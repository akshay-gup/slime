"""Inference-only parallel trajectory orchestration for multi_prompt_no_rl.

Runs K trajectories per sample with the shared bash-tool rollout flow, scores once,
ranks candidates, and optionally merges only the winner workspace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bash_tool_sandbox import tool_registry
from generate_with_bash_retool import _compute_bigmath_score, _split_multi_solution, generate
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    rollout_key: str
    sample_index: int
    trajectory_index: int
    solutions: list[str]
    score: float
    solved_count: int
    tool_calls: int
    turns: int


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_num}: {exc}") from exc
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _build_generate_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint=args.model_path,
        sglang_server_concurrency=1,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=args.temperature,
        rollout_top_p=args.top_p,
        rollout_top_k=-1,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        rollout_seed=0,
        n_samples_per_prompt=args.num_trajectories,
        sglang_dp_size=1,
        partial_rollout=False,
        sglang_router_ip=args.router_ip,
        sglang_router_port=args.router_port,
        rollout_max_response_len=args.max_new_tokens,
    )


async def run_trajectory(
    *,
    sample_index: int,
    trajectory_index: int,
    prompt_text: str,
    label_text: str,
    gold_solutions: list[str],
    generate_args: SimpleNamespace,
    sampling_params: dict[str, Any],
) -> TrajectoryResult:
    sample_id = sample_index * 1_000_000 + trajectory_index
    sample = Sample(index=sample_id, prompt=prompt_text, label=label_text, rollout_log_probs=[])
    generated = await generate(generate_args, sample, sampling_params)

    rollout_key = str(sample_id)
    solutions = list(getattr(generated, "generated_problem_solutions", []) or [])
    tool_calls = int(getattr(generated, "tool_call_count", 0) or 0)
    turns = generated.response.count("<tool_call>") if generated.response else 0

    scores: list[float] = []
    for idx, pred in enumerate(solutions):
        gold = gold_solutions[idx] if idx < len(gold_solutions) else ""
        scores.append(_compute_bigmath_score(pred, gold)["score"] if gold else (1.0 if pred else 0.0))

    score = (sum(scores) / len(scores)) if scores else 0.0
    solved_count = sum(1 for s in scores if s > 0)
    return TrajectoryResult(rollout_key, sample_index, trajectory_index, solutions, score, solved_count, tool_calls, turns)


async def run(args: argparse.Namespace) -> None:
    rows = _load_jsonl(Path(args.input_jsonl))
    generate_args = _build_generate_args(args)
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    all_results: list[dict[str, Any]] = []

    for sample_index, row in enumerate(rows[: args.max_samples]):
        prompt_text = row["prompt"]
        label_text = row.get("solution", "")
        gold_solutions = _split_multi_solution(label_text)

        tasks = [
            run_trajectory(
                sample_index=sample_index,
                trajectory_index=traj_idx,
                prompt_text=prompt_text,
                label_text=label_text,
                gold_solutions=gold_solutions,
                generate_args=generate_args,
                sampling_params=sampling_params,
            )
            for traj_idx in range(args.num_trajectories)
        ]
        results = await asyncio.gather(*tasks)

        ranked = sorted(results, key=lambda r: (r.score, r.solved_count, -r.tool_calls), reverse=True)
        winner = ranked[0]

        for traj in results:
            reward = 1.0 if (args.merge_winner and traj.rollout_key == winner.rollout_key) else 0.0
            tool_registry.finalize_rollout(traj.rollout_key, reward)

        all_results.append(
            {
                "sample_index": sample_index,
                "winner": winner.__dict__,
                "candidates": [candidate.__dict__ for candidate in ranked],
            }
        )
        logger.info("sample=%d winner=%s score=%.3f", sample_index, winner.rollout_key, winner.score)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d sample results to %s", len(all_results), out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel tool-use trajectories without RL training.")
    parser.add_argument("--input-jsonl", required=True, help="JSONL with at least {'prompt': ...}, optional {'solution': ...}")
    parser.add_argument("--output-json", default="no_rl_winners.json", help="Output JSON file for winner/candidate summaries")
    parser.add_argument("--model-path", required=True, help="HF checkpoint path used by GenerateState")
    parser.add_argument("--router-ip", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--num-trajectories", type=int, default=4)
    parser.add_argument("--max-turns-per-problem", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-samples", type=int, default=1_000_000)
    parser.add_argument(
        "--merge-winner",
        action="store_true",
        help="Merge only the winning trajectory workspace into main. Otherwise discard all trajectory workspaces.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

"""Example rollout for bash-tool RL with context reset after write events.

This example demonstrates:
1) Multi-turn tool calling with a bash executor.
2) Context reset immediately after file write/create operations.
3) Terminal-state reward, then per-step reward redistribution.

Configure with:
  --custom-generate-function-path examples.bash_reset_context.generate_with_bash_reset.generate
  --custom-rm-path examples.bash_reset_context.generate_with_bash_reset.reward_func
"""

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

BASH_RESET_CONFIG = {
    "max_turns": 8,
    "max_bash_chars": 4000,
    "workspace_key": "workspace_dir",
    "reset_bootstrap": (
        "You are continuing a coding task. Context was reset after a write operation. "
        "Rely on the current file state summary and continue from there."
    ),
}


@dataclass
class StepRecord:
    turn_idx: int
    action: str
    command: str
    wrote_files: bool


def _extract_action_and_command(prediction: str) -> tuple[str | None, str]:
    """Accept either <bash>...</bash> or <answer>...</answer>."""
    match = re.search(r"<(bash|answer)>(.*?)</\\1>", prediction, flags=re.DOTALL)
    if not match:
        return None, ""
    return match.group(1), match.group(2).strip()


def _snapshot_tree(root: str) -> dict[str, tuple[int, int]]:
    """Return a lightweight file snapshot: relpath -> (size, mtime_ns)."""
    state = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            state[rel] = (st.st_size, st.st_mtime_ns)
    return state


def _detect_writes(before: dict, after: dict) -> tuple[bool, list[str]]:
    changed = [p for p, v in after.items() if p not in before or before[p] != v]
    return (len(changed) > 0), sorted(changed)[:20]


def _run_bash(cmd: str, cwd: str) -> str:
    cmd = cmd[: BASH_RESET_CONFIG["max_bash_chars"]]
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    out = f"$ {shlex.quote(cmd)}\n"
    if proc.stdout:
        out += f"[stdout]\n{proc.stdout}\n"
    if proc.stderr:
        out += f"[stderr]\n{proc.stderr}\n"
    out += f"[exit_code] {proc.returncode}\n"
    return out


def _redistribute_terminal_reward(terminal_reward: float, records: list[StepRecord]) -> list[float]:
    """Write-aware distribution: write steps get 2x weight, read steps 1x."""
    if not records:
        return []
    weights = [2.0 if r.wrote_files else 1.0 for r in records]
    total = sum(weights)
    return [terminal_reward * (w / total) for w in weights]


async def _model_generate(url: str, text: str, sampling_params: dict):
    payload = {"text": text, "sampling_params": sampling_params, "return_logprob": True}
    return await post(url, payload)


async def generate(args, sample: Sample, sampling_params) -> Sample:
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    workspace_dir = sample.metadata.get(BASH_RESET_CONFIG["workspace_key"], os.getcwd())

    prompt_tokens = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_ids = []
    loss_mask = []
    rollout_log_probs = []
    step_records: list[StepRecord] = []

    active_context = sample.prompt

    for turn_idx in range(BASH_RESET_CONFIG["max_turns"]):
        output = await _model_generate(url=url, text=active_context + response, sampling_params=sampling_params)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_ids = [row[1] for row in output["meta_info"]["output_token_logprobs"]]
        cur_logps = [row[0] for row in output["meta_info"]["output_token_logprobs"]]
        cur_text = state.tokenizer.decode(cur_ids)

        response += cur_text
        response_ids += cur_ids
        rollout_log_probs += cur_logps
        loss_mask += [1] * len(cur_ids)

        action, content = _extract_action_and_command(cur_text)
        if action == "answer":
            break

        if action != "bash":
            tool_obs = "\n[tool] invalid action. Use <bash>...</bash> or <answer>...</answer>.\n"
            wrote_files = False
            changed_files = []
            command = ""
        else:
            pre = _snapshot_tree(workspace_dir)
            tool_obs = _run_bash(content, cwd=workspace_dir)
            post = _snapshot_tree(workspace_dir)
            wrote_files, changed_files = _detect_writes(pre, post)
            command = content

        step_records.append(StepRecord(turn_idx=turn_idx, action=action or "invalid", command=command, wrote_files=wrote_files))

        obs_suffix = f"\n<tool_observation>\n{tool_obs}\n</tool_observation>\n"
        obs_ids = state.tokenizer(obs_suffix, add_special_tokens=False)["input_ids"]
        response += obs_suffix
        response_ids += obs_ids
        rollout_log_probs += [0.0] * len(obs_ids)
        loss_mask += [0] * len(obs_ids)

        if wrote_files:
            summary = {
                "write_event": True,
                "changed_files": changed_files,
                "steps_so_far": len(step_records),
            }
            active_context = (
                f"{BASH_RESET_CONFIG['reset_bootstrap']}\n"
                f"Task: {sample.prompt}\n"
                f"StateSummary: {json.dumps(summary, ensure_ascii=False)}\n"
                "Continue."
            )
            response = ""
            response_ids = []
            loss_mask = []
            rollout_log_probs = []

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

    sample.tokens = prompt_tokens + response_ids
    sample.response = response
    sample.response_length = len(response_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.metadata["step_records"] = [r.__dict__ for r in step_records]

    sample.status = Sample.Status.COMPLETED
    return sample


async def reward_func(args, sample: Sample, **kwargs):
    """Terminal reward example; replace with your real evaluator/test harness."""
    if not isinstance(sample, Sample):
        raise TypeError("sample must be a Sample")

    # Placeholder terminal reward:
    # +1 if model emitted an <answer>, else -0.2
    terminal_reward = 1.0 if "<answer>" in sample.response else -0.2

    records = [StepRecord(**x) for x in sample.metadata.get("step_records", [])]
    per_step_rewards = _redistribute_terminal_reward(terminal_reward, records)

    return {
        "score": terminal_reward,
        "terminal_reward": terminal_reward,
        "step_rewards": per_step_rewards,
        "num_steps": len(records),
    }

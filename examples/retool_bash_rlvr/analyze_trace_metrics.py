#!/usr/bin/env python3
"""Collate retool_bash_rlvr trace JSONL files and emit debugging metrics.

Primary goal: detect cases where the model appears to write `answer.md` but reward
logic does not observe it (`reward_answer_file.exists == false`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RolloutSummary:
    rollout_key: str
    files: set[str] = field(default_factory=set)
    steps: Counter[str] = field(default_factory=Counter)
    answer_found_events: int = 0
    bash_result_file_changed_events: int = 0
    reward_answer_exists: bool | None = None
    reward_answer_content_nonempty: bool | None = None
    reward_score: float | None = None
    merge_message: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="+",
        help="Trace JSONL files and/or directories containing rollout_*.jsonl files.",
    )
    p.add_argument(
        "--glob",
        default="rollout_*.jsonl",
        help="Glob used when a directory is provided (default: rollout_*.jsonl).",
    )
    p.add_argument(
        "--show-bad",
        type=int,
        default=20,
        help="Max suspicious rollout rows to print in detail.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    return p.parse_args()


def expand_paths(inputs: list[str], pattern: str) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            files.append(p)
    uniq = sorted({f.resolve() for f in files})
    return uniq


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                events.append(
                    {
                        "step": "__parse_error__",
                        "error": str(e),
                        "source_line": i,
                    }
                )
                continue
            if isinstance(obj, dict):
                events.append(obj)
    return events


def update_summary(summary: RolloutSummary, ev: dict[str, Any]) -> None:
    step = str(ev.get("step", "<missing_step>"))
    summary.steps[step] += 1

    if step == "answer_found":
        summary.answer_found_events += 1

    if step == "bash_result":
        preview = str(ev.get("result_preview", ""))
        if "Files changed: yes" in preview:
            summary.bash_result_file_changed_events += 1

    if step == "reward_answer_file":
        exists = bool(ev.get("exists", False))
        content = str(ev.get("content", ""))
        summary.reward_answer_exists = exists
        summary.reward_answer_content_nonempty = bool(content.strip())

    if step == "reward_computed":
        score = ev.get("score")
        try:
            summary.reward_score = float(score)
        except (TypeError, ValueError):
            pass

    if step == "merge_result":
        summary.merge_message = str(ev.get("message", ""))


def classify_issue(s: RolloutSummary) -> str | None:
    # Most suspicious: model reached answer file detection, but reward couldn't read it.
    if s.answer_found_events > 0 and s.reward_answer_exists is False:
        return "answer_found_but_reward_missing"

    # Model modified files but reward says no answer file.
    if s.bash_result_file_changed_events > 0 and s.reward_answer_exists is False:
        return "files_changed_but_reward_missing"

    # Reward says answer exists but content is empty.
    if s.reward_answer_exists and s.reward_answer_content_nonempty is False:
        return "reward_found_empty_answer"

    # Answer seems present but score is zero: potential format/parsing mismatch.
    if s.reward_answer_exists and s.reward_score == 0.0:
        return "answer_present_but_zero_score"

    return None


def build_report(files: list[Path]) -> dict[str, Any]:
    rollouts: dict[str, RolloutSummary] = {}
    global_steps = Counter()
    parse_errors = 0

    for f in files:
        for ev in read_jsonl(f):
            step = str(ev.get("step", "<missing_step>"))
            if step == "__parse_error__":
                parse_errors += 1
                continue

            rollout_key = str(ev.get("rollout_key", "<missing_rollout_key>"))
            if rollout_key not in rollouts:
                rollouts[rollout_key] = RolloutSummary(rollout_key=rollout_key)
            s = rollouts[rollout_key]
            s.files.add(str(f))

            global_steps[step] += 1
            update_summary(s, ev)

    issue_counts: Counter[str] = Counter()
    suspicious: list[dict[str, Any]] = []
    for s in rollouts.values():
        issue = classify_issue(s)
        if issue:
            issue_counts[issue] += 1
            suspicious.append(
                {
                    "rollout_key": s.rollout_key,
                    "issue": issue,
                    "answer_found_events": s.answer_found_events,
                    "bash_result_file_changed_events": s.bash_result_file_changed_events,
                    "reward_answer_exists": s.reward_answer_exists,
                    "reward_answer_content_nonempty": s.reward_answer_content_nonempty,
                    "reward_score": s.reward_score,
                    "merge_message": s.merge_message,
                    "files": sorted(s.files),
                }
            )

    return {
        "num_files": len(files),
        "num_rollouts": len(rollouts),
        "parse_errors": parse_errors,
        "global_steps": dict(global_steps),
        "issue_counts": dict(issue_counts),
        "suspicious_rollouts": sorted(suspicious, key=lambda x: (x["issue"], x["rollout_key"])),
    }


def print_human(report: dict[str, Any], show_bad: int) -> None:
    print("=== retool_bash_rlvr trace metrics ===")
    print(f"files: {report['num_files']}")
    print(f"rollouts: {report['num_rollouts']}")
    print(f"parse_errors: {report['parse_errors']}")

    print("\n-- Step counts --")
    for k, v in sorted(report["global_steps"].items()):
        print(f"{k}: {v}")

    print("\n-- Issue counts --")
    if report["issue_counts"]:
        for k, v in sorted(report["issue_counts"].items()):
            print(f"{k}: {v}")
    else:
        print("none")

    rows = report["suspicious_rollouts"]
    if not rows:
        print("\nNo suspicious rollouts detected.")
        return

    print(f"\n-- Suspicious rollout details (showing up to {show_bad}) --")
    for row in rows[:show_bad]:
        print(json.dumps(row, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    files = expand_paths(args.paths, args.glob)
    if not files:
        print("No trace files found.")
        return 1

    report = build_report(files)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_human(report, args.show_bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

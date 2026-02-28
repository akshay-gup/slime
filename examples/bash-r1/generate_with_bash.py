"""Bash-tool multi-turn rollout example for slime.

This mirrors the Search-R1 style environment, but replaces retrieval with a
single bash tool call surface. The model should emit:
- <bash>...</bash> to execute shell commands
- <answer>...</answer> to finish
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

BASH_R1_CONFIGS = {
    "max_turns": 4,
    "tool_concurrency": 128,
    "timeout": 30,
    "max_output_chars": 8192,
    "blocked_patterns": [
        "rm -rf /",
        ":(){ :|:&};:",
        "mkfs",
        "dd if=",
        "> /dev/sd",
    ],
    "bad_call_penalty": -0.01,
    "format_score": 0.2,
    "return_logprob": False,
}

SEMAPHORE = asyncio.Semaphore(BASH_R1_CONFIGS["tool_concurrency"])

SYSTEM_PROMPT = (
    "You can use one tool: bash. "
    "When you need to run a command, respond with <bash>YOUR_COMMAND</bash>. "
    "When you are done, respond with <answer>FINAL_ANSWER</answer>."
)


def _truncate(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return output[:half] + f"\n\n... truncated ({len(output)} chars total) ...\n\n" + output[-half:]


def _postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    pattern = r"<(bash|answer)>(.*?)</\\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return None, ""


async def run_bash(command: str) -> tuple[str, bool]:
    command = command.strip()
    if not command:
        return "Error: 'command' must be a non-empty string.", False

    for pattern in BASH_R1_CONFIGS["blocked_patterns"]:
        if pattern in command:
            return "Error: command blocked by safety policy.", False

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            ),
            timeout=2,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=BASH_R1_CONFIGS["timeout"])
        exit_code = proc.returncode
    except asyncio.TimeoutError:
        return f"Error: command timed out after {BASH_R1_CONFIGS['timeout']}s.", False
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}", False

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    if exit_code != 0:
        parts.append(f"Exit code: {exit_code}")

    output = "\n".join(parts) if parts else "(no output)"
    output = _truncate(output, BASH_R1_CONFIGS["max_output_chars"])

    return output, exit_code == 0


async def execute_predictions(prediction: str) -> tuple[str, bool, bool]:
    action, content = _postprocess_predictions(prediction)

    if action == "bash":
        async with SEMAPHORE:
            tool_output, success = await run_bash(content)
        observation = f"\n\n<tool_response>{tool_output}</tool_response>\n\n"
        return observation, False, success

    if action == "answer":
        return "", True, True

    invalid_obs = (
        "\nMy previous action is invalid. "
        "Use <bash>...</bash> for a tool call, or <answer>...</answer> to finish.\n"
    )
    return invalid_obs, False, False


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text = f"{SYSTEM_PROMPT}\n\n{sample.prompt}"
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []

    output: dict[str, Any] | None = None

    for _turn_idx in range(BASH_R1_CONFIGS["max_turns"]):
        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
        }
        output = await post(url, payload)

        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)

        if finish_reason == "length":
            break

        next_obs, done, _success = await execute_predictions(cur_response)
        if done:
            break

        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    if output is None:
        sample.status = Sample.Status.ABORTED
        return sample

    finish_reason = output["meta_info"]["finish_reason"]["type"]
    if finish_reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_reason == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED

    return sample


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    prediction = _extract_answer(sample.response)
    target = sample.label

    score = 1.0 if _normalize_text(prediction) == _normalize_text(str(target)) else 0.0

    format_ok = float(bool(re.search(r"<answer>.*?</answer>", sample.response, re.DOTALL)))
    sample.reward = score + format_ok * BASH_R1_CONFIGS["format_score"]
    return sample

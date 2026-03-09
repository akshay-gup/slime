import logging
import re
from pathlib import Path
from typing import Any

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from bash_tool_sandbox import TOOL_CONFIGS, create_tracer, tool_registry

logger = logging.getLogger(__name__)

REWARD_RESULT_FILE = "answer.md"
PROBLEM_FILE = TOOL_CONFIGS["problem_file"]
TASK_FILE_TEMPLATE = f"""# Instructions

Your working memory resets frequently. Anything not written to a file
will be lost. This is normal.

## How to work

1. Start every cycle by reading the workspace: `ls` to see what exists.
2. If you see files from previous work, read them to understand where
   you left off. Do not start over — continue from what's there.
3. Do all your thinking and computation by writing files and running
   them. Do not try to solve problems in your head.
4. Before ending any cycle, make sure your progress is saved to a file.
   Write what you've figured out, what's left to do, and any partial
   results.

## How to finish

When you have your final answer, write it to `{REWARD_RESULT_FILE}` using
the format: Answer: \\boxed{{your_answer}}

## Workspace

This workspace persists across tasks. Files you create now will be here
for future tasks. If you build something useful — a script, a strategy,
a template — it stays. Organize however helps you work better over time.

# Problem

{{problem_text}}
"""


def _parse_model_answer(text: str):
    """Parse a model answer using boxed-math extraction rules."""
    return parse(
        text,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
        parsing_timeout=None,
    )


def _compute_bigmath_score(solution_str: str, ground_truth: str) -> dict[str, Any]:
    """Compute 0/1 reward with BigMath verifier-compatible parsing."""
    if not ground_truth:
        return {"score": 0.0, "pred": None}

    gold_parsed = parse(ground_truth, extraction_mode="first_match", parsing_timeout=None)
    if not gold_parsed:
        return {"score": 0.0, "pred": None}

    answer_parsed = _parse_model_answer(solution_str)
    if not answer_parsed:
        return {"score": 0.0, "pred": None}

    try:
        reward = float(verify(gold_parsed, answer_parsed, timeout_seconds=None))
    except Exception:
        logger.exception("BigMath verify failed")
        reward = 0.0

    return {
        "score": 1.0 if reward > 0.0 else 0.0,
        "pred": str(answer_parsed),
    }


def _extract_prompt_text(prompt: str | list[dict[str, str]]) -> str:
    """Extract plain text from a prompt that may be a chat-format message list."""
    if isinstance(prompt, str):
        return prompt
    return "\n".join(msg.get("content", "") for msg in prompt if msg.get("content"))

TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_conversation_with_tools(prompt: str, tools: list[dict[str, Any]] = None) -> str:
    template = Template(TOOL_TEMPLATE)
    messages = [
        {
            "role": "system",
            "content": (
                "You are working in an environment with a bash tool. "
                f"Start by reading {PROBLEM_FILE} in the current directory."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    rendered = template.render(messages=messages, tools=tools or [])
    logger.debug("System prompt rendered (%d chars): %.500s", len(rendered), rendered)
    return rendered


def postprocess_predictions(prediction: str):
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json

            tool_call_str = tool_call_match.group(1)
            try:
                tool_call_data = json.loads(tool_call_str)
            except json.JSONDecodeError:
                # Some model outputs include raw newlines inside JSON string values.
                # Retry with escaped newlines to salvage those cases.
                tool_call_data = json.loads(tool_call_str.replace("\n", "\\n"))

            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})
            if tool_name == "bash":
                command = arguments.get("command", "")
                if command.strip():
                    logger.debug("Parsed bash tool_call: %.200s", command)
                    return "bash", command
        except (json.JSONDecodeError, KeyError, AttributeError):
            logger.debug("Failed to parse tool_call JSON from: %.200s", prediction)

    logger.debug("No valid tool_call in prediction: %.200s", prediction)
    return None, ""


async def execute_predictions(prediction: str, rollout_key: str | int | None, tracer=None) -> tuple[str, bool]:
    action, content = postprocess_predictions(prediction)

    if action == "bash":
        logger.info("[rollout=%s] Executing bash: %.200s", rollout_key, content)
        if tracer:
            tracer.log("bash_execute", command=content[:300])
        result = await tool_registry.execute_tool("bash", {"command": content}, rollout_key=rollout_key)
        rollout_dir = tool_registry._resolve_rollout_workdir(rollout_key)
        done = (Path(rollout_dir) / REWARD_RESULT_FILE).is_file()
        logger.info("[rollout=%s] Bash result (%d chars), done=%s", rollout_key, len(result), done)
        if tracer:
            tracer.log("bash_result", result_length=len(result), done=done, result_preview=result[:500])
        return f"\n\n<tool_response>\n{result}\n</tool_response>\n\n", done

    if action is None:
        logger.debug("[rollout=%s] No tool call emitted; checking for answer file", rollout_key)
        if tracer:
            tracer.log("no_tool_call", prediction_preview=prediction[:300])
        rollout_dir = tool_registry._resolve_rollout_workdir(rollout_key)
        if (Path(rollout_dir) / REWARD_RESULT_FILE).is_file():
            return "", True
        return (f"\nRead {PROBLEM_FILE} in the current directory for full instructions.\n", False)

    logger.info("[rollout=%s] Invalid tool call (action=%s)", rollout_key, action)
    if tracer:
        tracer.log("invalid_tool_call", action=str(action), prediction_preview=prediction[:300])
    return (
        "\nMy previous action is invalid. If I want to use the tool, I should emit "
        "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"...\"}}</tool_call>. "
        "Let me try again.\n",
        False,
    )


def _has_file_change(tool_response: str) -> bool:
    return "Files changed: yes" in tool_response


def _resolve_rollout_key(sample: Sample) -> str | int | None:
    if TOOL_CONFIGS.get("shared_workspace_across_prompts", True):
        if sample.index is not None:
            return sample.index
        if sample.group_index is not None:
            return sample.group_index
        return "shared"
    return sample.index if sample.index is not None else sample.group_index


def _archive_and_reset_context_tokens(
    context_response_token_ids: list[int], archived_context_response_token_ids: list[list[int]]
) -> list[int]:
    if context_response_token_ids:
        archived_context_response_token_ids.append(context_response_token_ids.copy())
    return []


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    tool_specs = tool_registry.get_tool_specs()
    rollout_key = _resolve_rollout_key(sample)
    rollout_lock = tool_registry.get_rollout_lock(rollout_key)
    tracer = create_tracer(rollout_key)

    prompt_text = _extract_prompt_text(sample.prompt)
    logger.info("[rollout=%s] Starting generate for sample index=%s, prompt: %.150s", rollout_key, sample.index, prompt_text)
    if tracer:
        tracer.log("generate_start", sample_index=sample.index, prompt_preview=prompt_text[:300])

    async with rollout_lock:
        tool_registry.prepare_rollout(rollout_key)
        task_text = Template(TASK_FILE_TEMPLATE).render(problem_text=prompt_text.rstrip())
        tool_registry.write_problem_file(rollout_key=rollout_key, problem_text=task_text)
        prompt = format_conversation_with_tools(prompt="Please work on the task in the environment.", tools=tool_specs)

        prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        logger.info("[rollout=%s] Prompt tokenized: %d tokens", rollout_key, len(prompt_tokens_ids))
        if tracer:
            tracer.log("system_prompt", prompt_length=len(prompt), prompt_tokens=len(prompt_tokens_ids), content=prompt[:2000])
        response = ""
        response_token_ids = []
        loss_masks = []
        context_response_token_ids = []
        archived_context_response_token_ids = []
        tool_call_count = 0
        saw_length_stop = False

        for turn_num in range(TOOL_CONFIGS["max_turns"]):
            logger.info("[rollout=%s] Turn %d/%d, context_tokens=%d, tool_calls=%d", rollout_key, turn_num + 1, TOOL_CONFIGS["max_turns"], len(prompt_tokens_ids + context_response_token_ids), tool_call_count)
            if tracer:
                tracer.log("turn_start", turn=turn_num + 1, context_tokens=len(prompt_tokens_ids + context_response_token_ids), tool_calls_so_far=tool_call_count)

            current_token_ids = prompt_tokens_ids + context_response_token_ids
            payload = {
                "input_ids": current_token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            output = await post(url, payload)
            finish_reason = output["meta_info"]["finish_reason"]["type"]
            if finish_reason == "abort":
                logger.warning("[rollout=%s] Generation aborted", rollout_key)
                if tracer:
                    tracer.log("abort", turn=turn_num + 1)
                sample.status = Sample.Status.ABORTED
                return sample

            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

            logger.info("[rollout=%s] Turn %d: %d new tokens, finish_reason=%s", rollout_key, turn_num + 1, len(cur_response_token_ids), finish_reason)
            if tracer:
                tracer.log("model_output", turn=turn_num + 1, new_tokens=len(cur_response_token_ids), finish_reason=finish_reason, response_preview=cur_response[:500])

            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs

            response += cur_response
            response_token_ids += cur_response_token_ids
            context_response_token_ids += cur_response_token_ids
            loss_masks += [1] * len(cur_response_token_ids)

            if finish_reason == "length":
                logger.info("[rollout=%s] Hit length limit, stopping", rollout_key)
                if tracer:
                    tracer.log("length_stop", turn=turn_num + 1)
                saw_length_stop = True
                break

            next_obs, done = await execute_predictions(cur_response, rollout_key=rollout_key, tracer=tracer)
            if done:
                logger.info("[rollout=%s] Answer file detected, stopping", rollout_key)
                if tracer:
                    tracer.log("answer_found", turn=turn_num + 1)
                break

            if "<tool_response>" in next_obs:
                tool_call_count += 1

            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response += next_obs
            response_token_ids += obs_tokens_ids
            context_response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

            if _has_file_change(next_obs):
                # When files are modified, clear ongoing context for the next model step.
                # Keep the dropped context for downstream data pairing.
                logger.info("[rollout=%s] File change detected, archiving context (%d tokens)", rollout_key, len(context_response_token_ids))
                if tracer:
                    tracer.log("context_reset", turn=turn_num + 1, archived_tokens=len(context_response_token_ids))
                context_response_token_ids = _archive_and_reset_context_tokens(
                    context_response_token_ids, archived_context_response_token_ids
                )

            if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
                logger.info("[rollout=%s] Max tool calls (%d) reached, stopping", rollout_key, TOOL_CONFIGS["max_tool_calls"])
                if tracer:
                    tracer.log("max_tool_calls", turn=turn_num + 1, max_calls=TOOL_CONFIGS["max_tool_calls"])
                break

        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = response
        sample.loss_mask = loss_masks
        sample.tool_call_count = tool_call_count
        sample.context_reset_token_segments = archived_context_response_token_ids

        if saw_length_stop:
            sample.status = Sample.Status.TRUNCATED
        elif output["meta_info"]["finish_reason"]["type"] == "stop":
            sample.status = Sample.Status.COMPLETED

        logger.info("[rollout=%s] Generate complete: status=%s, tool_calls=%d, response_tokens=%d, context_resets=%d", rollout_key, sample.status, tool_call_count, len(response_token_ids), len(archived_context_response_token_ids))
        if tracer:
            tracer.log("generate_end", status=str(sample.status), tool_calls=tool_call_count, response_tokens=len(response_token_ids), context_resets=len(archived_context_response_token_ids))

    return sample


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    rollout_key = _resolve_rollout_key(sample)
    rollout_lock = tool_registry.get_rollout_lock(rollout_key)
    tracer = create_tracer(rollout_key)

    async with rollout_lock:
        rollout_dir = tool_registry._resolve_rollout_workdir(rollout_key)
        result_file = Path(rollout_dir) / REWARD_RESULT_FILE
        file_answer = ""
        if result_file.exists() and result_file.is_file():
            file_answer = result_file.read_text(encoding="utf-8", errors="replace").strip()

        logger.info("[rollout=%s] Answer file %s: %s", rollout_key, "found" if file_answer else "NOT found", result_file)
        if tracer:
            tracer.log("reward_answer_file", exists=bool(file_answer), content=file_answer[:500] if file_answer else "")

        if file_answer:
            solution_str = file_answer
        else:
            solution_str = ""

        ground_truth = sample.label if sample.label is not None else ""
        result = _compute_bigmath_score(solution_str, ground_truth)
        if result["pred"] is None:
            result["pred"] = ""

        logger.info("[rollout=%s] Reward: score=%s, pred=%.100s, ground_truth=%.100s", rollout_key, result["score"], str(result["pred"]), str(ground_truth))
        if tracer:
            tracer.log("reward_computed", score=result["score"], pred=str(result["pred"])[:200], ground_truth=str(ground_truth)[:200])

        result["reward_result_file"] = str(result_file)
        result["reward_result_content"] = file_answer
        merge_message = tool_registry.finalize_rollout(rollout_key=rollout_key, reward=result["score"])
        result["merge_message"] = merge_message

        logger.info("[rollout=%s] Finalize: %s", rollout_key, merge_message)
        if tracer:
            tracer.log("merge_result", message=merge_message)

    return result

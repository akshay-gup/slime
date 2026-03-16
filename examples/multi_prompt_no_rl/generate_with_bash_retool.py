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
from data_utils import PROBLEM_DELIMITER, SOLUTION_DELIMITER

logger = logging.getLogger(__name__)

REWARD_RESULT_FILE = "solution.md"
MULTI_SOLUTION_RESULT_FILE = "solutions.md"
README_FILE_TEMPLATE = """# Instructions

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

When you have your final answer, write it to `{{reward_result_file}}` using
the format: Answer: \\boxed{{your_answer}}

## Workspace

This workspace persists across tasks. Files you create now will be here
for future tasks. If you build something useful — a script, a strategy,
a template — it stays. Organize however helps you work better over time.

## Current task

Read `task.md` in the current directory for the current problem statement.
"""

TASK_FILE_TEMPLATE = """# Problem

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


def _split_multi_prompt(prompt_text: str) -> list[str]:
    parts = [part.strip() for part in prompt_text.split(PROBLEM_DELIMITER)]
    non_empty_parts = [part for part in parts if part]
    return non_empty_parts or [prompt_text.strip()]


def _split_multi_solution(solution_text: str) -> list[str]:
    if not solution_text:
        return []
    return [part.strip() for part in solution_text.split(SOLUTION_DELIMITER)]

TOOL_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" }}
    {%- raw %}{"name": <function-name>, "arguments": <args-json-object>}{%- endraw %}
    {{- "\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{\"name\": \"' }}
                {{- tool_call.name }}
                {{- '\", \"arguments\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""


def format_conversation_with_tools(tools: list[dict[str, Any]] = None) -> str:
    template = Template(TOOL_TEMPLATE)
    messages = [
        {
            "role": "system",
            "content": "Read files from the current workspace only. Start by reading README.md.",
        }
    ]
    rendered = template.render(messages=messages, tools=tools or [], add_generation_prompt=True)
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
        return ("\nRead README.md for instructions and the current problem.\n", False)

    logger.info("[rollout=%s] Invalid tool call (action=%s)", rollout_key, action)
    if tracer:
        tracer.log("invalid_tool_call", action=str(action), prediction_preview=prediction[:300])
    return (
        "\nMy previous action is invalid. If I want to use the tool, I should emit "
        "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"...\"}}</tool_call>. "
        "Let me try again.\n",
        False,
    )


def _resolve_rollout_key(sample: Sample) -> str | int | None:
    if TOOL_CONFIGS.get("shared_workspace_across_prompts", True):
        if sample.index is not None:
            return sample.index
        if sample.group_index is not None:
            return sample.group_index
        return "shared"
    return sample.index if sample.index is not None else sample.group_index


def _archive_and_reset_context_tokens(
    context_response_token_ids: list[int],
    archived_context_response_token_ids: list[list[int]],
    max_segment_length: int | None = None,
) -> list[int]:
    if not context_response_token_ids:
        return []

    if max_segment_length is None or max_segment_length <= 0:
        archived_context_response_token_ids.append(context_response_token_ids.copy())
        return []

    start = 0
    total = len(context_response_token_ids)
    while start < total:
        end = min(start + max_segment_length, total)
        archived_context_response_token_ids.append(context_response_token_ids[start:end])
        start = end
    return []


def _infer_max_segment_length(args) -> int | None:
    value = getattr(args, "rollout_max_response_len", None)
    if isinstance(value, int) and value > 0:
        return value
    return None



async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    tool_specs = tool_registry.get_tool_specs()
    rollout_key = _resolve_rollout_key(sample)
    rollout_lock = tool_registry.get_rollout_lock(rollout_key)
    tracer = create_tracer(rollout_key)

    prompt_text = _extract_prompt_text(sample.prompt)
    prompt_problems = _split_multi_prompt(prompt_text)
    logger.info("[rollout=%s] Starting generate for sample index=%s, prompt: %.150s", rollout_key, sample.index, prompt_text)
    if tracer:
        tracer.log("generate_start", sample_index=sample.index, prompt_preview=prompt_text[:300])

    async with rollout_lock:
        tool_registry.prepare_rollout(rollout_key)
        prompt = format_conversation_with_tools(tools=tool_specs)

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
        max_segment_length = _infer_max_segment_length(args)
        collected_solutions: list[str] = []
        rollout_dir = tool_registry._resolve_rollout_workdir(rollout_key)

        readme_text = Template(README_FILE_TEMPLATE).render(
            reward_result_file=REWARD_RESULT_FILE,
        )
        tool_registry.write_problem_file(
            rollout_key=rollout_key,
            problem_text=None,
            instruction_text=readme_text,
        )

        for problem_idx, problem_text in enumerate(prompt_problems):
            skip_problem_due_to_context_limit = False
            result_file = Path(rollout_dir) / REWARD_RESULT_FILE
            if result_file.exists() and result_file.is_file():
                # Ensure a stale solution from a previous problem cannot leak into
                # the current problem's done detection.
                result_file.unlink(missing_ok=True)
                logger.info(
                    "[rollout=%s] Removed stale %s before problem %d/%d",
                    rollout_key,
                    REWARD_RESULT_FILE,
                    problem_idx + 1,
                    len(prompt_problems),
                )
                if tracer:
                    tracer.log("stale_solution_file_removed", problem_index=problem_idx)

            task_text = Template(TASK_FILE_TEMPLATE).render(problem_text=problem_text.rstrip())
            tool_registry.write_problem_file(
                rollout_key=rollout_key,
                problem_text=task_text,
            )

            logger.info(
                "[rollout=%s] Starting problem %d/%d",
                rollout_key,
                problem_idx + 1,
                len(prompt_problems),
            )
            if tracer:
                tracer.log("problem_start", problem_index=problem_idx, total_problems=len(prompt_problems))

            turn_num = 0
            while True:
                turn_num += 1
                logger.info("[rollout=%s] Turn %d, context_tokens=%d, tool_calls=%d", rollout_key, turn_num, len(prompt_tokens_ids + context_response_token_ids), tool_call_count)
                if tracer:
                    tracer.log("turn_start", turn=turn_num, context_tokens=len(prompt_tokens_ids + context_response_token_ids), tool_calls_so_far=tool_call_count)

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
                        tracer.log("abort", turn=turn_num)
                    sample.status = Sample.Status.ABORTED
                    return sample

                cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_response = state.tokenizer.decode(cur_response_token_ids)
                cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

                logger.info("[rollout=%s] Turn %d: %d new tokens, finish_reason=%s", rollout_key, turn_num, len(cur_response_token_ids), finish_reason)
                if tracer:
                    tracer.log("model_output", turn=turn_num, new_tokens=len(cur_response_token_ids), finish_reason=finish_reason, response_preview=cur_response[:500])

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
                        tracer.log("length_stop", turn=turn_num)
                    saw_length_stop = True
                    break

                next_obs, done = await execute_predictions(cur_response, rollout_key=rollout_key, tracer=tracer)
                if done:
                    logger.info("[rollout=%s] Answer file detected, stopping", rollout_key)
                    if tracer:
                        tracer.log("answer_found", turn=turn_num)
                    if context_response_token_ids:
                        # Reset context only when the model has produced a solution file,
                        # not on every intermediate file mutation in the workspace.
                        logger.info(
                            "[rollout=%s] Solution file written, archiving context (%d tokens)",
                            rollout_key,
                            len(context_response_token_ids),
                        )
                        if tracer:
                            tracer.log("context_reset_solution_file", turn=turn_num, archived_tokens=len(context_response_token_ids))
                        context_response_token_ids = _archive_and_reset_context_tokens(
                            context_response_token_ids,
                            archived_context_response_token_ids,
                            max_segment_length=max_segment_length,
                        )
                    break

                if "<tool_response>" in next_obs:
                    tool_call_count += 1

                obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
                response += next_obs
                response_token_ids += obs_tokens_ids
                context_response_token_ids += obs_tokens_ids
                loss_masks += [0] * len(obs_tokens_ids)
                sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

                if max_segment_length is not None and len(context_response_token_ids) > max_segment_length:
                    logger.info(
                        "[rollout=%s] Context length %d exceeded max segment length %d for problem %d/%d; skipping problem",
                        rollout_key,
                        len(context_response_token_ids),
                        max_segment_length,
                        problem_idx + 1,
                        len(prompt_problems),
                    )
                    if tracer:
                        tracer.log(
                            "problem_skipped_context_limit",
                            problem_index=problem_idx,
                            turn=turn_num,
                            archived_tokens=len(context_response_token_ids),
                            max_segment_length=max_segment_length,
                        )
                    skip_problem_due_to_context_limit = True
                    context_response_token_ids = _archive_and_reset_context_tokens(
                        context_response_token_ids,
                        archived_context_response_token_ids,
                        max_segment_length=max_segment_length,
                    )
                    break

            if skip_problem_due_to_context_limit:
                if result_file.exists() and result_file.is_file():
                    result_file.unlink(missing_ok=True)
                collected_solutions.append("")
                logger.info(
                    "[rollout=%s] Skipped problem %d/%d after context limit; recorded empty solution",
                    rollout_key,
                    problem_idx + 1,
                    len(prompt_problems),
                )
                if tracer:
                    tracer.log("problem_solution_skipped", problem_index=problem_idx, reason="context_limit")
            elif result_file.exists() and result_file.is_file():
                problem_solution = result_file.read_text(encoding="utf-8", errors="replace").strip()
                collected_solutions.append(problem_solution)
                result_file.unlink(missing_ok=True)
                logger.info("[rollout=%s] Collected solution for problem %d/%d", rollout_key, problem_idx + 1, len(prompt_problems))
                if tracer:
                    tracer.log("problem_solution_collected", problem_index=problem_idx, has_solution=bool(problem_solution))
            else:
                collected_solutions.append("")
                logger.info("[rollout=%s] No solution file for problem %d/%d", rollout_key, problem_idx + 1, len(prompt_problems))
                if tracer:
                    tracer.log("problem_solution_missing", problem_index=problem_idx)

            if context_response_token_ids:
                context_response_token_ids = _archive_and_reset_context_tokens(
                    context_response_token_ids,
                    archived_context_response_token_ids,
                    max_segment_length=max_segment_length,
                )

        (Path(rollout_dir) / MULTI_SOLUTION_RESULT_FILE).write_text(
            SOLUTION_DELIMITER.join(collected_solutions),
            encoding="utf-8",
        )
        sample.generated_problem_solutions = collected_solutions

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
        multi_result_file = Path(rollout_dir) / MULTI_SOLUTION_RESULT_FILE
        file_answer = ""
        if multi_result_file.exists() and multi_result_file.is_file():
            file_answer = multi_result_file.read_text(encoding="utf-8", errors="replace").strip()
        elif result_file.exists() and result_file.is_file():
            file_answer = result_file.read_text(encoding="utf-8", errors="replace").strip()

        logger.info("[rollout=%s] Answer file %s: %s", rollout_key, "found" if file_answer else "NOT found", multi_result_file)
        if tracer:
            tracer.log("reward_answer_file", exists=bool(file_answer), content=file_answer[:500] if file_answer else "")

        if file_answer:
            solution_str = file_answer
        else:
            solution_str = ""

        ground_truth = sample.label if sample.label is not None else ""
        multi_predictions = _split_multi_solution(solution_str)
        multi_ground_truth = _split_multi_solution(ground_truth)

        if len(multi_ground_truth) > 1:
            per_problem_results = []
            for idx, gold in enumerate(multi_ground_truth):
                pred = multi_predictions[idx] if idx < len(multi_predictions) else ""
                per_problem_results.append(_compute_bigmath_score(pred, gold))

            avg_score = sum(item["score"] for item in per_problem_results) / len(per_problem_results)
            result = {
                "score": avg_score,
                "pred": SOLUTION_DELIMITER.join(str(item["pred"] or "") for item in per_problem_results),
                "per_problem_results": per_problem_results,
            }
        else:
            result = _compute_bigmath_score(solution_str, ground_truth)
            if result["pred"] is None:
                result["pred"] = ""

        logger.info("[rollout=%s] Reward: score=%s, pred=%.100s, ground_truth=%.100s", rollout_key, result["score"], str(result["pred"]), str(ground_truth))
        if tracer:
            tracer.log("reward_computed", score=result["score"], pred=str(result["pred"])[:200], ground_truth=str(ground_truth)[:200])

        result["reward_result_file"] = str(multi_result_file if multi_result_file.exists() else result_file)
        result["reward_result_content"] = file_answer
        merge_message = tool_registry.finalize_rollout(rollout_key=rollout_key, reward=result["score"])
        result["merge_message"] = merge_message

        logger.info("[rollout=%s] Finalize: %s", rollout_key, merge_message)
        if tracer:
            tracer.log("merge_result", message=merge_message)

    return result

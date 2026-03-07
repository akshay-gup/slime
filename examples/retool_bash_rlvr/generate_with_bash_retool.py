import re
from pathlib import Path
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

from bash_tool_sandbox import TOOL_CONFIGS, tool_registry

REWARD_RESULT_FILE = "answer.md"
PROBLEM_FILE = TOOL_CONFIGS["problem_file"]

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
                "You are working in an environment. "
                "Read the files in the current directory to understand what needs to be done. "
                "Use bash for any computation or inspection you need. "
                f"Write your final answer to `{REWARD_RESULT_FILE}` using the format: Answer: \\boxed{{...}} "
                "You may create, modify, and organize files in this workspace. "
                "Useful scripts, utilities, or notes you leave behind persist across tasks. "
                "Structure the workspace however helps you work best."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return template.render(messages=messages, tools=tools or [])


def postprocess_predictions(prediction: str):
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

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
                    return "bash", command
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    return None, ""


async def execute_predictions(prediction: str, rollout_key: str | int | None) -> tuple[str, bool]:
    action, content = postprocess_predictions(prediction)

    if action == "bash":
        result = await tool_registry.execute_tool("bash", {"command": content}, rollout_key=rollout_key)
        return f"\n\n<tool_response>\n{result}\n</tool_response>\n\n", False
    if action == "answer":
        return "", True

    return (
        "\nMy previous action is invalid. If I want to use the tool, I should emit "
        "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"...\"}}</tool_call>. "
        "If I want to finish, I should use Answer: \\boxed{...}. Let me try again.\n",
        False,
    )


def _has_file_change(tool_response: str) -> bool:
    return "Files changed: yes" in tool_response


def _resolve_rollout_key(sample: Sample) -> str | int | None:
    if TOOL_CONFIGS.get("shared_workspace_across_prompts", True):
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

    async with rollout_lock:
        tool_registry.prepare_rollout(rollout_key)
        tool_registry.write_problem_file(rollout_key=rollout_key, problem_text=sample.prompt)
        prompt = format_conversation_with_tools(prompt="Please work on the task in the environment.", tools=tool_specs)

        prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response = ""
        response_token_ids = []
        loss_masks = []
        context_response_token_ids = []
        archived_context_response_token_ids = []
        tool_call_count = 0
        saw_length_stop = False

        for _ in range(TOOL_CONFIGS["max_turns"]):
            current_token_ids = prompt_tokens_ids + context_response_token_ids
            payload = {
                "input_ids": current_token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            output = await post(url, payload)
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs

            response += cur_response
            response_token_ids += cur_response_token_ids
            context_response_token_ids += cur_response_token_ids
            loss_masks += [1] * len(cur_response_token_ids)

            if output["meta_info"]["finish_reason"]["type"] == "length":
                saw_length_stop = True
                break

            next_obs, done = await execute_predictions(cur_response, rollout_key=rollout_key)
            if done:
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
                context_response_token_ids = _archive_and_reset_context_tokens(
                    context_response_token_ids, archived_context_response_token_ids
                )

            if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
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

    return sample


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    rollout_key = _resolve_rollout_key(sample)
    rollout_lock = tool_registry.get_rollout_lock(rollout_key)

    async with rollout_lock:
        rollout_dir = tool_registry._resolve_rollout_workdir(rollout_key)
        result_file = Path(rollout_dir) / REWARD_RESULT_FILE
        file_answer = ""
        if result_file.exists() and result_file.is_file():
            file_answer = result_file.read_text(encoding="utf-8", errors="replace").strip()

        if file_answer:
            solution_str = f"Answer: \\boxed{{{file_answer}}}"
        else:
            solution_str = ""

        ground_truth = sample.label if sample.label is not None else ""
        result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
        if result["pred"] is None:
            result["pred"] = ""

        result["reward_result_file"] = str(result_file)
        result["reward_result_content"] = file_answer
        merge_message = tool_registry.finalize_rollout(rollout_key=rollout_key, reward=result["score"])
        result["merge_message"] = merge_message

    return result

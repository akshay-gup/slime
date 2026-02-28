import re
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
                "You are a helpful assistant. Use the bash tool for computations, file inspection, "
                "and shell-based reasoning when useful. Return final answers using Answer: \\boxed{...}."
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

            tool_call_data = json.loads(tool_call_match.group(1).replace("\n", "\\n"))
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})
            if tool_name == "bash":
                command = arguments.get("command", "")
                if command.strip():
                    return "bash", command
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    return None, ""


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    action, content = postprocess_predictions(prediction)

    if action == "bash":
        result = await tool_registry.execute_tool("bash", {"command": content})
        return f"\n\n<tool_response>\n{result}\n</tool_response>\n\n", False
    if action == "answer":
        return "", True

    return (
        "\nMy previous action is invalid. If I want to use the tool, I should emit "
        "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"...\"}}</tool_call>. "
        "If I want to finish, I should use Answer: \\boxed{...}. Let me try again.\n",
        False,
    )


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0

    for _ in range(TOOL_CONFIGS["max_turns"]):
        current_token_ids = prompt_tokens_ids + response_token_ids
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
        loss_masks += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        if "<tool_response>" in next_obs:
            tool_call_count += 1

        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)
        sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.tool_call_count = tool_call_count

    if output["meta_info"]["finish_reason"]["type"] == "length":
        sample.status = Sample.Status.TRUNCATED
    elif output["meta_info"]["finish_reason"]["type"] == "stop":
        sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    solution_str = sample.prompt + sample.response
    ground_truth = sample.label if sample.label is not None else ""
    num_turns = getattr(sample, "tool_call_count", 0)

    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)
    if result["pred"] is None:
        result["pred"] = ""
    return result

"""
Bash tool RLVR (Reinforcement Learning with Verifiable Rewards) for slime.

Gives the model a bash tool and trains it via multi-turn RL to solve
coding/debugging tasks by executing shell commands and observing outputs.

The model interacts with a repository in an isolated working directory,
issuing bash commands to explore code, run tests, apply patches, etc.
A verifiable reward is computed based on whether the task is solved
(e.g., tests pass, expected output matches).

Usage in slime training config:
  --custom-generate-function-path generate_with_bash.generate
  --custom-rm-path generate_with_bash.reward_func
"""

import json
import logging
import re
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from bash_sandbox import BASH_CONFIGS, SEMAPHORE, BashExecutor, tool_registry

logger = logging.getLogger(__name__)

# Jinja2 template for bash-tool-enabled conversations (Qwen3 chat format)
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

SYSTEM_PROMPT = """\
You are a skilled software engineer with access to a bash shell. \
You can run any command to explore the repository, understand the code, \
run tests, and make changes. Use the bash tool to execute commands.

When you have completed the task, respond with your final answer using:
Answer: \\boxed{done}

If you determine the task cannot be completed, respond with:
Answer: \\boxed{failed}"""


def format_conversation_with_tools(
    prompt: str,
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> str:
    """Format conversation using Jinja2 template with bash tool support."""
    template = Template(TOOL_TEMPLATE)

    messages_to_render = []
    messages_to_render.append({
        "role": "system",
        "content": system_prompt or SYSTEM_PROMPT,
    })

    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    if messages:
        messages_to_render.extend(messages)

    return template.render(messages=messages_to_render, tools=tools or [])


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    """
    Extract action type and content from model prediction.

    Returns:
        (action, content) where action is "bash", "answer", or None
    """
    # Check for Answer: \boxed{...} format
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    # Check for <tool_call> tags with bash tool
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            json_str = tool_call_match.group(1)
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "bash":
                command = arguments.get("command", "")
                if command.strip():
                    return "bash", command
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    # Check for ```bash or ```sh code blocks as fallback
    bash_pattern = r"```(?:bash|sh)\s*(.*?)\s*```"
    bash_match = re.search(bash_pattern, prediction, re.DOTALL)
    if bash_match:
        content = bash_match.group(1).strip()
        return "bash", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness."""
    # Handle <tool_call> tags
    if "<tool_call>" in resp:
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle ```bash code blocks
    if "```bash" in resp or "```sh" in resp:
        bash_pattern = r"```(?:bash|sh)\s*.*?```"
        matches = list(re.finditer(bash_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle Answer: \boxed{...}
    if "Answer:" in resp and "\\boxed{" in resp:
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp


async def execute_predictions(prediction: str, executor: BashExecutor) -> tuple[str, bool]:
    """
    Execute predictions: run bash commands or detect final answers.

    Returns:
        (next_observation, done) tuple
    """
    action, content = postprocess_predictions(prediction)

    if action == "bash":
        command = content.strip()
        if command:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("bash", {"command": command}, executor)
            next_obs = f"\n\n<bash_output>\n{result}\n</bash_output>\n\n"
            done = False
        else:
            next_obs = "\n\n<bash_output>\nError: empty command\n</bash_output>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "To execute a bash command, I should use the bash tool via <tool_call> tags. "
            "To give the final answer, I should use 'Answer: \\boxed{done}' or 'Answer: \\boxed{failed}'. "
            "Let me try again.\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Custom generation function supporting bash tool calls.

    This drives the multi-turn interaction loop:
    1. Format prompt with bash tool schema
    2. Generate model response
    3. Parse tool calls or final answer
    4. Execute bash commands, feed output back
    5. Repeat until done or max turns
    """
    assert not args.partial_rollout, "Partial rollout is not supported for bash tool rollouts."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Create a bash executor for this rollout
    executor = BashExecutor()
    if not executor.acquire_slot():
        logger.warning("All bash slots are busy, using default slot")

    try:
        # Set up initial prompt with system prompt and bash tool schema
        tool_specs = tool_registry.get_tool_specs()
        prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

        prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response = ""
        response_token_ids = []
        loss_masks = []
        tool_call_count = 0

        for turn in range(BASH_CONFIGS["max_turns"]):
            # Check total length against max context
            total_length = len(prompt_tokens_ids) + len(response_token_ids)
            if args.rollout_max_context_len is not None:
                max_context_length = args.rollout_max_context_len
            else:
                max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
            if total_length >= max_context_length:
                sample.status = Sample.Status.TRUNCATED
                break

            current_token_ids = prompt_tokens_ids + response_token_ids
            payload = {
                "input_ids": current_token_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            output = await post(url, payload)

            # Handle abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            if "output_token_logprobs" in output["meta_info"]:
                cur_response_token_ids = [
                    item[1] for item in output["meta_info"]["output_token_logprobs"]
                ]
                cur_response = state.tokenizer.decode(cur_response_token_ids)
                cur_log_probs = [
                    item[0] for item in output["meta_info"]["output_token_logprobs"]
                ]
                if sample.rollout_log_probs is None:
                    sample.rollout_log_probs = []
                sample.rollout_log_probs += cur_log_probs
            else:
                cur_response = output["text"]
                cur_response = postprocess_responses(cur_response)
                cur_response_token_ids = state.tokenizer(
                    cur_response, add_special_tokens=False
                )["input_ids"]

            response += cur_response
            response_token_ids += cur_response_token_ids
            loss_masks += [1] * len(cur_response_token_ids)

            # Check length limit
            if output["meta_info"]["finish_reason"]["type"] == "length":
                break

            next_obs, done = await execute_predictions(cur_response, executor)
            if done:
                break

            # Count tool calls
            if "<bash_output>" in next_obs:
                tool_call_count += 1

            assert next_obs != "", "Next observation should not be empty."
            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response += next_obs
            response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)

            # Add dummy log probs for observation tokens (masked out by loss_mask=0)
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
                assert len(response_token_ids) == len(sample.rollout_log_probs), (
                    f"Token/logp length mismatch at turn {turn}: "
                    f"{len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"
                )

            if tool_call_count >= BASH_CONFIGS["max_tool_calls"]:
                break

    finally:
        executor.release_slot()

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.tool_call_count = tool_call_count

    # Set status from last output
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> dict[str, Any]:
    """
    Reward function for bash tool RLVR.

    The reward is based on verifiable outcomes:
    - If the label contains a test command, run it and check exit code
    - If the label is a string, check if the model's answer matches
    - Bonus/penalty for tool usage patterns

    Expected label format (JSON string):
        {"test_command": "pytest tests/", "expected_exit_code": 0}
    or simply a string answer to match against.
    """
    if not isinstance(sample, Sample):
        raise TypeError("sample must be an instance of Sample class.")

    response = sample.response
    ground_truth = sample.label if sample.label is not None else ""
    num_turns = getattr(sample, "tool_call_count", 0)

    # Try to parse label as JSON with test_command
    test_config = None
    if ground_truth:
        try:
            test_config = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            pass

    score = 0.0
    pred = ""

    if test_config and isinstance(test_config, dict) and "test_command" in test_config:
        # Verification-based reward: run the test command in the executor's workdir
        # and check the exit code
        test_command = test_config["test_command"]
        expected_exit_code = test_config.get("expected_exit_code", 0)

        executor = BashExecutor()
        if executor.acquire_slot():
            try:
                output, success = await executor.execute(test_command)
                if success == (expected_exit_code == 0):
                    score = 1.0
                    pred = "pass"
                else:
                    score = 0.0
                    pred = f"fail: {output[:200]}"
            finally:
                executor.release_slot()
        else:
            score = 0.0
            pred = "no_slot"
    else:
        # String matching: extract \boxed{...} answer and compare
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        match = re.search(answer_pattern, response, re.DOTALL)
        if match:
            pred = match.group(1).strip()
            if pred.lower() == ground_truth.strip().lower():
                score = 1.0
            else:
                score = 0.0
        else:
            pred = ""
            score = -1.0  # No answer given

    # Encourage tool use: small bonus for using bash, penalty for not using it at all
    if score <= 0 and num_turns > 0:
        tool_bonus = min(num_turns, 5) * 0.02  # up to +0.1 for trying
        score = max(score + tool_bonus, -0.5)

    return {"score": score, "pred": pred}

# Retool Bash RLVR Example

This example mirrors `examples/retool`, but exposes a **bash** tool instead of a Python code interpreter.
It is intended for multi-turn RLVR-style training where the model can inspect files, run shell commands,
and then provide a final `Answer: \boxed{...}` response.

## Files

- `generate_with_bash_retool.py`: Custom rollout + reward function for bash tool calls.
- `custom_convert_samples_to_train_data.py`: Splits trajectories at context resets and expands train samples with divided rewards.
- `bash_tool_sandbox.py`: Bash tool registry and command execution safeguards.
- `retool_bash_qwen3_4b_rlvr.sh`: Launch script template for RLVR training.

## Tool format

The model receives a tool schema and should emit calls like:

```xml
<tool_call>
{"name": "bash", "arguments": {"command": "python -c 'print(2+2)'"}}
</tool_call>
```

Tool outputs are appended as:

```xml
<tool_response>
...
</tool_response>
```

## Safety controls

`bash_tool_sandbox.py` blocks several dangerous command patterns and enforces timeout/output truncation:

- timeout (`bash_timeout`)
- output cap (`max_output_chars`)
- blocked substrings (e.g. `rm -rf /`, `mkfs`)
- default workspace root is `/opt/NeMo/slime_bash_tool_workspace` (override with `SLIME_BASH_TOOL_WORKDIR`)

## Rollout workspace merge policy

Each rollout runs in an isolated copy of `workdir/main`. After reward is computed:

- reward `> 0`: changes are merged back into `workdir/main`, auto-merged with `git merge-file` when possible, then committed.
- reward `<= 0`: all rollout workspace changes are discarded.
- merge conflicts: both sides are kept as `*.main` and `*.rollout` companion files.

By default (`shared_workspace_across_prompts=True`), all prompts share one bash workspace lineage:

- at rollout start, the current sample problem is written to `task.md` inside the rollout workspace
- system prompt only gives general environment instructions; the model reads `task.md` via bash commands
- before merge/discard in reward finalization, `task.md` is removed from rollout/base/main workspaces


- every sample starts by refreshing its rollout copy from the latest `workdir/main`
- samples are sharded across rollout slots (`SLIME_BASH_NUM_ROLLOUT_ENVS`, default `8`) so multiple GPUs can execute tool rollouts concurrently
- split/merge logic still happens per sample (branch, score, merge-or-discard)

By default, `retool_bash_qwen3_4b_rlvr.sh` sets `SLIME_BASH_NUM_ROLLOUT_ENVS=${NUM_GPUS}` so one rollout slot is available per GPU.

This behavior is implemented in `bash_tool_sandbox.py` via `ToolRegistry.prepare_rollout()` and `ToolRegistry.finalize_rollout()`, called from `generate()` and `reward_func()`.

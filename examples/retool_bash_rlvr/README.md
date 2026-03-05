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

## Rollout workspace merge policy

Each rollout runs in an isolated copy of `workdir/main`. After reward is computed:

- reward `> 0`: changes are merged back into `workdir/main`, auto-merged with `git merge-file` when possible, then committed.
- reward `<= 0`: all rollout workspace changes are discarded.
- merge conflicts: both sides are kept as `*.main` and `*.rollout` companion files.

This behavior is implemented in `bash_tool_sandbox.py` via `ToolRegistry.finalize_rollout()`, called from `reward_func()`.

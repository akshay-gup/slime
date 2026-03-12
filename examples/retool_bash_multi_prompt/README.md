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

By default (`shared_workspace_across_prompts=False`), each sample uses an isolated rollout workspace lineage:

- every sample starts by refreshing its rollout copy from the latest `workdir/main`
- each rollout key gets a dedicated workspace directory and lock (no shared slot hashing collisions)
- split/merge logic still happens per sample (branch, score, merge-or-discard)

When shared lineage is enabled (`shared_workspace_across_prompts=True`):

- at rollout start, the current sample problem is written to `task.md` inside the rollout workspace
- system prompt only gives general environment instructions; the model reads `task.md` via bash commands
- before merge/discard in reward finalization, `task.md` is removed from rollout/base/main workspaces

`retool_bash_qwen3_4b_rlvr.sh` is tuned for a single-node 4xH100 setup with a 4B model:

- fixed `NUM_GPUS=4` (no auto detection)
- fixed `NUM_GPUS_PER_NODE=4` and passes `--num-gpus-per-node 4` explicitly under `--colocate`
- `--rollout-num-gpus-per-engine 1` to run one rollout engine per GPU
- always builds a multi-problem parquet at launch using `data_utils.build_verl_parquet_openr1_bigmath_multi` with `OPEN_R1_MULTI_SUBSET`, optional `OPEN_R1_MULTI_DOMAIN`, `PROBLEMS_PER_PROMPT`, and output path `PROMPT_DATA_FILE`
- rollout workspaces are created per-key and cleaned up after finalization to avoid disk growth
- memory-safer defaults for 4xH100 RLVR (`--max-tokens-per-gpu 5120`, `--rollout-max-response-len 4096`, `--sglang-mem-fraction-static 0.4`)
- these limits can be overridden via env vars (`MAX_TOKENS_PER_GPU`, `ROLLOUT_MAX_RESPONSE_LEN`) when debugging throughput vs. stability

The launcher now fails fast if conflicting environment overrides are present (for example,
`ACTOR_NUM_GPUS_PER_NODE!=4`, `NUM_GPUS_PER_NODE!=4`, or `ROLLOUT_NUM_GPUS_PER_ENGINE!=1`) so
4-GPU behavior stays explicit.

The current launch template does not include eval arguments; it focuses on training rollouts only.

This behavior is implemented in `bash_tool_sandbox.py` via `ToolRegistry.prepare_rollout()` and `ToolRegistry.finalize_rollout()`, called from `generate()` and `reward_func()`.

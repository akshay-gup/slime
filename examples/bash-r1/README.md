# Bash-R1 lite

This example mirrors the **Search-R1** multi-turn setup, but replaces search with a single `bash` tool.

The model is expected to emit one of two XML actions:

- `<bash>...</bash>` to execute a shell command
- `<answer>...</answer>` to finish the episode

## Key files

- `generate_with_bash.py`: custom rollout + reward function

## Configuration

In your training script, point slime to this custom generator/reward pair:

```bash
CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_bash.generate
  --custom-rm-path generate_with_bash.reward_func
)
```

Inside `generate_with_bash.py`, configure behavior through `BASH_R1_CONFIGS`:

- `max_turns`: max assistant turns per trajectory
- `timeout`: per-command timeout in seconds
- `max_output_chars`: truncation limit for tool output
- `blocked_patterns`: simple safety deny-list
- `tool_concurrency`: max concurrent bash executions

## Notes

- Commands run with `asyncio.create_subprocess_shell`.
- Tool output is fed back into the dialogue via `<tool_response>...</tool_response>`.
- Invalid model actions are corrected with an in-context retry message.

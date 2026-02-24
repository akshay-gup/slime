# Bash Tool + Context Reset Example

This example shows how to build an RL rollout where:

1. The model can call a bash tool using `<bash>...</bash>`.
2. After any file write/create event, model-visible context is reset.
3. Reward is computed from terminal state, then redistributed across steps.

## Files

- `generate_with_bash_reset.py`: custom generate + reward hooks.

## Key Ideas

- Multi-turn loop follows Search-R1 / Retool style:
  - assistant tokens get `loss_mask=1`
  - tool observation tokens get `loss_mask=0`
- Write event detection is done by filesystem snapshot diff before/after a bash command.
- On write event, this example rebuilds a compact prompt and clears previous context.
- Reward hook returns terminal reward and per-step redistributed rewards.

## Configure in training

```bash
CUSTOM_ARGS=(
  --custom-generate-function-path examples.bash_reset_context.generate_with_bash_reset.generate
  --custom-rm-path examples.bash_reset_context.generate_with_bash_reset.reward_func
)
```

## Notes

- This is a reference implementation; replace `_run_bash` and terminal scoring with your production sandbox and evaluator.
- The example stores step metadata in `sample.metadata["step_records"]`.

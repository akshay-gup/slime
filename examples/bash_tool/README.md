# Bash Tool RLVR

Train language models to use bash via multi-turn reinforcement learning with verifiable rewards (RLVR).

## Overview

This example gives the model a single **bash** tool and trains it via GRPO to solve coding and debugging tasks by executing shell commands in an isolated sandbox. Rewards are computed by running verifiable test commands (e.g., `pytest`, `grep`, exit code checks).

Key differences from the [retool](../retool/) example:
- **bash** tool instead of `code_interpreter` (Python sandbox)
- Reward based on running a **test command** with an expected exit code, not boxed math answers
- Slot-based working directory isolation for parallel rollouts

## Files

| File | Description |
|------|-------------|
| `bash_sandbox.py` | Bash execution sandbox with slot pool, safety checks, and output truncation |
| `generate_with_bash.py` | Multi-turn generation loop and reward function |
| `rl_data_preprocess.py` | Data preprocessing utilities |
| `bash_tool_qwen3_4b_rl.sh` | Training launch script |

## Quick Start

### 1. Install dependencies

```bash
cd slime
pip install -e . --no-deps
pip install -r examples/bash_tool/requirements.txt
```

### 2. Prepare model

```bash
# Download model
hf download Qwen/Qwen3-4B --local-dir /root/models/qwen3-4b

# Convert to torch distributed format
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/qwen3-4b \
    --rotary-base 5000000 \
    --save /root/models/qwen3-4b_torch_dist
```

### 3. Prepare data

Generate example data for testing:
```bash
python examples/bash_tool/rl_data_preprocess.py --mode example --output /root/data/bash_tool_tasks.jsonl
```

Or convert your own dataset:
```bash
# SWE-bench style (JSONL with problem_statement + test_cmd fields)
python examples/bash_tool/rl_data_preprocess.py --mode swebench \
    --input /root/data/swebench_raw.jsonl \
    --output /root/data/bash_tool_tasks.jsonl
```

### 4. Train

```bash
bash examples/bash_tool/bash_tool_qwen3_4b_rl.sh
```

## Data Format

Training data is JSONL with `prompt` and `label` fields:

```jsonl
{"prompt": "Create a Python file hello.py that prints 'Hello, World!'", "label": "{\"test_command\": \"python hello.py | grep -q 'Hello, World!'\", \"expected_exit_code\": 0}"}
{"prompt": "Fix the bug in sort.py so it handles empty input", "label": "{\"test_command\": \"pytest tests/test_sort.py\", \"expected_exit_code\": 0}"}
```

The `label` is either:
- **JSON string** with `test_command` and optional `expected_exit_code` (default 0) for verification-based rewards
- **Plain string** for simple answer matching via `Answer: \boxed{...}`

## Tool Format

The model sees the bash tool via the standard OpenAI function calling format:

```
<tools>
{"type": "function", "function": {"name": "bash", "description": "Run a command in a bash shell...", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
</tools>
```

The model calls it with:
```
<tool_call>
{"name": "bash", "arguments": {"command": "python hello.py"}}
</tool_call>
```

And receives output in:
```
<bash_output>
Hello, World!
</bash_output>
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  generate_with_bash.py                          │
│                                                 │
│  generate() ──► format prompt with bash tool    │
│       │         schema (Jinja2 template)        │
│       │                                         │
│       ▼                                         │
│  multi-turn loop:                               │
│    1. LLM generates response                    │
│    2. Parse <tool_call> or Answer: \boxed{...}  │
│    3. Execute bash command in sandbox slot       │
│    4. Feed <bash_output> back to LLM            │
│    5. Repeat until done or max_turns            │
│                                                 │
│  reward_func() ──► run test_command from label  │
│                    check exit code → reward      │
└─────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│  bash_sandbox.py                                │
│                                                 │
│  BashSlotPool: N isolated workdirs for parallel │
│  BashExecutor: async subprocess with timeout    │
│  BashToolRegistry: OpenAI tool schema           │
│  Safety: blocked patterns, output truncation    │
└─────────────────────────────────────────────────┘
```

## Configuration

Key settings in `bash_sandbox.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_turns` | 16 | Max interaction rounds per sample |
| `max_tool_calls` | 16 | Max bash executions per sample |
| `timeout` | 30 | Seconds before command is killed |
| `max_output_chars` | 8192 | Truncate output beyond this |
| `num_slots` | 8 | Parallel isolated workdirs |
| `blocked_patterns` | see code | Commands blocked for safety |

## Safety

- Each rollout runs in an isolated slot directory
- Dangerous commands are blocked (`rm -rf /`, fork bombs, `mkfs`, etc.)
- Commands time out after 30s
- Output is truncated to prevent context window overflow
- `GIT_TERMINAL_PROMPT=0` prevents interactive git prompts

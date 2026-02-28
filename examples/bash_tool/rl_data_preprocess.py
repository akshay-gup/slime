"""
Data preprocessing for bash tool RLVR training.

Converts coding task datasets into the format expected by slime:
  {"prompt": "<task description>", "label": "<verification config>"}

The label can be either:
  - A JSON string: {"test_command": "pytest tests/", "expected_exit_code": 0}
  - A plain string answer for simple matching

Example input datasets:
  - SWE-bench style: repo + issue description + test patch
  - Coding challenges: problem statement + test cases
  - Custom: any task with a verifiable bash command
"""

import json


def make_bash_task(
    task_description: str,
    test_command: str | None = None,
    expected_exit_code: int = 0,
    answer: str | None = None,
) -> dict[str, str]:
    """
    Create a single training sample for bash tool RLVR.

    Args:
        task_description: Natural language description of the task.
        test_command: Bash command to verify the solution (e.g., "pytest tests/").
        expected_exit_code: Expected exit code of test_command (default 0).
        answer: Plain string answer (used if test_command is None).

    Returns:
        Dict with "prompt" and "label" keys.
    """
    if test_command is not None:
        label = json.dumps({
            "test_command": test_command,
            "expected_exit_code": expected_exit_code,
        })
    elif answer is not None:
        label = answer
    else:
        label = "done"

    return {"prompt": task_description, "label": label}


def preprocess_swebench_style(input_path: str, output_path: str) -> None:
    """
    Convert a SWE-bench style dataset to bash tool RLVR format.

    Expected input format (JSONL):
        {"problem_statement": "...", "test_cmd": "pytest ...", "repo": "..."}
    """
    samples = []
    with open(input_path) as f:
        for line in f:
            item = json.loads(line)
            prompt = item.get("problem_statement", item.get("prompt", ""))
            test_cmd = item.get("test_cmd", item.get("test_command", None))

            sample = make_bash_task(
                task_description=prompt,
                test_command=test_cmd,
            )
            samples.append(sample)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(samples)} samples to {output_path}")


def preprocess_coding_challenges(input_path: str, output_path: str) -> None:
    """
    Convert coding challenge datasets to bash tool RLVR format.

    Expected input format (JSONL):
        {"prompt": "Write a function ...", "answer": "done", "test_code": "python -c '...'"}
    """
    samples = []
    with open(input_path) as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]
            test_code = item.get("test_code")
            answer = item.get("answer", "done")

            sample = make_bash_task(
                task_description=prompt,
                test_command=test_code,
                answer=answer if test_code is None else None,
            )
            samples.append(sample)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(samples)} samples to {output_path}")


def generate_example_data(output_path: str, num_samples: int = 100) -> None:
    """Generate example training data for quick testing."""
    import random

    tasks = [
        {
            "prompt": "Create a Python file called hello.py that prints 'Hello, World!'",
            "test_command": "python hello.py | grep -q 'Hello, World!'",
        },
        {
            "prompt": "Create a bash script called count.sh that counts from 1 to 10, one number per line.",
            "test_command": "bash count.sh | wc -l | grep -q '10'",
        },
        {
            "prompt": "Create a file called fibonacci.py that prints the first 10 Fibonacci numbers separated by spaces.",
            "test_command": "python fibonacci.py | grep -q '0 1 1 2 3 5 8 13 21 34'",
        },
        {
            "prompt": "Write a Python script sort_numbers.py that reads numbers from stdin (one per line) and prints them sorted.",
            "test_command": "echo -e '3\\n1\\n2' | python sort_numbers.py | tr '\\n' ' ' | grep -q '1 2 3'",
        },
        {
            "prompt": "Create a Python file called reverse.py that takes a string argument and prints it reversed.",
            "test_command": "python reverse.py hello | grep -q 'olleh'",
        },
    ]

    samples = []
    for _ in range(num_samples):
        task = random.choice(tasks)
        sample = make_bash_task(
            task_description=task["prompt"],
            test_command=task["test_command"],
        )
        samples.append(sample)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(samples)} example samples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data for bash tool RLVR")
    parser.add_argument("--mode", choices=["swebench", "coding", "example"], default="example")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default="/root/data/bash_tool_tasks.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "swebench":
        assert args.input, "--input required for swebench mode"
        preprocess_swebench_style(args.input, args.output)
    elif args.mode == "coding":
        assert args.input, "--input required for coding mode"
        preprocess_coding_challenges(args.input, args.output)
    elif args.mode == "example":
        generate_example_data(args.output, args.num_samples)

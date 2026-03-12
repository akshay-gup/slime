from datasets import load_dataset, Dataset


PROBLEM_DELIMITER = "\n\n=== PROBLEM DELIMITER ===\n\n"
SOLUTION_DELIMITER = "\n\n=== SOLUTION DELIMITER ===\n\n"


def build_math_multi_prompt(questions: list[str]) -> str:
    return PROBLEM_DELIMITER.join(q.strip() for q in questions)


def build_math_multi_solution(solutions: list[str]) -> str:
    return SOLUTION_DELIMITER.join(str(s).strip() for s in solutions)


def build_record(questions: list[str], solutions: list[str]) -> dict:
    return {
        "prompt": build_math_multi_prompt(questions),
        "solution": build_math_multi_solution(solutions),
    }


def build_verl_parquet_openr1_bigmath_multi(
    subset="level_5",
    problems_per_prompt=5,
    domain: str | None = None,
) -> Dataset:
    raw_ds = load_dataset(
        "open-r1/Big-Math-RL-Verified-Processed",
        subset,
        split="train",
    )

    if domain is not None:
        raw_ds = raw_ds.filter(lambda ex: ex.get("domain") == domain)

    if len(raw_ds) == 0:
        raise ValueError(
            f"No samples found for subset={subset!r}"
            + (f" and domain={domain!r}" if domain is not None else "")
        )

    all_questions = [ex["prompt"] for ex in raw_ds]
    all_solutions = [ex["solution"] for ex in raw_ds]

    records = []
    for start in range(0, len(all_questions), problems_per_prompt):
        end = start + problems_per_prompt
        records.append(
            build_record(
                questions=all_questions[start:end],
                solutions=all_solutions[start:end],
            )
        )

    return Dataset.from_list(records)

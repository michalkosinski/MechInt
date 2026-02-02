### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###
### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###
### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###

"""
Behavioral analysis for Theory of Mind experiments.

Response parsing and evaluation logic. Uses model_runner for inference.
"""

import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import List

from .task_generator import ToMTask
from .model_runner import ModelRunner

# Simple task wrapper for model_runner (needs: task_id, task_type, full_prompt)
PromptTask = namedtuple("PromptTask", ["task_id", "task_type", "full_prompt"])

# Question templates applied to narrative at inference time
QUESTION_TEMPLATES = {
    "reality": "The {obj} is in the",
    "protagonist": "{protagonist} will look for the {obj} in the",
    "observer": "{observer} will look for the {obj} in the",
}

def build_prompt(task: ToMTask, question_type: str) -> str:
    """Build full prompt from narrative + question template."""
    template = QUESTION_TEMPLATES[question_type]
    question = template.format(
        obj=task.obj,
        protagonist=task.protagonist,
        observer=task.observer,
    )
    return f"{task.narrative} {question}"


def get_expected_answer(task: ToMTask, question_type: str) -> str:
    """Get expected answer based on question type."""
    if question_type == "reality":
        return task.final_location
    elif question_type == "protagonist":
        return task.protagonist_belief
    elif question_type == "observer":
        return task.final_location
    else:
        raise ValueError(f"Unknown question type: {question_type}")


def parse_response(raw: str, valid_containers: List[str]) -> str:
    """Parse model response to extract the predicted container."""
    if not raw or not raw.strip():
        return "[UNCERTAIN]"

    text = " ".join(raw.strip().lower().split())

    for c in valid_containers:
        if f"or the {c.lower()}" in text:
            return "[UNCERTAIN]"

    first_pos = float('inf')
    first_container = None
    for c in valid_containers:
        pos = text.find(c.lower())
        if pos != -1 and pos < first_pos:
            first_pos = pos
            first_container = c

    return first_container if first_container else "[UNCERTAIN]"


def run_behavioral_analysis(
    runner: ModelRunner,
    tasks: List[ToMTask],
    question_type: str = "protagonist",
    max_new_tokens: int = 20,
) -> tuple[List[dict], dict]:
    """Run behavioral analysis on tasks. Returns (results, summary)."""
    prompt_tasks = [
        PromptTask(t.task_id, t.task_type, build_prompt(t, question_type))
        for t in tasks
    ]

    # Run inference
    batch_output = runner.run_batch(
        prompt_tasks,
        max_new_tokens=max_new_tokens,
        extract_attention=False,
        stop_strings=[".", "?", "!"],
    )

    # Evaluate outputs
    task_lookup = {t.task_id: t for t in tasks}
    results = []
    for output in batch_output.outputs:
        task = task_lookup[output.task_id]
        containers = [task.initial_location, task.final_location]
        parsed = parse_response(output.model_response, containers)
        expected = get_expected_answer(task, question_type)

        results.append({
            "task_id": output.task_id,
            "task_type": output.task_type,
            "family_id": task.family_id,
            "question_type": question_type,
            "prompt": output.prompt,
            "response": output.model_response,
            "parsed_response": parsed,
            "expected": expected,
            "is_correct": parsed.lower() == expected.lower(),
            "is_uncertain": parsed == "[UNCERTAIN]",
        })

    # Compute summary
    correct = sum(1 for r in results if r["is_correct"])
    uncertain = sum(1 for r in results if r["is_uncertain"])
    fb_results = [r for r in results if r["task_type"] == "false_belief"]
    tb_results = [r for r in results if r["task_type"] == "true_belief"]

    summary = {
        "model_name": runner.model_name,
        "question_type": question_type,
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0,
        "uncertain": uncertain,
        "false_belief_accuracy": sum(1 for r in fb_results if r["is_correct"]) / len(fb_results) if fb_results else 0,
        "true_belief_accuracy": sum(1 for r in tb_results if r["is_correct"]) / len(tb_results) if tb_results else 0,
        "total_time_ms": batch_output.total_time_ms,
    }

    return results, summary


def save_results(results: List[dict], summary: dict, output_dir: Path) -> Path:
    """Save behavioral results to JSON file."""
    behavioral_dir = output_dir / "behavioral"
    behavioral_dir.mkdir(parents=True, exist_ok=True)

    model_slug = summary["model_name"].replace("/", "_")
    output_path = behavioral_dir / f"{model_slug}_{summary['question_type']}.json"

    data = {"summary": summary, "details": results}
    output_path.write_text(json.dumps(data, indent=2))
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run behavioral analysis on ToM tasks")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Model name")
    parser.add_argument("--tasks", type=int, default=None, help="Number of tasks to run (default: all)")
    parser.add_argument("--question", default="protagonist", choices=["protagonist", "reality", "observer"])
    args = parser.parse_args()

    from .task_generator import load_tasks

    tasks = load_tasks(Path(__file__).parent.parent / "tasks.json")
    if args.tasks is not None:
        tasks = tasks[:args.tasks]
    runner = ModelRunner(args.model)
    results, summary = run_behavioral_analysis(runner, tasks, question_type=args.question)
    output_path = save_results(results, summary, Path(__file__).parent.parent / "results")

    print(f"Model: {summary['model_name']}")
    print(f"Tasks: {summary['total_tasks']} | Correct: {summary['correct']}")
    print(f"Accuracy: {summary['accuracy']:.1%} (TB: {summary['true_belief_accuracy']:.1%}, FB: {summary['false_belief_accuracy']:.1%})")
    print(f"Saved: {output_path}")

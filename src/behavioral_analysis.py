### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###
### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###
### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###

"""
Behavioral analysis for Theory of Mind experiments.

Response parsing and evaluation logic. Uses model_runner for inference.

Usage:
    python -m src.behavioral_analysis
    python -m src.behavioral_analysis --model "Qwen/Qwen2.5-0.5B"
    python -m src.behavioral_analysis --tasks-file tasks.json --batch-size 8
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

from .task_generator import ToMTask, load_tasks
from .model_runner import ModelRunner, ModelOutput

# NOTE: Reasoning/thinking mode is ALWAYS disabled for behavioral analysis.
# Models like SmolLM3 support /think and /no_think modes - we use /no_think
# to ensure fast, direct responses without chain-of-thought reasoning.
# This keeps responses comparable across models and avoids confounding ToM
# results with reasoning capabilities.
DISABLE_REASONING_PREFIX = "/no_think "
REASONING_MODELS = ["SmolLM3", "smollm3"]  # Models that support thinking mode

def parse_response(raw: str, valid_containers: List[str]) -> str:
    """
    Parse model response to extract the predicted container.
    """
    if not raw or not raw.strip():
        return "[UNCERTAIN]"

    text = " ".join(raw.strip().lower().split())  # normalize whitespace

    # Expand synonyms in text for matching
    text = text.replace("cabinet", "cupboard")

    # Check for hedging: "or the [container]"
    for c in valid_containers:
        if f"or the {c.lower()}" in text:
            return "[UNCERTAIN]"

    # Find first mentioned container
    first_pos = float('inf')
    first_container = None
    for c in valid_containers:
        pos = text.find(c.lower())
        if pos != -1 and pos < first_pos:
            first_pos = pos
            first_container = c

    if first_container:
        return first_container

    # No container found → uncertain
    return "[UNCERTAIN]"


def evaluate_outputs(outputs: List[ModelOutput], tasks: List[ToMTask]) -> List[dict]:
    """
    Evaluate model outputs using proper response parsing.

    Args:
        outputs: List of ModelOutput from model_runner
        tasks: List of tasks (for container info)

    Returns:
        List of evaluation dicts with parsed responses
    """
    task_lookup = {t.task_id: t for t in tasks}

    results = []
    for output in outputs:
        task = task_lookup[output.task_id]
        parsed = parse_response(output.model_response, [task.c1, task.c2])

        results.append({
            "task_id": output.task_id,
            "task_type": output.task_type,
            "family_id": task.family_id,
            "prompt": output.prompt,
            "response": output.model_response,
            "parsed_response": parsed,
            "expected": output.expected_answer,
            "is_correct": parsed.lower() == task.expected_answer.lower(),
            "is_uncertain": parsed == "[UNCERTAIN]",
        })

    return results


def _should_disable_reasoning(model_name: str) -> bool:
    """Check if model supports thinking mode that should be disabled."""
    return any(rm.lower() in model_name.lower() for rm in REASONING_MODELS)


def _prepare_tasks_for_inference(tasks: List[ToMTask], model_name: str) -> List[ToMTask]:
    """Prepare tasks, disabling reasoning mode if needed."""
    if not _should_disable_reasoning(model_name):
        return tasks

    # Prepend /no_think to disable reasoning mode
    from dataclasses import replace
    return [
        replace(t, full_prompt=DISABLE_REASONING_PREFIX + t.full_prompt)
        for t in tasks
    ]


def run_behavioral_analysis(
    runner: ModelRunner,
    tasks: List[ToMTask],
    max_new_tokens: int = 20,
) -> tuple[List[dict], dict]:
    """
    Run behavioral analysis on tasks.

    Args:
        runner: ModelRunner instance
        tasks: List of ToM tasks
        max_new_tokens: Max tokens to generate

    Returns:
        Tuple of (results list, summary dict)
    """
    # Disable reasoning mode for models that support it
    inference_tasks = _prepare_tasks_for_inference(tasks, runner.model_name)

    # Run inference (batch_size auto-optimized by runner)
    batch_output = runner.run_batch(
        inference_tasks,
        max_new_tokens=max_new_tokens,
        extract_attention=False,
        stop_strings=[".", "?", "!"],
    )

    # Evaluate with proper parsing
    results = evaluate_outputs(batch_output.outputs, tasks)

    # Compute summary
    correct = sum(1 for r in results if r["is_correct"])
    uncertain = sum(1 for r in results if r["is_uncertain"])

    fb_results = [r for r in results if r["task_type"] == "false_belief"]
    tb_results = [r for r in results if r["task_type"] == "true_belief"]

    fb_correct = sum(1 for r in fb_results if r["is_correct"])
    tb_correct = sum(1 for r in tb_results if r["is_correct"])

    summary = {
        "model_name": runner.model_name,
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0,
        "uncertain": uncertain,
        "false_belief_accuracy": fb_correct / len(fb_results) if fb_results else 0,
        "true_belief_accuracy": tb_correct / len(tb_results) if tb_results else 0,
        "total_time_ms": batch_output.total_time_ms,
    }

    return results, summary


def save_results(results: List[dict], summary: dict, output_dir: Path) -> Path:
    """Save behavioral results to JSON file."""
    behavioral_dir = output_dir / "behavioral"
    behavioral_dir.mkdir(parents=True, exist_ok=True)

    model_slug = summary["model_name"].replace("/", "_")
    output_path = behavioral_dir / f"{model_slug}.json"

    data = {
        "model": summary["model_name"],
        "timestamp": summary["timestamp"],
        "summary": {
            "overall_accuracy": summary["accuracy"],
            "false_belief_accuracy": summary["false_belief_accuracy"],
            "true_belief_accuracy": summary["true_belief_accuracy"],
            "num_tasks": summary["total_tasks"],
            "num_correct": summary["correct"],
            "num_uncertain": summary["uncertain"],
            "total_time_ms": summary["total_time_ms"],
        },
        "details": results,
    }

    output_path.write_text(json.dumps(data, indent=2))
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run behavioral analysis from command line."""
    parser = argparse.ArgumentParser(description="Run behavioral analysis on ToM tasks")
    parser.add_argument("--model", "-m", default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B"))
    parser.add_argument("--tasks-file", "-t", type=Path, default=Path("tasks.json"))
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("results"))
    parser.add_argument("--max-tokens", type=int, default=20)

    args = parser.parse_args()

    if not args.tasks_file.exists():
        print(f"Error: Tasks file not found: {args.tasks_file}")
        print("Run 'python -m src.task_generator' first to generate tasks.")
        return 1

    print(f"Loading tasks from {args.tasks_file}")
    tasks = load_tasks(args.tasks_file)
    print(f"Loaded {len(tasks)} tasks")

    # Load model and run analysis
    runner = ModelRunner(args.model)
    results, summary = run_behavioral_analysis(
        runner, tasks,
        max_new_tokens=args.max_tokens,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("BEHAVIORAL ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Model: {summary['model_name']}")
    print(f"Overall accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total_tasks']})")
    print(f"  False belief: {summary['false_belief_accuracy']:.1%}")
    print(f"  True belief:  {summary['true_belief_accuracy']:.1%}")
    print(f"  Uncertain:    {summary['uncertain']}")
    print(f"Total time: {summary['total_time_ms'] / 1000:.1f}s")

    # Save results
    output_path = save_results(results, summary, args.output_dir)
    print(f"\nResults saved to: {output_path}")

    # Show sample errors
    errors = [r for r in results if not r["is_correct"]]
    if errors:
        print(f"\nSample errors ({min(5, len(errors))} of {len(errors)}):")
        for r in errors[:5]:
            print(f"  [{r['task_type']}] {r['task_id']}: got '{r['response']}' → '{r['parsed_response']}', expected '{r['expected']}'")

    return 0


if __name__ == "__main__":
    exit(main())

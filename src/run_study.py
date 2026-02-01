"""
Main study orchestrator for MechInt.

Runs the behavioral analysis pipeline:
1. Load or generate tasks
2. Run model inference
3. Analyze behavioral results
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .task_generator import ToMTask, load_tasks
from .model_runner import ModelRunner, ModelOutput, BatchOutput
from .probing_analysis import ProbingConfig, run_probing_analysis


@dataclass
class StudyConfig:
    """Configuration for a MechInt study run."""
    model_name: str = "mistralai/Mistral-7B-v0.3"
    num_false_belief_tasks: int = 20
    num_true_belief_tasks: int = 20
    batch_size: int = 16
    max_new_tokens: int = 3
    results_dir: Path = Path("results")
    tasks_file: Path = Path("tasks.json")
    run_probing: bool = False
    probing_num_families: int = 100
    probing_seed: int = 42
    probing_filter_correct: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results_dir"] = str(self.results_dir)
        d["tasks_file"] = str(self.tasks_file)
        return d


@dataclass
class StudyResults:
    """Results from a study run."""
    config: StudyConfig
    timestamp: str
    behavioral_accuracy: float
    false_belief_accuracy: float
    true_belief_accuracy: float
    num_tasks: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


def run_behavioral_study(
    runner: ModelRunner,
    tasks: List[ToMTask],
    config: StudyConfig,
    results_dir: Path,
) -> Tuple[BatchOutput, List[ModelOutput]]:
    """
    Run behavioral verification.

    Tests whether the model can correctly answer ToM tasks.
    """
    print("\n" + "=" * 60)
    print("Behavioral Analysis")
    print("=" * 60)

    # Run all tasks with batching
    batch_output = runner.run_batch(
        tasks,
        max_new_tokens=config.max_new_tokens,
        stop_strings=[".", "?", "!"],
        batch_size=config.batch_size,
        extract_attention=False,
        attention_sample_size=0,
    )

    # Compute accuracy by task type
    fb_outputs = [o for o in batch_output.outputs if o.task_type == "false_belief"]
    tb_outputs = [o for o in batch_output.outputs if o.task_type == "true_belief"]

    fb_accuracy = sum(1 for o in fb_outputs if o.is_correct) / len(fb_outputs) if fb_outputs else 0
    tb_accuracy = sum(1 for o in tb_outputs if o.is_correct) / len(tb_outputs) if tb_outputs else 0

    print(f"\nResults:")
    print(f"  Overall accuracy: {batch_output.accuracy:.1%} ({batch_output.correct_count}/{len(tasks)})")
    print(f"  False belief accuracy: {fb_accuracy:.1%}")
    print(f"  True belief accuracy: {tb_accuracy:.1%}")
    print(f"  Total time: {batch_output.total_time_ms / 1000:.1f}s")

    # Save behavioral results
    behavioral_results = {
        "model": config.model_name,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "overall_accuracy": batch_output.accuracy,
            "false_belief_accuracy": fb_accuracy,
            "true_belief_accuracy": tb_accuracy,
            "num_tasks": len(tasks),
            "num_correct": batch_output.correct_count,
        },
        "details": [
            {
                "task_id": o.task_id,
                "task_type": o.task_type,
                "prompt": o.prompt,
                "response": o.model_response,
                "expected": o.expected_answer,
                "correct": o.is_correct,
            }
            for o in batch_output.outputs
        ],
    }

    behavioral_path = results_dir / "behavioral" / f"{config.model_name.replace('/', '_')}.json"
    behavioral_path.parent.mkdir(parents=True, exist_ok=True)
    behavioral_path.write_text(json.dumps(behavioral_results, indent=2))
    print(f"  Saved to: {behavioral_path}")

    return batch_output, batch_output.outputs


def run_study(config: StudyConfig) -> StudyResults:
    """
    Run the behavioral study pipeline.
    """
    print("\n" + "=" * 60)
    print("MechInt: Theory of Mind Behavioral Study")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Results dir: {config.results_dir}")

    required_model = "mistralai/Mistral-7B-v0.3"
    if config.model_name != required_model:
        raise ValueError(
            f"Study runs are restricted to {required_model}. "
            f"Got: {config.model_name}"
        )

    # Ensure directories exist
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks (do not generate new tasks here)
    if not config.tasks_file.exists():
        raise FileNotFoundError(
            f"Tasks file not found: {config.tasks_file}. "
            "Please provide an existing tasks.json."
        )

    print(f"\nLoading tasks from {config.tasks_file}")
    tasks = load_tasks(config.tasks_file)
    fb_count = sum(1 for t in tasks if t.task_type == "false_belief")
    tb_count = sum(1 for t in tasks if t.task_type == "true_belief")
    print(f"Tasks loaded: {fb_count} false belief + {tb_count} true belief ({len(tasks)} total)")

    # Load model
    runner = ModelRunner(config.model_name)

    # Run behavioral study
    batch_output, outputs = run_behavioral_study(runner, tasks, config, config.results_dir)

    fb_outputs = [o for o in outputs if o.task_type == "false_belief"]
    tb_outputs = [o for o in outputs if o.task_type == "true_belief"]
    fb_accuracy = sum(1 for o in fb_outputs if o.is_correct) / len(fb_outputs) if fb_outputs else 0
    tb_accuracy = sum(1 for o in tb_outputs if o.is_correct) / len(tb_outputs) if tb_outputs else 0

    # Create results
    results = StudyResults(
        config=config,
        timestamp=datetime.now().isoformat(),
        behavioral_accuracy=batch_output.accuracy,
        false_belief_accuracy=fb_accuracy,
        true_belief_accuracy=tb_accuracy,
        num_tasks=len(tasks),
    )

    # Optional probing analysis
    if config.run_probing:
        probing_config = ProbingConfig(
            num_families=config.probing_num_families,
            seed=config.probing_seed,
            filter_correct=config.probing_filter_correct,
        )
        run_probing_analysis(
            runner,
            tasks,
            outputs=outputs,
            results_dir=config.results_dir,
            config=probing_config,
        )

    # Save summary
    summary_path = config.results_dir / "study_summary.json"
    summary_path.write_text(json.dumps(results.to_dict(), indent=2))

    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)
    print(f"Summary saved to: {summary_path}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run MechInt behavioral study on Theory of Mind in LLMs"
    )
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.3"),
        help="Model name (HuggingFace model ID)"
    )
    parser.add_argument(
        "--num-false-belief", "-nfb",
        type=int,
        default=int(os.getenv("NUM_FALSE_BELIEF_TASKS", "20")),
        help="Number of false belief tasks"
    )
    parser.add_argument(
        "--num-true-belief", "-ntb",
        type=int,
        default=int(os.getenv("NUM_TRUE_BELIEF_TASKS", "20")),
        help="Number of true belief tasks"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "4")),
        help="Batch size for processing"
    )
    parser.add_argument(
        "--results-dir", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--tasks-file", "-t",
        type=Path,
        default=Path("tasks.json"),
        help="Path to tasks file (must already exist)"
    )
    parser.add_argument(
        "--run-probing",
        action="store_true",
        help="Run probing analysis (extracts hidden states)"
    )
    parser.add_argument(
        "--probe-families",
        type=int,
        default=int(os.getenv("PROBE_FAMILIES", "100")),
        help="Number of task families to use for probing"
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=int(os.getenv("PROBE_SEED", "42")),
        help="Random seed for probing family selection"
    )
    parser.add_argument(
        "--probe-filter-correct",
        action="store_true",
        help="Use only behaviorally-correct tasks for probing"
    )
    # Keep --behavioral-only for backwards compatibility (now a no-op)
    parser.add_argument(
        "--behavioral-only",
        action="store_true",
        help="(Deprecated: behavioral analysis is now the default)"
    )

    args = parser.parse_args()

    config = StudyConfig(
        model_name=args.model,
        num_false_belief_tasks=args.num_false_belief,
        num_true_belief_tasks=args.num_true_belief,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        tasks_file=args.tasks_file,
        run_probing=args.run_probing,
        probing_num_families=args.probe_families,
        probing_seed=args.probe_seed,
        probing_filter_correct=args.probe_filter_correct,
    )

    run_study(config)


if __name__ == "__main__":
    main()

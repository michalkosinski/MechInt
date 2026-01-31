"""
Main study orchestrator for MechInt.

Runs the complete pipeline:
1. Generate tasks
2. Run behavioral verification
3. Extract and analyze attention patterns
4. Identify ToM heads
5. Run intervention experiments
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .task_generator import ToMTask, generate_tasks, save_tasks, load_tasks
from .model_runner import ModelRunner, ModelOutput, BatchOutput
from .attention_analyzer import (
    AttentionAnalyzer, AttentionPattern, HeadScore,
    save_patterns, save_head_scores, load_patterns,
    save_layer_selectivity, save_token_attention, save_attention_matrices,
)
from .intervention import (
    InterventionRunner, InterventionResult, InterventionSummary,
    save_intervention_results
)


@dataclass
class StudyConfig:
    """Configuration for a MechInt study run."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    num_false_belief_tasks: int = 20
    num_true_belief_tasks: int = 20
    batch_size: int = 16  # Larger batches for efficiency
    max_new_tokens: int = 3  # Completion format only needs 3 tokens
    run_attention_analysis: bool = True
    attention_sample_size: int = 40  # Sample tasks for attention extraction
    run_interventions: bool = True
    top_k_heads: int = 5  # Number of top heads for interventions
    boost_scale: float = 2.0
    results_dir: Path = Path("results")
    tasks_file: Path = Path("tasks.json")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results_dir"] = str(self.results_dir)
        d["tasks_file"] = str(self.tasks_file)
        return d


@dataclass
class StudyResults:
    """Results from a complete study run."""
    config: StudyConfig
    timestamp: str
    # Behavioral results
    behavioral_accuracy: float
    false_belief_accuracy: float
    true_belief_accuracy: float
    num_tasks: int
    # Attention analysis results
    attention_analyzed: bool
    num_patterns: int
    top_tom_heads: List[Tuple[int, int]]
    # Intervention results
    interventions_run: bool
    ablation_accuracy_delta: Optional[float]
    boost_accuracy_delta: Optional[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        d["top_tom_heads"] = [list(t) for t in self.top_tom_heads]
        return d


def run_behavioral_study(
    runner: ModelRunner,
    tasks: List[ToMTask],
    config: StudyConfig,
    results_dir: Path,
) -> Tuple[BatchOutput, List[ModelOutput]]:
    """
    Step 1: Run behavioral verification.

    Tests whether the model can correctly answer ToM tasks.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Behavioral Verification")
    print("=" * 60)

    # Run all tasks with batching
    batch_output = runner.run_batch(
        tasks,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        extract_attention=config.run_attention_analysis,
        attention_sample_size=config.attention_sample_size if config.run_attention_analysis else 0,
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


def run_attention_analysis(
    outputs: List[ModelOutput],
    tasks: List[ToMTask],
    config: StudyConfig,
    results_dir: Path,
    runner: ModelRunner,
) -> Tuple[List[AttentionPattern], List[HeadScore], List[Tuple[int, int]]]:
    """
    Step 2: Analyze attention patterns.

    Extracts attention patterns and identifies ToM-relevant heads.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Attention Analysis")
    print("=" * 60)

    analyzer = AttentionAnalyzer()

    # Build task lookup
    task_lookup = {t.task_id: t for t in tasks}

    # Extract patterns from all outputs
    all_patterns = []
    for output in outputs:
        if output.has_attentions():
            task = task_lookup[output.task_id]
            patterns = analyzer.extract_attention_patterns(output, task)
            all_patterns.extend(patterns)

    print(f"Extracted {len(all_patterns)} attention patterns")

    # Identify ToM heads
    head_scores = analyzer.identify_tom_heads(all_patterns)
    top_heads = analyzer.get_top_tom_heads(all_patterns, top_k=config.top_k_heads)

    print(f"\nTop {len(top_heads)} ToM-relevant heads:")
    for score in head_scores[:config.top_k_heads]:
        print(f"  Layer {score.layer}, Head {score.head}: "
              f"FB ratio={score.false_belief_ratio:.2f}, "
              f"diff={score.condition_diff:.2f}, "
              f"score={score.tom_score:.3f}")

    # Save attention results
    attention_dir = results_dir / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)

    save_patterns(all_patterns, attention_dir / "patterns.json")
    save_head_scores(head_scores, attention_dir / "head_scores.json")

    # Save heatmap data
    heatmap_data = {
        "false_belief": analyzer.create_heatmap_data(
            all_patterns, runner.num_layers, runner.num_heads,
            task_type="false_belief"
        ).tolist(),
        "true_belief": analyzer.create_heatmap_data(
            all_patterns, runner.num_layers, runner.num_heads,
            task_type="true_belief"
        ).tolist(),
        "num_layers": runner.num_layers,
        "num_heads": runner.num_heads,
    }
    (attention_dir / "heatmap_data.json").write_text(json.dumps(heatmap_data, indent=2))

    # Save layer selectivity data
    layer_selectivity = analyzer.compute_layer_selectivity(all_patterns, runner.num_layers)
    save_layer_selectivity(layer_selectivity, attention_dir / "layer_selectivity.json")

    # Extract and save token-level attention for top heads
    # Get top 10 heads for visualization
    top_10_heads = [(s.layer, s.head) for s in head_scores[:10]]
    all_token_attention = []
    all_attention_matrices = []

    # Process outputs that have attention data
    outputs_with_attention = [o for o in outputs if o.has_attentions()]
    sample_outputs = outputs_with_attention[:10]  # Limit to 10 samples for matrices

    for output in outputs_with_attention:
        task = task_lookup[output.task_id]
        token_data = analyzer.extract_token_attention(output, task, top_10_heads)
        all_token_attention.extend(token_data)

    # Save full attention matrices for a smaller sample
    for output in sample_outputs:
        task = task_lookup[output.task_id]
        for layer, head in top_10_heads[:3]:  # Top 3 heads only
            matrix = analyzer.extract_attention_matrix(output, task, layer, head)
            if matrix:
                all_attention_matrices.append(matrix)

    save_token_attention(all_token_attention, attention_dir / "token_attention.json")
    save_attention_matrices(all_attention_matrices, attention_dir / "attention_matrices.json")

    print(f"  Saved patterns to: {attention_dir / 'patterns.json'}")
    print(f"  Saved head scores to: {attention_dir / 'head_scores.json'}")
    print(f"  Saved layer selectivity to: {attention_dir / 'layer_selectivity.json'}")
    print(f"  Saved token attention ({len(all_token_attention)} samples)")
    print(f"  Saved attention matrices ({len(all_attention_matrices)} samples)")

    return all_patterns, head_scores, top_heads


def run_intervention_study(
    runner: ModelRunner,
    tasks: List[ToMTask],
    baseline_outputs: List[ModelOutput],
    top_heads: List[Tuple[int, int]],
    config: StudyConfig,
    results_dir: Path,
) -> Tuple[Optional[InterventionSummary], Optional[InterventionSummary]]:
    """
    Step 3: Run intervention experiments.

    Tests causal effects by ablating and boosting identified ToM heads.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Intervention Experiments")
    print("=" * 60)

    if not top_heads:
        print("No ToM heads identified. Skipping interventions.")
        return None, None

    print(f"Target heads: {top_heads}")

    intervention_runner = InterventionRunner(runner)
    intervention_dir = results_dir / "interventions"
    intervention_dir.mkdir(parents=True, exist_ok=True)

    # Ablation experiment
    print("\nRunning ablation experiment (scale=0.0)...")
    ablation_results, ablation_summary = intervention_runner.run_ablation_experiment(
        tasks, top_heads, baseline_outputs
    )
    save_intervention_results(
        ablation_results, ablation_summary,
        intervention_dir / "ablation.json"
    )

    print(f"  Ablation results:")
    print(f"    Original accuracy: {ablation_summary.original_accuracy:.1%}")
    print(f"    Modified accuracy: {ablation_summary.modified_accuracy:.1%}")
    print(f"    Delta: {ablation_summary.accuracy_delta:+.1%}")
    print(f"    Flip rate: {ablation_summary.flip_rate:.1%}")

    # Boosting experiment
    print(f"\nRunning boost experiment (scale={config.boost_scale})...")
    boost_results, boost_summary = intervention_runner.run_boost_experiment(
        tasks, top_heads, scale_factor=config.boost_scale, baseline_outputs=baseline_outputs
    )
    save_intervention_results(
        boost_results, boost_summary,
        intervention_dir / "boost.json"
    )

    print(f"  Boost results:")
    print(f"    Original accuracy: {boost_summary.original_accuracy:.1%}")
    print(f"    Modified accuracy: {boost_summary.modified_accuracy:.1%}")
    print(f"    Delta: {boost_summary.accuracy_delta:+.1%}")
    print(f"    Flip rate: {boost_summary.flip_rate:.1%}")

    return ablation_summary, boost_summary


def run_full_study(config: StudyConfig) -> StudyResults:
    """
    Run the complete MechInt study pipeline.
    """
    print("\n" + "=" * 60)
    print("MechInt: Mechanistic Interpretability of ToM")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Tasks: {config.num_false_belief_tasks} false belief + {config.num_true_belief_tasks} true belief")
    print(f"Results dir: {config.results_dir}")

    # Ensure directories exist
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Load or generate tasks
    if config.tasks_file.exists():
        print(f"\nLoading tasks from {config.tasks_file}")
        tasks = load_tasks(config.tasks_file)
    else:
        print(f"\nGenerating {config.num_false_belief_tasks + config.num_true_belief_tasks} tasks...")
        tasks = generate_tasks(
            num_false_belief=config.num_false_belief_tasks,
            num_true_belief=config.num_true_belief_tasks,
        )
        save_tasks(tasks, config.tasks_file)
        print(f"Saved tasks to {config.tasks_file}")

    # Load model
    runner = ModelRunner(config.model_name)

    # Step 1: Behavioral verification
    batch_output, outputs = run_behavioral_study(runner, tasks, config, config.results_dir)

    fb_outputs = [o for o in outputs if o.task_type == "false_belief"]
    tb_outputs = [o for o in outputs if o.task_type == "true_belief"]
    fb_accuracy = sum(1 for o in fb_outputs if o.is_correct) / len(fb_outputs) if fb_outputs else 0
    tb_accuracy = sum(1 for o in tb_outputs if o.is_correct) / len(tb_outputs) if tb_outputs else 0

    # Step 2: Attention analysis
    top_heads = []
    num_patterns = 0
    if config.run_attention_analysis:
        patterns, head_scores, top_heads = run_attention_analysis(
            outputs, tasks, config, config.results_dir, runner
        )
        num_patterns = len(patterns)

    # Step 3: Intervention experiments
    ablation_summary = None
    boost_summary = None
    if config.run_interventions and top_heads:
        ablation_summary, boost_summary = run_intervention_study(
            runner, tasks, outputs, top_heads, config, config.results_dir
        )

    # Create final results
    results = StudyResults(
        config=config,
        timestamp=datetime.now().isoformat(),
        behavioral_accuracy=batch_output.accuracy,
        false_belief_accuracy=fb_accuracy,
        true_belief_accuracy=tb_accuracy,
        num_tasks=len(tasks),
        attention_analyzed=config.run_attention_analysis,
        num_patterns=num_patterns,
        top_tom_heads=top_heads,
        interventions_run=config.run_interventions and bool(top_heads),
        ablation_accuracy_delta=ablation_summary.accuracy_delta if ablation_summary else None,
        boost_accuracy_delta=boost_summary.accuracy_delta if boost_summary else None,
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
        description="Run MechInt study on Theory of Mind in LLMs"
    )
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
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
        "--behavioral-only",
        action="store_true",
        help="Run only behavioral verification (skip attention analysis)"
    )
    parser.add_argument(
        "--no-interventions",
        action="store_true",
        help="Skip intervention experiments"
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
        help="Path to tasks file (will be created if doesn't exist)"
    )

    args = parser.parse_args()

    config = StudyConfig(
        model_name=args.model,
        num_false_belief_tasks=args.num_false_belief,
        num_true_belief_tasks=args.num_true_belief,
        batch_size=args.batch_size,
        run_attention_analysis=not args.behavioral_only,
        run_interventions=not args.no_interventions and not args.behavioral_only,
        results_dir=args.results_dir,
        tasks_file=args.tasks_file,
    )

    run_full_study(config)


if __name__ == "__main__":
    main()

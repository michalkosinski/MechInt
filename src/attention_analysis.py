"""
Attention pattern analysis for Theory of Mind tasks.

Analyzes attention weights to find heads that:
- Track protagonist's belief vs reality
- Differentiate between false-belief and true-belief scenarios

Aggregates metrics across many tasks (no per-task visualization).
"""

import json
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
import torch

from .task_generator import ToMTask, load_tasks
from .model_runner import ModelRunner, ModelOutput


@dataclass
class AttentionConfig:
    num_families: int = 100
    seed: int = 42
    attention_sample_size: int = 200  # Tasks to extract attention for
    filter_correct: bool = False
    source_position: str = "prompt_end"  # Where to look FROM


@dataclass
class SanityCheckResults:
    token_position_warnings: int = 0
    attention_sum_warnings: int = 0
    metric_range_warnings: int = 0
    tasks_analyzed: int = 0
    tasks_skipped: int = 0
    fb_tasks: int = 0
    tb_tasks: int = 0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def _select_tasks_by_family(tasks: List[ToMTask], num_families: int, seed: int) -> List[ToMTask]:
    """Select tasks from a random subset of families."""
    family_ids = sorted({t.family_id for t in tasks})
    rng = random.Random(seed)
    if num_families >= len(family_ids):
        selected = set(family_ids)
    else:
        selected = set(rng.sample(family_ids, num_families))
    return [t for t in tasks if t.family_id in selected]


def _find_subsequence(haystack: List[int], needle: List[int], find_first: bool = False) -> Optional[int]:
    """Find subsequence in token list."""
    if not needle or len(needle) > len(haystack):
        return None
    found_idx = None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            if find_first:
                return i
            found_idx = i
    return found_idx


def _find_token_index_for_word(
    tokenizer,
    input_ids: List[int],
    word: str,
    find_first: bool = False,
) -> Optional[int]:
    """Find token index for a word in the input."""
    variants = [word, f" {word}"]
    for variant in variants:
        word_ids = tokenizer(variant, add_special_tokens=False).input_ids
        start_idx = _find_subsequence(input_ids, word_ids, find_first=find_first)
        if start_idx is not None:
            return start_idx + len(word_ids) - 1
    return None


def _validate_attention_sums(
    attentions: Tuple[torch.Tensor, ...],
    tolerance: float = 0.05,
) -> List[str]:
    """Verify attention weights sum to approximately 1."""
    warnings = []
    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn shape: (batch, num_heads, seq_len, seq_len) or (num_heads, seq_len, seq_len)
        # Convert to float for numerical stability
        attn = layer_attn.float()
        if attn.dim() == 4:
            attn = attn[0]  # Take batch 0
        row_sums = attn.sum(dim=-1)
        max_deviation = (row_sums - 1.0).abs().max().item()
        if max_deviation > tolerance:
            warnings.append(
                f"Layer {layer_idx}: attention rows deviate from sum=1 "
                f"(max deviation: {max_deviation:.4f})"
            )
    return warnings


def _extract_attention_to_positions(
    attentions: Tuple[torch.Tensor, ...],
    source_idx: int,
    target_indices: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """
    Extract attention weights from source token to target tokens.

    Args:
        attentions: Tuple of attention tensors per layer, each (batch, num_heads, seq_len, seq_len)
                    or (num_heads, seq_len, seq_len)
        source_idx: Token position to look FROM
        target_indices: Dict mapping target name to token position

    Returns:
        Dict mapping target name to array of shape (num_layers, num_heads)
    """
    num_layers = len(attentions)
    # Handle both (batch, heads, seq, seq) and (heads, seq, seq) shapes
    first_attn = attentions[0]
    if first_attn.dim() == 4:
        num_heads = first_attn.shape[1]
    else:
        num_heads = first_attn.shape[0]

    result = {}
    for target_name, target_idx in target_indices.items():
        if target_idx is None or target_idx < 0:
            continue
        weights = np.zeros((num_layers, num_heads))
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn shape: (batch, num_heads, seq_len, seq_len) or (num_heads, seq_len, seq_len)
            # attention[b, h, i, j] = attention from token i to token j at head h, batch b
            # Convert to float32 for numpy compatibility (handles bfloat16)
            if layer_attn.dim() == 4:
                # Shape: (batch, num_heads, seq_len, seq_len) - take batch 0
                weights[layer_idx, :] = layer_attn[0, :, source_idx, target_idx].float().numpy()
            else:
                # Shape: (num_heads, seq_len, seq_len)
                weights[layer_idx, :] = layer_attn[:, source_idx, target_idx].float().numpy()
        result[target_name] = weights

    return result


def run_attention_analysis(
    runner: ModelRunner,
    tasks: List[ToMTask],
    outputs: Optional[List[ModelOutput]],
    results_dir: Path,
    config: Optional[AttentionConfig] = None,
) -> Dict:
    """
    Run attention pattern analysis.

    Analyzes attention weights to find heads that track belief vs reality.
    """
    config = config or AttentionConfig()
    start_time = time.time()

    # Select tasks
    selected_tasks = _select_tasks_by_family(tasks, config.num_families, config.seed)

    # Filter to correct only if requested
    correct_lookup = {}
    if outputs:
        correct_lookup = {o.task_id: o.is_correct for o in outputs}

    if config.filter_correct and correct_lookup:
        selected_tasks = [t for t in selected_tasks if correct_lookup.get(t.task_id, False)]

    # Sample tasks for attention extraction
    rng = random.Random(config.seed)
    if len(selected_tasks) > config.attention_sample_size:
        selected_tasks = rng.sample(selected_tasks, config.attention_sample_size)

    sanity = SanityCheckResults()
    tokenizer = runner.tokenizer
    num_layers = runner.num_layers
    num_heads = runner.num_heads

    # Per-task attention data
    # Structure: list of dicts with task_type, attention weights to belief/world
    task_attention_data = []

    print(f"Extracting attention for {len(selected_tasks)} tasks...")

    for output in runner.run_batch(
        selected_tasks,
        max_new_tokens=20,
        stop_strings=[".", "?", "!"],
        extract_attention=True,
        attention_sample_size=len(selected_tasks),
    ).outputs:
        task = next((t for t in selected_tasks if t.task_id == output.task_id), None)
        if task is None:
            continue

        if not output.has_attentions():
            sanity.tasks_skipped += 1
            continue

        attentions = output.attentions
        input_ids = output.input_ids[0].tolist() if output.input_ids is not None else []

        # Get actual sequence length from attention tensor (not from tokens which may include generated)
        # Attention shape: (batch, num_heads, seq_len, seq_len) or (num_heads, seq_len, seq_len)
        first_attn = attentions[0] if attentions else None
        if first_attn is None:
            sanity.tasks_skipped += 1
            continue

        # Get seq_len from the correct dimension based on tensor shape
        if first_attn.dim() == 4:
            attn_seq_len = first_attn.shape[2]  # (batch, heads, seq_len, seq_len)
        else:
            attn_seq_len = first_attn.shape[1]  # (heads, seq_len, seq_len)

        if attn_seq_len == 0:
            sanity.tasks_skipped += 1
            continue

        # Use the attention tensor's sequence length as the true input length
        input_length = attn_seq_len

        # Validate attention sums
        sum_warnings = _validate_attention_sums(attentions)
        if sum_warnings:
            sanity.attention_sum_warnings += len(sum_warnings)
            sanity.warnings.extend(sum_warnings[:3])

        # Find token positions
        # World is mentioned FIRST ("A obj is in the {world}")
        world_idx = _find_token_index_for_word(tokenizer, input_ids, task.world, find_first=True)
        # Belief is mentioned LAST ("X believes the obj is in the {belief}")
        belief_idx = _find_token_index_for_word(tokenizer, input_ids, task.belief, find_first=False)

        # Determine source position
        if config.source_position == "prompt_end":
            source_idx = input_length - 1
        else:
            source_idx = input_length - 1

        # Validate positions
        if world_idx is None or belief_idx is None:
            sanity.tasks_skipped += 1
            sanity.token_position_warnings += 1
            continue

        if world_idx >= belief_idx:
            sanity.token_position_warnings += 1
            sanity.warnings.append(
                f"Task {task.task_id}: world_idx ({world_idx}) >= belief_idx ({belief_idx})"
            )

        # Validate all indices are within bounds
        if source_idx >= input_length or belief_idx >= input_length or world_idx >= input_length:
            sanity.tasks_skipped += 1
            sanity.token_position_warnings += 1
            sanity.warnings.append(
                f"Task {task.task_id}: index out of bounds (src={source_idx}, belief={belief_idx}, "
                f"world={world_idx}, seq_len={input_length})"
            )
            continue

        # Extract attention weights
        target_indices = {
            "belief": belief_idx,
            "world": world_idx,
        }

        weights = _extract_attention_to_positions(attentions, source_idx, target_indices)

        if "belief" not in weights or "world" not in weights:
            sanity.tasks_skipped += 1
            continue

        task_attention_data.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "family_id": task.family_id,
            "belief_attention": weights["belief"],  # (num_layers, num_heads)
            "world_attention": weights["world"],
            "is_correct": correct_lookup.get(task.task_id, None),
        })

        sanity.tasks_analyzed += 1
        if task.task_type == "false_belief":
            sanity.fb_tasks += 1
        else:
            sanity.tb_tasks += 1

    if not task_attention_data:
        print("No tasks with attention data found!")
        return {"error": "No attention data extracted"}

    print(f"Analyzed {sanity.tasks_analyzed} tasks ({sanity.fb_tasks} FB, {sanity.tb_tasks} TB)")

    # Aggregate metrics across tasks
    # Separate by task type
    fb_data = [d for d in task_attention_data if d["task_type"] == "false_belief"]
    tb_data = [d for d in task_attention_data if d["task_type"] == "true_belief"]

    # Stack attention arrays
    if fb_data:
        fb_belief = np.stack([d["belief_attention"] for d in fb_data])  # (n_fb, layers, heads)
        fb_world = np.stack([d["world_attention"] for d in fb_data])
    else:
        fb_belief = np.zeros((0, num_layers, num_heads))
        fb_world = np.zeros((0, num_layers, num_heads))

    if tb_data:
        tb_belief = np.stack([d["belief_attention"] for d in tb_data])
        tb_world = np.stack([d["world_attention"] for d in tb_data])
    else:
        tb_belief = np.zeros((0, num_layers, num_heads))
        tb_world = np.zeros((0, num_layers, num_heads))

    # Compute per-head metrics
    head_metrics = []
    epsilon = 1e-8

    for layer in range(num_layers):
        for head in range(num_heads):
            metrics = {
                "layer": layer,
                "head": head,
            }

            # Overall attention to belief/world (across all tasks)
            all_belief = np.concatenate([
                fb_belief[:, layer, head] if len(fb_data) else np.array([]),
                tb_belief[:, layer, head] if len(tb_data) else np.array([]),
            ])
            all_world = np.concatenate([
                fb_world[:, layer, head] if len(fb_data) else np.array([]),
                tb_world[:, layer, head] if len(tb_data) else np.array([]),
            ])

            if len(all_belief) > 0:
                metrics["belief_attention_mean"] = float(np.mean(all_belief))
                metrics["belief_attention_std"] = float(np.std(all_belief))
                metrics["world_attention_mean"] = float(np.mean(all_world))
                metrics["world_attention_std"] = float(np.std(all_world))

                # Belief over world ratio
                ratio = (np.mean(all_belief) + epsilon) / (np.mean(all_world) + epsilon)
                metrics["belief_over_world_ratio"] = float(ratio)
            else:
                metrics["belief_attention_mean"] = 0.0
                metrics["world_attention_mean"] = 0.0
                metrics["belief_over_world_ratio"] = 1.0

            # FB-specific metrics
            if len(fb_data) > 0:
                fb_belief_vals = fb_belief[:, layer, head]
                fb_world_vals = fb_world[:, layer, head]
                metrics["belief_attention_fb"] = float(np.mean(fb_belief_vals))
                metrics["world_attention_fb"] = float(np.mean(fb_world_vals))

                # Belief tracking score: log ratio of belief/world attention on FB tasks
                log_ratios = np.log(fb_belief_vals + epsilon) - np.log(fb_world_vals + epsilon)
                metrics["belief_tracking_score"] = float(np.mean(log_ratios))
            else:
                metrics["belief_attention_fb"] = 0.0
                metrics["world_attention_fb"] = 0.0
                metrics["belief_tracking_score"] = 0.0

            # TB-specific metrics
            if len(tb_data) > 0:
                tb_belief_vals = tb_belief[:, layer, head]
                tb_world_vals = tb_world[:, layer, head]
                metrics["belief_attention_tb"] = float(np.mean(tb_belief_vals))
                metrics["world_attention_tb"] = float(np.mean(tb_world_vals))
            else:
                metrics["belief_attention_tb"] = 0.0
                metrics["world_attention_tb"] = 0.0

            # Differential score: FB belief attention - TB belief attention
            metrics["differential_score"] = metrics["belief_attention_fb"] - metrics["belief_attention_tb"]

            # Statistical test: FB vs TB belief attention
            if len(fb_data) > 1 and len(tb_data) > 1:
                fb_vals = fb_belief[:, layer, head]
                tb_vals = tb_belief[:, layer, head]
                try:
                    _, p_value = stats.ttest_ind(fb_vals, tb_vals)
                    metrics["p_value"] = float(p_value) if not np.isnan(p_value) else 1.0
                except Exception:
                    metrics["p_value"] = 1.0
            else:
                metrics["p_value"] = 1.0

            # Validate metrics
            if not (0 <= metrics["belief_attention_mean"] <= 1):
                sanity.metric_range_warnings += 1
            if not (0 <= metrics["world_attention_mean"] <= 1):
                sanity.metric_range_warnings += 1

            head_metrics.append(metrics)

    # Find top belief-tracking heads (sorted by belief_tracking_score)
    top_belief_tracking = sorted(
        head_metrics,
        key=lambda m: m.get("belief_tracking_score", 0),
        reverse=True
    )[:20]

    # Find top differentiating heads (sorted by |differential_score| with low p-value)
    significant_heads = [m for m in head_metrics if m.get("p_value", 1.0) < 0.05]
    top_differentiating = sorted(
        significant_heads if significant_heads else head_metrics,
        key=lambda m: abs(m.get("differential_score", 0)),
        reverse=True
    )[:20]

    # Layer aggregates
    layer_aggregates = {}
    for layer in range(num_layers):
        layer_heads = [m for m in head_metrics if m["layer"] == layer]
        if layer_heads:
            layer_aggregates[layer] = {
                "max_belief_tracking": max(m["belief_tracking_score"] for m in layer_heads),
                "mean_belief_tracking": np.mean([m["belief_tracking_score"] for m in layer_heads]),
                "max_differential": max(abs(m["differential_score"]) for m in layer_heads),
                "mean_belief_attention": np.mean([m["belief_attention_mean"] for m in layer_heads]),
                "mean_world_attention": np.mean([m["world_attention_mean"] for m in layer_heads]),
            }

    # Build results
    elapsed_ms = (time.time() - start_time) * 1000

    results = {
        "model_name": runner.model_name,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "timing": {
            "total_time_ms": elapsed_ms,
        },
        "model_info": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "hidden_size": runner.hidden_size,
        },
        "sanity_checks": asdict(sanity),
        "summary": {
            "tasks_analyzed": sanity.tasks_analyzed,
            "fb_tasks": sanity.fb_tasks,
            "tb_tasks": sanity.tb_tasks,
            "best_belief_tracking_score": top_belief_tracking[0]["belief_tracking_score"] if top_belief_tracking else 0,
            "best_belief_tracking_head": {
                "layer": top_belief_tracking[0]["layer"],
                "head": top_belief_tracking[0]["head"],
            } if top_belief_tracking else None,
        },
        "results": {
            "head_metrics": head_metrics,
            "top_belief_tracking_heads": [
                {"layer": m["layer"], "head": m["head"], "score": m["belief_tracking_score"]}
                for m in top_belief_tracking
            ],
            "top_differentiating_heads": [
                {
                    "layer": m["layer"],
                    "head": m["head"],
                    "diff": m["differential_score"],
                    "p_value": m.get("p_value", 1.0),
                }
                for m in top_differentiating
            ],
            "layer_aggregates": {str(k): v for k, v in layer_aggregates.items()},
        },
    }

    # Save results
    safe_model_name = runner.model_name.replace("/", "_")
    attention_dir = results_dir / "attention" / safe_model_name
    attention_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest = {
        "model_name": runner.model_name,
        "timestamp": results["timestamp"],
        "num_tasks": sanity.tasks_analyzed,
        "num_fb_tasks": sanity.fb_tasks,
        "num_tb_tasks": sanity.tb_tasks,
        "config": asdict(config),
        "model_info": results["model_info"],
        "sanity_checks": asdict(sanity),
    }
    (attention_dir / "attention_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save full metrics
    (attention_dir / "attention_metrics.json").write_text(json.dumps(results, indent=2))

    print(f"Attention analysis complete in {elapsed_ms:.0f}ms")
    print(f"Results saved to {attention_dir}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run attention analysis for ToM tasks")
    parser.add_argument("--model", "-m", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--tasks-file", "-t", type=Path, default=Path("tasks.json"))
    parser.add_argument("--results-dir", "-o", type=Path, default=Path("results"))
    parser.add_argument("--num-families", type=int, default=100)
    parser.add_argument("--attention-sample-size", type=int, default=200,
                        help="Number of tasks to extract attention for")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-correct", action="store_true")

    args = parser.parse_args()

    if not args.tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {args.tasks_file}")

    tasks = load_tasks(args.tasks_file)
    runner = ModelRunner(args.model)

    config = AttentionConfig(
        num_families=args.num_families,
        attention_sample_size=args.attention_sample_size,
        seed=args.seed,
        filter_correct=args.filter_correct,
    )

    results = run_attention_analysis(
        runner, tasks, outputs=None, results_dir=args.results_dir, config=config
    )

    if "error" not in results:
        top = results["results"]["top_belief_tracking_heads"][:5]
        print("\nTop 5 Belief-Tracking Heads:")
        for i, h in enumerate(top, 1):
            print(f"  {i}. Layer {h['layer']}, Head {h['head']}: score={h['score']:.3f}")


if __name__ == "__main__":
    main()

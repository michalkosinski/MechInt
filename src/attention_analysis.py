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
from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
import torch

from .task_generator import ToMTask, load_tasks
from .model_runner import ModelRunner, ModelOutput

# Simple task wrapper for model_runner (needs: task_id, task_type, full_prompt)
PromptTask = namedtuple("PromptTask", ["task_id", "task_type", "full_prompt"])

# Question template for attention analysis - uses protagonist belief question
QUESTION_TEMPLATE = "{protagonist} will look for the {obj} in the"


def build_prompt(task: ToMTask) -> str:
    """Build full prompt from narrative + question template."""
    question = QUESTION_TEMPLATE.format(
        obj=task.obj,
        protagonist=task.protagonist,
    )
    return f"{task.narrative} {question}"


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def _attention_entropy(attn_weights: np.ndarray) -> float:
    """Compute entropy of attention distribution (higher = more diffuse)."""
    attn = attn_weights + 1e-10  # Avoid log(0)
    attn = attn / attn.sum()
    return float(-np.sum(attn * np.log(attn)))


@dataclass
class AttentionConfig:
    num_families: int = 100
    seed: int = 42
    attention_sample_size: int = 200  # Tasks to extract attention for
    filter_correct: bool = False
    save_sample_heatmaps: int = 5  # Number of tasks to save full attention matrices for


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
    # Enhanced tracking
    tokens_not_found: Dict[str, int] = None  # Count per token type
    position_order_violations: int = 0  # world_idx > belief_idx
    zero_attention_warnings: int = 0  # Heads with all-zero attention
    high_entropy_warnings: int = 0  # Heads with near-uniform attention

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.tokens_not_found is None:
            self.tokens_not_found = {}


def _select_tasks_by_family(tasks: List[ToMTask], num_families: int, seed: int) -> List[ToMTask]:
    """Select tasks from a random subset of families."""
    family_ids = sorted({t.family_id for t in tasks})
    rng = random.Random(seed)
    if num_families >= len(family_ids):
        selected = set(family_ids)
    else:
        selected = set(rng.sample(family_ids, num_families))
    return [t for t in tasks if t.family_id in selected]


def _get_attention_shape(attn: torch.Tensor) -> Tuple[int, int]:
    """Extract (num_heads, seq_len) from attention tensor of shape (batch, heads, seq, seq) or (heads, seq, seq)."""
    if attn.dim() == 4:
        return attn.shape[1], attn.shape[2]
    return attn.shape[0], attn.shape[1]


def _stack_attention_data(
    data_list: List[Dict],
    num_layers: int,
    num_heads: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack belief and world attention arrays from task data list.

    Returns (belief_array, world_array) each of shape (n_tasks, num_layers, num_heads).
    Returns zero-sized arrays if data_list is empty.
    """
    if data_list:
        belief = np.stack([d["belief_attention"] for d in data_list])
        world = np.stack([d["world_attention"] for d in data_list])
        return belief, world
    return np.zeros((0, num_layers, num_heads)), np.zeros((0, num_layers, num_heads))


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
    num_heads, _ = _get_attention_shape(attentions[0])

    result = {}
    for target_name, target_idx in target_indices.items():
        if target_idx is None or target_idx < 0:
            continue
        weights = np.zeros((num_layers, num_heads))
        for layer_idx, layer_attn in enumerate(attentions):
            # attention[b, h, i, j] = attention from token i to token j at head h, batch b
            # Convert to float32 for numpy compatibility (handles bfloat16)
            if layer_attn.dim() == 4:
                weights[layer_idx, :] = layer_attn[0, :, source_idx, target_idx].float().numpy()
            else:
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

    # Build task lookup for later reference
    task_lookup = {t.task_id: t for t in selected_tasks}

    # Wrap tasks with prompts for model runner
    prompt_tasks = [
        PromptTask(t.task_id, t.task_type, build_prompt(t))
        for t in selected_tasks
    ]

    batch_result = runner.run_batch(
        prompt_tasks,
        max_new_tokens=20,
        stop_strings=[".", "?", "!"],
        extract_attention=True,
        attention_sample_size=len(prompt_tasks),
    )

    for output in batch_result.outputs:
        task = task_lookup.get(output.task_id)
        if task is None:
            continue

        if not output.has_attentions():
            sanity.tasks_skipped += 1
            continue

        attentions = output.attentions

        # Get sequence length from attention tensor shape (input only, not generated)
        first_attn = attentions[0] if attentions else None
        if first_attn is None:
            sanity.tasks_skipped += 1
            continue

        _, input_length = _get_attention_shape(first_attn)
        if input_length == 0:
            sanity.tasks_skipped += 1
            continue

        # Truncate input_ids to input_length (exclude generated tokens)
        input_ids = output.input_ids[0].tolist() if output.input_ids is not None else []
        input_ids = input_ids[:input_length]

        # Validate attention sums
        sum_warnings = _validate_attention_sums(attentions)
        if sum_warnings:
            sanity.attention_sum_warnings += len(sum_warnings)
            sanity.warnings.extend(sum_warnings[:3])

        # Find token positions for all targets
        # World/reality = final_location (where object actually is after move)
        world_idx = _find_token_index_for_word(tokenizer, input_ids, task.final_location, find_first=False)
        # Belief = protagonist_belief (what protagonist thinks - initial for FB, final for TB)
        belief_idx = _find_token_index_for_word(tokenizer, input_ids, task.protagonist_belief, find_first=False)
        # Additional targets for richer analysis
        protagonist_idx = _find_token_index_for_word(tokenizer, input_ids, task.protagonist, find_first=True)
        believes_idx = _find_token_index_for_word(tokenizer, input_ids, "believes", find_first=False)
        question_where_idx = _find_token_index_for_word(tokenizer, input_ids, "Where", find_first=False)

        # Track which tokens were not found
        token_positions = {
            "belief": belief_idx,
            "world": world_idx,
            "protagonist": protagonist_idx,
            "believes": believes_idx,
            "question_where": question_where_idx,
        }
        for token_name, idx in token_positions.items():
            if idx is None:
                sanity.tokens_not_found[token_name] = sanity.tokens_not_found.get(token_name, 0) + 1

        # Source position: last token of prompt (where model attends from)
        source_idx = input_length - 1

        # Validate required positions (belief and world are mandatory)
        if world_idx is None or belief_idx is None:
            sanity.tasks_skipped += 1
            sanity.token_position_warnings += 1
            continue

        # For TB tasks, belief == world (same token), so equal indices are expected
        # For FB tasks, world should appear after belief in the narrative
        if task.task_type == "false_belief" and world_idx <= belief_idx:
            sanity.position_order_violations += 1
            sanity.warnings.append(
                f"Task {task.task_id}: world_idx ({world_idx}) <= belief_idx ({belief_idx}) for FB task"
            )

        # Validate token indices are within bounds (source_idx is always valid since it's input_length-1)
        if belief_idx >= input_length or world_idx >= input_length:
            sanity.tasks_skipped += 1
            sanity.token_position_warnings += 1
            sanity.warnings.append(
                f"Task {task.task_id}: index out of bounds (belief={belief_idx}, "
                f"world={world_idx}, seq_len={input_length})"
            )
            continue

        # Build target indices (include all found tokens)
        target_indices = {k: v for k, v in token_positions.items() if v is not None and v < input_length}

        weights = _extract_attention_to_positions(attentions, source_idx, target_indices)

        task_attention_data.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "family_id": task.family_id,
            "belief_attention": weights.get("belief", np.zeros((num_layers, num_heads))),
            "world_attention": weights.get("world", np.zeros((num_layers, num_heads))),
            "protagonist_attention": weights.get("protagonist"),  # May be None
            "believes_attention": weights.get("believes"),
            "question_where_attention": weights.get("question_where"),
            "is_correct": correct_lookup.get(task.task_id, None),
            "token_positions": token_positions,
            "source_idx": source_idx,
            "input_length": input_length,
            "validation": {
                "all_required_found": belief_idx is not None and world_idx is not None,
                "position_order_valid": world_idx < belief_idx if (world_idx and belief_idx) else False,
            },
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

    # Stack attention arrays (returns zero-sized arrays if empty)
    fb_belief, fb_world = _stack_attention_data(fb_data, num_layers, num_heads)
    tb_belief, tb_world = _stack_attention_data(tb_data, num_layers, num_heads)

    # Stack all belief attention in original task order (for correctness correlation)
    all_belief_stacked = np.stack([d["belief_attention"] for d in task_attention_data])  # (n_tasks, layers, heads)

    # Stack optional attention targets (protagonist, believes, question_where)
    # These may be None for some tasks, so we filter to tasks where they exist
    protagonist_data = [d for d in task_attention_data if d.get("protagonist_attention") is not None]
    believes_data = [d for d in task_attention_data if d.get("believes_attention") is not None]
    question_where_data = [d for d in task_attention_data if d.get("question_where_attention") is not None]

    protagonist_attn = np.stack([d["protagonist_attention"] for d in protagonist_data]) if protagonist_data else None
    believes_attn = np.stack([d["believes_attention"] for d in believes_data]) if believes_data else None
    question_where_attn = np.stack([d["question_where_attention"] for d in question_where_data]) if question_where_data else None

    # Prepare correctness data ONCE (not per layer/head)
    correctness_data = [(i, d["is_correct"]) for i, d in enumerate(task_attention_data) if d["is_correct"] is not None]
    has_correctness_variance = len(correctness_data) > 2 and len(set(c for _, c in correctness_data)) > 1
    if has_correctness_variance:
        correct_indices = np.array([i for i, _ in correctness_data])
        correct_mask = np.array([c for _, c in correctness_data], dtype=float)

    # Compute per-head metrics
    head_metrics = []
    epsilon = 1e-8

    for layer in range(num_layers):
        for head in range(num_heads):
            # Initialize with defaults
            metrics = {
                "layer": layer,
                "head": head,
                "belief_attention_mean": 0.0,
                "belief_attention_std": 0.0,
                "belief_attention_ci95": 0.0,
                "world_attention_mean": 0.0,
                "world_attention_std": 0.0,
                "world_attention_ci95": 0.0,
                "belief_over_world_ratio": 1.0,
                "belief_attention_fb": 0.0,
                "world_attention_fb": 0.0,
                "belief_tracking_score": 0.0,
                "belief_attention_tb": 0.0,
                "world_attention_tb": 0.0,
                "differential_score": 0.0,
                "effect_size_fb_vs_tb": 0.0,
                "p_value": 1.0,
                # Optional token metrics
                "protagonist_attention_mean": 0.0,
                "believes_attention_mean": 0.0,
                "question_where_attention_mean": 0.0,
            }

            # Extract per-head values (slicing empty arrays returns empty arrays)
            fb_belief_vals = fb_belief[:, layer, head]
            fb_world_vals = fb_world[:, layer, head]
            tb_belief_vals = tb_belief[:, layer, head]
            tb_world_vals = tb_world[:, layer, head]

            # Overall attention to belief/world (across all tasks)
            all_belief = np.concatenate([fb_belief_vals, tb_belief_vals])
            all_world = np.concatenate([fb_world_vals, tb_world_vals])

            if len(all_belief) > 0:
                metrics["belief_attention_mean"] = float(np.mean(all_belief))
                metrics["belief_attention_std"] = float(np.std(all_belief))
                metrics["world_attention_mean"] = float(np.mean(all_world))
                metrics["world_attention_std"] = float(np.std(all_world))
                ratio = (np.mean(all_belief) + epsilon) / (np.mean(all_world) + epsilon)
                metrics["belief_over_world_ratio"] = float(ratio)
                # 95% confidence intervals
                if len(all_belief) > 1:
                    metrics["belief_attention_ci95"] = float(1.96 * np.std(all_belief) / np.sqrt(len(all_belief)))
                    metrics["world_attention_ci95"] = float(1.96 * np.std(all_world) / np.sqrt(len(all_world)))

            # FB-specific metrics
            if len(fb_belief_vals) > 0:
                metrics["belief_attention_fb"] = float(np.mean(fb_belief_vals))
                metrics["world_attention_fb"] = float(np.mean(fb_world_vals))
                # Belief tracking score: log ratio of belief/world attention on FB tasks
                log_ratios = np.log(fb_belief_vals + epsilon) - np.log(fb_world_vals + epsilon)
                mean_log_ratio = np.mean(log_ratios)
                metrics["belief_tracking_score"] = float(mean_log_ratio) if np.isfinite(mean_log_ratio) else 0.0

            # TB-specific metrics
            if len(tb_belief_vals) > 0:
                metrics["belief_attention_tb"] = float(np.mean(tb_belief_vals))
                metrics["world_attention_tb"] = float(np.mean(tb_world_vals))

            # Optional token attention metrics
            if protagonist_attn is not None and len(protagonist_attn) > 0:
                metrics["protagonist_attention_mean"] = float(np.mean(protagonist_attn[:, layer, head]))
            if believes_attn is not None and len(believes_attn) > 0:
                metrics["believes_attention_mean"] = float(np.mean(believes_attn[:, layer, head]))
            if question_where_attn is not None and len(question_where_attn) > 0:
                metrics["question_where_attention_mean"] = float(np.mean(question_where_attn[:, layer, head]))

            # Differential score: FB belief attention - TB belief attention
            metrics["differential_score"] = metrics["belief_attention_fb"] - metrics["belief_attention_tb"]

            # Statistical test: FB vs TB belief attention
            if len(fb_belief_vals) > 1 and len(tb_belief_vals) > 1:
                try:
                    _, p_value = stats.ttest_ind(fb_belief_vals, tb_belief_vals)
                    if np.isfinite(p_value):
                        metrics["p_value"] = float(p_value)
                except Exception:
                    pass  # Keep default p_value=1.0
                # Effect size (Cohen's d)
                metrics["effect_size_fb_vs_tb"] = _cohens_d(fb_belief_vals, tb_belief_vals)

            # Correlation with correctness (using pre-computed aligned arrays)
            if has_correctness_variance:
                belief_vals_for_corr = all_belief_stacked[correct_indices, layer, head]
                try:
                    corr, corr_p = stats.pointbiserialr(correct_mask, belief_vals_for_corr)
                    if np.isfinite(corr):
                        metrics["belief_correctness_corr"] = float(corr)
                        metrics["belief_correctness_p"] = float(corr_p)
                except Exception:
                    pass

            # Attention entropy (how focused vs diffuse)
            if len(all_belief) > 0:
                metrics["belief_attention_entropy"] = _attention_entropy(all_belief)

            # Check for degenerate attention patterns
            if len(all_belief) > 0:
                if np.allclose(all_belief, 0, atol=1e-8):
                    sanity.zero_attention_warnings += 1
                # High entropy = near-uniform attention (threshold: 90% of max possible entropy)
                max_entropy = np.log(len(all_belief)) if len(all_belief) > 1 else 1.0
                if metrics.get("belief_attention_entropy", 0) > 0.9 * max_entropy:
                    sanity.high_entropy_warnings += 1

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
                "mean_belief_tracking": float(np.mean([m["belief_tracking_score"] for m in layer_heads])),
                "max_differential": max(abs(m["differential_score"]) for m in layer_heads),
                "mean_belief_attention": float(np.mean([m["belief_attention_mean"] for m in layer_heads])),
                "mean_world_attention": float(np.mean([m["world_attention_mean"] for m in layer_heads])),
                "max_effect_size": max(abs(m["effect_size_fb_vs_tb"]) for m in layer_heads),
            }

    # Layer summary for easy plotting
    layer_summary = {
        "layers": list(range(num_layers)),
        "mean_belief_attention": [layer_aggregates.get(l, {}).get("mean_belief_attention", 0.0) for l in range(num_layers)],
        "mean_world_attention": [layer_aggregates.get(l, {}).get("mean_world_attention", 0.0) for l in range(num_layers)],
        "max_belief_tracking_score": [layer_aggregates.get(l, {}).get("max_belief_tracking", 0.0) for l in range(num_layers)],
        "max_effect_size": [layer_aggregates.get(l, {}).get("max_effect_size", 0.0) for l in range(num_layers)],
    }

    # Distribution data for histograms (aggregate across all heads for top layer)
    # Find the layer with the best belief tracking
    best_layer = max(range(num_layers), key=lambda l: layer_aggregates.get(l, {}).get("max_belief_tracking", 0.0))
    distributions = {
        "best_layer": best_layer,
        "fb_belief_attention": fb_belief[:, best_layer, :].flatten().tolist() if len(fb_data) > 0 else [],
        "tb_belief_attention": tb_belief[:, best_layer, :].flatten().tolist() if len(tb_data) > 0 else [],
        "fb_world_attention": fb_world[:, best_layer, :].flatten().tolist() if len(fb_data) > 0 else [],
        "tb_world_attention": tb_world[:, best_layer, :].flatten().tolist() if len(tb_data) > 0 else [],
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
                {
                    "layer": m["layer"],
                    "head": m["head"],
                    "score": m["belief_tracking_score"],
                    "effect_size": m["effect_size_fb_vs_tb"],
                    "p_value": m.get("p_value", 1.0),
                }
                for m in top_belief_tracking
            ],
            "top_differentiating_heads": [
                {
                    "layer": m["layer"],
                    "head": m["head"],
                    "diff": m["differential_score"],
                    "effect_size": m["effect_size_fb_vs_tb"],
                    "p_value": m.get("p_value", 1.0),
                }
                for m in top_differentiating
            ],
            "layer_aggregates": {str(k): v for k, v in layer_aggregates.items()},
            "layer_summary": layer_summary,
            "distributions": distributions,
            # Sample tasks with detailed attention data (for inspection/visualization)
            "sample_tasks": [
                {
                    "task_id": d["task_id"],
                    "task_type": d["task_type"],
                    "is_correct": d["is_correct"],
                    "token_positions": d["token_positions"],
                    "belief_attention_by_layer": d["belief_attention"].mean(axis=1).tolist(),  # Mean across heads per layer
                    "world_attention_by_layer": d["world_attention"].mean(axis=1).tolist(),
                }
                for d in task_attention_data[:config.save_sample_heatmaps]
            ],
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

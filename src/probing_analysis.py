"""
Hidden-state probing analysis for Theory of Mind tasks.

Runs a linear probe to decode:
- false belief (belief != world)
- world location (where object actually is)
- belief location (where agent thinks object is)

Uses hidden states from ModelRunner without generation.
Uses 10-fold cross-validation with family-based splits.
Probes at multiple positions: after_world, after_belief, prompt_end.
"""

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

from .task_generator import ToMTask, load_tasks
from .model_runner import ModelRunner, ModelOutput


# Position types for probing
PROBE_POSITIONS = ("after_world", "after_belief", "prompt_end")


@dataclass
class ProbingConfig:
    seed: int = 42
    label_definitions: Tuple[str, ...] = ("false_belief", "world_location", "belief_location")
    probe_positions: Tuple[str, ...] = PROBE_POSITIONS
    include_embedding_layer: bool = True
    n_folds: int = 10
    normalize: bool = True
    filter_correct: bool = False


def _compute_tasks_hash(tasks: List[ToMTask], config: ProbingConfig, num_layers: int) -> str:
    """Compute a hash of task IDs and config for cache validation."""
    task_ids = sorted(t.task_id for t in tasks)
    # Include config settings and model architecture that affect cached data
    positions_str = ",".join(config.probe_positions)
    content = ",".join(task_ids) + f"|embed={config.include_embedding_layer}|layers={num_layers}|pos={positions_str}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _article(word: str) -> str:
    """Return 'a' or 'an' depending on word's first letter."""
    return "an" if word[0].lower() in "aeiou" else "a"


def _build_partial_prompts(task: ToMTask) -> Dict[str, str]:
    """
    Build partial prompts ending at each probe position.

    Returns dict mapping position name to partial prompt string.
    Task structure:
        "There is a [c1] and a [c2]. A [obj] is in the [world].
         [Name] believes the [obj] is in the [belief]. [Name] will look..."
    """
    order = task.order
    obj = task.obj
    world = task.world
    belief = task.belief
    agent = task.protagonist

    # Build each segment
    preamble = f"There is {_article(order[0])} {order[0]} and {_article(order[1])} {order[1]}."
    world_stmt = f"{_article(obj).capitalize()} {obj} is in the {world}."
    belief_stmt = f"{agent} believes the {obj} is in the {belief}."

    return {
        "after_world": f"{preamble} {world_stmt}",
        "after_belief": f"{preamble} {world_stmt} {belief_stmt}",
        "prompt_end": task.full_prompt,
    }


def _find_token_positions(
    task: ToMTask,
    tokenizer,
    full_input_ids: "torch.Tensor",
) -> Dict[str, int]:
    """
    Find token indices for each probe position within the full tokenized prompt.

    Returns dict mapping position name to token index (0-indexed).
    The index points to the last token of each partial prompt.
    """
    partial_prompts = _build_partial_prompts(task)
    positions = {}

    for pos_name, partial_prompt in partial_prompts.items():
        # Tokenize the partial prompt
        partial_ids = tokenizer(partial_prompt, return_tensors="pt").input_ids[0]
        # The position is the last token of the partial prompt (0-indexed)
        positions[pos_name] = len(partial_ids) - 1

    return positions


def _create_family_folds(
    families: List[str],
    n_folds: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    """Create n_folds train/test splits based on families."""
    unique_families = sorted(set(families))
    rng = random.Random(seed)
    rng.shuffle(unique_families)

    # Assign families to folds
    fold_size = max(1, len(unique_families) // n_folds)
    family_to_fold = {}
    for i, fam in enumerate(unique_families):
        family_to_fold[fam] = i // fold_size if i // fold_size < n_folds else n_folds - 1

    # Create train/test indices for each fold
    folds = []
    for fold_idx in range(n_folds):
        test_fams = {f for f, fold in family_to_fold.items() if fold == fold_idx}
        train_idx = [i for i, f in enumerate(families) if f not in test_fams]
        test_idx = [i for i, f in enumerate(families) if f in test_fams]
        folds.append((train_idx, test_idx))

    return folds


def _train_probe_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normalize: bool,
    seed: int,
) -> Tuple[Dict, np.ndarray]:
    """Train logistic regression probe on one fold."""
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=5000, random_state=seed, solver='lbfgs')
    clf.fit(X_train, y_train)

    # Handle single-class case where predict_proba returns only one column
    proba = clf.predict_proba(X_test)
    probs = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probs)) if len(y_test) > 0 and len(np.unique(y_test)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_test, probs)) if len(y_test) > 0 else 0.0,
    }

    return metrics, probs


def _build_label(task: ToMTask, label_type: str) -> int:
    if label_type == "false_belief":
        return 1 if task.belief != task.world else 0
    if label_type == "world_location":
        return 1 if task.world == task.c1 else 0  # 1=c1, 0=c2
    if label_type == "belief_location":
        return 1 if task.belief == task.c1 else 0  # 1=c1, 0=c2
    raise ValueError(f"Unknown label type: {label_type}")


def run_probing_analysis(
    runner: ModelRunner,
    tasks: List[ToMTask],
    outputs: Optional[List[ModelOutput]],
    results_dir: Path,
    config: Optional[ProbingConfig] = None,
) -> Dict:
    config = config or ProbingConfig()

    required_model = "mistralai/Mistral-7B-v0.3"
    if runner.model_name != required_model:
        raise ValueError(
            f"Probing is restricted to {required_model}. "
            f"Got: {runner.model_name}"
        )

    # Use all tasks (optionally filter to correct-only)
    if config.filter_correct and outputs:
        correct_lookup = {o.task_id: o.is_correct for o in outputs}
        selected_tasks = [t for t in tasks if correct_lookup.get(t.task_id, False)]
    else:
        selected_tasks = tasks

    layer_count = runner.num_layers + 1 if config.include_embedding_layer else runner.num_layers

    # Compute hash for cache validation (includes config and model architecture)
    tasks_hash = _compute_tasks_hash(selected_tasks, config, runner.num_layers)
    layer_indices = list(range(layer_count))

    safe_model_name = runner.model_name.replace("/", "_")
    probing_dir = results_dir / "probing" / safe_model_name
    probing_dir.mkdir(parents=True, exist_ok=True)
    cache_file = probing_dir / "hidden_states_multipos.npz"

    # Try to load cached hidden states (with hash validation)
    cache_status = "miss"
    hidden_state_time_ms = 0.0
    cache_load_time_ms = 0.0
    cache_valid = False
    task_ids: List[str] = []
    family_ids: List[str] = []
    # X_by_position[pos][layer] = array of shape (n_tasks, hidden_size)
    X_by_position: Dict[str, List[np.ndarray]] = {pos: [] for pos in config.probe_positions}
    labels: Dict[str, List[int]] = {}

    if cache_file.exists():
        cache_start = time.time()
        cached = np.load(cache_file, allow_pickle=True)
        cached_hash = str(cached.get("tasks_hash", ""))
        if cached_hash == tasks_hash:
            print(f"Loading cached hidden states from {cache_file}")
            cache_status = "hit"
            cache_valid = True
            task_ids = cached["task_ids"].tolist()
            family_ids = cached["family_ids"].tolist()
            for pos in config.probe_positions:
                X_by_position[pos] = [cached[f"{pos}_layer_{i}"] for i in layer_indices]
            labels = {lt: cached[f"labels_{lt}"].tolist() for lt in config.label_definitions}
            cache_load_time_ms = (time.time() - cache_start) * 1000
        else:
            print("Cache invalid (hash mismatch), re-extracting hidden states...")

    if not cache_valid:
        print("Extracting hidden states from model at multiple positions...")
        extract_start = time.time()

        # Collect features and metadata
        # features_by_position[pos][layer] = list of vectors
        features_by_position: Dict[str, List[List[np.ndarray]]] = {
            pos: [[] for _ in layer_indices] for pos in config.probe_positions
        }
        labels = {lt: [] for lt in config.label_definitions}

        for task, input_ids, tokens, hidden_states, input_length in runner.iter_hidden_states(selected_tasks):
            task_ids.append(task.task_id)
            family_ids.append(task.family_id)

            # Compute token positions for each probe location
            positions = _find_token_positions(task, runner.tokenizer, input_ids)

            for label_type in config.label_definitions:
                labels[label_type].append(_build_label(task, label_type))

            # Extract hidden states at each position
            for pos_name in config.probe_positions:
                token_idx = positions[pos_name]
                for layer_idx in layer_indices:
                    vec = hidden_states[layer_idx][token_idx].float().cpu().numpy()
                    features_by_position[pos_name][layer_idx].append(vec)

        # Stack features
        for pos in config.probe_positions:
            X_by_position[pos] = [np.stack(vecs) for vecs in features_by_position[pos]]

        # Save to cache (include hash for validation)
        print(f"Saving hidden states to {cache_file}")
        save_dict = {
            "tasks_hash": np.array(tasks_hash),
            "task_ids": np.array(task_ids),
            "family_ids": np.array(family_ids),
        }
        for pos in config.probe_positions:
            for i, X in enumerate(X_by_position[pos]):
                save_dict[f"{pos}_layer_{i}"] = X
        for lt, lbl in labels.items():
            save_dict[f"labels_{lt}"] = np.array(lbl)
        np.savez(cache_file, **save_dict)
        hidden_state_time_ms = (time.time() - extract_start) * 1000

    # Create CV folds
    folds = _create_family_folds(family_ids, config.n_folds, config.seed)

    timestamp = datetime.now().isoformat()
    manifest = {
        "model_name": runner.model_name,
        "timestamp": timestamp,
        "num_tasks": len(selected_tasks),
        "num_families": len(set(family_ids)),
        "label_definitions": list(config.label_definitions),
        "probe_positions": list(config.probe_positions),
        "layers": layer_indices,
        "n_folds": config.n_folds,
        "filter_correct": config.filter_correct,
        "seed": config.seed,
        "cache_status": cache_status,
        "hidden_state_time_ms": hidden_state_time_ms,
        "cache_load_time_ms": cache_load_time_ms,
    }
    (probing_dir / "probe_manifest.json").write_text(json.dumps(manifest, indent=2))

    results = {
        "model_name": runner.model_name,
        "timestamp": timestamp,
        "config": asdict(config),
        "probe_positions": list(config.probe_positions),
        "timing": {
            "cache_status": cache_status,
            "hidden_state_time_ms": hidden_state_time_ms,
            "cache_load_time_ms": cache_load_time_ms,
            "probe_train_time_ms": 0.0,
        },
        "results": {},  # results[position][label_type] = {...}
    }

    # Per-task predictions: raw_predictions[position][label_type][layer][task_id]
    raw_predictions: Dict[str, Dict[str, Dict[int, Dict[str, dict]]]] = {}

    probe_start = time.time()

    # Iterate over positions × labels
    for pos_name in config.probe_positions:
        print(f"\nProbing at position: {pos_name}")
        results["results"][pos_name] = {}
        raw_predictions[pos_name] = {}
        X_by_layer = X_by_position[pos_name]

        for label_type in config.label_definitions:
            y = np.array(labels[label_type])
            raw_predictions[pos_name][label_type] = {}

            layer_metrics = []
            best_layer = None
            best_layer_i = None
            best_acc = -1.0

            for layer_i, layer_idx in enumerate(layer_indices):
                X = X_by_layer[layer_i]

                # Collect metrics across folds
                fold_metrics = []
                all_probs = np.zeros(len(y))

                for train_idx, test_idx in folds:
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    metrics, probs = _train_probe_fold(
                        X_train, y_train, X_test, y_test,
                        config.normalize, config.seed
                    )
                    fold_metrics.append(metrics)

                    # Store predictions for test samples
                    all_probs[test_idx] = probs

                # Average metrics across folds
                avg_metrics = {
                    "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
                    "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics])),
                    "f1": float(np.mean([m["f1"] for m in fold_metrics])),
                    "auc": float(np.mean([m["auc"] for m in fold_metrics])),
                    "brier": float(np.mean([m["brier"] for m in fold_metrics])),
                    "n_samples": len(y),
                    "n_folds": config.n_folds,
                }

                # Store per-task predictions
                layer_preds = {
                    task_ids[idx]: {
                        "prob": float(all_probs[idx]),
                        "label": int(y[idx]),
                        "family": family_ids[idx],
                    }
                    for idx in range(len(y))
                }
                raw_predictions[pos_name][label_type][layer_idx] = layer_preds

                layer_metrics.append({
                    "layer": layer_idx,
                    **avg_metrics,
                })

                if avg_metrics["accuracy"] > best_acc:
                    best_acc = avg_metrics["accuracy"]
                    best_layer = layer_idx
                    best_layer_i = layer_i

            results["results"][pos_name][label_type] = {
                "layers": layer_indices,
                "metrics": layer_metrics,
                "best_layer": best_layer,
                "best_accuracy": best_acc,
            }

            print(f"  {label_type}: best_layer={best_layer}, best_acc={best_acc:.3f}")

            # Train final probe on all data for best layer and save weights
            if best_layer_i is not None and best_layer is not None:
                X_best = X_by_layer[best_layer_i]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_best)
                clf = LogisticRegression(max_iter=5000, random_state=config.seed, solver='lbfgs')
                clf.fit(X_scaled, y)

                weights_file = probing_dir / f"probe_weights_{pos_name}_{label_type}.npz"
                assert scaler.mean_ is not None and scaler.scale_ is not None
                np.savez(
                    weights_file,
                    coef=clf.coef_,
                    intercept=clf.intercept_,
                    scaler_mean=scaler.mean_,
                    scaler_scale=scaler.scale_,
                    best_layer=best_layer,
                    position=pos_name,
                )

    results["timing"]["probe_train_time_ms"] = (time.time() - probe_start) * 1000

    # Save aggregate metrics
    (probing_dir / "probe_metrics_multipos.json").write_text(json.dumps(results, indent=2))

    # Save raw per-task predictions
    (probing_dir / "probe_predictions_multipos.json").write_text(json.dumps(raw_predictions, indent=2))

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run probing analysis for ToM tasks")
    parser.add_argument("--model", "-m", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--tasks-file", "-t", type=Path, default=Path("tasks.json"))
    parser.add_argument("--results-dir", "-o", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-correct", action="store_true")

    args = parser.parse_args()

    if not args.tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {args.tasks_file}")

    tasks = load_tasks(args.tasks_file)
    runner = ModelRunner(args.model)
    # Model validation happens in run_probing_analysis()
    config = ProbingConfig(
        seed=args.seed,
        filter_correct=args.filter_correct,
    )

    run_probing_analysis(runner, tasks, outputs=None, results_dir=args.results_dir, config=config)


if __name__ == "__main__":
    main()

"""
Hidden-state probing analysis for Theory of Mind tasks.

Runs a linear probe to decode:
- false belief (belief != world)

Uses hidden states from ModelRunner without generation.
Uses 10-fold cross-validation with family-based splits.
"""

import json
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
import torch

from .task_generator import ToMTask, load_tasks
from .model_runner import ModelRunner, ModelOutput


@dataclass
class ProbingConfig:
    num_families: int = 200
    seed: int = 42
    label_definitions: Tuple[str, ...] = ("false_belief",)
    include_embedding_layer: bool = True
    n_folds: int = 10
    normalize: bool = True
    filter_correct: bool = False


def _select_tasks_by_family(tasks: List[ToMTask], num_families: int, seed: int) -> List[ToMTask]:
    family_ids = sorted({t.family_id for t in tasks})
    rng = random.Random(seed)
    if num_families >= len(family_ids):
        selected = set(family_ids)
    else:
        selected = set(rng.sample(family_ids, num_families))
    return [t for t in tasks if t.family_id in selected]


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
    fold_size = len(unique_families) // n_folds
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

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_test, probs)),
    }

    return metrics, probs


def _build_label(task: ToMTask, label_type: str) -> int:
    if label_type == "false_belief":
        return 1 if task.belief != task.world else 0
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

    selected_tasks = _select_tasks_by_family(tasks, config.num_families, config.seed)

    correct_lookup = {}
    if outputs:
        correct_lookup = {o.task_id: o.is_correct for o in outputs}

    if config.filter_correct and correct_lookup:
        selected_tasks = [t for t in selected_tasks if correct_lookup.get(t.task_id, False)]

    layer_count = runner.num_layers + 1 if config.include_embedding_layer else runner.num_layers
    layer_indices = list(range(layer_count))

    safe_model_name = runner.model_name.replace("/", "_")
    probing_dir = results_dir / "probing" / safe_model_name
    probing_dir.mkdir(parents=True, exist_ok=True)
    cache_file = probing_dir / "hidden_states.npz"

    # Try to load cached hidden states
    cache_status = "miss"
    hidden_state_time_ms = 0.0
    cache_load_time_ms = 0.0
    if cache_file.exists():
        print(f"Loading cached hidden states from {cache_file}")
        cache_status = "hit"
        cache_start = time.time()
        cached = np.load(cache_file, allow_pickle=True)
        task_ids = cached["task_ids"].tolist()
        family_ids = cached["family_ids"].tolist()
        X_by_layer = [cached[f"layer_{i}"] for i in layer_indices]
        labels = {lt: cached[f"labels_{lt}"].tolist() for lt in config.label_definitions}
        cache_load_time_ms = (time.time() - cache_start) * 1000
    else:
        print(f"Extracting hidden states from model...")
        extract_start = time.time()
        # Collect features and metadata
        task_ids: List[str] = []
        family_ids: List[str] = []
        features_by_layer: List[List[np.ndarray]] = [[] for _ in layer_indices]
        labels: Dict[str, List[int]] = {lt: [] for lt in config.label_definitions}

        for task, _, _, hidden_states, input_length in runner.iter_hidden_states(selected_tasks):
            prompt_end_idx = input_length - 1
            task_ids.append(task.task_id)
            family_ids.append(task.family_id)

            for label_type in config.label_definitions:
                labels[label_type].append(_build_label(task, label_type))

            for layer_i, layer_idx in enumerate(layer_indices):
                vec = hidden_states[layer_idx][prompt_end_idx].float().cpu().numpy()
                features_by_layer[layer_i].append(vec)

        # Stack features
        X_by_layer = [np.stack(vecs) for vecs in features_by_layer]

        # Save to cache
        print(f"Saving hidden states to {cache_file}")
        save_dict = {
            "task_ids": np.array(task_ids),
            "family_ids": np.array(family_ids),
        }
        for i, X in enumerate(X_by_layer):
            save_dict[f"layer_{i}"] = X
        for lt, lbl in labels.items():
            save_dict[f"labels_{lt}"] = np.array(lbl)
        np.savez(cache_file, **save_dict)
        hidden_state_time_ms = (time.time() - extract_start) * 1000

    # Create CV folds
    folds = _create_family_folds(family_ids, config.n_folds, config.seed)

    safe_model_name = runner.model_name.replace("/", "_")
    probing_dir = results_dir / "probing" / safe_model_name
    probing_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_name": runner.model_name,
        "timestamp": datetime.now().isoformat(),
        "num_tasks": len(selected_tasks),
        "num_families": len(set(family_ids)),
        "label_definitions": list(config.label_definitions),
        "layers": layer_indices,
        "n_folds": config.n_folds,
        "filter_correct": config.filter_correct,
        "seed": config.seed,
        "feature_position": "prompt_end",
        "cache_status": cache_status,
        "hidden_state_time_ms": hidden_state_time_ms,
        "cache_load_time_ms": cache_load_time_ms,
    }
    (probing_dir / "probe_manifest.json").write_text(json.dumps(manifest, indent=2))

    results = {
        "model_name": runner.model_name,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "feature_position": "prompt_end",
        "timing": {
            "cache_status": cache_status,
            "hidden_state_time_ms": hidden_state_time_ms,
            "cache_load_time_ms": cache_load_time_ms,
            "probe_train_time_ms": 0.0,
        },
        "results": {},
    }

    # Per-task predictions (raw probabilities) - aggregate across folds
    raw_predictions: Dict[str, Dict[int, Dict[str, dict]]] = {}

    probe_start = time.time()
    for label_type in config.label_definitions:
        y = np.array(labels[label_type])
        raw_predictions[label_type] = {}

        layer_metrics = []
        best_layer = None
        best_acc = -1.0

        for layer_i, layer_idx in enumerate(layer_indices):
            X = X_by_layer[layer_i]

            # Collect metrics across folds
            fold_metrics = []
            all_probs = np.zeros(len(y))
            test_mask = np.zeros(len(y), dtype=bool)

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                metrics, probs = _train_probe_fold(
                    X_train, y_train, X_test, y_test,
                    config.normalize, config.seed
                )
                fold_metrics.append(metrics)

                # Store predictions for test samples
                for i, idx in enumerate(test_idx):
                    all_probs[idx] = probs[i]
                    test_mask[idx] = True

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
            layer_preds = {}
            for idx in range(len(y)):
                if test_mask[idx]:
                    layer_preds[task_ids[idx]] = {
                        "prob": float(all_probs[idx]),
                        "label": int(y[idx]),
                        "family": family_ids[idx],
                    }
            raw_predictions[label_type][layer_idx] = layer_preds

            layer_metrics.append({
                "layer": layer_idx,
                **avg_metrics,
            })

            if avg_metrics["accuracy"] > best_acc:
                best_acc = avg_metrics["accuracy"]
                best_layer = layer_idx

        results["results"][label_type] = {
            "layers": layer_indices,
            "metrics": layer_metrics,
            "best_layer": best_layer,
            "best_accuracy": best_acc,
        }

    results["timing"]["probe_train_time_ms"] = (time.time() - probe_start) * 1000

    # Save aggregate metrics
    (probing_dir / "probe_metrics.json").write_text(json.dumps(results, indent=2))

    # Save raw per-task predictions
    (probing_dir / "probe_predictions.json").write_text(json.dumps(raw_predictions, indent=2))

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run probing analysis for ToM tasks")
    parser.add_argument("--model", "-m", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--tasks-file", "-t", type=Path, default=Path("tasks.json"))
    parser.add_argument("--results-dir", "-o", type=Path, default=Path("results"))
    parser.add_argument("--num-families", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-correct", action="store_true")

    args = parser.parse_args()

    if not args.tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {args.tasks_file}")

    tasks = load_tasks(args.tasks_file)
    runner = ModelRunner(args.model)
    if runner.model_name != "mistralai/Mistral-7B-v0.3":
        raise ValueError(
            "Probing is restricted to mistralai/Mistral-7B-v0.3. "
            f"Got: {runner.model_name}"
        )
    config = ProbingConfig(
        num_families=args.num_families,
        seed=args.seed,
        filter_correct=args.filter_correct,
    )

    run_probing_analysis(runner, tasks, outputs=None, results_dir=args.results_dir, config=config)


if __name__ == "__main__":
    main()

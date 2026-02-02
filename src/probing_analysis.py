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


# Wrapper that adds full_prompt to tasks for model runner compatibility
@dataclass
class ProbingTask:
    """Task wrapper with full_prompt for model runner compatibility."""
    task_id: str
    family_id: str
    task_type: str
    full_prompt: str
    # Store original task reference for label building
    original: ToMTask = None  # type: ignore

    @classmethod
    def from_tom_task(cls, task: ToMTask, question_type: str = "protagonist") -> "ProbingTask":
        """Create a ProbingTask from a ToMTask with the specified question type."""
        question_templates = {
            "reality": f"The {task.obj} is in the",
            "protagonist": f"{task.protagonist} will look for the {task.obj} in the",
            "observer": f"{task.observer} will look for the {task.obj} in the",
        }
        question = question_templates.get(question_type, question_templates["protagonist"])
        full_prompt = f"{task.narrative} {question}"

        return cls(
            task_id=task.task_id,
            family_id=task.family_id,
            task_type=task.task_type,
            full_prompt=full_prompt,
            original=task,
        )


# Position types for probing
# - after_put: After protagonist places object in initial location
# - before_return: After all events, just before protagonist returns
# - prompt_end: Full narrative + question
PROBE_POSITIONS = ("after_put", "before_return", "prompt_end")


@dataclass
class ProbingConfig:
    seed: int = 42
    label_definitions: Tuple[str, ...] = ("false_belief", "world_location", "belief_location")
    probe_positions: Tuple[str, ...] = PROBE_POSITIONS
    include_embedding_layer: bool = True
    n_folds: int = 10
    normalize: bool = True
    filter_correct: bool = False


def _article(word: str) -> str:
    """Return 'a' or 'an' depending on word's first letter."""
    return "an" if word[0].lower() in "aeiou" else "a"


def _build_partial_prompts(task: ToMTask, question_type: str = "protagonist") -> Dict[str, str]:
    """
    Build partial prompts ending at each probe position.

    Returns dict mapping position name to partial prompt string.

    New task structure (unexpected transfer paradigm):
        "In a room there are {protagonist}, {observer}, a {c1}, and a {c2}.
         {protagonist} puts a {obj} in the {initial}.
         [FB: {protagonist} leaves. {observer} moves to {final}. {protagonist} returns.]
         [TB: {observer} moves to {final}. {protagonist} leaves. {protagonist} returns.]"

    Probe positions:
        - after_put: After initial object placement
        - before_return: Just before protagonist returns (after all belief-relevant events)
        - prompt_end: Full prompt with question
    """
    narrative = task.narrative
    protagonist = task.protagonist
    obj = task.obj

    # Parse narrative to find key positions
    # Narrative format: "In a room... {protagonist} puts... [events] {protagonist} returns."
    sentences = [s.strip() + "." for s in narrative.rstrip(".").split(". ")]

    # Find the sentence indices for key positions
    put_idx = None
    return_idx = None
    for i, sent in enumerate(sentences):
        if f"puts {_article(obj)} {obj}" in sent or f"puts a {obj}" in sent or f"puts an {obj}" in sent:
            put_idx = i
        if f"{protagonist} returns" in sent:
            return_idx = i

    # Build partial prompts
    if put_idx is not None:
        after_put = " ".join(sentences[:put_idx + 1])
    else:
        # Fallback: use first two sentences (intro + put)
        after_put = " ".join(sentences[:2])

    if return_idx is not None and return_idx > 0:
        before_return = " ".join(sentences[:return_idx])
    else:
        # Fallback: use all but last sentence
        before_return = " ".join(sentences[:-1]) if len(sentences) > 1 else narrative

    # Build full prompt with question (matching behavioral_analysis.py)
    question_templates = {
        "reality": f"The {obj} is in the",
        "protagonist": f"{protagonist} will look for the {obj} in the",
        "observer": f"{task.observer} will look for the {obj} in the",
    }
    question = question_templates.get(question_type, question_templates["protagonist"])
    full_prompt = f"{narrative} {question}"

    return {
        "after_put": after_put,
        "before_return": before_return,
        "prompt_end": full_prompt,
    }


def _find_token_positions(
    task: ToMTask,
    tokenizer,
    full_input_ids: "torch.Tensor",
    question_type: str = "protagonist",
) -> Dict[str, int]:
    """
    Find token indices for each probe position within the full tokenized prompt.

    Returns dict mapping position name to token index (0-indexed).
    The index points to the last token of each partial prompt.
    """
    partial_prompts = _build_partial_prompts(task, question_type)
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


def _get_first_container(task: ToMTask) -> str:
    """Get the alphabetically first container for consistent encoding."""
    containers = sorted([task.initial_location, task.final_location])
    return containers[0]


def _build_label(task: ToMTask, label_type: str) -> int:
    """
    Build binary labels for probing.

    Labels (using alphabetically first container as reference):
        false_belief: 1 if protagonist has false belief (belief != reality), 0 otherwise
        world_location: 1 if object is in first container (alphabetically), 0 otherwise
        belief_location: 1 if protagonist believes object is in first container, 0 otherwise
    """
    first_container = _get_first_container(task)

    if label_type == "false_belief":
        # False belief = protagonist's belief differs from reality (final_location)
        return 1 if task.protagonist_belief != task.final_location else 0
    if label_type == "world_location":
        # Reality: where is the object actually? (always final_location after the move)
        # Encode: 1 if at alphabetically first container, 0 otherwise
        return 1 if task.final_location == first_container else 0
    if label_type == "belief_location":
        # Belief: where does protagonist think the object is?
        # Encode: 1 if believes it's at alphabetically first container, 0 otherwise
        return 1 if task.protagonist_belief == first_container else 0
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

    # Create wrapper tasks with full_prompt for model runner
    # Use "protagonist" question type for probing (main ToM question)
    question_type = "protagonist"
    probing_tasks = [ProbingTask.from_tom_task(t, question_type) for t in selected_tasks]
    # Keep a lookup from task_id to original task for label building
    original_task_lookup = {t.task_id: t for t in selected_tasks}

    layer_count = runner.num_layers + 1 if config.include_embedding_layer else runner.num_layers
    layer_indices = list(range(layer_count))

    safe_model_name = runner.model_name.replace("/", "_")
    probing_dir = results_dir / "probing" / safe_model_name
    probing_dir.mkdir(parents=True, exist_ok=True)

    # Extract hidden states (model_runner handles caching internally)
    print("Extracting hidden states from model at multiple positions...")
    extract_start = time.time()

    task_ids: List[str] = []
    family_ids: List[str] = []
    # X_by_position[pos][layer] = array of shape (n_tasks, hidden_size)
    X_by_position: Dict[str, List[np.ndarray]] = {pos: [] for pos in config.probe_positions}
    labels: Dict[str, List[int]] = {}

    # Collect features and metadata
    # features_by_position[pos][layer] = list of vectors
    features_by_position: Dict[str, List[List[np.ndarray]]] = {
        pos: [[] for _ in layer_indices] for pos in config.probe_positions
    }
    labels = {lt: [] for lt in config.label_definitions}

    for probing_task, input_ids, tokens, hidden_states, input_length in runner.iter_hidden_states(probing_tasks):
        task_ids.append(probing_task.task_id)
        family_ids.append(probing_task.family_id)

        # Get original task for label building and partial prompts
        original_task = original_task_lookup[probing_task.task_id]

        # Compute token positions for each probe location
        positions = _find_token_positions(original_task, runner.tokenizer, input_ids, question_type)

        for label_type in config.label_definitions:
            labels[label_type].append(_build_label(original_task, label_type))

        # Extract hidden states at each position
        for pos_name in config.probe_positions:
            token_idx = positions[pos_name]
            for layer_idx in layer_indices:
                vec = hidden_states[layer_idx][token_idx].float().cpu().numpy()
                features_by_position[pos_name][layer_idx].append(vec)

    # Stack features
    for pos in config.probe_positions:
        X_by_position[pos] = [np.stack(vecs) for vecs in features_by_position[pos]]

    hidden_state_time_ms = (time.time() - extract_start) * 1000
    cache_stats = runner.get_cache_stats()
    cache_status = "hit" if cache_stats["hits"] > 0 else "miss"

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
        "cache_stats": cache_stats,
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
            "cache_stats": cache_stats,
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

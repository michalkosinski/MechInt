"""
Behavioral probing analysis for Theory of Mind experiments.

Sentence-by-sentence analysis tracking probability of each location
for reality vs. protagonist/observer belief probes.

Similar to Study 1.4 in Kosinski (2024) PNAS paper.
"""

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .model_runner import ModelRunner
from .task_generator import ToMTask, load_tasks


def split_narrative_sentences(narrative: str) -> List[str]:
    """Split narrative into individual sentences."""
    sentences = []
    current = ""
    for char in narrative:
        current += char
        if char == '.':
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def run_behavioral_probing(
    runner: ModelRunner,
    tasks: List[ToMTask],
    num_families: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Run sentence-by-sentence probing analysis.

    Args:
        runner: ModelRunner instance
        tasks: List of ToM tasks
        num_families: Number of families to sample (None = all)
        seed: Random seed for sampling

    Returns:
        Dict with summary, aggregate results, and details
    """
    # Sample families if specified
    if num_families is not None:
        random.seed(seed)
        all_families = list(set(t.family_id for t in tasks))
        sampled_families = set(random.sample(
            all_families,
            min(num_families, len(all_families))
        ))
        tasks = [t for t in tasks if t.family_id in sampled_families]

    results = []

    # Calculate total probes for progress bar
    total_probes = len(tasks) * 6 * 10  # 6 positions × 10 probes per position
    pbar = tqdm(total=total_probes, desc="Probing", unit="probe")

    for task in tasks:
        sentences = split_narrative_sentences(task.narrative)
        containers = [task.initial_location, task.final_location]

        # Query at each position (0 = before any sentences, up to 5 = after all)
        for pos in range(len(sentences) + 1):
            partial = " ".join(sentences[:pos])

            # Query reality: "The [obj] is in the ___"
            reality_prompt = f"{partial} The {task.obj} is in the" if partial else f"The {task.obj} is in the"
            reality_probs = runner.get_candidate_probabilities(reality_prompt, containers)
            pbar.update(1)

            # Query protagonist belief: "[P] thinks the [obj] is in the ___"
            protagonist_prompt = f"{partial} {task.protagonist} thinks the {task.obj} is in the" if partial else f"{task.protagonist} thinks the {task.obj} is in the"
            protagonist_probs = runner.get_candidate_probabilities(protagonist_prompt, containers)
            pbar.update(1)

            # Query observer belief: "[O] thinks the [obj] is in the ___"
            observer_prompt = f"{partial} {task.observer} thinks the {task.obj} is in the" if partial else f"{task.observer} thinks the {task.obj} is in the"
            observer_probs = runner.get_candidate_probabilities(observer_prompt, containers)
            pbar.update(1)

            # Query fallout: "The [obj] falls out of the ___"
            fallout_prompt = f"{partial} The {task.obj} falls out of the" if partial else f"The {task.obj} falls out of the"
            fallout_probs = runner.get_candidate_probabilities(fallout_prompt, containers)
            pbar.update(1)

            # Query protagonist look: "[P] will look for the [obj] in the ___"
            protagonist_look_prompt = f"{partial} {task.protagonist} will look for the {task.obj} in the" if partial else f"{task.protagonist} will look for the {task.obj} in the"
            protagonist_look_probs = runner.get_candidate_probabilities(protagonist_look_prompt, containers)
            pbar.update(1)

            # Query observer look: "[O] will look for the [obj] in the ___"
            observer_look_prompt = f"{partial} {task.observer} will look for the {task.obj} in the" if partial else f"{task.observer} will look for the {task.obj} in the"
            observer_look_probs = runner.get_candidate_probabilities(observer_look_prompt, containers)
            pbar.update(1)

            # Query protagonist gets: "[P] goes to get the [obj] from the ___"
            protagonist_gets_prompt = f"{partial} {task.protagonist} goes to get the {task.obj} from the" if partial else f"{task.protagonist} goes to get the {task.obj} from the"
            protagonist_gets_probs = runner.get_candidate_probabilities(protagonist_gets_prompt, containers)
            pbar.update(1)

            # Query observer gets: "[O] goes to get the [obj] from the ___"
            observer_gets_prompt = f"{partial} {task.observer} goes to get the {task.obj} from the" if partial else f"{task.observer} goes to get the {task.obj} from the"
            observer_gets_probs = runner.get_candidate_probabilities(observer_gets_prompt, containers)
            pbar.update(1)

            # Query protagonist walks: "To retrieve the [obj], [P] walks to the ___"
            protagonist_walks_prompt = f"{partial} To retrieve the {task.obj}, {task.protagonist} walks to the" if partial else f"To retrieve the {task.obj}, {task.protagonist} walks to the"
            protagonist_walks_probs = runner.get_candidate_probabilities(protagonist_walks_prompt, containers)
            pbar.update(1)

            # Query observer walks: "To retrieve the [obj], [O] walks to the ___"
            observer_walks_prompt = f"{partial} To retrieve the {task.obj}, {task.observer} walks to the" if partial else f"To retrieve the {task.obj}, {task.observer} walks to the"
            observer_walks_probs = runner.get_candidate_probabilities(observer_walks_prompt, containers)
            pbar.update(1)

            results.append({
                "task_id": task.task_id,
                "task_type": task.task_type,
                "family_id": task.family_id,
                "position": pos,
                "initial_location": task.initial_location,
                "final_location": task.final_location,
                "reality_probs": reality_probs,
                "protagonist_probs": protagonist_probs,
                "observer_probs": observer_probs,
                "fallout_probs": fallout_probs,
                "protagonist_look_probs": protagonist_look_probs,
                "observer_look_probs": observer_look_probs,
                "protagonist_gets_probs": protagonist_gets_probs,
                "observer_gets_probs": observer_gets_probs,
                "protagonist_walks_probs": protagonist_walks_probs,
                "observer_walks_probs": observer_walks_probs,
            })

    pbar.close()

    # Aggregate results
    aggregate = compute_aggregate(results)

    return {
        "summary": {
            "model_name": runner.model_name,
            "num_families": len(set(r["family_id"] for r in results)),
            "num_tasks": len(set(r["task_id"] for r in results)),
            "timestamp": datetime.now().isoformat(),
        },
        "aggregate": aggregate,
        "details": results,
    }


def compute_aggregate(results: List[Dict]) -> Dict:
    """
    Compute aggregate statistics by task type and position.

    Returns mean P(initial_location) and P(final_location) for each probe type.
    """
    # Group by task_type and position
    groups = defaultdict(list)
    for r in results:
        groups[(r["task_type"], r["position"])].append(r)

    aggregate = {
        "false_belief": {"by_position": []},
        "true_belief": {"by_position": []},
    }

    for task_type in ["false_belief", "true_belief"]:
        positions_data = []
        for pos in range(6):  # 0-5
            key = (task_type, pos)
            if key not in groups:
                continue

            pos_results = groups[key]

            # Compute mean probability for each location and probe
            reality_initial = []
            reality_final = []
            protagonist_initial = []
            protagonist_final = []
            observer_initial = []
            observer_final = []
            fallout_initial = []
            fallout_final = []
            protagonist_look_initial = []
            protagonist_look_final = []
            observer_look_initial = []
            observer_look_final = []
            protagonist_gets_initial = []
            protagonist_gets_final = []
            observer_gets_initial = []
            observer_gets_final = []
            protagonist_walks_initial = []
            protagonist_walks_final = []
            observer_walks_initial = []
            observer_walks_final = []

            for r in pos_results:
                init_loc = r["initial_location"]
                final_loc = r["final_location"]

                reality_initial.append(r["reality_probs"].get(init_loc, 0))
                reality_final.append(r["reality_probs"].get(final_loc, 0))
                protagonist_initial.append(r["protagonist_probs"].get(init_loc, 0))
                protagonist_final.append(r["protagonist_probs"].get(final_loc, 0))
                observer_initial.append(r["observer_probs"].get(init_loc, 0))
                observer_final.append(r["observer_probs"].get(final_loc, 0))
                fallout_initial.append(r.get("fallout_probs", {}).get(init_loc, 0))
                fallout_final.append(r.get("fallout_probs", {}).get(final_loc, 0))
                protagonist_look_initial.append(r.get("protagonist_look_probs", {}).get(init_loc, 0))
                protagonist_look_final.append(r.get("protagonist_look_probs", {}).get(final_loc, 0))
                observer_look_initial.append(r.get("observer_look_probs", {}).get(init_loc, 0))
                observer_look_final.append(r.get("observer_look_probs", {}).get(final_loc, 0))
                protagonist_gets_initial.append(r.get("protagonist_gets_probs", {}).get(init_loc, 0))
                protagonist_gets_final.append(r.get("protagonist_gets_probs", {}).get(final_loc, 0))
                observer_gets_initial.append(r.get("observer_gets_probs", {}).get(init_loc, 0))
                observer_gets_final.append(r.get("observer_gets_probs", {}).get(final_loc, 0))
                protagonist_walks_initial.append(r.get("protagonist_walks_probs", {}).get(init_loc, 0))
                protagonist_walks_final.append(r.get("protagonist_walks_probs", {}).get(final_loc, 0))
                observer_walks_initial.append(r.get("observer_walks_probs", {}).get(init_loc, 0))
                observer_walks_final.append(r.get("observer_walks_probs", {}).get(final_loc, 0))

            n = len(pos_results)
            positions_data.append({
                "position": pos,
                "n": n,
                "reality_initial": sum(reality_initial) / n,
                "reality_final": sum(reality_final) / n,
                "protagonist_initial": sum(protagonist_initial) / n,
                "protagonist_final": sum(protagonist_final) / n,
                "observer_initial": sum(observer_initial) / n,
                "observer_final": sum(observer_final) / n,
                "fallout_initial": sum(fallout_initial) / n,
                "fallout_final": sum(fallout_final) / n,
                "protagonist_look_initial": sum(protagonist_look_initial) / n,
                "protagonist_look_final": sum(protagonist_look_final) / n,
                "observer_look_initial": sum(observer_look_initial) / n,
                "observer_look_final": sum(observer_look_final) / n,
                "protagonist_gets_initial": sum(protagonist_gets_initial) / n,
                "protagonist_gets_final": sum(protagonist_gets_final) / n,
                "observer_gets_initial": sum(observer_gets_initial) / n,
                "observer_gets_final": sum(observer_gets_final) / n,
                "protagonist_walks_initial": sum(protagonist_walks_initial) / n,
                "protagonist_walks_final": sum(protagonist_walks_final) / n,
                "observer_walks_initial": sum(observer_walks_initial) / n,
                "observer_walks_final": sum(observer_walks_final) / n,
            })

        aggregate[task_type]["by_position"] = positions_data

    return aggregate


def save_results(results: Dict, output_dir: Path, model_name: str) -> Path:
    """Save probing results to JSON and CSV files."""
    model_slug = model_name.replace("/", "_")
    probing_dir = output_dir / model_slug / "behavioral_probing"
    probing_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    output_path = probing_dir / "probing_results.json"
    output_path.write_text(json.dumps(results, indent=2))

    # Save aggregated CSV
    csv_path = probing_dir / "probing_aggregate.csv"
    save_aggregate_csv(results["aggregate"], csv_path)

    # Save detailed/raw CSV
    raw_csv_path = probing_dir / "probing_raw.csv"
    save_raw_csv(results["details"], raw_csv_path)

    return output_path


def save_raw_csv(details: List[Dict], csv_path: Path) -> None:
    """Save raw per-task probing results to CSV."""
    headers = [
        "task_id", "family_id", "task_type", "position",
        "initial_location", "final_location",
        "reality_p_initial", "reality_p_final",
        "protagonist_p_initial", "protagonist_p_final",
        "observer_p_initial", "observer_p_final",
        "fallout_p_initial", "fallout_p_final",
        "protagonist_look_p_initial", "protagonist_look_p_final",
        "observer_look_p_initial", "observer_look_p_final",
        "protagonist_gets_p_initial", "protagonist_gets_p_final",
        "observer_gets_p_initial", "observer_gets_p_final",
        "protagonist_walks_p_initial", "protagonist_walks_p_final",
        "observer_walks_p_initial", "observer_walks_p_final",
    ]

    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in details:
            init_loc = r["initial_location"]
            final_loc = r["final_location"]
            row = [
                r["task_id"],
                r["family_id"],
                r["task_type"],
                r["position"],
                init_loc,
                final_loc,
                r["reality_probs"].get(init_loc, 0),
                r["reality_probs"].get(final_loc, 0),
                r["protagonist_probs"].get(init_loc, 0),
                r["protagonist_probs"].get(final_loc, 0),
                r["observer_probs"].get(init_loc, 0),
                r["observer_probs"].get(final_loc, 0),
                r.get("fallout_probs", {}).get(init_loc, 0),
                r.get("fallout_probs", {}).get(final_loc, 0),
                r.get("protagonist_look_probs", {}).get(init_loc, 0),
                r.get("protagonist_look_probs", {}).get(final_loc, 0),
                r.get("observer_look_probs", {}).get(init_loc, 0),
                r.get("observer_look_probs", {}).get(final_loc, 0),
                r.get("protagonist_gets_probs", {}).get(init_loc, 0),
                r.get("protagonist_gets_probs", {}).get(final_loc, 0),
                r.get("observer_gets_probs", {}).get(init_loc, 0),
                r.get("observer_gets_probs", {}).get(final_loc, 0),
                r.get("protagonist_walks_probs", {}).get(init_loc, 0),
                r.get("protagonist_walks_probs", {}).get(final_loc, 0),
                r.get("observer_walks_probs", {}).get(init_loc, 0),
                r.get("observer_walks_probs", {}).get(final_loc, 0),
            ]
            f.write(",".join(str(v) for v in row) + "\n")


def save_aggregate_csv(aggregate: Dict, csv_path: Path) -> None:
    """Save aggregated probing results to CSV."""
    headers = [
        "task_type", "position",
        "reality_initial", "reality_final",
        "protagonist_initial", "protagonist_final",
        "observer_initial", "observer_final",
        "fallout_initial", "fallout_final",
        "protagonist_look_initial", "protagonist_look_final",
        "observer_look_initial", "observer_look_final",
        "protagonist_gets_initial", "protagonist_gets_final",
        "observer_gets_initial", "observer_gets_final",
        "protagonist_walks_initial", "protagonist_walks_final",
        "observer_walks_initial", "observer_walks_final",
        "n"
    ]

    rows = []
    for task_type in ["false_belief", "true_belief"]:
        for pos_data in aggregate[task_type]["by_position"]:
            rows.append([
                task_type,
                pos_data["position"],
                pos_data.get("reality_initial", 0),
                pos_data.get("reality_final", 0),
                pos_data.get("protagonist_initial", 0),
                pos_data.get("protagonist_final", 0),
                pos_data.get("observer_initial", 0),
                pos_data.get("observer_final", 0),
                pos_data.get("fallout_initial", 0),
                pos_data.get("fallout_final", 0),
                pos_data.get("protagonist_look_initial", 0),
                pos_data.get("protagonist_look_final", 0),
                pos_data.get("observer_look_initial", 0),
                pos_data.get("observer_look_final", 0),
                pos_data.get("protagonist_gets_initial", 0),
                pos_data.get("protagonist_gets_final", 0),
                pos_data.get("observer_gets_initial", 0),
                pos_data.get("observer_gets_final", 0),
                pos_data.get("protagonist_walks_initial", 0),
                pos_data.get("protagonist_walks_final", 0),
                pos_data.get("observer_walks_initial", 0),
                pos_data.get("observer_walks_final", 0),
                pos_data.get("n", 0),
            ])

    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run behavioral probing analysis")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Model name")
    parser.add_argument("--families", type=int, default=50, help="Number of families to analyze")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load tasks
    tasks = load_tasks(Path(__file__).parent.parent / "tasks.json")
    print(f"Loaded {len(tasks)} tasks")

    # Load model
    print(f"Loading {args.model}...")
    runner = ModelRunner(args.model)
    print(f"Model loaded in {runner.load_time_ms:.0f}ms")

    # Run probing analysis
    print(f"Running probing on {args.families} families...")
    results = run_behavioral_probing(runner, tasks, num_families=args.families, seed=args.seed)

    # Save results
    output_path = save_results(
        results,
        Path(__file__).parent.parent / "results",
        args.model,
    )

    # Print summary
    print(f"\nResults saved to: {output_path}")
    print(f"Model: {results['summary']['model_name']}")
    print(f"Families: {results['summary']['num_families']}, Tasks: {results['summary']['num_tasks']}")

    print("\nAggregate by position (P(initial) / P(final)):")
    for task_type in ["false_belief", "true_belief"]:
        print(f"\n{task_type.upper()}:")
        print("Pos | Reality      | P thinks     | O thinks     | Fallout      | P looks      | O looks      | P gets       | O gets       | P walks      | O walks")
        print("-" * 143)
        for pos_data in results["aggregate"][task_type]["by_position"]:
            ri, rf = pos_data['reality_initial'], pos_data['reality_final']
            pi, pf = pos_data['protagonist_initial'], pos_data['protagonist_final']
            oi, of = pos_data['observer_initial'], pos_data['observer_final']
            fi, ff = pos_data.get('fallout_initial', 0), pos_data.get('fallout_final', 0)
            pli, plf = pos_data.get('protagonist_look_initial', 0), pos_data.get('protagonist_look_final', 0)
            oli, olf = pos_data.get('observer_look_initial', 0), pos_data.get('observer_look_final', 0)
            pgi, pgf = pos_data.get('protagonist_gets_initial', 0), pos_data.get('protagonist_gets_final', 0)
            ogi, ogf = pos_data.get('observer_gets_initial', 0), pos_data.get('observer_gets_final', 0)
            pwi, pwf = pos_data.get('protagonist_walks_initial', 0), pos_data.get('protagonist_walks_final', 0)
            owi, owf = pos_data.get('observer_walks_initial', 0), pos_data.get('observer_walks_final', 0)
            print(f"  {pos_data['position']} | {ri:.2f} / {rf:.2f} | {pi:.2f} / {pf:.2f} | {oi:.2f} / {of:.2f} | {fi:.2f} / {ff:.2f} | {pli:.2f} / {plf:.2f} | {oli:.2f} / {olf:.2f} | {pgi:.2f} / {pgf:.2f} | {ogi:.2f} / {ogf:.2f} | {pwi:.2f} / {pwf:.2f} | {owi:.2f} / {owf:.2f}")

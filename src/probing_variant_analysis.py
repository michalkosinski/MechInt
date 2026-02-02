"""
Variant-level analysis of behavioral probing results.

Breaks down probing results by fb1/fb2/tb1/tb2 variants to detect
any asymmetries that might indicate container biases or primacy effects.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def extract_variant(task_id: str) -> str:
    """Extract variant (fb1/fb2/tb1/tb2) from task_id like 'f021_fb1'."""
    return task_id.split("_")[-1]


def analyze_raw_csv(csv_path: Path) -> dict:
    """
    Read probing_raw.csv and aggregate by (variant, position).

    Returns dict with structure:
    {
        (variant, position): {
            'n': count,
            'reality_initial': [values],
            'reality_final': [values],
            ...
        }
    }
    """
    groups = defaultdict(lambda: defaultdict(list))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = extract_variant(row["task_id"])
            position = int(row["position"])
            key = (variant, position)

            # Collect all probe values
            for probe in ["reality", "protagonist", "observer", "fallout",
                          "protagonist_look", "observer_look"]:
                groups[key][f"{probe}_initial"].append(float(row[f"{probe}_p_initial"]))
                groups[key][f"{probe}_final"].append(float(row[f"{probe}_p_final"]))

    return groups


def compute_aggregates(groups: dict) -> list[dict]:
    """Compute mean and stdev for each (variant, position) group."""
    results = []

    for (variant, position), data in sorted(groups.items()):
        n = len(data["reality_initial"])
        row = {
            "variant": variant,
            "position": position,
            "n": n,
        }

        for probe in ["reality", "protagonist", "observer", "fallout",
                      "protagonist_look", "observer_look"]:
            initial_vals = data[f"{probe}_initial"]
            final_vals = data[f"{probe}_final"]

            row[f"{probe}_initial"] = mean(initial_vals)
            row[f"{probe}_final"] = mean(final_vals)
            if n > 1:
                row[f"{probe}_initial_std"] = stdev(initial_vals)
                row[f"{probe}_final_std"] = stdev(final_vals)
            else:
                row[f"{probe}_initial_std"] = 0
                row[f"{probe}_final_std"] = 0

        results.append(row)

    return results


def save_variant_csv(results: list[dict], output_path: Path) -> None:
    """Save variant-level aggregates to CSV."""
    headers = [
        "variant", "position", "n",
        "reality_initial", "reality_final",
        "protagonist_initial", "protagonist_final",
        "observer_initial", "observer_final",
        "fallout_initial", "fallout_final",
        "protagonist_look_initial", "protagonist_look_final",
        "observer_look_initial", "observer_look_final",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def print_comparison(results: list[dict]) -> None:
    """Print summary comparing fb1 vs fb2 and tb1 vs tb2."""
    # Group by position for comparison
    by_pos = defaultdict(dict)
    for row in results:
        by_pos[row["position"]][row["variant"]] = row

    print("\n" + "=" * 100)
    print("VARIANT COMPARISON: Observer Look Probe (P(initial) / P(final))")
    print("=" * 100)

    print("\n### FALSE BELIEF: fb1 vs fb2")
    print("-" * 90)
    print(f"{'Pos':<4} | {'fb1: P(init)':<12} {'P(final)':<12} | {'fb2: P(init)':<12} {'P(final)':<12} | {'Diff init':<10} {'Diff final':<10}")
    print("-" * 90)

    for pos in sorted(by_pos.keys()):
        fb1 = by_pos[pos].get("fb1", {})
        fb2 = by_pos[pos].get("fb2", {})
        if fb1 and fb2:
            fb1_init = fb1.get("observer_look_initial", 0)
            fb1_final = fb1.get("observer_look_final", 0)
            fb2_init = fb2.get("observer_look_initial", 0)
            fb2_final = fb2.get("observer_look_final", 0)
            diff_init = fb1_init - fb2_init
            diff_final = fb1_final - fb2_final
            print(f"{pos:<4} | {fb1_init:<12.3f} {fb1_final:<12.3f} | {fb2_init:<12.3f} {fb2_final:<12.3f} | {diff_init:+.3f}      {diff_final:+.3f}")

    print("\n### TRUE BELIEF: tb1 vs tb2")
    print("-" * 90)
    print(f"{'Pos':<4} | {'tb1: P(init)':<12} {'P(final)':<12} | {'tb2: P(init)':<12} {'P(final)':<12} | {'Diff init':<10} {'Diff final':<10}")
    print("-" * 90)

    for pos in sorted(by_pos.keys()):
        tb1 = by_pos[pos].get("tb1", {})
        tb2 = by_pos[pos].get("tb2", {})
        if tb1 and tb2:
            tb1_init = tb1.get("observer_look_initial", 0)
            tb1_final = tb1.get("observer_look_final", 0)
            tb2_init = tb2.get("observer_look_initial", 0)
            tb2_final = tb2.get("observer_look_final", 0)
            diff_init = tb1_init - tb2_init
            diff_final = tb1_final - tb2_final
            print(f"{pos:<4} | {tb1_init:<12.3f} {tb1_final:<12.3f} | {tb2_init:<12.3f} {tb2_final:<12.3f} | {diff_init:+.3f}      {diff_final:+.3f}")

    # Print all probes at position 5 (final)
    print("\n" + "=" * 100)
    print("ALL PROBES AT POSITION 5 (after full narrative)")
    print("=" * 100)

    probes = ["reality", "protagonist", "observer", "fallout", "protagonist_look", "observer_look"]

    for variant in ["fb1", "fb2", "tb1", "tb2"]:
        row = by_pos.get(5, {}).get(variant, {})
        if row:
            print(f"\n{variant.upper()} (n={row.get('n', 0)}):")
            print(f"  {'Probe':<20} {'P(initial)':<12} {'P(final)':<12} {'Winner':<10}")
            print("  " + "-" * 56)
            for probe in probes:
                p_init = row.get(f"{probe}_initial", 0)
                p_final = row.get(f"{probe}_final", 0)
                winner = "initial" if p_init > p_final else "final"
                print(f"  {probe:<20} {p_init:<12.3f} {p_final:<12.3f} {winner:<10}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze probing results by variant")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Model name")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()

    # Find the raw CSV
    model_slug = args.model.replace("/", "_")
    results_dir = Path(__file__).parent.parent / args.results_dir
    csv_path = results_dir / model_slug / "behavioral_probing" / "probing_raw.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    print(f"Analyzing: {csv_path}")

    # Analyze
    groups = analyze_raw_csv(csv_path)
    results = compute_aggregates(groups)

    # Save
    output_path = csv_path.parent / "probing_by_variant.csv"
    save_variant_csv(results, output_path)
    print(f"Saved: {output_path}")

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()

### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###
### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###
### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###

"""
Task generator for Theory of Mind experiments.

Unexpected transfer paradigm (Sally-Anne style):
- Protagonist puts object in container, then leaves (or sees move, then leaves)
- Observer moves object to different container
- Protagonist returns

4 variants per family:
- tb1/tb2: true belief (protagonist sees move before leaving)
- fb1/fb2: false belief (protagonist leaves before move)
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

OBJECTS = [
    "ring", "coin", "key", "ball", "card", "marble", "button", "chip",
    "token", "disc", "cube", "peg", "bead", "badge", "tag", "toy",
    "marker", "pin", "seal", "medal", "clip"
]

CONTAINERS = ["box", "basket", "drawer", "cupboard", "bag"]

NAMES = [
    "Anna", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Ryan", "Sara", "Tom", "Uma", "Victor", "Wendy", "Xander",
    "Yuki", "Zoe", "Alex", "Beth", "Chris", "Dana", "Emma", "Felix"
]


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


@dataclass
class ToMTask:
    task_id: str
    family_id: str
    task_type: str           # "true_belief" or "false_belief"
    narrative: str
    protagonist: str
    observer: str
    obj: str
    initial_location: str    # where protagonist puts object
    final_location: str      # where observer moves it (= reality)
    protagonist_belief: str  # initial_location for FB, final_location for TB


def make_task(family_id: str, protagonist: str, observer: str,
              obj: str, initial: str, final: str, is_true_belief: bool,
              variant: int) -> ToMTask:
    """Create a single task."""
    suffix = f"tb{variant}" if is_true_belief else f"fb{variant}"

    intro = f"In a room there are {protagonist}, {observer}, {article(initial)} {initial}, and {article(final)} {final}."
    put = f"{protagonist} puts {article(obj)} {obj} in the {initial}."
    leave = f"{protagonist} leaves the room."
    move = f"{observer} moves the {obj} to the {final}."
    returns = f"{protagonist} returns."

    if is_true_belief:
        narrative = f"{intro} {put} {move} {leave} {returns}"
        belief = final
    else:
        narrative = f"{intro} {put} {leave} {move} {returns}"
        belief = initial

    return ToMTask(
        task_id=f"{family_id}_{suffix}",
        family_id=family_id,
        task_type="true_belief" if is_true_belief else "false_belief",
        narrative=narrative,
        protagonist=protagonist,
        observer=observer,
        obj=obj,
        initial_location=initial,
        final_location=final,
        protagonist_belief=belief,
    )


def generate_family(family_id: str, protagonist: str, observer: str,
                    obj: str, c1: str, c2: str) -> list[ToMTask]:
    """Generate 4 variants for one family."""
    tasks = []
    for is_tb in [True, False]:
        for variant, (init, fin) in enumerate([(c1, c2), (c2, c1)], 1):
            tasks.append(make_task(family_id, protagonist, observer, obj, init, fin, is_tb, variant))
    return tasks


def generate_tasks(num_families: int = 500, seed: int = 42) -> list[ToMTask]:
    """Generate num_families × 4 tasks."""
    random.seed(seed)
    tasks = []
    used: set[tuple[str, str, str, str, str]] = set()

    for i in range(num_families):
        while True:
            c1, c2 = random.sample(CONTAINERS, 2)
            protagonist, observer = random.sample(NAMES, 2)
            obj = random.choice(OBJECTS)
            config = (protagonist, observer, obj, c1, c2)
            if config not in used:
                used.add(config)
                break

        family_id = f"f{i + 1:03d}"
        tasks.extend(generate_family(family_id, protagonist, observer, obj, c1, c2))

    return tasks


def save_tasks(tasks: list[ToMTask], path: Path) -> None:
    """Save tasks to JSON and markdown files."""
    tasks = sorted(tasks, key=lambda t: t.task_id)
    families = {t.family_id for t in tasks}
    tb_count = sum(1 for t in tasks if t.task_type == "true_belief")

    data = {
        "version": "7.0",
        "num_tasks": len(tasks),
        "num_families": len(families),
        "task_type_counts": {"true_belief": tb_count, "false_belief": len(tasks) - tb_count},
        "tasks": [asdict(t) for t in tasks],
    }

    Path(path).write_text(json.dumps(data, indent=2))

    md_path = path.with_suffix(".md")
    lines = [f"# ToM Tasks ({len(tasks)} total, {len(families)} families)\n"]
    for i, task in enumerate(tasks, 1):
        lines.append(f"{i}. **{task.task_id}** ({task.task_type}): {task.narrative}\n")
    md_path.write_text("\n".join(lines))


def load_tasks(path: Path) -> list[ToMTask]:
    """Load tasks from JSON file."""
    data = json.loads(Path(path).read_text())
    return [ToMTask(**t) for t in data["tasks"]]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate ToM tasks")
    parser.add_argument("--families", type=int, default=500, help="Number of families (4 tasks each)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="tasks.json", help="Output file path")
    args = parser.parse_args()

    tasks = generate_tasks(num_families=args.families, seed=args.seed)
    save_tasks(tasks, Path(args.output))
    print(f"Generated {len(tasks)} tasks ({args.families} families) -> {args.output}")


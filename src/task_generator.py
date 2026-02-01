### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###
### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###
### DO NOT CHANGE THIS FILE WITHOUT HUMAN APPROVAL. ###

"""
Task generator for Theory of Mind experiments.

8-row truth table design per family (agent, obj, C1, C2):
- order: (C1,C2) or (C2,C1) in preamble
- world: C1 or C2 (where object actually is)
- belief: C1 or C2 (where agent thinks it is)

TB: world == belief (tb1-tb4)
FB: world != belief (fb1-fb4)
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

# (suffix, order_idx, world_idx, belief_idx)
ROWS = [
    ("tb1", (1, 2), 1, 1),
    ("tb2", (2, 1), 1, 1),
    ("tb3", (1, 2), 2, 2),
    ("tb4", (2, 1), 2, 2),
    ("fb1", (1, 2), 1, 2),
    ("fb2", (2, 1), 1, 2),
    ("fb3", (1, 2), 2, 1),
    ("fb4", (2, 1), 2, 1),
]


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


@dataclass
class ToMTask:
    task_id: str                # e.g., "f001_tb3"
    family_id: str              # e.g., "f001"
    task_type: str              # "true_belief" or "false_belief"
    full_prompt: str
    expected_answer: str        # always = belief
    protagonist: str
    obj: str
    c1: str                     # canonical "first" container
    c2: str                     # canonical "second" container
    order: tuple[str, str]      # (first_mentioned, second_mentioned) in preamble
    world: str                  # reality location
    belief: str                 # belief location


def make_task(family_id: str, agent: str, obj: str, c1: str, c2: str,
              row_def: tuple) -> ToMTask:
    """Create a single task from a row definition."""
    suffix, order_idx, world_idx, belief_idx = row_def
    containers = {1: c1, 2: c2}

    order = (containers[order_idx[0]], containers[order_idx[1]])
    world = containers[world_idx]
    belief = containers[belief_idx]
    task_type = "true_belief" if suffix.startswith("tb") else "false_belief"

    prompt = (
        f"There is {article(order[0])} {order[0]} and {article(order[1])} {order[1]}. "
        f"{article(obj).capitalize()} {obj} is in the {world}. "
        f"{agent} believes the {obj} is in the {belief}. "
        f"{agent} will look for the {obj} in the"
    )

    return ToMTask(
        task_id=f"{family_id}_{suffix}",
        family_id=family_id,
        task_type=task_type,
        full_prompt=prompt,
        expected_answer=belief,
        protagonist=agent,
        obj=obj,
        c1=c1,
        c2=c2,
        order=order,
        world=world,
        belief=belief,
    )


def generate_family(family_id: str, agent: str, obj: str, c1: str, c2: str) -> list[ToMTask]:
    """Generate all 8 rows for one family."""
    return [make_task(family_id, agent, obj, c1, c2, row) for row in ROWS]


def generate_tasks(num_families: int = 10, seed: int = 42) -> list[ToMTask]:
    """Generate num_families × 8 tasks."""
    random.seed(seed)
    tasks = []
    used: set[tuple[str, str, str, str]] = set()

    for i in range(num_families):
        while True:
            c1, c2 = random.sample(CONTAINERS, 2)
            agent = random.choice(NAMES)
            obj = random.choice(OBJECTS)
            config = (agent, obj, c1, c2)
            if config not in used:
                used.add(config)
                break

        family_id = f"f{i + 1:03d}"
        tasks.extend(generate_family(family_id, agent, obj, c1, c2))

    random.shuffle(tasks)
    return tasks


def save_tasks(tasks: list[ToMTask], path: Path) -> None:
    """Save tasks to JSON and markdown files."""
    families = {t.family_id for t in tasks}
    tb_count = sum(1 for t in tasks if t.task_type == "true_belief")

    data = {
        "version": "6.0",
        "num_tasks": len(tasks),
        "num_families": len(families),
        "task_type_counts": {"true_belief": tb_count, "false_belief": len(tasks) - tb_count},
        "tasks": [asdict(t) for t in tasks],
    }

    Path(path).write_text(json.dumps(data, indent=2))

    # Save markdown
    md_path = path.with_suffix(".md")
    lines = [f"# ToM Tasks ({len(tasks)} total, {len(families)} families)\n"]
    for i, task in enumerate(tasks, 1):
        lines.append(
            f"{i}. **{task.task_id}** ({task.task_type}): "
            f"{task.full_prompt}**{task.expected_answer}**\n"
        )
    md_path.write_text("\n".join(lines))


def load_tasks(path: Path) -> list[ToMTask]:
    """Load tasks from JSON file."""
    data = json.loads(Path(path).read_text())
    tasks = []
    for t in data["tasks"]:
        t["order"] = tuple(t["order"])
        tasks.append(ToMTask(**t))
    return tasks


if __name__ == "__main__":
    tasks = generate_tasks(num_families=20)
    output_path = Path(__file__).parent.parent / "tasks.json"
    save_tasks(tasks, output_path)

    tb = sum(1 for t in tasks if t.task_type == "true_belief")
    fb = len(tasks) - tb
    print(f"Generated {len(tasks)} tasks ({len(tasks)//8} families × 8)")
    print(f"  Types: {tb} true_belief, {fb} false_belief")
    print(f"\nSaved to: {output_path} and {output_path.with_suffix('.md')}")

"""
Task generator for Theory of Mind experiments.

5 task types (each with reversed variant = 10 total):
- true_belief: belief matches reality
- false_belief: belief ≠ reality
- mismatch: belief ≠ reality (opposite direction)
- negation_trap: "believes not in X"
- belief_overwrite: belief updated mid-story
"""

import json
import random
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

OBJECTS = [
    "ball", "key", "book", "apple", "toy", "coin", "letter", "phone",
    "watch", "ring", "pen", "cup", "doll", "card", "hat", "scarf",
    "wallet", "photo", "ticket", "cookie", "orange", "banana", "marble",
    "bracelet", "glove", "sock", "ribbon", "feather", "shell", "button",
    "crayon", "sticker", "candy", "chocolate", "sandwich", "notebook"
]

CONTAINERS = [
    "box", "basket", "drawer", "cupboard", "bag", "bucket", "chest",
    "jar", "tin", "suitcase", "backpack", "crate", "bin", "trunk", "case"
]

NAMES = [
    "Anna", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Ryan", "Sara", "Tom", "Uma", "Victor", "Wendy", "Xander",
    "Yuki", "Zoe", "Alex", "Beth", "Chris", "Dana", "Emma", "Felix"
]

BASE_TYPES = ["true_belief", "false_belief", "mismatch", "negation_trap", "belief_overwrite"]
PREFIXES = {"true_belief": "tb", "false_belief": "fb", "mismatch": "mm", "negation_trap": "nt", "belief_overwrite": "bo"}


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


@dataclass
class ToMTask:
    task_id: str
    task_type: str
    full_prompt: str
    expected_answer: str
    reality_location: str
    belief_location: str
    protagonist: str
    obj: str


def generate_task(task_id: str, task_type: str, name: str, obj: str, c1: str, c2: str) -> ToMTask:
    """Generate a single ToM task."""
    base = task_type.replace("_reversed", "")
    loc_a, loc_b = (c2, c1) if "_reversed" in task_type else (c1, c2)

    preamble = f"There is {article(c1)} {c1} and {article(c2)} {c2}."
    question = f"{name} will look for the {obj} in the "

    # Task-specific logic
    if base == "true_belief":
        reality, belief = loc_b, loc_b
    elif base == "false_belief":
        reality, belief = loc_a, loc_b
    elif base == "mismatch":
        reality, belief = loc_b, loc_a
    elif base == "negation_trap":
        reality, belief = loc_b, loc_a
    elif base == "belief_overwrite":
        reality, belief = loc_a, loc_b
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Build belief statement
    if base == "negation_trap":
        belief_stmt = f"{name} believes the {obj} is not in the {reality}."
    elif base == "belief_overwrite":
        belief_stmt = f"{name} believes the {obj} is in the {reality}. {name} now believes the {obj} is in the {belief}."
    else:
        belief_stmt = f"{name} believes the {obj} is in the {belief}."

    story = f"{article(obj).capitalize()} {obj} is in the {reality}."
    full_prompt = f"{preamble} {story} {belief_stmt} {question}"

    return ToMTask(
        task_id=task_id,
        task_type=task_type,
        full_prompt=full_prompt,
        expected_answer=belief,
        reality_location=reality,
        belief_location=belief,
        protagonist=name,
        obj=obj,
    )


def generate_tasks(num_per_type: int = 10, seed: int = 42) -> list[ToMTask]:
    """Generate tasks for all 10 task types (5 base + 5 reversed)."""
    random.seed(seed)
    tasks = []
    all_types = BASE_TYPES + [f"{t}_reversed" for t in BASE_TYPES]

    for task_type in all_types:
        base = task_type.replace("_reversed", "")
        prefix = PREFIXES[base]
        suffix = "r" if "_reversed" in task_type else ""
        for i in range(num_per_type):
            c1, c2 = random.sample(CONTAINERS, 2)
            tasks.append(generate_task(
                task_id=f"{prefix}_{i+1}{suffix}",
                task_type=task_type,
                name=random.choice(NAMES),
                obj=random.choice(OBJECTS),
                c1=c1,
                c2=c2,
            ))

    random.shuffle(tasks)
    return tasks


def save_tasks(tasks: list[ToMTask], path: Path) -> None:
    counts = Counter(t.task_type for t in tasks)
    data = {"version": "2.0", "num_tasks": len(tasks), "task_type_counts": dict(counts), "tasks": [asdict(t) for t in tasks]}
    Path(path).write_text(json.dumps(data, indent=2))


def load_tasks(path: Path) -> list[ToMTask]:
    data = json.loads(Path(path).read_text())
    return [ToMTask(**t) for t in data["tasks"]]


if __name__ == "__main__":
    tasks = generate_tasks(num_per_type=10)
    output_path = Path(__file__).parent.parent / "tasks.json"
    save_tasks(tasks, output_path)

    counts = Counter(t.task_type for t in tasks)
    print(f"Generated {len(tasks)} tasks")
    for t, c in sorted(counts.items()):
        print(f"  {t}: {c}")

    print(f"\nSaved to: {output_path}")
    print("\n" + "="*60 + "\nEXAMPLES\n" + "="*60)

    shown = set()
    for task in tasks:
        base = task.task_type.replace("_reversed", "")
        if base not in shown:
            shown.add(base)
            print(f"\n--- {task.task_type.upper()} ---")
            print(task.full_prompt)
            print(f"Answer: {task.expected_answer}")

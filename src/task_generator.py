"""
Task generator for Theory of Mind false-belief experiments.

Generates explicit belief statement tasks:
- False belief: "A ball is in the blue box. Anna believes the ball is in the red box."
- True belief: "A ball is in the blue box. Anna believes the ball is in the blue box."
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Literal, Optional

# Default vocabularies
OBJECTS = [
    "ball", "key", "book", "apple", "toy", "coin", "letter", "phone",
    "watch", "ring", "pen", "cup", "doll", "card", "hat", "scarf"
]

CONTAINERS = [
    "red box", "blue box", "green basket", "wooden drawer",
    "canvas bag", "metal cupboard", "yellow bucket", "white container"
]

NAMES = [
    "Anna", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul"
]


@dataclass
class ToMTask:
    """A Theory of Mind task with explicit belief statement."""
    task_id: str
    task_type: Literal["false_belief", "true_belief"]
    story: str                    # "A ball is in the blue box."
    belief_statement: str         # "Anna believes the ball is in the red box."
    question: str                 # "Where will Anna look for the ball?"
    full_prompt: str              # Complete prompt for the model
    expected_answer: str          # "red box" or "blue box"
    reality_location: str         # The actual location
    belief_location: str          # What protagonist believes
    protagonist_name: str         # "Anna"
    object_name: str              # "ball"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ToMTask":
        return cls(**d)


def get_article(word: str) -> str:
    """Return 'an' if word starts with a vowel sound, otherwise 'a'."""
    return "an" if word[0].lower() in "aeiou" else "a"


def generate_single_task(
    task_id: str,
    task_type: Literal["false_belief", "true_belief"],
    protagonist: str,
    obj: str,
    location1: str,
    location2: str,
) -> ToMTask:
    """
    Generate a single ToM task.

    For false belief: protagonist believes object is in location2, but it's in location1.
    For true belief: protagonist correctly believes object is in location1.
    """
    reality_location = location1

    if task_type == "false_belief":
        belief_location = location2
    else:
        belief_location = location1

    article = get_article(obj)
    story = f"{article.capitalize()} {obj} is in the {reality_location}."
    belief_statement = f"{protagonist} believes the {obj} is in the {belief_location}."
    # Use completion format for faster inference (only need 3 tokens)
    question = f"{protagonist} will look for the {obj} in the"

    full_prompt = f"{story} {belief_statement} {question}"

    return ToMTask(
        task_id=task_id,
        task_type=task_type,
        story=story,
        belief_statement=belief_statement,
        question=question,
        full_prompt=full_prompt,
        expected_answer=belief_location,
        reality_location=reality_location,
        belief_location=belief_location,
        protagonist_name=protagonist,
        object_name=obj,
    )


def generate_tasks(
    num_false_belief: int = 20,
    num_true_belief: int = 20,
    objects: Optional[List[str]] = None,
    containers: Optional[List[str]] = None,
    names: Optional[List[str]] = None,
    seed: Optional[int] = 42,
) -> List[ToMTask]:
    """
    Generate a set of ToM tasks.

    Args:
        num_false_belief: Number of false belief tasks to generate
        num_true_belief: Number of true belief tasks to generate
        objects: List of objects to use (default: OBJECTS)
        containers: List of containers to use (default: CONTAINERS)
        names: List of protagonist names to use (default: NAMES)
        seed: Random seed for reproducibility

    Returns:
        List of ToMTask instances
    """
    if seed is not None:
        random.seed(seed)

    objects = objects or OBJECTS
    containers = containers or CONTAINERS
    names = names or NAMES

    tasks = []

    # Generate false belief tasks
    for i in range(num_false_belief):
        protagonist = random.choice(names)
        obj = random.choice(objects)
        loc1, loc2 = random.sample(containers, 2)

        task = generate_single_task(
            task_id=f"fb_{i:03d}",
            task_type="false_belief",
            protagonist=protagonist,
            obj=obj,
            location1=loc1,
            location2=loc2,
        )
        tasks.append(task)

    # Generate true belief tasks
    for i in range(num_true_belief):
        protagonist = random.choice(names)
        obj = random.choice(objects)
        loc1 = random.choice(containers)
        # For true belief, we still pick loc2 for variety but don't use it
        loc2 = random.choice([c for c in containers if c != loc1])

        task = generate_single_task(
            task_id=f"tb_{i:03d}",
            task_type="true_belief",
            protagonist=protagonist,
            obj=obj,
            location1=loc1,
            location2=loc2,
        )
        tasks.append(task)

    # Shuffle tasks
    random.shuffle(tasks)

    return tasks


def save_tasks(tasks: List[ToMTask], path: Path) -> None:
    """Save tasks to a JSON file."""
    path = Path(path)
    data = {
        "version": "1.0",
        "num_tasks": len(tasks),
        "num_false_belief": sum(1 for t in tasks if t.task_type == "false_belief"),
        "num_true_belief": sum(1 for t in tasks if t.task_type == "true_belief"),
        "tasks": [t.to_dict() for t in tasks],
    }
    path.write_text(json.dumps(data, indent=2))


def load_tasks(path: Path) -> List[ToMTask]:
    """Load tasks from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return [ToMTask.from_dict(t) for t in data["tasks"]]


def get_task_by_id(tasks: List[ToMTask], task_id: str) -> Optional[ToMTask]:
    """Find a task by its ID."""
    for task in tasks:
        if task.task_id == task_id:
            return task
    return None


if __name__ == "__main__":
    # Generate and save default tasks
    tasks = generate_tasks(num_false_belief=20, num_true_belief=20)
    output_path = Path(__file__).parent.parent / "tasks.json"
    save_tasks(tasks, output_path)

    print(f"Generated {len(tasks)} tasks")
    print(f"  False belief: {sum(1 for t in tasks if t.task_type == 'false_belief')}")
    print(f"  True belief: {sum(1 for t in tasks if t.task_type == 'true_belief')}")
    print(f"Saved to: {output_path}")

    # Show example tasks
    print("\nExample false belief task:")
    fb_task = next(t for t in tasks if t.task_type == "false_belief")
    print(f"  Prompt: {fb_task.full_prompt}")
    print(f"  Expected: {fb_task.expected_answer}")

    print("\nExample true belief task:")
    tb_task = next(t for t in tasks if t.task_type == "true_belief")
    print(f"  Prompt: {tb_task.full_prompt}")
    print(f"  Expected: {tb_task.expected_answer}")

"""Tests for task generator."""

import pytest
from pathlib import Path
import tempfile

from src.task_generator import (
    ToMTask, generate_tasks, save_tasks, load_tasks,
    generate_single_task,
)


class TestToMTask:
    """Tests for ToMTask dataclass."""

    def test_create_false_belief_task(self):
        """Test creating a false belief task."""
        task = generate_single_task(
            task_id="test_001",
            task_type="false_belief",
            protagonist="Anna",
            obj="ball",
            location1="blue box",
            location2="red box",
        )

        assert task.task_type == "false_belief"
        assert task.reality_location == "blue box"
        assert task.belief_location == "red box"
        assert task.expected_answer == "red box"
        assert "Anna" in task.full_prompt
        assert "ball" in task.full_prompt

    def test_create_true_belief_task(self):
        """Test creating a true belief task."""
        task = generate_single_task(
            task_id="test_002",
            task_type="true_belief",
            protagonist="Bob",
            obj="key",
            location1="drawer",
            location2="basket",
        )

        assert task.task_type == "true_belief"
        assert task.reality_location == "drawer"
        assert task.belief_location == "drawer"  # Same as reality
        assert task.expected_answer == "drawer"

    def test_false_belief_locations_differ(self):
        """Verify false belief tasks have different reality and belief locations."""
        tasks = generate_tasks(num_false_belief=10, num_true_belief=0)

        for task in tasks:
            assert task.reality_location != task.belief_location, \
                f"Task {task.task_id}: reality and belief locations should differ"

    def test_true_belief_locations_match(self):
        """Verify true belief tasks have matching reality and belief locations."""
        tasks = generate_tasks(num_false_belief=0, num_true_belief=10)

        for task in tasks:
            assert task.reality_location == task.belief_location, \
                f"Task {task.task_id}: reality and belief locations should match"


class TestGenerateTasks:
    """Tests for generate_tasks function."""

    def test_generate_correct_counts(self):
        """Test that correct number of tasks are generated."""
        tasks = generate_tasks(num_false_belief=15, num_true_belief=25)

        fb_count = sum(1 for t in tasks if t.task_type == "false_belief")
        tb_count = sum(1 for t in tasks if t.task_type == "true_belief")

        assert fb_count == 15
        assert tb_count == 25
        assert len(tasks) == 40

    def test_reproducible_with_seed(self):
        """Test that same seed produces same tasks."""
        tasks1 = generate_tasks(num_false_belief=5, num_true_belief=5, seed=123)
        tasks2 = generate_tasks(num_false_belief=5, num_true_belief=5, seed=123)

        for t1, t2 in zip(tasks1, tasks2):
            assert t1.task_id == t2.task_id
            assert t1.full_prompt == t2.full_prompt

    def test_different_with_different_seed(self):
        """Test that different seeds produce different tasks."""
        tasks1 = generate_tasks(num_false_belief=5, num_true_belief=5, seed=123)
        tasks2 = generate_tasks(num_false_belief=5, num_true_belief=5, seed=456)

        # At least some tasks should differ
        prompts1 = {t.full_prompt for t in tasks1}
        prompts2 = {t.full_prompt for t in tasks2}

        assert prompts1 != prompts2


class TestSaveLoadTasks:
    """Tests for save/load functionality."""

    def test_save_and_load(self):
        """Test saving and loading tasks."""
        tasks = generate_tasks(num_false_belief=3, num_true_belief=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_tasks.json"

            save_tasks(tasks, path)
            loaded = load_tasks(path)

            assert len(loaded) == len(tasks)

            for orig, loaded_task in zip(tasks, loaded):
                assert orig.task_id == loaded_task.task_id
                assert orig.task_type == loaded_task.task_type
                assert orig.full_prompt == loaded_task.full_prompt
                assert orig.expected_answer == loaded_task.expected_answer


class TestTaskPromptFormat:
    """Tests for correct prompt formatting."""

    def test_prompt_structure(self):
        """Test that prompts follow the expected structure."""
        task = generate_single_task(
            task_id="test",
            task_type="false_belief",
            protagonist="Anna",
            obj="ball",
            location1="blue box",
            location2="red box",
        )

        # Check story format
        assert "A ball is in the blue box" in task.story

        # Check belief format
        assert "Anna believes the ball is in the red box" in task.belief_statement

        # Check question format
        assert "Where will Anna look for the ball" in task.question

        # Check full prompt combines all
        assert task.story in task.full_prompt
        assert task.belief_statement in task.full_prompt
        assert task.question in task.full_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

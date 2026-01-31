"""
Intervention module for ablating and boosting attention heads.

Uses PyTorch hooks to modify attention weights during inference
to test causal effects of identified ToM heads.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
from tqdm import tqdm

from .task_generator import ToMTask
from .model_runner import ModelRunner, ModelOutput


@dataclass
class InterventionResult:
    """Result from running a task with attention intervention."""
    task_id: str
    task_type: str
    intervention_type: Literal["ablation", "boost", "none"]
    target_heads: List[Tuple[int, int]]  # [(layer, head), ...]
    scale_factor: float                   # 0.0 for ablation, >1 for boost
    original_response: str
    modified_response: str
    original_correct: bool
    modified_correct: bool
    flipped: bool  # Did the answer change?

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d["target_heads"] = [list(t) for t in self.target_heads]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "InterventionResult":
        d = d.copy()
        d["target_heads"] = [tuple(t) for t in d["target_heads"]]
        return cls(**d)


@dataclass
class InterventionSummary:
    """Summary statistics for intervention experiments."""
    intervention_type: str
    target_heads: List[Tuple[int, int]]
    scale_factor: float
    num_tasks: int
    original_accuracy: float
    modified_accuracy: float
    accuracy_delta: float
    flip_rate: float  # Fraction of answers that changed
    # Breakdown by task type
    false_belief_original_acc: float
    false_belief_modified_acc: float
    true_belief_original_acc: float
    true_belief_modified_acc: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_heads"] = [list(t) for t in self.target_heads]
        return d


class AttentionHook:
    """
    Hook for modifying attention weights during forward pass.

    Can scale attention for specific heads (0 for ablation, >1 for boosting).
    """

    def __init__(
        self,
        target_heads: List[int],  # Head indices to modify in this layer
        scale_factor: float = 0.0,  # 0 = ablate, >1 = boost
    ):
        self.target_heads = target_heads
        self.scale_factor = scale_factor
        self.handle = None

    def __call__(self, module, input, output):
        """
        Hook function called during forward pass.

        For attention layers, output is typically a tuple where the first
        element contains the attention output tensor.
        """
        # Get attention output
        if isinstance(output, tuple):
            attn_output = output[0]
            rest = output[1:]
        else:
            attn_output = output
            rest = ()

        # Modify attention for target heads
        # attn_output shape: (batch, seq_len, hidden_dim)
        # We need to work with the attention weights, which are computed internally

        # For most transformer implementations, we can scale the output
        # of specific attention heads by modifying the hidden states
        # This is an approximation - ideally we'd hook into attention weights directly

        # Note: This is a simplified approach. For more precise control,
        # we'd need to modify the attention computation itself.

        return output

    def register(self, module):
        """Register the hook on a module."""
        self.handle = module.register_forward_hook(self)
        return self

    def remove(self):
        """Remove the hook."""
        if self.handle:
            self.handle.remove()
            self.handle = None


class InterventionRunner:
    """
    Runs attention intervention experiments.

    Supports:
    - Ablation: Zero out attention from specific heads
    - Boosting: Scale up attention from specific heads
    """

    def __init__(self, model_runner: ModelRunner):
        """
        Initialize with an existing model runner.

        Args:
            model_runner: The ModelRunner instance with loaded model
        """
        self.model_runner = model_runner
        self.model = model_runner.model
        self.tokenizer = model_runner.tokenizer
        self._hooks: List[AttentionHook] = []

    def _get_attention_layer(self, layer_idx: int):
        """
        Get the attention module for a specific layer.

        This method needs to be adapted based on the model architecture.
        """
        # Try common transformer architectures
        model = self.model

        # Qwen / LLaMA style
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx].self_attn

        # GPT-2 style
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx].attn

        # BERT style
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer[layer_idx].attention.self

        raise ValueError(f"Unknown model architecture: {type(model)}")

    def _create_attention_scale_hook(
        self,
        layer_idx: int,
        head_indices: List[int],
        scale_factor: float,
    ) -> Callable:
        """
        Create a hook that scales attention output for specific heads.

        This modifies the attention output by scaling the contribution
        from specific heads.
        """
        num_heads = self.model_runner.num_heads
        head_dim = self.model.config.hidden_size // num_heads

        def hook(module, input, output):
            # output is typically (attn_output, attn_weights, ...)
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            # attn_output shape: (batch, seq_len, hidden_size)
            batch_size, seq_len, hidden_size = attn_output.shape

            # Reshape to (batch, seq_len, num_heads, head_dim)
            attn_reshaped = attn_output.view(batch_size, seq_len, num_heads, head_dim)

            # Scale target heads
            for head_idx in head_indices:
                attn_reshaped[:, :, head_idx, :] *= scale_factor

            # Reshape back
            modified_output = attn_reshaped.view(batch_size, seq_len, hidden_size)

            if isinstance(output, tuple):
                return (modified_output,) + output[1:]
            return modified_output

        return hook

    def _install_hooks(
        self,
        target_heads: List[Tuple[int, int]],
        scale_factor: float,
    ) -> List:
        """Install hooks for all target heads."""
        handles = []

        # Group heads by layer
        heads_by_layer: Dict[int, List[int]] = {}
        for layer, head in target_heads:
            if layer not in heads_by_layer:
                heads_by_layer[layer] = []
            heads_by_layer[layer].append(head)

        # Install hooks
        for layer_idx, head_indices in heads_by_layer.items():
            try:
                attn_module = self._get_attention_layer(layer_idx)
                hook_fn = self._create_attention_scale_hook(
                    layer_idx, head_indices, scale_factor
                )
                handle = attn_module.register_forward_hook(hook_fn)
                handles.append(handle)
            except Exception as e:
                print(f"Warning: Could not install hook for layer {layer_idx}: {e}")

        return handles

    def _remove_hooks(self, handles: List):
        """Remove all installed hooks."""
        for handle in handles:
            handle.remove()

    def run_intervention(
        self,
        task: ToMTask,
        target_heads: List[Tuple[int, int]],
        scale_factor: float,
        baseline_output: Optional[ModelOutput] = None,
    ) -> InterventionResult:
        """
        Run a single task with attention intervention.

        Args:
            task: The task to run
            target_heads: List of (layer, head) tuples to intervene on
            scale_factor: Scale factor (0 for ablation, >1 for boost)
            baseline_output: Optional pre-computed baseline (no intervention)

        Returns:
            InterventionResult with original and modified outputs
        """
        # Get baseline if not provided
        if baseline_output is None:
            baseline_output = self.model_runner.run_task(
                task, extract_attention=False
            )

        # Install hooks
        handles = self._install_hooks(target_heads, scale_factor)

        try:
            # Run with intervention
            modified_output = self.model_runner.run_task(
                task, extract_attention=False
            )
        finally:
            # Always remove hooks
            self._remove_hooks(handles)

        # Determine intervention type
        if scale_factor == 0.0:
            intervention_type = "ablation"
        elif scale_factor > 1.0:
            intervention_type = "boost"
        else:
            intervention_type = "none"

        return InterventionResult(
            task_id=task.task_id,
            task_type=task.task_type,
            intervention_type=intervention_type,
            target_heads=target_heads,
            scale_factor=scale_factor,
            original_response=baseline_output.model_response,
            modified_response=modified_output.model_response,
            original_correct=baseline_output.is_correct,
            modified_correct=modified_output.is_correct,
            flipped=baseline_output.model_response != modified_output.model_response,
        )

    def run_ablation_experiment(
        self,
        tasks: List[ToMTask],
        target_heads: List[Tuple[int, int]],
        baseline_outputs: Optional[List[ModelOutput]] = None,
    ) -> Tuple[List[InterventionResult], InterventionSummary]:
        """
        Run ablation experiment (zero out target heads).

        Args:
            tasks: List of tasks to run
            target_heads: List of (layer, head) tuples to ablate
            baseline_outputs: Optional pre-computed baselines

        Returns:
            Tuple of (results, summary)
        """
        return self._run_experiment(
            tasks, target_heads, scale_factor=0.0,
            baseline_outputs=baseline_outputs, intervention_type="ablation"
        )

    def run_boost_experiment(
        self,
        tasks: List[ToMTask],
        target_heads: List[Tuple[int, int]],
        scale_factor: float = 2.0,
        baseline_outputs: Optional[List[ModelOutput]] = None,
    ) -> Tuple[List[InterventionResult], InterventionSummary]:
        """
        Run boosting experiment (scale up target heads).

        Args:
            tasks: List of tasks to run
            target_heads: List of (layer, head) tuples to boost
            scale_factor: Scale factor (default 2.0 = double)
            baseline_outputs: Optional pre-computed baselines

        Returns:
            Tuple of (results, summary)
        """
        return self._run_experiment(
            tasks, target_heads, scale_factor=scale_factor,
            baseline_outputs=baseline_outputs, intervention_type="boost"
        )

    def _run_experiment(
        self,
        tasks: List[ToMTask],
        target_heads: List[Tuple[int, int]],
        scale_factor: float,
        baseline_outputs: Optional[List[ModelOutput]],
        intervention_type: str,
    ) -> Tuple[List[InterventionResult], InterventionSummary]:
        """Run an intervention experiment."""
        results = []

        # Get baselines if not provided
        if baseline_outputs is None:
            print("Computing baseline outputs...")
            baseline_outputs = []
            for task in tqdm(tasks, desc="Baselines"):
                output = self.model_runner.run_task(task, extract_attention=False)
                baseline_outputs.append(output)

        # Run interventions
        print(f"Running {intervention_type} experiment...")
        for task, baseline in tqdm(
            zip(tasks, baseline_outputs), total=len(tasks), desc=intervention_type.capitalize()
        ):
            result = self.run_intervention(
                task, target_heads, scale_factor, baseline
            )
            results.append(result)

        # Compute summary
        summary = self._compute_summary(results, target_heads, scale_factor, intervention_type)

        return results, summary

    def _compute_summary(
        self,
        results: List[InterventionResult],
        target_heads: List[Tuple[int, int]],
        scale_factor: float,
        intervention_type: str,
    ) -> InterventionSummary:
        """Compute summary statistics from intervention results."""
        num_tasks = len(results)

        original_correct = sum(1 for r in results if r.original_correct)
        modified_correct = sum(1 for r in results if r.modified_correct)
        flipped = sum(1 for r in results if r.flipped)

        original_acc = original_correct / num_tasks if num_tasks > 0 else 0
        modified_acc = modified_correct / num_tasks if num_tasks > 0 else 0

        # By task type
        fb_results = [r for r in results if r.task_type == "false_belief"]
        tb_results = [r for r in results if r.task_type == "true_belief"]

        fb_orig_acc = (sum(1 for r in fb_results if r.original_correct) /
                       len(fb_results) if fb_results else 0)
        fb_mod_acc = (sum(1 for r in fb_results if r.modified_correct) /
                      len(fb_results) if fb_results else 0)
        tb_orig_acc = (sum(1 for r in tb_results if r.original_correct) /
                       len(tb_results) if tb_results else 0)
        tb_mod_acc = (sum(1 for r in tb_results if r.modified_correct) /
                      len(tb_results) if tb_results else 0)

        return InterventionSummary(
            intervention_type=intervention_type,
            target_heads=target_heads,
            scale_factor=scale_factor,
            num_tasks=num_tasks,
            original_accuracy=original_acc,
            modified_accuracy=modified_acc,
            accuracy_delta=modified_acc - original_acc,
            flip_rate=flipped / num_tasks if num_tasks > 0 else 0,
            false_belief_original_acc=fb_orig_acc,
            false_belief_modified_acc=fb_mod_acc,
            true_belief_original_acc=tb_orig_acc,
            true_belief_modified_acc=tb_mod_acc,
        )


def save_intervention_results(
    results: List[InterventionResult],
    summary: InterventionSummary,
    path: Path,
) -> None:
    """Save intervention results to JSON."""
    data = {
        "summary": summary.to_dict(),
        "results": [r.to_dict() for r in results],
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_intervention_results(
    path: Path,
) -> Tuple[List[InterventionResult], InterventionSummary]:
    """Load intervention results from JSON."""
    data = json.loads(Path(path).read_text())
    results = [InterventionResult.from_dict(r) for r in data["results"]]

    summary_data = data["summary"]
    summary_data["target_heads"] = [tuple(t) for t in summary_data["target_heads"]]
    summary = InterventionSummary(**summary_data)

    return results, summary


if __name__ == "__main__":
    print("Intervention module loaded successfully.")
    print("Use InterventionRunner with a ModelRunner to run experiments.")

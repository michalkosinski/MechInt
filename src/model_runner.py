"""
Model runner for HuggingFace models with attention extraction.

Loads models with output_attentions=True and extracts attention weights
for analysis of Theory of Mind processing.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .task_generator import ToMTask


@dataclass
class ModelOutput:
    """Result from running a single task through the model."""
    task_id: str
    task_type: str
    prompt: str
    model_response: str
    expected_answer: str
    is_correct: bool
    # Attention tensors: tuple of (batch, num_heads, seq_len, seq_len) per layer
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    input_ids: Optional[torch.Tensor] = None
    tokens: List[str] = field(default_factory=list)
    logits: Optional[torch.Tensor] = None
    generation_time_ms: float = 0.0

    def has_attentions(self) -> bool:
        return self.attentions is not None


@dataclass
class BatchOutput:
    """Result from running a batch of tasks."""
    outputs: List[ModelOutput]
    total_time_ms: float
    correct_count: int
    accuracy: float


class ModelRunner:
    """
    Runs HuggingFace models with attention extraction.

    Example:
        runner = ModelRunner("Qwen/Qwen2.5-3B-Instruct")
        output = runner.run_task(task, extract_attention=True)
        print(output.model_response)
        print(output.attentions[0].shape)  # (1, num_heads, seq_len, seq_len)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_memory: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the model runner.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ("auto", "cuda", "cpu", "mps")
            torch_dtype: Torch dtype for model weights
            max_memory: Optional memory constraints per device
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"Loading model: {model_name}")
        start = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for batched generation

        # Load model with attention output enabled
        # IMPORTANT: Use attn_implementation="eager" to get attention weights
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",  # Required for attention output
            max_memory=max_memory,
        )
        self.model.eval()

        # Get model config info
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

        load_time = time.time() - start
        print(f"Model loaded in {load_time:.1f}s")
        print(f"  Layers: {self.num_layers}, Heads: {self.num_heads}")
        print(f"  Device: {next(self.model.parameters()).device}")

    def _prepare_prompt(self, task: ToMTask) -> str:
        """Prepare the prompt for a task."""
        # For completion-style prompts (ending with "in the"), use direct completion
        # Don't use chat template as it causes the model to reason instead of complete
        return task.full_prompt

    def run_task(
        self,
        task: ToMTask,
        max_new_tokens: int = 3,
        extract_attention: bool = True,
        temperature: float = 0.0,
    ) -> ModelOutput:
        """
        Run a single task and extract model output with attention.

        Args:
            task: The ToM task to run
            max_new_tokens: Maximum tokens to generate (default 3 for completion format)
            extract_attention: Whether to extract attention weights
            temperature: Generation temperature (0 for greedy)

        Returns:
            ModelOutput with response, correctness, and optionally attention
        """
        start_time = time.time()

        input_text = self._prepare_prompt(task)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        input_length = inputs.input_ids.shape[1]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        # Generate with attention
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=extract_attention,
                return_dict_in_generate=True,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        generated_ids = outputs.sequences[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Check correctness
        is_correct = self._check_answer(response, task.expected_answer)

        # Extract attention if requested
        attentions = None
        if extract_attention and hasattr(outputs, "attentions") and outputs.attentions:
            if outputs.attentions[0] is not None:
                attentions = tuple(
                    layer_attn.cpu() for layer_attn in outputs.attentions[0]
                )

        generation_time_ms = (time.time() - start_time) * 1000

        return ModelOutput(
            task_id=task.task_id,
            task_type=task.task_type,
            prompt=task.full_prompt,
            model_response=response,
            expected_answer=task.expected_answer,
            is_correct=is_correct,
            attentions=attentions,
            input_ids=inputs.input_ids.cpu(),
            tokens=tokens,
            generation_time_ms=generation_time_ms,
        )

    def run_batch(
        self,
        tasks: List[ToMTask],
        batch_size: int = 8,
        max_new_tokens: int = 3,
        extract_attention: bool = False,
        attention_sample_size: int = 20,
        on_progress: Optional[Callable[[int, int, ModelOutput], None]] = None,
    ) -> BatchOutput:
        """
        Run multiple tasks in batches for efficiency.

        Args:
            tasks: List of tasks to run
            batch_size: Number of tasks to process together
            max_new_tokens: Maximum tokens to generate per task (default 3)
            extract_attention: Whether to extract attention (slows down processing)
            attention_sample_size: If extract_attention is True, only extract for this many tasks
            on_progress: Callback for progress updates

        Returns:
            BatchOutput with all results and summary statistics
        """
        start_time = time.time()
        outputs = []

        # Decide which tasks get attention extraction (sample for efficiency)
        attention_task_ids = set()
        if extract_attention and attention_sample_size > 0:
            import random
            sample_tasks = random.sample(tasks, min(attention_sample_size, len(tasks)))
            attention_task_ids = {t.task_id for t in sample_tasks}

        # Process in batches
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Running batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]

            batch_outputs = self._run_batch_internal(
                batch_tasks,
                max_new_tokens=max_new_tokens,
                extract_attention=False,  # Don't extract in batch mode for speed
            )
            outputs.extend(batch_outputs)

            if on_progress:
                on_progress(batch_end, len(tasks), batch_outputs[-1])

        # Now extract attention for sampled tasks (one at a time)
        if attention_task_ids:
            print(f"\nExtracting attention for {len(attention_task_ids)} sampled tasks...")
            task_lookup = {t.task_id: t for t in tasks}
            output_lookup = {o.task_id: o for o in outputs}

            for task_id in tqdm(attention_task_ids, desc="Extracting attention"):
                task = task_lookup[task_id]
                attn_output = self.run_task(task, max_new_tokens=max_new_tokens, extract_attention=True)
                # Update the output with attention
                orig_output = output_lookup[task_id]
                orig_output.attentions = attn_output.attentions
                orig_output.tokens = attn_output.tokens
                orig_output.input_ids = attn_output.input_ids

        total_time_ms = (time.time() - start_time) * 1000
        correct_count = sum(1 for o in outputs if o.is_correct)
        accuracy = correct_count / len(outputs) if outputs else 0.0

        return BatchOutput(
            outputs=outputs,
            total_time_ms=total_time_ms,
            correct_count=correct_count,
            accuracy=accuracy,
        )

    def _run_batch_internal(
        self,
        tasks: List[ToMTask],
        max_new_tokens: int = 3,
        extract_attention: bool = False,
    ) -> List[ModelOutput]:
        """
        Internal method to run a batch of tasks together.

        Uses padding to batch multiple inputs for parallel processing.
        """
        if not tasks:
            return []

        start_time = time.time()

        # Prepare all prompts
        prompts = [self._prepare_prompt(task) for task in tasks]

        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=extract_attention,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Process each output
        results = []
        batch_time = (time.time() - start_time) * 1000

        for i, task in enumerate(tasks):
            # Find where the actual input ends (not padding)
            input_ids = inputs.input_ids[i]
            # Find first non-pad token position
            non_pad_mask = input_ids != self.tokenizer.pad_token_id
            input_length = non_pad_mask.sum().item()

            # Get generated tokens for this sample
            generated_ids = outputs.sequences[i][input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            is_correct = self._check_answer(response, task.expected_answer)

            results.append(ModelOutput(
                task_id=task.task_id,
                task_type=task.task_type,
                prompt=task.full_prompt,
                model_response=response,
                expected_answer=task.expected_answer,
                is_correct=is_correct,
                attentions=None,
                input_ids=None,
                tokens=[],
                generation_time_ms=batch_time / len(tasks),
            ))

        return results

    def _check_answer(self, response: str, expected: str) -> bool:
        """
        Check if the model's response contains the expected answer.

        Uses case-insensitive matching and checks if the expected location
        appears in the response.
        """
        response_lower = response.lower()
        expected_lower = expected.lower()

        # Direct containment check
        if expected_lower in response_lower:
            return True

        # Check individual words of expected answer
        expected_words = expected_lower.split()
        if all(word in response_lower for word in expected_words):
            return True

        return False

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(self.torch_dtype),
        }


def load_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    **kwargs,
) -> ModelRunner:
    """Convenience function to load a model."""
    return ModelRunner(model_name, **kwargs)


if __name__ == "__main__":
    # Quick test
    from .task_generator import generate_tasks

    print("Testing ModelRunner...")

    # Generate test tasks
    tasks = generate_tasks(num_false_belief=4, num_true_belief=4)

    # Load model (use small model for testing)
    runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct")

    # Run tasks in batch
    results = runner.run_batch(tasks, batch_size=4, extract_attention=True, attention_sample_size=2)

    print(f"\nResults:")
    print(f"  Accuracy: {results.accuracy:.1%}")
    print(f"  Correct: {results.correct_count}/{len(tasks)}")
    print(f"  Total time: {results.total_time_ms:.0f}ms")

    # Show individual results
    for output in results.outputs:
        status = "✓" if output.is_correct else "✗"
        has_attn = "📊" if output.has_attentions() else ""
        print(f"{status} {has_attn} [{output.task_type}] {output.task_id}: {output.model_response}")

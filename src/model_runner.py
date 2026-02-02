"""
Model runner for HuggingFace models with attention extraction.

Loads models with output_attentions=True and extracts attention weights
for analysis of Theory of Mind processing.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


class HasFullPrompt(Protocol):
    """Protocol for objects with full_prompt attribute."""
    task_id: str
    task_type: str
    full_prompt: str

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .task_generator import ToMTask

# Large models that need CPU offloading on MPS
LARGE_MODELS_MAX_MEMORY = {
    "Qwen/Qwen2.5-32B": {"mps": "50GB", "cpu": "40GB"},
}

# Prefix to disable reasoning/thinking mode (always applied)
NO_REASONING_PREFIX = "/no_think "


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SpeedMetrics:
    """Detailed speed metrics for model execution."""
    model_load_time_ms: float = 0.0
    tokenization_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    decoding_time_ms: float = 0.0
    total_time_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelConfig:
    """Persisted configuration for a model."""
    model_name: str
    optimal_batch_size: int
    num_layers: int
    num_heads: int
    hidden_size: int
    device: str
    dtype: str
    last_optimized: str
    optimization_device: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)


@dataclass
class ModelOutput:
    """Result from running a single task through the model."""
    task_id: str
    task_type: str
    prompt: str
    model_response: str
    expected_answer: Optional[str]
    is_correct: bool
    # Full text fields
    full_input_text: str
    full_output_text: str
    raw_generated_text: str
    # Speed metrics
    speed_metrics: SpeedMetrics
    input_token_count: int
    output_token_count: int
    # Attention data
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    input_ids: Optional[torch.Tensor] = None
    tokens: List[str] = field(default_factory=list)

    def has_attentions(self) -> bool:
        return self.attentions is not None


@dataclass
class BatchOutput:
    """Result from running a batch of tasks."""
    outputs: List[ModelOutput]
    total_time_ms: float
    correct_count: int
    accuracy: float
    speed_metrics: SpeedMetrics
    batch_size_used: int
    total_input_tokens: int
    total_output_tokens: int
    tokens_per_second: float


# =============================================================================
# models.json Helpers
# =============================================================================

def load_models_config(path: Path = None) -> Dict[str, ModelConfig]:
    """Load model configurations from models.json."""
    if path is None:
        path = Path(__file__).parent.parent / "models.json"

    if not path.exists():
        return {}

    data = json.loads(path.read_text())
    return {
        name: ModelConfig.from_dict(config)
        for name, config in data.get("models", {}).items()
    }


def save_models_config(configs: Dict[str, ModelConfig], path: Path = None) -> None:
    """Save model configurations to models.json."""
    if path is None:
        path = Path(__file__).parent.parent / "models.json"

    data = {
        "version": "1.0",
        "models": {name: config.to_dict() for name, config in configs.items()}
    }
    path.write_text(json.dumps(data, indent=2))


# =============================================================================
# Model Runner
# =============================================================================

class ModelRunner:
    """
    Runs HuggingFace models with attention extraction.

    Example:
        runner = ModelRunner("mistralai/Mistral-7B-v0.3")
        output = runner.run_task(task, max_new_tokens=3, stop_strings=["."], extract_attention=True)
        print(output.model_response)
        print(output.attentions[0].shape)  # (1, num_heads, seq_len, seq_len)
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.3",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_memory: Optional[Dict[str, str]] = None,
        auto_optimize_batch_size: bool = True,
        models_config_path: Optional[Path] = None,
    ):
        """
        Initialize the model runner.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ("auto", "cuda", "cpu", "mps")
            torch_dtype: Torch dtype for model weights
            max_memory: Optional memory constraints per device
            auto_optimize_batch_size: Whether to auto-optimize batch size
            models_config_path: Path to models.json (default: project root)
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self._models_config_path = models_config_path or Path(__file__).parent.parent / "models.json"

        start = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Auto-configure max_memory for large models on MPS
        if max_memory is None and model_name in LARGE_MODELS_MAX_MEMORY:
            max_memory = LARGE_MODELS_MAX_MEMORY[model_name]
            print(f"Using CPU offloading for large model: {max_memory}")

        # Load model with attention output enabled
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
            max_memory=max_memory,
        )
        self.model.eval()

        # Model config info
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size

        # Timing and device
        self.load_time_ms = (time.time() - start) * 1000
        self.actual_device = str(next(self.model.parameters()).device)

        # Batch size optimization
        self.optimal_batch_size: Optional[int] = None
        self._auto_optimize = auto_optimize_batch_size
        if auto_optimize_batch_size:
            self._load_cached_batch_size()

    def _load_cached_batch_size(self) -> None:
        """Load cached batch size from models.json if available."""
        configs = load_models_config(self._models_config_path)

        if self.model_name in configs:
            config = configs[self.model_name]
            if config.optimization_device == self.actual_device:
                self.optimal_batch_size = config.optimal_batch_size

    def _try_batch_size(self, tasks: List[ToMTask], batch_size: int) -> Tuple[bool, float]:
        """
        Try running with a specific batch size.

        Returns:
            Tuple of (success, time_per_task_ms)
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            start = time.time()
            batch = tasks[:batch_size]
            self._run_batch_internal(batch, max_new_tokens=3, stop_strings=["."], extract_attention=False)
            elapsed = (time.time() - start) * 1000
            return True, elapsed / batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False, float('inf')
            raise

    def find_optimal_batch_size(
        self,
        sample_tasks: List[ToMTask],
        min_batch: int = 1,
        max_batch: int = 64,
    ) -> int:
        """
        Find optimal batch size using binary search with OOM detection.

        Args:
            sample_tasks: Tasks to use for benchmarking
            min_batch: Minimum batch size to try
            max_batch: Maximum batch size to try

        Returns:
            Optimal batch size
        """
        if len(sample_tasks) < max_batch:
            max_batch = len(sample_tasks)

        optimal = min_batch
        low, high = min_batch, max_batch

        while low <= high:
            mid = (low + high) // 2
            success, _ = self._try_batch_size(sample_tasks, mid)

            if success:
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1

        return optimal

    def _optimize_and_save_batch_size(self, sample_tasks: List[ToMTask]) -> int:
        """Run batch size optimization and save results."""
        optimal = self.find_optimal_batch_size(sample_tasks)

        configs = load_models_config(self._models_config_path)
        configs[self.model_name] = ModelConfig(
            model_name=self.model_name,
            optimal_batch_size=optimal,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            device=self.actual_device,
            dtype=str(self.torch_dtype),
            last_optimized=datetime.now().isoformat(),
            optimization_device=self.actual_device,
        )
        save_models_config(configs, self._models_config_path)

        self.optimal_batch_size = optimal
        return optimal

    def run_task(
        self,
        task: ToMTask,
        max_new_tokens: int,
        stop_strings: List[str],
        extract_attention: bool = True,
        temperature: float = 0.0,
    ) -> ModelOutput:
        """
        Run a single task and extract model output with attention.

        Args:
            task: The ToM task to run
            max_new_tokens: Maximum tokens to generate (required)
            stop_strings: Strings that stop generation (required)
            extract_attention: Whether to extract attention weights
            temperature: Generation temperature (0 for greedy)

        Returns:
            ModelOutput with response, correctness, timing, and optionally attention
        """
        total_start = time.time()

        # Tokenization
        tokenize_start = time.time()
        input_text = NO_REASONING_PREFIX + task.full_prompt
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        tokenize_time_ms = (time.time() - tokenize_start) * 1000

        # Generation
        gen_start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=extract_attention,
                return_dict_in_generate=True,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
            )
        gen_time_ms = (time.time() - gen_start) * 1000

        # Decoding
        decode_start = time.time()
        full_output_ids = outputs.sequences[0]
        generated_ids = full_output_ids[input_length:]

        full_output_text = self.tokenizer.decode(full_output_ids, skip_special_tokens=True)
        raw_generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = raw_generated_text.strip()
        decode_time_ms = (time.time() - decode_start) * 1000

        total_time_ms = (time.time() - total_start) * 1000

        # Attention extraction
        attentions = None
        if extract_attention and hasattr(outputs, "attentions") and outputs.attentions:
            if outputs.attentions[0] is not None:
                attentions = tuple(
                    layer_attn.cpu() for layer_attn in outputs.attentions[0]
                )

        # Speed metrics
        output_token_count = len(generated_ids)
        speed_metrics = SpeedMetrics(
            model_load_time_ms=0.0,
            tokenization_time_ms=tokenize_time_ms,
            generation_time_ms=gen_time_ms,
            decoding_time_ms=decode_time_ms,
            total_time_ms=total_time_ms,
            input_tokens=input_length,
            output_tokens=output_token_count,
            tokens_per_second=(input_length + output_token_count) / (total_time_ms / 1000) if total_time_ms > 0 else 0,
        )

        return ModelOutput(
            task_id=task.task_id,
            task_type=task.task_type,
            prompt=task.full_prompt,
            model_response=response,
            expected_answer=getattr(task, 'expected_answer', None),
            is_correct=False,  # Scoring done by behavioral_analysis
            full_input_text=input_text,
            full_output_text=full_output_text,
            raw_generated_text=raw_generated_text,
            speed_metrics=speed_metrics,
            input_token_count=input_length,
            output_token_count=output_token_count,
            attentions=attentions,
            input_ids=inputs.input_ids.cpu(),
            tokens=tokens,
        )

    def run_batch(
        self,
        tasks: List[ToMTask],
        max_new_tokens: int,
        stop_strings: List[str],
        batch_size: Optional[int] = None,
        extract_attention: bool = False,
        attention_sample_size: int = 20,
        on_progress: Optional[Callable[[int, int, ModelOutput], None]] = None,
        temperature: float = 0.0,
    ) -> BatchOutput:
        """
        Run multiple tasks in batches for efficiency.

        Args:
            tasks: List of tasks to run
            max_new_tokens: Maximum tokens to generate per task (required)
            stop_strings: Stop generation when these strings are produced (required)
            batch_size: Batch size (None = use optimal)
            extract_attention: Whether to extract attention
            attention_sample_size: Number of tasks to extract attention for
            on_progress: Callback for progress updates
            temperature: Generation temperature (0 for greedy/deterministic)

        Returns:
            BatchOutput with all results and summary statistics
        """
        start_time = time.time()

        # Determine batch size
        if batch_size is None:
            if self.optimal_batch_size is None and self._auto_optimize:
                self._optimize_and_save_batch_size(tasks)
            batch_size = self.optimal_batch_size or 8

        outputs = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Attention sampling
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
                extract_attention=False,
                stop_strings=stop_strings,
                temperature=temperature,
            )

            for output in batch_outputs:
                total_input_tokens += output.input_token_count
                total_output_tokens += output.output_token_count

            outputs.extend(batch_outputs)

            if on_progress:
                on_progress(batch_end, len(tasks), batch_outputs[-1])

        # Extract attention for sampled tasks
        if attention_task_ids:
            task_lookup = {t.task_id: t for t in tasks}
            output_lookup = {o.task_id: o for o in outputs}

            for task_id in tqdm(attention_task_ids, desc="Extracting attention"):
                task = task_lookup[task_id]
                attn_output = self.run_task(task, max_new_tokens=max_new_tokens, stop_strings=stop_strings, extract_attention=True)
                orig_output = output_lookup[task_id]
                orig_output.attentions = attn_output.attentions
                orig_output.tokens = attn_output.tokens
                orig_output.input_ids = attn_output.input_ids

        total_time_ms = (time.time() - start_time) * 1000
        correct_count = sum(1 for o in outputs if o.is_correct)
        accuracy = correct_count / len(outputs) if outputs else 0.0

        speed_metrics = SpeedMetrics(
            model_load_time_ms=self.load_time_ms,
            total_time_ms=total_time_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            tokens_per_second=(total_input_tokens + total_output_tokens) / (total_time_ms / 1000) if total_time_ms > 0 else 0,
        )

        return BatchOutput(
            outputs=outputs,
            total_time_ms=total_time_ms,
            correct_count=correct_count,
            accuracy=accuracy,
            speed_metrics=speed_metrics,
            batch_size_used=batch_size,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            tokens_per_second=speed_metrics.tokens_per_second,
        )

    def iter_hidden_states(
        self,
        tasks: List[HasFullPrompt],
        batch_size: Optional[int] = None,
    ):
        """
        Yield hidden states for each task without generation.

        Yields:
            Tuple of (task, input_ids, tokens, hidden_states, input_length)
        """
        if not tasks:
            return

        if batch_size is None:
            if self.optimal_batch_size is None and self._auto_optimize:
                self._optimize_and_save_batch_size(tasks)
            batch_size = self.optimal_batch_size or 8

        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Extracting hidden states"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]

            prompts = [task.full_prompt for task in batch_tasks]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states

            for i, task in enumerate(batch_tasks):
                input_ids = inputs.input_ids[i]
                non_pad_mask = input_ids != self.tokenizer.pad_token_id
                input_length = int(non_pad_mask.sum().item())
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

                per_task_hidden = tuple(layer[i].cpu() for layer in hidden_states)

                yield task, input_ids.cpu(), tokens, per_task_hidden, input_length

    def _run_batch_internal(
        self,
        tasks: List[ToMTask],
        max_new_tokens: int,
        stop_strings: List[str],
        extract_attention: bool = False,
        temperature: float = 0.0,
    ) -> List[ModelOutput]:
        """
        Internal method to run a batch of tasks together.

        Uses padding to batch multiple inputs for parallel processing.
        """
        if not tasks:
            return []

        start_time = time.time()

        # Tokenization
        tokenize_start = time.time()
        prompts = [NO_REASONING_PREFIX + task.full_prompt for task in tasks]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        tokenize_time_ms = (time.time() - tokenize_start) * 1000

        # Generation
        gen_start = time.time()
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "output_attentions": extract_attention,
                "return_dict_in_generate": True,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            if stop_strings:
                generate_kwargs["stop_strings"] = stop_strings
                generate_kwargs["tokenizer"] = self.tokenizer
            outputs = self.model.generate(**generate_kwargs)
        gen_time_ms = (time.time() - gen_start) * 1000

        # Process each output
        results = []
        batch_time_ms = (time.time() - start_time) * 1000
        per_task_time_ms = batch_time_ms / len(tasks)

        for i, task in enumerate(tasks):
            # Get actual input length (excluding padding)
            input_ids = inputs.input_ids[i]
            non_pad_mask = input_ids != self.tokenizer.pad_token_id
            input_length = non_pad_mask.sum().item()

            # Get generated tokens
            full_output_ids = outputs.sequences[i]
            generated_ids = full_output_ids[input_length:]

            # Decode
            full_output_text = self.tokenizer.decode(full_output_ids, skip_special_tokens=True)
            raw_generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            response = raw_generated_text.strip()

            output_token_count = len(generated_ids)

            speed_metrics = SpeedMetrics(
                tokenization_time_ms=tokenize_time_ms / len(tasks),
                generation_time_ms=gen_time_ms / len(tasks),
                total_time_ms=per_task_time_ms,
                input_tokens=input_length,
                output_tokens=output_token_count,
                tokens_per_second=(input_length + output_token_count) / (per_task_time_ms / 1000) if per_task_time_ms > 0 else 0,
            )

            results.append(ModelOutput(
                task_id=task.task_id,
                task_type=task.task_type,
                prompt=task.full_prompt,
                model_response=response,
                expected_answer=getattr(task, 'expected_answer', None),
                is_correct=False,  # Scoring done by behavioral_analysis
                full_input_text=task.full_prompt,
                full_output_text=full_output_text,
                raw_generated_text=raw_generated_text,
                speed_metrics=speed_metrics,
                input_token_count=input_length,
                output_token_count=output_token_count,
                attentions=None,
                input_ids=None,
                tokens=[],
            ))

        return results

    def get_model_info(self) -> Dict:
        """Get comprehensive information about the loaded model."""
        return {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "device": self.actual_device,
            "dtype": str(self.torch_dtype),
            "load_time_ms": self.load_time_ms,
            "optimal_batch_size": self.optimal_batch_size,
        }

    def get_candidate_probabilities(
        self,
        prompt: str,
        candidates: List[str],
    ) -> Dict[str, float]:
        """
        Get probability of each candidate being the next token(s).

        Performs a forward pass (no generation) and computes softmax
        probabilities for each candidate's first token.

        Args:
            prompt: The input text ending where completion is expected
            candidates: List of possible completions (e.g., ["box", "basket"])

        Returns:
            Dict mapping each candidate to its probability
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids)
            logits = outputs.logits[:, -1, :]  # Last position logits
            probs = torch.softmax(logits, dim=-1)

        result = {}
        for candidate in candidates:
            # Encode candidate and get first token ID
            token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if token_ids:
                # Use first token probability
                result[candidate] = probs[0, token_ids[0]].item()
            else:
                result[candidate] = 0.0

        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test model runner")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.3", help="Model name")
    parser.add_argument("--prompt", default="The capital of France is", help="Test prompt")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    runner = ModelRunner(args.model)
    info = runner.get_model_info()
    print(f"Loaded in {info['load_time_ms']:.0f}ms | {info['num_layers']} layers | {info['device']}")

    # Simple completion test
    from collections import namedtuple
    TestTask = namedtuple("TestTask", ["task_id", "task_type", "full_prompt"])
    task = TestTask("test", "test", args.prompt)
    output = runner.run_task(task, max_new_tokens=20, stop_strings=[".", "\n"], extract_attention=False)
    print(f"Prompt: {args.prompt}")
    print(f"Response: {output.model_response}")


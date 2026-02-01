"""
Attention pattern analyzer for Theory of Mind studies.

Identifies attention heads that track protagonist beliefs vs reality
by analyzing attention patterns in false-belief vs true-belief tasks.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .task_generator import ToMTask
from .model_runner import ModelOutput


@dataclass
class TokenRegion:
    """A region of tokens with semantic meaning."""
    name: str           # "protagonist", "belief_location", "reality_location", "question"
    start_idx: int
    end_idx: int        # exclusive

    @property
    def span(self) -> range:
        return range(self.start_idx, self.end_idx)

    def __len__(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class AttentionPattern:
    """Attention statistics for a single task, layer, and head."""
    task_id: str
    task_type: str      # "false_belief" or "true_belief"
    layer: int
    head: int
    # Attention metrics (averaged over query positions in question region)
    attention_to_protagonist: float
    attention_to_belief_location: float
    attention_to_reality_location: float
    attention_to_question: float
    # Derived metrics
    belief_vs_reality_ratio: float  # belief_location / reality_location

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeadScore:
    """Score for a single attention head based on ToM analysis."""
    layer: int
    head: int
    # Average belief-reality ratio in false belief condition
    false_belief_ratio: float
    # Average belief-reality ratio in true belief condition
    true_belief_ratio: float
    # Difference between conditions
    condition_diff: float
    # Combined ToM score
    tom_score: float
    # Number of tasks analyzed
    num_tasks: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayerSelectivity:
    """Selectivity metrics aggregated by layer."""
    layer: int
    task_type: str
    mean_belief_ratio: float
    mean_attention_to_belief: float
    mean_attention_to_reality: float
    num_heads: int
    num_tasks: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenAttentionData:
    """Token-level attention weights for visualization."""
    task_id: str
    task_type: str
    layer: int
    head: int
    tokens: List[str]
    # Attention from last query token to all keys
    attention_weights: List[float]
    # Region boundaries
    regions: Dict[str, Tuple[int, int]]  # name -> (start, end)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "layer": self.layer,
            "head": self.head,
            "tokens": self.tokens,
            "attention_weights": self.attention_weights,
            "regions": self.regions,
        }


@dataclass
class TaskAttentionMatrix:
    """Full attention matrix for a specific task and head."""
    task_id: str
    task_type: str
    layer: int
    head: int
    tokens: List[str]
    # Full attention matrix (seq_len x seq_len)
    attention_matrix: List[List[float]]
    regions: Dict[str, Tuple[int, int]]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "layer": self.layer,
            "head": self.head,
            "tokens": self.tokens,
            "attention_matrix": self.attention_matrix,
            "regions": self.regions,
        }


class AttentionAnalyzer:
    """
    Analyzes attention patterns to identify ToM-relevant heads.

    A head is considered ToM-relevant if:
    1. In false belief tasks: attends more to belief_location than reality_location
    2. Shows significant difference between false and true belief conditions
    """

    def __init__(
        self,
        belief_ratio_threshold: float = 1.5,
        condition_diff_threshold: float = 0.3,
    ):
        """
        Initialize the analyzer.

        Args:
            belief_ratio_threshold: Minimum belief/reality ratio for ToM heads
            condition_diff_threshold: Minimum difference between conditions
        """
        self.belief_ratio_threshold = belief_ratio_threshold
        self.condition_diff_threshold = condition_diff_threshold

    def identify_token_regions(
        self,
        tokens: List[str],
        task: ToMTask,
    ) -> Dict[str, TokenRegion]:
        """
        Map token positions to semantic regions in the task.

        Identifies regions for:
        - protagonist: tokens containing the protagonist name
        - belief_location: tokens containing the believed location
        - reality_location: tokens containing the actual location
        - question: tokens in the question part

        Args:
            tokens: List of token strings from the tokenizer
            task: The ToM task being analyzed

        Returns:
            Dictionary mapping region names to TokenRegion objects
        """
        tokens_lower = [t.lower().replace("▁", "").replace("Ġ", "") for t in tokens]
        joined_tokens = " ".join(tokens_lower)

        regions = {}

        # Find protagonist region
        protagonist_lower = task.protagonist.lower()
        regions["protagonist"] = self._find_region(
            tokens_lower, protagonist_lower, "protagonist"
        )

        # Find belief location region
        belief_parts = task.belief.lower().split()
        regions["belief_location"] = self._find_region(
            tokens_lower, belief_parts, "belief_location"
        )

        # Find reality location region (may overlap with belief in true-belief tasks)
        reality_parts = task.world.lower().split()
        regions["reality_location"] = self._find_region(
            tokens_lower, reality_parts, "reality_location"
        )

        # Find question region (starts with "where")
        question_start = None
        for i, token in enumerate(tokens_lower):
            if "where" in token:
                question_start = i
                break

        if question_start is not None:
            regions["question"] = TokenRegion(
                name="question",
                start_idx=question_start,
                end_idx=len(tokens),
            )
        else:
            # Fallback: last 20% of tokens
            regions["question"] = TokenRegion(
                name="question",
                start_idx=int(len(tokens) * 0.8),
                end_idx=len(tokens),
            )

        return regions

    def _find_region(
        self,
        tokens_lower: List[str],
        target: str | List[str],
        name: str,
    ) -> TokenRegion:
        """Find a region containing the target string(s)."""
        if isinstance(target, str):
            target = [target]

        # Find first occurrence of target words
        start_idx = None
        end_idx = None

        for target_word in target:
            for i, token in enumerate(tokens_lower):
                if target_word in token:
                    if start_idx is None or i < start_idx:
                        start_idx = i
                    if end_idx is None or i + 1 > end_idx:
                        end_idx = i + 1
                    break

        if start_idx is None:
            # Fallback: return middle of sequence
            mid = len(tokens_lower) // 2
            return TokenRegion(name=name, start_idx=mid, end_idx=mid + 1)

        return TokenRegion(name=name, start_idx=start_idx, end_idx=end_idx)

    def extract_attention_patterns(
        self,
        output: ModelOutput,
        task: ToMTask,
    ) -> List[AttentionPattern]:
        """
        Extract attention patterns for all layers and heads.

        Args:
            output: Model output with attention weights
            task: The task that was run

        Returns:
            List of AttentionPattern objects for each (layer, head)
        """
        if not output.has_attentions():
            return []

        # Identify token regions
        regions = self.identify_token_regions(output.tokens, task)
        patterns = []

        # Process each layer
        for layer_idx, layer_attn in enumerate(output.attentions):
            # layer_attn shape: (batch, num_heads, seq_len, seq_len)
            # We take batch=0
            attn = layer_attn[0].float().numpy()  # (num_heads, seq_len, seq_len) - convert bfloat16 to float32
            num_heads = attn.shape[0]

            for head_idx in range(num_heads):
                head_attn = attn[head_idx]  # (seq_len, seq_len)

                # Get attention FROM question tokens TO each region
                question_span = regions["question"].span
                if len(question_span) == 0:
                    continue

                # Average attention from question tokens to each region
                attn_to_protagonist = self._mean_attention_to_region(
                    head_attn, question_span, regions["protagonist"].span
                )
                attn_to_belief = self._mean_attention_to_region(
                    head_attn, question_span, regions["belief_location"].span
                )
                attn_to_reality = self._mean_attention_to_region(
                    head_attn, question_span, regions["reality_location"].span
                )
                attn_to_question = self._mean_attention_to_region(
                    head_attn, question_span, question_span
                )

                # Compute ratio (avoid division by zero)
                ratio = attn_to_belief / max(attn_to_reality, 1e-8)

                pattern = AttentionPattern(
                    task_id=output.task_id,
                    task_type=output.task_type,
                    layer=layer_idx,
                    head=head_idx,
                    attention_to_protagonist=float(attn_to_protagonist),
                    attention_to_belief_location=float(attn_to_belief),
                    attention_to_reality_location=float(attn_to_reality),
                    attention_to_question=float(attn_to_question),
                    belief_vs_reality_ratio=float(ratio),
                )
                patterns.append(pattern)

        return patterns

    def _mean_attention_to_region(
        self,
        attn_matrix: np.ndarray,
        from_span: range,
        to_span: range,
    ) -> float:
        """Compute mean attention from one region to another."""
        if len(from_span) == 0 or len(to_span) == 0:
            return 0.0

        from_indices = list(from_span)
        to_indices = list(to_span)

        # Extract submatrix and compute mean
        submatrix = attn_matrix[np.ix_(from_indices, to_indices)]
        return float(np.mean(submatrix))

    def analyze_by_head(
        self,
        patterns: List[AttentionPattern],
    ) -> pd.DataFrame:
        """
        Aggregate attention patterns by (layer, head) across tasks.

        Args:
            patterns: List of patterns from multiple tasks

        Returns:
            DataFrame with aggregated statistics per head
        """
        if not patterns:
            return pd.DataFrame()

        df = pd.DataFrame([p.to_dict() for p in patterns])

        # Aggregate by layer, head, task_type
        agg = df.groupby(["layer", "head", "task_type"]).agg({
            "attention_to_protagonist": "mean",
            "attention_to_belief_location": "mean",
            "attention_to_reality_location": "mean",
            "belief_vs_reality_ratio": "mean",
            "task_id": "count",
        }).rename(columns={"task_id": "num_tasks"})

        return agg.reset_index()

    def identify_tom_heads(
        self,
        patterns: List[AttentionPattern],
    ) -> List[HeadScore]:
        """
        Identify attention heads that track protagonist beliefs.

        A head is ToM-relevant if:
        1. belief_vs_reality_ratio > threshold in false belief condition
        2. Significant difference between false and true belief conditions

        Args:
            patterns: List of patterns from multiple tasks

        Returns:
            List of HeadScore objects sorted by ToM score (descending)
        """
        if not patterns:
            return []

        # Separate by task type
        fb_patterns = [p for p in patterns if p.task_type == "false_belief"]
        tb_patterns = [p for p in patterns if p.task_type == "true_belief"]

        # Aggregate by head
        fb_by_head: Dict[Tuple[int, int], List[float]] = {}
        tb_by_head: Dict[Tuple[int, int], List[float]] = {}

        for p in fb_patterns:
            key = (p.layer, p.head)
            if key not in fb_by_head:
                fb_by_head[key] = []
            fb_by_head[key].append(p.belief_vs_reality_ratio)

        for p in tb_patterns:
            key = (p.layer, p.head)
            if key not in tb_by_head:
                tb_by_head[key] = []
            tb_by_head[key].append(p.belief_vs_reality_ratio)

        # Score each head
        scores = []
        all_heads = set(fb_by_head.keys()) | set(tb_by_head.keys())

        for layer, head in all_heads:
            fb_ratios = fb_by_head.get((layer, head), [1.0])
            tb_ratios = tb_by_head.get((layer, head), [1.0])

            fb_mean = np.mean(fb_ratios)
            tb_mean = np.mean(tb_ratios)
            diff = fb_mean - tb_mean

            # ToM score: high false-belief ratio * large condition difference
            tom_score = max(0, fb_mean - 1) * max(0, diff)

            scores.append(HeadScore(
                layer=layer,
                head=head,
                false_belief_ratio=float(fb_mean),
                true_belief_ratio=float(tb_mean),
                condition_diff=float(diff),
                tom_score=float(tom_score),
                num_tasks=len(fb_ratios) + len(tb_ratios),
            ))

        # Sort by ToM score descending
        scores.sort(key=lambda x: -x.tom_score)

        return scores

    def get_top_tom_heads(
        self,
        patterns: List[AttentionPattern],
        top_k: int = 10,
    ) -> List[Tuple[int, int]]:
        """
        Get the top K ToM-relevant heads.

        Args:
            patterns: List of attention patterns
            top_k: Number of heads to return

        Returns:
            List of (layer, head) tuples
        """
        scores = self.identify_tom_heads(patterns)

        # Filter by thresholds
        filtered = [
            s for s in scores
            if s.false_belief_ratio > self.belief_ratio_threshold
            and s.condition_diff > self.condition_diff_threshold
        ]

        return [(s.layer, s.head) for s in filtered[:top_k]]

    def create_heatmap_data(
        self,
        patterns: List[AttentionPattern],
        num_layers: int,
        num_heads: int,
        metric: str = "belief_vs_reality_ratio",
        task_type: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create a heatmap matrix for visualization.

        Args:
            patterns: List of attention patterns
            num_layers: Number of layers in the model
            num_heads: Number of attention heads per layer
            metric: Which metric to use for the heatmap
            task_type: Filter by task type (None for all)

        Returns:
            numpy array of shape (num_layers, num_heads)
        """
        # Filter by task type if specified
        if task_type:
            patterns = [p for p in patterns if p.task_type == task_type]

        # Initialize with zeros
        heatmap = np.zeros((num_layers, num_heads))
        counts = np.zeros((num_layers, num_heads))

        # Aggregate
        for p in patterns:
            value = getattr(p, metric, 0)
            heatmap[p.layer, p.head] += value
            counts[p.layer, p.head] += 1

        # Average
        counts[counts == 0] = 1  # Avoid division by zero
        heatmap = heatmap / counts

        return heatmap

    def compute_layer_selectivity(
        self,
        patterns: List[AttentionPattern],
        num_layers: int,
    ) -> List[LayerSelectivity]:
        """
        Compute selectivity metrics aggregated by layer.

        Args:
            patterns: List of attention patterns
            num_layers: Number of layers in the model

        Returns:
            List of LayerSelectivity objects
        """
        results = []

        for task_type in ["false_belief", "true_belief"]:
            type_patterns = [p for p in patterns if p.task_type == task_type]

            for layer in range(num_layers):
                layer_patterns = [p for p in type_patterns if p.layer == layer]
                if not layer_patterns:
                    continue

                mean_ratio = np.mean([p.belief_vs_reality_ratio for p in layer_patterns])
                mean_belief = np.mean([p.attention_to_belief_location for p in layer_patterns])
                mean_reality = np.mean([p.attention_to_reality_location for p in layer_patterns])

                # Count unique heads
                unique_heads = len(set((p.head for p in layer_patterns)))
                unique_tasks = len(set((p.task_id for p in layer_patterns)))

                results.append(LayerSelectivity(
                    layer=layer,
                    task_type=task_type,
                    mean_belief_ratio=float(mean_ratio),
                    mean_attention_to_belief=float(mean_belief),
                    mean_attention_to_reality=float(mean_reality),
                    num_heads=unique_heads,
                    num_tasks=unique_tasks,
                ))

        return results

    def extract_token_attention(
        self,
        output: ModelOutput,
        task: ToMTask,
        target_heads: List[Tuple[int, int]],
    ) -> List[TokenAttentionData]:
        """
        Extract token-level attention weights for specified heads.

        Args:
            output: Model output with attention weights
            task: The task that was run
            target_heads: List of (layer, head) tuples to extract

        Returns:
            List of TokenAttentionData objects
        """
        if not output.has_attentions():
            return []

        regions = self.identify_token_regions(output.tokens, task)
        region_dict = {
            name: (r.start_idx, r.end_idx)
            for name, r in regions.items()
        }

        results = []
        for layer, head in target_heads:
            if layer >= len(output.attentions):
                continue

            layer_attn = output.attentions[layer]
            attn = layer_attn[0].float().numpy()  # (num_heads, seq_len, seq_len)

            if head >= attn.shape[0]:
                continue

            head_attn = attn[head]  # (seq_len, seq_len)

            # Get attention from last token to all other tokens
            last_token_attn = head_attn[-1, :].tolist()

            results.append(TokenAttentionData(
                task_id=output.task_id,
                task_type=output.task_type,
                layer=layer,
                head=head,
                tokens=output.tokens,
                attention_weights=last_token_attn,
                regions=region_dict,
            ))

        return results

    def extract_attention_matrix(
        self,
        output: ModelOutput,
        task: ToMTask,
        layer: int,
        head: int,
    ) -> Optional[TaskAttentionMatrix]:
        """
        Extract full attention matrix for a specific task and head.

        Args:
            output: Model output with attention weights
            task: The task that was run
            layer: Layer index
            head: Head index

        Returns:
            TaskAttentionMatrix object or None
        """
        if not output.has_attentions():
            return None

        if layer >= len(output.attentions):
            return None

        regions = self.identify_token_regions(output.tokens, task)
        region_dict = {
            name: (r.start_idx, r.end_idx)
            for name, r in regions.items()
        }

        layer_attn = output.attentions[layer]
        attn = layer_attn[0].float().numpy()  # (num_heads, seq_len, seq_len)

        if head >= attn.shape[0]:
            return None

        head_attn = attn[head]  # (seq_len, seq_len)

        return TaskAttentionMatrix(
            task_id=output.task_id,
            task_type=output.task_type,
            layer=layer,
            head=head,
            tokens=output.tokens,
            attention_matrix=head_attn.tolist(),
            regions=region_dict,
        )


def save_patterns(patterns: List[AttentionPattern], path: Path) -> None:
    """Save attention patterns to JSON."""
    data = [p.to_dict() for p in patterns]
    Path(path).write_text(json.dumps(data, indent=2))


def load_patterns(path: Path) -> List[AttentionPattern]:
    """Load attention patterns from JSON."""
    data = json.loads(Path(path).read_text())
    return [AttentionPattern(**p) for p in data]


def save_head_scores(scores: List[HeadScore], path: Path) -> None:
    """Save head scores to JSON."""
    data = [s.to_dict() for s in scores]
    Path(path).write_text(json.dumps(data, indent=2))


def load_head_scores(path: Path) -> List[HeadScore]:
    """Load head scores from JSON."""
    data = json.loads(Path(path).read_text())
    return [HeadScore(**s) for s in data]


def save_layer_selectivity(selectivity: List[LayerSelectivity], path: Path) -> None:
    """Save layer selectivity data to JSON."""
    data = [s.to_dict() for s in selectivity]
    Path(path).write_text(json.dumps(data, indent=2))


def load_layer_selectivity(path: Path) -> List[LayerSelectivity]:
    """Load layer selectivity data from JSON."""
    data = json.loads(Path(path).read_text())
    return [LayerSelectivity(**s) for s in data]


def save_token_attention(token_data: List[TokenAttentionData], path: Path) -> None:
    """Save token attention data to JSON."""
    data = [t.to_dict() for t in token_data]
    Path(path).write_text(json.dumps(data, indent=2))


def load_token_attention(path: Path) -> List[TokenAttentionData]:
    """Load token attention data from JSON."""
    data = json.loads(Path(path).read_text())
    return [TokenAttentionData(**d) for d in data]


def save_attention_matrices(matrices: List[TaskAttentionMatrix], path: Path) -> None:
    """Save attention matrices to JSON."""
    data = [m.to_dict() for m in matrices]
    Path(path).write_text(json.dumps(data, indent=2))


def load_attention_matrices(path: Path) -> List[TaskAttentionMatrix]:
    """Load attention matrices from JSON."""
    data = json.loads(Path(path).read_text())
    return [TaskAttentionMatrix(**m) for m in data]


if __name__ == "__main__":
    # Test with mock data
    print("Testing AttentionAnalyzer...")

    analyzer = AttentionAnalyzer()

    # Create mock patterns
    mock_patterns = []
    for task_type in ["false_belief", "true_belief"]:
        for layer in range(3):
            for head in range(4):
                # In false belief, some heads should have high belief ratio
                if task_type == "false_belief" and layer == 2 and head == 1:
                    ratio = 2.5
                else:
                    ratio = 0.9 + np.random.random() * 0.3

                mock_patterns.append(AttentionPattern(
                    task_id=f"test_{task_type}_{layer}_{head}",
                    task_type=task_type,
                    layer=layer,
                    head=head,
                    attention_to_protagonist=0.1,
                    attention_to_belief_location=0.3 * ratio,
                    attention_to_reality_location=0.3,
                    attention_to_question=0.3,
                    belief_vs_reality_ratio=ratio,
                ))

    # Analyze
    scores = analyzer.identify_tom_heads(mock_patterns)
    print("\nTop ToM heads:")
    for score in scores[:5]:
        print(f"  Layer {score.layer}, Head {score.head}: "
              f"FB ratio={score.false_belief_ratio:.2f}, "
              f"diff={score.condition_diff:.2f}, "
              f"score={score.tom_score:.3f}")

    # Create heatmap
    heatmap = analyzer.create_heatmap_data(
        mock_patterns, num_layers=3, num_heads=4,
        task_type="false_belief"
    )
    print(f"\nHeatmap shape: {heatmap.shape}")
    print(f"Heatmap:\n{heatmap}")

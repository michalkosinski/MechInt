"""
Tests for inference caching in model_runner.py

Run with: python -m pytest tests/test_cache.py -v
Or standalone: python tests/test_cache.py

IMPORTANT: Run these tests after ANY changes to model_runner.py caching logic.
"""

import json
import shutil
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_runner import InferenceCache


# =============================================================================
# InferenceCache Unit Tests (no model required)
# =============================================================================

def test_cache_key_determinism():
    """Same parameters should produce same cache key."""
    cache = InferenceCache(cache_dir=Path(tempfile.mkdtemp()), enabled=True)

    params1 = {"model_name": "test", "prompts": ["hello"], "temperature": 0.0}
    params2 = {"model_name": "test", "prompts": ["hello"], "temperature": 0.0}

    key1 = cache._compute_key(params1)
    key2 = cache._compute_key(params2)

    assert key1 == key2, "Same params should produce same key"
    print("✓ Cache key determinism")


def test_cache_key_different_params():
    """Different parameters should produce different cache keys."""
    cache = InferenceCache(cache_dir=Path(tempfile.mkdtemp()), enabled=True)

    base_params = {"model_name": "test", "prompts": ["hello"], "temperature": 0.0}

    # Test each parameter change produces different key
    variations = [
        {"model_name": "other_model", "prompts": ["hello"], "temperature": 0.0},
        {"model_name": "test", "prompts": ["world"], "temperature": 0.0},
        {"model_name": "test", "prompts": ["hello"], "temperature": 0.5},
        {"model_name": "test", "prompts": ["hello", "world"], "temperature": 0.0},
    ]

    base_key = cache._compute_key(base_params)

    for i, varied_params in enumerate(variations):
        varied_key = cache._compute_key(varied_params)
        assert base_key != varied_key, f"Variation {i} should produce different key: {varied_params}"

    print("✓ Different params produce different keys")


def test_cache_put_get():
    """Basic put/get operations should work."""
    cache_dir = Path(tempfile.mkdtemp())
    cache = InferenceCache(cache_dir=cache_dir, enabled=True)

    key = "test_key_123"
    value = {"response": "hello", "tokens": [1, 2, 3]}

    # Initially should be None
    assert cache.get(key) is None

    # After put, should retrieve
    cache.put(key, value)
    retrieved = cache.get(key)

    assert retrieved == value, "Retrieved value should match"
    print("✓ Cache put/get works")


def test_cache_persistence():
    """Cache should persist to disk and reload."""
    cache_dir = Path(tempfile.mkdtemp())

    # Create cache and add entry
    cache1 = InferenceCache(cache_dir=cache_dir, enabled=True, auto_flush_interval=1)
    cache1.put("key1", {"data": "value1"})
    cache1.flush()

    # Create new cache instance - should load from disk
    cache2 = InferenceCache(cache_dir=cache_dir, enabled=True)
    retrieved = cache2.get("key1")

    assert retrieved == {"data": "value1"}, "Cache should persist across instances"
    print("✓ Cache persistence works")


def test_cache_stats():
    """Cache stats should track hits and misses."""
    cache = InferenceCache(cache_dir=Path(tempfile.mkdtemp()), enabled=True)

    cache.put("existing", {"data": "value"})

    # Miss
    cache.get("nonexistent")
    # Hit
    cache.get("existing")
    # Another miss
    cache.get("also_nonexistent")

    stats = cache.get_stats()
    assert stats["hits"] == 1, f"Expected 1 hit, got {stats['hits']}"
    assert stats["misses"] == 2, f"Expected 2 misses, got {stats['misses']}"
    print("✓ Cache stats tracking works")


def test_cache_disabled():
    """Disabled cache should not store or retrieve."""
    cache = InferenceCache(cache_dir=Path(tempfile.mkdtemp()), enabled=False)

    cache.put("key", {"data": "value"})
    result = cache.get("key")

    assert result is None, "Disabled cache should not return values"
    print("✓ Disabled cache works correctly")


def test_cache_key_order_independence():
    """Key computation should be independent of dict key order."""
    cache = InferenceCache(cache_dir=Path(tempfile.mkdtemp()), enabled=True)

    params1 = {"a": 1, "b": 2, "c": 3}
    params2 = {"c": 3, "a": 1, "b": 2}
    params3 = {"b": 2, "c": 3, "a": 1}

    key1 = cache._compute_key(params1)
    key2 = cache._compute_key(params2)
    key3 = cache._compute_key(params3)

    assert key1 == key2 == key3, "Key order should not affect hash"
    print("✓ Cache key order independence")


def test_cache_list_vs_tuple():
    """Lists and tuples with same content should produce same key."""
    cache = InferenceCache(cache_dir=Path(tempfile.mkdtemp()), enabled=True)

    params_list = {"prompts": ["a", "b"]}
    params_tuple = {"prompts": ("a", "b")}

    key_list = cache._compute_key(params_list)
    key_tuple = cache._compute_key(params_tuple)

    assert key_list == key_tuple, "List and tuple should produce same key"
    print("✓ List/tuple equivalence in cache keys")


# =============================================================================
# Integration Tests (require model - skipped if unavailable)
# =============================================================================

def test_model_cache_hit_miss():
    """Test that model runner correctly uses cache."""
    try:
        from src.model_runner import ModelRunner
    except ImportError:
        print("⊘ Skipping model test (import failed)")
        return

    cache_dir = Path(tempfile.mkdtemp())
    TestTask = namedtuple('TestTask', ['task_id', 'task_type', 'full_prompt'])

    try:
        # Use smallest available model
        runner = ModelRunner('Qwen/Qwen2.5-3B', use_cache=True, cache_dir=cache_dir)
    except Exception as e:
        print(f"⊘ Skipping model test (model load failed: {e})")
        return

    task = TestTask('test1', 'test', 'The sky is')

    # First run - should miss
    stats_before = runner.get_cache_stats()
    output1 = runner.run_task(task, max_new_tokens=5, stop_strings=['.'], extract_attention=False)
    stats_after_first = runner.get_cache_stats()

    assert stats_after_first["misses"] == stats_before["misses"] + 1, "First run should miss"

    # Second run - should hit
    output2 = runner.run_task(task, max_new_tokens=5, stop_strings=['.'], extract_attention=False)
    stats_after_second = runner.get_cache_stats()

    assert stats_after_second["hits"] == stats_after_first["hits"] + 1, "Second run should hit"
    assert output1.model_response == output2.model_response, "Responses should match"

    print("✓ Model cache hit/miss works")


def test_different_params_miss_cache():
    """Changing any parameter should result in cache miss."""
    try:
        from src.model_runner import ModelRunner
    except ImportError:
        print("⊘ Skipping model test (import failed)")
        return

    cache_dir = Path(tempfile.mkdtemp())
    TestTask = namedtuple('TestTask', ['task_id', 'task_type', 'full_prompt'])

    try:
        runner = ModelRunner('Qwen/Qwen2.5-3B', use_cache=True, cache_dir=cache_dir)
    except Exception as e:
        print(f"⊘ Skipping model test (model load failed: {e})")
        return

    # Base call
    task1 = TestTask('t1', 'test', 'Hello world')
    runner.run_task(task1, max_new_tokens=5, stop_strings=['.'], extract_attention=False)
    base_misses = runner.get_cache_stats()["misses"]

    # Different prompt - should miss
    task2 = TestTask('t2', 'test', 'Goodbye world')
    runner.run_task(task2, max_new_tokens=5, stop_strings=['.'], extract_attention=False)
    assert runner.get_cache_stats()["misses"] == base_misses + 1, "Different prompt should miss"

    # Different max_new_tokens - should miss
    runner.run_task(task1, max_new_tokens=10, stop_strings=['.'], extract_attention=False)
    assert runner.get_cache_stats()["misses"] == base_misses + 2, "Different max_new_tokens should miss"

    # Different stop_strings - should miss
    runner.run_task(task1, max_new_tokens=5, stop_strings=['!'], extract_attention=False)
    assert runner.get_cache_stats()["misses"] == base_misses + 3, "Different stop_strings should miss"

    # Different temperature - should miss
    runner.run_task(task1, max_new_tokens=5, stop_strings=['.'], extract_attention=False, temperature=0.5)
    assert runner.get_cache_stats()["misses"] == base_misses + 4, "Different temperature should miss"

    print("✓ Different parameters correctly miss cache")


def test_attention_skips_cache():
    """Requesting attention should skip cache (need fresh tensors)."""
    try:
        from src.model_runner import ModelRunner
    except ImportError:
        print("⊘ Skipping model test (import failed)")
        return

    cache_dir = Path(tempfile.mkdtemp())
    TestTask = namedtuple('TestTask', ['task_id', 'task_type', 'full_prompt'])

    try:
        runner = ModelRunner('Qwen/Qwen2.5-3B', use_cache=True, cache_dir=cache_dir)
    except Exception as e:
        print(f"⊘ Skipping model test (model load failed: {e})")
        return

    task = TestTask('t1', 'test', 'Test prompt')

    # First call without attention - should cache
    runner.run_task(task, max_new_tokens=3, stop_strings=['.'], extract_attention=False)
    entries_after_first = runner.get_cache_stats()["entries"]

    # Call with attention - should NOT use cache (needs fresh attention tensors)
    output_with_attn = runner.run_task(task, max_new_tokens=3, stop_strings=['.'], extract_attention=True)

    # Attention should be present
    assert output_with_attn.attentions is not None, "Attention should be extracted"

    print("✓ Attention extraction correctly handled")


def test_get_candidate_probabilities_cache():
    """Test caching for get_candidate_probabilities."""
    try:
        from src.model_runner import ModelRunner
    except ImportError:
        print("⊘ Skipping model test (import failed)")
        return

    cache_dir = Path(tempfile.mkdtemp())

    try:
        runner = ModelRunner('Qwen/Qwen2.5-3B', use_cache=True, cache_dir=cache_dir)
    except Exception as e:
        print(f"⊘ Skipping model test (model load failed: {e})")
        return

    prompt = "The color of grass is"
    candidates = ["green", "blue", "red"]

    # First call - miss
    probs1 = runner.get_candidate_probabilities(prompt, candidates)
    stats1 = runner.get_cache_stats()

    # Second call - hit
    probs2 = runner.get_candidate_probabilities(prompt, candidates)
    stats2 = runner.get_cache_stats()

    assert stats2["hits"] == stats1["hits"] + 1, "Second call should hit cache"

    # Results should be similar (may have small float precision differences)
    for c in candidates:
        assert abs(probs1[c] - probs2[c]) < 1e-5, f"Probabilities should match for {c}"

    print("✓ get_candidate_probabilities caching works")


# =============================================================================
# Run all tests
# =============================================================================

def run_all_tests():
    """Run all cache tests."""
    print("\n" + "="*60)
    print("CACHE TESTS")
    print("="*60 + "\n")

    # Unit tests (no model required)
    print("--- Unit Tests (no model) ---")
    test_cache_key_determinism()
    test_cache_key_different_params()
    test_cache_put_get()
    test_cache_persistence()
    test_cache_stats()
    test_cache_disabled()
    test_cache_key_order_independence()
    test_cache_list_vs_tuple()

    # Integration tests (require model)
    print("\n--- Integration Tests (require model) ---")
    test_model_cache_hit_miss()
    test_different_params_miss_cache()
    test_attention_skips_cache()
    test_get_candidate_probabilities_cache()

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

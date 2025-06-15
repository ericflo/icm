"""
Comprehensive Test Suite for ICM Implementation
Tests individual components and full algorithm with different configurations
"""

import unittest
import torch
import numpy as np
from typing import List, Tuple
import json
import time
from pathlib import Path

# Import ICM components (assuming icm.py contains the main implementation)
from icm_implementation import (
    ICMConfig, DataPoint, ModelBackend, TransformersBackend,
    LogicalConsistency, ICM, create_truthfulness_dataset,
    create_math_correctness_dataset, create_comparison_dataset
)


class MockModelBackend(ModelBackend):
    """Mock backend for testing without actual model inference"""
    def __init__(self, config: ICMConfig):
        super().__init__(config)
        self.call_count = 0
        
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Mock generation - returns alternating True/False"""
        self.call_count += 1
        results = []
        for i, prompt in enumerate(prompts):
            if "true" in prompt.lower() or i % 2 == 0:
                results.append("True")
            else:
                results.append("False")
        return results
        
    def get_log_probs(self, prompts: List[str], targets: List[str]) -> List[float]:
        """Mock log probabilities - higher for consistent patterns"""
        self.call_count += 1
        log_probs = []
        for prompt, target in zip(prompts, targets):
            # Higher probability if target matches pattern in prompt
            if ("true" in prompt.lower() and "true" in target.lower()) or \
               ("false" in prompt.lower() and "false" in target.lower()):
                log_probs.append(-0.5)  # High probability
            else:
                log_probs.append(-2.0)  # Lower probability
        return log_probs


class TestLogicalConsistency(unittest.TestCase):
    """Test logical consistency checking"""
    
    def setUp(self):
        self.checker = LogicalConsistency()
        
    def test_general_consistency(self):
        """Test general consistency (no specific constraints)"""
        x1 = DataPoint(1, "Test 1")
        x2 = DataPoint(2, "Test 2")
        
        # Should always return True for general consistency
        self.assertTrue(self.checker.check_consistency(x1, 0, x2, 0))
        self.assertTrue(self.checker.check_consistency(x1, 0, x2, 1))
        self.assertTrue(self.checker.check_consistency(x1, 1, x2, 0))
        self.assertTrue(self.checker.check_consistency(x1, 1, x2, 1))
        
    def test_asymmetry_consistency(self):
        """Test asymmetry consistency for comparisons"""
        self.checker.consistency_type = "asymmetry"
        
        # Create comparison data points
        x1 = DataPoint(1, "A vs B", metadata={"comparison": ("A", "B")})
        x2 = DataPoint(2, "B vs A", metadata={"comparison": ("B", "A")})
        x3 = DataPoint(3, "A vs C", metadata={"comparison": ("A", "C")})
        
        # Opposite comparisons should have opposite labels
        self.assertFalse(self.checker.check_consistency(x1, 1, x2, 1))  # Both True
        self.assertFalse(self.checker.check_consistency(x1, 0, x2, 0))  # Both False
        self.assertTrue(self.checker.check_consistency(x1, 1, x2, 0))   # Opposite
        self.assertTrue(self.checker.check_consistency(x1, 0, x2, 1))   # Opposite
        
        # Different comparisons have no constraints
        self.assertTrue(self.checker.check_consistency(x1, 1, x3, 1))
        self.assertTrue(self.checker.check_consistency(x1, 0, x3, 0))
        
    def test_math_consistency(self):
        """Test mathematical correctness consistency"""
        self.checker.consistency_type = "math_correctness"
        
        # Same problem, different answers
        x1 = DataPoint(1, "2+2=?", metadata={"problem_id": 1, "answer": "4"})
        x2 = DataPoint(2, "2+2=?", metadata={"problem_id": 1, "answer": "5"})
        x3 = DataPoint(3, "3+3=?", metadata={"problem_id": 2, "answer": "6"})
        
        # Different answers can't both be correct
        self.assertFalse(self.checker.check_consistency(x1, 1, x2, 1))  # Both correct
        self.assertTrue(self.checker.check_consistency(x1, 1, x2, 0))   # One correct
        self.assertTrue(self.checker.check_consistency(x1, 0, x2, 1))   # One correct
        self.assertTrue(self.checker.check_consistency(x1, 0, x2, 0))   # Both wrong OK
        
        # Different problems have no constraints
        self.assertTrue(self.checker.check_consistency(x1, 1, x3, 1))
        
    def test_count_inconsistencies(self):
        """Test counting inconsistencies in dataset"""
        self.checker.consistency_type = "asymmetry"
        
        # Create dataset with known inconsistencies
        data = [
            (DataPoint(1, "A>B", metadata={"comparison": ("A", "B")}), 1),  # A > B
            (DataPoint(2, "B>A", metadata={"comparison": ("B", "A")}), 1),  # B > A (inconsistent!)
            (DataPoint(3, "A>C", metadata={"comparison": ("A", "C")}), 0),  # A < C
        ]
        
        count = self.checker.count_inconsistencies(data)
        self.assertEqual(count, 1)  # One inconsistency
        
        pairs = self.checker.get_inconsistent_pairs(data)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], (0, 1))  # Indices of inconsistent pair


class TestICMCore(unittest.TestCase):
    """Test core ICM functionality"""
    
    def setUp(self):
        self.config = ICMConfig(
            num_labels=2,
            label_names=["False", "True"],
            initial_examples=2,
            alpha=10.0
        )
        self.mock_backend = MockModelBackend(self.config)
        self.icm = ICM(self.config, backend=self.mock_backend)
        
    def test_format_in_context_prompt(self):
        """Test prompt formatting"""
        x_query = DataPoint(3, "Is the sky blue?")
        labeled_data = [
            (DataPoint(1, "Is grass green?"), 1),
            (DataPoint(2, "Is snow black?"), 0),
        ]
        
        prompt = self.icm.format_in_context_prompt(x_query, labeled_data)
        
        # Check prompt structure
        self.assertIn("Is grass green?", prompt)
        self.assertIn("Label: True", prompt)
        self.assertIn("Is snow black?", prompt)
        self.assertIn("Label: False", prompt)
        self.assertIn("Is the sky blue?", prompt)
        self.assertTrue(prompt.endswith("Label:"))
        
    def test_calculate_mutual_predictability(self):
        """Test mutual predictability calculation"""
        labeled_data = [
            (DataPoint(1, "True statement"), 1),
            (DataPoint(2, "False statement"), 0),
            (DataPoint(3, "Another true"), 1),
        ]
        
        score = self.icm.calculate_mutual_predictability(labeled_data)
        
        # Should return negative log probability
        self.assertLess(score, 0)
        self.assertEqual(self.mock_backend.call_count, 1)  # One batch call
        
    def test_calculate_score(self):
        """Test overall score calculation"""
        # Create data with known consistency
        labeled_data = [
            (DataPoint(1, "A>B", metadata={"comparison": ("A", "B")}), 1),
            (DataPoint(2, "B>A", metadata={"comparison": ("B", "A")}), 0),  # Consistent
        ]
        
        self.icm.consistency_checker.consistency_type = "asymmetry"
        score, predictability, inconsistencies = self.icm.calculate_score(labeled_data)
        
        self.assertEqual(inconsistencies, 0)
        self.assertLess(predictability, 0)  # Negative log prob
        self.assertAlmostEqual(score, self.config.alpha * predictability)
        
    def test_consistency_fix(self):
        """Test consistency fixing algorithm"""
        # Create inconsistent data
        self.icm.consistency_checker.consistency_type = "asymmetry"
        
        labeled_data = [
            (DataPoint(1, "A>B", metadata={"comparison": ("A", "B")}), 1),  # A > B
            (DataPoint(2, "B>A", metadata={"comparison": ("B", "A")}), 1),  # B > A (wrong!)
            (DataPoint(3, "C>D", metadata={"comparison": ("C", "D")}), 0),
        ]
        
        # Count initial inconsistencies
        initial_inconsistencies = self.icm.consistency_checker.count_inconsistencies(labeled_data)
        self.assertGreater(initial_inconsistencies, 0)
        
        # Fix inconsistencies
        fixed_data = self.icm.consistency_fix(labeled_data)
        
        # Check that inconsistencies are reduced or resolved
        final_inconsistencies = self.icm.consistency_checker.count_inconsistencies(fixed_data)
        # With mock backend, we may not always achieve 0 inconsistencies
        self.assertLessEqual(final_inconsistencies, initial_inconsistencies)
        
    def test_temperature_scheduling(self):
        """Test temperature scheduling"""
        initial_temp = self.config.initial_temperature
        
        # Run a few iterations
        dataset = [DataPoint(i, f"Test {i}") for i in range(5)]
        self.icm.run(dataset, max_iterations=3)
        
        # Temperature should decrease
        self.assertLess(self.icm.temperature, initial_temp)
        self.assertGreater(self.icm.temperature, self.config.final_temperature)


class TestDatasetCreation(unittest.TestCase):
    """Test dataset creation utilities"""
    
    def test_truthfulness_dataset(self):
        """Test TruthfulQA-style dataset creation"""
        data = [
            ("What is 2+2?", "2+2 equals 4", True),
            ("What is the capital of France?", "The capital is London", False),
        ]
        
        dataset = create_truthfulness_dataset(data)
        
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].label, 1)  # True
        self.assertEqual(dataset[1].label, 0)  # False
        self.assertIn("What is 2+2?", dataset[0].input_text)
        self.assertIn("2+2 equals 4", dataset[0].input_text)
        
    def test_math_correctness_dataset(self):
        """Test GSM8K-style dataset creation"""
        data = [
            ("If John has 3 apples...", "John has 5 apples total", "5", True),
            ("Calculate 10/2", "10 divided by 2 is 3", "3", False),
        ]
        
        dataset = create_math_correctness_dataset(data)
        
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].metadata["answer"], "5")
        self.assertEqual(dataset[0].metadata["problem_id"], hash(data[0][0]))
        
    def test_comparison_dataset(self):
        """Test Alpaca-style dataset creation"""
        data = [
            ("Write a poem", "Roses are red...", "Error 404", True),
            ("Explain gravity", "It pulls things", "Gravity is a force...", False),
        ]
        
        dataset = create_comparison_dataset(data)
        
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].metadata["comparison"], ("A", "B"))
        self.assertIn("Response A is more helpful", dataset[0].input_text)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_small_truthfulness_task(self):
        """Test on small truthfulness dataset"""
        # Create small dataset
        data = [
            ("Is 2+2=4?", "Yes, 2+2=4", None),
            ("Is the Earth flat?", "No, the Earth is round", None),
            ("Is water wet?", "Yes, water is wet", None),
            ("Can pigs fly?", "No, pigs cannot fly", None),
        ]
        
        dataset = create_truthfulness_dataset(data)
        
        # Run ICM with mock backend
        config = ICMConfig(initial_examples=2, alpha=5.0)
        mock_backend = MockModelBackend(config)
        icm = ICM(config, backend=mock_backend)
        
        labeled_data = icm.run(dataset, max_iterations=5)
        
        # Verify results
        self.assertEqual(len(labeled_data), 4)
        self.assertTrue(all(label in [0, 1] for _, label in labeled_data))
        
        # Check final metrics
        score, _, inconsistencies = icm.calculate_score(labeled_data)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(inconsistencies, 0)
        
    def test_performance_tracking(self):
        """Test performance metrics tracking"""
        dataset = [DataPoint(i, f"Test {i}") for i in range(10)]
        
        config = ICMConfig(initial_examples=3)
        mock_backend = MockModelBackend(config)
        icm = ICM(config, backend=mock_backend)
        
        icm.run(dataset, max_iterations=5)
        
        # Check tracking
        self.assertEqual(len(icm.score_history), 5)
        self.assertEqual(len(icm.acceptance_history), 5)
        self.assertTrue(all(isinstance(s, float) for s in icm.score_history))
        self.assertTrue(all(isinstance(a, bool) for a in icm.acceptance_history))


class BenchmarkTests(unittest.TestCase):
    """Benchmark performance tests"""
    
    def test_scaling_with_dataset_size(self):
        """Test how algorithm scales with dataset size"""
        sizes = [10, 50, 100]
        times = []
        
        for size in sizes:
            dataset = [DataPoint(i, f"Test {i}") for i in range(size)]
            
            config = ICMConfig(initial_examples=min(5, size))
            mock_backend = MockModelBackend(config)
            icm = ICM(config, backend=mock_backend)
            
            start_time = time.time()
            icm.run(dataset, max_iterations=10)
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            print(f"Size {size}: {elapsed:.2f}s")
            
        # Verify reasonable scaling
        self.assertLess(times[-1], times[0] * 20)  # Should not scale too badly
        
    def test_memory_usage(self):
        """Test memory usage stays reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run with larger dataset
        dataset = [DataPoint(i, f"Test string {i}" * 10) for i in range(100)]
        
        config = ICMConfig(initial_examples=5)
        mock_backend = MockModelBackend(config)
        icm = ICM(config, backend=mock_backend)
        
        icm.run(dataset, max_iterations=20)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f} MB")
        self.assertLess(memory_increase, 500)  # Should not use excessive memory


def run_specific_tests():
    """Run specific test scenarios for debugging"""
    print("Running specific ICM tests...")
    print("=" * 60)
    
    # Test 1: Logical consistency
    print("\n1. Testing Logical Consistency Component")
    checker = LogicalConsistency("asymmetry")
    x1 = DataPoint(1, "A>B", metadata={"comparison": ("A", "B")})
    x2 = DataPoint(2, "B>A", metadata={"comparison": ("B", "A")})
    
    consistent = checker.check_consistency(x1, 1, x2, 0)
    inconsistent = checker.check_consistency(x1, 1, x2, 1)
    
    print(f"   A>B=True, B>A=False: Consistent? {consistent}")
    print(f"   A>B=True, B>A=True: Consistent? {inconsistent}")
    
    # Test 2: Score calculation
    print("\n2. Testing Score Calculation")
    config = ICMConfig(alpha=10.0)
    mock_backend = MockModelBackend(config)
    icm = ICM(config, backend=mock_backend)
    
    labeled_data = [
        (DataPoint(1, "True fact"), 1),
        (DataPoint(2, "False fact"), 0),
    ]
    
    score, pred, incons = icm.calculate_score(labeled_data)
    print(f"   Score: {score:.2f}")
    print(f"   Predictability: {pred:.2f}")
    print(f"   Inconsistencies: {incons}")
    
    # Test 3: Small ICM run
    print("\n3. Running Small ICM Example")
    dataset = [
        DataPoint(1, "The sky is blue"),
        DataPoint(2, "Grass is purple"),
        DataPoint(3, "Water is wet"),
    ]
    
    config = ICMConfig(initial_examples=1, alpha=5.0)
    mock_backend = MockModelBackend(config)
    icm = ICM(config, backend=mock_backend)
    
    labeled_data = icm.run(dataset, max_iterations=3)
    
    print("   Results:")
    for dp, label in labeled_data:
        print(f"   - '{dp.input_text}' -> {config.label_names[label]}")


if __name__ == "__main__":
    # Run specific tests first
    run_specific_tests()
    
    # Then run full test suite
    print("\n" + "=" * 60)
    print("Running Full Test Suite")
    print("=" * 60)
    
    unittest.main(verbosity=2)
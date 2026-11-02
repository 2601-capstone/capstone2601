"""
Unit tests for Federated Averaging (FedAvg) logic (src/aggregation.py).
"""

import unittest
import numpy as np

from src.aggregation import federated_averaging


class TestFederatedAveraging(unittest.TestCase):
    """Test suite for FedAvg mathematical weighting and error handling."""

    def setUp(self):
        # Standard mock parameters for 3 clients
        self.c1 = {
            "W1": np.ones((5, 8)) * 1.0,
            "b1": np.ones((1, 8)) * 1.0,
            "W2": np.ones((8, 1)) * 1.0,
            "b2": np.ones((1, 1)) * 1.0
        }
        self.c2 = {
            "W1": np.ones((5, 8)) * 2.0,
            "b1": np.ones((1, 8)) * 2.0,
            "W2": np.ones((8, 1)) * 2.0,
            "b2": np.ones((1, 1)) * 2.0
        }
        self.c3 = {
            "W1": np.ones((5, 8)) * 3.0,
            "b1": np.ones((1, 8)) * 3.0,
            "W2": np.ones((8, 1)) * 3.0,
            "b2": np.ones((1, 1)) * 3.0
        }

    def test_equal_weighted_aggregation(self):
        """When sample sizes are equal (400 each), FedAvg should equal simple average."""
        samples = [400, 400, 400]
        agg = federated_averaging([self.c1, self.c2, self.c3], samples)

        # Expected value: (1.0 + 2.0 + 3.0) / 3 = 2.0
        for key in ["W1", "b1", "W2", "b2"]:
            self.assertTrue(np.allclose(agg[key], 2.0))

    def test_sample_weighted_aggregation(self):
        """When sample sizes differ, weights should be proportional: sum(n_k/N * W_k)."""
        # Samples: 100, 200, 700 (total = 1000)
        # Expected: 0.1 * 1.0 + 0.2 * 2.0 + 0.7 * 3.0 = 0.1 + 0.4 + 2.1 = 2.6
        samples = [100, 200, 700]
        agg = federated_averaging([self.c1, self.c2, self.c3], samples)

        for key in ["W1", "b1", "W2", "b2"]:
            self.assertTrue(np.allclose(agg[key], 2.6))

    def test_correct_parameter_shapes(self):
        """Aggregated parameters must preserve exact matrix dimensions."""
        agg = federated_averaging([self.c1, self.c2], [100, 100])
        self.assertEqual(agg["W1"].shape, (5, 8))
        self.assertEqual(agg["b1"].shape, (1, 8))
        self.assertEqual(agg["W2"].shape, (8, 1))
        self.assertEqual(agg["b2"].shape, (1, 1))

    def test_original_parameters_not_modified(self):
        """Aggregation must return new copies and never mutate client inputs."""
        c1_w1_orig = self.c1["W1"].copy()
        agg = federated_averaging([self.c1, self.c2], [100, 100])
        agg["W1"] += 50.0  # mutate result

        # Original c1 must remain untouched
        self.assertTrue(np.array_equal(self.c1["W1"], c1_w1_orig))

    def test_mismatched_parameter_keys_raise_error(self):
        """Missing or extraneous keys should raise ValueError."""
        bad_client = {"W1": np.ones((5, 8)), "b1": np.zeros((1, 8))}  # missing W2, b2
        with self.assertRaises(ValueError):
            federated_averaging([self.c1, bad_client], [100, 100])

    def test_mismatched_shapes_raise_error(self):
        """Incompatible matrix shapes across clients must raise ValueError."""
        bad_shape_client = {
            "W1": np.ones((3, 3)),  # should be (5, 8)
            "b1": np.ones((1, 8)),
            "W2": np.ones((8, 1)),
            "b2": np.ones((1, 1))
        }
        with self.assertRaises(ValueError):
            federated_averaging([self.c1, bad_shape_client], [100, 100])

    def test_invalid_sample_counts_raise_error(self):
        """Zero or negative sample counts must raise ValueError."""
        with self.assertRaises(ValueError):
            federated_averaging([self.c1, self.c2], [100, 0])
        with self.assertRaises(ValueError):
            federated_averaging([self.c1, self.c2], [100, -50])
        with self.assertRaises(ValueError):
            federated_averaging([self.c1, self.c2], [100])  # length mismatch


if __name__ == "__main__":
    unittest.main()

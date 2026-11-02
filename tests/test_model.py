"""
Unit tests for Local Model Architecture (src/model.py).
"""

import unittest
import numpy as np
from src.model import HealthcareMLP


class TestHealthcareMLP(unittest.TestCase):
    """Test suite for the HealthcareMLP neural network architecture."""

    def setUp(self):
        self.input_dim = 5
        self.hidden_dim = 8
        self.output_dim = 1
        self.model = HealthcareMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            random_seed=42
        )

    def test_parameter_initialization_shapes(self):
        """Test parameter initialization and exact matrix dimensions."""
        params = self.model.get_parameters()
        self.assertIn("W1", params)
        self.assertIn("b1", params)
        self.assertIn("W2", params)
        self.assertIn("b2", params)

        self.assertEqual(params["W1"].shape, (5, 8))
        self.assertEqual(params["b1"].shape, (1, 8))
        self.assertEqual(params["W2"].shape, (8, 1))
        self.assertEqual(params["b2"].shape, (1, 1))

    def test_forward_propagation_range(self):
        """Forward pass should output probabilities strictly bounded in [0, 1]."""
        X = np.random.randn(20, self.input_dim)
        probs = self.model.predict_proba(X)
        self.assertEqual(probs.shape, (20, 1))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    def test_loss_computation(self):
        """Binary cross-entropy loss should be finite, non-negative, and scalar."""
        y_true = np.array([0, 1, 1, 0]).reshape(-1, 1)
        y_pred = np.array([0.1, 0.8, 0.9, 0.2]).reshape(-1, 1)
        loss = self.model.compute_loss(y_true, y_pred)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
        self.assertGreater(loss, 0.0)

    def test_predict_binary_labels(self):
        """Predictions should be integer binary labels (0 or 1)."""
        X = np.random.randn(15, self.input_dim)
        preds = self.model.predict(X, threshold=0.5)
        self.assertEqual(preds.shape, (15, 1))
        unique_vals = set(np.unique(preds))
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_training_updates_parameters(self):
        """Weights and biases must change after training on dummy data."""
        X = np.random.randn(50, self.input_dim)
        y = np.random.randint(0, 2, size=(50, 1))

        initial_params = self.model.get_parameters()
        loss_history = self.model.fit(X, y, epochs=5, batch_size=16, learning_rate=0.05)

        updated_params = self.model.get_parameters()

        # Verify losses are finite
        for loss in loss_history:
            self.assertFalse(np.isnan(loss))
            self.assertFalse(np.isinf(loss))

        # Check parameter changes
        w1_diff = np.sum(np.abs(updated_params["W1"] - initial_params["W1"]))
        w2_diff = np.sum(np.abs(updated_params["W2"] - initial_params["W2"]))
        self.assertGreater(w1_diff, 1e-5)
        self.assertGreater(w2_diff, 1e-5)

    def test_get_parameters_returns_deep_copies(self):
        """Modifying dictionary from get_parameters should not mutate internal weights."""
        params = self.model.get_parameters()
        original_w1_val = params["W1"][0, 0]
        params["W1"][0, 0] += 999.0

        internal_params = self.model.get_parameters()
        self.assertEqual(internal_params["W1"][0, 0], original_w1_val)

    def test_set_parameters_validation(self):
        """Setting invalid parameter keys or shapes must raise ValueError."""
        # Missing key
        with self.assertRaises(ValueError):
            self.model.set_parameters({"W1": np.zeros((5, 8))})

        # Wrong shape
        with self.assertRaises(ValueError):
            invalid_shape_params = self.model.get_parameters()
            invalid_shape_params["W1"] = np.zeros((3, 3))
            self.model.set_parameters(invalid_shape_params)


if __name__ == "__main__":
    unittest.main()

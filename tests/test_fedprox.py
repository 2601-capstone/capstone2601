"""
Unit tests for FedProx Algorithm and Comparison Module (Phase 10).
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.model import HealthcareMLP, compute_proximal_loss
from src.client import HospitalClient
from src.experiment import run_federated_experiment
from src.fedprox_comparison import (
    run_fedavg_vs_fedprox_comparison,
    evaluate_model_on_central_test,
    evaluate_model_client_robustness
)
from src.communication_analysis import calculate_model_parameter_bytes


class TestFedProx(unittest.TestCase):
    """Comprehensive test suite for FedProx algorithmic implementation and empirical comparisons."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_proximal_loss_computation_exact(self):
        """Verify (mu/2) * sum(||w - w_t||^2) against manual mathematical calculation."""
        # W1: diff is [[1.0, 2.0]] -> norm^2 = 1 + 4 = 5
        # b1: diff is [[2.0]]      -> norm^2 = 4
        # Total squared norm = 5 + 4 = 9
        # Proximal loss with mu = 0.5: 0.5/2 * 9 = 2.25
        params = {
            "W1": np.array([[1.0, 2.0]], dtype=float),
            "b1": np.array([[3.0]], dtype=float)
        }
        global_params = {
            "W1": np.array([[0.0, 0.0]], dtype=float),
            "b1": np.array([[1.0]], dtype=float)
        }
        loss = compute_proximal_loss(params, global_params, mu=0.5)
        self.assertAlmostEqual(loss, 2.25, places=6)

    def test_proximal_loss_zero_when_identical(self):
        """Verify proximal loss evaluates to exactly 0 when parameters match or mu=0."""
        params = {
            "W1": np.array([[1.5, -0.5]]),
            "b1": np.array([[0.25]])
        }
        loss_identical = compute_proximal_loss(params, params, mu=0.05)
        self.assertEqual(loss_identical, 0.0)

        loss_mu_zero = compute_proximal_loss(params, {"W1": np.zeros((1, 2)), "b1": np.zeros((1, 1))}, mu=0.0)
        self.assertEqual(loss_mu_zero, 0.0)

    def test_mu_validation(self):
        """Verify negative mu values raise ValueError across model and experiment coordinator."""
        params = {"W1": np.array([[1.0]])}
        with self.assertRaises(ValueError):
            compute_proximal_loss(params, params, mu=-0.01)

        model = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1)
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, (10, 1))
        with self.assertRaises(ValueError):
            model.fit(X, y, epochs=1, mu=-0.05)

        with self.assertRaises(ValueError):
            run_federated_experiment(num_rounds=1, algorithm="FedProx", mu=-0.01)

    def test_proximal_gradient_application(self):
        """Verify that proximal gradient applies an additional mu * (w - w_t) term."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, (20, 1)).astype(float)

        # Baseline model with no proximal regularization (mu=0)
        model_standard = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=10)
        initial_params = model_standard.get_parameters()

        # Global reference parameters (slightly offset)
        global_params = {k: v + 0.1 for k, v in initial_params.items()}

        model_standard.train_step(X, y, learning_rate=0.01, global_parameters=global_params, mu=0.0)
        params_standard = model_standard.get_parameters()

        # Model with proximal regularization (mu=0.05)
        model_prox = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=10)
        model_prox.train_step(X, y, learning_rate=0.01, global_parameters=global_params, mu=0.05)
        params_prox = model_prox.get_parameters()

        # Analytical check: the extra gradient is mu * (initial_w - w_t)
        # Therefore, proximal w = standard_w - lr * mu * (initial_w - w_t)
        lr = 0.01
        mu = 0.05
        for k in ["W1", "b1", "W2", "b2"]:
            expected_diff = -lr * mu * (initial_params[k] - global_params[k])
            actual_diff = params_prox[k] - params_standard[k]
            np.testing.assert_allclose(actual_diff, expected_diff, rtol=1e-5, atol=1e-6)

    def test_client_retains_global_parameters(self):
        """Ensure HospitalClient preserves initial round parameters without corruption."""
        client = HospitalClient(hospital_id="hospital_1", data_path=os.path.join("data", "hospital_1_train.csv"))
        initial_p = client.get_model_parameters()

        # Verify client deep-copies parameters
        client.set_model_parameters(initial_p)
        self.assertIsNotNone(client.global_parameters)
        for k in initial_p:
            np.testing.assert_array_equal(client.global_parameters[k], initial_p[k])

        # Mutate local model parameters and verify client.global_parameters is unaffected
        mutated_p = {k: v + 1.0 for k, v in initial_p.items()}
        client.model.set_parameters(mutated_p)
        for k in initial_p:
            self.assertFalse(np.array_equal(client.model.get_parameters()[k], client.global_parameters[k]))

    def test_fedavg_behavior_remains_unchanged(self):
        """Verify that default FedAvg execution is completely identical to previous phases."""
        res_default = run_federated_experiment(
            num_rounds=1,
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            random_seed=42,
            save_history=False
        )
        res_explicit = run_federated_experiment(
            num_rounds=1,
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            random_seed=42,
            save_history=False,
            algorithm="FedAvg",
            mu=0.0
        )
        for k in ["W1", "b1", "W2", "b2"]:
            np.testing.assert_array_equal(
                res_default[0].get_parameters()[k],
                res_explicit[0].get_parameters()[k]
            )

    def test_fedprox_mu_zero_matches_fedavg(self):
        """Verify mathematical equivalence: FedProx with mu=0.0 is strictly identical to FedAvg."""
        model_fedavg, hist_fedavg = run_federated_experiment(
            num_rounds=2,
            local_epochs=3,
            batch_size=32,
            learning_rate=0.01,
            random_seed=99,
            save_history=False,
            algorithm="FedAvg",
            mu=0.0
        )
        model_fedprox_zero, hist_fedprox_zero = run_federated_experiment(
            num_rounds=2,
            local_epochs=3,
            batch_size=32,
            learning_rate=0.01,
            random_seed=99,
            save_history=False,
            algorithm="FedProx",
            mu=0.0
        )
        # Parameters must match exactly
        for k in ["W1", "b1", "W2", "b2"]:
            np.testing.assert_allclose(
                model_fedavg.get_parameters()[k],
                model_fedprox_zero.get_parameters()[k],
                rtol=1e-7,
                atol=1e-7
            )
        # Training loss history must match exactly
        np.testing.assert_allclose(
            hist_fedavg["average_local_loss"].to_numpy(),
            hist_fedprox_zero["average_local_loss"].to_numpy(),
            rtol=1e-7,
            atol=1e-7
        )

    def test_reproducibility_with_fixed_seed(self):
        """Ensure repeated FedProx experiments with the same seed yield identical weights and losses."""
        m1, h1 = run_federated_experiment(
            num_rounds=2,
            local_epochs=2,
            learning_rate=0.01,
            random_seed=77,
            save_history=False,
            algorithm="FedProx",
            mu=0.02
        )
        m2, h2 = run_federated_experiment(
            num_rounds=2,
            local_epochs=2,
            learning_rate=0.01,
            random_seed=77,
            save_history=False,
            algorithm="FedProx",
            mu=0.02
        )
        for k in ["W1", "b1", "W2", "b2"]:
            np.testing.assert_array_equal(m1.get_parameters()[k], m2.get_parameters()[k])
        np.testing.assert_array_equal(h1["average_local_loss"], h2["average_local_loss"])

    def test_run_fedavg_vs_fedprox_comparison_pipeline(self):
        """Verify the complete comparison pipeline executes and saves all expected artifacts."""
        summary = run_fedavg_vs_fedprox_comparison(
            num_rounds=1,
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            mu=0.01,
            random_seed=42,
            output_dir=self.temp_dir
        )
        self.assertIn("experiment_configuration", summary)
        self.assertIn("central_evaluation", summary)
        self.assertIn("client_robustness", summary)
        self.assertIn("communication_and_runtime", summary)
        self.assertIn("comparison_table", summary)

        # Verify artifacts written to output_dir
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "fedavg_vs_fedprox_comparison.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "fedavg_vs_fedprox_summary.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "federated_training_history.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "fedprox_training_history.csv")))

        # Check central test evaluation metrics
        for alg in ["fedavg", "fedprox"]:
            metrics = summary["central_evaluation"][alg]
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1_score", metrics)
            self.assertIn("roc_auc", metrics)
            self.assertGreaterEqual(metrics["accuracy"], 0.0)
            self.assertLessEqual(metrics["accuracy"], 1.0)

        # Check communication metrics
        comm = summary["communication_and_runtime"]
        self.assertEqual(comm["parameter_count"], 57)
        self.assertEqual(comm["parameter_bytes"], 456)
        self.assertEqual(comm["total_communication_bytes"], 2736)  # 1 round * 2736 bytes


if __name__ == "__main__":
    unittest.main()

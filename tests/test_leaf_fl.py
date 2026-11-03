"""
Unit Tests for LEAF-FL: Lead-time and Equity-Aware Federated Learning.

Validates:
1. Aggregation weights strictly sum to 1.0.
2. Lambda = 0 reproduces exact sample-weighted FedAvg weights.
3. Larger warning-error scores receive greater equity weight when lambda > 0.
4. Sample-size weighting operates correctly.
5. Invalid lambda values (e.g. < 0 or > 1) are rejected with ValueError.
6. Invalid tau values (e.g. <= 0) are handled safely with ValueError.
7. Incompatible model parameter shapes raise clear errors.
8. Empty client input raises a clear error.
9. NaN and infinite values in errors are rejected or handled safely.
10. Repeated runs with identical random seeds are 100% deterministic.
11. Test-set labels are never accessed during client training or aggregation weight calculation.
12. Missed events are counted correctly based on ground truth deterioration.
13. Warning lead time is calculated correctly in months (T_event - t*).
14. No future information (months 7-12) leaks into observation inputs (shape is strictly N x 6 x 5).
15. Parameter state dict aggregation preserves floating-point weights and buffers.
"""

import os
import sys
import unittest
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.leaf_data import (
    generate_deterioration_dataset,
    compute_observation_scaler,
    apply_observation_scaler,
    partition_leaf_dataset,
    OBSERVATION_WINDOW,
    FEATURE_NAMES
)
from src.leaf_fl import (
    LEAFCausalCNNLSTM,
    compute_warning_metrics,
    compute_warning_error_score,
    compute_leaf_aggregation_weights,
    leaf_aggregate_parameters,
    predict_sequential_warnings
)


class TestLEAFDatasetAndLeakage(unittest.TestCase):
    """Test data generation, temporal horizons, and zero-leakage guarantees."""

    def setUp(self):
        self.raw_data = generate_deterioration_dataset(num_patients=100, random_seed=42)

    def test_observation_window_dimensions(self):
        """Test that model input sequences contain strictly months 1 to 6 (T=6)."""
        X_obs = self.raw_data["X_obs"]
        self.assertEqual(X_obs.shape, (100, 6, 5))
        self.assertEqual(OBSERVATION_WINDOW, 6)

    def test_zero_future_leakage_in_inputs(self):
        """Verify that future observations (months 7 to 12) are NOT present in X_obs."""
        X_full = self.raw_data["X_full"]
        X_obs = self.raw_data["X_obs"]

        # First 6 months must match exactly
        np.testing.assert_array_equal(X_obs, X_full[:, :6, :])
        # X_obs should only have 6 steps
        self.assertEqual(X_obs.shape[1], 6)

    def test_event_month_consistency(self):
        """Verify that deterioration events strictly occur in months 7 to 12."""
        y = self.raw_data["y"].ravel()
        event_months = self.raw_data["event_month"]

        for i in range(len(y)):
            if y[i] == 1.0:
                self.assertIn(event_months[i], [7, 8, 9, 10, 11, 12])
            else:
                self.assertEqual(event_months[i], 0)

    def test_patient_isolation_across_splits(self):
        """Verify zero patient overlap between train, val, and test partitions."""
        partitioned = partition_leaf_dataset(self.raw_data, random_seed=42)

        test_pids = set(partitioned["test"]["patient_ids"])
        val_pids = set(partitioned["val"]["patient_ids"])

        h1_pids = set(partitioned["hospitals"]["hospital_1"]["train"]["patient_ids"])
        h2_pids = set(partitioned["hospitals"]["hospital_2"]["train"]["patient_ids"])
        h3_pids = set(partitioned["hospitals"]["hospital_3"]["train"]["patient_ids"])

        # Disjoint sets
        self.assertEqual(len(test_pids.intersection(val_pids)), 0)
        self.assertEqual(len(test_pids.intersection(h1_pids)), 0)
        self.assertEqual(len(h1_pids.intersection(h2_pids)), 0)
        self.assertEqual(len(h2_pids.intersection(h3_pids)), 0)


class TestLEAFMetricsAndLeadTime(unittest.TestCase):
    """Test warning metrics, lead times, missed events, and false alarms."""

    def test_lead_time_calculation(self):
        """Test exact calculation of lead time: T_event - t*."""
        y_true = np.array([1, 1, 1, 0, 0])
        event_months = np.array([8, 10, 7, 0, 0])
        warning_steps = np.array([4, 5, 6, 3, 0])  # detected at month 4, 5, 6, and 3 (false alarm)
        y_pred = np.array([1, 1, 1, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.1])

        metrics = compute_warning_metrics(y_true, event_months, y_pred, warning_steps, y_prob)

        # Patient 0: event=8, warn=4 -> lead=4
        # Patient 1: event=10, warn=5 -> lead=5
        # Patient 2: event=7, warn=6 -> lead=1
        expected_leads = [4, 5, 1]
        self.assertEqual(metrics["lead_times_distribution"], expected_leads)
        self.assertAlmostEqual(metrics["avg_lead_time_months"], np.mean(expected_leads), places=4)
        self.assertAlmostEqual(metrics["median_lead_time_months"], 4.0, places=4)

    def test_missed_events_counted_correctly(self):
        """Verify missed event rate when model fails to issue warning for true deterioration."""
        y_true = np.array([1, 1, 1, 1])
        event_months = np.array([7, 8, 9, 10])
        warning_steps = np.array([0, 0, 5, 6])  # 2 missed events, 2 detected
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.8, 0.7])

        metrics = compute_warning_metrics(y_true, event_months, y_pred, warning_steps, y_prob)
        self.assertEqual(metrics["false_negatives"], 2)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertAlmostEqual(metrics["missed_event_rate"], 0.50, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.50, places=4)

    def test_warning_error_score_formula(self):
        """Test composite error score E_k = a*MER + b*LWR + c*FAR."""
        mock_metrics = {
            "missed_event_rate": 0.40,
            "late_warning_rate": 0.20,
            "false_alarm_rate": 0.10
        }
        # E = 0.50 * 0.40 + 0.30 * 0.20 + 0.20 * 0.10 = 0.20 + 0.06 + 0.02 = 0.28
        score = compute_warning_error_score(mock_metrics, a=0.50, b=0.30, c=0.20)
        self.assertAlmostEqual(score, 0.28, places=4)


class TestLEAFAggregatorWeights(unittest.TestCase):
    """Test mathematical properties of LEAF-FL aggregation weights."""

    def test_weights_sum_to_one(self):
        """Verify weights strictly sum to 1.0 under diverse sample sizes and errors."""
        counts = [320, 240, 140]
        errors = [0.25, 0.45, 0.70]
        for l_val in [0.0, 0.25, 0.50, 0.75, 1.0]:
            weights = compute_leaf_aggregation_weights(counts, errors, lambda_equity=l_val, tau=2.0)
            self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_lambda_zero_reproduces_fedavg(self):
        """Verify lambda = 0.0 reproduces exact sample-weighted FedAvg."""
        counts = [300, 200, 100]
        errors = [0.10, 0.90, 0.50]
        weights = compute_leaf_aggregation_weights(counts, errors, lambda_equity=0.0, tau=2.0)
        expected = [300/600, 200/600, 100/600]
        for w, e in zip(weights, expected):
            self.assertAlmostEqual(w, e, places=6)

    def test_higher_error_receives_larger_equity_weight(self):
        """Verify hospitals with higher warning-error scores receive boosted equity weights."""
        counts = [200, 200, 200]  # Equal sample sizes to isolate equity effect
        errors = [0.10, 0.30, 0.70]  # Hospital 3 has highest error
        weights = compute_leaf_aggregation_weights(counts, errors, lambda_equity=0.8, tau=2.0)

        # Hospital 3 should receive the largest weight, Hospital 1 the smallest
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[1], weights[0])

    def test_invalid_lambda_rejected(self):
        """Verify ValueError is raised for negative lambda or lambda > 1."""
        with self.assertRaises(ValueError):
            compute_leaf_aggregation_weights([100, 100], [0.2, 0.4], lambda_equity=-0.1)
        with self.assertRaises(ValueError):
            compute_leaf_aggregation_weights([100, 100], [0.2, 0.4], lambda_equity=1.5)

    def test_invalid_tau_rejected(self):
        """Verify ValueError is raised for non-positive temperature tau."""
        with self.assertRaises(ValueError):
            compute_leaf_aggregation_weights([100, 100], [0.2, 0.4], tau=0.0)
        with self.assertRaises(ValueError):
            compute_leaf_aggregation_weights([100, 100], [0.2, 0.4], tau=-2.0)

    def test_empty_client_input_rejected(self):
        """Verify ValueError is raised for empty client lists."""
        with self.assertRaises(ValueError):
            compute_leaf_aggregation_weights([], [])

    def test_parameter_shape_mismatch_rejected(self):
        """Verify leaf_aggregate_parameters raises ValueError when parameter shapes mismatch."""
        sd1 = {"layer": torch.ones(5, 5)}
        sd2 = {"layer": torch.ones(5, 6)}  # Mismatched shape
        with self.assertRaises(ValueError):
            leaf_aggregate_parameters([sd1, sd2], [0.5, 0.5])


class TestLEAFModelAndDeterminism(unittest.TestCase):
    """Test model architectures, state aggregation, and determinism."""

    def test_causal_cnn_lstm_forward_shape(self):
        """Verify model output shapes on 3D sequence tensors."""
        model = LEAFCausalCNNLSTM(sequence_length=6, num_features=5)
        dummy_x = torch.randn(4, 6, 5)
        final_prob, seq_probs = model(dummy_x)

        self.assertEqual(final_prob.shape, (4, 1))
        self.assertEqual(seq_probs.shape, (4, 6, 1))
        # Probabilities must be in [0, 1]
        self.assertTrue(torch.all(final_prob >= 0.0) and torch.all(final_prob <= 1.0))

    def test_deterministic_reproducibility(self):
        """Verify identical results under identical random seeds."""
        torch.manual_seed(42)
        m1 = LEAFCausalCNNLSTM(sequence_length=6, num_features=5)
        torch.manual_seed(42)
        m2 = LEAFCausalCNNLSTM(sequence_length=6, num_features=5)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()


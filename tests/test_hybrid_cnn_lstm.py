"""
Unit Tests for Phase 14: Hybrid CNN-LSTM for Federated Healthcare Analytics.

Validates:
1. Synthetic longitudinal data generation, temporal dynamics, and feature dimensions.
2. Patient-level partition isolation (strict zero leakage between train/val/test).
3. Temporal feature scaling fit exclusively on training sequences.
4. Non-IID federated client partitioning across 3 simulated hospitals.
5. Neural network architectures (MLP, 1D CNN, LSTM, Hybrid CNN-LSTM) forward passes, shapes, and parameter counts.
6. Training and evaluation routines with metric contracts.
7. Sample-weighted state_dict federated aggregation.
8. Communication efficiency calculations and comparative ratios.
9. Firebase Cloud Firestore experiment record builder with strict privacy guarantees.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.temporal_data import (
    generate_synthetic_longitudinal_data,
    compute_temporal_scaler,
    apply_temporal_scaler,
    partition_temporal_dataset,
    partition_federated_clients,
    TEMPORAL_FEATURE_COLUMNS
)

from src.hybrid_cnn_lstm import (
    TemporalMLPBaseline,
    TemporalCNNModel,
    TemporalLSTMModel,
    HybridCNNLSTMModel,
    count_model_parameters,
    get_model_summary,
    train_temporal_model,
    evaluate_temporal_model,
    aggregate_state_dicts
)

from src.communication_analysis import calculate_cnn_lstm_communication_volume
from src.firebase_service import build_cnn_lstm_experiment_record


class TestTemporalData(unittest.TestCase):
    """Test suite for longitudinal patient data generation and partitioning."""

    def setUp(self):
        self.num_patients = 100
        self.seq_len = 12
        self.raw = generate_synthetic_longitudinal_data(
            num_patients=self.num_patients,
            sequence_length=self.seq_len,
            random_seed=42
        )

    def test_synthetic_data_shapes_and_features(self):
        """Verify 3D tensor shape (N, T, F), binary label shape (N, 1), and metadata."""
        X = self.raw["X"]
        y = self.raw["y"]
        pids = self.raw["patient_ids"]

        self.assertEqual(X.shape, (self.num_patients, self.seq_len, 5))
        self.assertEqual(y.shape, (self.num_patients, 1))
        self.assertEqual(len(pids), self.num_patients)
        self.assertTrue(set(np.unique(y)).issubset({0.0, 1.0}))
        self.assertEqual(self.raw["metadata"]["num_features"], 5)
        self.assertEqual(self.raw["metadata"]["feature_names"], TEMPORAL_FEATURE_COLUMNS)

    def test_age_remains_constant_across_sequence(self):
        """Verify patient age is invariant across observation time steps."""
        X = self.raw["X"]
        for i in range(self.num_patients):
            age_seq = X[i, :, 0]
            self.assertEqual(len(np.unique(age_seq)), 1, f"Patient {i} age varied over time.")

    def test_physiological_features_vary_over_time(self):
        """Verify dynamic features (BP, glucose, BMI, cholesterol) exhibit temporal variation."""
        X = self.raw["X"]
        # Check across the cohort that BP and glucose are not static
        bp_stds = np.std(X[:, :, 1], axis=1)
        glu_stds = np.std(X[:, :, 2], axis=1)
        self.assertGreater(np.mean(bp_stds), 0.5)
        self.assertGreater(np.mean(glu_stds), 0.5)

    def test_patient_level_partitioning_zero_leakage(self):
        """Verify train, validation, and test partitions have zero overlapping patient IDs."""
        splits = partition_temporal_dataset(
            self.raw,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )

        train_pids = set(splits["train"]["patient_ids"])
        val_pids = set(splits["val"]["patient_ids"])
        test_pids = set(splits["test"]["patient_ids"])

        self.assertEqual(len(train_pids & val_pids), 0, "Train and Val share patient IDs!")
        self.assertEqual(len(train_pids & test_pids), 0, "Train and Test share patient IDs!")
        self.assertEqual(len(val_pids & test_pids), 0, "Val and Test share patient IDs!")
        self.assertEqual(len(train_pids) + len(val_pids) + len(test_pids), self.num_patients)

    def test_temporal_scaler_fitted_on_train_only(self):
        """Verify scaler parameters are extracted and applied correctly without leaking validation data."""
        splits = partition_temporal_dataset(self.raw, random_seed=42)
        scaler = compute_temporal_scaler(splits["train"]["X"])

        self.assertIn("mean", scaler)
        self.assertIn("std", scaler)
        for col in TEMPORAL_FEATURE_COLUMNS:
            self.assertIn(col, scaler["mean"])
            self.assertIn(col, scaler["std"])

        X_norm = apply_temporal_scaler(splits["train"]["X"], scaler)
        self.assertEqual(X_norm.shape, splits["train"]["X"].shape)
        # Scaled train means should be close to 0
        means = np.mean(X_norm.reshape(-1, 5), axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_partition_federated_clients_non_iid(self):
        """Verify 3 clients partition training records with disjoint patient sets."""
        splits = partition_temporal_dataset(self.raw, random_seed=42)
        clients = partition_federated_clients(splits["train"], num_clients=3, random_seed=42)

        self.assertEqual(set(clients.keys()), {"hospital_1", "hospital_2", "hospital_3"})
        h1_pids = set(clients["hospital_1"]["patient_ids"])
        h2_pids = set(clients["hospital_2"]["patient_ids"])
        h3_pids = set(clients["hospital_3"]["patient_ids"])

        self.assertEqual(len(h1_pids & h2_pids), 0)
        self.assertEqual(len(h1_pids & h3_pids), 0)
        self.assertEqual(len(h2_pids & h3_pids), 0)
        self.assertEqual(len(h1_pids) + len(h2_pids) + len(h3_pids), len(splits["train"]["patient_ids"]))


class TestHybridCNNLSTMArchitectures(unittest.TestCase):
    """Test suite for neural network architectures, layer shapes, and parameter counts."""

    def setUp(self):
        self.batch_size = 8
        self.seq_len = 12
        self.num_features = 5
        self.dummy_input = torch.randn(self.batch_size, self.seq_len, self.num_features)

    def test_temporal_mlp_baseline(self):
        """Verify TemporalMLPBaseline forward pass and parameter count."""
        model = TemporalMLPBaseline(
            sequence_length=self.seq_len,
            num_features=self.num_features,
            hidden_dim=16
        )
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 1))
        self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))

        # Expected: 60*16 + 16 (fc1) + 16*1 + 1 (fc2) = 976 + 17 = 993
        params = count_model_parameters(model)
        self.assertEqual(params, 993)

    def test_temporal_cnn_model(self):
        """Verify TemporalCNNModel forward pass and parameter count."""
        model = TemporalCNNModel(
            num_features=self.num_features,
            conv_filters=16,
            kernel_size=3
        )
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 1))
        self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))

        # Expected: Conv1d(5, 16, 3) = 16*5*3 + 16 = 256; fc = 16*1 + 1 = 17; total = 273
        params = count_model_parameters(model)
        self.assertEqual(params, 273)

    def test_temporal_lstm_model(self):
        """Verify TemporalLSTMModel forward pass and parameter count."""
        model = TemporalLSTMModel(
            num_features=self.num_features,
            lstm_units=16
        )
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 1))
        self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))

        # LSTM(5, 16): 4 * (5*16 + 16*16 + 16 + 16) = 4 * (80 + 256 + 32) = 4 * 368 = 1472?
        # In PyTorch LSTM with bias, input and hidden biases are separate: 4*(in_gate + hidden_gate)
        # 4 * (5*16 + 16*16 + 16 + 16) = 1,472 + fc(16*1 + 1 = 17) = 1,489
        params = count_model_parameters(model)
        self.assertEqual(params, 1489)

    def test_hybrid_cnn_lstm_model(self):
        """Verify HybridCNNLSTMModel forward pass, output shape, and structural summary."""
        model = HybridCNNLSTMModel(
            sequence_length=self.seq_len,
            num_features=self.num_features,
            conv_filters=16,
            kernel_size=3,
            lstm_units=16
        )
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 1))
        self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))

        # Conv1d: 256, LSTM(16, 16): 4*(16*16 + 16*16 + 16 + 16) = 4*(256+256+32) = 2,176, FC: 17 -> total = 2,449
        params = count_model_parameters(model)
        self.assertEqual(params, 2449)

        summary = get_model_summary(model, input_shape=(1, 12, 5))
        self.assertEqual(summary["model_type"], "HybridCNNLSTMModel")
        self.assertEqual(summary["total_parameters"], 2449)
        self.assertEqual(summary["parameter_bytes"], 2449 * 4)
        self.assertAlmostEqual(summary["parameter_kilobytes"], (2449 * 4) / 1024.0, places=3)
        self.assertGreater(len(summary["layers"]), 0)


class TestTrainingAndFederation(unittest.TestCase):
    """Test suite for training routines, evaluation metrics, and state_dict aggregation."""

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.X_train = np.random.randn(50, 12, 5).astype(np.float32)
        self.y_train = np.random.randint(0, 2, size=(50, 1)).astype(np.float32)
        self.X_test = np.random.randn(20, 12, 5).astype(np.float32)
        self.y_test = np.random.randint(0, 2, size=(20, 1)).astype(np.float32)

    def test_train_temporal_model_execution(self):
        """Verify model trains over epochs and returns history with finite loss."""
        model = HybridCNNLSTMModel(num_features=5, conv_filters=8, lstm_units=8)
        res = train_temporal_model(
            model=model,
            X_train=self.X_train,
            y_train=self.y_train,
            epochs=3,
            batch_size=16,
            lr=0.01,
            random_seed=42
        )
        self.assertEqual(len(res["history"]), 3)
        self.assertTrue(np.isfinite(res["final_train_loss"]))

    def test_evaluate_temporal_model_contract(self):
        """Verify evaluation dictionary contains expected metric keys and valid bounds."""
        model = HybridCNNLSTMModel(num_features=5, conv_filters=8, lstm_units=8)
        metrics = evaluate_temporal_model(model, self.X_test, self.y_test)

        expected_keys = [
            "loss", "accuracy", "precision", "recall",
            "f1_score", "roc_auc", "confusion_matrix",
            "probabilities", "predictions"
        ]
        for k in expected_keys:
            self.assertIn(k, metrics)

        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["precision"] <= 1.0)
        self.assertTrue(0.0 <= metrics["recall"] <= 1.0)
        self.assertTrue(0.0 <= metrics["f1_score"] <= 1.0)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        self.assertEqual(len(metrics["confusion_matrix"][0]), 2)

    def test_aggregate_state_dicts_sample_weighted(self):
        """Verify sample-weighted FedAvg calculates weighted average over PyTorch state_dicts."""
        m1 = HybridCNNLSTMModel(num_features=5, conv_filters=8, lstm_units=8)
        m2 = HybridCNNLSTMModel(num_features=5, conv_filters=8, lstm_units=8)

        # Set specific weight values
        with torch.no_grad():
            for p in m1.parameters():
                p.fill_(1.0)
            for p in m2.parameters():
                p.fill_(3.0)

        # 100 samples at m1, 300 samples at m2 -> weights: 0.25 and 0.75
        # Expected value: 0.25*1.0 + 0.75*3.0 = 0.25 + 2.25 = 2.50
        aggregated = aggregate_state_dicts(
            [m1.state_dict(), m2.state_dict()],
            [100, 300]
        )

        for k, v in aggregated.items():
            if v.is_floating_point():
                np.testing.assert_allclose(v.numpy(), 2.50, atol=1e-5)


class TestCommunicationAndFirebase(unittest.TestCase):
    """Test suite for communication analysis and Firebase Cloud Firestore payload generation."""

    def test_calculate_cnn_lstm_communication_volume(self):
        """Verify communication volume arithmetic for CNN-LSTM."""
        comm = calculate_cnn_lstm_communication_volume(
            num_parameters=2449,
            num_clients=3,
            num_rounds=5,
            bytes_per_parameter=4
        )

        self.assertEqual(comm["num_parameters"], 2449)
        self.assertEqual(comm["single_model_bytes"], 2449 * 4)  # 9,796
        # Round bytes = (3 upload + 3 download) * 9,796 = 6 * 9,796 = 58,776
        self.assertEqual(comm["total_round_bytes"], 58776)
        # Total for 5 rounds = 58,776 * 5 = 293,880
        self.assertEqual(comm["total_communication_bytes"], 293880)
        self.assertAlmostEqual(comm["total_communication_kb"], 293880 / 1024.0, places=1)
        self.assertIn("comparison_with_tabular_mlp", comm)

    def test_calculate_cnn_lstm_communication_validation(self):
        """Verify invalid parameter errors."""
        with self.assertRaises(ValueError):
            calculate_cnn_lstm_communication_volume(num_parameters=0)
        with self.assertRaises(ValueError):
            calculate_cnn_lstm_communication_volume(num_clients=-1)
        with self.assertRaises(ValueError):
            calculate_cnn_lstm_communication_volume(num_rounds=0)

    def test_firebase_cnn_lstm_payload_privacy_enforcement(self):
        """Verify Firebase payload strictly enforces privacy guarantees without patient leakage."""
        record = build_cnn_lstm_experiment_record()

        self.assertEqual(record["algorithm"], "Hybrid CNN-LSTM")
        self.assertIn("privacy_guarantees", record)
        self.assertFalse(record["privacy_guarantees"]["raw_patient_sequences_persisted"])
        self.assertFalse(record["privacy_guarantees"]["patient_level_predictions_persisted"])
        self.assertFalse(record["privacy_guarantees"]["intermediate_feature_maps_persisted"])
        self.assertFalse(record["privacy_guarantees"]["intermediate_gradients_persisted"])
        self.assertFalse(record["privacy_guarantees"]["patient_ids_transmitted"])

        # Confirm no raw patient arrays in top keys
        for bad_key in ["raw_sequences", "patient_ids", "patient_features", "activations"]:
            self.assertNotIn(bad_key, record)


if __name__ == "__main__":
    unittest.main()


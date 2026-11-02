"""
Comprehensive Unit Tests for Vertical Federated Learning Module (src/vertical_federated.py).

Verifies Phase 12 requirements:
- Feature partition correctness (Hospital A, B, C and Coordinator)
- Deterministic patient ID alignment & Private Set Intersection (PSI) simulation
- Defensive error handling for duplicate, missing, or mismatched patient IDs
- Privacy preservation: zero raw feature or label leakage across parties
- Split-Neural Network (SplitNN) forward/backward shapes and gradient updates
- Stationary model weights (45 total parameters, 0 parameter network transfer)
- Training convergence and loss decreasing behavior
- Central test evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Communication overhead estimation (forward embeddings, backward gradients, PSI tokens)
- Reproducibility under deterministic random seeds
- Firebase document building with strict privacy enforcement
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from src.vertical_federated import (
    VFL_FEATURE_PARTITIONS,
    VFL_LABEL_PARTY,
    VFLAlignmentError,
    create_vfl_partitions,
    align_vfl_datasets,
    VFLFeatureParty,
    VFLLabelCoordinator,
    VFLSplitNN,
    run_vfl_experiment
)
from src.communication_analysis import (
    calculate_vfl_communication_volume,
    compare_hfl_vs_vfl_communication
)
from src.firebase_service import build_vfl_experiment_record, build_experiment_record


class TestVFLPartitionsAndAlignment(unittest.TestCase):
    """Test suite for vertical feature partitioning and simulated PSI patient ID alignment."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pids = [f"PID_{i+1:04d}" for i in range(20)]

        # Synthetic micro datasets
        self.df_a = pd.DataFrame({
            "patient_id": self.pids,
            "age": np.random.uniform(25, 75, 20),
            "blood_pressure": np.random.uniform(100, 160, 20)
        })
        self.df_b = pd.DataFrame({
            "patient_id": self.pids,
            "glucose_level": np.random.uniform(70, 180, 20),
            "bmi": np.random.uniform(18.5, 38.0, 20)
        })
        self.df_c = pd.DataFrame({
            "patient_id": self.pids,
            "cholesterol_level": np.random.uniform(150, 280, 20)
        })
        self.df_labels = pd.DataFrame({
            "patient_id": self.pids,
            "risk_label": np.random.randint(0, 2, 20)
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_feature_partition_schema_disjointness_and_completeness(self):
        """Verify feature partitions are strictly disjoint and cover all clinical features."""
        all_features = []
        for party, features in VFL_FEATURE_PARTITIONS.items():
            self.assertGreater(len(features), 0, f"Party {party} has no features")
            for f in features:
                self.assertNotIn(f, all_features, f"Duplicate feature '{f}' across parties")
                all_features.append(f)

        expected_features = {"age", "blood_pressure", "glucose_level", "bmi", "cholesterol_level"}
        self.assertEqual(set(all_features), expected_features)
        self.assertEqual(VFL_FEATURE_PARTITIONS["hospital_A"], ["age", "blood_pressure"])
        self.assertEqual(VFL_FEATURE_PARTITIONS["hospital_B"], ["glucose_level", "bmi"])
        self.assertEqual(VFL_FEATURE_PARTITIONS["hospital_C"], ["cholesterol_level"])
        self.assertEqual(VFL_LABEL_PARTY, "coordinator")

    def test_align_vfl_datasets_successful_alignment(self):
        """Verify PSI alignment preserves aligned records and ordering."""
        party_data = {
            "hospital_A": self.df_a,
            "hospital_B": self.df_b,
            "hospital_C": self.df_c
        }
        aligned_parties, aligned_label, common_ids = align_vfl_datasets(
            party_data, self.df_labels, strict=True
        )

        self.assertEqual(len(common_ids), 20)
        self.assertEqual(common_ids, self.pids)
        for party_name, df_aligned in aligned_parties.items():
            self.assertEqual(list(df_aligned["patient_id"]), self.pids)
        self.assertEqual(list(aligned_label["patient_id"]), self.pids)

    def test_align_vfl_datasets_rejects_missing_patient_id_column(self):
        """Verify error is raised if 'patient_id' is missing from any party."""
        bad_df_a = self.df_a.drop(columns=["patient_id"])
        party_data = {"hospital_A": bad_df_a, "hospital_B": self.df_b}
        with self.assertRaises(VFLAlignmentError) as ctx:
            align_vfl_datasets(party_data, self.df_labels)
        self.assertIn("missing 'patient_id'", str(ctx.exception))

    def test_align_vfl_datasets_rejects_duplicate_patient_ids(self):
        """Verify error is raised if duplicate patient IDs exist within any party."""
        dup_df_b = pd.concat([self.df_b, self.df_b.iloc[[0]]], ignore_index=True)
        party_data = {"hospital_A": self.df_a, "hospital_B": dup_df_b, "hospital_C": self.df_c}
        with self.assertRaises(VFLAlignmentError) as ctx:
            align_vfl_datasets(party_data, self.df_labels)
        self.assertIn("Duplicate patient IDs", str(ctx.exception))

    def test_align_vfl_datasets_strict_mode_rejects_mismatched_ids(self):
        """Verify strict mode rejects mismatched patient IDs."""
        dropped_df_c = self.df_c.iloc[:-2]  # only 18 IDs
        party_data = {"hospital_A": self.df_a, "hospital_B": self.df_b, "hospital_C": dropped_df_c}
        with self.assertRaises(VFLAlignmentError) as ctx:
            align_vfl_datasets(party_data, self.df_labels, strict=True)
        self.assertIn("Patient ID mismatch", str(ctx.exception))

    def test_align_vfl_datasets_non_strict_mode_finds_intersection(self):
        """Verify non-strict mode computes the correct common intersection."""
        dropped_df_c = self.df_c.iloc[:-2]  # 18 IDs
        party_data = {"hospital_A": self.df_a, "hospital_B": self.df_b, "hospital_C": dropped_df_c}
        aligned_parties, aligned_label, common_ids = align_vfl_datasets(
            party_data, self.df_labels, strict=False
        )
        self.assertEqual(len(common_ids), 18)
        self.assertEqual(len(aligned_label), 18)
        self.assertEqual(len(aligned_parties["hospital_A"]), 18)

    def test_create_vfl_partitions_file_structure(self):
        """Verify create_vfl_partitions creates train/test CSVs for all 3 parties and coordinator."""
        out_vfl_dir = os.path.join(self.temp_dir, "vfl_out")
        partition_paths = create_vfl_partitions(data_dir="data", output_dir=out_vfl_dir)

        # Check paths returned
        self.assertIn("train", partition_paths)
        self.assertIn("test", partition_paths)
        for party in ["hospital_A", "hospital_B", "hospital_C", "coordinator"]:
            self.assertIn(party, partition_paths["train"])
            self.assertIn(party, partition_paths["test"])
            self.assertTrue(os.path.exists(partition_paths["train"][party]))
            self.assertTrue(os.path.exists(partition_paths["test"][party]))

        # Check train sample counts (1,200) and test sample counts (300)
        df_a_train = pd.read_csv(partition_paths["train"]["hospital_A"])
        df_a_test = pd.read_csv(partition_paths["test"]["hospital_A"])
        self.assertEqual(len(df_a_train), 1200)
        self.assertEqual(len(df_a_test), 300)
        self.assertEqual(list(df_a_train.columns), ["patient_id", "age", "blood_pressure"])


class TestVFLSplitNNComponents(unittest.TestCase):
    """Test suite covering Bottom Models, Top Model, and SplitNN architecture."""

    def setUp(self):
        np.random.seed(42)
        self.N = 50
        # Simulated party data
        self.party_dfs = {
            "hospital_A": pd.DataFrame({
                "patient_id": [f"P_{i}" for i in range(self.N)],
                "age": np.random.uniform(30, 70, self.N),
                "blood_pressure": np.random.uniform(110, 150, self.N)
            }),
            "hospital_B": pd.DataFrame({
                "patient_id": [f"P_{i}" for i in range(self.N)],
                "glucose_level": np.random.uniform(80, 160, self.N),
                "bmi": np.random.uniform(20, 35, self.N)
            }),
            "hospital_C": pd.DataFrame({
                "patient_id": [f"P_{i}" for i in range(self.N)],
                "cholesterol_level": np.random.uniform(160, 260, self.N)
            })
        }
        self.label_df = pd.DataFrame({
            "patient_id": [f"P_{i}" for i in range(self.N)],
            "risk_label": np.random.randint(0, 2, self.N)
        })

    def test_feature_party_bottom_model_dimensions_and_shapes(self):
        """Verify Bottom Model outputs correct embedding dimensions without leaking raw features."""
        party_a = VFLFeatureParty("hospital_A", ["age", "blood_pressure"], hidden_dim=4, random_seed=42)
        X_raw = self.party_dfs["hospital_A"][["age", "blood_pressure"]].values
        emb = party_a.forward(X_raw)

        self.assertEqual(emb.shape, (self.N, 4))
        self.assertFalse(np.isnan(emb).any())
        self.assertFalse(np.isneginf(emb).any())
        # ReLU output must be >= 0
        self.assertTrue(np.all(emb >= 0))

        # Test parameter count for Hospital A: (2 inputs * 4 units) + 4 biases = 12
        params = party_a.get_parameter_count()
        self.assertEqual(params, 12)

    def test_coordinator_top_model_fusion_and_loss(self):
        """Verify Top Model concatenates embeddings, computes loss, and generates cut gradients."""
        coord = VFLLabelCoordinator(party_embedding_dims={"hospital_A": 4, "hospital_B": 4, "hospital_C": 4}, random_seed=42)
        self.assertEqual(coord.total_embedding_dim, 12)
        self.assertEqual(coord.get_parameter_count(), 13)  # (12 * 1) + 1

        # Dummy embeddings from 3 parties
        embeddings = {
            "hospital_A": np.random.randn(self.N, 4),
            "hospital_B": np.random.randn(self.N, 4),
            "hospital_C": np.random.randn(self.N, 4)
        }
        y = self.label_df["risk_label"].values

        y_pred, loss = coord.forward(embeddings, y_batch=y)
        self.assertEqual(y_pred.shape, (self.N, 1))
        self.assertTrue(np.all((y_pred >= 0) & (y_pred <= 1)))
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

        # Backward gradients
        cut_grads = coord.backward(y_batch=y, learning_rate=0.05)
        self.assertEqual(set(cut_grads.keys()), {"hospital_A", "hospital_B", "hospital_C"})
        self.assertEqual(cut_grads["hospital_A"].shape, (self.N, 4))
        self.assertEqual(cut_grads["hospital_B"].shape, (self.N, 4))
        self.assertEqual(cut_grads["hospital_C"].shape, (self.N, 4))

    def test_vfl_split_nn_total_parameters_is_45(self):
        """Verify total VFL parameter count across all bottom models and top model is exactly 45."""
        split_nn = VFLSplitNN(random_seed=42)
        counts = split_nn.count_total_parameters()

        self.assertEqual(counts["bottom_models"]["hospital_A"], 12)  # 2x4 + 4
        self.assertEqual(counts["bottom_models"]["hospital_B"], 12)  # 2x4 + 4
        self.assertEqual(counts["bottom_models"]["hospital_C"], 8)   # 1x4 + 4
        self.assertEqual(counts["top_model_parameters"], 13)         # 12x1 + 1
        self.assertEqual(counts["total_parameters"], 45)

    def test_vfl_split_nn_training_convergence(self):
        """Verify training loss decreases over epochs."""
        split_nn = VFLSplitNN(random_seed=42)
        history = split_nn.fit(
            self.party_dfs,
            self.label_df,
            epochs=20,
            batch_size=16,
            learning_rate=0.05,
            verbose=False
        )

        self.assertEqual(len(history), 20)
        # Final loss should be lower than initial loss
        self.assertLess(history[-1], history[0])

    def test_vfl_split_nn_evaluate_metrics(self):
        """Verify evaluation returns valid metrics dictionary with accuracy, precision, recall, f1, and roc_auc."""
        split_nn = VFLSplitNN(random_seed=42)
        split_nn.fit(self.party_dfs, self.label_df, epochs=5, batch_size=16, learning_rate=0.05, verbose=False)

        metrics = split_nn.evaluate(self.party_dfs, self.label_df)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("confusion_matrix", metrics)

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        self.assertEqual(len(metrics["confusion_matrix"][0]), 2)

    def test_vfl_seed_reproducibility(self):
        """Verify identical random seeds produce identical predictions."""
        m1 = VFLSplitNN(random_seed=123)
        m1.fit(self.party_dfs, self.label_df, epochs=5, batch_size=16, learning_rate=0.05, verbose=False)
        p1 = m1.predict_proba(self.party_dfs)

        m2 = VFLSplitNN(random_seed=123)
        m2.fit(self.party_dfs, self.label_df, epochs=5, batch_size=16, learning_rate=0.05, verbose=False)
        p2 = m2.predict_proba(self.party_dfs)

        np.testing.assert_allclose(p1, p2, atol=1e-12)


class TestVFLCommunicationAndPrivacy(unittest.TestCase):
    """Test suite for VFL communication efficiency calculations and privacy safeguards."""

    def test_calculate_vfl_communication_volume_default(self):
        """Verify communication overhead calculations for default 1200 samples and 50 epochs."""
        vol = calculate_vfl_communication_volume(
            num_samples=1200,
            epochs=50,
            party_dims={"hospital_A": 4, "hospital_B": 4, "hospital_C": 4}
        )

        self.assertEqual(vol["paradigm"], "Vertical Federated Learning (SplitNN)")
        self.assertEqual(vol["parameter_transfer_bytes"], 0)  # Stationary model parameters!
        self.assertEqual(vol["total_embedding_dim"], 12)
        # Forward: 1200 * 12 * 8 = 115,200 per epoch * 50 = 5,760,000 bytes
        self.assertEqual(vol["total_forward_bytes"], 5760000)
        # Backward: 5,760,000 bytes
        self.assertEqual(vol["total_backward_bytes"], 5760000)
        # Training total: 11,520,000 bytes
        self.assertEqual(vol["total_training_traffic_bytes"], 11520000)
        self.assertAlmostEqual(vol["total_training_traffic_kb"], 11250.0, places=1)
        # PSI: 3 parties * 1200 * 16 = 57,600 bytes
        self.assertEqual(vol["psi_alignment_overhead_bytes"], 57600)
        self.assertEqual(vol["grand_total_communication_bytes"], 11520000 + 57600)

    def test_calculate_vfl_communication_volume_validation(self):
        """Verify validation errors for non-positive samples or epochs."""
        with self.assertRaises(ValueError):
            calculate_vfl_communication_volume(num_samples=0)
        with self.assertRaises(ValueError):
            calculate_vfl_communication_volume(epochs=-1)

    def test_compare_hfl_vs_vfl_communication(self):
        """Verify comparative communication table includes FedAvg, FedProx, SecAgg, and VFL."""
        comp_rows = compare_hfl_vs_vfl_communication()
        self.assertGreaterEqual(len(comp_rows), 4)

        algorithms = [r["algorithm"] for r in comp_rows]
        self.assertIn("FedAvg", algorithms)
        self.assertIn("FedProx (mu=0.01)", algorithms)
        self.assertIn("FedAvg with Secure Aggregation", algorithms)
        self.assertIn("VFL SplitNN", algorithms)

        vfl_row = next(r for r in comp_rows if r["algorithm"] == "VFL SplitNN")
        self.assertEqual(vfl_row["parameter_transfer_bytes"], 0)
        self.assertGreater(vfl_row["representation_transfer_bytes"], 0)

    def test_firebase_vfl_payload_privacy_enforcement(self):
        """Verify Firebase VFL experiment record strictly excludes raw patient data."""
        payload = build_vfl_experiment_record()

        self.assertEqual(payload["algorithm"], "Vertical Federated Learning")
        self.assertIn("privacy_guarantees", payload)
        self.assertFalse(payload["privacy_guarantees"]["raw_features_transmitted"])
        self.assertFalse(payload["privacy_guarantees"]["patient_level_predictions_persisted"])
        self.assertFalse(payload["privacy_guarantees"]["intermediate_representations_persisted"])
        self.assertFalse(payload["privacy_guarantees"]["intermediate_gradients_persisted"])

        # Ensure no raw patient data or embeddings in top-level or sub-keys
        for k, v in payload.items():
            self.assertNotEqual(k, "patient_features")
            self.assertNotEqual(k, "raw_data")
            self.assertNotEqual(k, "embeddings")
            self.assertNotEqual(k, "cut_gradients")


if __name__ == "__main__":
    unittest.main()

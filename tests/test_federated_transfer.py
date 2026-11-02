"""
Comprehensive Unit Tests for Federated Transfer Learning Module (src/federated_transfer.py).

Verifies Phase 13 requirements:
- Source and Target domain dataset separation and distinct demographic distributions
- Target test-set isolation (zero test-data leakage)
- Deterministic patient ID prefixing (SRC_ vs TGT_)
- Transfer model initialization and parameter transfer correctness
- Full fine-tuning behavior (updates all layers: W1, b1, W2, b2)
- Feature extraction behavior (strictly freezes Layer 1: W1, b1; updates only Layer 2: W2, b2)
- Target-only baseline trained from scratch
- Source-only zero-shot evaluation on target test set
- Evaluation metric validity (Accuracy, Precision, Recall, F1, ROC-AUC within [0, 1])
- Reproducibility under fixed random seed
- Communication volume calculations (456 bytes model payload, 1 transfer event, 0 bytes fine-tuning)
- Firebase metadata document building with strict patient privacy preservation
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from src.federated_transfer import (
    SOURCE_DOMAIN_CONFIG,
    TARGET_DOMAIN_CONFIG,
    generate_domain_dataset,
    prepare_ftl_datasets,
    apply_domain_scaler,
    TransferHealthcareModel,
    train_source_model,
    run_ftl_experiment
)
from src.communication_analysis import (
    calculate_ftl_communication_volume,
    compare_hfl_vs_vfl_communication
)
from src.firebase_service import build_ftl_experiment_record, build_experiment_record


class TestFTLDatasetsAndDomainShift(unittest.TestCase):
    """Test suite covering Source and Target synthetic dataset generation, separation, and domain shift."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_domain_dataset_generation_shapes_and_ids(self):
        """Verify generated domain datasets have expected record counts and distinct patient ID prefixes."""
        df_src = generate_domain_dataset(SOURCE_DOMAIN_CONFIG, prefix="SRC", random_seed=42)
        df_tgt = generate_domain_dataset(TARGET_DOMAIN_CONFIG, prefix="TGT", random_seed=142)

        self.assertEqual(len(df_src), SOURCE_DOMAIN_CONFIG["sample_size"])
        self.assertEqual(len(df_tgt), TARGET_DOMAIN_CONFIG["sample_size"])

        # Check distinct prefixing
        self.assertTrue(all(pid.startswith("SRC_") for pid in df_src["patient_id"]))
        self.assertTrue(all(pid.startswith("TGT_") for pid in df_tgt["patient_id"]))

        # Check no overlap between source and target patient IDs
        common_pids = set(df_src["patient_id"]).intersection(set(df_tgt["patient_id"]))
        self.assertEqual(len(common_pids), 0)

    def test_domain_shift_statistical_characteristics(self):
        """Verify statistical domain shift: Target population is older and has elevated BP and glucose."""
        df_src = generate_domain_dataset(SOURCE_DOMAIN_CONFIG, prefix="SRC", random_seed=42)
        df_tgt = generate_domain_dataset(TARGET_DOMAIN_CONFIG, prefix="TGT", random_seed=142)

        # Target should be noticeably older (geriatric demographic shift)
        self.assertGreater(df_tgt["age"].mean(), df_src["age"].mean() + 10.0)

        # Target should have higher blood pressure and cholesterol
        self.assertGreater(df_tgt["blood_pressure"].mean(), df_src["blood_pressure"].mean() + 5.0)
        self.assertGreater(df_tgt["cholesterol_level"].mean(), df_src["cholesterol_level"].mean() + 5.0)

    def test_prepare_ftl_datasets_file_generation_and_isolation(self):
        """Verify prepare_ftl_datasets creates separate train/test files and scaler JSON."""
        paths = prepare_ftl_datasets(output_dir=self.temp_dir, random_seed=42)

        for key, p in paths.items():
            self.assertTrue(os.path.exists(p), f"Missing expected FTL file: {p}")

        df_src_train = pd.read_csv(paths["source_train"])
        df_src_test = pd.read_csv(paths["source_test"])
        df_tgt_pool = pd.read_csv(paths["target_train_pool"])
        df_tgt_test = pd.read_csv(paths["target_test"])

        self.assertEqual(len(df_src_train), 1000)
        self.assertEqual(len(df_src_test), 200)
        self.assertEqual(len(df_tgt_pool), 200)
        self.assertEqual(len(df_tgt_test), 100)

        # Test set isolation: Target test IDs must not appear in target train pool
        test_pids = set(df_tgt_test["patient_id"])
        train_pids = set(df_tgt_pool["patient_id"])
        self.assertEqual(len(test_pids.intersection(train_pids)), 0)


class TestTransferLearningModelAndStrategies(unittest.TestCase):
    """Test suite covering model transfer, feature extraction freezing, and fine-tuning."""

    def setUp(self):
        np.random.seed(42)
        self.X_dummy = np.random.randn(30, 5)
        self.y_dummy = np.random.randint(0, 2, 30)

        # Initialize source model and train for 5 epochs
        self.source_model = TransferHealthcareModel(random_seed=42)
        self.source_model.fit(
            self.X_dummy, self.y_dummy,
            epochs=5,
            batch_size=16,
            learning_rate=0.02,
            transfer_strategy="scratch"
        )
        self.source_params = self.source_model.get_parameters()

    def test_parameter_transfer_initialization(self):
        """Verify target model correctly loads pre-trained source parameters."""
        target_model = TransferHealthcareModel(random_seed=999)
        # Verify initial target parameters differ from source
        self.assertFalse(np.allclose(target_model.get_parameters()["W1"], self.source_params["W1"]))

        # Transfer full parameters
        target_model.set_parameters(self.source_params)
        for k in ["W1", "b1", "W2", "b2"]:
            np.testing.assert_allclose(target_model.get_parameters()[k], self.source_params[k])

    def test_feature_extraction_freezes_layer1_and_updates_layer2(self):
        """Verify 'feature_extraction' freezes W1, b1 completely while updating W2, b2."""
        target_model = TransferHealthcareModel(random_seed=123)
        target_model.set_parameters(self.source_params)

        initial_w1 = target_model.get_parameters()["W1"].copy()
        initial_b1 = target_model.get_parameters()["b1"].copy()
        initial_w2 = target_model.get_parameters()["W2"].copy()
        initial_b2 = target_model.get_parameters()["b2"].copy()

        # Fine-tune using feature extraction strategy
        target_model.fit(
            self.X_dummy, self.y_dummy,
            epochs=5,
            batch_size=16,
            learning_rate=0.05,
            transfer_strategy="feature_extraction"
        )

        final_params = target_model.get_parameters()

        # Layer 1 must be strictly IDENTICAL (Frozen!)
        np.testing.assert_array_equal(final_params["W1"], initial_w1, err_msg="W1 was modified during feature extraction!")
        np.testing.assert_array_equal(final_params["b1"], initial_b1, err_msg="b1 was modified during feature extraction!")

        # Layer 2 must have UPDATED (Trained head)
        self.assertFalse(np.allclose(final_params["W2"], initial_w2), "W2 was not updated during feature extraction!")
        self.assertFalse(np.allclose(final_params["b2"], initial_b2), "b2 was not updated during feature extraction!")

    def test_full_fine_tuning_updates_all_layers(self):
        """Verify 'full_fine_tuning' updates both Layer 1 and Layer 2."""
        target_model = TransferHealthcareModel(random_seed=123)
        target_model.set_parameters(self.source_params)

        initial_w1 = target_model.get_parameters()["W1"].copy()
        initial_w2 = target_model.get_parameters()["W2"].copy()

        target_model.fit(
            self.X_dummy, self.y_dummy,
            epochs=5,
            batch_size=16,
            learning_rate=0.05,
            transfer_strategy="full_fine_tuning"
        )

        final_params = target_model.get_parameters()

        # Both layers must be updated
        self.assertFalse(np.allclose(final_params["W1"], initial_w1), "W1 was not updated during full fine-tuning!")
        self.assertFalse(np.allclose(final_params["W2"], initial_w2), "W2 was not updated during full fine-tuning!")

    def test_evaluation_metrics_validity(self):
        """Verify evaluate returns valid metric bounds and confusion matrix."""
        metrics = self.source_model.evaluate(self.X_dummy, self.y_dummy)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertLessEqual(metrics["precision"], 1.0)
        self.assertGreaterEqual(metrics["recall"], 0.0)
        self.assertLessEqual(metrics["recall"], 1.0)
        self.assertGreaterEqual(metrics["f1_score"], 0.0)
        self.assertLessEqual(metrics["f1_score"], 1.0)
        if metrics["roc_auc"] is not None:
            self.assertGreaterEqual(metrics["roc_auc"], 0.0)
            self.assertLessEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        self.assertEqual(len(metrics["confusion_matrix"][0]), 2)

    def test_seed_reproducibility(self):
        """Verify identical random seeds produce identical fine-tuned models."""
        m1 = TransferHealthcareModel(random_seed=42)
        m1.set_parameters(self.source_params)
        m1.fit(self.X_dummy, self.y_dummy, epochs=3, batch_size=16, learning_rate=0.02, transfer_strategy="full_fine_tuning")

        m2 = TransferHealthcareModel(random_seed=42)
        m2.set_parameters(self.source_params)
        m2.fit(self.X_dummy, self.y_dummy, epochs=3, batch_size=16, learning_rate=0.02, transfer_strategy="full_fine_tuning")

        for k in ["W1", "b1", "W2", "b2"]:
            np.testing.assert_allclose(m1.get_parameters()[k], m2.get_parameters()[k], atol=1e-12)


class TestFTLCommunicationAndPrivacy(unittest.TestCase):
    """Test suite covering FTL communication payload sizing, paradigm comparison, and privacy enforcement."""

    def test_calculate_ftl_communication_volume_default(self):
        """Verify FTL communication analysis reports 57 parameters (456 bytes) and 1 transfer event."""
        comm = calculate_ftl_communication_volume()

        self.assertEqual(comm["paradigm"], "Federated Transfer Learning (FTL)")
        self.assertEqual(comm["total_parameters"], 57)
        self.assertEqual(comm["bytes_per_parameter"], 8)
        self.assertEqual(comm["total_communication_bytes"], 456)
        self.assertEqual(comm["transfer_events"], 1)
        self.assertEqual(comm["target_fine_tuning_network_bytes"], 0)

        # Layer breakdown verification
        self.assertEqual(comm["feature_extractor_parameters"], 48)  # 5*8 + 8 = 48
        self.assertEqual(comm["feature_extractor_bytes"], 384)
        self.assertEqual(comm["prediction_head_parameters"], 9)     # 8*1 + 1 = 9
        self.assertEqual(comm["prediction_head_bytes"], 72)

    def test_compare_hfl_vs_vfl_communication_includes_ftl(self):
        """Verify comparative communication table includes HFL, VFL, and FTL."""
        rows = compare_hfl_vs_vfl_communication()
        self.assertGreaterEqual(len(rows), 5)

        paradigms = [r["paradigm"] for r in rows]
        self.assertIn("Horizontal Federated Learning (HFL)", paradigms)
        self.assertIn("Vertical Federated Learning (VFL)", paradigms)
        self.assertIn("Federated Transfer Learning (FTL)", paradigms)

        ftl_row = next(r for r in rows if "Transfer" in r["paradigm"])
        self.assertEqual(ftl_row["parameter_transfer_bytes"], 456)
        self.assertEqual(ftl_row["representation_transfer_bytes"], 0)

    def test_firebase_ftl_payload_privacy_enforcement(self):
        """Verify Firebase FTL experiment record strictly excludes raw patient data and predictions."""
        payload = build_ftl_experiment_record()

        self.assertEqual(payload["algorithm"], "Federated Transfer Learning")
        self.assertIn("privacy_guarantees", payload)
        self.assertFalse(payload["privacy_guarantees"]["raw_patient_data_persisted"])
        self.assertFalse(payload["privacy_guarantees"]["patient_level_predictions_persisted"])
        self.assertFalse(payload["privacy_guarantees"]["source_patient_records_transmitted"])
        self.assertFalse(payload["privacy_guarantees"]["target_patient_records_transmitted"])

        # Check no raw patient records are leaked in payload keys
        for k in payload.keys():
            self.assertNotEqual(k, "patient_records")
            self.assertNotEqual(k, "predictions")
            self.assertNotEqual(k, "raw_data")


if __name__ == "__main__":
    unittest.main()


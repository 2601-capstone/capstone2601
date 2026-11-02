"""
Unit tests for Global Model Evaluation and Baseline Comparison (src/evaluation.py).
"""

import unittest
import os
import shutil
import numpy as np
import pandas as pd

from src.evaluation import (
    calculate_metrics,
    evaluate_model,
    load_preprocessed_test_data,
    train_centralized_model,
    train_isolated_local_model,
    run_full_evaluation
)
from src.model import HealthcareMLP
from src.data_preparation import load_scaler, load_central_test_data, FEATURE_COLUMNS


class TestEvaluation(unittest.TestCase):
    """Test suite for metric calculation, baseline training, and central evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.scaler_path = os.path.join("data", "scaler_params.json")
        cls.scaler = load_scaler(cls.scaler_path)

    def test_calculate_metrics_known_example(self):
        """Verify metric calculation on a deterministic, hand-crafted truth table."""
        # y_true: [0, 0, 1, 1]
        # y_pred: [0, 1, 0, 1] -> TN=1, FP=1, FN=1, TP=1
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.7, 0.3, 0.9])

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        self.assertAlmostEqual(metrics["accuracy"], 0.50, places=4)
        self.assertAlmostEqual(metrics["precision"], 0.50, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.50, places=4)
        self.assertAlmostEqual(metrics["f1_score"], 0.50, places=4)
        self.assertIsInstance(metrics["roc_auc"], float)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])

    def test_roc_auc_single_class_edge_case(self):
        """Ensure ROC-AUC returns None safely when only a single class is present."""
        y_true_single = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0])
        y_prob = np.array([0.2, 0.1, 0.8, 0.3])

        metrics = calculate_metrics(y_true_single, y_pred, y_prob)
        self.assertIsNone(metrics["roc_auc"])
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_evaluate_model_output_structure(self):
        """Verify evaluate_model produces expected dictionary keys on HealthcareMLP."""
        model = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=42)
        X_dummy = np.random.randn(20, 5)
        y_dummy = np.array([0, 1] * 10)

        metrics = evaluate_model(model, X_dummy, y_dummy)
        expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"}
        self.assertTrue(expected_keys.issubset(set(metrics.keys())))
        self.assertIsInstance(metrics["confusion_matrix"], list)

    def test_load_preprocessed_test_data(self):
        """Ensure central test dataset loads exactly 300 samples with 5 features."""
        X_test, y_test = load_preprocessed_test_data(data_dir="data")
        self.assertEqual(X_test.shape, (300, 5))
        self.assertEqual(y_test.shape, (300,))
        self.assertEqual(set(np.unique(y_test)), {0, 1})
        self.assertFalse(np.isnan(X_test).any())

    def test_test_set_isolation_from_scaler(self):
        """Ensure central test data was NOT used to fit the feature scaler (zero data leakage)."""
        test_df = load_central_test_data(data_dir="data")
        scaler = load_scaler(os.path.join("data", "scaler_params.json"))

        # Test set mean is distinct from the training-set mean stored in scaler
        for col in FEATURE_COLUMNS:
            test_col_mean = float(test_df[col].mean())
            scaler_col_mean = float(scaler["mean"][col])
            # They should not be identical (would indicate scaler was fitted on test set)
            self.assertNotAlmostEqual(
                test_col_mean, scaler_col_mean, places=3,
                msg=f"Potential data leakage: test mean matches scaler mean for {col}"
            )

    def test_evaluation_of_all_five_model_types(self):
        """Run a lightweight evaluation pipeline to test all 5 models and persistence."""
        test_output_dir = os.path.join("results", "metrics", "test_evaluation_tmp")

        comp_df, detailed_summary = run_full_evaluation(
            num_federated_rounds=1,
            epochs_per_baseline=2,
            random_seed=42,
            data_dir="data",
            output_dir=test_output_dir
        )

        expected_models = {
            "Federated Global Model",
            "Centralized Baseline",
            "Hospital 1 Local Baseline",
            "Hospital 2 Local Baseline",
            "Hospital 3 Local Baseline"
        }

        # 1. Verify all 5 models are present in comparison table
        self.assertEqual(set(comp_df["Model"]), expected_models)

        # 2. Verify all 5 models are in detailed summary with confusion matrices
        self.assertEqual(set(detailed_summary["models"].keys()), expected_models)
        for model_name in expected_models:
            cm = detailed_summary["models"][model_name]["confusion_matrix"]
            self.assertEqual(len(cm), 2)
            self.assertEqual(len(cm[0]), 2)
            self.assertEqual(len(cm[1]), 2)

        # 3. Verify files were saved
        csv_file = os.path.join(test_output_dir, "model_comparison.csv")
        json_file = os.path.join(test_output_dir, "evaluation_summary.json")
        self.assertTrue(os.path.exists(csv_file))
        self.assertTrue(os.path.exists(json_file))

        # Cleanup
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)


if __name__ == "__main__":
    unittest.main()

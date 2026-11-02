"""
Unit tests for Robustness Analysis Across Hospitals (src/robustness.py).
"""

import unittest
import os
import numpy as np
import pandas as pd

from src.robustness import (
    partition_hospital_test_sets,
    get_or_create_hospital_test_sets,
    evaluate_predictions,
    calculate_robustness_statistics,
    run_robustness_analysis
)
from src.data_preparation import TARGET_COLUMN


class TestRobustnessAnalysis(unittest.TestCase):
    """Test suite verifying client-level evaluation, variance, and Non-IID robustness."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = "data"

    def test_partition_hospital_test_sets_counts_and_ratios(self):
        """Verify central test set (300) is partitioned into 3 x 100 sets with exact Non-IID ratios."""
        partitions = partition_hospital_test_sets(data_dir=self.data_dir, random_seed=42)

        self.assertEqual(set(partitions.keys()), {"hospital_1", "hospital_2", "hospital_3"})

        # Check sizes
        self.assertEqual(len(partitions["hospital_1"]), 100)
        self.assertEqual(len(partitions["hospital_2"]), 100)
        self.assertEqual(len(partitions["hospital_3"]), 100)

        # Check exact class allocations
        h1_c1 = (partitions["hospital_1"][TARGET_COLUMN] == 1).sum()
        h2_c1 = (partitions["hospital_2"][TARGET_COLUMN] == 1).sum()
        h3_c1 = (partitions["hospital_3"][TARGET_COLUMN] == 1).sum()

        self.assertEqual(h1_c1, 65, "Hospital 1 must have 65% high-risk samples")
        self.assertEqual(h2_c1, 50, "Hospital 2 must have 50% high-risk samples")
        self.assertEqual(h3_c1, 39, "Hospital 3 must have 39% high-risk samples")

        # Sum of class 1 equals 154 (exact central_test.csv count)
        self.assertEqual(h1_c1 + h2_c1 + h3_c1, 154)

    def test_evaluate_predictions_known_example(self):
        """Verify metric calculation on a known binary vector."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])  # TN=1, FP=1, FN=0, TP=2
        y_prob = np.array([0.2, 0.8, 0.6, 0.9])

        metrics = evaluate_predictions(y_true, y_pred, y_prob)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2.0 / 3.0, places=4)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertIsInstance(metrics["roc_auc"], float)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])

    def test_calculate_robustness_statistics_mathematical_precision(self):
        """Verify variance, std dev, min/max, best/worst, and disparity gap calculations."""
        mock_hospital_metrics = {
            "hospital_1": {"accuracy": 0.80, "f1_score": 0.82},
            "hospital_2": {"accuracy": 0.74, "f1_score": 0.75},
            "hospital_3": {"accuracy": 0.70, "f1_score": 0.68}
        }

        stats = calculate_robustness_statistics(mock_hospital_metrics)

        # Accuracies: [0.80, 0.74, 0.70]
        # Mean: (0.80 + 0.74 + 0.70) / 3 = 0.746666... -> 0.7467
        self.assertAlmostEqual(stats["mean_accuracy"], 0.7467, places=3)
        self.assertAlmostEqual(stats["min_accuracy"], 0.70, places=2)
        self.assertAlmostEqual(stats["max_accuracy"], 0.80, places=2)
        self.assertAlmostEqual(stats["accuracy_gap"], 0.10, places=2)

        # Variance & Standard Deviation explicit checks
        self.assertAlmostEqual(stats["accuracy_variance"], 0.001689, places=5)
        self.assertAlmostEqual(stats["accuracy_std"], 0.0411, places=3)
        self.assertAlmostEqual(stats["accuracy_std"], np.sqrt(stats["accuracy_variance"]), places=2)

        self.assertEqual(stats["best_hospital_accuracy"], "hospital_1")
        self.assertEqual(stats["worst_hospital_accuracy"], "hospital_3")

        # F1 gap: 0.82 - 0.68 = 0.14
        self.assertAlmostEqual(stats["f1_gap"], 0.14, places=2)
        self.assertEqual(stats["best_hospital_f1"], "hospital_1")
        self.assertEqual(stats["worst_hospital_f1"], "hospital_3")

    def test_evaluate_predictions_single_class_edge_case(self):
        """Test evaluation when true labels contain only a single class (ROC-AUC undefined)."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 1])
        y_prob = np.array([0.9, 0.4, 0.8, 0.7])

        metrics = evaluate_predictions(y_true, y_pred, y_prob)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIsNone(metrics["roc_auc"], "ROC-AUC must be None when only one class is present")
        self.assertEqual(metrics["recall"], 0.75)

    def test_evaluate_predictions_mismatched_lengths_raises_error(self):
        """Test evaluation with mismatched array lengths raises ValueError."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1, 1])
        y_prob = np.array([0.1, 0.8])

        with self.assertRaises(ValueError):
            evaluate_predictions(y_true, y_pred, y_prob)

    def test_non_iid_class_distribution_analysis(self):
        """Verify positive and negative class distributions match exact simulated hospital demographics."""
        partitions = partition_hospital_test_sets(data_dir=self.data_dir, random_seed=42)

        expected_distributions = {
            "hospital_1": {"pos": 65, "neg": 35, "pos_ratio": 0.65},
            "hospital_2": {"pos": 50, "neg": 50, "pos_ratio": 0.50},
            "hospital_3": {"pos": 39, "neg": 61, "pos_ratio": 0.39}
        }

        for h_id, expected in expected_distributions.items():
            df = partitions[h_id]
            pos_count = int((df[TARGET_COLUMN] == 1).sum())
            neg_count = int((df[TARGET_COLUMN] == 0).sum())
            total = len(df)

            self.assertEqual(pos_count, expected["pos"])
            self.assertEqual(neg_count, expected["neg"])
            self.assertEqual(total, 100)
            self.assertAlmostEqual(pos_count / total, expected["pos_ratio"], places=2)

    def test_empty_metrics_raises_value_error(self):
        """Empty input to calculate_robustness_statistics must raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_robustness_statistics({})

    def test_get_or_create_hospital_test_sets_persistence(self):
        """Verify CSV files exist and can be loaded repeatedly."""
        partitions = get_or_create_hospital_test_sets(data_dir=self.data_dir, random_seed=42)
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "hospital_1_test.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "hospital_2_test.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "hospital_3_test.csv")))
        self.assertEqual(len(partitions["hospital_1"]), 100)


if __name__ == "__main__":
    unittest.main()


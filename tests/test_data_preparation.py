"""
Unit tests for Healthcare Dataset Preparation Module (src/data_preparation.py).
"""

import unittest
import numpy as np
import pandas as pd
from src.data_preparation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    generate_synthetic_healthcare_data,
    validate_dataset,
    split_features_and_target,
    prepare_train_test_split,
    partition_hospitals,
    validate_hospital_partitions,
    load_hospital_data,
    load_central_test_data
)


class TestDataPreparation(unittest.TestCase):
    """Test suite for Phase 2 synthetic dataset creation and client separation."""

    def setUp(self):
        self.random_seed = 42
        self.num_samples = 1500
        self.df = generate_synthetic_healthcare_data(
            num_samples=self.num_samples,
            random_seed=self.random_seed
        )

    def test_dataset_generation_shape_and_columns(self):
        """Test dataset shape and presence of all required feature columns."""
        self.assertEqual(len(self.df), self.num_samples)
        expected_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
        for col in expected_cols:
            self.assertIn(col, self.df.columns)

    def test_no_missing_or_infinite_values(self):
        """Ensure no NaN, null, or inf values exist in generated data."""
        self.assertFalse(self.df.isnull().any().any())
        self.assertFalse(np.isinf(self.df[FEATURE_COLUMNS].to_numpy()).any())

    def test_target_label_values(self):
        """Ensure binary target consists strictly of {0, 1}."""
        unique_targets = set(self.df[TARGET_COLUMN].unique())
        self.assertEqual(unique_targets, {0, 1})

    def test_reproducibility(self):
        """Verify identical output when generated with identical seed."""
        df_duplicate = generate_synthetic_healthcare_data(
            num_samples=self.num_samples,
            random_seed=self.random_seed
        )
        pd.testing.assert_frame_equal(self.df, df_duplicate)

    def test_feature_target_split(self):
        """Test separating X (features) and y (target)."""
        X, y = split_features_and_target(self.df)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertEqual(len(X), self.num_samples)
        self.assertEqual(len(y), self.num_samples)
        self.assertEqual(y.name, TARGET_COLUMN)

    def test_train_test_split(self):
        """Test 80/20 train/test split proportion and isolation."""
        train_df, test_df = prepare_train_test_split(self.df, test_size=0.2, random_seed=self.random_seed)
        self.assertEqual(len(train_df), 1200)
        self.assertEqual(len(test_df), 300)
        # Ensure disjoint index records
        self.assertEqual(len(train_df) + len(test_df), self.num_samples)

    def test_hospital_partition_counts_and_classes(self):
        """Verify each of the 3 hospital clients receives 400 samples with both classes."""
        train_df, _ = prepare_train_test_split(self.df, test_size=0.2, random_seed=self.random_seed)
        hospitals = partition_hospitals(train_df, random_seed=self.random_seed)

        self.assertEqual(set(hospitals.keys()), {"hospital_1", "hospital_2", "hospital_3"})

        total_client_samples = 0
        for name, client_df in hospitals.items():
            # Check length is 400
            self.assertEqual(len(client_df), 400)
            total_client_samples += len(client_df)

            # Check both classes are well represented (at least 50 samples each)
            c0 = (client_df[TARGET_COLUMN] == 0).sum()
            c1 = (client_df[TARGET_COLUMN] == 1).sum()
            self.assertGreaterEqual(c0, 50, f"{name} has too few class 0 samples")
            self.assertGreaterEqual(c1, 50, f"{name} has too few class 1 samples")

        # Sum of client training datasets equals total train set
        self.assertEqual(total_client_samples, len(train_df))

    def test_non_iid_distribution_differences(self):
        """Verify hospitals have distinct (Non-IID) risk profiles."""
        train_df, _ = prepare_train_test_split(self.df, test_size=0.2, random_seed=self.random_seed)
        hospitals = partition_hospitals(train_df, random_seed=self.random_seed)

        pct_h1 = (hospitals["hospital_1"][TARGET_COLUMN] == 1).mean()
        pct_h2 = (hospitals["hospital_2"][TARGET_COLUMN] == 1).mean()
        pct_h3 = (hospitals["hospital_3"][TARGET_COLUMN] == 1).mean()

        # Hospital 1 should have highest concentration of high risk, Hospital 3 lowest
        self.assertGreater(pct_h1, pct_h2)
        self.assertGreater(pct_h2, pct_h3)

    def test_saved_csv_files_loadable(self):
        """Check loading saved CSV data files for hospitals and test set."""
        h1 = load_hospital_data("hospital_1")
        h2 = load_hospital_data("hospital_2")
        h3 = load_hospital_data("hospital_3")
        test_df = load_central_test_data()

        self.assertEqual(len(h1), 400)
        self.assertEqual(len(h2), 400)
        self.assertEqual(len(h3), 400)
        self.assertEqual(len(test_df), 300)


if __name__ == "__main__":
    unittest.main()

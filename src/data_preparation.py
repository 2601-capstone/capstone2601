"""
Data Preparation Module for Federated Healthcare Analytics Simulation.

This module handles:
1. Generating synthetic tabular healthcare records (age, blood pressure, glucose, BMI, cholesterol).
2. Data validation (checking schema, missing/infinite values, binary risk labels).
3. Train/test splitting (reserving a separate centralized test set for evaluation).
4. Partitioning training data across three hospital client nodes with non-identical (Non-IID) distributions.
5. Exporting and loading partitioned datasets.

NOTE: This is a simulation using synthetic records. It does not represent real patients
or clinical diagnostic outcomes.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Expected dataset schema
FEATURE_COLUMNS = [
    "age",
    "blood_pressure",
    "glucose_level",
    "bmi",
    "cholesterol_level"
]
TARGET_COLUMN = "risk_label"


def generate_synthetic_healthcare_data(num_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic binary classification healthcare dataset.

    Parameters:
        num_samples (int): Total number of synthetic records (default: 1500).
        random_seed (int): Seed for reproducible generation (default: 42).

    Returns:
        pd.DataFrame: Synthetic dataset with clinical features and binary risk_label.
    """
    rng = np.random.RandomState(random_seed)

    # 1. Feature simulation based on realistic clinical ranges
    # Age: between 20 and 80 years
    age = rng.normal(loc=52.0, scale=13.0, size=num_samples)
    age = np.clip(age, 20.0, 80.0).round().astype(int)

    # Blood Pressure: Systolic BP (mmHg), typical range 90-180
    blood_pressure = rng.normal(loc=126.0, scale=17.0, size=num_samples)
    blood_pressure = np.clip(blood_pressure, 90.0, 185.0).round(1)

    # Fasting Glucose Level: (mg/dL), typical range 70-240
    glucose_level = rng.normal(loc=112.0, scale=30.0, size=num_samples)
    glucose_level = np.clip(glucose_level, 70.0, 250.0).round(1)

    # Body Mass Index (BMI): (kg/m^2), typical range 18-45
    bmi = rng.normal(loc=27.4, scale=5.1, size=num_samples)
    bmi = np.clip(bmi, 18.0, 46.0).round(1)

    # Total Cholesterol: (mg/dL), typical range 130-320
    cholesterol_level = rng.normal(loc=204.0, scale=36.0, size=num_samples)
    cholesterol_level = np.clip(cholesterol_level, 130.0, 320.0).round(1)

    # 2. Risk label calculation via a logistic risk score
    # Standardize features for linear combination
    z_age = (age - 52.0) / 13.0
    z_bp = (blood_pressure - 126.0) / 17.0
    z_glu = (glucose_level - 112.0) / 30.0
    z_bmi = (bmi - 27.4) / 5.1
    z_chol = (cholesterol_level - 204.0) / 36.0

    # Linear risk combination with realistic weights + noise
    noise = rng.normal(loc=0.0, scale=0.5, size=num_samples)
    risk_score = (
        0.50 * z_age +
        0.55 * z_bp +
        0.65 * z_glu +
        0.45 * z_bmi +
        0.40 * z_chol +
        noise
    )

    # Sigmoid function to derive probabilities
    probability = 1.0 / (1.0 + np.exp(-risk_score))

    # Binary assignment: 0 = Lower Risk, 1 = Higher Risk
    risk_label = (probability >= 0.50).astype(int)

    df = pd.DataFrame({
        "age": age,
        "blood_pressure": blood_pressure,
        "glucose_level": glucose_level,
        "bmi": bmi,
        "cholesterol_level": cholesterol_level,
        "risk_label": risk_label
    })

    return df


def validate_dataset(df: pd.DataFrame, required_features=None, target_col: str = TARGET_COLUMN) -> bool:
    """
    Validate that the dataset meets required schema and quality checks.

    Raises:
        ValueError: If validation fails.

    Returns:
        bool: True if validation passes.
    """
    if required_features is None:
        required_features = FEATURE_COLUMNS

    # Check required columns
    expected_cols = set(required_features + [target_col])
    actual_cols = set(df.columns)
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Check for NaN / null values
    if df[required_features + [target_col]].isnull().any().any():
        raise ValueError("Dataset contains null or NaN values.")

    # Check for infinite values
    if np.isinf(df[required_features].to_numpy()).any():
        raise ValueError("Dataset contains infinite numerical values.")

    # Check binary target
    unique_targets = set(df[target_col].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target column '{target_col}' must contain only 0 and 1. Found: {unique_targets}")

    # Ensure both classes are present
    if len(unique_targets) < 2:
        raise ValueError("Dataset target must contain both classes (0 and 1).")

    return True


def split_features_and_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN):
    """
    Separate features (X) and target label (y).

    Returns:
        tuple (pd.DataFrame, pd.Series): X (features), y (target).
    """
    validate_dataset(df, target_col=target_col)
    X = df[FEATURE_COLUMNS].copy()
    y = df[target_col].copy()
    return X, y


def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42):
    """
    Split the synthetic dataset into a training pool (for federated hospital partition)
    and an isolated centralized test set (for global evaluation).

    Returns:
        tuple (pd.DataFrame, pd.DataFrame): (train_df, test_df)
    """
    validate_dataset(df)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[TARGET_COLUMN]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def partition_hospitals(train_df: pd.DataFrame, random_seed: int = 42) -> dict:
    """
    Divide the training dataset across three simulated hospital clients with
    a realistic non-identical (Non-IID) risk distribution.

    Simulation Context:
    - Hospital 1 (Specialty Referral Center): Higher risk patient concentration (~65% Class 1)
    - Hospital 2 (Community Hospital): Balanced patient population (~50% Class 1)
    - Hospital 3 (Primary Care Clinic): Lower risk patient concentration (~35% Class 1)

    Each hospital receives an equal sample count (e.g., 400 records each from 1,200 training samples),
    and all hospitals have sufficient examples from both classes.

    Parameters:
        train_df (pd.DataFrame): Training dataset of 1,200 samples.
        random_seed (int): Seed for reproducibility.

    Returns:
        dict: Mapping {"hospital_1": df, "hospital_2": df, "hospital_3": df}.
    """
    validate_dataset(train_df)

    rng = np.random.RandomState(random_seed)

    # Separate class 0 and class 1 samples with reproducible shuffle
    class_0_df = train_df[train_df[TARGET_COLUMN] == 0].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    class_1_df = train_df[train_df[TARGET_COLUMN] == 1].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    total_samples = len(train_df)
    samples_per_hospital = total_samples // 3

    # Target class 1 counts per hospital to simulate realistic clinical differences:
    # Hospital 1 (Specialty center): ~65% high risk
    # Hospital 2 (Community hospital): ~50% high risk
    # Hospital 3 (Primary care clinic): remaining high risk (~39%)
    h1_c1 = int(round(samples_per_hospital * 0.65))
    h2_c1 = int(round(samples_per_hospital * 0.50))
    h3_c1 = len(class_1_df) - (h1_c1 + h2_c1)

    # Class 0 counts complement each hospital's quota to ensure equal sample size
    h1_c0 = samples_per_hospital - h1_c1
    h2_c0 = samples_per_hospital - h2_c1
    h3_c0 = len(class_0_df) - (h1_c0 + h2_c0)

    # Slice partitions
    h1_df = pd.concat([
        class_1_df.iloc[:h1_c1],
        class_0_df.iloc[:h1_c0]
    ]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    h2_df = pd.concat([
        class_1_df.iloc[h1_c1:h1_c1 + h2_c1],
        class_0_df.iloc[h1_c0:h1_c0 + h2_c0]
    ]).sample(frac=1.0, random_state=random_seed + 1).reset_index(drop=True)

    h3_df = pd.concat([
        class_1_df.iloc[h1_c1 + h2_c1:],
        class_0_df.iloc[h1_c0 + h2_c0:]
    ]).sample(frac=1.0, random_state=random_seed + 2).reset_index(drop=True)

    hospital_dict = {
        "hospital_1": h1_df,
        "hospital_2": h2_df,
        "hospital_3": h3_df
    }

    validate_hospital_partitions(hospital_dict, expected_total_samples=len(train_df))
    return hospital_dict


def validate_hospital_partitions(
    hospital_dict: dict,
    expected_total_samples: int = None,
    min_samples_per_class: int = 20
) -> bool:
    """
    Validate partitioned hospital datasets for completeness and Non-IID integrity.

    Raises:
        ValueError: If any partition validation fails.

    Returns:
        bool: True if all client validations pass.
    """
    total_partition_samples = 0

    for name, client_df in hospital_dict.items():
        # Validate schema and data quality
        validate_dataset(client_df)

        if len(client_df) == 0:
            raise ValueError(f"Partition '{name}' is empty.")

        # Verify representation of both classes
        counts = client_df[TARGET_COLUMN].value_counts().to_dict()
        count_0 = counts.get(0, 0)
        count_1 = counts.get(1, 0)

        if count_0 < min_samples_per_class:
            raise ValueError(
                f"Partition '{name}' has insufficient Class 0 samples: {count_0} (< {min_samples_per_class})"
            )
        if count_1 < min_samples_per_class:
            raise ValueError(
                f"Partition '{name}' has insufficient Class 1 samples: {count_1} (< {min_samples_per_class})"
            )

        total_partition_samples += len(client_df)

    if expected_total_samples is not None and total_partition_samples != expected_total_samples:
        raise ValueError(
            f"Total client samples ({total_partition_samples}) does not match expected total ({expected_total_samples})."
        )

    return True


def save_datasets(
    full_df: pd.DataFrame,
    hospital_dict: dict,
    test_df: pd.DataFrame,
    output_dir: str = "data"
) -> dict:
    """
    Save the generated datasets and client partitions to CSV files in the data directory.

    Returns:
        dict: Mapping of dataset key to saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_paths = {
        "full_synthetic": os.path.join(output_dir, "synthetic_healthcare_data.csv"),
        "hospital_1": os.path.join(output_dir, "hospital_1_train.csv"),
        "hospital_2": os.path.join(output_dir, "hospital_2_train.csv"),
        "hospital_3": os.path.join(output_dir, "hospital_3_train.csv"),
        "central_test": os.path.join(output_dir, "central_test.csv")
    }

    full_df.to_csv(file_paths["full_synthetic"], index=False)
    hospital_dict["hospital_1"].to_csv(file_paths["hospital_1"], index=False)
    hospital_dict["hospital_2"].to_csv(file_paths["hospital_2"], index=False)
    hospital_dict["hospital_3"].to_csv(file_paths["hospital_3"], index=False)
    test_df.to_csv(file_paths["central_test"], index=False)

    return file_paths


def load_hospital_data(hospital_name: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Load an individual hospital client's local training dataset.

    Parameters:
        hospital_name (str): 'hospital_1', 'hospital_2', or 'hospital_3'.
        data_dir (str): Directory where CSV files are stored.

    Returns:
        pd.DataFrame: Local client training dataset.
    """
    file_path = os.path.join(data_dir, f"{hospital_name}_train.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset for {hospital_name} not found at {file_path}")
    df = pd.read_csv(file_path)
    validate_dataset(df)
    return df


def load_central_test_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the central holdout evaluation test dataset.

    Returns:
        pd.DataFrame: Centralized evaluation test dataset.
    """
    file_path = os.path.join(data_dir, "central_test.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Central test dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    validate_dataset(df)
    return df


def compute_training_scaler(train_df: pd.DataFrame) -> dict:
    """
    Compute standardization parameters (mean and std) exclusively on the training pool.
    This guarantees that all hospital clients and evaluation datasets operate in
    an identical feature space without data leakage.

    Returns:
        dict: {"mean": {col: float}, "std": {col: float}}
    """
    validate_dataset(train_df)
    means = {col: float(train_df[col].mean()) for col in FEATURE_COLUMNS}
    stds = {col: float(max(train_df[col].std(), 1e-7)) for col in FEATURE_COLUMNS}
    return {"mean": means, "std": stds}


def apply_scaler(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    """
    Apply standardization: X_scaled = (X - mean) / std.

    Parameters:
        df (pd.DataFrame): Dataset containing FEATURE_COLUMNS.
        scaler (dict): Dictionary with 'mean' and 'std' mappings.

    Returns:
        np.ndarray: Scaled feature matrix of shape (N, 5).
    """
    scaled_cols = []
    for col in FEATURE_COLUMNS:
        mean_val = scaler["mean"][col]
        std_val = scaler["std"][col]
        scaled = (df[col].astype(float) - mean_val) / std_val
        scaled_cols.append(scaled.to_numpy())
    return np.column_stack(scaled_cols)


def save_scaler(scaler: dict, filepath: str = os.path.join("data", "scaler_params.json")) -> str:
    """Save shared feature scaler parameters to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(scaler, f, indent=4)
    return filepath


def load_scaler(filepath: str = os.path.join("data", "scaler_params.json")) -> dict:
    """Load shared feature scaler parameters from JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scaler parameters not found at {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 2 & 3: DATA PREPARATION & SHARED PREPROCESSING")
    print("=" * 65)

    # 1. Generate synthetic dataset
    print("\n1. Generating synthetic healthcare dataset (1,500 samples)...")
    full_dataset = generate_synthetic_healthcare_data(num_samples=1500, random_seed=42)
    validate_dataset(full_dataset)
    print(f"   Generated total records: {len(full_dataset)}")
    print(f"   Features: {FEATURE_COLUMNS}")
    print(f"   Class distribution: {full_dataset[TARGET_COLUMN].value_counts().to_dict()}")

    # 2. Split train and centralized test set
    print("\n2. Splitting into training pool (80%) and holdout test set (20%)...")
    train_pool, test_pool = prepare_train_test_split(full_dataset, test_size=0.2, random_seed=42)
    print(f"   Training pool samples: {len(train_pool)}")
    print(f"   Holdout test samples:  {len(test_pool)}")

    # 3. Fit shared scaler on training pool only (prevents data leakage)
    print("\n3. Computing shared standardization parameters on training pool...")
    scaler_params = compute_training_scaler(train_pool)
    scaler_path = save_scaler(scaler_params)
    print(f"   Saved shared scaler to: {scaler_path}")

    # 4. Partition across 3 hospital clients
    print("\n4. Partitioning training data across 3 simulated hospitals (Non-IID)...")
    hospitals = partition_hospitals(train_pool, random_seed=42)

    for h_name, h_df in hospitals.items():
        c0 = (h_df[TARGET_COLUMN] == 0).sum()
        c1 = (h_df[TARGET_COLUMN] == 1).sum()
        pct_c1 = (c1 / len(h_df)) * 100
        print(f"   {h_name.upper()}: {len(h_df)} samples | Class 0: {c0} | Class 1: {c1} ({pct_c1:.1f}% High Risk)")

    # 5. Save to data/
    print("\n5. Saving datasets to 'data/' directory...")
    saved_paths = save_datasets(full_dataset, hospitals, test_pool, output_dir="data")
    for key, path in saved_paths.items():
        print(f"   Saved {key:<16}: {path}")

    print("\nDataset preparation and shared preprocessing completed successfully.")
    print("=" * 65)


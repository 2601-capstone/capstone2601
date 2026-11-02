"""
Longitudinal & Temporal Patient Data Module for Healthcare Analytics Simulation.

This module generates and partitions synthetic longitudinal patient time-series data:
1. Temporal Trajectory Simulation:
   - Sequence length: Configurable (default: 12 monthly time steps).
   - Features per time step:
     - Age: Constant baseline per patient.
     - Blood Pressure: Temporal baseline + linear drift + autoregressive autocorrelation + noise.
     - Glucose Level: Temporal baseline + progressive metabolic drift + episodic glycemic spikes.
     - Body Mass Index (BMI): Slow gradual longitudinal drift + low noise.
     - Cholesterol Level: Baseline + seasonal variation + temporal fluctuation.
2. Temporal Risk Mechanism:
   - Incorporates trajectory slope (rising BP and glucose over time).
   - Incorporates threshold exceedance duration (months with BP >= 140 mmHg or Glucose >= 126 mg/dL).
   - Incorporates temporal volatility (glycemic fluctuation / variance).
   - Combines baseline cardiovascular factors to compute a clinically plausible binary risk_label.
3. Patient-Level Partitioning & Isolation:
   - Partitioning is performed strictly at the patient level (patient_id).
   - Never allows time steps of a patient to appear in multiple splits (zero leakage).
   - Produces Train (70%), Validation (15%), and Test (15%) partitions.
   - Partitions training pool across 3 simulated hospital clients with Non-IID risk distributions.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

TEMPORAL_FEATURE_COLUMNS = [
    "age",
    "blood_pressure",
    "glucose_level",
    "bmi",
    "cholesterol_level"
]


def generate_synthetic_longitudinal_data(
    num_patients: int = 1000,
    sequence_length: int = 12,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Generate synthetic longitudinal measurement sequences for healthcare risk prediction.

    Parameters:
        num_patients (int): Total number of unique patients (default: 1,000).
        sequence_length (int): Number of consecutive temporal observation steps (default: 12).
        random_seed (int): Deterministic RNG seed.

    Returns:
        dict:
            - "X": np.ndarray of shape (num_patients, sequence_length, 5)
            - "y": np.ndarray of shape (num_patients, 1) (binary risk label)
            - "patient_ids": List[str] of IDs ("PID_T0001"...)
            - "trajectories_df": pd.DataFrame containing flattened step-by-step records.
    """
    rng = np.random.RandomState(random_seed)

    # 1. Baseline patient profile generation
    base_age = rng.normal(loc=54.0, scale=12.0, size=num_patients)
    base_age = np.clip(base_age, 25.0, 85.0).round().astype(float)

    base_bp = rng.normal(loc=126.0, scale=14.0, size=num_patients)
    base_bp = np.clip(base_bp, 95.0, 175.0)

    base_glucose = rng.normal(loc=108.0, scale=24.0, size=num_patients)
    base_glucose = np.clip(base_glucose, 75.0, 220.0)

    base_bmi = rng.normal(loc=27.2, scale=4.8, size=num_patients)
    base_bmi = np.clip(base_bmi, 18.5, 44.0)

    base_chol = rng.normal(loc=202.0, scale=32.0, size=num_patients)
    base_chol = np.clip(base_chol, 130.0, 310.0)

    # Patient latent progression propensity (determines disease worsening velocity over time)
    progression_propensity = rng.beta(a=2.0, b=2.5, size=num_patients)  # in [0, 1]

    # Initialize 3D tensor: (N, T, F)
    X = np.zeros((num_patients, sequence_length, 5), dtype=np.float32)
    time_steps = np.arange(sequence_length)

    flat_records = []
    patient_ids = [f"PID_T{i+1:04d}" for i in range(num_patients)]

    # Temporal feature metrics for risk scoring
    bp_slopes = np.zeros(num_patients)
    glucose_slopes = np.zeros(num_patients)
    bp_exceed_fractions = np.zeros(num_patients)
    glucose_volatilities = np.zeros(num_patients)

    for i in range(num_patients):
        pid = patient_ids[i]
        prop = progression_propensity[i]

        # Trajectory parameters
        # High propensity -> positive drift in BP and glucose
        bp_drift = (prop - 0.45) * 1.8  # mmHg per step
        glu_drift = (prop - 0.40) * 2.5  # mg/dL per step
        bmi_drift = (prop - 0.48) * 0.15

        # Temporal sequences with AR(1) dynamics + seasonal fluctuation
        bp_seq = np.zeros(sequence_length)
        glu_seq = np.zeros(sequence_length)
        bmi_seq = np.zeros(sequence_length)
        chol_seq = np.zeros(sequence_length)

        cur_bp = base_bp[i]
        cur_glu = base_glucose[i]
        cur_bmi = base_bmi[i]
        cur_chol = base_chol[i]

        for t in range(sequence_length):
            # AR(1) autoregressive noise
            ar_bp = 0.6 * (cur_bp - base_bp[i]) + bp_drift + rng.normal(0.0, 2.8)
            cur_bp = np.clip(base_bp[i] + ar_bp, 85.0, 205.0)
            bp_seq[t] = cur_bp

            # Glucose drift with intermittent metabolic spikes
            spike = 25.0 if rng.rand() < (0.08 + 0.15 * prop) else 0.0
            ar_glu = 0.5 * (cur_glu - base_glucose[i]) + glu_drift + rng.normal(0.0, 4.5) + spike
            cur_glu = np.clip(base_glucose[i] + ar_glu, 65.0, 290.0)
            glu_seq[t] = cur_glu

            # BMI slow drift
            cur_bmi = np.clip(base_bmi[i] + t * bmi_drift + rng.normal(0.0, 0.2), 17.0, 48.0)
            bmi_seq[t] = cur_bmi

            # Cholesterol variation
            cur_chol = np.clip(base_chol[i] + 6.0 * np.sin(2 * np.pi * t / sequence_length) + rng.normal(0.0, 5.0), 120.0, 340.0)
            chol_seq[t] = cur_chol

            # Store in tensor
            X[i, t, 0] = base_age[i]
            X[i, t, 1] = cur_bp
            X[i, t, 2] = cur_glu
            X[i, t, 3] = cur_bmi
            X[i, t, 4] = cur_chol

            flat_records.append({
                "patient_id": pid,
                "time_step": t,
                "age": base_age[i],
                "blood_pressure": round(float(cur_bp), 1),
                "glucose_level": round(float(cur_glu), 1),
                "bmi": round(float(cur_bmi), 1),
                "cholesterol_level": round(float(cur_chol), 1)
            })

        # Calculate temporal summary dynamics for patient risk scoring
        bp_slopes[i] = (bp_seq[-1] - bp_seq[0]) / max(1, sequence_length - 1)
        glucose_slopes[i] = (glu_seq[-1] - glu_seq[0]) / max(1, sequence_length - 1)
        bp_exceed_fractions[i] = np.mean(bp_seq >= 140.0)
        glucose_volatilities[i] = np.std(glu_seq)

    # 2. Longitudinal Risk Label Mechanism
    # Standardize indicators across population
    z_age = (base_age - 54.0) / 12.0
    z_bp_mean = (np.mean(X[:, :, 1], axis=1) - 126.0) / 14.0
    z_glu_mean = (np.mean(X[:, :, 2], axis=1) - 108.0) / 24.0
    z_bmi_mean = (np.mean(X[:, :, 3], axis=1) - 27.2) / 4.8
    z_chol_mean = (np.mean(X[:, :, 4], axis=1) - 202.0) / 32.0

    z_bp_slope = (bp_slopes - np.mean(bp_slopes)) / max(1e-5, np.std(bp_slopes))
    z_glu_slope = (glucose_slopes - np.mean(glucose_slopes)) / max(1e-5, np.std(glucose_slopes))
    z_glu_vol = (glucose_volatilities - np.mean(glucose_volatilities)) / max(1e-5, np.std(glucose_volatilities))

    # Composite temporal logit: baseline cardiovascular factors + longitudinal worsening dynamics
    noise = rng.normal(loc=0.0, scale=0.45, size=num_patients)
    temporal_risk_score = (
        0.30 * z_age +
        0.35 * z_bp_mean +
        0.40 * z_glu_mean +
        0.25 * z_bmi_mean +
        0.20 * z_chol_mean +
        0.45 * z_bp_slope +          # Trajectory slope penalty
        0.50 * z_glu_slope +         # Glycemic progression penalty
        0.35 * bp_exceed_fractions + # Chronic hypertension duration penalty
        0.30 * z_glu_vol +           # Volatility penalty
        noise
    )

    risk_prob = 1.0 / (1.0 + np.exp(-temporal_risk_score))
    risk_label = (risk_prob >= 0.50).astype(np.float32).reshape(-1, 1)

    # Update flat dataframe with risk_label
    df_flat = pd.DataFrame(flat_records)
    pid_to_label = dict(zip(patient_ids, risk_label.ravel()))
    df_flat["risk_label"] = df_flat["patient_id"].map(pid_to_label)

    return {
        "X": X,
        "y": risk_label,
        "patient_ids": patient_ids,
        "trajectories_df": df_flat,
        "metadata": {
            "num_patients": num_patients,
            "sequence_length": sequence_length,
            "num_features": 5,
            "feature_names": TEMPORAL_FEATURE_COLUMNS,
            "positive_risk_ratio": float(np.mean(risk_label))
        }
    }


def compute_temporal_scaler(X_train: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std normalization parameters exclusively on the training sequences.
    X_train shape: (N_train, T, F).
    Computes global mean and std per feature across all (N_train * T) measurements.

    Returns:
        dict: {"mean": {feature: float}, "std": {feature: float}}
    """
    N, T, F = X_train.shape
    X_flat = X_train.reshape(-1, F)

    means = {TEMPORAL_FEATURE_COLUMNS[f]: float(np.mean(X_flat[:, f])) for f in range(F)}
    stds = {TEMPORAL_FEATURE_COLUMNS[f]: float(max(1e-7, np.std(X_flat[:, f]))) for f in range(F)}

    return {"mean": means, "std": stds}


def apply_temporal_scaler(X: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    """
    Normalize 3D temporal feature tensor using pre-computed scaling parameters.
    X shape: (N, T, F).
    Returns normalized tensor of identical shape.
    """
    X_norm = X.copy()
    F = X.shape[2]
    for f in range(F):
        feat = TEMPORAL_FEATURE_COLUMNS[f]
        m = scaler["mean"].get(feat, 0.0)
        s = scaler["std"].get(feat, 1.0)
        s = s if s > 1e-12 else 1.0
        X_norm[:, :, f] = (X_norm[:, :, f] - m) / s
    return X_norm.astype(np.float32)


def partition_temporal_dataset(
    data_dict: Dict[str, Any],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Partition longitudinal patient records strictly at the patient level.
    Guarantees zero overlap between train, val, and test patient IDs.

    Returns:
        dict:
            - "train": {"X", "y", "patient_ids"}
            - "val": {"X", "y", "patient_ids"}
            - "test": {"X", "y", "patient_ids"}
    """
    X = data_dict["X"]
    y = data_dict["y"]
    patient_ids = data_dict["patient_ids"]
    N = len(patient_ids)

    rng = np.random.RandomState(random_seed)
    indices = np.arange(N)
    rng.shuffle(indices)

    n_train = int(round(N * train_ratio))
    n_val = int(round(N * val_ratio))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def make_split(idx):
        return {
            "X": X[idx],
            "y": y[idx],
            "patient_ids": [patient_ids[i] for i in idx]
        }

    return {
        "train": make_split(train_idx),
        "val": make_split(val_idx),
        "test": make_split(test_idx)
    }


def partition_federated_clients(
    train_split: Dict[str, Any],
    num_clients: int = 3,
    random_seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Partition the training sequences across 3 simulated hospital clients with Non-IID distributions:
    - hospital_1 (Specialty Center): Higher risk skew
    - hospital_2 (Community Hospital): Balanced
    - hospital_3 (Primary Clinic): Lower risk skew
    """
    X_train = train_split["X"]
    y_train = train_split["y"].ravel()
    pids = train_split["patient_ids"]

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    rng = np.random.RandomState(random_seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Distribute with Non-IID skew
    # H1: ~65% pos, H2: ~50% pos, H3: ~35% pos
    n_total = len(y_train)
    n_per_client = n_total // num_clients

    # Allocation targets
    # H1: 65% pos
    h1_pos_count = int(round(n_per_client * 0.65))
    h1_neg_count = n_per_client - h1_pos_count

    # H2: 50% pos
    h2_pos_count = int(round(n_per_client * 0.50))
    h2_neg_count = n_per_client - h2_pos_count

    # H3: Remaining
    h1_idx = np.concatenate([pos_idx[:h1_pos_count], neg_idx[:h1_neg_count]])
    h2_idx = np.concatenate([pos_idx[h1_pos_count:h1_pos_count + h2_pos_count], neg_idx[h1_neg_count:h1_neg_count + h2_neg_count]])
    h3_idx = np.concatenate([pos_idx[h1_pos_count + h2_pos_count:], neg_idx[h1_neg_count + h2_neg_count:]])

    def pack(idx):
        rng.shuffle(idx)
        return {
            "X": X_train[idx],
            "y": train_split["y"][idx],
            "patient_ids": [pids[i] for i in idx]
        }

    return {
        "hospital_1": pack(h1_idx),
        "hospital_2": pack(h2_idx),
        "hospital_3": pack(h3_idx)
    }


def prepare_and_save_temporal_data(
    output_dir: str = os.path.join("data", "temporal"),
    num_patients: int = 1000,
    sequence_length: int = 12,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Generate, normalize, and save all longitudinal datasets to disk:
    - temporal_train.npz
    - temporal_val.npz
    - temporal_test.npz
    - hospital_1_temporal.npz, hospital_2_temporal.npz, hospital_3_temporal.npz
    - temporal_scaler.json
    - temporal_patient_summary.csv

    Returns:
        dict: Mapping of split keys to saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate full raw dataset
    raw_data = generate_synthetic_longitudinal_data(
        num_patients=num_patients,
        sequence_length=sequence_length,
        random_seed=random_seed
    )

    # 2. Partition at patient level (70% train, 15% val, 15% test)
    splits = partition_temporal_dataset(
        raw_data,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed
    )

    # 3. Fit scaler exclusively on train split
    scaler = compute_temporal_scaler(splits["train"]["X"])

    # 4. Normalize tensors
    splits["train"]["X"] = apply_temporal_scaler(splits["train"]["X"], scaler)
    splits["val"]["X"] = apply_temporal_scaler(splits["val"]["X"], scaler)
    splits["test"]["X"] = apply_temporal_scaler(splits["test"]["X"], scaler)

    # 5. Partition federated clients from normalized train split
    clients = partition_federated_clients(splits["train"], num_clients=3, random_seed=random_seed)

    # 6. Save NPZ and JSON files
    paths = {
        "train": os.path.join(output_dir, "temporal_train.npz"),
        "val": os.path.join(output_dir, "temporal_val.npz"),
        "test": os.path.join(output_dir, "temporal_test.npz"),
        "hospital_1": os.path.join(output_dir, "hospital_1_temporal.npz"),
        "hospital_2": os.path.join(output_dir, "hospital_2_temporal.npz"),
        "hospital_3": os.path.join(output_dir, "hospital_3_temporal.npz"),
        "scaler": os.path.join(output_dir, "temporal_scaler.json"),
        "summary_csv": os.path.join(output_dir, "temporal_patient_summary.csv")
    }

    np.savez_compressed(paths["train"], X=splits["train"]["X"], y=splits["train"]["y"], patient_ids=splits["train"]["patient_ids"])
    np.savez_compressed(paths["val"], X=splits["val"]["X"], y=splits["val"]["y"], patient_ids=splits["val"]["patient_ids"])
    np.savez_compressed(paths["test"], X=splits["test"]["X"], y=splits["test"]["y"], patient_ids=splits["test"]["patient_ids"])

    for c_name, c_data in clients.items():
        np.savez_compressed(paths[c_name], X=c_data["X"], y=c_data["y"], patient_ids=c_data["patient_ids"])

    with open(paths["scaler"], "w") as f:
        json.dump(scaler, f, indent=4)

    # Export patient-level summary CSV
    summary_rows = []
    for pid in raw_data["patient_ids"]:
        sub = raw_data["trajectories_df"][raw_data["trajectories_df"]["patient_id"] == pid]
        summary_rows.append({
            "patient_id": pid,
            "age": sub["age"].iloc[0],
            "initial_bp": sub["blood_pressure"].iloc[0],
            "final_bp": sub["blood_pressure"].iloc[-1],
            "mean_glucose": round(sub["glucose_level"].mean(), 1),
            "final_bmi": sub["bmi"].iloc[-1],
            "mean_cholesterol": round(sub["cholesterol_level"].mean(), 1),
            "risk_label": int(sub["risk_label"].iloc[0])
        })
    pd.DataFrame(summary_rows).to_csv(paths["summary_csv"], index=False)

    return paths


if __name__ == "__main__":
    paths = prepare_and_save_temporal_data()
    print("Temporal dataset generated successfully:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


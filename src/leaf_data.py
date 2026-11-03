"""
LEAF-FL Dataset Module: Future Healthcare Deterioration Forecasting.

This module generates, preprocesses, and partitions synthetic longitudinal patient data
specifically for the early warning forecasting task:
- Observation Window: Months 1 to 6 (t = 0 to 5) - model input sequences (N, 6, 5).
- Forecast Horizon: Months 7 to 12 (t = 6 to 11) - future clinical deterioration window.
- Clinical Deterioration Event Definition:
  A binary event (Y = 1) occurring if during months 7 to 12 the patient experiences:
    1. Acute Hypertensive Crisis: Systolic BP >= 155.0 mmHg, OR
    2. Severe Hyperglycemic Spike: Fasting Glucose >= 160.0 mg/dL, OR
    3. Sustained Hypertension: Systolic BP >= 142.0 mmHg for >= 2 consecutive months, OR
    4. Sustained Hyperglycemia: Fasting Glucose >= 145.0 mg/dL for >= 2 consecutive months.
  Event month T_event in {7, 8, 9, 10, 11, 12} records the earliest onset month.
- Hospital Partitions with Unequal Cohort Sizes & Clinical Equity Gaps:
  - Hospital 1 (Specialty Referral): Large cohort (320 patients, ~58% deterioration).
  - Hospital 2 (Community Hospital): Medium cohort (240 patients, ~42% deterioration).
  - Hospital 3 (Rural Clinic): Resource-constrained, small cohort (140 patients, ~24% deterioration).
- Zero Data Leakage:
  - Patient IDs are strictly disjoint across train, val, and test splits.
  - Shared feature scaler is fitted strictly on observation window (months 1-6) of training patients.
  - Future observations (months 7-12) are NEVER provided as model input.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_NAMES = [
    "age",
    "blood_pressure",
    "glucose_level",
    "bmi",
    "cholesterol_level"
]

OBSERVATION_WINDOW = 6  # Months 1 to 6 (Input)
FORECAST_HORIZON = 6     # Months 7 to 12 (Prediction Horizon)
TOTAL_MONTHS = 12


def generate_deterioration_dataset(
    num_patients: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Generate synthetic longitudinal records and evaluate deterioration events in the forecast window.

    Parameters:
        num_patients (int): Total patient trajectories (default: 1,000).
        random_seed (int): Deterministic RNG seed.

    Returns:
        dict:
            - "X_obs": np.ndarray of shape (num_patients, 6, 5) - Observation window features.
            - "X_full": np.ndarray of shape (num_patients, 12, 5) - Complete 12-month trajectory.
            - "y": np.ndarray of shape (num_patients, 1) - Binary deterioration indicator.
            - "event_month": np.ndarray of shape (num_patients,) - Month of event (7-12) or 0 if none.
            - "patient_ids": List[str] - Unique patient identifiers.
            - "metadata": dict - Summary statistics of deterioration rates.
    """
    rng = np.random.RandomState(random_seed)

    base_age = rng.normal(loc=55.0, scale=11.5, size=num_patients)
    base_age = np.clip(base_age, 26.0, 84.0).round().astype(float)

    base_bp = rng.normal(loc=125.0, scale=13.5, size=num_patients)
    base_bp = np.clip(base_bp, 95.0, 168.0)

    base_glu = rng.normal(loc=106.0, scale=22.0, size=num_patients)
    base_glu = np.clip(base_glu, 72.0, 210.0)

    base_bmi = rng.normal(loc=27.0, scale=4.6, size=num_patients)
    base_bmi = np.clip(base_bmi, 18.5, 43.0)

    base_chol = rng.normal(loc=200.0, scale=30.0, size=num_patients)
    base_chol = np.clip(base_chol, 130.0, 305.0)

    # Latent progression velocity (controls acceleration in the forecast window)
    progression_propensity = rng.beta(a=2.0, b=2.8, size=num_patients)

    X_full = np.zeros((num_patients, TOTAL_MONTHS, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros((num_patients, 1), dtype=np.float32)
    event_month = np.zeros(num_patients, dtype=np.int32)
    patient_ids = [f"PID_DET_{i+1:04d}" for i in range(num_patients)]

    for i in range(num_patients):
        prop = progression_propensity[i]
        cur_bp = base_bp[i]
        cur_glu = base_glu[i]
        cur_bmi = base_bmi[i]
        cur_chol = base_chol[i]

        bp_trend = (prop - 0.40) * 2.2
        glu_trend = (prop - 0.38) * 3.0
        bmi_trend = (prop - 0.45) * 0.12

        bp_seq = np.zeros(TOTAL_MONTHS)
        glu_seq = np.zeros(TOTAL_MONTHS)

        for m in range(TOTAL_MONTHS):
            # AR(1) noise + progressive drift
            ar_bp = 0.55 * (cur_bp - base_bp[i]) + bp_trend + rng.normal(0.0, 2.5)
            # In forecast window (m >= 6), high propensity patients experience acute deterioration
            if m >= OBSERVATION_WINDOW and prop > 0.50:
                acute_bp = (prop - 0.50) * 8.0 * (m - OBSERVATION_WINDOW + 1)
                ar_bp += acute_bp
            cur_bp = float(np.clip(base_bp[i] + ar_bp, 85.0, 210.0))
            bp_seq[m] = cur_bp

            ar_glu = 0.50 * (cur_glu - base_glu[i]) + glu_trend + rng.normal(0.0, 4.0)
            if m >= OBSERVATION_WINDOW and prop > 0.48:
                spike = 30.0 if rng.rand() < (0.15 + 0.35 * prop) else 10.0 * (prop - 0.48)
                ar_glu += spike
            cur_glu = float(np.clip(base_glu[i] + ar_glu, 65.0, 295.0))
            glu_seq[m] = cur_glu

            cur_bmi = float(np.clip(base_bmi[i] + m * bmi_trend + rng.normal(0.0, 0.18), 17.0, 48.0))
            cur_chol = float(np.clip(base_chol[i] + 5.0 * np.sin(2 * np.pi * m / 12) + rng.normal(0.0, 4.5), 120.0, 330.0))

            X_full[i, m, 0] = base_age[i]
            X_full[i, m, 1] = cur_bp
            X_full[i, m, 2] = cur_glu
            X_full[i, m, 3] = cur_bmi
            X_full[i, m, 4] = cur_chol

        # Evaluate deterioration criteria in the forecast window: months 7 to 12 (indices 6 to 11)
        det_found = False
        onset_step = 0

        for m in range(OBSERVATION_WINDOW, TOTAL_MONTHS):
            bp_m = bp_seq[m]
            glu_m = glu_seq[m]

            # Criterion 1: Acute Hypertensive Crisis (>= 155 mmHg)
            crit1 = (bp_m >= 155.0)
            # Criterion 2: Severe Hyperglycemic Spike (>= 160 mg/dL)
            crit2 = (glu_m >= 160.0)
            # Criterion 3: Sustained Hypertension (BP >= 142 for 2 consecutive steps)
            crit3 = False
            if m > OBSERVATION_WINDOW:
                crit3 = (bp_seq[m-1] >= 142.0 and bp_m >= 142.0)
            # Criterion 4: Sustained Hyperglycemia (Glucose >= 145 for 2 consecutive steps)
            crit4 = False
            if m > OBSERVATION_WINDOW:
                crit4 = (glu_seq[m-1] >= 145.0 and glu_m >= 145.0)

            if crit1 or crit2 or crit3 or crit4:
                det_found = True
                onset_step = m + 1  # 1-indexed month (7 to 12)
                break

        if det_found:
            y[i, 0] = 1.0
            event_month[i] = onset_step
        else:
            y[i, 0] = 0.0
            event_month[i] = 0

    # Model input is strictly observation window: months 1 to 6 (t = 0 to 5)
    X_obs = X_full[:, :OBSERVATION_WINDOW, :].copy()

    metadata = {
        "num_patients": num_patients,
        "observation_window_months": OBSERVATION_WINDOW,
        "forecast_horizon_months": FORECAST_HORIZON,
        "total_deterioration_events": int(np.sum(y)),
        "event_rate": float(np.mean(y)),
        "event_month_distribution": {
            int(m): int(np.sum(event_month == m)) for m in range(7, 13)
        }
    }

    return {
        "X_obs": X_obs,
        "X_full": X_full,
        "y": y,
        "event_month": event_month,
        "patient_ids": patient_ids,
        "metadata": metadata
    }


def compute_observation_scaler(X_train: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std normalization parameters exclusively on the training observation sequences.
    X_train shape: (N_train, 6, 5).
    """
    N, T, F = X_train.shape
    X_flat = X_train.reshape(-1, F)

    means = {FEATURE_NAMES[f]: float(np.mean(X_flat[:, f])) for f in range(F)}
    stds = {FEATURE_NAMES[f]: float(max(1e-7, np.std(X_flat[:, f]))) for f in range(F)}

    return {"mean": means, "std": stds}


def apply_observation_scaler(X: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    """Normalize 3D feature tensor (N, T, 5) using pre-computed scaling parameters."""
    X_norm = X.copy()
    F = X.shape[2]
    for f in range(F):
        feat = FEATURE_NAMES[f]
        m = scaler["mean"].get(feat, 0.0)
        s = scaler["std"].get(feat, 1.0)
        s = s if s > 1e-12 else 1.0
        X_norm[:, :, f] = (X_norm[:, :, f] - m) / s
    return X_norm.astype(np.float32)


def partition_leaf_dataset(
    dataset: Dict[str, Any],
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Partition patient trajectories into Central Test (15%), Central Val (15%),
    and 3 Non-IID Hospital Clients with heterogeneous cohort sizes and deterioration rates:
    - Hospital 1 (Specialty): 320 patients (260 train, 60 local val), ~58% positive
    - Hospital 2 (Community): 240 patients (190 train, 50 local val), ~42% positive
    - Hospital 3 (Rural/Clinic): 140 patients (110 train, 30 local val), ~24% positive
    """
    rng = np.random.RandomState(random_seed)
    N = dataset["X_obs"].shape[0]
    indices = np.arange(N)

    # Separate positive and negative indices for stratified holdout
    pos_idx = indices[dataset["y"].ravel() == 1.0]
    neg_idx = indices[dataset["y"].ravel() == 0.0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Reserve 150 test (15%) and 150 central val (15%)
    n_test = 150
    n_test_pos = int(round(n_test * np.mean(dataset["y"])))
    n_test_neg = n_test - n_test_pos

    test_pos = pos_idx[:n_test_pos]
    test_neg = neg_idx[:n_test_neg]
    test_idx = np.concatenate([test_pos, test_neg])
    rng.shuffle(test_idx)

    rem_pos = pos_idx[n_test_pos:]
    rem_neg = neg_idx[n_test_neg:]

    n_val = 150
    n_val_pos = int(round(n_val * np.mean(dataset["y"])))
    n_val_neg = n_val - n_val_pos

    val_pos = rem_pos[:n_val_pos]
    val_neg = rem_neg[:n_val_neg]
    val_idx = np.concatenate([val_pos, val_neg])
    rng.shuffle(val_idx)

    # Remaining 700 patients assigned to 3 hospitals with controlled skew
    train_pool_pos = rem_pos[n_val_pos:]
    train_pool_neg = rem_neg[n_val_neg:]

    # Target allocations:
    # H1: 320 total (185 pos, 135 neg) -> ~57.8% positive
    # H2: 240 total (100 pos, 140 neg) -> ~41.7% positive
    # H3: 140 total (35 pos, 105 neg) -> ~25.0% positive
    h1_pos_count = min(185, len(train_pool_pos) - 135)
    h2_pos_count = 100
    h3_pos_count = len(train_pool_pos) - (h1_pos_count + h2_pos_count)

    h1_neg_count = 320 - h1_pos_count
    h2_neg_count = 240 - h2_pos_count
    h3_neg_count = len(train_pool_neg) - (h1_neg_count + h2_neg_count)

    h1_idx = np.concatenate([train_pool_pos[:h1_pos_count], train_pool_neg[:h1_neg_count]])
    h2_idx = np.concatenate([
        train_pool_pos[h1_pos_count:h1_pos_count + h2_pos_count],
        train_pool_neg[h1_neg_count:h1_neg_count + h2_neg_count]
    ])
    h3_idx = np.concatenate([
        train_pool_pos[h1_pos_count + h2_pos_count:],
        train_pool_neg[h1_neg_count + h2_neg_count:]
    ])

    rng.shuffle(h1_idx)
    rng.shuffle(h2_idx)
    rng.shuffle(h3_idx)

    # Split each hospital into local train and local val:
    # H1: 260 train, 60 val
    # H2: 190 train, 50 val
    # H3: 110 train, 30 val
    h1_train_idx, h1_val_idx = h1_idx[:260], h1_idx[260:]
    h2_train_idx, h2_val_idx = h2_idx[:190], h2_idx[190:]
    h3_train_idx, h3_val_idx = h3_idx[:110], h3_idx[110:]

    # Compute scaler strictly on all hospital training observations (260 + 190 + 110 = 560 patients)
    all_train_idx = np.concatenate([h1_train_idx, h2_train_idx, h3_train_idx])
    scaler = compute_observation_scaler(dataset["X_obs"][all_train_idx])

    def build_split(idx_array):
        return {
            "X": apply_observation_scaler(dataset["X_obs"][idx_array], scaler),
            "X_raw": dataset["X_obs"][idx_array],
            "y": dataset["y"][idx_array].astype(np.float32),
            "event_month": dataset["event_month"][idx_array],
            "patient_ids": [dataset["patient_ids"][i] for i in idx_array],
            "num_samples": len(idx_array),
            "event_rate": float(np.mean(dataset["y"][idx_array]))
        }

    return {
        "scaler": scaler,
        "test": build_split(test_idx),
        "val": build_split(val_idx),
        "hospitals": {
            "hospital_1": {
                "name": "Hospital 1 (Specialty Center)",
                "train": build_split(h1_train_idx),
                "val": build_split(h1_val_idx),
                "total_patients": len(h1_idx)
            },
            "hospital_2": {
                "name": "Hospital 2 (Community Hospital)",
                "train": build_split(h2_train_idx),
                "val": build_split(h2_val_idx),
                "total_patients": len(h2_idx)
            },
            "hospital_3": {
                "name": "Hospital 3 (Rural Clinic)",
                "train": build_split(h3_train_idx),
                "val": build_split(h3_val_idx),
                "total_patients": len(h3_idx)
            }
        }
    }


def prepare_and_save_leaf_data(
    output_dir: str = os.path.join("data", "leaf"),
    random_seed: int = 42
) -> Dict[str, Any]:
    """Generate, partition, and serialize LEAF-FL datasets."""
    os.makedirs(output_dir, exist_ok=True)
    raw_data = generate_deterioration_dataset(num_patients=1000, random_seed=random_seed)
    partitioned = partition_leaf_dataset(raw_data, random_seed=random_seed)

    # Save scaler
    scaler_path = os.path.join(output_dir, "leaf_scaler.json")
    with open(scaler_path, "w") as f:
        json.dump(partitioned["scaler"], f, indent=4)

    # Save central test & val
    np.savez_compressed(
        os.path.join(output_dir, "leaf_test.npz"),
        X=partitioned["test"]["X"],
        y=partitioned["test"]["y"],
        event_month=partitioned["test"]["event_month"],
        patient_ids=np.array(partitioned["test"]["patient_ids"])
    )
    np.savez_compressed(
        os.path.join(output_dir, "leaf_val.npz"),
        X=partitioned["val"]["X"],
        y=partitioned["val"]["y"],
        event_month=partitioned["val"]["event_month"],
        patient_ids=np.array(partitioned["val"]["patient_ids"])
    )

    # Save hospital client partitions
    for h_id, h_data in partitioned["hospitals"].items():
        np.savez_compressed(
            os.path.join(output_dir, f"{h_id}_train.npz"),
            X=h_data["train"]["X"],
            y=h_data["train"]["y"],
            event_month=h_data["train"]["event_month"],
            patient_ids=np.array(h_data["train"]["patient_ids"])
        )
        np.savez_compressed(
            os.path.join(output_dir, f"{h_id}_val.npz"),
            X=h_data["val"]["X"],
            y=h_data["val"]["y"],
            event_month=h_data["val"]["event_month"],
            patient_ids=np.array(h_data["val"]["patient_ids"])
        )

    # Save patient summary CSV for exploration
    summary_rows = []
    for i in range(1000):
        summary_rows.append({
            "patient_id": raw_data["patient_ids"][i],
            "age": float(raw_data["X_full"][i, 0, 0]),
            "mean_obs_bp": float(np.mean(raw_data["X_obs"][i, :, 1])),
            "final_obs_bp": float(raw_data["X_obs"][i, -1, 1]),
            "mean_obs_glu": float(np.mean(raw_data["X_obs"][i, :, 2])),
            "final_obs_glu": float(raw_data["X_obs"][i, -1, 2]),
            "deterioration_event": int(raw_data["y"][i, 0]),
            "event_month": int(raw_data["event_month"][i])
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "patient_deterioration_summary.csv"), index=False)

    return partitioned


def load_leaf_dataset_from_disk(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load previously partitioned LEAF-FL datasets from data/leaf/, or generate if absent."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "leaf")

    scaler_path = os.path.join(data_dir, "leaf_scaler.json")
    test_path = os.path.join(data_dir, "leaf_test.npz")
    val_path = os.path.join(data_dir, "leaf_val.npz")

    if not (os.path.exists(scaler_path) and os.path.exists(test_path) and os.path.exists(val_path)):
        return prepare_and_save_leaf_data(output_dir=data_dir)

    with open(scaler_path, "r") as f:
        scaler = json.load(f)

    test_npz = np.load(test_path)
    val_npz = np.load(val_path)

    test_dict = {
        "X": test_npz["X"],
        "y": test_npz["y"],
        "event_month": test_npz["event_month"],
        "patient_ids": test_npz["patient_ids"].tolist(),
        "num_samples": len(test_npz["y"]),
        "event_rate": float(np.mean(test_npz["y"]))
    }

    val_dict = {
        "X": val_npz["X"],
        "y": val_npz["y"],
        "event_month": val_npz["event_month"],
        "patient_ids": val_npz["patient_ids"].tolist(),
        "num_samples": len(val_npz["y"]),
        "event_rate": float(np.mean(val_npz["y"]))
    }

    hospitals = {}
    h_configs = [
        ("hospital_1", "Hospital 1 (Specialty Center)"),
        ("hospital_2", "Hospital 2 (Community General)"),
        ("hospital_3", "Hospital 3 (Rural/Clinic)")
    ]
    for h_id, h_name in h_configs:
        tr_npz = np.load(os.path.join(data_dir, f"{h_id}_train.npz"))
        va_npz = np.load(os.path.join(data_dir, f"{h_id}_val.npz"))
        hospitals[h_id] = {
            "name": h_name,
            "train": {
                "X": tr_npz["X"],
                "y": tr_npz["y"],
                "event_month": tr_npz["event_month"],
                "patient_ids": tr_npz["patient_ids"].tolist(),
                "num_samples": len(tr_npz["y"]),
                "event_rate": float(np.mean(tr_npz["y"]))
            },
            "val": {
                "X": va_npz["X"],
                "y": va_npz["y"],
                "event_month": va_npz["event_month"],
                "patient_ids": va_npz["patient_ids"].tolist(),
                "num_samples": len(va_npz["y"]),
                "event_rate": float(np.mean(va_npz["y"]))
            }
        }

    return {
        "scaler": scaler,
        "test": test_dict,
        "val": val_dict,
        "hospitals": hospitals
    }


if __name__ == "__main__":
    print("Generating and verifying LEAF-FL Deterioration Forecasting Dataset...")
    data = prepare_and_save_leaf_data()
    print("LEAF-FL Dataset generation complete.")
    for h_id, info in data["hospitals"].items():
        tr = info["train"]
        va = info["val"]
        print(f"[{h_id.upper()}] {info['name']}: Train N={tr['num_samples']} (Event Rate={tr['event_rate']:.1%}), Val N={va['num_samples']} (Event Rate={va['event_rate']:.1%})")
    te = data["test"]
    print(f"[TEST] Central Holdout Test: N={te['num_samples']} (Event Rate={te['event_rate']:.1%})")

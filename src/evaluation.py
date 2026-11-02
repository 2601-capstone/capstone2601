"""
Evaluation and Baseline Comparison Module for Federated Healthcare Analytics.

This module provides comprehensive benchmarking on the isolated central test dataset:
1. Evaluates the Global Federated Model against traditional baselines:
   - Centralized Baseline: Model trained on pooled hospital data (hypothetical non-private ceiling).
   - Hospital 1 Local Baseline: Model trained strictly on Hospital 1 data (specialty clinic).
   - Hospital 2 Local Baseline: Model trained strictly on Hospital 2 data (community hospital).
   - Hospital 3 Local Baseline: Model trained strictly on Hospital 3 data (primary clinic).
2. Calculates standard clinical classification metrics:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
3. Persists evaluation tables to CSV and JSON formats in results/metrics/.

NOTE: All evaluations are conducted on synthetic data. This simulation demonstrates
privacy-preserving trade-offs and does not represent real clinical diagnoses.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Ensure project root is accessible on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import HealthcareMLP
from src.experiment import run_federated_experiment
from src.data_preparation import (
    TARGET_COLUMN,
    load_central_test_data,
    load_hospital_data,
    load_scaler,
    apply_scaler
)


def load_preprocessed_test_data(data_dir: str = "data") -> tuple:
    """
    Load the central holdout evaluation test dataset and apply the shared training scaler.

    Returns:
        tuple (np.ndarray, np.ndarray): (X_test, y_test)
            - X_test: Standardized feature matrix of shape (300, 5).
            - y_test: Ground truth binary labels of shape (300,).
    """
    test_df = load_central_test_data(data_dir=data_dir)
    scaler_path = os.path.join(data_dir, "scaler_params.json")
    scaler = load_scaler(filepath=scaler_path)

    X_test = apply_scaler(test_df, scaler)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=int).ravel()

    return X_test, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute standard classification evaluation metrics.

    Parameters:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_pred (np.ndarray): Predicted binary labels (0 or 1).
        y_prob (np.ndarray): Predicted probabilities in [0.0, 1.0].

    Returns:
        dict: Metric scores including accuracy, precision, recall, f1, roc_auc, and confusion_matrix.
    """
    y_true_flat = np.asarray(y_true, dtype=int).ravel()
    y_pred_flat = np.asarray(y_pred, dtype=int).ravel()
    y_prob_flat = np.asarray(y_prob, dtype=float).ravel()

    acc = float(accuracy_score(y_true_flat, y_pred_flat))
    prec = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))
    rec = float(recall_score(y_true_flat, y_pred_flat, zero_division=0))
    f1 = float(f1_score(y_true_flat, y_pred_flat, zero_division=0))

    # ROC-AUC requires at least two distinct classes present in y_true
    unique_classes = np.unique(y_true_flat)
    if len(unique_classes) >= 2:
        auc = float(roc_auc_score(y_true_flat, y_prob_flat))
    else:
        auc = None

    cm = confusion_matrix(y_true_flat, y_pred_flat).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm
    }


def evaluate_model(model: HealthcareMLP, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a HealthcareMLP model on the preprocessed test set.

    Returns:
        dict: Calculated metrics dictionary.
    """
    y_prob = model.predict_proba(X_test).ravel()
    y_pred = model.predict(X_test).ravel()
    return calculate_metrics(y_test, y_pred, y_prob)


def train_centralized_model(
    scaler: dict,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    random_seed: int = 42,
    data_dir: str = "data"
) -> HealthcareMLP:
    """
    Train a centralized baseline model on the pooled training records from all 3 hospitals.
    Represents the theoretical upper performance bound without data partitioning.

    Returns:
        HealthcareMLP: Trained centralized model.
    """
    h1 = load_hospital_data("hospital_1", data_dir=data_dir)
    h2 = load_hospital_data("hospital_2", data_dir=data_dir)
    h3 = load_hospital_data("hospital_3", data_dir=data_dir)

    combined_df = pd.concat([h1, h2, h3], ignore_index=True)
    X_train = apply_scaler(combined_df, scaler)
    y_train = combined_df[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)

    model = HealthcareMLP(
        input_dim=5,
        hidden_dim=8,
        output_dim=1,
        random_seed=random_seed
    )
    model.fit(
        X=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=False
    )
    return model


def train_isolated_local_model(
    hospital_id: str,
    scaler: dict,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    random_seed: int = 42,
    data_dir: str = "data"
) -> HealthcareMLP:
    """
    Train an isolated local baseline model using only a single hospital's private dataset.
    Demonstrates model performance when a hospital refuses to participate in collaboration.

    Returns:
        HealthcareMLP: Trained local hospital model.
    """
    df = load_hospital_data(hospital_id, data_dir=data_dir)
    X_train = apply_scaler(df, scaler)
    y_train = df[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)

    model = HealthcareMLP(
        input_dim=5,
        hidden_dim=8,
        output_dim=1,
        random_seed=random_seed
    )
    model.fit(
        X=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=False
    )
    return model


def run_full_evaluation(
    num_federated_rounds: int = 5,
    epochs_per_baseline: int = 50,
    random_seed: int = 42,
    data_dir: str = "data",
    output_dir: str = os.path.join("results", "metrics")
) -> tuple:
    """
    Benchmark the 5 models on the identical isolated central test dataset:
    1. Federated Global Model (FedAvg)
    2. Centralized Baseline (Pooled Data)
    3. Hospital 1 Local Baseline (Isolated)
    4. Hospital 2 Local Baseline (Isolated)
    5. Hospital 3 Local Baseline (Isolated)

    Returns:
        tuple (pd.DataFrame, dict): (comparison_df, detailed_summary)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 68)
    print("PHASE 5: GLOBAL MODEL EVALUATION & BASELINE COMPARISON")
    print("=" * 68)

    # 1. Load test data and scaler
    print("\n1. Loading and preprocessing isolated central test dataset (300 samples)...")
    X_test, y_test = load_preprocessed_test_data(data_dir=data_dir)
    scaler_path = os.path.join(data_dir, "scaler_params.json")
    scaler = load_scaler(filepath=scaler_path)
    print(f"   Test samples: {len(y_test)} | Class 0: {(y_test == 0).sum()} | Class 1: {(y_test == 1).sum()}")

    # 2. Retrieve Federated Global Model
    print(f"\n2. Training/retrieving Federated Global Model ({num_federated_rounds} communication rounds)...")
    fed_history_path = os.path.join(output_dir, "federated_training_history.csv")
    federated_model, _ = run_federated_experiment(
        num_rounds=num_federated_rounds,
        local_epochs=10,
        batch_size=32,
        learning_rate=0.01,
        random_seed=random_seed,
        save_history=True,
        history_output_path=fed_history_path
    )

    # 3. Train Centralized Baseline (Pooled 1,200 training samples)
    print(f"\n3. Training Centralized Baseline (1,200 pooled samples, {epochs_per_baseline} epochs)...")
    centralized_model = train_centralized_model(
        scaler=scaler,
        epochs=epochs_per_baseline,
        batch_size=32,
        learning_rate=0.01,
        random_seed=random_seed,
        data_dir=data_dir
    )

    # 4. Train Isolated Local Baselines (400 samples each)
    print(f"\n4. Training Isolated Local Hospital Baselines (400 samples each, {epochs_per_baseline} epochs)...")
    h1_model = train_isolated_local_model("hospital_1", scaler, epochs=epochs_per_baseline, random_seed=random_seed, data_dir=data_dir)
    h2_model = train_isolated_local_model("hospital_2", scaler, epochs=epochs_per_baseline, random_seed=random_seed, data_dir=data_dir)
    h3_model = train_isolated_local_model("hospital_3", scaler, epochs=epochs_per_baseline, random_seed=random_seed, data_dir=data_dir)

    # 5. Evaluate all models on central test set
    models_to_evaluate = {
        "Federated Global Model": federated_model,
        "Centralized Baseline": centralized_model,
        "Hospital 1 Local Baseline": h1_model,
        "Hospital 2 Local Baseline": h2_model,
        "Hospital 3 Local Baseline": h3_model
    }

    print("\n5. Computing evaluation metrics on isolated central test set...")
    comparison_rows = []
    detailed_summary = {
        "evaluation_dataset": {
            "source": os.path.join(data_dir, "central_test.csv"),
            "num_samples": len(y_test),
            "class_distribution": {
                "class_0": int((y_test == 0).sum()),
                "class_1": int((y_test == 1).sum())
            }
        },
        "models": {}
    }

    for model_name, model_instance in models_to_evaluate.items():
        metrics = evaluate_model(model_instance, X_test, y_test)
        detailed_summary["models"][model_name] = metrics

        comparison_rows.append({
            "Model": model_name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1_Score": round(metrics["f1_score"], 4),
            "ROC_AUC": round(metrics["roc_auc"], 4) if metrics["roc_auc"] is not None else None
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # 6. Save results to CSV and JSON
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    json_path = os.path.join(output_dir, "evaluation_summary.json")

    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(detailed_summary, f, indent=4)

    print(f"\n   Saved comparison table to: {csv_path}")
    print(f"   Saved detailed summary to: {json_path}")

    print("\n" + "=" * 68)
    print("EVALUATION RESULTS SUMMARY (Tested on Central Test Set)")
    print("=" * 68)
    print(comparison_df.to_string(index=False))
    print("=" * 68)

    return comparison_df, detailed_summary


if __name__ == "__main__":
    run_full_evaluation()

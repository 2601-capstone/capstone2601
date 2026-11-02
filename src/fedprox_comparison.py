"""
FedAvg vs. FedProx Comparative Analysis for Federated Healthcare Analytics.

This module conducts a fair, reproducible experimental comparison between:
1. Sample-Weighted Federated Averaging (FedAvg) [McMahan et al., 2017]
2. Federated Proximal Optimization (FedProx) [Li et al., 2020]

Both algorithms are evaluated under strictly controlled, identical conditions:
- Identical synthetic healthcare dataset (1,500 records)
- Identical Non-IID hospital client partitions (H1: 65%, H2: 50%, H3: 39% risk)
- Identical isolated central holdout test set (300 records)
- Identical client demographic holdout partitions (100 records per hospital)
- Identical model architecture (HealthcareMLP: 5 -> 8 -> 1)
- Identical initialization seeds, communication rounds, local epochs, and learning rates
"""

import os
import sys
import json
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Ensure project root is accessible on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiment import run_federated_experiment
from src.data_preparation import (
    TARGET_COLUMN,
    load_central_test_data,
    load_scaler,
    apply_scaler
)
from src.robustness import (
    evaluate_predictions,
    calculate_robustness_statistics,
    get_or_create_hospital_test_sets
)
from src.communication_analysis import (
    calculate_model_parameter_bytes,
    calculate_round_communication_volume
)


def evaluate_model_on_central_test(
    model,
    data_dir: str = "data"
) -> Dict[str, Any]:
    """
    Evaluate a trained global model on the 300-sample isolated central holdout test set.

    Returns:
        dict: Classification metrics (accuracy, precision, recall, f1_score, roc_auc, confusion_matrix).
    """
    scaler_path = os.path.join(data_dir, "scaler_params.json")
    scaler = load_scaler(scaler_path)
    test_df = load_central_test_data(data_dir=data_dir)

    X_test = apply_scaler(test_df, scaler)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=int).ravel()

    probs = model.predict_proba(X_test).ravel()
    preds = model.predict(X_test).ravel()

    return evaluate_predictions(y_test, preds, probs)


def evaluate_model_client_robustness(
    model,
    data_dir: str = "data",
    random_seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a trained global model on each hospital's local demographic holdout partition (100 records each).

    Returns:
        tuple (hospital_metrics, robustness_statistics)
    """
    scaler_path = os.path.join(data_dir, "scaler_params.json")
    scaler = load_scaler(scaler_path)
    hospital_tests = get_or_create_hospital_test_sets(data_dir=data_dir, random_seed=random_seed)

    hospital_metrics = {}
    for h_id, df_test in hospital_tests.items():
        X_h = apply_scaler(df_test, scaler)
        y_h = df_test[TARGET_COLUMN].to_numpy(dtype=int).ravel()

        probs = model.predict_proba(X_h).ravel()
        preds = model.predict(X_h).ravel()
        hospital_metrics[h_id] = evaluate_predictions(y_h, preds, probs)

    stats = calculate_robustness_statistics(hospital_metrics)
    return hospital_metrics, stats


def run_fedavg_vs_fedprox_comparison(
    num_rounds: int = 5,
    local_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    mu: float = 0.01,
    random_seed: int = 42,
    data_dir: str = "data",
    output_dir: str = os.path.join("results", "metrics")
) -> Dict[str, Any]:
    """
    Execute end-to-end fair experimental comparison between FedAvg and FedProx.

    Parameters:
        num_rounds (int): Communication rounds (default: 5).
        local_epochs (int): Local client epochs per round (default: 10).
        batch_size (int): Mini-batch size (default: 32).
        learning_rate (float): Optimization step size (default: 0.01).
        mu (float): FedProx proximal coefficient (default: 0.01).
        random_seed (int): Shared initialization seed (default: 42).
        data_dir (str): Directory containing synthetic hospital datasets.
        output_dir (str): Directory where comparison artifacts will be saved.

    Returns:
        dict: Complete comparison metrics, histories, and summary tables.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 68)
    print("PHASE 10: FEDAVG VS. FEDPROX ALGORITHMIC COMPARISON")
    print(f"Rounds: {num_rounds} | Local Epochs: {local_epochs} | LR: {learning_rate} | FedProx Mu: {mu}")
    print("=" * 68)

    # -------------------------------------------------------------
    # 1. Train FedAvg (mu = 0.0)
    # -------------------------------------------------------------
    print("\n>>> [1/2] Training Federated Averaging (FedAvg) Baseline...")
    t0_fedavg = time.perf_counter()
    fedavg_history_path = os.path.join(output_dir, "federated_training_history.csv")
    model_fedavg, history_fedavg = run_federated_experiment(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        save_history=True,
        history_output_path=fedavg_history_path,
        algorithm="FedAvg",
        mu=0.0
    )
    fedavg_train_time = time.perf_counter() - t0_fedavg

    # -------------------------------------------------------------
    # 2. Train FedProx (mu > 0.0)
    # -------------------------------------------------------------
    print(f"\n>>> [2/2] Training FedProx with Proximal Term (mu = {mu})...")
    t0_fedprox = time.perf_counter()
    fedprox_history_path = os.path.join(output_dir, "fedprox_training_history.csv")
    model_fedprox, history_fedprox = run_federated_experiment(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        save_history=True,
        history_output_path=fedprox_history_path,
        algorithm="FedProx",
        mu=mu
    )
    fedprox_train_time = time.perf_counter() - t0_fedprox

    # -------------------------------------------------------------
    # 3. Evaluate both on Central Test Set (300 records)
    # -------------------------------------------------------------
    print("\n>>> Computing Central Holdout Test Set Evaluations...")
    fedavg_central_metrics = evaluate_model_on_central_test(model_fedavg, data_dir=data_dir)
    fedprox_central_metrics = evaluate_model_on_central_test(model_fedprox, data_dir=data_dir)

    # -------------------------------------------------------------
    # 4. Evaluate Client-Level Robustness on Local Demographic Partitions
    # -------------------------------------------------------------
    print(">>> Computing Client-Level Robustness & Dispersion Across Hospitals...")
    fedavg_client_metrics, fedavg_robustness = evaluate_model_client_robustness(
        model_fedavg, data_dir=data_dir, random_seed=random_seed
    )
    fedprox_client_metrics, fedprox_robustness = evaluate_model_client_robustness(
        model_fedprox, data_dir=data_dir, random_seed=random_seed
    )

    # -------------------------------------------------------------
    # 5. Communication & Complexity Analysis
    # -------------------------------------------------------------
    param_info = calculate_model_parameter_bytes(model_fedavg.get_parameters())
    round_comm = calculate_round_communication_volume(model_fedavg.get_parameters(), num_clients=3)
    total_comm_bytes = round_comm["total_communication_bytes"] * num_rounds

    # -------------------------------------------------------------
    # 6. Build Side-by-Side Comparison Table
    # -------------------------------------------------------------
    comparison_rows = [
        {
            "Metric Category": "Central Holdout Evaluation (300 records)",
            "Metric": "Accuracy",
            "FedAvg (mu=0)": round(fedavg_central_metrics["accuracy"], 4),
            "FedProx (mu=0.01)": round(fedprox_central_metrics["accuracy"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_central_metrics["accuracy"] - fedavg_central_metrics["accuracy"], 4)
        },
        {
            "Metric Category": "Central Holdout Evaluation (300 records)",
            "Metric": "Precision",
            "FedAvg (mu=0)": round(fedavg_central_metrics["precision"], 4),
            "FedProx (mu=0.01)": round(fedprox_central_metrics["precision"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_central_metrics["precision"] - fedavg_central_metrics["precision"], 4)
        },
        {
            "Metric Category": "Central Holdout Evaluation (300 records)",
            "Metric": "Recall",
            "FedAvg (mu=0)": round(fedavg_central_metrics["recall"], 4),
            "FedProx (mu=0.01)": round(fedprox_central_metrics["recall"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_central_metrics["recall"] - fedavg_central_metrics["recall"], 4)
        },
        {
            "Metric Category": "Central Holdout Evaluation (300 records)",
            "Metric": "F1-Score",
            "FedAvg (mu=0)": round(fedavg_central_metrics["f1_score"], 4),
            "FedProx (mu=0.01)": round(fedprox_central_metrics["f1_score"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_central_metrics["f1_score"] - fedavg_central_metrics["f1_score"], 4)
        },
        {
            "Metric Category": "Central Holdout Evaluation (300 records)",
            "Metric": "ROC-AUC",
            "FedAvg (mu=0)": round(fedavg_central_metrics["roc_auc"], 4) if fedavg_central_metrics["roc_auc"] else None,
            "FedProx (mu=0.01)": round(fedprox_central_metrics["roc_auc"], 4) if fedprox_central_metrics["roc_auc"] else None,
            "Delta (FedProx - FedAvg)": round(fedprox_central_metrics["roc_auc"] - fedavg_central_metrics["roc_auc"], 4) if fedavg_central_metrics["roc_auc"] and fedprox_central_metrics["roc_auc"] else None
        },
        {
            "Metric Category": "Client Robustness & Fairness (100 records / node)",
            "Metric": "Mean Client Accuracy",
            "FedAvg (mu=0)": round(fedavg_robustness["mean_accuracy"], 4),
            "FedProx (mu=0.01)": round(fedprox_robustness["mean_accuracy"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_robustness["mean_accuracy"] - fedavg_robustness["mean_accuracy"], 4)
        },
        {
            "Metric Category": "Client Robustness & Fairness (100 records / node)",
            "Metric": "Accuracy Variance",
            "FedAvg (mu=0)": round(fedavg_robustness["accuracy_variance"], 6),
            "FedProx (mu=0.01)": round(fedprox_robustness["accuracy_variance"], 6),
            "Delta (FedProx - FedAvg)": round(fedprox_robustness["accuracy_variance"] - fedavg_robustness["accuracy_variance"], 6)
        },
        {
            "Metric Category": "Client Robustness & Fairness (100 records / node)",
            "Metric": "Accuracy Standard Deviation",
            "FedAvg (mu=0)": round(fedavg_robustness["accuracy_std"], 4),
            "FedProx (mu=0.01)": round(fedprox_robustness["accuracy_std"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_robustness["accuracy_std"] - fedavg_robustness["accuracy_std"], 4)
        },
        {
            "Metric Category": "Client Robustness & Fairness (100 records / node)",
            "Metric": "Disparity Gap (Best - Worst Node)",
            "FedAvg (mu=0)": round(fedavg_robustness["accuracy_gap"], 4),
            "FedProx (mu=0.01)": round(fedprox_robustness["accuracy_gap"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_robustness["accuracy_gap"] - fedavg_robustness["accuracy_gap"], 4)
        },
        {
            "Metric Category": "Client Robustness & Fairness (100 records / node)",
            "Metric": "Hospital 1 Accuracy (65% High Risk)",
            "FedAvg (mu=0)": round(fedavg_client_metrics["hospital_1"]["accuracy"], 4),
            "FedProx (mu=0.01)": round(fedprox_client_metrics["hospital_1"]["accuracy"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_client_metrics["hospital_1"]["accuracy"] - fedavg_client_metrics["hospital_1"]["accuracy"], 4)
        },
        {
            "Metric Category": "Client Robustness & Fairness (100 records / node)",
            "Metric": "Hospital 2 Accuracy (50% Balanced)",
            "FedAvg (mu=0)": round(fedavg_client_metrics["hospital_2"]["accuracy"], 4),
            "FedProx (mu=0.01)": round(fedprox_client_metrics["hospital_2"]["accuracy"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_client_metrics["hospital_2"]["accuracy"] - fedavg_client_metrics["hospital_2"]["accuracy"], 4)
        },
        {
            "Metric Category": "Client Robustness & Fairness (100 records / node)",
            "Metric": "Hospital 3 Accuracy (39% Low Risk)",
            "FedAvg (mu=0)": round(fedavg_client_metrics["hospital_3"]["accuracy"], 4),
            "FedProx (mu=0.01)": round(fedprox_client_metrics["hospital_3"]["accuracy"], 4),
            "Delta (FedProx - FedAvg)": round(fedprox_client_metrics["hospital_3"]["accuracy"] - fedavg_client_metrics["hospital_3"]["accuracy"], 4)
        },
        {
            "Metric Category": "Convergence & Computational Efficiency",
            "Metric": "Final Average Local Loss",
            "FedAvg (mu=0)": round(history_fedavg["average_local_loss"].iloc[-1], 4),
            "FedProx (mu=0.01)": round(history_fedprox["average_local_loss"].iloc[-1], 4),
            "Delta (FedProx - FedAvg)": round(history_fedprox["average_local_loss"].iloc[-1] - history_fedavg["average_local_loss"].iloc[-1], 4)
        },
        {
            "Metric Category": "Convergence & Computational Efficiency",
            "Metric": "Total Experiment Time (seconds)",
            "FedAvg (mu=0)": round(fedavg_train_time, 4),
            "FedProx (mu=0.01)": round(fedprox_train_time, 4),
            "Delta (FedProx - FedAvg)": round(fedprox_train_time - fedavg_train_time, 4)
        },
        {
            "Metric Category": "Communication Footprint",
            "Metric": "Model Parameters Count",
            "FedAvg (mu=0)": param_info["total_parameters"],
            "FedProx (mu=0.01)": param_info["total_parameters"],
            "Delta (FedProx - FedAvg)": 0
        },
        {
            "Metric Category": "Communication Footprint",
            "Metric": "Total Communication Volume (Bytes)",
            "FedAvg (mu=0)": total_comm_bytes,
            "FedProx (mu=0.01)": total_comm_bytes,
            "Delta (FedProx - FedAvg)": 0
        }
    ]

    comp_df = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(output_dir, "fedavg_vs_fedprox_comparison.csv")
    comp_df.to_csv(csv_path, index=False)

    summary_payload = {
        "experiment_configuration": {
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "fedprox_mu": mu,
            "random_seed": random_seed
        },
        "central_evaluation": {
            "fedavg": fedavg_central_metrics,
            "fedprox": fedprox_central_metrics
        },
        "client_robustness": {
            "fedavg_clients": fedavg_client_metrics,
            "fedavg_statistics": fedavg_robustness,
            "fedprox_clients": fedprox_client_metrics,
            "fedprox_statistics": fedprox_robustness
        },
        "communication_and_runtime": {
            "parameter_count": param_info["total_parameters"],
            "parameter_bytes": param_info["total_bytes"],
            "total_communication_bytes": total_comm_bytes,
            "fedavg_training_time_seconds": round(fedavg_train_time, 4),
            "fedprox_training_time_seconds": round(fedprox_train_time, 4),
            "fedavg_final_loss": round(float(history_fedavg["average_local_loss"].iloc[-1]), 4),
            "fedprox_final_loss": round(float(history_fedprox["average_local_loss"].iloc[-1]), 4)
        },
        "comparison_table": comparison_rows
    }

    json_path = os.path.join(output_dir, "fedavg_vs_fedprox_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_payload, f, indent=4)

    print("\n" + "=" * 68)
    print("COMPARATIVE EVALUATION SUMMARY TABLE")
    print("=" * 68)
    print(comp_df.to_string(index=False))
    print("=" * 68)
    print(f"Artifacts saved to:\n  - {csv_path}\n  - {json_path}\n  - {fedprox_history_path}")

    return summary_payload


if __name__ == "__main__":
    run_fedavg_vs_fedprox_comparison()

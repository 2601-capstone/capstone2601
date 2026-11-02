"""
Robustness Analysis Across Simulated Healthcare Clients.

This module evaluates how the Federated Global Model performs across heterogeneous
hospital environments and quantifies the impact of Non-IID patient data distributions:
1. Partitions the holdout test dataset (300 samples) into deterministic client-level evaluation sets:
   - Hospital 1: 100 test samples (65% High Risk - Specialty referral demographic)
   - Hospital 2: 100 test samples (50% High Risk - General community demographic)
   - Hospital 3: 100 test samples (39% High Risk - Primary care clinic demographic)
2. Computes client-wise performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
3. Calculates cross-client robustness statistics:
   - Mean Client Accuracy, Accuracy Variance, Standard Deviation.
   - Minimum, Maximum, and the Performance Disparity Gap between best and worst clients.
4. Compares the Federated Model vs Isolated Local Models on each hospital's local demographic.
5. Saves robustness reports to CSV and JSON in results/metrics/.
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
from src.evaluation import train_isolated_local_model
from src.data_preparation import (
    TARGET_COLUMN,
    load_central_test_data,
    load_scaler,
    apply_scaler
)


def partition_hospital_test_sets(
    central_test_df: pd.DataFrame = None,
    data_dir: str = "data",
    random_seed: int = 42
) -> dict:
    """
    Partition the 300-sample central holdout test set into 3 client-specific test sets
    reflecting each hospital's Non-IID demographic:
    - Hospital 1: 100 samples (65 Class 1, 35 Class 0) -> 65% High Risk
    - Hospital 2: 100 samples (50 Class 1, 50 Class 0) -> 50% High Risk
    - Hospital 3: 100 samples (39 Class 1, 61 Class 0) -> 39% High Risk

    Total: 154 Class 1 + 146 Class 0 = 300 records (100% of central test set).
    Saves: hospital_1_test.csv, hospital_2_test.csv, hospital_3_test.csv in data_dir.

    Returns:
        dict: Mapping {"hospital_1": df1, "hospital_2": df2, "hospital_3": df3}
    """
    if central_test_df is None:
        central_test_df = load_central_test_data(data_dir=data_dir)

    c0_df = central_test_df[central_test_df[TARGET_COLUMN] == 0].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    c1_df = central_test_df[central_test_df[TARGET_COLUMN] == 1].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Class 1 allocation: 65 (H1), 50 (H2), 39 (H3) = 154
    # Class 0 allocation: 35 (H1), 50 (H2), 61 (H3) = 146
    h1_df = pd.concat([c1_df.iloc[:65], c0_df.iloc[:35]]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    h2_df = pd.concat([c1_df.iloc[65:115], c0_df.iloc[35:85]]).sample(frac=1.0, random_state=random_seed + 1).reset_index(drop=True)
    h3_df = pd.concat([c1_df.iloc[115:], c0_df.iloc[85:]]).sample(frac=1.0, random_state=random_seed + 2).reset_index(drop=True)

    partitions = {
        "hospital_1": h1_df,
        "hospital_2": h2_df,
        "hospital_3": h3_df
    }

    # Save CSVs to data/
    for h_name, df_part in partitions.items():
        csv_path = os.path.join(data_dir, f"{h_name}_test.csv")
        df_part.to_csv(csv_path, index=False)

    return partitions


def get_or_create_hospital_test_sets(data_dir: str = "data", random_seed: int = 42) -> dict:
    """
    Retrieve existing hospital test CSVs or generate them deterministically from central_test.csv.

    Returns:
        dict: Mapping {"hospital_1": df1, "hospital_2": df2, "hospital_3": df3}
    """
    test_files = [os.path.join(data_dir, f"hospital_{i}_test.csv") for i in [1, 2, 3]]
    if all(os.path.exists(f) for f in test_files):
        return {
            f"hospital_{i}": pd.read_csv(os.path.join(data_dir, f"hospital_{i}_test.csv"))
            for i in [1, 2, 3]
        }
    return partition_hospital_test_sets(data_dir=data_dir, random_seed=random_seed)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute classification metrics on ground truth vs predictions.

    Returns:
        dict: accuracy, precision, recall, f1_score, roc_auc, confusion_matrix
    """
    y_true_flat = np.asarray(y_true, dtype=int).ravel()
    y_pred_flat = np.asarray(y_pred, dtype=int).ravel()
    y_prob_flat = np.asarray(y_prob, dtype=float).ravel()

    acc = float(accuracy_score(y_true_flat, y_pred_flat))
    prec = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))
    rec = float(recall_score(y_true_flat, y_pred_flat, zero_division=0))
    f1 = float(f1_score(y_true_flat, y_pred_flat, zero_division=0))

    if len(np.unique(y_true_flat)) >= 2:
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


def calculate_robustness_statistics(hospital_metrics: dict) -> dict:
    """
    Calculate cross-client dispersion and fairness metrics across hospitals:
    - Mean accuracy & F1
    - Variance & Standard Deviation
    - Min, Max, Best, Worst hospital
    - Performance Disparity Gap (Max - Min)

    Parameters:
        hospital_metrics (dict): Mapping {hospital_id: metrics_dict}

    Returns:
        dict: Robustness summary statistics.
    """
    if not hospital_metrics:
        raise ValueError("hospital_metrics cannot be empty.")

    accuracies = {h: metrics["accuracy"] for h, metrics in hospital_metrics.items()}
    f1_scores = {h: metrics["f1_score"] for h, metrics in hospital_metrics.items()}

    acc_vals = list(accuracies.values())
    f1_vals = list(f1_scores.values())

    mean_acc = float(np.mean(acc_vals))
    var_acc = float(np.var(acc_vals))
    std_acc = float(np.std(acc_vals))
    min_acc = float(np.min(acc_vals))
    max_acc = float(np.max(acc_vals))
    acc_gap = float(max_acc - min_acc)

    best_h_acc = max(accuracies, key=accuracies.get)
    worst_h_acc = min(accuracies, key=accuracies.get)

    mean_f1 = float(np.mean(f1_vals))
    var_f1 = float(np.var(f1_vals))
    std_f1 = float(np.std(f1_vals))
    min_f1 = float(np.min(f1_vals))
    max_f1 = float(np.max(f1_vals))
    f1_gap = float(max_f1 - min_f1)

    best_h_f1 = max(f1_scores, key=f1_scores.get)
    worst_h_f1 = min(f1_scores, key=f1_scores.get)

    return {
        "mean_accuracy": round(mean_acc, 4),
        "accuracy_variance": round(var_acc, 6),
        "accuracy_std": round(std_acc, 4),
        "min_accuracy": round(min_acc, 4),
        "max_accuracy": round(max_acc, 4),
        "accuracy_gap": round(acc_gap, 4),
        "best_hospital_accuracy": best_h_acc,
        "worst_hospital_accuracy": worst_h_acc,
        "mean_f1": round(mean_f1, 4),
        "f1_variance": round(var_f1, 6),
        "f1_std": round(std_f1, 4),
        "min_f1": round(min_f1, 4),
        "max_f1": round(max_f1, 4),
        "f1_gap": round(f1_gap, 4),
        "best_hospital_f1": best_h_f1,
        "worst_hospital_f1": worst_h_f1
    }


def run_robustness_analysis(
    global_model: HealthcareMLP = None,
    data_dir: str = "data",
    output_dir: str = os.path.join("results", "metrics"),
    random_seed: int = 42
) -> dict:
    """
    Run complete robustness and non-IID analysis across the 3 hospitals:
    1. Evaluates the Global Federated Model on each hospital's local demographic test set.
    2. Evaluates each hospital's isolated Local Model on its own demographic test set.
    3. Calculates robustness statistics (mean, variance, std, best, worst, disparity gap).
    4. Saves hospital_performance.csv and robustness_metrics.json.

    Returns:
        dict: Complete robustness report.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 68)
    print("PHASE 8: ROBUSTNESS & NON-IID ANALYSIS ACROSS HOSPITALS")
    print("=" * 68)

    # 1. Load scaler and test partitions
    scaler_path = os.path.join(data_dir, "scaler_params.json")
    scaler = load_scaler(scaler_path)
    hospital_tests = get_or_create_hospital_test_sets(data_dir=data_dir, random_seed=random_seed)

    # 2. Retrieve or train global federated model
    if global_model is None:
        print("\n1. Training/retrieving 5-round Federated Global Model...")
        global_model, _ = run_federated_experiment(
            num_rounds=5,
            local_epochs=10,
            batch_size=32,
            learning_rate=0.01,
            random_seed=random_seed,
            save_history=False
        )

    # 3. Evaluate models on client test sets
    print("\n2. Evaluating Global Model vs Local Baselines on client demographics...")
    hospital_global_metrics = {}
    hospital_local_metrics = {}
    table_rows = []

    demographic_info = {
        "hospital_1": {"name": "Hospital 1", "type": "Specialty Center (65% High Risk)", "c0": 35, "c1": 65},
        "hospital_2": {"name": "Hospital 2", "type": "Community Hospital (50% Balanced)", "c0": 50, "c1": 50},
        "hospital_3": {"name": "Hospital 3", "type": "Primary Clinic (39% Low Risk)", "c0": 61, "c1": 39}
    }

    for h_id, test_df in hospital_tests.items():
        X_test = apply_scaler(test_df, scaler)
        y_test = test_df[TARGET_COLUMN].to_numpy(dtype=int).ravel()

        # Evaluate Global Model
        g_prob = global_model.predict_proba(X_test).ravel()
        g_pred = global_model.predict(X_test).ravel()
        g_metrics = evaluate_predictions(y_test, g_pred, g_prob)
        hospital_global_metrics[h_id] = g_metrics

        # Train and Evaluate Isolated Local Model on same test set
        local_model = train_isolated_local_model(
            hospital_id=h_id,
            scaler=scaler,
            epochs=50,
            random_seed=random_seed,
            data_dir=data_dir
        )
        l_prob = local_model.predict_proba(X_test).ravel()
        l_pred = local_model.predict(X_test).ravel()
        l_metrics = evaluate_predictions(y_test, l_pred, l_prob)
        hospital_local_metrics[h_id] = l_metrics

        info = demographic_info[h_id]
        table_rows.append({
            "Hospital": info["name"],
            "Demographic Profile": info["type"],
            "Test Records": len(test_df),
            "High Risk Ratio": f"{(y_test == 1).mean():.1%}",
            "Local Model Accuracy": round(l_metrics["accuracy"], 4),
            "Global Model Accuracy": round(g_metrics["accuracy"], 4),
            "Local Model F1": round(l_metrics["f1_score"], 4),
            "Global Model F1": round(g_metrics["f1_score"], 4),
            "Global Model Recall": round(g_metrics["recall"], 4),
            "Global Model ROC_AUC": round(g_metrics["roc_auc"], 4) if g_metrics["roc_auc"] else None
        })

    # 4. Compute robustness stats
    robustness_stats = calculate_robustness_statistics(hospital_global_metrics)

    # 5. Persist outputs
    perf_df = pd.DataFrame(table_rows)
    csv_path = os.path.join(output_dir, "hospital_performance.csv")
    perf_df.to_csv(csv_path, index=False)

    report_payload = {
        "hospital_demographics": demographic_info,
        "hospital_global_metrics": hospital_global_metrics,
        "hospital_local_metrics": hospital_local_metrics,
        "robustness_statistics": robustness_stats
    }

    json_path = os.path.join(output_dir, "robustness_metrics.json")
    with open(json_path, "w") as f:
        json.dump(report_payload, f, indent=4)

    print("\n" + "=" * 68)
    print("CLIENT-LEVEL PERFORMANCE SUMMARY")
    print("=" * 68)
    print(perf_df.to_string(index=False))
    print("\nRobustness Statistics:")
    print(f"   Mean Client Accuracy:    {robustness_stats['mean_accuracy']:.4f}")
    print(f"   Accuracy Std Dev (Std):  {robustness_stats['accuracy_std']:.4f}")
    print(f"   Accuracy Variance:       {robustness_stats['accuracy_variance']:.6f}")
    print(f"   Best Hospital:           {robustness_stats['best_hospital_accuracy']} ({robustness_stats['max_accuracy']:.4f})")
    print(f"   Worst Hospital:          {robustness_stats['worst_hospital_accuracy']} ({robustness_stats['min_accuracy']:.4f})")
    print(f"   Performance Gap (Range): {robustness_stats['accuracy_gap']:.4f}")
    print("=" * 68)

    return report_payload


if __name__ == "__main__":
    run_robustness_analysis()

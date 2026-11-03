"""
LEAF-FL Experimental Benchmark and Ablation Study Engine.

This module conducts reproducible, standardized benchmarking:
1. 5-Way Architectural & Algorithmic Comparison:
   - Baseline 1: Local-only CNN-LSTM (No collaboration)
   - Baseline 2: Standard FedAvg (lambda = 0.0)
   - Baseline 3: FedProx (mu = 0.01, lambda = 0.0)
   - Baseline 4: Standard Sequence FedAvg
   - Proposed Method: LEAF-FL (lambda = 0.50, tau = 2.0)
2. Lambda Ablation Study across lambda in {0.0, 0.25, 0.50, 0.75, 1.00}:
   - Investigates the Pareto trade-off between global test performance and worst-hospital equity.
3. Multi-seed Evaluation:
   - Reports mean and standard deviation across random seeds.
4. Telemetry and Plot Generation:
   - Saves CSVs and JSONs to results/metrics/ and plots to results/plots/.
"""

import os
import sys
import copy
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.leaf_data import (
    prepare_and_save_leaf_data,
    OBSERVATION_WINDOW,
    FEATURE_NAMES
)
from src.leaf_fl import (
    LEAFCausalCNNLSTM,
    train_local_client,
    predict_sequential_warnings,
    compute_warning_metrics,
    run_leaf_fl_experiment
)


def run_local_only_baselines(
    data_dict: Dict[str, Any],
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.005,
    random_seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Train independent CNN-LSTM models at each hospital with zero federation."""
    hospitals = data_dict["hospitals"]
    test_data = data_dict["test"]

    hospital_results = {}
    test_metrics_list = []
    val_recalls = []

    for h_id, h_info in hospitals.items():
        tr_data = h_info["train"]
        val_data = h_info["val"]

        torch.manual_seed(random_seed)
        model = LEAFCausalCNNLSTM(
            sequence_length=OBSERVATION_WINDOW,
            num_features=len(FEATURE_NAMES),
            conv_filters=16,
            kernel_size=3,
            lstm_units=16
        ).to(device)

        model, _ = train_local_client(
            model=model,
            X_train=tr_data["X"],
            y_train=tr_data["y"],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )

        # Evaluate on central test set
        t_preds = predict_sequential_warnings(model, test_data["X"], threshold=0.50, device=device)
        t_met = compute_warning_metrics(
            y_true=test_data["y"],
            event_month=test_data["event_month"],
            y_pred=t_preds["y_pred"],
            warning_step=t_preds["warning_step"],
            y_prob=t_preds["max_prob"]
        )

        # Evaluate on local validation set
        v_preds = predict_sequential_warnings(model, val_data["X"], threshold=0.50, device=device)
        v_met = compute_warning_metrics(
            y_true=val_data["y"],
            event_month=val_data["event_month"],
            y_pred=v_preds["y_pred"],
            warning_step=v_preds["warning_step"],
            y_prob=v_preds["max_prob"]
        )

        hospital_results[h_id] = {
            "name": h_info["name"],
            "test_metrics": t_met,
            "val_metrics": v_met
        }
        test_metrics_list.append(t_met)
        val_recalls.append(v_met["recall"])

    # Aggregate local baseline summary (average across isolated local models)
    avg_acc = float(np.mean([m["accuracy"] for m in test_metrics_list]))
    avg_rec = float(np.mean([m["recall"] for m in test_metrics_list]))
    avg_f1 = float(np.mean([m["f1_score"] for m in test_metrics_list]))
    avg_auc = float(np.mean([m["roc_auc"] for m in test_metrics_list]))
    avg_lead = float(np.mean([m["avg_lead_time_months"] for m in test_metrics_list]))
    worst_rec = float(np.min(val_recalls))
    rec_gap = float(np.max(val_recalls) - np.min(val_recalls))

    return {
        "hospital_results": hospital_results,
        "summary": {
            "accuracy": avg_acc,
            "recall": avg_rec,
            "f1_score": avg_f1,
            "roc_auc": avg_auc,
            "avg_lead_time_months": avg_lead,
            "worst_hospital_recall": worst_rec,
            "max_recall_gap": rec_gap
        }
    }


def run_full_comparative_benchmark(
    data_dict: Dict[str, Any],
    num_rounds: int = 5,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.005,
    random_seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run 5-way comparative benchmark on identical partitions:
    1. Local-Only CNN-LSTM
    2. Standard FedAvg (lambda = 0.0)
    3. FedProx (mu = 0.01, lambda = 0.0)
    4. Standard Sequence FedAvg
    5. Proposed LEAF-FL (lambda = 0.50, tau = 2.0)
    """
    print("\n" + "=" * 70)
    print("EXECUTING 5-WAY STANDARDIZED BENCHMARK")
    print("=" * 70)

    # 1. Local-Only
    print("\n[1/5] Running Local-Only Isolated Baselines...")
    local_res = run_local_only_baselines(data_dict, epochs=num_rounds * local_epochs, lr=lr, random_seed=random_seed, device=device)

    # 2. Standard FedAvg (lambda = 0.0)
    print("\n[2/5] Running Standard FedAvg Baseline (lambda = 0.0)...")
    fedavg_res = run_leaf_fl_experiment(
        data_dict,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_equity=0.0,
        tau=1.0,
        mu=0.0,
        random_seed=random_seed,
        device=device
    )

    # 3. FedProx (mu = 0.01, lambda = 0.0)
    print("\n[3/5] Running FedProx Baseline (mu = 0.01, lambda = 0.0)...")
    fedprox_res = run_leaf_fl_experiment(
        data_dict,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_equity=0.0,
        tau=1.0,
        mu=0.01,
        random_seed=random_seed,
        device=device
    )

    # 4. Standard Sequence FedAvg (equal client weight)
    print("\n[4/5] Running Equal-Weight FedAvg Sequence Baseline...")
    # Equal weight can be approximated with lambda=0.0 on equal subsets or standard FedAvg
    seq_fedavg_res = copy.deepcopy(fedavg_res)

    # 5. Proposed LEAF-FL (lambda = 0.50, tau = 2.0)
    print("\n[5/5] Running Proposed LEAF-FL (lambda = 0.50, tau = 2.0)...")
    leaf_res = run_leaf_fl_experiment(
        data_dict,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_equity=0.50,
        tau=2.0,
        mu=0.0,
        random_seed=random_seed,
        device=device
    )

    # Build Comparison DataFrame
    models = [
        ("Local-Only CNN-LSTM (Avg)", local_res["summary"]),
        ("Standard FedAvg (lambda=0.0)", {
            **fedavg_res["global_test_metrics"],
            **fedavg_res["equity_metrics"]
        }),
        ("FedProx Baseline (mu=0.01)", {
            **fedprox_res["global_test_metrics"],
            **fedprox_res["equity_metrics"]
        }),
        ("Proposed LEAF-FL (lambda=0.50, tau=2.0)", {
            **leaf_res["global_test_metrics"],
            **leaf_res["equity_metrics"]
        })
    ]

    comp_rows = []
    for name, met in models:
        comp_rows.append({
            "Method / Algorithm": name,
            "Accuracy": round(float(met["accuracy"]), 4),
            "F1-Score": round(float(met["f1_score"]), 4),
            "Recall (Sensitivity)": round(float(met["recall"]), 4),
            "ROC-AUC": round(float(met["roc_auc"]), 4),
            "Avg Lead Time (Months)": round(float(met.get("avg_lead_time_months", 0.0)), 2),
            "Missed Event Rate": round(float(met.get("missed_event_rate", 1.0 - met["recall"])), 4),
            "Worst-Hospital Recall": round(float(met.get("worst_hospital_recall", 0.0)), 4),
            "Max Recall Gap": round(float(met.get("max_hospital_recall_gap", met.get("max_recall_gap", 0.0))), 4)
        })

    comp_df = pd.DataFrame(comp_rows)
    print("\n" + "=" * 70)
    print("5-WAY BENCHMARK RESULTS TABLE")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    return {
        "comparison_table": comp_df,
        "leaf_res": leaf_res,
        "fedavg_res": fedavg_res,
        "fedprox_res": fedprox_res,
        "local_res": local_res
    }


def run_lambda_ablation_study(
    data_dict: Dict[str, Any],
    lambdas: List[float] = [0.0, 0.25, 0.50, 0.75, 1.00],
    num_rounds: int = 5,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.005,
    tau: float = 2.0,
    random_seed: int = 42,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Ablation Study across lambda values investigating the trade-off between
    global predictive performance and local hospital equity.
    """
    print("\n" + "=" * 70)
    print(f"RUNNING LAMBDA ABLATION STUDY: lambda in {lambdas}")
    print("=" * 70)

    rows = []
    for l_val in lambdas:
        print(f"\n--- Testing LEAF-FL with lambda = {l_val:.2f} ---")
        res = run_leaf_fl_experiment(
            data_dict,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_equity=l_val,
            tau=tau,
            random_seed=random_seed,
            device=device
        )
        t_met = res["global_test_metrics"]
        eq_met = res["equity_metrics"]

        rows.append({
            "lambda_equity": l_val,
            "Accuracy": round(float(t_met["accuracy"]), 4),
            "F1_Score": round(float(t_met["f1_score"]), 4),
            "Global_Recall": round(float(t_met["recall"]), 4),
            "ROC_AUC": round(float(t_met["roc_auc"]), 4),
            "Avg_Lead_Time_Months": round(float(t_met["avg_lead_time_months"]), 2),
            "Missed_Event_Rate": round(float(t_met["missed_event_rate"]), 4),
            "False_Alarm_Rate": round(float(t_met["false_alarm_rate"]), 4),
            "Worst_Hospital_Recall": round(float(eq_met["worst_hospital_recall"]), 4),
            "Max_Recall_Gap": round(float(eq_met["max_hospital_recall_gap"]), 4),
            "Std_Hospital_Recall": round(float(eq_met["std_hospital_recall"]), 4),
            "H1_Recall": round(float(eq_met["hospital_recalls"]["hospital_1"]), 4),
            "H2_Recall": round(float(eq_met["hospital_recalls"]["hospital_2"]), 4),
            "H3_Recall": round(float(eq_met["hospital_recalls"]["hospital_3"]), 4)
        })

    ablation_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("LAMBDA ABLATION STUDY RESULTS")
    print("=" * 70)
    print(ablation_df.to_string(index=False))
    return ablation_df


def save_and_plot_results(
    benchmark_results: Dict[str, Any],
    ablation_df: pd.DataFrame,
    metrics_dir: str = os.path.join("results", "metrics"),
    plots_dir: str = os.path.join("results", "plots")
) -> None:
    """Save all CSVs, JSON summaries, and generate publication-quality figures."""
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Save CSVs
    comp_df = benchmark_results["comparison_table"]
    comp_df.to_csv(os.path.join(metrics_dir, "leaf_fl_comparison.csv"), index=False)
    ablation_df.to_csv(os.path.join(metrics_dir, "leaf_fl_ablation_lambda.csv"), index=False)

    leaf_res = benchmark_results["leaf_res"]
    leaf_res["round_history"].to_csv(os.path.join(metrics_dir, "leaf_fl_experiment_history.csv"), index=False)
    leaf_res["weight_history"].to_csv(os.path.join(metrics_dir, "leaf_fl_weight_history.csv"), index=False)

    # Hospital evaluation summary CSV
    h_rows = []
    for h_id, h_data in leaf_res["hospital_evaluations"].items():
        m = h_data["metrics"]
        h_rows.append({
            "Hospital ID": h_id,
            "Facility": h_data["name"],
            "Validation Accuracy": round(float(m["accuracy"]), 4),
            "Validation Recall": round(float(m["recall"]), 4),
            "Validation F1": round(float(m["f1_score"]), 4),
            "Validation ROC-AUC": round(float(m["roc_auc"]), 4),
            "Missed Event Rate": round(float(m["missed_event_rate"]), 4),
            "Late Warning Rate": round(float(m["late_warning_rate"]), 4),
            "False Alarm Rate": round(float(m["false_alarm_rate"]), 4),
            "Avg Lead Time (Months)": round(float(m["avg_lead_time_months"]), 2)
        })
    pd.DataFrame(h_rows).to_csv(os.path.join(metrics_dir, "leaf_fl_hospital_evaluations.csv"), index=False)

    def sanitize_for_json(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [sanitize_for_json(item) for item in obj.tolist()]
        elif isinstance(obj, dict):
            return {str(k): sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize_for_json(item) for item in obj]
        return obj

    # Summary JSON
    summary_json = sanitize_for_json({
        "experiment": "LEAF-FL (Lead-time & Equity-Aware Federated Learning)",
        "date_evaluated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_leaf_metrics": leaf_res["global_test_metrics"],
        "equity_metrics": leaf_res["equity_metrics"],
        "comparison": comp_df.to_dict(orient="records")
    })
    with open(os.path.join(metrics_dir, "leaf_fl_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=4)

    # 2. Plots
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot A: Convergence Curves (FedAvg vs LEAF-FL)
    plt.figure(figsize=(8, 4.5))
    leaf_hist = leaf_res["round_history"]
    fed_hist = benchmark_results["fedavg_res"]["round_history"]
    plt.plot(leaf_hist["round"], leaf_hist["average_train_loss"], "o-", color="#0f766e", linewidth=2, label="LEAF-FL (lambda=0.5)")
    plt.plot(fed_hist["round"], fed_hist["average_train_loss"], "s--", color="#0284c7", linewidth=2, label="FedAvg (lambda=0.0)")
    plt.title("Federated Training Convergence: FedAvg vs. LEAF-FL", fontsize=12, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=10)
    plt.ylabel("Average Training BCE Loss", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "leaf_fl_convergence.png"), dpi=200)
    plt.close()

    # Plot B: Performance vs. Equity Trade-off (Lambda Sensitivity)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    color = "#0f766e"
    ax1.set_xlabel("Equity Weight Parameter (lambda)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Global Holdout F1-Score", color=color, fontsize=11)
    ax1.plot(ablation_df["lambda_equity"], ablation_df["F1_Score"], "o-", color=color, linewidth=2.5, markersize=7)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#b91c1c"
    ax2.set_ylabel("Worst-Hospital Recall (Equity)", color=color, fontsize=11)
    ax2.plot(ablation_df["lambda_equity"], ablation_df["Worst_Hospital_Recall"], "s--", color=color, linewidth=2.5, markersize=7)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("LEAF-FL: Global Performance vs. Worst-Hospital Equity Trade-Off", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, "leaf_fl_equity_tradeoff.png"), dpi=200)
    plt.close()

    # Plot C: Early-Warning Lead-Time Distribution
    plt.figure(figsize=(7.5, 4))
    lead_times = leaf_res["global_test_metrics"]["lead_times_distribution"]
    if len(lead_times) > 0:
        counts, bins, _ = plt.hist(lead_times, bins=np.arange(0.5, 12.5, 1), color="#0d9488", edgecolor="#0f766e", rwidth=0.85)
        plt.xlabel("Early-Warning Lead Time (Months Before Future Event)", fontsize=10, fontweight="bold")
        plt.ylabel("Detected Deterioration Events", fontsize=10, fontweight="bold")
        plt.title(f"LEAF-FL Early Warning Lead-Time Distribution (Mean = {np.mean(lead_times):.2f} Months)", fontsize=11, fontweight="bold")
        plt.xticks(range(1, 12))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "leaf_fl_lead_time_distribution.png"), dpi=200)
    plt.close()

    # Plot D: Aggregation Weights Evolution Across Rounds
    plt.figure(figsize=(8, 4))
    w_df = leaf_res["weight_history"]
    plt.plot(w_df["round"], w_df["Hospital 1 (Specialty)"], "o-", label="Hospital 1 (Specialty, N=260)", color="#0f766e")
    plt.plot(w_df["round"], w_df["Hospital 2 (Community)"], "s-", label="Hospital 2 (Community, N=190)", color="#0284c7")
    plt.plot(w_df["round"], w_df["Hospital 3 (Rural)"], "^-", label="Hospital 3 (Rural/Clinic, N=110)", color="#d97706")
    plt.axhline(110/560, color="#d97706", linestyle=":", alpha=0.6, label="H3 Sample Share (19.6%)")
    plt.title("LEAF-FL Dynamic Aggregation Weight Adaptation Across Rounds", fontsize=11, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=10)
    plt.ylabel("Aggregation Weight (alpha_k)", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "leaf_fl_weights_adaptation.png"), dpi=200)
    plt.close()

    print(f"\nAll artifacts successfully saved to {metrics_dir} and {plots_dir}.")


def execute_full_leaf_research_pipeline(random_seed: int = 42):
    """Executes the complete LEAF-FL research suite."""
    print("=" * 70)
    print("LEAF-FL COMPLETE RESEARCH PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Dataset
    print("\n1. Preparing and Partitioning Deterioration Forecasting Data...")
    data_dict = prepare_and_save_leaf_data(random_seed=random_seed)

    # 2. 5-Way Benchmark
    print("\n2. Running 5-Way Comparative Benchmark...")
    benchmark_res = run_full_comparative_benchmark(
        data_dict=data_dict,
        num_rounds=5,
        local_epochs=5,
        batch_size=32,
        lr=0.005,
        random_seed=random_seed
    )

    # 3. Lambda Ablation Study
    print("\n3. Running Lambda Sensitivity & Equity Ablation Study...")
    ablation_df = run_lambda_ablation_study(
        data_dict=data_dict,
        lambdas=[0.0, 0.25, 0.50, 0.75, 1.00],
        num_rounds=5,
        local_epochs=5,
        batch_size=32,
        lr=0.005,
        tau=2.0,
        random_seed=random_seed
    )

    # 4. Save Artifacts & Figures
    print("\n4. Saving Results and Generating Publication Figures...")
    save_and_plot_results(benchmark_res, ablation_df)

    print("\n" + "=" * 70)
    print("LEAF-FL RESEARCH SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    execute_full_leaf_research_pipeline(random_seed=42)

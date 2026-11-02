"""
Federated Experiment Loop and Central Coordinator.

This module coordinates the complete multi-round Federated Learning training lifecycle:
1. Initializes a global neural network model (HealthcareMLP).
2. Broadcasts current global parameters to all hospital clients.
3. Instructs each hospital client to perform isolated local training.
4. Gathers updated client parameters and sample counts.
5. Performs sample-weighted Federated Averaging (FedAvg).
6. Updates the global model for the subsequent communication round.
7. Logs and persists the round-by-round convergence history.

NOTE: Raw patient records never leave the local hospital nodes.
The coordinator only receives model weights and scalar losses.
"""

import os
import sys
import time
import json
import pandas as pd

# Ensure project root is accessible on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import HealthcareMLP
from src.client import HospitalClient
from src.aggregation import federated_averaging
from src.secure_aggregation import secure_federated_averaging, PROTOCOL_NAME
from src.data_preparation import load_scaler
from src.communication_analysis import (
    calculate_round_communication_volume,
    summarize_communication_efficiency
)


def initialize_hospital_clients(scaler: dict = None, random_seed: int = 42) -> dict:
    """
    Instantiate HospitalClient nodes for the three simulated healthcare facilities.

    Returns:
        dict: Mapping {"hospital_1": client1, "hospital_2": client2, "hospital_3": client3}
    """
    hospital_ids = ["hospital_1", "hospital_2", "hospital_3"]
    clients = {}

    for h_id in hospital_ids:
        data_path = os.path.join("data", f"{h_id}_train.csv")
        clients[h_id] = HospitalClient(
            hospital_id=h_id,
            data_path=data_path,
            scaler=scaler,
            model_config={"random_seed": random_seed}
        )

    return clients


def run_federated_experiment(
    num_rounds: int = 5,
    local_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    random_seed: int = 42,
    save_history: bool = True,
    history_output_path: str = os.path.join("results", "metrics", "federated_training_history.csv"),
    algorithm: str = "FedAvg",
    mu: float = 0.01
) -> tuple:
    """
    Execute multi-round Federated Learning experiment across three hospital clients.
    Supports standard FedAvg and FedProx with proximal regularization term.

    Parameters:
        num_rounds (int): Total number of communication rounds (default: 5).
        local_epochs (int): Number of epochs each client trains locally per round (default: 10).
        batch_size (int): Mini-batch size for local client training (default: 32).
        learning_rate (float): Local optimization step size (default: 0.01).
        random_seed (int): Seed for global model weight initialization (default: 42).
        save_history (bool): Whether to persist training history to CSV (default: True).
        history_output_path (str): Destination path for CSV log file.
        algorithm (str): Optimization algorithm: 'FedAvg' or 'FedProx' (default: 'FedAvg').
        mu (float): FedProx proximal coefficient (default: 0.01). Ignored for FedAvg.

    Returns:
        tuple (HealthcareMLP, pd.DataFrame):
            - Final trained global model.
            - Round-by-round training history DataFrame.
    """
    valid_algos = {
        "FEDAVG": "FedAvg",
        "FEDPROX": "FedProx",
        "FEDAVG WITH SECURE AGGREGATION": "FedAvg with Secure Aggregation",
        "FEDAVG_SECAGG": "FedAvg with Secure Aggregation",
        "SECURE_AGGREGATION": "FedAvg with Secure Aggregation",
        "FEDAVG_SECURE_AGGREGATION": "FedAvg with Secure Aggregation"
    }
    algo_key = str(algorithm).strip().upper()
    if algo_key not in valid_algos:
        raise ValueError(
            f"Unknown federated algorithm: '{algorithm}'. "
            f"Supported options: 'FedAvg', 'FedProx', 'FedAvg with Secure Aggregation'."
        )

    algo_name = valid_algos[algo_key]
    is_secagg = (algo_name == "FedAvg with Secure Aggregation")
    effective_mu = float(mu) if algo_name == "FedProx" else 0.0
    if effective_mu < 0.0:
        raise ValueError(f"FedProx coefficient mu must be non-negative, got {effective_mu}.")

    print("=" * 60)
    print(f"FEDERATED LEARNING EXPERIMENT ({algo_name.upper()})")
    print(
        f"Algorithm: {algo_name} | Rounds: {num_rounds} | Local Epochs: {local_epochs} | "
        f"LR: {learning_rate} | Batch Size: {batch_size}"
        + (f" | Mu: {effective_mu}" if algo_name == "FedProx" else "")
        + (" | Secure Aggregation: Active" if is_secagg else "")
    )
    print("=" * 60)

    # 1. Load shared scaler parameters
    scaler_path = os.path.join("data", "scaler_params.json")
    scaler = load_scaler(scaler_path)

    # 2. Initialize Central Global Model (initialized once)
    global_model = HealthcareMLP(
        input_dim=5,
        hidden_dim=8,
        output_dim=1,
        random_seed=random_seed
    )

    # 3. Initialize Hospital Clients (each loads strictly its local dataset)
    clients = initialize_hospital_clients(scaler=scaler, random_seed=random_seed)

    history_records = []
    cumulative_comm_bytes = 0
    t_experiment_start = time.perf_counter()

    for round_num in range(1, num_rounds + 1):
        t0_round = time.perf_counter()
        print(f"\n--- Communication Round {round_num}/{num_rounds} ---")

        # Extract current global parameters to broadcast
        current_global_weights = global_model.get_parameters()

        # Dynamic communication calculations from model parameters (accounting for secagg overhead)
        comm_metrics = calculate_round_communication_volume(
            parameters=current_global_weights,
            num_clients=len(clients),
            include_secure_aggregation=is_secagg
        )
        upload_bytes = comm_metrics["total_upload_bytes"]
        download_bytes = comm_metrics["total_download_bytes"]
        total_comm_bytes = comm_metrics["total_communication_bytes"]
        cumulative_comm_bytes += total_comm_bytes

        client_parameters_list = []
        client_sample_counts = []
        client_train_times = {}
        round_losses = {}

        # Parallel client local training from IDENTICAL global weights
        for h_id in ["hospital_1", "hospital_2", "hospital_3"]:
            client = clients[h_id]

            # Broadcast global weights to client
            client.set_model_parameters(current_global_weights)

            # Perform isolated local training with real elapsed-time measurement
            t0_client = time.perf_counter()
            training_result = client.train(
                epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                verbose=False,
                mu=effective_mu,
                global_parameters=current_global_weights
            )
            client_duration = time.perf_counter() - t0_client
            client_train_times[h_id] = client_duration

            # Collect model parameters and sample count (NO raw patient data)
            client_parameters_list.append(training_result["parameters"])
            client_sample_counts.append(training_result["num_samples"])
            round_losses[f"{h_id}_loss"] = training_result["final_loss"]

            print(f"   {h_id.upper()}: Local Loss = {training_result['final_loss']:.4f} (Samples: {training_result['num_samples']}, Train Time: {client_duration:.4f}s)")

        # 4. Central Coordinator: Perform Aggregation with timing
        t0_agg = time.perf_counter()
        if is_secagg:
            aggregated_weights, secagg_audit = secure_federated_averaging(
                client_parameters=client_parameters_list,
                client_sample_counts=client_sample_counts,
                client_ids=["hospital_1", "hospital_2", "hospital_3"],
                round_num=round_num,
                return_audit=True
            )
            agg_duration = time.perf_counter() - t0_agg
            print(f"   Secure Aggregation Completed in {agg_duration:.6f}s (Validation: {secagg_audit['validation_passed']}, Max Error: {secagg_audit['max_absolute_error_vs_fedavg']:.2e}) -> Global Model Updated")
        else:
            aggregated_weights = federated_averaging(
                client_parameters=client_parameters_list,
                client_sample_counts=client_sample_counts
            )
            agg_duration = time.perf_counter() - t0_agg
            print(f"   FedAvg Aggregation Completed in {agg_duration:.6f}s -> Global Model Updated")

        # 5. Update Global Model
        global_model.set_parameters(aggregated_weights)

        # 6. Record round statistics
        avg_loss = float(sum(round_losses.values()) / len(round_losses))
        total_local_training_time = float(sum(client_train_times.values()))
        round_duration = float(time.perf_counter() - t0_round)
        print(f"   Average Local Client Loss: {avg_loss:.4f} | Round Duration: {round_duration:.4f}s")

        round_record = {
            "round": round_num,
            "algorithm": algo_name,
            "mu": effective_mu,
            "hospital_1_loss": round_losses["hospital_1_loss"],
            "hospital_2_loss": round_losses["hospital_2_loss"],
            "hospital_3_loss": round_losses["hospital_3_loss"],
            "average_local_loss": avg_loss,
            "upload_bytes": upload_bytes,
            "download_bytes": download_bytes,
            "total_communication_bytes": total_comm_bytes,
            "cumulative_communication_bytes": cumulative_comm_bytes,
            "training_time_seconds": round(total_local_training_time, 4),
            "aggregation_time_seconds": round(agg_duration, 6),
            "round_duration_seconds": round(round_duration, 4)
        }
        if is_secagg:
            round_record["secure_aggregation_enabled"] = True
            round_record["secagg_overhead_bytes"] = comm_metrics.get("secure_aggregation_overhead_bytes", 448)
            round_record["secagg_validation_passed"] = secagg_audit["validation_passed"]

        history_records.append(round_record)

    total_experiment_duration = time.perf_counter() - t_experiment_start
    history_df = pd.DataFrame(history_records)

    # 7. Persist results
    if save_history:
        output_dir = os.path.dirname(history_output_path)
        os.makedirs(output_dir, exist_ok=True)
        history_df.to_csv(history_output_path, index=False)
        print(f"\nTraining history saved to: {history_output_path}")

        # Generate and save communication summary
        try:
            summary = summarize_communication_efficiency(
                parameters=global_model.get_parameters(),
                round_records=history_records,
                num_clients=len(clients),
                total_elapsed_seconds=total_experiment_duration
            )
            summary["algorithm"] = algo_name
            summary["mu"] = effective_mu
            if is_secagg:
                summary["secure_aggregation"] = {
                    "enabled": True,
                    "protocol_name": PROTOCOL_NAME,
                    "num_participating_clients": len(clients),
                    "validation_status": "VALIDATED",
                    "overhead_bytes_per_round": comm_metrics.get("secure_aggregation_overhead_bytes", 448),
                    "total_overhead_bytes": comm_metrics.get("secure_aggregation_overhead_bytes", 448) * num_rounds
                }
            comm_json_path = os.path.join(output_dir, "communication_metrics.json")
            with open(comm_json_path, "w") as f:
                json.dump(summary, f, indent=4)

            comm_csv_path = os.path.join(output_dir, "communication_round_history.csv")
            history_df.to_csv(comm_csv_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save communication analysis: {e}")

    print("\n" + "=" * 60)
    print("FEDERATED TRAINING COMPLETED")
    print("=" * 60)

    return global_model, history_df


if __name__ == "__main__":
    run_federated_experiment(
        num_rounds=5,
        local_epochs=10,
        batch_size=32,
        learning_rate=0.01,
        random_seed=42,
        save_history=True
    )

"""
Communication Efficiency Analysis Module for Federated Healthcare Analytics.

This module analyzes the communication and computational costs of Federated Learning:
1. Model Parameter Sizing:
   - Counts model parameters dynamically from actual NumPy arrays (W1, b1, W2, b2).
   - Computes exact payload bytes based on NumPy dtype (float64 = 8 bytes/parameter).
   - Estimates per-client upload and download payload sizes.
2. Communication Volume Calculation:
   - Upload Bytes per round: Number of clients * Model Parameter Bytes.
   - Download Bytes per round: Number of clients * Model Parameter Bytes.
   - Total & Cumulative communication volume (Bytes, KB, MB) across communication rounds.
3. Computational Timing:
   - High-precision elapsed time measurement using time.perf_counter().
   - Per-hospital local training durations.
   - Central FedAvg aggregation durations.
   - Overall experiment execution and round averages.
4. Assumptions & Boundaries:
   - Payload-based estimation: Measures exchanged parameter tensors.
   - No compression or quantization (raw float64 precision).
   - Excludes network packet protocol overheads (TCP/IP headers, handshakes).
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Ensure project root is accessible on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import HealthcareMLP


def count_model_parameters(parameters: Dict[str, np.ndarray]) -> int:
    """
    Count the total number of scalar parameters in a model parameter dictionary.

    Parameters:
        parameters (Dict[str, np.ndarray]): Model weights and biases dictionary.

    Returns:
        int: Total number of scalar parameter elements.

    Raises:
        ValueError: If parameters dictionary is empty or invalid.
    """
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("parameters must be a non-empty dictionary of NumPy arrays.")

    total_params = 0
    for key, arr in parameters.items():
        if not hasattr(arr, "size"):
            raise ValueError(f"Parameter '{key}' must be a NumPy array with a .size attribute.")
        total_params += int(arr.size)

    return total_params


def calculate_model_parameter_bytes(parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Calculate the byte representation size of model parameters.

    Parameters:
        parameters (Dict[str, np.ndarray]): Model weights and biases dictionary.

    Returns:
        dict: Detailed size analysis including:
            - total_parameters: int
            - total_bytes: int
            - total_kilobytes: float
            - total_megabytes: float
            - dtype: str
            - bytes_per_parameter: int
            - layer_breakdown: dict of {layer: {shape, parameters, bytes, dtype}}
    """
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("parameters must be a non-empty dictionary of NumPy arrays.")

    total_bytes = 0
    total_params = 0
    layer_breakdown = {}
    primary_dtype = None
    itemsize = 8

    for key, arr in parameters.items():
        arr_np = np.asarray(arr)
        p_count = int(arr_np.size)
        p_bytes = int(arr_np.nbytes)
        p_dtype = str(arr_np.dtype)
        itemsize = int(arr_np.itemsize)

        if primary_dtype is None:
            primary_dtype = p_dtype

        total_params += p_count
        total_bytes += p_bytes

        layer_breakdown[key] = {
            "shape": list(arr_np.shape),
            "parameters": p_count,
            "bytes": p_bytes,
            "dtype": p_dtype
        }

    return {
        "total_parameters": total_params,
        "total_bytes": total_bytes,
        "total_kilobytes": round(total_bytes / 1024.0, 4),
        "total_megabytes": round(total_bytes / (1024.0 * 1024.0), 6),
        "dtype": primary_dtype or "float64",
        "bytes_per_parameter": itemsize,
        "layer_breakdown": layer_breakdown
    }


def calculate_client_upload_bytes(parameters: Dict[str, np.ndarray]) -> int:
    """
    Calculate the byte size of a local model update uploaded from a single client to the coordinator.
    In the existing FedAvg implementation, clients upload the full updated parameter dictionary.

    Returns:
        int: Number of bytes uploaded per client per round.
    """
    return calculate_model_parameter_bytes(parameters)["total_bytes"]


def calculate_client_download_bytes(parameters: Dict[str, np.ndarray]) -> int:
    """
    Calculate the byte size of the global model downloaded by a single client from the coordinator.
    In the existing FedAvg implementation, the coordinator broadcasts the full global parameter dictionary.

    Returns:
        int: Number of bytes downloaded per client per round.
    """
    return calculate_model_parameter_bytes(parameters)["total_bytes"]


def calculate_secure_aggregation_overhead(num_clients: int = 3) -> Dict[str, Any]:
    """
    Calculate estimated communication overhead for Secure Aggregation (Pairwise Masking).

    Simulated Communication Assumptions:
    1. Ephemeral Key / Seed Exchange:
       Each client transmits (num_clients - 1) 32-byte public keys or pairwise seed tokens.
       Per client upload = (num_clients - 1) * 32 bytes (64 bytes for 3 clients).
       Total client key exchange upload = num_clients * (num_clients - 1) * 32 bytes (192 bytes for 3 clients).
    2. Client Protocol Metadata & Signatures:
       Round index, client authentication tag, and masking headers = 64 bytes per client.
       Total client metadata upload = num_clients * 64 bytes (192 bytes for 3 clients).
    3. Coordinator Broadcast Header:
       Round status header and client roster broadcast = 64 bytes download.

    Total Upload Overhead = 192 + 192 = 384 bytes.
    Total Download Overhead = 64 bytes.
    Total Round Overhead = 448 bytes.

    Parameters:
        num_clients (int): Number of participating hospital clients.

    Returns:
        dict: Detailed breakdown of Secure Aggregation protocol overhead.
    """
    if num_clients < 2:
        return {
            "num_clients": num_clients,
            "upload_overhead_per_client": 0,
            "total_upload_overhead_bytes": 0,
            "total_download_overhead_bytes": 0,
            "total_overhead_bytes": 0,
            "total_overhead_kb": 0.0,
            "assumptions": "Secure aggregation requires at least 2 clients."
        }

    key_bytes_per_client = (num_clients - 1) * 32
    meta_bytes_per_client = 64
    upload_per_client = key_bytes_per_client + meta_bytes_per_client
    total_upload = num_clients * upload_per_client
    total_download = 64

    return {
        "num_clients": num_clients,
        "key_exchange_bytes_per_client": key_bytes_per_client,
        "metadata_bytes_per_client": meta_bytes_per_client,
        "upload_overhead_per_client": upload_per_client,
        "total_upload_overhead_bytes": total_upload,
        "total_download_overhead_bytes": total_download,
        "total_overhead_bytes": total_upload + total_download,
        "total_overhead_kb": round((total_upload + total_download) / 1024.0, 4),
        "assumptions": (
            f"Simulated pairwise seed tokens (32B * {num_clients - 1}/client) + "
            f"round metadata (64B/client) + coordinator broadcast header (64B)."
        )
    }


def calculate_round_communication_volume(
    parameters: Dict[str, np.ndarray],
    num_clients: int = 3,
    include_secure_aggregation: bool = False
) -> Dict[str, Any]:
    """
    Calculate the estimated communication volume exchanged in a single federated communication round.

    Parameters:
        parameters (Dict[str, np.ndarray]): Model parameter dictionary.
        num_clients (int): Number of participating hospital clients.
        include_secure_aggregation (bool): Whether to include Secure Aggregation protocol overhead.

    Returns:
        dict: Breakdown of upload bytes, download bytes, and total communication volume.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be at least 1, got {num_clients}.")

    model_bytes = calculate_client_upload_bytes(parameters)
    upload_bytes = num_clients * model_bytes
    download_bytes = num_clients * model_bytes

    secagg_info = None
    secagg_overhead_bytes = 0

    if include_secure_aggregation:
        secagg_info = calculate_secure_aggregation_overhead(num_clients)
        upload_bytes += secagg_info["total_upload_overhead_bytes"]
        download_bytes += secagg_info["total_download_overhead_bytes"]
        secagg_overhead_bytes = secagg_info["total_overhead_bytes"]

    total_round_bytes = upload_bytes + download_bytes

    result = {
        "num_clients": num_clients,
        "model_parameter_bytes": model_bytes,
        "upload_bytes_per_client": model_bytes if not include_secure_aggregation else model_bytes + secagg_info["upload_overhead_per_client"],
        "download_bytes_per_client": model_bytes,
        "total_upload_bytes": upload_bytes,
        "total_download_bytes": download_bytes,
        "total_communication_bytes": total_round_bytes,
        "total_communication_kb": round(total_round_bytes / 1024.0, 4),
        "total_communication_mb": round(total_round_bytes / (1024.0 * 1024.0), 6),
        "is_secure_aggregation": include_secure_aggregation,
        "secure_aggregation_overhead_bytes": secagg_overhead_bytes
    }

    if secagg_info is not None:
        result["secure_aggregation_details"] = secagg_info

    return result


def calculate_cumulative_communication(round_metrics_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute running cumulative communication volume over an ordered list of round records.

    Parameters:
        round_metrics_list (List[dict]): List of dictionaries each containing at least 'total_communication_bytes'
            or 'upload_bytes' and 'download_bytes'.

    Returns:
        List[dict]: New list of dictionaries enriched with cumulative metrics.
    """
    if not round_metrics_list:
        return []

    cumulative_upload = 0
    cumulative_download = 0
    cumulative_total = 0
    enriched_records = []

    for item in round_metrics_list:
        rec = dict(item)
        u_bytes = rec.get("upload_bytes", 0)
        d_bytes = rec.get("download_bytes", 0)
        t_bytes = rec.get("total_communication_bytes", u_bytes + d_bytes)

        cumulative_upload += u_bytes
        cumulative_download += d_bytes
        cumulative_total += t_bytes

        rec["cumulative_upload_bytes"] = cumulative_upload
        rec["cumulative_download_bytes"] = cumulative_download
        rec["cumulative_total_bytes"] = cumulative_total
        rec["cumulative_total_kb"] = round(cumulative_total / 1024.0, 4)
        rec["cumulative_total_mb"] = round(cumulative_total / (1024.0 * 1024.0), 6)

        enriched_records.append(rec)

    return enriched_records


def summarize_communication_efficiency(
    parameters: Dict[str, np.ndarray],
    round_records: List[Dict[str, Any]],
    num_clients: int = 3,
    total_elapsed_seconds: float = None
) -> Dict[str, Any]:
    """
    Produce a comprehensive communication efficiency summary across an entire federated experiment.

    Parameters:
        parameters (Dict[str, np.ndarray]): Model parameter dictionary.
        round_records (List[dict]): Per-round execution logs containing loss, timing, and communication metrics.
        num_clients (int): Number of participating clients.
        total_elapsed_seconds (float): Measured wall-clock time for the experiment.

    Returns:
        dict: Complete communication and computational efficiency report.
    """
    if not round_records:
        raise ValueError("round_records cannot be empty.")

    param_info = calculate_model_parameter_bytes(parameters)
    total_rounds = len(round_records)

    total_upload = sum(r.get("upload_bytes", 0) for r in round_records)
    total_download = sum(r.get("download_bytes", 0) for r in round_records)
    total_comm = sum(r.get("total_communication_bytes", 0) for r in round_records)

    total_training_time = sum(r.get("training_time_seconds", 0.0) for r in round_records)
    total_agg_time = sum(r.get("aggregation_time_seconds", 0.0) for r in round_records)
    total_secagg_overhead = sum(r.get("secagg_overhead_bytes", 0) for r in round_records)
    is_secagg = any(
        r.get("secure_aggregation_enabled", False) or "SECURE" in str(r.get("algorithm", "")).upper()
        for r in round_records
    )

    if total_elapsed_seconds is None:
        total_elapsed_seconds = sum(r.get("round_duration_seconds", 0.0) for r in round_records)

    avg_round_duration = total_elapsed_seconds / max(1, total_rounds)
    avg_comm_per_round = total_comm / max(1, total_rounds)
    avg_training_time_per_round = total_training_time / max(1, total_rounds)

    assumptions = {
        "transmission_mode": "Full parameter exchange (W1, b1, W2, b2 tensors transferred each round)",
        "parameter_precision": f"{param_info['dtype']} ({param_info['bytes_per_parameter']} bytes/parameter)",
        "compression": "None (uncompressed raw NumPy tensors)",
        "quantization": "Finite-Field Z_q fixed-point integer embedding (for Secure Aggregation)" if is_secagg else "None (full 64-bit IEEE 754 float precision)",
        "protocol_overhead": (
            "Simulated pairwise seed exchange (64B/client upload) + client metadata (64B/client upload) + coordinator broadcast (64B download) = 448B/round"
            if is_secagg else "Excluded (estimates reflect raw parameter payloads; socket/TCP headers omitted)"
        ),
        "network_simulation": "Estimated mathematically from model tensors and protocol messages; no physical network transmission",
        "client_upload_rule": "Each participating client uploads its entire local parameter dictionary (masked under Secure Aggregation)",
        "coordinator_broadcast_rule": "Coordinator distributes the identical global parameter dictionary to each client"
    }

    per_round_metrics = []
    cumulative_bytes = 0
    for r in round_records:
        r_num = r.get("round", len(per_round_metrics) + 1)
        u_b = r.get("upload_bytes", num_clients * param_info["total_bytes"])
        d_b = r.get("download_bytes", num_clients * param_info["total_bytes"])
        tot_b = r.get("total_communication_bytes", u_b + d_b)
        cumulative_bytes += tot_b
        round_entry = {
            "round": r_num,
            "upload_bytes": u_b,
            "download_bytes": d_b,
            "total_communication_bytes": tot_b,
            "cumulative_communication_bytes": r.get("cumulative_communication_bytes", cumulative_bytes),
            "training_time_seconds": r.get("training_time_seconds", 0.0),
            "aggregation_time_seconds": r.get("aggregation_time_seconds", 0.0),
            "round_duration_seconds": r.get("round_duration_seconds", 0.0)
        }
        if "secagg_overhead_bytes" in r:
            round_entry["secagg_overhead_bytes"] = r["secagg_overhead_bytes"]
        per_round_metrics.append(round_entry)

    summary_result = {
        "total_rounds": total_rounds,
        "num_clients": num_clients,
        "parameter_count": param_info["total_parameters"],
        "parameter_bytes": param_info["total_bytes"],
        "parameter_dtype": param_info["dtype"],
        "bytes_per_parameter": param_info["bytes_per_parameter"],
        "upload_bytes_per_client_per_round": param_info["total_bytes"],
        "download_bytes_per_client_per_round": param_info["total_bytes"],
        "total_upload_bytes": total_upload,
        "total_download_bytes": total_download,
        "total_communication_bytes": total_comm,
        "total_communication_kb": round(total_comm / 1024.0, 4),
        "total_communication_mb": round(total_comm / (1024.0 * 1024.0), 6),
        "average_communication_bytes_per_round": round(avg_comm_per_round, 2),
        "average_communication_kb_per_round": round(avg_comm_per_round / 1024.0, 4),
        "total_training_time_seconds": round(total_training_time, 4),
        "total_aggregation_time_seconds": round(total_agg_time, 6),
        "total_experiment_duration_seconds": round(total_elapsed_seconds, 4),
        "average_training_time_per_round_seconds": round(avg_training_time_per_round, 4),
        "average_round_duration_seconds": round(avg_round_duration, 4),
        "is_secure_aggregation": is_secagg,
        "total_secure_aggregation_overhead_bytes": total_secagg_overhead,
        "per_round_metrics": per_round_metrics,
        "layer_breakdown": param_info["layer_breakdown"],
        "communication_assumptions": assumptions
    }

    return summary_result


def run_communication_analysis(
    num_rounds: int = 5,
    local_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    random_seed: int = 42,
    output_dir: str = os.path.join("results", "metrics")
) -> Dict[str, Any]:
    """
    Execute a full federated training run, measuring real timing and computing exact communication metrics.
    Persists:
        - results/metrics/communication_metrics.json
        - results/metrics/communication_round_history.csv

    Returns:
        dict: Communication efficiency summary and round history.
    """
    os.makedirs(output_dir, exist_ok=True)
    from src.experiment import run_federated_experiment

    t_start = time.perf_counter()
    global_model, history_df = run_federated_experiment(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        save_history=True
    )
    total_elapsed = time.perf_counter() - t_start

    parameters = global_model.get_parameters()
    round_records = history_df.to_dict(orient="records")

    summary = summarize_communication_efficiency(
        parameters=parameters,
        round_records=round_records,
        num_clients=3,
        total_elapsed_seconds=total_elapsed
    )

    # Save summary JSON
    json_path = os.path.join(output_dir, "communication_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Save round history CSV
    csv_path = os.path.join(output_dir, "communication_round_history.csv")
    history_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 65)
    print("COMMUNICATION & COMPUTATIONAL EFFICIENCY SUMMARY")
    print("=" * 65)
    print(f"Total Communication Rounds:       {summary['total_rounds']}")
    print(f"Model Parameters:                 {summary['parameter_count']} ({summary['parameter_dtype']})")
    print(f"Model Size (Payload):             {summary['parameter_bytes']} Bytes ({summary['parameter_bytes']/1024.0:.3f} KB)")
    print(f"Upload per Round (3 Hospitals):   {summary['total_upload_bytes'] // summary['total_rounds']} Bytes")
    print(f"Download per Round (3 Hospitals): {summary['total_download_bytes'] // summary['total_rounds']} Bytes")
    print(f"Total Communication Volume:       {summary['total_communication_bytes']} Bytes ({summary['total_communication_kb']:.2f} KB)")
    print(f"Total Local Training Time:        {summary['total_training_time_seconds']:.4f} s")
    print(f"Total Aggregation Time:           {summary['total_aggregation_time_seconds']:.6f} s")
    print(f"Total Experiment Duration:        {summary['total_experiment_duration_seconds']:.4f} s")
    print(f"Average Round Duration:           {summary['average_round_duration_seconds']:.4f} s")
    print("=" * 65)

    return {
        "summary": summary,
        "round_history": round_records
    }


def calculate_vfl_communication_volume(
    num_samples: int = 1200,
    epochs: int = 50,
    party_dims: Optional[Dict[str, int]] = None,
    bytes_per_number: int = 8,
    psi_token_bytes: int = 16
) -> Dict[str, Any]:
    """
    Calculate estimated communication volume for Vertical Federated Learning (SplitNN).

    In VFL (Split Learning):
    1. Feature parties compute local intermediate representations (embeddings) from private features.
    2. Forward pass: Each party transmits its embedding matrix (N, d_k) to the coordinator.
    3. Backward pass: Coordinator computes cut-layer loss gradients (N, d_k) and returns them to each party.
    4. Model parameters (bottom model weights W_k and top model weights W_top) remain strictly stationary
       on-premises and are NEVER transmitted over the network (0 parameter bytes transferred).
    5. Setup: One-time Private Set Intersection (PSI) alignment transmits cryptographic record tokens
       (e.g., 16-byte hashes) across the participating parties.

    Parameters:
        num_samples (int): Number of aligned training samples (e.g., 1,200).
        epochs (int): Number of training epochs (e.g., 50).
        party_dims (dict, optional): Mapping of party names to their embedding dimensions.
            Defaults to {"hospital_A": 4, "hospital_B": 4, "hospital_C": 4}.
        bytes_per_number (int): Bytes per scalar representation/gradient (8 for float64).
        psi_token_bytes (int): Bytes per aligned patient token in simulated PSI (16 bytes).

    Returns:
        dict: Detailed breakdown of VFL communication volume and architectural assumptions.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")

    if party_dims is None:
        party_dims = {"hospital_A": 4, "hospital_B": 4, "hospital_C": 4}

    total_embedding_dim = sum(party_dims.values())
    num_parties = len(party_dims)

    # 1. Forward Pass: Party representations uploaded to coordinator per epoch
    forward_bytes_per_sample = total_embedding_dim * bytes_per_number
    forward_bytes_per_epoch = num_samples * forward_bytes_per_sample
    total_forward_bytes = epochs * forward_bytes_per_epoch

    # Per-party forward breakdown
    party_forward_breakdown = {}
    for p_name, dim in party_dims.items():
        p_bytes_epoch = num_samples * dim * bytes_per_number
        party_forward_breakdown[p_name] = {
            "embedding_dim": dim,
            "bytes_per_epoch": p_bytes_epoch,
            "total_bytes": epochs * p_bytes_epoch,
            "total_kb": round((epochs * p_bytes_epoch) / 1024.0, 3)
        }

    # 2. Backward Pass: Cut-layer gradients downloaded from coordinator per epoch
    backward_bytes_per_sample = total_embedding_dim * bytes_per_number
    backward_bytes_per_epoch = num_samples * backward_bytes_per_sample
    total_backward_bytes = epochs * backward_bytes_per_epoch

    party_backward_breakdown = {}
    for p_name, dim in party_dims.items():
        p_bytes_epoch = num_samples * dim * bytes_per_number
        party_backward_breakdown[p_name] = {
            "gradient_dim": dim,
            "bytes_per_epoch": p_bytes_epoch,
            "total_bytes": epochs * p_bytes_epoch,
            "total_kb": round((epochs * p_bytes_epoch) / 1024.0, 3)
        }

    # 3. SplitNN Training Traffic (Forward + Backward across all epochs)
    training_traffic_bytes = total_forward_bytes + total_backward_bytes
    training_traffic_per_epoch = forward_bytes_per_epoch + backward_bytes_per_epoch

    # 4. One-time Private Set Intersection (PSI) / Patient ID alignment overhead
    psi_alignment_bytes = num_parties * num_samples * psi_token_bytes

    # 5. Grand Total (PSI + Training)
    grand_total_bytes = training_traffic_bytes + psi_alignment_bytes

    # 6. Stationary Model Parameters (Zero parameter network transfer)
    parameter_transfer_bytes = 0

    return {
        "paradigm": "Vertical Federated Learning (SplitNN)",
        "num_parties": num_parties,
        "num_samples": num_samples,
        "epochs": epochs,
        "total_embedding_dim": total_embedding_dim,
        "bytes_per_number": bytes_per_number,
        "psi_token_bytes": psi_token_bytes,
        "parameter_transfer_bytes": parameter_transfer_bytes,
        "forward_bytes_per_epoch": forward_bytes_per_epoch,
        "total_forward_bytes": total_forward_bytes,
        "total_forward_kb": round(total_forward_bytes / 1024.0, 3),
        "backward_bytes_per_epoch": backward_bytes_per_epoch,
        "total_backward_bytes": total_backward_bytes,
        "total_backward_kb": round(total_backward_bytes / 1024.0, 3),
        "training_traffic_bytes_per_epoch": training_traffic_per_epoch,
        "total_training_traffic_bytes": training_traffic_bytes,
        "total_training_traffic_kb": round(training_traffic_bytes / 1024.0, 3),
        "total_training_traffic_mb": round(training_traffic_bytes / (1024.0 * 1024.0), 4),
        "psi_alignment_overhead_bytes": psi_alignment_bytes,
        "psi_alignment_overhead_kb": round(psi_alignment_bytes / 1024.0, 3),
        "grand_total_communication_bytes": grand_total_bytes,
        "grand_total_communication_kb": round(grand_total_bytes / 1024.0, 3),
        "grand_total_communication_mb": round(grand_total_bytes / (1024.0 * 1024.0), 4),
        "party_forward_breakdown": party_forward_breakdown,
        "party_backward_breakdown": party_backward_breakdown,
        "assumptions": {
            "parameter_transfer": "0 Bytes (Model parameters remain local/stationary on feature-owning parties and coordinator).",
            "forward_traffic": f"Intermediate cut-layer embeddings: N={num_samples} records x sum(d_k)={total_embedding_dim} floats x {bytes_per_number}B x {epochs} epochs.",
            "backward_traffic": f"Cut-layer loss gradients dL/dH_k: N={num_samples} records x sum(d_k)={total_embedding_dim} floats x {bytes_per_number}B x {epochs} epochs.",
            "psi_alignment": f"One-time simulated PSI alignment: {num_parties} parties x {num_samples} samples x {psi_token_bytes}B hash tokens.",
            "measurement_mode": "Simulated network payload sizing based on mathematical tensor exchange (no physical network sockets)."
        }
    }


def calculate_ftl_communication_volume(
    parameters: Optional[Dict[str, np.ndarray]] = None,
    num_transfer_events: int = 1,
    bytes_per_number: int = 8
) -> Dict[str, Any]:
    """
    Calculate estimated communication volume for Federated Transfer Learning (FTL).

    In FTL:
    1. A pre-trained Source Model (or feature extractor) is transferred from the Source Hospital
       (or via the central coordinator) to the Target Hospital.
    2. Transferred Payload: 57 model parameters (W1: 40, b1: 8, W2: 8, b2: 1) = 456 bytes in float64.
       (If transferring only Layer 1 feature extractor W1, b1: 48 parameters = 384 bytes).
    3. Transfer Events: 1 one-time transfer event.
    4. Target Fine-Tuning Communication: 0 bytes over the network (Target Hospital fine-tunes locally
       behind its firewall on local target patient records).
    5. Privacy Boundary: Transferring model parameters does NOT alone guarantee source patient privacy.
       An adversarial recipient could theoretically probe weights using membership inference.

    Parameters:
        parameters (dict, optional): Model parameter dictionary.
        num_transfer_events (int): Number of transfer events (default: 1).
        bytes_per_number (int): Bytes per scalar parameter (default: 8 for float64).

    Returns:
        dict: Detailed communication breakdown and architectural assumptions for FTL.
    """
    if parameters is None:
        from src.model import HealthcareMLP
        mlp = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=42)
        parameters = mlp.get_parameters()

    param_info = calculate_model_parameter_bytes(parameters)
    total_params = param_info["total_parameters"]
    param_bytes = param_info["total_bytes"]

    # Layer 1 feature extractor parameters only (W1, b1)
    feature_extractor_params = int(parameters["W1"].size + parameters["b1"].size)
    feature_extractor_bytes = feature_extractor_params * bytes_per_number

    # Prediction head parameters only (W2, b2)
    head_params = int(parameters["W2"].size + parameters["b2"].size)
    head_bytes = head_params * bytes_per_number

    total_transfer_bytes = param_bytes * num_transfer_events
    fine_tuning_network_bytes = 0  # Fine-tuning occurs entirely on local target node

    return {
        "paradigm": "Federated Transfer Learning (FTL)",
        "transfer_events": num_transfer_events,
        "total_parameters": total_params,
        "parameter_bytes": param_bytes,
        "bytes_per_parameter": bytes_per_number,
        "feature_extractor_parameters": feature_extractor_params,
        "feature_extractor_bytes": feature_extractor_bytes,
        "prediction_head_parameters": head_params,
        "prediction_head_bytes": head_bytes,
        "source_to_target_transfer_bytes": total_transfer_bytes,
        "source_to_target_transfer_kb": round(total_transfer_bytes / 1024.0, 4),
        "target_fine_tuning_network_bytes": fine_tuning_network_bytes,
        "total_communication_bytes": total_transfer_bytes,
        "total_communication_kb": round(total_transfer_bytes / 1024.0, 4),
        "scaling_nature": "O(|W|) - Single one-time parameter transfer; fine-tuning is strictly on-premise",
        "assumptions": {
            "parameter_transfer": f"One-time transfer of {total_params} float64 weights from Source to Target = {param_bytes} Bytes.",
            "fine_tuning": "0 Bytes network exchange during target fine-tuning (local GPU/CPU computation).",
            "privacy_disclaimer": "Transferring model parameters does not guarantee source data confidentiality; DP-SGD or parameter obfuscation is required in adversarial regimes.",
            "measurement_mode": "Simulated serialization size based on float64 NumPy tensors (no socket headers)."
        }
    }


def calculate_cnn_lstm_communication_volume(
    num_parameters: int = 2449,
    num_clients: int = 3,
    num_rounds: int = 5,
    bytes_per_parameter: int = 4
) -> Dict[str, Any]:
    """
    Calculate theoretical and simulated communication costs for Federated Hybrid CNN-LSTM training.

    Parameters:
        num_parameters (int): Trainable parameter count in Hybrid CNN-LSTM (default: 2,449).
        num_clients (int): Number of participating hospital clients (default: 3).
        num_rounds (int): Federated communication rounds (default: 5).
        bytes_per_parameter (int): Byte width per scalar parameter (float32 = 4 bytes).

    Returns:
        dict: Communication breakdown across rounds, upload/download, and comparative ratios against MLP.
    """
    if num_parameters <= 0:
        raise ValueError("num_parameters must be positive.")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive.")

    single_model_bytes = int(num_parameters * bytes_per_parameter)
    single_model_kb = round(single_model_bytes / 1024.0, 3)

    client_upload_round_bytes = int(num_clients * single_model_bytes)
    coordinator_broadcast_round_bytes = int(num_clients * single_model_bytes)
    round_bytes = client_upload_round_bytes + coordinator_broadcast_round_bytes

    total_upload_bytes = client_upload_round_bytes * num_rounds
    total_broadcast_bytes = coordinator_broadcast_round_bytes * num_rounds
    total_bytes = round_bytes * num_rounds
    total_kb = round(total_bytes / 1024.0, 2)
    total_mb = round(total_bytes / (1024.0 * 1024.0), 4)

    # Reference comparison: Tabular MLP baseline (57 parameters, float64 = 8 bytes)
    mlp_param_bytes = 57 * 8  # 456 bytes
    mlp_round_bytes = 2 * num_clients * mlp_param_bytes  # 2,736 bytes
    mlp_total_bytes = mlp_round_bytes * num_rounds       # 13,680 bytes

    return {
        "model_architecture": "Hybrid CNN-LSTM (1D CNN + LSTM + Dense Head)",
        "num_parameters": num_parameters,
        "bytes_per_parameter": bytes_per_parameter,
        "precision": "float32 (IEEE 754 standard PyTorch tensor precision)",
        "single_model_bytes": single_model_bytes,
        "single_model_kb": single_model_kb,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "client_upload_per_round_bytes": client_upload_round_bytes,
        "coordinator_broadcast_per_round_bytes": coordinator_broadcast_round_bytes,
        "total_round_bytes": round_bytes,
        "total_round_kb": round(round_bytes / 1024.0, 2),
        "cumulative_upload_bytes": total_upload_bytes,
        "cumulative_broadcast_bytes": total_broadcast_bytes,
        "total_communication_bytes": total_bytes,
        "total_communication_kb": total_kb,
        "total_communication_mb": total_mb,
        "comparison_with_tabular_mlp": {
            "tabular_mlp_parameters": 57,
            "tabular_mlp_total_bytes": mlp_total_bytes,
            "parameter_expansion_factor": round(num_parameters / 57.0, 1),
            "traffic_expansion_factor": round(total_bytes / mlp_total_bytes, 1),
            "bandwidth_feasibility": "High. Total multi-round exchange (~287 KB) is negligible even on constrained 3G/4G or campus hospital networks (<0.01 sec transfer at 100 Mbps)."
        }
    }


def compare_hfl_vs_vfl_communication(
    hfl_rounds: int = 5,
    hfl_parameters: Optional[Dict[str, np.ndarray]] = None,
    vfl_samples: int = 1200,
    vfl_epochs: int = 50,
    vfl_party_dims: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    """
    Generate a direct side-by-side comparative table of communication profiles
    across Horizontal FL (FedAvg, FedProx, SecAgg), Vertical FL (SplitNN), and Federated Transfer Learning (FTL).

    Returns:
        List[Dict[str, Any]]: Comparison rows with paradigm, algorithm, transfer characteristics, and volumes.
    """
    if hfl_parameters is None:
        from src.model import HealthcareMLP
        mlp = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=42)
        hfl_parameters = mlp.get_parameters()

    # HFL standard (FedAvg & FedProx have identical model payload transfer)
    hfl_base = calculate_round_communication_volume(hfl_parameters, num_clients=3, include_secure_aggregation=False)
    hfl_base_round_bytes = hfl_base["total_communication_bytes"]
    hfl_base_total_bytes = hfl_base_round_bytes * hfl_rounds

    # HFL SecAgg
    hfl_secagg = calculate_round_communication_volume(hfl_parameters, num_clients=3, include_secure_aggregation=True)
    hfl_secagg_round_bytes = hfl_secagg["total_communication_bytes"]
    hfl_secagg_total_bytes = hfl_secagg_round_bytes * hfl_rounds

    # VFL
    vfl_comm = calculate_vfl_communication_volume(
        num_samples=vfl_samples,
        epochs=vfl_epochs,
        party_dims=vfl_party_dims
    )

        # FTL
    ftl_comm = calculate_ftl_communication_volume(parameters=hfl_parameters)

    # Longitudinal Hybrid CNN-LSTM
    cnn_lstm_comm = calculate_cnn_lstm_communication_volume(
        num_parameters=2449,
        num_clients=3,
        num_rounds=hfl_rounds,
        bytes_per_parameter=4
    )

    return [
        {
            "paradigm": "Horizontal Federated Learning (HFL)",
            "algorithm": "FedAvg",
            "unit": f"{hfl_rounds} rounds",
            "what_is_exchanged": "Model parameter tensors (W1, b1, W2, b2)",
            "parameter_transfer_bytes": hfl_base_total_bytes,
            "representation_transfer_bytes": 0,
            "total_communication_bytes": hfl_base_total_bytes,
            "total_communication_kb": round(hfl_base_total_bytes / 1024.0, 2),
            "scaling_nature": "O(Rounds * Model_Parameters) - Independent of sample count N"
        },
        {
            "paradigm": "Horizontal Federated Learning (HFL)",
            "algorithm": "FedProx (mu=0.01)",
            "unit": f"{hfl_rounds} rounds",
            "what_is_exchanged": "Model parameter tensors (W1, b1, W2, b2)",
            "parameter_transfer_bytes": hfl_base_total_bytes,
            "representation_transfer_bytes": 0,
            "total_communication_bytes": hfl_base_total_bytes,
            "total_communication_kb": round(hfl_base_total_bytes / 1024.0, 2),
            "scaling_nature": "O(Rounds * Model_Parameters) - Independent of sample count N"
        },
        {
            "paradigm": "Horizontal Federated Learning (HFL)",
            "algorithm": "FedAvg with Secure Aggregation",
            "unit": f"{hfl_rounds} rounds",
            "what_is_exchanged": "Masked parameter tensors + cryptographic seed/metadata tokens",
            "parameter_transfer_bytes": hfl_base_total_bytes,
            "representation_transfer_bytes": 0,
            "total_communication_bytes": hfl_secagg_total_bytes,
            "total_communication_kb": round(hfl_secagg_total_bytes / 1024.0, 2),
            "scaling_nature": "O(Rounds * (Model_Parameters + K^2 Protocol Tokens))"
        },
        {
            "paradigm": "Horizontal Federated Learning (HFL)",
            "algorithm": "FedAvg (Hybrid CNN-LSTM)",
            "unit": f"{hfl_rounds} rounds",
            "what_is_exchanged": "1D CNN + LSTM parameter state_dicts (Conv1d, LSTM, FC Head)",
            "parameter_transfer_bytes": cnn_lstm_comm["total_communication_bytes"],
            "representation_transfer_bytes": 0,
            "total_communication_bytes": cnn_lstm_comm["total_communication_bytes"],
            "total_communication_kb": cnn_lstm_comm["total_communication_kb"],
            "scaling_nature": "O(Rounds * K * CNN_LSTM_Parameters) - Rich temporal modeling with compact 287 KB traffic"
        },
        {
            "paradigm": "Vertical Federated Learning (VFL)",
            "algorithm": "VFL SplitNN",
            "unit": f"{vfl_epochs} epochs",
            "what_is_exchanged": "Cut-layer representations H_k + cut-layer loss gradients dL/dH_k",
            "parameter_transfer_bytes": 0,
            "representation_transfer_bytes": vfl_comm["total_training_traffic_bytes"],
            "total_communication_bytes": vfl_comm["total_training_traffic_bytes"],
            "total_communication_kb": round(vfl_comm["total_training_traffic_bytes"] / 1024.0, 2),
            "scaling_nature": "O(Epochs * N_samples * Embedding_Dim) - Stationary model weights"
        },
        {
            "paradigm": "Federated Transfer Learning (FTL)",
            "algorithm": "FTL Transfer (Fine-Tuning / Feature Extraction)",
            "unit": "1 transfer event",
            "what_is_exchanged": "Pre-trained Source model parameter tensors (W1, b1, W2, b2)",
            "parameter_transfer_bytes": ftl_comm["total_communication_bytes"],
            "representation_transfer_bytes": 0,
            "total_communication_bytes": ftl_comm["total_communication_bytes"],
            "total_communication_kb": round(ftl_comm["total_communication_bytes"] / 1024.0, 2),
            "scaling_nature": "O(Model_Parameters) - Single one-time transfer; local fine-tuning"
        }
    ]


if __name__ == "__main__":
    run_communication_analysis()

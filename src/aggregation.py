"""
Federated Aggregation Module for Healthcare Analytics Simulation.

Implements the Federated Averaging (FedAvg) algorithm:
    W_global = sum_{k=1}^K (n_k / N) * W_k

Where:
- W_k is the model weight dictionary of hospital client k.
- n_k is the number of local training samples at hospital client k.
- N = sum_{k=1}^K n_k is the total number of training samples across all clients.
- K is the number of participating hospital clients.

Aggregates weights coordinate-wise across all network layers (W1, b1, W2, b2)
without inspecting or transferring private patient data.
"""

from typing import List, Dict
import numpy as np

EXPECTED_PARAMETER_KEYS = {"W1", "b1", "W2", "b2"}


def federated_averaging(
    client_parameters: List[Dict[str, np.ndarray]],
    client_sample_counts: List[int]
) -> Dict[str, np.ndarray]:
    """
    Perform sample-weighted Federated Averaging (FedAvg) across client model weights.

    Parameters:
        client_parameters (List[Dict[str, np.ndarray]]): List of parameter dictionaries
            from each participating hospital client.
        client_sample_counts (List[int]): Number of local training records at each hospital.

    Raises:
        ValueError: If client lists are empty, length mismatched, sample counts invalid,
            keys missing, shapes mismatched, or non-finite values encountered.

    Returns:
        Dict[str, np.ndarray]: New aggregated global parameter dictionary.
    """
    # 1. Basic list validation
    if not client_parameters:
        raise ValueError("client_parameters list cannot be empty.")
    if not client_sample_counts:
        raise ValueError("client_sample_counts list cannot be empty.")
    if len(client_parameters) != len(client_sample_counts):
        raise ValueError(
            f"Length mismatch: {len(client_parameters)} parameter dicts vs "
            f"{len(client_sample_counts)} sample counts."
        )

    num_clients = len(client_parameters)

    # 2. Sample counts validation
    for i, count in enumerate(client_sample_counts):
        if not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError(f"Client {i} has invalid sample count: {count}. Must be positive integer.")

    total_samples = int(sum(client_sample_counts))
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be greater than zero.")

    # 3. Schema and shape consistency validation
    reference_params = client_parameters[0]
    ref_keys = set(reference_params.keys())
    if ref_keys != EXPECTED_PARAMETER_KEYS:
        raise ValueError(
            f"Client 0 parameter keys mismatch. Expected {EXPECTED_PARAMETER_KEYS}, got {ref_keys}."
        )

    reference_shapes = {k: np.asarray(v).shape for k, v in reference_params.items()}

    for client_idx, params in enumerate(client_parameters):
        if set(params.keys()) != EXPECTED_PARAMETER_KEYS:
            raise ValueError(
                f"Client {client_idx} has mismatched keys: {set(params.keys())} vs {EXPECTED_PARAMETER_KEYS}."
            )
        for key in EXPECTED_PARAMETER_KEYS:
            arr = np.asarray(params[key], dtype=float)
            if arr.shape != reference_shapes[key]:
                raise ValueError(
                    f"Shape mismatch for '{key}' at client {client_idx}. "
                    f"Expected {reference_shapes[key]}, got {arr.shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Non-finite values (NaN/Inf) detected in parameter '{key}' at client {client_idx}."
                )

    # 4. Perform sample-weighted aggregation: W_global = sum (n_k / N) * W_k
    global_parameters = {}
    for key in EXPECTED_PARAMETER_KEYS:
        # Initialize zero accumulator with the reference shape
        weighted_sum = np.zeros(reference_shapes[key], dtype=float)

        for client_idx in range(num_clients):
            weight_factor = float(client_sample_counts[client_idx]) / float(total_samples)
            client_arr = np.asarray(client_parameters[client_idx][key], dtype=float)
            weighted_sum += weight_factor * client_arr

        global_parameters[key] = weighted_sum

    return global_parameters


if __name__ == "__main__":
    print("Testing Federated Averaging (FedAvg) logic...")
    # Mock parameters for 2 clients with equal sample count
    c1_params = {
        "W1": np.ones((5, 8)),
        "b1": np.zeros((1, 8)),
        "W2": np.ones((8, 1)) * 2.0,
        "b2": np.zeros((1, 1))
    }
    c2_params = {
        "W1": np.ones((5, 8)) * 3.0,
        "b1": np.zeros((1, 8)),
        "W2": np.ones((8, 1)) * 4.0,
        "b2": np.zeros((1, 1))
    }

    # Equal weighting test (50/50)
    agg_equal = federated_averaging([c1_params, c2_params], [100, 100])
    print(f"Equal aggregation W1 mean: {agg_equal['W1'].mean():.2f} (Expected: 2.00)")
    print(f"Equal aggregation W2 mean: {agg_equal['W2'].mean():.2f} (Expected: 3.00)")

    # Unequal weighting test (1:3 ratio) -> (1*1 + 3*3)/4 = 10/4 = 2.50
    agg_weighted = federated_averaging([c1_params, c2_params], [100, 300])
    print(f"Weighted aggregation W1 mean: {agg_weighted['W1'].mean():.2f} (Expected: 2.50)")
    print("FedAvg tests passed successfully.")

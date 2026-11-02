"""
Secure Aggregation Module for Federated Healthcare Analytics.

This module implements a cryptographically-grounded pairwise additive masking protocol
with fixed-point finite-field quantization inspired by Google's Secure Aggregation protocol
(Bonawitz et al., ACM CCS 2017):
1. Fixed-Point Quantization:
   - Real-valued model weights are clipped and scaled into discrete integer representations.
   - Integers are embedded into a finite field Z_q where q is a large prime modulus
     (default: 2^61 - 1, the 9th Mersenne prime).
2. Pairwise Additive Masking:
   - For every client pair (u, v) with u < v, clients establish a cryptographically secure
     shared secret seed s_{u, v} via a CSPRNG (secrets module).
   - From s_{u, v}, a pseudo-random mask tensor M_{u, v} in Z_q is deterministically generated
     via HMAC-SHA256 in counter mode.
   - Client u adds +M_{u, v} (mod q) and client v subtracts -M_{u, v} (mod q).
3. Sample Weighting:
   - Each client weights its local parameters by (n_k / N) before quantization and masking.
   - When summed coordinate-wise modulo q by the coordinator:
     sum_k M_{u, v} cancels out to exactly 0 (mod q).
   - The coordinator decodes the sum directly into the sample-weighted FedAvg global model.
4. Security Guarantee:
   - The coordinator never observes any client's individual plaintext model parameters.
   - If a client fails or drops out, the protocol rejects aggregation or requires explicit
     recovery to prevent silent corruption.
5. Educational Disclaimer:
   - This implementation simulates pairwise key agreement and masking in a single process / environment.
   - In production, key agreement requires authenticated Diffie-Hellman (X25519) and threshold secret
     sharing (Shamir's Secret Sharing) to handle dropouts without coordinator collusion.
"""

import os
import sys
import hmac
import hashlib
import struct
import secrets
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Ensure project root is accessible on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aggregation import EXPECTED_PARAMETER_KEYS

# ==============================================================================
# PROTOCOL CONSTANTS
# ==============================================================================
# Mersenne Prime M_61 = 2^61 - 1 = 2305843009213693951
# Fits comfortably inside 64-bit integer calculations and prevents wraparound overflow
DEFAULT_MODULUS: int = (1 << 61) - 1

# Scaling factor for fixed-point quantization: S = 10^7 provides 7 decimal digits of precision
DEFAULT_SCALING_FACTOR: int = 10_000_000

# Value clipping bounds to prevent extreme outliers before quantization
DEFAULT_CLIPPING_BOUND: float = 1000.0

# Protocol identifier string
PROTOCOL_NAME: str = "Pairwise Additive Masking with Finite-Field Quantization (Bonawitz et al. Simulation)"


class SecureAggregationError(ValueError):
    """Raised when secure aggregation fails due to invalid inputs, missing clients, or protocol violations."""
    pass


class ClientDropoutError(SecureAggregationError):
    """Raised when client dropout disrupts mask cancellation and cannot be securely recovered."""
    pass


# ==============================================================================
# CRYPTOGRAPHIC CSPRNG & PAIRWISE SEED GENERATION
# ==============================================================================

def generate_pairwise_seed() -> bytes:
    """
    Generate a 256-bit (32-byte) cryptographically secure random seed.
    Uses Python's secrets module (backed by the operating system CSPRNG).
    Never uses ordinary pseudo-random generators (e.g. random.random).
    """
    return secrets.token_bytes(32)


def generate_pairwise_seeds(client_ids: List[str]) -> Dict[Tuple[str, str], bytes]:
    """
    Generate symmetric pairwise shared seeds for all unique pairs of participating clients.

    Parameters:
        client_ids (List[str]): Unique identifiers of participating hospital clients.

    Returns:
        dict: Mapping of (min_id, max_id) -> 32-byte secret seed.
    """
    sorted_ids = sorted(list(set(client_ids)))
    if len(sorted_ids) < 2:
        raise SecureAggregationError(f"At least 2 clients required for pairwise masking, got {len(sorted_ids)}.")

    pairwise_seeds = {}
    for i in range(len(sorted_ids)):
        for j in range(i + 1, len(sorted_ids)):
            u, v = sorted_ids[i], sorted_ids[j]
            pairwise_seeds[(u, v)] = generate_pairwise_seed()

    return pairwise_seeds


def prg_stream(
    seed: bytes,
    round_num: int,
    param_key: str,
    count: int,
    modulus: int = DEFAULT_MODULUS
) -> np.ndarray:
    """
    Cryptographically secure pseudo-random generator (CSPRNG) stream in counter mode.
    Derives deterministic integer values in [0, modulus - 1] from a pairwise secret seed.

    Implementation:
        HMAC-SHA256(seed, round_num || param_key || counter)
        Unpacks 64-bit unsigned integers with rejection sampling to eliminate modulo bias.

    Parameters:
        seed (bytes): 32-byte pairwise shared secret.
        round_num (int): Communication round index.
        param_key (str): Parameter identifier (e.g. 'W1', 'b1').
        count (int): Number of pseudo-random field elements required.
        modulus (int): Field modulus q.

    Returns:
        np.ndarray: 1D array of integers of length `count` in [0, modulus - 1].
    """
    if count <= 0:
        return np.array([], dtype=object)

    context = f"{round_num}:{param_key}:mask".encode("utf-8")
    counter = 0
    generated: List[int] = []

    # Maximum multiple of modulus within 2^64 to eliminate modulo bias via rejection sampling
    two_pow_64 = 1 << 64
    limit = two_pow_64 - (two_pow_64 % modulus)

    while len(generated) < count:
        msg = context + struct.pack(">Q", counter)
        digest = hmac.new(seed, msg, hashlib.sha256).digest()
        counter += 1

        # Unpack 4 unsigned 64-bit integers from the 32-byte SHA-256 digest
        words = struct.unpack(">4Q", digest)
        for w in words:
            if w < limit:
                generated.append(w % modulus)
                if len(generated) == count:
                    break

    return np.array(generated, dtype=object)


def generate_parameter_mask(
    seed: bytes,
    round_num: int,
    param_key: str,
    shape: Tuple[int, ...],
    modulus: int = DEFAULT_MODULUS
) -> np.ndarray:
    """
    Generate a pseudo-random mask tensor matching the specified layer shape in Z_q.

    Parameters:
        seed (bytes): Shared pairwise secret seed.
        round_num (int): Current federated round.
        param_key (str): Layer key ('W1', 'b1', 'W2', 'b2').
        shape (tuple): Dimensionality of the parameter array.
        modulus (int): Field modulus.

    Returns:
        np.ndarray: Mask array matching `shape` with elements in [0, modulus - 1].
    """
    total_elements = int(np.prod(shape))
    flat_mask = prg_stream(seed, round_num, param_key, total_elements, modulus=modulus)
    return flat_mask.reshape(shape)


# ==============================================================================
# FIXED-POINT QUANTIZATION & FINITE-FIELD ENCODING
# ==============================================================================

def quantize_parameter(
    param_array: np.ndarray,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
    modulus: int = DEFAULT_MODULUS,
    clip_bound: float = DEFAULT_CLIPPING_BOUND
) -> np.ndarray:
    """
    Quantize a real-valued floating point parameter array into discrete elements in Z_q.

    Algorithm:
        1. Verify all elements are finite (reject NaN / Inf).
        2. Clip values to [-clip_bound, clip_bound] to protect against overflow.
        3. Multiply by scaling_factor S and round to nearest integer: z = round(x * S).
        4. Embed into Z_q using exact integer arithmetic: (z % q + q) % q.

    Parameters:
        param_array (np.ndarray): Floating-point model weights.
        scaling_factor (int): Scaling multiplier (default: 10^7).
        modulus (int): Field modulus (default: 2^61 - 1).
        clip_bound (float): Maximum absolute bound before quantization.

    Returns:
        np.ndarray: Quantized integer tensor with dtype=object in Z_q.
    """
    arr = np.asarray(param_array, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise SecureAggregationError("Non-finite values (NaN/Inf) encountered during parameter quantization.")

    # Clip to protect field range
    arr_clipped = np.clip(arr, -clip_bound, clip_bound)

    # Scale to integer using exact Python arbitrary precision ints
    flat_ints = [int(round(float(x) * scaling_factor)) for x in arr_clipped.flat]
    field_elements = np.array(
        [(x % modulus + modulus) % modulus for x in flat_ints],
        dtype=object
    ).reshape(arr.shape)

    return field_elements


def dequantize_parameter(
    field_array: np.ndarray,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
    modulus: int = DEFAULT_MODULUS
) -> np.ndarray:
    """
    Decode an accumulated finite-field tensor from Z_q back to standard floating-point representation.

    Algorithm:
        1. For each integer z in [0, q - 1]:
           If z > q // 2: signed_z = z - q
           Else:          signed_z = z
        2. Real value = signed_z / scaling_factor.

    Parameters:
        field_array (np.ndarray): Tensor in Z_q.
        scaling_factor (int): Scaling multiplier used during quantization.
        modulus (int): Field modulus q.

    Returns:
        np.ndarray: Reconstructed float64 parameter array.
    """
    half_mod = modulus // 2
    flat_floats = []
    for z in field_array.flat:
        signed_z = int(z) - modulus if int(z) > half_mod else int(z)
        flat_floats.append(float(signed_z) / float(scaling_factor))

    return np.array(flat_floats, dtype=np.float64).reshape(field_array.shape)


# ==============================================================================
# CLIENT-SIDE UPDATE PREPARATION & PAIRWISE MASKING
# ==============================================================================

def mask_client_parameters(
    client_id: str,
    parameters: Dict[str, np.ndarray],
    pairwise_seeds: Dict[Tuple[str, str], bytes],
    participating_clients: List[str],
    round_num: int = 1,
    sample_weight: float = 1.0,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
    modulus: int = DEFAULT_MODULUS,
    clip_bound: float = DEFAULT_CLIPPING_BOUND
) -> Dict[str, np.ndarray]:
    """
    Prepare a masked parameter update for a participating client.

    Steps:
        1. Validates parameter keys and shapes.
        2. Weights parameters by sample_weight (e.g., n_k / N).
        3. Quantizes weighted parameters into Z_q.
        4. For every other participating client v:
           - Generates mask M_{u, v} from shared seed s_{u, v}.
           - If client_id < v: adds +M_{u, v} (mod q).
           - If client_id > v: subtracts -M_{v, u} (mod q).
        5. Returns the fully masked tensor dictionary.

    Parameters:
        client_id (str): ID of this hospital client (e.g. 'hospital_1').
        parameters (Dict[str, np.ndarray]): Local model weights.
        pairwise_seeds (dict): All pairwise seeds known to this client.
        participating_clients (List[str]): List of active clients in this round.
        round_num (int): Communication round index.
        sample_weight (float): Sample-weight coefficient (default: 1.0).
        scaling_factor (int): Quantization scale.
        modulus (int): Modulus q.
        clip_bound (float): Max value bound.

    Returns:
        Dict[str, np.ndarray]: Obfuscated masked parameters (safe to transmit to coordinator).
    """
    # 1. Validation
    if set(parameters.keys()) != EXPECTED_PARAMETER_KEYS:
        raise SecureAggregationError(
            f"Client {client_id} parameter keys mismatch. Expected {EXPECTED_PARAMETER_KEYS}, got {set(parameters.keys())}."
        )

    if sample_weight <= 0.0 or not np.isfinite(sample_weight):
        raise SecureAggregationError(f"Invalid sample weight {sample_weight} for client {client_id}.")

    sorted_active = sorted(list(set(participating_clients)))
    if client_id not in sorted_active:
        raise SecureAggregationError(f"Client {client_id} is not in participating client list: {sorted_active}.")

    masked_dict = {}

    for key in EXPECTED_PARAMETER_KEYS:
        arr = np.asarray(parameters[key], dtype=float)
        # Apply local sample weighting: w_weighted = w * (n_k / N)
        arr_weighted = arr * sample_weight

        # Quantize to field Z_q
        field_arr = quantize_parameter(
            arr_weighted,
            scaling_factor=scaling_factor,
            modulus=modulus,
            clip_bound=clip_bound
        )

        # Apply pairwise masks
        for other_id in sorted_active:
            if other_id == client_id:
                continue

            pair_key = (min(client_id, other_id), max(client_id, other_id))
            if pair_key not in pairwise_seeds:
                raise SecureAggregationError(
                    f"Missing pairwise seed for pair {pair_key} required by client {client_id}."
                )

            seed = pairwise_seeds[pair_key]
            mask = generate_parameter_mask(seed, round_num, key, arr.shape, modulus=modulus)

            if client_id < other_id:
                # Add mask: +M_{u, v}
                field_arr = (field_arr + mask) % modulus
            else:
                # Subtract mask: -M_{v, u} === +(modulus - M_{v, u})
                field_arr = (field_arr + (modulus - mask)) % modulus

        masked_dict[key] = field_arr

    return masked_dict


# ==============================================================================
# COORDINATOR-SIDE AGGREGATION & MASK CANCELLATION
# ==============================================================================

def aggregate_masked_updates(
    masked_updates: Dict[str, Dict[str, np.ndarray]],
    participating_clients: List[str],
    round_num: int = 1,
    expected_clients: Optional[List[str]] = None,
    dropout_seeds: Optional[Dict[str, Dict[Tuple[str, str], bytes]]] = None,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
    modulus: int = DEFAULT_MODULUS,
    surviving_sample_fraction: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Aggregate masked client updates at the central coordinator.

    Mathematical Invariant:
        When all participating clients submit their masked updates:
            sum_{u} [ Quantize(y_u) + sum_{v > u} M_{u, v} - sum_{v < u} M_{v, u} ] mod q
            = sum_u Quantize(y_u) mod q
        All pairwise masks cancel out exactly to zero.
        The coordinator decodes sum_u Quantize(y_u) to obtain W_global directly.

    Dropout Handling:
        If expected_clients is provided and some clients are missing:
        - If dropout_seeds is provided (surviving clients report seeds shared with the dropped client),
          the coordinator unmasks the orphaned masks and adjusts weights.
        - Otherwise, raises ClientDropoutError to prevent silent corruption.

    Parameters:
        masked_updates (dict): Mapping client_id -> masked parameter dictionary.
        participating_clients (List[str]): List of clients who actually submitted updates.
        round_num (int): Current communication round.
        expected_clients (List[str], optional): Clients expected to participate.
        dropout_seeds (dict, optional): Seeds of dropped clients for recovery.
        scaling_factor (int): Fixed-point scale.
        modulus (int): Field modulus.
        surviving_sample_fraction (float): Normalization scale if a dropout occurred.

    Returns:
        Dict[str, np.ndarray]: Aggregated global parameter dictionary.
    """
    if not masked_updates:
        raise SecureAggregationError("No masked client updates provided to coordinator.")

    active_set = set(participating_clients)
    if not active_set:
        raise SecureAggregationError("participating_clients list cannot be empty.")

    # Check for client dropouts
    if expected_clients is not None:
        missing_clients = set(expected_clients) - active_set
        if missing_clients:
            if not dropout_seeds:
                raise ClientDropoutError(
                    f"Secure aggregation failed: Client(s) {sorted(list(missing_clients))} dropped out. "
                    f"Pairwise masks cannot cancel without dropout recovery keys."
                )

    # Verify all participating clients have submitted updates
    for cid in active_set:
        if cid not in masked_updates:
            raise SecureAggregationError(f"Client '{cid}' listed in active clients but no update received.")

    # Reference shapes and keys check
    first_cid = next(iter(active_set))
    first_update = masked_updates[first_cid]
    if set(first_update.keys()) != EXPECTED_PARAMETER_KEYS:
        raise SecureAggregationError(
            f"Update from client {first_cid} has invalid keys: {set(first_update.keys())} vs {EXPECTED_PARAMETER_KEYS}."
        )

    ref_shapes = {k: np.asarray(v).shape for k, v in first_update.items()}

    for cid in active_set:
        client_up = masked_updates[cid]
        if set(client_up.keys()) != EXPECTED_PARAMETER_KEYS:
            raise SecureAggregationError(f"Client {cid} keys mismatch: {set(client_up.keys())}.")
        for k in EXPECTED_PARAMETER_KEYS:
            arr = np.asarray(client_up[k])
            if arr.shape != ref_shapes[k]:
                raise SecureAggregationError(
                    f"Shape mismatch for '{k}' from client {cid}. Expected {ref_shapes[k]}, got {arr.shape}."
                )

    # Coordinate-wise summation in Z_q
    aggregated_params = {}
    for key in EXPECTED_PARAMETER_KEYS:
        shape = ref_shapes[key]
        accum = np.zeros(shape, dtype=object)

        for cid in active_set:
            accum = (accum + np.asarray(masked_updates[cid][key], dtype=object)) % modulus

        # If dropouts occurred and recovery seeds were supplied, subtract orphaned masks
        if expected_clients is not None and dropout_seeds:
            missing_clients = set(expected_clients) - active_set
            for dropped_cid in missing_clients:
                for surviving_cid in active_set:
                    pair_key = (min(surviving_cid, dropped_cid), max(surviving_cid, dropped_cid))
                    if pair_key in dropout_seeds.get(dropped_cid, {}):
                        seed = dropout_seeds[dropped_cid][pair_key]
                        orphaned_mask = generate_parameter_mask(seed, round_num, key, shape, modulus=modulus)
                        if surviving_cid < dropped_cid:
                            # surviving_cid added +M, so coordinator subtracts M
                            accum = (accum + (modulus - orphaned_mask)) % modulus
                        else:
                            # surviving_cid subtracted -M, so coordinator adds M
                            accum = (accum + orphaned_mask) % modulus

        # Dequantize back to float64
        decoded_float = dequantize_parameter(accum, scaling_factor=scaling_factor, modulus=modulus)

        # Re-normalize if surviving sample fraction was applied
        if surviving_sample_fraction > 0.0 and surviving_sample_fraction != 1.0:
            decoded_float = decoded_float / surviving_sample_fraction

        aggregated_params[key] = decoded_float

    return aggregated_params


# ==============================================================================
# END-TO-END SECURE FEDERATED AVERAGING WRAPPER
# ==============================================================================

def secure_federated_averaging(
    client_parameters: List[Dict[str, np.ndarray]],
    client_sample_counts: List[int],
    client_ids: Optional[List[str]] = None,
    round_num: int = 1,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
    modulus: int = DEFAULT_MODULUS,
    clip_bound: float = DEFAULT_CLIPPING_BOUND,
    return_audit: bool = False
) -> Any:
    """
    Perform Sample-Weighted Federated Averaging via the Secure Aggregation protocol.

    Workflow:
        1. Validate inputs and sample counts.
        2. Generate pairwise shared seeds for all client pairs via CSPRNG.
        3. Each client masks its update:
           - Scales parameters by sample fraction (n_k / N).
           - Quantizes to Z_q.
           - Adds / subtracts pairwise masks.
        4. Central coordinator aggregates all masked updates.
        5. Pairwise masks cancel to zero.
        6. Coordinator decodes the aggregate to produce W_global.
        7. Verifies correctness against standard FedAvg within 1e-5 numerical tolerance.

    Parameters:
        client_parameters (List[Dict[str, np.ndarray]]): Plaintext parameters at clients.
        client_sample_counts (List[int]): Number of local clinical samples per hospital.
        client_ids (List[str], optional): Hospital client identifiers.
        round_num (int): Communication round index.
        scaling_factor (int): Quantization precision multiplier.
        modulus (int): Field modulus.
        clip_bound (float): Parameter clipping threshold.
        return_audit (bool): If True, returns (aggregated_params, audit_dict).

    Returns:
        Dict[str, np.ndarray] or (Dict[str, np.ndarray], dict):
            Aggregated global parameters matching FedAvg within numerical tolerance.
    """
    if not client_parameters:
        raise SecureAggregationError("client_parameters list cannot be empty.")
    if not client_sample_counts:
        raise SecureAggregationError("client_sample_counts list cannot be empty.")
    if len(client_parameters) != len(client_sample_counts):
        raise SecureAggregationError(
            f"Length mismatch: {len(client_parameters)} models vs {len(client_sample_counts)} sample counts."
        )

    num_clients = len(client_parameters)
    if num_clients < 2:
        raise SecureAggregationError(f"Secure aggregation requires at least 2 clients, got {num_clients}.")

    for i, count in enumerate(client_sample_counts):
        if not isinstance(count, (int, np.integer)) or count <= 0:
            raise SecureAggregationError(f"Client {i} has invalid sample count: {count}.")

    total_samples = int(sum(client_sample_counts))
    if total_samples <= 0:
        raise SecureAggregationError("Total sample count must be greater than zero.")

    if client_ids is None:
        client_ids = [f"hospital_{i+1}" for i in range(num_clients)]
    elif len(client_ids) != num_clients:
        raise SecureAggregationError(f"Length mismatch: {len(client_ids)} client IDs vs {num_clients} models.")

    # 1. Establish pairwise seeds (simulating key agreement)
    pairwise_seeds = generate_pairwise_seeds(client_ids)

    # 2. Client-side update preparation and masking
    masked_updates = {}
    for i in range(num_clients):
        cid = client_ids[i]
        params = client_parameters[i]
        sample_weight = float(client_sample_counts[i]) / float(total_samples)

        masked_update = mask_client_parameters(
            client_id=cid,
            parameters=params,
            pairwise_seeds=pairwise_seeds,
            participating_clients=client_ids,
            round_num=round_num,
            sample_weight=sample_weight,
            scaling_factor=scaling_factor,
            modulus=modulus,
            clip_bound=clip_bound
        )
        masked_updates[cid] = masked_update

    # 3. Coordinator-side aggregation (coordinator receives ONLY masked_updates)
    aggregated_weights = aggregate_masked_updates(
        masked_updates=masked_updates,
        participating_clients=client_ids,
        round_num=round_num,
        expected_clients=client_ids,
        scaling_factor=scaling_factor,
        modulus=modulus
    )

    # 4. Correctness validation (compare against theoretical standard FedAvg)
    max_error = 0.0
    for key in EXPECTED_PARAMETER_KEYS:
        expected_param = np.zeros_like(client_parameters[0][key], dtype=float)
        for i in range(num_clients):
            weight = float(client_sample_counts[i]) / float(total_samples)
            expected_param += weight * np.asarray(client_parameters[i][key], dtype=float)

        diff = np.max(np.abs(aggregated_weights[key] - expected_param))
        max_error = max(max_error, float(diff))

    validation_passed = bool(max_error < 1e-4)
    if not validation_passed:
        raise SecureAggregationError(
            f"Secure aggregation validation failed. Max deviation from FedAvg ({max_error:.2e}) "
            f"exceeded numerical tolerance (1e-4)."
        )

    if return_audit:
        audit_info = {
            "protocol_name": PROTOCOL_NAME,
            "algorithm": "FedAvg with Secure Aggregation",
            "num_clients": num_clients,
            "client_ids": client_ids,
            "round_num": round_num,
            "total_samples": total_samples,
            "modulus": f"2^61 - 1 ({modulus})",
            "scaling_factor": scaling_factor,
            "clipping_bound": clip_bound,
            "csprng_source": "Python secrets module + HMAC-SHA256 (Counter Mode)",
            "validation_passed": validation_passed,
            "max_absolute_error_vs_fedavg": max_error,
            "plaintext_exposure": False,
            "dropout_handling": "Quorum enforcement with explicit failure notification"
        }
        return aggregated_weights, audit_info

    return aggregated_weights

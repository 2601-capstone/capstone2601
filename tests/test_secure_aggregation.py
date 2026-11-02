"""
Comprehensive Unit Tests for Secure Aggregation Module (src/secure_aggregation.py).

Verifies Phase 11 requirements:
- Cryptographic pairwise seed generation via secrets module
- Deterministic CSPRNG parameter mask generation
- Exact pairwise mask cancellation
- Quantization precision in finite field Z_q
- Client update obfuscation (zero plaintext leakage)
- Correct weighted aggregation matching FedAvg within 1e-5 numerical tolerance
- Unequal sample count weighting
- Defensive input rejection (NaN, Inf, shape mismatch, key mismatch)
- Client dropout detection and explicit failure rejection
- Dropout recovery via surviving client seeds
- End-to-end integration with the central experiment coordinator
- Preservation of existing FedAvg and FedProx behavior
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from src.secure_aggregation import (
    generate_pairwise_seed,
    generate_pairwise_seeds,
    generate_parameter_mask,
    quantize_parameter,
    dequantize_parameter,
    mask_client_parameters,
    aggregate_masked_updates,
    secure_federated_averaging,
    SecureAggregationError,
    ClientDropoutError,
    DEFAULT_MODULUS,
    DEFAULT_SCALING_FACTOR,
    DEFAULT_CLIPPING_BOUND,
    PROTOCOL_NAME
)
from src.aggregation import federated_averaging, EXPECTED_PARAMETER_KEYS
from src.communication_analysis import (
    calculate_secure_aggregation_overhead,
    calculate_round_communication_volume
)
from src.model import HealthcareMLP
from src.experiment import run_federated_experiment


class TestSecureAggregationProtocol(unittest.TestCase):
    """Test suite covering cryptographic masking, numerical precision, and coordinator aggregation."""

    def setUp(self):
        """Set up deterministic parameter fixtures for 3 hospital clients."""
        self.client_ids = ["hospital_1", "hospital_2", "hospital_3"]

        # Deterministic dummy model parameters for 3 clients
        self.p1 = {
            "W1": np.ones((5, 8)) * 0.5,
            "b1": np.zeros((1, 8)) + 0.1,
            "W2": np.ones((8, 1)) * 1.5,
            "b2": np.zeros((1, 1)) + 0.2
        }
        self.p2 = {
            "W1": np.ones((5, 8)) * -0.2,
            "b1": np.zeros((1, 8)) - 0.3,
            "W2": np.ones((8, 1)) * 2.0,
            "b2": np.zeros((1, 1)) - 0.1
        }
        self.p3 = {
            "W1": np.ones((5, 8)) * 0.8,
            "b1": np.zeros((1, 8)) + 0.4,
            "W2": np.ones((8, 1)) * 0.5,
            "b2": np.zeros((1, 1)) + 0.6
        }
        self.client_params = [self.p1, self.p2, self.p3]
        self.sample_counts = [400, 400, 400]

    def test_pairwise_seed_generation(self):
        """Verify generation of 256-bit cryptographically secure pairwise seeds."""
        seed = generate_pairwise_seed()
        self.assertIsInstance(seed, bytes)
        self.assertEqual(len(seed), 32)

        # Pairwise map for 3 clients has 3 unique pairs: (H1, H2), (H1, H3), (H2, H3)
        pairwise_map = generate_pairwise_seeds(self.client_ids)
        expected_pairs = {
            ("hospital_1", "hospital_2"),
            ("hospital_1", "hospital_3"),
            ("hospital_2", "hospital_3")
        }
        self.assertEqual(set(pairwise_map.keys()), expected_pairs)

        for pair, s in pairwise_map.items():
            self.assertEqual(len(s), 32)
            self.assertIsInstance(s, bytes)

        # All pairwise seeds must be mutually distinct
        all_seeds = list(pairwise_map.values())
        self.assertEqual(len(all_seeds), len(set(all_seeds)))

        # Reject fewer than 2 clients
        with self.assertRaises(SecureAggregationError):
            generate_pairwise_seeds(["hospital_1"])

    def test_csprng_parameter_mask_properties(self):
        """Verify CSPRNG mask shapes, bounds in Z_q, and deterministic repeatability."""
        seed = generate_pairwise_seed()
        shape = (5, 8)
        mask = generate_parameter_mask(seed, round_num=1, param_key="W1", shape=shape, modulus=DEFAULT_MODULUS)

        # Shape verification
        self.assertEqual(mask.shape, shape)

        # Field element range verification: all in [0, modulus - 1]
        for val in mask.flat:
            self.assertGreaterEqual(int(val), 0)
            self.assertLess(int(val), DEFAULT_MODULUS)

        # Deterministic reproducibility: same seed + round + key yields identical mask
        mask_repeat = generate_parameter_mask(seed, round_num=1, param_key="W1", shape=shape, modulus=DEFAULT_MODULUS)
        self.assertTrue(np.array_equal(mask, mask_repeat))

        # Different round yields different mask
        mask_round2 = generate_parameter_mask(seed, round_num=2, param_key="W1", shape=shape, modulus=DEFAULT_MODULUS)
        self.assertFalse(np.array_equal(mask, mask_round2))

        # Different parameter key yields different mask
        mask_b1 = generate_parameter_mask(seed, round_num=1, param_key="b1", shape=(1, 8), modulus=DEFAULT_MODULUS)
        self.assertFalse(np.array_equal(mask.flat[:8], mask_b1.flat[:8]))

    def test_pairwise_mask_exact_cancellation(self):
        """Verify that pairwise masks sum to zero coordinate-wise modulo q."""
        pairwise_seeds = generate_pairwise_seeds(self.client_ids)
        shape = (5, 8)
        modulus = DEFAULT_MODULUS

        # hospital_1: +M(1,2) + M(1,3)
        m12 = generate_parameter_mask(pairwise_seeds[("hospital_1", "hospital_2")], 1, "W1", shape, modulus)
        m13 = generate_parameter_mask(pairwise_seeds[("hospital_1", "hospital_3")], 1, "W1", shape, modulus)
        h1_mask = (m12 + m13) % modulus

        # hospital_2: -M(1,2) + M(2,3)
        m23 = generate_parameter_mask(pairwise_seeds[("hospital_2", "hospital_3")], 1, "W1", shape, modulus)
        h2_mask = ((modulus - m12) + m23) % modulus

        # hospital_3: -M(1,3) - M(2,3)
        h3_mask = ((modulus - m13) + (modulus - m23)) % modulus

        # Sum of all client masks must equal exactly zero (modulus)
        mask_sum = (h1_mask + h2_mask + h3_mask) % modulus
        self.assertTrue(np.all(mask_sum == 0))

    def test_quantization_and_dequantization_precision(self):
        """Verify fixed-point quantization round-trip numerical precision (< 1e-6 error)."""
        test_array = np.array([
            [-10.54321, -0.00001, 0.0],
            [0.1234567, 45.98765, 999.999]
        ])

        quantized = quantize_parameter(test_array, scaling_factor=DEFAULT_SCALING_FACTOR, modulus=DEFAULT_MODULUS)
        dequantized = dequantize_parameter(quantized, scaling_factor=DEFAULT_SCALING_FACTOR, modulus=DEFAULT_MODULUS)

        # Max error between original and dequantized must be within 1e-6
        max_error = np.max(np.abs(test_array - dequantized))
        self.assertLess(max_error, 1e-6)

        # Test clipping of extreme values
        extreme_array = np.array([[-2000.0, 1500.0]])
        q_clipped = quantize_parameter(extreme_array, clip_bound=DEFAULT_CLIPPING_BOUND)
        deq_clipped = dequantize_parameter(q_clipped)
        self.assertAlmostEqual(deq_clipped[0, 0], -1000.0, places=4)
        self.assertAlmostEqual(deq_clipped[0, 1], 1000.0, places=4)

    def test_client_parameter_masking_obfuscation(self):
        """Verify client updates are fully obfuscated (individual plaintext is never exposed)."""
        pairwise_seeds = generate_pairwise_seeds(self.client_ids)

        masked_update = mask_client_parameters(
            client_id="hospital_1",
            parameters=self.p1,
            pairwise_seeds=pairwise_seeds,
            participating_clients=self.client_ids,
            round_num=1,
            sample_weight=1.0 / 3.0
        )

        for key in EXPECTED_PARAMETER_KEYS:
            masked_arr = masked_update[key]
            orig_arr = self.p1[key]

            # Masked elements must be in Z_q and look pseudo-random
            self.assertEqual(masked_arr.shape, orig_arr.shape)
            # Masked values must NOT equal original values
            self.assertFalse(np.all(masked_arr == orig_arr))
            # Average magnitude in Z_q should be huge (order of 10^17 - 10^18)
            mean_val = float(np.mean(masked_arr))
            self.assertGreater(mean_val, 1e15)

    def test_secure_aggregation_equal_weights(self):
        """Verify secure aggregation with equal weights matches theoretical FedAvg."""
        aggregated, audit = secure_federated_averaging(
            client_parameters=self.client_params,
            client_sample_counts=self.sample_counts,
            client_ids=self.client_ids,
            round_num=1,
            return_audit=True
        )

        expected = federated_averaging(self.client_params, self.sample_counts)

        # Audit checks
        self.assertTrue(audit["validation_passed"])
        self.assertFalse(audit["plaintext_exposure"])
        self.assertEqual(audit["num_clients"], 3)
        self.assertEqual(audit["protocol_name"], PROTOCOL_NAME)

        # Numerical equivalence coordinate-wise
        for key in EXPECTED_PARAMETER_KEYS:
            diff = np.max(np.abs(aggregated[key] - expected[key]))
            self.assertLess(diff, 1e-5, f"Discrepancy for parameter '{key}' was {diff:.2e}")

    def test_secure_aggregation_unequal_weights(self):
        """Verify secure aggregation handles unequal hospital sample distributions correctly."""
        unequal_counts = [200, 400, 600]  # total = 1200 (1/6, 2/6, 3/6)
        aggregated = secure_federated_averaging(
            client_parameters=self.client_params,
            client_sample_counts=unequal_counts,
            client_ids=self.client_ids,
            round_num=2
        )

        expected = federated_averaging(self.client_params, unequal_counts)

        for key in EXPECTED_PARAMETER_KEYS:
            diff = np.max(np.abs(aggregated[key] - expected[key]))
            self.assertLess(diff, 1e-5, f"Discrepancy for unequal weights on '{key}' was {diff:.2e}")

    def test_invalid_input_rejections(self):
        """Verify defensive validation against invalid models, shapes, and inputs."""
        # Empty parameters list
        with self.assertRaises(SecureAggregationError):
            secure_federated_averaging([], [])

        # Mismatched lengths
        with self.assertRaises(SecureAggregationError):
            secure_federated_averaging(self.client_params, [400, 400])

        # Invalid sample count (zero or negative)
        with self.assertRaises(SecureAggregationError):
            secure_federated_averaging(self.client_params, [400, -10, 400])

        # Missing keys
        bad_params = [dict(self.p1), dict(self.p2), {"W1": self.p3["W1"]}]
        with self.assertRaises(SecureAggregationError):
            secure_federated_averaging(bad_params, self.sample_counts)

        # Non-finite values (NaN / Inf)
        nan_p = {k: v.copy() for k, v in self.p1.items()}
        nan_p["W1"][0, 0] = np.nan
        with self.assertRaises(SecureAggregationError):
            secure_federated_averaging([nan_p, self.p2, self.p3], self.sample_counts)

        # Single client rejection (at least 2 clients required for secure aggregation)
        with self.assertRaises(SecureAggregationError):
            secure_federated_averaging([self.p1], [400])

    def test_client_dropout_without_recovery_raises_error(self):
        """Verify that a dropped client raises ClientDropoutError and does NOT silently corrupt the aggregate."""
        pairwise_seeds = generate_pairwise_seeds(self.client_ids)
        u1 = mask_client_parameters("hospital_1", self.p1, pairwise_seeds, self.client_ids, 1, 400 / 1200)
        u2 = mask_client_parameters("hospital_2", self.p2, pairwise_seeds, self.client_ids, 1, 400 / 1200)

        # hospital_3 dropped out! Coordinator only receives hospital_1 and hospital_2 updates
        masked_updates = {"hospital_1": u1, "hospital_2": u2}

        # Attempt aggregation with missing client and no recovery keys
        with self.assertRaises(ClientDropoutError) as ctx:
            aggregate_masked_updates(
                masked_updates=masked_updates,
                participating_clients=["hospital_1", "hospital_2"],
                round_num=1,
                expected_clients=self.client_ids
            )

        self.assertIn("hospital_3", str(ctx.exception))
        self.assertIn("dropped out", str(ctx.exception))

    def test_client_dropout_with_recovery_succeeds(self):
        """Verify that when dropout recovery keys are provided, orphaned masks cancel exactly."""
        pairwise_seeds = generate_pairwise_seeds(self.client_ids)
        u1 = mask_client_parameters("hospital_1", self.p1, pairwise_seeds, self.client_ids, 1, 400 / 1200)
        u2 = mask_client_parameters("hospital_2", self.p2, pairwise_seeds, self.client_ids, 1, 400 / 1200)

        masked_updates = {"hospital_1": u1, "hospital_2": u2}

        # Surviving clients report seeds shared with hospital_3
        dropout_recovery_seeds = {
            "hospital_3": {
                ("hospital_1", "hospital_3"): pairwise_seeds[("hospital_1", "hospital_3")],
                ("hospital_2", "hospital_3"): pairwise_seeds[("hospital_2", "hospital_3")]
            }
        }

        # Surviving fraction = (400 + 400) / 1200 = 800 / 1200
        surviving_fraction = 800.0 / 1200.0

        recovered_agg = aggregate_masked_updates(
            masked_updates=masked_updates,
            participating_clients=["hospital_1", "hospital_2"],
            round_num=1,
            expected_clients=self.client_ids,
            dropout_seeds=dropout_recovery_seeds,
            surviving_sample_fraction=surviving_fraction
        )

        # Expected aggregate is FedAvg across hospital_1 and hospital_2 only
        expected_surviving = federated_averaging([self.p1, self.p2], [400, 400])

        for key in EXPECTED_PARAMETER_KEYS:
            diff = np.max(np.abs(recovered_agg[key] - expected_surviving[key]))
            self.assertLess(diff, 1e-5, f"Recovery discrepancy on '{key}' was {diff:.2e}")

    def test_communication_overhead_calculations(self):
        """Verify Secure Aggregation communication overhead math."""
        overhead = calculate_secure_aggregation_overhead(num_clients=3)

        self.assertEqual(overhead["num_clients"], 3)
        self.assertEqual(overhead["key_exchange_bytes_per_client"], 2 * 32)  # 64 B
        self.assertEqual(overhead["metadata_bytes_per_client"], 64)          # 64 B
        self.assertEqual(overhead["upload_overhead_per_client"], 128)        # 128 B
        self.assertEqual(overhead["total_upload_overhead_bytes"], 3 * 128)   # 384 B
        self.assertEqual(overhead["total_download_overhead_bytes"], 64)      # 64 B
        self.assertEqual(overhead["total_overhead_bytes"], 448)              # 448 B

        # Verify round volume with secagg enabled
        vol_plain = calculate_round_communication_volume(self.p1, num_clients=3, include_secure_aggregation=False)
        vol_secagg = calculate_round_communication_volume(self.p1, num_clients=3, include_secure_aggregation=True)

        self.assertEqual(vol_plain["total_communication_bytes"], 2736)
        self.assertEqual(vol_secagg["total_communication_bytes"], 2736 + 448)
        self.assertTrue(vol_secagg["is_secure_aggregation"])
        self.assertEqual(vol_secagg["secure_aggregation_overhead_bytes"], 448)


class TestSecureAggregationIntegration(unittest.TestCase):
    """Test suite verifying end-to-end integration with the central experiment coordinator."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_experiment_coordinator_fedavg_with_secure_aggregation(self):
        """Verify run_federated_experiment executes successfully with Secure Aggregation."""
        hist_path = os.path.join(self.tmp_dir, "secagg_hist.csv")

        model, hist_df = run_federated_experiment(
            num_rounds=2,
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            random_seed=42,
            save_history=True,
            history_output_path=hist_path,
            algorithm="FedAvg with Secure Aggregation"
        )

        # 1. Returned model is HealthcareMLP
        self.assertIsInstance(model, HealthcareMLP)
        params = model.get_parameters()
        self.assertEqual(params["W1"].shape, (5, 8))
        self.assertEqual(params["b1"].shape, (1, 8))
        self.assertEqual(params["W2"].shape, (8, 1))
        self.assertEqual(params["b2"].shape, (1, 1))

        # 2. DataFrame checks
        self.assertEqual(len(hist_df), 2)
        self.assertEqual(list(hist_df["round"]), [1, 2])
        self.assertEqual(hist_df["algorithm"].iloc[0], "FedAvg with Secure Aggregation")
        self.assertTrue(bool(hist_df["secure_aggregation_enabled"].iloc[0]))
        self.assertEqual(int(hist_df["secagg_overhead_bytes"].iloc[0]), 448)
        self.assertTrue(bool(hist_df["secagg_validation_passed"].iloc[0]))

        # 3. CSV file checks
        self.assertTrue(os.path.exists(hist_path))
        loaded_df = pd.read_csv(hist_path)
        self.assertEqual(len(loaded_df), 2)

    def test_existing_fedavg_and_fedprox_unmodified(self):
        """Verify that existing FedAvg and FedProx options remain completely intact."""
        # FedAvg
        hist_fedavg = os.path.join(self.tmp_dir, "fedavg_hist.csv")
        m_avg, df_avg = run_federated_experiment(
            num_rounds=1,
            local_epochs=1,
            random_seed=42,
            save_history=True,
            history_output_path=hist_fedavg,
            algorithm="FedAvg"
        )
        self.assertEqual(df_avg["algorithm"].iloc[0], "FedAvg")
        self.assertNotIn("secure_aggregation_enabled", df_avg.columns)

        # FedProx
        hist_fedprox = os.path.join(self.tmp_dir, "fedprox_hist.csv")
        m_prx, df_prx = run_federated_experiment(
            num_rounds=1,
            local_epochs=1,
            random_seed=42,
            save_history=True,
            history_output_path=hist_fedprox,
            algorithm="FedProx",
            mu=0.01
        )
        self.assertEqual(df_prx["algorithm"].iloc[0], "FedProx")
        self.assertEqual(float(df_prx["mu"].iloc[0]), 0.01)
        self.assertNotIn("secure_aggregation_enabled", df_prx.columns)


if __name__ == "__main__":
    unittest.main()


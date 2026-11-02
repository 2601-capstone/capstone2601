"""
Unit tests for Communication Efficiency Analysis Module (src/communication_analysis.py).
"""

import unittest
import numpy as np

from src.communication_analysis import (
    count_model_parameters,
    calculate_model_parameter_bytes,
    calculate_client_upload_bytes,
    calculate_client_download_bytes,
    calculate_round_communication_volume,
    calculate_cumulative_communication,
    summarize_communication_efficiency
)
from src.model import HealthcareMLP


class TestCommunicationAnalysis(unittest.TestCase):
    """Test suite for model parameter sizing, communication volumes, and timing metrics."""

    @classmethod
    def setUpClass(cls):
        cls.mlp = HealthcareMLP(input_dim=5, hidden_dim=8, output_dim=1, random_seed=42)
        cls.params = cls.mlp.get_parameters()

    def test_count_model_parameters_known_architecture(self):
        """Verify parameter counting on the 5-8-1 HealthcareMLP: (5*8) + 8 + (8*1) + 1 = 57."""
        param_count = count_model_parameters(self.params)
        self.assertEqual(param_count, 57)

    def test_count_model_parameters_dynamic_shapes(self):
        """Verify dynamic parameter counting on arbitrary multi-dimensional tensors."""
        custom_params = {
            "weight_matrix": np.zeros((10, 20)),  # 200
            "bias_vector": np.zeros((1, 20)),     # 20
            "scalar_weight": np.zeros((1, 1))     # 1
        }
        self.assertEqual(count_model_parameters(custom_params), 221)

    def test_calculate_model_parameter_bytes_float64(self):
        """Verify exact byte size calculation for float64 tensors (57 * 8 = 456 bytes)."""
        size_info = calculate_model_parameter_bytes(self.params)

        self.assertEqual(size_info["total_parameters"], 57)
        self.assertEqual(size_info["total_bytes"], 456)
        self.assertEqual(size_info["bytes_per_parameter"], 8)
        self.assertEqual(size_info["dtype"], "float64")
        self.assertAlmostEqual(size_info["total_kilobytes"], 456 / 1024.0, places=4)

        # Layer breakdown assertions
        breakdown = size_info["layer_breakdown"]
        self.assertEqual(breakdown["W1"]["shape"], [5, 8])
        self.assertEqual(breakdown["W1"]["bytes"], 40 * 8)
        self.assertEqual(breakdown["b1"]["bytes"], 8 * 8)
        self.assertEqual(breakdown["W2"]["bytes"], 8 * 8)
        self.assertEqual(breakdown["b2"]["bytes"], 1 * 8)

    def test_calculate_model_parameter_bytes_float32(self):
        """Verify byte calculation respects float32 data types (57 * 4 = 228 bytes)."""
        f32_params = {k: v.astype(np.float32) for k, v in self.params.items()}
        size_info = calculate_model_parameter_bytes(f32_params)

        self.assertEqual(size_info["total_parameters"], 57)
        self.assertEqual(size_info["total_bytes"], 228)
        self.assertEqual(size_info["bytes_per_parameter"], 4)
        self.assertEqual(size_info["dtype"], "float32")

    def test_upload_and_download_bytes_per_client(self):
        """Verify client upload and download byte size matches total parameter bytes."""
        upload_bytes = calculate_client_upload_bytes(self.params)
        download_bytes = calculate_client_download_bytes(self.params)

        self.assertEqual(upload_bytes, 456)
        self.assertEqual(download_bytes, 456)

    def test_round_communication_volume_three_clients(self):
        """Verify round volume for 3 clients: Upload = 3*456, Download = 3*456, Total = 2736."""
        volume = calculate_round_communication_volume(self.params, num_clients=3)

        self.assertEqual(volume["num_clients"], 3)
        self.assertEqual(volume["upload_bytes_per_client"], 456)
        self.assertEqual(volume["download_bytes_per_client"], 456)
        self.assertEqual(volume["total_upload_bytes"], 1368)
        self.assertEqual(volume["total_download_bytes"], 1368)
        self.assertEqual(volume["total_communication_bytes"], 2736)
        self.assertAlmostEqual(volume["total_communication_kb"], 2736 / 1024.0, places=4)

    def test_cumulative_communication_volume_calculation(self):
        """Verify running cumulative communication totals accumulate monotonically."""
        mock_rounds = [
            {"round": 1, "upload_bytes": 1368, "download_bytes": 1368, "total_communication_bytes": 2736},
            {"round": 2, "upload_bytes": 1368, "download_bytes": 1368, "total_communication_bytes": 2736},
            {"round": 3, "upload_bytes": 1368, "download_bytes": 1368, "total_communication_bytes": 2736}
        ]

        cumulative_list = calculate_cumulative_communication(mock_rounds)

        self.assertEqual(len(cumulative_list), 3)
        self.assertEqual(cumulative_list[0]["cumulative_total_bytes"], 2736)
        self.assertEqual(cumulative_list[1]["cumulative_total_bytes"], 5472)
        self.assertEqual(cumulative_list[2]["cumulative_total_bytes"], 8208)
        self.assertAlmostEqual(cumulative_list[2]["cumulative_total_kb"], 8208 / 1024.0, places=4)

    def test_time_aggregation_and_summary_metrics(self):
        """Verify time aggregation and overall communication summary statistics."""
        mock_records = [
            {
                "round": 1,
                "upload_bytes": 1368,
                "download_bytes": 1368,
                "total_communication_bytes": 2736,
                "training_time_seconds": 0.05,
                "aggregation_time_seconds": 0.001,
                "round_duration_seconds": 0.06
            },
            {
                "round": 2,
                "upload_bytes": 1368,
                "download_bytes": 1368,
                "total_communication_bytes": 2736,
                "training_time_seconds": 0.04,
                "aggregation_time_seconds": 0.001,
                "round_duration_seconds": 0.05
            }
        ]

        summary = summarize_communication_efficiency(
            parameters=self.params,
            round_records=mock_records,
            num_clients=3,
            total_elapsed_seconds=0.12
        )

        self.assertEqual(summary["total_rounds"], 2)
        self.assertEqual(summary["num_clients"], 3)
        self.assertEqual(summary["parameter_count"], 57)
        self.assertEqual(summary["total_upload_bytes"], 2736)
        self.assertEqual(summary["total_download_bytes"], 2736)
        self.assertEqual(summary["total_communication_bytes"], 5472)
        self.assertEqual(summary["average_communication_bytes_per_round"], 2736.0)

        # Timing assertions
        self.assertAlmostEqual(summary["total_training_time_seconds"], 0.09, places=3)
        self.assertAlmostEqual(summary["total_aggregation_time_seconds"], 0.002, places=4)
        self.assertAlmostEqual(summary["total_experiment_duration_seconds"], 0.12, places=3)
        self.assertAlmostEqual(summary["average_round_duration_seconds"], 0.06, places=3)

        # Assumptions structure
        self.assertIn("communication_assumptions", summary)
        self.assertIn("transmission_mode", summary["communication_assumptions"])

    def test_invalid_inputs_raise_value_errors(self):
        """Verify defensive handling of invalid parameter and round inputs."""
        # Empty parameters dict
        with self.assertRaises(ValueError):
            count_model_parameters({})

        with self.assertRaises(ValueError):
            calculate_model_parameter_bytes({})

        # Non-dict parameter input
        with self.assertRaises(ValueError):
            count_model_parameters(["not_a_dict"])

        # Invalid client count
        with self.assertRaises(ValueError):
            calculate_round_communication_volume(self.params, num_clients=0)

        with self.assertRaises(ValueError):
            calculate_round_communication_volume(self.params, num_clients=-2)

        # Empty round records in summary
        with self.assertRaises(ValueError):
            summarize_communication_efficiency(self.params, round_records=[])

    def test_missing_metrics_handling_in_cumulative_communication(self):
        """Verify calculate_cumulative_communication handles empty list and missing keys gracefully."""
        self.assertEqual(calculate_cumulative_communication([]), [])

        incomplete_records = [{"round": 1}]
        result = calculate_cumulative_communication(incomplete_records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["cumulative_total_bytes"], 0)


if __name__ == "__main__":
    unittest.main()

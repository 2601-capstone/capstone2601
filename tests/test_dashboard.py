"""
Unit tests for Streamlit Dashboard Module (app.py).
"""

import unittest
import py_compile
import os


class TestStreamlitDashboard(unittest.TestCase):
    """Test suite verifying app.py compilation, dependencies, and structure."""

    def test_app_compilation(self):
        """Ensure app.py has completely valid Python syntax and compiles cleanly."""
        app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
        compiled = py_compile.compile(app_path)
        self.assertTrue(os.path.exists(compiled))

    def test_dashboard_file_contains_core_sections(self):
        """Verify app.py includes all required Phase 7 functional modules and pages."""
        app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check required function sections
        self.assertIn("render_overview", content)
        self.assertIn("render_federated_training", content)
        self.assertIn("render_model_comparison", content)
        self.assertIn("render_evaluation_details", content)
        self.assertIn("render_robustness_analysis", content)
        self.assertIn("render_communication_efficiency", content)
        self.assertIn("render_fedprox_comparison", content)
        self.assertIn("render_secure_aggregation", content)
        self.assertIn("render_vertical_federated", content)
        self.assertIn("render_federated_transfer", content)
        self.assertIn("render_hybrid_cnn_lstm", content)
        self.assertIn("render_leaf_fl", content)
        self.assertIn("render_experiment_history", content)

        # Check core integration hooks
        self.assertIn("run_federated_experiment", content)
        self.assertIn("run_full_evaluation", content)
        self.assertIn("run_robustness_analysis", content)
        self.assertIn("run_communication_analysis", content)
        self.assertIn("run_fedavg_vs_fedprox_comparison", content)
        self.assertIn("run_secagg_three_way_comparison", content)
        self.assertIn("FedAvg with Secure Aggregation", content)
        self.assertIn("run_vfl_experiment", content)
        self.assertIn("calculate_vfl_communication_volume", content)
        self.assertIn("build_vfl_experiment_record", content)
        self.assertIn("run_ftl_experiment", content)
        self.assertIn("calculate_ftl_communication_volume", content)
        self.assertIn("build_ftl_experiment_record", content)
        self.assertIn("run_cnn_lstm_baseline_comparison", content)
        self.assertIn("train_federated_cnn_lstm", content)
        self.assertIn("calculate_cnn_lstm_communication_volume", content)
        self.assertIn("build_cnn_lstm_experiment_record", content)
        self.assertIn("run_leaf_fl_experiment", content)
        self.assertIn("check_firestore_connection", content)
        self.assertIn("save_experiment", content)


if __name__ == "__main__":
    unittest.main()

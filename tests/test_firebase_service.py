"""
Unit tests for Firebase Firestore Service (src/firebase_service.py).
Uses mocks to verify cloud operations without requiring live credentials or external network calls.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.firebase_service import (
    sanitize_for_firestore,
    check_firestore_connection,
    save_experiment,
    get_experiment_history,
    get_experiment_by_id,
    build_experiment_record
)


class TestFirebaseService(unittest.TestCase):
    """Test suite for Firebase serialization, mock Firestore storage, and error handling."""

    def test_missing_credentials_graceful_degradation(self):
        """When credentials are absent, service must report False and not crash."""
        is_connected, msg = check_firestore_connection()
        # In a development environment without serviceAccountKey.json, it should be False
        self.assertIsInstance(is_connected, bool)
        self.assertIsInstance(msg, str)

        # Attempting save/query without credentials should return safe fallbacks
        success, save_msg = save_experiment({"test": 123}, client=None)
        if not is_connected:
            self.assertFalse(success)
            self.assertIn("Firestore save skipped", save_msg)

            history = get_experiment_history(limit=5, client=None)
            self.assertEqual(history, [])

            doc = get_experiment_by_id("fake_id", client=None)
            self.assertIsNone(doc)

    def test_sanitize_numpy_types_for_firestore(self):
        """Verify recursive conversion of all NumPy types, arrays, NaNs, and Infs."""
        payload = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14159),
            "bool_val": np.bool_(True),
            "array_1d": np.array([1, 2, 3]),
            "array_2d": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "nested": {
                "sub_array": np.array([5.5, 6.5]),
                "sub_int": np.int32(100)
            },
            "list_of_floats": [np.float32(0.5), np.float32(0.9)]
        }

        sanitized = sanitize_for_firestore(payload)

        # Verify python primitive types
        self.assertIs(type(sanitized["int_val"]), int)
        self.assertIs(type(sanitized["float_val"]), float)
        self.assertIs(type(sanitized["bool_val"]), bool)
        self.assertIs(type(sanitized["array_1d"]), list)
        self.assertEqual(sanitized["array_1d"], [1, 2, 3])
        self.assertIs(type(sanitized["array_2d"]), dict)
        self.assertEqual(sanitized["array_2d"], {"row_0": [1.0, 2.0], "row_1": [3.0, 4.0]})

        # NaN and Inf must be converted to None
        self.assertIsNone(sanitized["nan_val"])
        self.assertIsNone(sanitized["inf_val"])

        # Nested assertions
        self.assertIs(type(sanitized["nested"]["sub_int"]), int)
        self.assertIs(type(sanitized["nested"]["sub_array"]), list)
        self.assertEqual(sanitized["nested"]["sub_array"], [5.5, 6.5])
        self.assertIs(type(sanitized["list_of_floats"][0]), float)

    def test_save_experiment_with_mock_client(self):
        """Verify save_experiment writes sanitized document to collection('experiments')."""
        mock_client = MagicMock()
        mock_doc_ref = MagicMock()
        mock_client.collection.return_value.document.return_value = mock_doc_ref

        data = {
            "num_hospitals": 3,
            "accuracy": np.float64(0.85),
            "weights": np.array([0.1, 0.2])
        }

        success, exp_id = save_experiment(data, experiment_id="test_exp_001", client=mock_client)

        self.assertTrue(success)
        self.assertEqual(exp_id, "test_exp_001")
        mock_client.collection.assert_called_with("experiments")
        mock_client.collection.return_value.document.assert_called_with("test_exp_001")

        # Verify .set was called with sanitized dictionary
        mock_doc_ref.set.assert_called_once()
        saved_dict = mock_doc_ref.set.call_args[0][0]
        self.assertEqual(saved_dict["experiment_id"], "test_exp_001")
        self.assertIn("created_at", saved_dict)
        self.assertIs(type(saved_dict["accuracy"]), float)
        self.assertIs(type(saved_dict["weights"]), list)

    def test_get_experiment_history_with_mock_client(self):
        """Verify get_experiment_history queries and parses document stream."""
        mock_client = MagicMock()

        # Create 2 mock Firestore document snapshots
        doc1 = MagicMock()
        doc1.exists = True
        doc1.to_dict.return_value = {"experiment_id": "exp_1", "accuracy": 0.85}

        doc2 = MagicMock()
        doc2.exists = True
        doc2.to_dict.return_value = {"experiment_id": "exp_2", "accuracy": 0.88}

        mock_query = MagicMock()
        mock_query.stream.return_value = [doc1, doc2]
        mock_client.collection.return_value.order_by.return_value.limit.return_value = mock_query

        history = get_experiment_history(limit=2, client=mock_client)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["experiment_id"], "exp_1")
        self.assertEqual(history[1]["experiment_id"], "exp_2")

    def test_get_experiment_by_id_with_mock_client(self):
        """Verify get_experiment_by_id fetches single document."""
        mock_client = MagicMock()
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {"experiment_id": "exp_target", "status": "completed"}
        mock_client.collection.return_value.document.return_value.get.return_value = mock_doc

        result = get_experiment_by_id("exp_target", client=mock_client)
        self.assertIsNotNone(result)
        self.assertEqual(result["experiment_id"], "exp_target")

        # Non-existent doc
        mock_doc_missing = MagicMock()
        mock_doc_missing.exists = False
        mock_client.collection.return_value.document.return_value.get.return_value = mock_doc_missing
        self.assertIsNone(get_experiment_by_id("missing_id", client=mock_client))

    def test_graceful_handling_of_firestore_exceptions(self):
        """Client exceptions during save or read must be caught without raising."""
        mock_client = MagicMock()
        mock_client.collection.return_value.document.return_value.set.side_effect = RuntimeError("Connection timeout")

        success, err_msg = save_experiment({"test": 1}, client=mock_client)
        self.assertFalse(success)
        self.assertIn("Failed to save experiment to Firestore", err_msg)
        self.assertIn("Connection timeout", err_msg)

        # Exception during history stream
        mock_client.collection.side_effect = RuntimeError("Firestore unavailable")
        history = get_experiment_history(limit=5, client=mock_client)
        self.assertEqual(history, [])

    def test_build_experiment_record(self):
        """Verify build_experiment_record loads available metrics into a clean payload."""
        payload = build_experiment_record()
        expected_keys = {
            "num_hospitals",
            "num_communication_rounds",
            "model_architecture",
            "aggregation_algorithm",
            "model_config",
            "training_history",
            "model_comparison",
            "evaluation_summary",
            "dataset_description"
        }
        self.assertTrue(expected_keys.issubset(set(payload.keys())))
        self.assertEqual(payload["num_hospitals"], 3)
        self.assertIsInstance(payload["training_history"], list)
        self.assertIsInstance(payload["model_comparison"], list)


if __name__ == "__main__":
    unittest.main()

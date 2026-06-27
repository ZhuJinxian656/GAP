import os
import sys
import unittest

import numpy as np

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)

from gap_policy.triadic import TriadicConfig, build_triadic_state, infer_triadic_dim


class TestTriadicState(unittest.TestCase):
    def test_pairwise_shape_and_values(self):
        obs = {
            "/observation/left_eef_pos": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "/observation/right_eef_pos": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            "/observation/object_pos": np.array([[0.5, 1.0, 1.5]], dtype=np.float32),
            "/observation/left_gripper": np.array([[0.25]], dtype=np.float32),
            "/observation/right_gripper": np.array([[0.75]], dtype=np.float32),
        }
        state = build_triadic_state(obs, TriadicConfig(mode="pairwise"))
        self.assertEqual(state.shape, (1, infer_triadic_dim("pairwise")))
        np.testing.assert_allclose(
            state[0],
            np.array([0.5, 1.0, 1.5, 3.5, 4.0, 4.5, 0.25, 0.75], dtype=np.float32),
        )

    def test_triadic_shape(self):
        obs = {
            "observation": {
                "left_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "right_eef_pos": np.array([4.0, 5.0, 6.0], dtype=np.float32),
                "object_pos": np.array([0.5, 1.0, 1.5], dtype=np.float32),
            },
            "agent_pos": np.arange(14, dtype=np.float32),
        }
        state = build_triadic_state(obs, TriadicConfig(mode="triadic"))
        self.assertEqual(state.shape, (infer_triadic_dim("triadic"),))

    def test_missing_object_returns_none(self):
        obs = {
            "/observation/left_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "/observation/right_eef_pos": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        }
        messages = []
        state = build_triadic_state(obs, TriadicConfig(mode="triadic"), warn_fn=messages.append)
        self.assertIsNone(state)
        self.assertTrue(any("object" in message.lower() for message in messages))

    def test_proprio_only_fallback(self):
        obs = {"agent_pos": np.arange(14, dtype=np.float32)}
        messages = []
        state = build_triadic_state(
            obs,
            TriadicConfig(mode="proprio_only_fallback"),
            warn_fn=messages.append,
        )
        self.assertEqual(state.shape, (infer_triadic_dim("proprio_only_fallback"),))
        np.testing.assert_allclose(state[:3], np.array([-7.0, -7.0, -7.0], dtype=np.float32))
        np.testing.assert_allclose(state[3:], np.array([6.0, 13.0], dtype=np.float32))
        self.assertTrue(any("fallback" in message for message in messages))


if __name__ == "__main__":
    unittest.main()

"""Transport content checks without OpenPI, a model, or dataset dependencies."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from checks import check_policy_transport as transport


class PolicyTransportTests(unittest.TestCase):
    def setUp(self):
        self.request = {"observation/image": np.full((224, 224, 3), 17, np.uint8),
                        "observation/wrist_image": np.full((224, 224, 3), 91, np.uint8),
                        "observation/state": np.arange(8, dtype=np.float32),
                        "prompt": "Put the cup on the plate. 放好杯子。"}
        server = mock.Mock()
        modules = {"openpi_client.base_policy": SimpleNamespace(BasePolicy=object),
                   "openpi.serving.websocket_policy_server": SimpleNamespace(WebsocketPolicyServer=server)}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, modules):
            metadata = Path(directory) / "metadata.json"
            metadata.write_text("{}")
            transport.serve(8000, metadata)
        self.policy = server.call_args.args[0]
        server.return_value.serve_forever.assert_called_once()

    def client(self, received=None):
        return SimpleNamespace(infer=lambda request: self.policy.infer(
            copy.deepcopy(request) if received is None else received))

    def test_exact_round_trip_preserves_both_cameras_state_and_unicode_prompt(self):
        transport.check_round_trip(self.client(), self.request)

    def test_changed_camera_pixels_and_swapped_cameras_are_rejected(self):
        for key in ("observation/image", "observation/wrist_image"):
            with self.subTest(key=key):
                received = copy.deepcopy(self.request)
                received[key][0, 0, 0] += 1
                with self.assertRaisesRegex(ValueError, f"content changed.*{key}"):
                    transport.check_round_trip(self.client(received), self.request)
        received = copy.deepcopy(self.request)
        received["observation/image"], received["observation/wrist_image"] = (
            received["observation/wrist_image"], received["observation/image"])
        with self.assertRaisesRegex(ValueError, "content changed"):
            transport.check_round_trip(self.client(received), self.request)

    def test_cleared_images_prompt_and_last_state_element_are_rejected(self):
        for key in self.request:
            with self.subTest(key=key):
                received = copy.deepcopy(self.request)
                if key == "prompt":
                    received[key] = "Different task"
                elif key == "observation/state":
                    received[key][-1] += 1  # This element is absent from echoed actions.
                else:
                    received[key].fill(0)
                with self.assertRaisesRegex(ValueError, f"content changed.*{key}"):
                    transport.check_round_trip(self.client(received), self.request)

    def test_dtype_change_with_identical_bytes_is_rejected(self):
        received = copy.deepcopy(self.request)
        received["observation/state"] = received["observation/state"].view(np.int32)
        with self.assertRaisesRegex(ValueError, "content changed.*observation/state"):
            transport.check_round_trip(self.client(received), self.request)

    def test_invalid_shapes_and_prompt_type_fail_explicitly(self):
        for key in self.request:
            with self.subTest(key=key):
                received = copy.deepcopy(self.request)
                received[key] = 17 if key == "prompt" else received[key].reshape(-1, 1)
                with self.assertRaisesRegex(ValueError, key):
                    self.policy.infer(received)

    def test_missing_fingerprint_and_changed_response_actions_are_rejected(self):
        response = self.policy.infer(self.request)
        del response["observation_fingerprints"]
        with self.assertRaisesRegex(ValueError, "Missing or invalid observation fingerprints"):
            transport.check_round_trip(SimpleNamespace(infer=lambda request: response), self.request)
        response = self.policy.infer(self.request)
        response["actions"][0, 0] += 1
        with self.assertRaises(AssertionError):
            transport.check_round_trip(SimpleNamespace(infer=lambda request: response), self.request)

    def test_missing_field_invalid_and_extra_fingerprints_are_rejected(self):
        for key in self.request:
            with self.subTest(key=key):
                response = self.policy.infer(self.request)
                del response["observation_fingerprints"][key]
                with self.assertRaisesRegex(ValueError, f"content changed.*{key}"):
                    transport.check_round_trip(SimpleNamespace(infer=lambda request: response), self.request)
        response = self.policy.infer(self.request)
        response["observation_fingerprints"] = []
        with self.assertRaisesRegex(ValueError, "Missing or invalid observation fingerprints"):
            transport.check_round_trip(SimpleNamespace(infer=lambda request: response), self.request)
        response = self.policy.infer(self.request)
        response["observation_fingerprints"]["extra"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "Unexpected observation fingerprints"):
            transport.check_round_trip(SimpleNamespace(infer=lambda request: response), self.request)


if __name__ == "__main__":
    unittest.main()

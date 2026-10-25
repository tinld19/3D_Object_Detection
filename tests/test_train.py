import sys
import types
import unittest

import train


class TrainEntrypointTest(unittest.TestCase):
    def test_enables_early_stopping_and_rich_progress(self):
        calls = []

        class FakeModel:
            def train(self, **kwargs):
                calls.append(kwargs)

        previous_argv = sys.argv
        previous_rfdetr = sys.modules.get("rfdetr")
        sys.argv = ["train.py", "--device", "cuda"]
        sys.modules["rfdetr"] = types.SimpleNamespace(RFDETRNano=FakeModel)
        try:
            train.main()
        finally:
            sys.argv = previous_argv
            if previous_rfdetr is None:
                del sys.modules["rfdetr"]
            else:
                sys.modules["rfdetr"] = previous_rfdetr

        self.assertEqual(15, calls[0]["early_stopping_patience"])
        self.assertTrue(calls[0]["early_stopping"])
        self.assertEqual("rich", calls[0]["progress_bar"])
        self.assertEqual(384, calls[0]["resolution"])


if __name__ == "__main__":
    unittest.main()

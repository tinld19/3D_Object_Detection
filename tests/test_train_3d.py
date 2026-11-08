import unittest

import torch

from train_3d import set_seed


class Train3DEntrypointTest(unittest.TestCase):
    def test_set_seed_repeats_torch_initialization(self):
        set_seed(42)
        first = torch.rand(4)
        set_seed(42)

        self.assertTrue(torch.equal(first, torch.rand(4)))


if __name__ == "__main__":
    unittest.main()

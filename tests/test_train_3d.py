import unittest

import torch

from train_3d import set_seed, split_indices


class Train3DEntrypointTest(unittest.TestCase):
    def test_set_seed_repeats_torch_initialization(self):
        set_seed(42)
        first = torch.rand(4)
        set_seed(42)

        self.assertTrue(torch.equal(first, torch.rand(4)))

    def test_split_indices_partitions_by_val_ids(self):
        stems = ["000000", "000001", "000002", "000003"]

        self.assertEqual(split_indices(stems, {"000001", "000003"}), ([0, 2], [1, 3]))
        self.assertEqual(split_indices(stems + ["000009"], {"000001"}), ([0, 2, 3, 4], [1]))


if __name__ == "__main__":
    unittest.main()

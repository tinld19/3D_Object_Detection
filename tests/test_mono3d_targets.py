import math
import unittest

from monocular_3d import depth_bin
import mono3d_targets as t

P2 = [721.5377, 0.0, 609.5593, 44.85728, 0.0, 721.5377, 172.854, 0.2163791, 0.0, 0.0, 1.0, 0.002745884]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mono3DTargetsTest(unittest.TestCase):
    def test_bin_value_round_trips_depth_bin(self):
        for i in (0, 1, 40, 79):
            self.assertEqual(depth_bin(t.bin_value(i) + 1e-6), i)

    def test_bin_value_mid_index_hand_value(self):
        delta = 2 * 58.0 / (80 * 81)
        self.assertAlmostEqual(t.bin_value(40), 2.0 + delta * 40 * 41 / 2)
        self.assertAlmostEqual(t.bin_value(0), 2.0)

    def test_depth_outside_range_has_no_bin(self):
        self.assertIsNone(depth_bin(1.9))
        self.assertIsNone(depth_bin(60.1))

    def test_rescale_p2_independent_axes(self):
        out = t.rescale_p2(P2, (1242, 375), (384, 384))
        sx, sy = 384 / 1242, 384 / 375
        self.assertEqual(len(out), 12)
        self.assertAlmostEqual(out[0], P2[0] * sx)
        self.assertAlmostEqual(out[2], P2[2] * sx)
        self.assertAlmostEqual(out[5], P2[5] * sy)
        self.assertAlmostEqual(out[6], P2[6] * sy)

    def test_geometric_depth(self):
        self.assertAlmostEqual(t.geometric_depth(700.0, 1.5, 50.0), 21.0)
        self.assertTrue(math.isfinite(t.geometric_depth(700.0, 1.5, 0.0)))

    def test_alpha_ry_round_trip(self):
        for ry in [-math.pi, -2.0, -1.0, -0.3, 0.0, 0.7, 2.5, math.pi - 1e-6]:
            alpha = t.alpha_from_ry(ry, 3.0, 20.0)
            self.assertTrue(-math.pi <= alpha < math.pi)
            back = t.ry_from_alpha(alpha, 3.0, 20.0)
            self.assertAlmostEqual(_wrap(back - ry), 0.0, places=9)

    def test_ray_angle(self):
        self.assertAlmostEqual(t.ray_angle(1.0, 1.0), math.pi / 4)

    def test_multibin_round_trip(self):
        for alpha in [-math.pi, -1.3, 0.0, 0.4, 2.9, math.pi - 1e-6]:
            b, r = t.multibin_target(alpha)
            self.assertTrue(0 <= b < 12)
            self.assertLessEqual(abs(r), math.pi / 12 + 1e-9)
            dec = t.multibin_decode(b, r)
            self.assertTrue(-math.pi <= dec < math.pi)
            self.assertAlmostEqual(_wrap(dec - alpha), 0.0, places=9)

    def test_project_backproject_round_trip(self):
        x, y, z = 2.3, 1.1, 15.0
        u, v = t.project_center(x, y, z, P2)
        bx, by = t.backproject_center(u, v, z, P2)
        self.assertAlmostEqual(bx, x, places=6)
        self.assertAlmostEqual(by, y, places=6)


if __name__ == "__main__":
    unittest.main()

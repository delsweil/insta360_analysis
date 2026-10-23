"""Tests for ground_model.py: box height and foot position combined."""
import math
import unittest

import numpy as np

import ground_model as GM
import scene_check as S

FY = S.SQUARE_FY
H = 8.0                                     # camera height of the synthetic world, metres


def world(n=2000, seed=1, noise=0.06, d_lo=12.0, d_hi=75.0):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-50, 50, n)
    d = rng.uniform(d_lo, d_hi, n)
    lat = -np.degrees(np.arctan2(H, d))
    cen = np.array([min([-40.0, 0.0, 40.0], key=lambda v: abs(v - l)) for l in lon])
    off = np.radians(0.75 * (lon - cen))
    h_true = 1.7 * FY / (0.75 * d * np.cos(off))
    return lon, lat, h_true * rng.normal(1, noise, n), cen, d, h_true


class TestGroundModel(unittest.TestCase):
    def test_fit_finds_the_camera(self):
        lon, lat, h, cen, d, _ = world()
        c, delta, res, n = GM.fit(lon, lat, h, cen)
        self.assertAlmostEqual(c, 1.7 * FY / (0.75 * H), delta=0.04 * c)
        self.assertGreater(n, 500)

    def test_not_enough_players_means_no_fit(self):
        lon, lat, h, cen, *_ = world(n=100)
        self.assertIsNone(GM.fit(lon, lat, h, cen))

    def test_combined_distance_is_closer_to_the_truth_for_far_players(self):
        lon, lat, h, cen, d, h_true = world(seed=2, noise=0.08, d_lo=45.0, d_hi=75.0)
        c, delta, *_ = GM.fit(*world(seed=3)[:4])
        used, state = GM.fuse(lon, lat, h, cen, c, delta)
        err_box = np.median(np.abs(np.log(h / h_true)))
        err_fused = np.median(np.abs(np.log(used / h_true)))
        self.assertLess(err_fused, 0.6 * err_box)
        self.assertGreater(state.mean(), 0.95)

    def test_near_players_are_not_made_worse(self):
        lon, lat, h, cen, d, h_true = world(seed=4, d_lo=10.0, d_hi=14.0)
        c, delta, *_ = GM.fit(*world(seed=3)[:4])
        used, _ = GM.fuse(lon, lat, h, cen, c, delta)
        self.assertLessEqual(np.median(np.abs(np.log(used / h_true))), np.median(np.abs(np.log(h / h_true))))

    def test_hidden_legs_keep_the_box_height(self):
        c, delta, *_ = GM.fit(*world(seed=3)[:4])
        d = 40.0
        lat = -math.degrees(math.atan2(H, d)) + 4.0                      # feet seen 4 degrees too high (legs behind a wall)
        h = 1.7 * FY / (0.75 * d)
        used, state = GM.fuse([0.0], [lat], [h], [0.0], c, delta)
        self.assertEqual(int(state[0]), 0)
        self.assertAlmostEqual(float(used[0]), h, places=6)

    def test_near_the_horizon_is_not_used(self):
        c, delta, *_ = GM.fit(*world(seed=3)[:4])
        used, state = GM.fuse([0.0], [-1.5], [20.0], [0.0], c, delta)
        self.assertEqual(int(state[0]), 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for board_smooth.py: synthetic walks with noise, no footage."""
import unittest

import numpy as np

import board_clean as BC
import board_smooth as BS


def walk(n=60, dt=1 / 3, vx=3.0, vy=1.0, y0=40.0, sig=0.8, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    tx, ty = 10 + vx * t, y0 + vy * t
    return t, tx, ty, tx + rng.normal(0, sig, n), ty + rng.normal(0, sig, n)


class TestSmooth(unittest.TestCase):
    def test_noise_goes_down(self):
        t, tx, ty, x, y = walk()
        frame = np.arange(len(t))
        tr = BC.link_tracks(frame, x, y, t)
        nx, ny, st = BS.smooth(frame, t, x, y, tr)
        before = np.hypot(x - tx, y - ty).mean()
        after = np.hypot(nx - tx, ny - ty).mean()
        self.assertLess(after, 0.7 * before)
        self.assertEqual(st["smoothed"], len(t))

    def test_a_standing_player_stops_jittering(self):
        t, tx, ty, x, y = walk(vx=0, vy=0)
        frame = np.arange(len(t))
        nx, ny, _ = BS.smooth(frame, t, x, y, np.zeros(len(t), int))
        self.assertLess(nx.std(), 0.5 * x.std())

    def test_a_sprint_is_not_flattened(self):
        t, tx, ty, x, y = walk(vx=7.0, vy=0.0, sig=0.3)
        frame = np.arange(len(t))
        nx, _, _ = BS.smooth(frame, t, x, y, np.zeros(len(t), int))
        self.assertLess(np.abs(nx - tx)[3:-3].max(), 0.6)           # follows a 7 m/s run without lag

    def test_one_wild_detection_is_gated(self):
        t, tx, ty, x, y = walk(sig=0.2)
        x = x.copy(); x[30] += 25.0
        frame = np.arange(len(t))
        nx, _, st = BS.smooth(frame, t, x, y, np.zeros(len(t), int))
        self.assertLess(abs(nx[30] - tx[30]), 3.0)
        self.assertGreaterEqual(st["gated"], 1)

    def test_short_tracks_and_unlinked_are_left_alone(self):
        frame = np.array([0, 1, 2, 3])
        vt = np.arange(4) / 3
        x = np.array([1.0, 2.0, 3.0, 4.0]); y = np.full(4, 30.0)
        nx, ny, st = BS.smooth(frame, vt, x, y, np.array([0, 0, -1, -1]))
        self.assertTrue(np.array_equal(nx, x) and np.array_equal(ny, y))
        self.assertEqual(st["smoothed"], 0)

    def test_nothing_is_added_or_dropped(self):
        t, tx, ty, x, y = walk(n=10)
        nx, ny, _ = BS.smooth(np.arange(10), t, x, y, np.zeros(10, int))
        self.assertEqual((len(nx), len(ny)), (10, 10))
        self.assertTrue(np.isfinite(nx).all() and np.isfinite(ny).all())


if __name__ == "__main__":
    unittest.main()

"""Tests for ball_track.py and the ball-view rule of auto_keyframes.py (python3 test_ball_track.py)."""
import sys
import unittest

import numpy as np

import auto_keyframes as AK
import ball_track as BT


def frame(t, players, balls):
    return {"t": t, "players": [[x, -5.0, 30.0, 0.8, True] for x in players], "balls": balls}


class TrackTests(unittest.TestCase):
    def clip(self, n=100):
        fr = []
        for i in range(n):
            x = 10 + i * 1.5                                     # a ball rolling right with 3 players around it
            balls = [[x, -6.0, 0.7], [-70.0, -8.0, 0.6]]         # + a fixed false spot in every frame
            if i == 50:
                balls.append([-30.0, -6.0, 0.9])                 # lone high-confidence false hit, nobody near
            fr.append(frame(i / 3, [x - 3, x + 2, x + 6, -40], balls))
        return fr

    def test_follows_ball_and_ignores_fixed_spot_and_lone_hit(self):
        fr = self.clip()
        out = BT.track_ball(fr)
        self.assertGreater(np.mean(~np.isnan(out)), 0.95)
        self.assertLess(np.nanmax(np.abs(out - np.array([10 + i * 1.5 for i in range(len(fr))]))), 0.01)

    def test_short_false_stretch_dropped(self):
        fr = [frame(i / 3, [50, 52, 54], []) for i in range(30)]
        fr[10]["balls"] = [[51.0, -6.0, 0.8]]
        fr[11]["balls"] = [[51.5, -6.0, 0.8]]
        self.assertTrue(np.isnan(BT.track_ball(fr)).all())

    def test_no_balls_gives_nan(self):
        self.assertTrue(np.isnan(BT.track_ball([frame(0, [1, 2], [])])).all())

    def test_hold(self):
        x = np.array([1, np.nan, np.nan, np.nan, np.nan, 2.0])
        h = BT.hold(x, 2)
        self.assertEqual(list(h[:3]), [1, 1, 1])
        self.assertTrue(np.isnan(h[3]) and np.isnan(h[4]))


class ViewTests(unittest.TestCase):
    N = 400

    def frames(self, with_ball):
        # players bunched near 0; the ball (two players next to it) sweeps out to 100 degrees
        fr = []
        for i in range(self.N):
            b = 20 + 80 * i / self.N
            pl = [-10, -5, 0, 5, 10, 14] + ([b - 2, b + 3] if with_ball else [])
            fr.append(frame(i / 3, pl, [[b, -6.0, 0.8]] if with_ball else []))
        return fr

    def run_(self, with_ball, **kw):
        return AK.build_path({"fps": 3.0}, self.frames(with_ball), fov_fixed=90.0, fov_min=70, fov_max=100, **kw)

    def test_ball_far_out_is_kept_in_crop(self):
        i = 330                                                           # ball at 86 degrees
        ball = 20 + 80 * i / self.N
        off = self.run_(True)
        on = self.run_(True, ball_view=True, ball_margin=8.0, ball_edge_extra=0.0)
        self.assertGreaterEqual(-on["pan"][i] + 45.0 - 8.0, ball - 1.0)   # inside, ~8 degrees from the edge
        self.assertLess(-off["pan"][i] + 45.0 - 8.0, ball - 5.0)          # and it was NOT inside before

    def test_swing_is_limited_and_ball_stays_in(self):
        fr = []
        for i in range(300):                                     # the ball switches from far left to far right
            b = -70.0 + 2.0 * i if i < 70 else 70.0 - 0.1 * (i - 70)
            fr.append(frame(i / 3, [-10, 0, 10, b - 2, b + 3], [[b, -6.0, 0.8]]))
        r = AK.build_path({"fps": 3.0}, fr, fov_fixed=90.0, fov_min=70, fov_max=100, ball_view=True, ball_max_speed=30.0)
        self.assertLessEqual(np.abs(np.diff(r["pan"])).max() * 3.0, 30.0 + 1e-6)

    def test_zoom_in_when_ball_at_the_end(self):
        fr = []
        for i in range(400):                                    # the ball sweeps out to 96 degrees
            b = 20 + 76 * i / 399.0
            fr.append(frame(i / 3, [-10, 0, 10, b - 2, b + 3], [[b, -6.0, 0.8]]))
        kw = dict(fov_a=0.0, fov_b=100.0, fov_min=70, fov_max=100, projection="flat", base_tilt=-20.0, ball_view=True)
        z = AK.build_path({"fps": 3.0, "iv_fov": 112.0}, fr, **kw)
        n = AK.build_path({"fps": 3.0, "iv_fov": 112.0}, fr, ball_zoom=False, **kw)
        self.assertLess(z["fov"][385], 95.0)
        self.assertGreaterEqual(n["fov"][385], 99.9)

    def test_no_ball_changes_nothing(self):
        a = self.run_(False)
        b = self.run_(False, ball_view=True)
        self.assertTrue(np.allclose(a["pan"], b["pan"]))


if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    unittest.main(verbosity=1)

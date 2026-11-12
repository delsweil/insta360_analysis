import unittest
from unittest import mock

import numpy as np

import preview_people as PP


def player(lon, lat, h=60.0, conf=0.9):
    return {"lon": lon, "lat": lat, "h": h, "conf": conf}


class TestPreviewPeople(unittest.TestCase):
    def test_times_are_spread_inside_the_video(self):
        ts = PP.sample_times(100.0, 1000.0, 12)
        self.assertEqual(len(ts), 12)
        self.assertAlmostEqual(ts[0], 200.0)
        self.assertAlmostEqual(ts[-1], 1000.0)
        self.assertEqual(ts, sorted(ts))

    def test_one_picture_is_taken_in_the_middle(self):
        self.assertEqual(PP.sample_times(0.0, 100.0, 1), [50.0])

    def test_only_sure_and_big_enough_people_count(self):
        out = PP.feet_pixels([player(0, -10), player(10, -10, conf=0.3), player(20, -10, h=5)])
        self.assertEqual(len(out), 1)
        # the middle of the picture: lon 0 is half the width, lat -10 is below the middle
        self.assertAlmostEqual(out[0][0], 1439.5, delta=1)
        self.assertGreater(out[0][1], 720)

    def test_run_collects_feet_and_reports_progress(self):
        seen = []
        frames = {"n": 0}

        def grab(t):
            frames["n"] += 1
            return None if frames["n"] == 2 else np.zeros((4, 4, 3), np.uint8)       # one picture could not be read
        with mock.patch.object(PP.S, "detect_scene", return_value=[player(0, -10), player(30, -12)]):
            res = PP.run("x.insv", 0.0, 100.0, 4, True, model=None, device="cpu", grab=grab,
                         progress=lambda k, n: seen.append((k, n)))
        self.assertEqual(len(res["times"]), 3)
        self.assertEqual(len(res["feet"]), 6)
        self.assertEqual(seen[-1], (4, 4))


if __name__ == "__main__":
    unittest.main()

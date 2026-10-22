"""Tests for board_data.py: player positions in the time of the joined video, teams, camera."""
import json
import os
import tempfile
import unittest

import numpy as np

import board_data as BD
import kf_core as K


class FakeClusters:
    names = ["Altenberg", "ASN"]
    colours = ["#ececec", "#d9392b"]

    def team_of(self, bgr):                      # bright shirts = team 0, red-ish = team 1, dark = nobody
        bgr = np.asarray(bgr, float)
        return np.where(bgr.mean(1) > 180, 0, np.where(bgr[:, 2] > 150, 1, -1))


def write_dets(path, start, times, extra=None):
    with open(path, "w") as fh:
        fh.write(json.dumps({"type": "header", "view_lons": [-40, 0, 40], "start": start, "fps": 3.0,
                             "square_views": True, "colours": True}) + "\n")
        for t in times:
            players = [[0.0, -10.0, 60.0, 0.9, None, [230, 230, 230]],       # white shirt
                       [10.0, -10.0, 60.0, 0.9, None, [30, 30, 220]],        # red shirt
                       [-5.0, -10.0, 60.0, 0.9, None, [20, 20, 20]],         # dark: referee
                       [5.0, -10.0, 60.0, 0.2, None, [230, 230, 230]],       # not confident: dropped
                       [20.0, -10.0, 8.0, 0.9, None, [230, 230, 230]],       # too small for a trusted colour
                       [110.0, -10.0, 60.0, 0.9, None, [230, 230, 230]]]     # outside the lens: dropped
            if extra:
                players += extra
            fh.write(json.dumps({"t": t, "players": players, "balls": []}) + "\n")


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def p(self, name):
        return os.path.join(self.d, name)

    def test_times_are_those_of_the_joined_video(self):
        a, b = self.p("a.jsonl"), self.p("b.jsonl")
        write_dets(a, 10.0, [10.0, 10.333, 10.667, 11.0])
        write_dets(b, 500.0, [500.0, 500.333])
        data = BD.build([{"det": a, "half": 1, "video_start": 0.0, "duration": 1.2, "box": [-50, 50, 12, 80], "a_attacks_right": True},
                         {"det": b, "half": 2, "video_start": 1.2, "duration": 1.0, "box": [-50, 50, 12, 80], "a_attacks_right": False}],
                        FakeClusters(), "Game", "en")
        ts = [f[0] for f in data["frames"]]
        self.assertEqual(ts, [0.0, 0.33, 0.67, 1.0, 1.2, 1.53])       # first part from 0 s, the second after it
        self.assertEqual([p["half"] for p in data["parts"]], [1, 2])
        self.assertEqual(data["parts"][1]["video_start"], 1.2)
        self.assertEqual(data["parts"][1]["a_attacks_right"], False)
        self.assertEqual(data["teams"]["B"], {"name": "ASN", "colour": "#d9392b"})

    def test_players_teams_and_units(self):
        a = self.p("a.jsonl")
        write_dets(a, 0.0, [0.0])
        data = BD.build([{"det": a, "half": 1, "video_start": 0.0, "duration": 5.0}], FakeClusters())
        f = data["frames"][0]
        self.assertEqual((len(f) - 3) % 3, 0)
        trip = [tuple(f[3 + 3 * i:6 + 3 * i]) for i in range((len(f) - 3) // 3)]
        self.assertEqual(len(trip), 4)                                # confident + inside the lens only
        self.assertEqual(sorted(t[2] for t in trip), [0, 1, 2, 2])    # white, red, dark referee, small box = unknown
        white = [t for t in trip if t[2] == 0][0]
        self.assertAlmostEqual(abs(white[0]) / 10.0, 0.0, delta=0.1)  # straight ahead: x ~ 0, in decimetres
        self.assertGreater(white[1] / 10.0, 5.0)                      # some metres from the camera
        self.assertIsNone(f[1])                                       # no camera path given

    def test_camera_and_trim(self):
        a, kf = self.p("a.jsonl"), self.p("a.keyframes.json")
        write_dets(a, 0.0, [2.0, 3.0, 4.0])
        json.dump([{"t": 0.0, "pan_deg": 0.0, "fov_deg": 80.0}, {"t": 10.0, "pan_deg": 20.0, "fov_deg": 60.0}], open(kf, "w"))
        K.save_meta(kf, tilt_deg=-20.0, interp="pchip", trim_in=2.0, trim_out=4.0)   # the video starts at 2 s
        data = BD.build([{"det": a, "kf": kf, "half": 1, "video_start": 100.0, "duration": 2.0}], FakeClusters())
        self.assertEqual([f[0] for f in data["frames"]], [100.0, 101.0, 102.0])
        self.assertIsNotNone(data["frames"][0][1])
        self.assertTrue(55 < data["frames"][0][2] < 85)               # zoom (fov) read from the camera path
        self.assertNotEqual(data["frames"][0][1], data["frames"][2][1])

    def test_cli_and_compact_file(self):
        a = self.p("a.jsonl")
        write_dets(a, 0.0, [i / 3.0 for i in range(30)])
        labels = self.p("labels.json")
        json.dump({}, open(labels, "w"))
        cd = self.p("g_half1_chartdata.json")
        json.dump({"pitch_box_m": [-45, 45, 15, 70], "team_a_attacks_right": True}, open(cd, "w"))
        out = self.p("g_board.json")
        import team_labels as TL
        orig = TL.Labelled
        TL.Labelled = lambda path: FakeClusters()
        try:
            self.assertEqual(BD.main(["--detections", a, "--halves", "1", "--durations", "10", "--team-labels", labels,
                                      "--chart-data", cd, "--out", out]), 0)
        finally:
            TL.Labelled = orig
        data = json.load(open(out))
        self.assertEqual(data["format"], BD.FORMAT)
        self.assertEqual(data["parts"][0]["box"], [-45.0, 45.0, 15.0, 70.0])
        self.assertEqual(len(data["frames"]), 30)
        self.assertNotIn(" ", open(out).read().split('"note"')[0])    # compact separators
        self.assertLess(os.path.getsize(out), 12000)

    def test_no_players_is_an_error(self):
        a = self.p("a.jsonl")
        with open(a, "w") as fh:
            fh.write(json.dumps({"type": "header", "view_lons": [0], "start": 0.0, "fps": 3.0, "colours": True}) + "\n")
        with self.assertRaises(SystemExit):
            BD.build([{"det": a, "half": 1, "video_start": 0.0, "duration": 5.0}], FakeClusters())


if __name__ == "__main__":
    unittest.main()

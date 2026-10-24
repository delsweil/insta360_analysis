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


class RoleClusters(FakeClusters):
    """white = A, red = B, dark = referee, [10,200,10] (green) = goalkeeper group labelled as B's, [200,10,200] = A's."""
    has_roles = True

    def role_of(self, bgr):
        bgr = np.asarray(bgr, float)
        out = np.full(len(bgr), -1)
        out[bgr.mean(1) < 40] = 3
        out[(bgr[:, 1] > 150) & (bgr[:, 0] < 50)] = 5
        out[(bgr[:, 0] > 150) & (bgr[:, 2] > 150) & (bgr[:, 1] < 50)] = 4
        return out


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

    def test_roles_referee_and_goalkeepers_by_goal(self):
        a = self.p("a.jsonl")
        # two keepers wearing the colour of B's keeper: one stands far left, one far right
        extra = [[-60.0, -10.0, 60.0, 0.9, None, [10, 200, 10]], [60.0, -10.0, 60.0, 0.9, None, [10, 200, 10]]]
        write_dets(a, 0.0, [0.0], extra=extra)
        box = [-50, 50, 12, 80]
        left_is_a = BD.build([{"det": a, "half": 1, "video_start": 0.0, "duration": 5.0, "box": box, "a_attacks_right": True}],
                             RoleClusters())["frames"][0]
        trip = [tuple(left_is_a[3 + 3 * i:6 + 3 * i]) for i in range((len(left_is_a) - 3) // 3)]
        codes = sorted(t[2] for t in trip)
        self.assertIn(3, codes)                                       # the dark one is the referee now, not 'other'
        self.assertEqual(codes.count(4), 1)
        self.assertEqual(codes.count(5), 1)
        gka = [t for t in trip if t[2] == 4][0]
        self.assertLess(gka[0], 0)                                    # A attacks right: A's keeper is at the left goal
        flipped = BD.build([{"det": a, "half": 2, "video_start": 0.0, "duration": 5.0, "box": box, "a_attacks_right": False}],
                           RoleClusters())["frames"][0]
        trip2 = [tuple(flipped[3 + 3 * i:6 + 3 * i]) for i in range((len(flipped) - 3) // 3)]
        self.assertGreater([t for t in trip2 if t[2] == 4][0][0], 0)  # second half: sides swapped

    def test_without_roles_nothing_changes(self):
        a = self.p("a.jsonl")
        write_dets(a, 0.0, [0.0])
        f = BD.build([{"det": a, "half": 1, "video_start": 0.0, "duration": 5.0}], FakeClusters())["frames"][0]
        self.assertNotIn(3, [f[5 + 3 * i] for i in range((len(f) - 3) // 3)])

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

    def test_looser_board_thresholds_do_not_change_the_detections_file(self):
        a = self.p("a.jsonl")
        write_dets(a, 0.0, [0.0], extra=[[-20.0, -10.0, 60.0, 0.35, None, [230, 230, 230]]])   # found at 0.35: shown now
        before = open(a).read()
        data = BD.build([{"det": a, "half": 1, "video_start": 0.0, "duration": 5.0}], FakeClusters())
        self.assertEqual(len(data["frames"][0]) - 3, 5 * 3)           # 4 as before + the 0.35 one
        strict = BD.build([{"det": a, "half": 1, "video_start": 0.0, "duration": 5.0}], FakeClusters(), min_conf=0.5)
        self.assertEqual(len(strict["frames"][0]) - 3, 4 * 3)
        self.assertEqual(open(a).read(), before)                       # the camera planning reads this same file

    def test_board_check_reports_bands(self):
        import io
        import contextlib
        import board_check as BC
        a = self.p("a.jsonl")
        write_dets(a, 0.0, [i / 3 for i in range(9)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(BC.main(["--detections", a]), 0)
        out = buf.getvalue()
        self.assertIn("near (< 30 m)", out)
        self.assertIn("far (> 55 m)", out)
        self.assertIn(">= 0.3", out)

    def test_rescue_gives_washed_out_shirts_a_team_but_not_the_referee(self):
        import scene_check as S
        import team_labels as TL
        labels = self.p("l.json")
        groups = np.array([[235, 235, 235], [30, 30, 220], [20, 20, 20], [60, 150, 80]], float)   # white, red, dark, grass-green
        json.dump({"centres": S.colour_features(groups).tolist(), "sizes": [50, 40, 5, 20], "mean_bgr": groups.tolist(),
                   "assign": {"A": [0], "B": [1]}, "names": {"A": "A", "B": "B"}, "colours": {"A": "#fff", "B": "#f00"}},
                  open(labels, "w"))
        cl = TL.Labelled(labels)
        test = np.array([[235, 235, 235],      # clean white shirt
                         [20, 20, 20],         # the referee: clearly the dark group
                         [140, 200, 150]], float)   # white shirt washed out by grass: nearest group is the green one
        self.assertEqual(list(cl.team_of(test)), [0, -1, -1])
        self.assertEqual(list(cl.team_of_rescue(test, 1.0)), [0, -1, -1])           # ratio 1 = off
        self.assertEqual(list(cl.team_of_rescue(test, 2.0)), [0, -1, 0])            # rescued for white, referee stays unknown

    def test_view_focal_length_follows_the_view_fov(self):
        import math
        import scene_check as S
        self.assertAlmostEqual(S.view_fy({"square_views": True, "view_fov": 100.0}), S.SQUARE_FY, places=6)
        self.assertAlmostEqual(S.view_fy({"square_views": True}), S.SQUARE_FY, places=6)          # old files: 100 degrees
        self.assertAlmostEqual(S.view_fy({"square_views": True, "view_fov": 55.0}), 640 / math.tan(math.radians(27.5)), places=6)
        self.assertEqual(S.view_fy({"view_fov": 55.0}), S.LEGACY_FY)                              # legacy views ignore it
        a = self.p("a.jsonl")                                      # a narrow-view file puts the same box height much farther away
        for fov, name in ((100.0, "wide.jsonl"), (55.0, "narrow.jsonl")):
            with open(self.p(name), "w") as fh:
                fh.write(json.dumps({"type": "header", "view_lons": [0], "start": 0.0, "fps": 3.0, "square_views": True,
                                     "colours": True, "view_fov": fov}) + "\n")
                fh.write(json.dumps({"t": 0.0, "players": [[0.0, -10.0, 30.0, 0.9, None, [230, 230, 230]]], "balls": []}) + "\n")
        wide = BD.part_frames(self.p("wide.jsonl"), None, FakeClusters(), 0.0, 98.0, 0.3, 14.0, 0.75)[0]["players"][0]
        narrow = BD.part_frames(self.p("narrow.jsonl"), None, FakeClusters(), 0.0, 98.0, 0.3, 14.0, 0.75)[0]["players"][0]
        self.assertGreater(narrow[1], 2.0 * wide[1])

    def test_outline_margin_is_in_metres_not_degrees(self):
        import kf_core as K
        import pitch_filter as PF
        calib = self.p("pitch.json")
        json.dump({"pixel_polygon": [[1100, 600], [1780, 600], [1780, 1100], [1100, 1100]]}, open(calib, "w"))
        ft = PF.FootTester(calib, margin_px=25.0, margin_m=1.5)
        lon, lat = K.xy_to_lonlat(1030.0, 1000.0, 2880, 1440)          # 70 px left of the outline
        near_box, far_box = 243.0, 15.0                               # a player 5 m from the camera / one 80 m away
        self.assertTrue(ft(float(lon), float(lat), near_box))         # 1.5 m on the ground is a wide angle up close
        self.assertFalse(ft(float(lon), float(lat), far_box))         # far away the same angle is many metres: off
        lon2, lat2 = K.xy_to_lonlat(1090.0, 1000.0, 2880, 1440)        # 10 px outside: the old rule keeps it too
        self.assertTrue(ft(float(lon2), float(lat2), far_box))
        lon3, lat3 = K.xy_to_lonlat(1400.0, 900.0, 2880, 1440)         # inside
        self.assertTrue(ft(float(lon3), float(lat3), far_box))

    def test_board_keeps_the_player_on_the_near_touchline(self):
        import kf_core as K
        calib, a = self.p("pitch.json"), self.p("a.jsonl")
        json.dump({"pixel_polygon": [[1100, 600], [1780, 600], [1780, 1100], [1100, 1100]]}, open(calib, "w"))
        lon, lat = K.xy_to_lonlat(1030.0, 1000.0, 2880, 1440)
        with open(a, "w") as fh:
            fh.write(json.dumps({"type": "header", "view_lons": [0], "start": 0.0, "fps": 3.0, "square_views": True, "colours": True}) + "\n")
            fh.write(json.dumps({"t": 0.0, "players": [[float(lon), float(lat), 243.0, 0.9, None, [230, 230, 230]],   # close, beside the line
                                                       [float(lon), float(lat), 15.0, 0.9, None, [230, 230, 230]]],  # far, beside the line
                                 "balls": []}) + "\n")
        part = {"det": a, "half": 1, "video_start": 0.0, "duration": 5.0, "calib": calib}
        data = BD.build([part], FakeClusters())
        self.assertEqual((len(data["frames"][0]) - 3) // 3, 1)        # only the close one stays
        data = BD.build([{**part, "calib": None}], FakeClusters())
        self.assertEqual((len(data["frames"][0]) - 3) // 3, 2)        # no outline: both

    def test_tidy_corrects_a_flickering_colour_and_can_be_switched_off(self):
        a = self.p("a.jsonl")
        with open(a, "w") as fh:
            fh.write(json.dumps({"type": "header", "view_lons": [0], "start": 0.0, "fps": 3.0, "square_views": True, "colours": True}) + "\n")
            for i in range(12):
                shirt = [30, 30, 220] if i == 6 else [230, 230, 230]          # one moment the white shirt looked red
                fh.write(json.dumps({"t": i / 3.0, "players": [[0.0, -10.0, 60.0, 0.9, None, shirt]], "balls": []}) + "\n")
        part = {"det": a, "half": 1, "video_start": 0.0, "duration": 10.0}
        teams = lambda d: [f[5] for f in d["frames"]]
        self.assertEqual(set(teams(BD.build([part], FakeClusters()))), {0})
        self.assertEqual(teams(BD.build([part], FakeClusters(), tidy=False))[6], 1)
        self.assertEqual(BD.build([part], FakeClusters())["tidy"]["relabelled"], 1)

    def test_feet_make_far_positions_steadier(self):
        import test_ground_model as TG
        lon, lat, h, cen, d, _ = TG.world(n=1500, seed=7, noise=0.08, d_lo=40.0, d_hi=75.0)
        a = self.p("w.jsonl")
        with open(a, "w") as fh:
            fh.write(json.dumps({"type": "header", "view_lons": [-40, 0, 40], "start": 0.0, "fps": 3.0, "square_views": True,
                                 "colours": True}) + "\n")
            for i in range(150):                                              # 150 moments of 10 players each
                sl = slice(i * 10, i * 10 + 10)
                pl = [[float(lo), float(la), float(hh), 0.9, None, [230, 230, 230]] for lo, la, hh in zip(lon[sl], lat[sl], h[sl])]
                fh.write(json.dumps({"t": i / 3.0, "players": pl, "balls": []}) + "\n")
        part = {"det": a, "half": 1, "video_start": 0.0, "duration": 100.0}
        def ys(feet):
            d_ = BD.build([part], FakeClusters(), feet=feet, tidy=False)
            return np.array([[f[k + 1] / 10.0 for k in range(3, len(f), 3)] for f in d_["frames"]]).ravel()
        truth = (d * np.cos(np.radians(0.75 * lon)))[:1500].reshape(150, 10).ravel()    # the board's y: distance x cos(real angle)
        err = lambda y: np.median(np.abs(y - truth))
        self.assertLess(err(ys(True)), 0.7 * err(ys(False)))

    def test_probe_explains_a_drop(self):
        import io
        import contextlib
        import kf_core as K
        import board_probe as BP
        calib, a = self.p("pitch.json"), self.p("a.jsonl")
        json.dump({"pixel_polygon": [[1100, 600], [1780, 600], [1780, 1100], [1100, 1100]]}, open(calib, "w"))
        lon, lat = K.xy_to_lonlat(1030.0, 1000.0, 2880, 1440)
        with open(a, "w") as fh:
            fh.write(json.dumps({"type": "header", "view_lons": [0], "start": 100.0, "fps": 3.0, "square_views": True}) + "\n")
            fh.write(json.dumps({"t": 102.0, "players": [[float(lon), float(lat), 15.0, 0.9, None],
                                                         [0.0, -10.0, 60.0, 0.1, None]], "balls": []}) + "\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            BP.main(["--detections", a, "--calib", calib, "--at", "102", "--source-time"])
        out = buf.getvalue()
        self.assertIn("outside the outline", out)
        self.assertIn("confidence < 0.3", out)

    def test_no_players_is_an_error(self):
        a = self.p("a.jsonl")
        with open(a, "w") as fh:
            fh.write(json.dumps({"type": "header", "view_lons": [0], "start": 0.0, "fps": 3.0, "colours": True}) + "\n")
        with self.assertRaises(SystemExit):
            BD.build([{"det": a, "half": 1, "video_start": 0.0, "duration": 5.0}], FakeClusters())


if __name__ == "__main__":
    unittest.main()

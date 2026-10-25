"""Tests for board_clean.py: tracks, team votes, outline, 11 a side. Synthetic data, no footage."""
import unittest

import numpy as np

import board_clean as BC


def moments(n, per, dt=1 / 3):
    """n moments with `per` players walking slowly in a row: frame, vt, x, y."""
    frame = np.repeat(np.arange(n), per)
    vt = np.arange(n) * dt
    x = np.tile(np.arange(per) * 6.0, n) + frame * 0.3
    y = np.tile(np.full(per, 30.0), n) + np.tile(np.arange(per) * 0.5, n)
    return frame, vt, x, y


class TestTracks(unittest.TestCase):
    def test_a_walking_player_keeps_one_track(self):
        frame, vt, x, y = moments(20, 3)
        tr = BC.link_tracks(frame, x, y, vt)
        self.assertEqual(len(set(tr)), 3)
        self.assertTrue(all(len(set(tr[np.arange(k, 60, 3)])) == 1 for k in range(3)))

    def test_a_jump_starts_a_new_track(self):
        frame = np.array([0, 1])
        tr = BC.link_tracks(frame, np.array([0.0, 40.0]), np.array([30.0, 30.0]), np.array([0.0, 0.33]))
        self.assertNotEqual(tr[0], tr[1])

    def test_a_gap_in_time_breaks_the_track(self):
        frame = np.array([0, 1])
        tr = BC.link_tracks(frame, np.array([0.0, 0.2]), np.array([30.0, 30.0]), np.array([0.0, 5.0]))
        self.assertNotEqual(tr[0], tr[1])


class TestVote(unittest.TestCase):
    def test_a_wrong_colour_in_a_track_is_outvoted(self):
        frame, vt, x, y = moments(15, 1)
        team = np.zeros(15, int)
        team[7] = 1                                           # one frame the colour looked like the other team
        weight = np.full(15, 0.6)
        weight[7] = 0.2                                       # and it was not sure
        new, st = BC.clean(frame, vt, x, y, team, weight, np.ones(15), np.zeros(15))
        self.assertTrue((new == 0).all())
        self.assertEqual(st["relabelled"], 1)

    def test_a_sure_other_colour_is_not_overruled(self):
        frame, vt, x, y = moments(15, 1)
        team = np.full(15, 2)                                 # a referee, seen clearly in every moment
        new, _ = BC.clean(frame, vt, x, y, team, np.full(15, 0.5), np.ones(15), np.zeros(15))
        self.assertTrue((new == 2).all())

    def test_a_small_far_player_takes_the_team_of_its_track(self):
        frame, vt, x, y = moments(15, 1)
        team = np.full(15, 1)
        weight = np.full(15, 0.7)
        team[5:10] = 2                                        # five moments too small for a colour
        weight[5:10] = 0.0
        new, st = BC.clean(frame, vt, x, y, team, weight, np.ones(15), np.zeros(15))
        self.assertTrue((new == 1).all())
        self.assertEqual(st["recovered"], 5)

    def test_without_any_evidence_nothing_changes(self):
        frame, vt, x, y = moments(10, 1)
        team = np.full(10, 2)
        new, _ = BC.clean(frame, vt, x, y, team, np.zeros(10), np.ones(10), np.zeros(10))
        self.assertTrue((new == 2).all())


class TestOutline(unittest.TestCase):
    def test_a_team_colour_outside_the_pitch_becomes_other(self):
        frame, vt, x, y = moments(10, 2)
        team = np.zeros(20, int)
        out = np.zeros(20)
        out[1::2] = 1.2                                       # the second one stands 1.2 m outside, all the time
        new, st = BC.clean(frame, vt, x, y, team, np.full(20, 0.7), np.ones(20), out)
        self.assertTrue((new[0::2] == 0).all())
        self.assertTrue((new[1::2] == 2).all())
        self.assertEqual(st["outside"], 10)

    def test_someone_outside_is_no_witness(self):
        frame, vt, x, y = moments(12, 1)
        team = np.zeros(12, int)
        team[:6] = 1                                          # first half of the track looked like the other team ...
        out = np.zeros(12)
        out[:6] = 2.0                                         # ... but was outside the pitch there: that does not count
        new, _ = BC.clean(frame, vt, x, y, team, np.full(12, 0.7), np.ones(12), out)
        self.assertTrue((new[6:] == 0).all())


class TestElevenASide(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros(14, int)
        self.vt = np.array([0.0])
        self.x = np.arange(14) * 5.0
        self.y = np.full(14, 30.0)

    def test_twelve_of_a_team_are_cut_to_eleven(self):
        team = np.zeros(14, int)
        weight = np.linspace(0.9, 0.1, 14)
        team[12:] = 2
        new, _ = BC.clean(self.frame, self.vt, self.x, self.y, team, weight, np.ones(14), np.zeros(14))
        self.assertEqual((new == 0).sum(), 11)
        self.assertEqual(new[11], 2)                          # the least certain of the 12 lost its team (sure colour: other)
        self.assertTrue((new[:11] == 0).all())

    def test_an_uncertain_player_goes_to_the_team_with_room(self):
        team = np.zeros(14, int)
        team[12:] = 1
        weight = np.full(14, 0.8)
        margin = np.ones(14)
        weight[5], margin[5] = 0.05, 0.1                      # nearly as close to the other team
        new, st = BC.clean(self.frame, self.vt, self.x, self.y, team, weight, margin, np.zeros(14))
        self.assertEqual(new[5], 1)
        self.assertEqual(st["to_other_team"], 1)
        self.assertEqual((new == 0).sum(), 11)

    def test_a_full_other_team_does_not_take_more(self):
        team = np.array([0] * 12 + [1] * 11 + [2])
        n = len(team)
        weight = np.full(n, 0.8)
        margin = np.ones(n)
        weight[3], margin[3] = 0.05, 0.1
        new, st = BC.clean(np.zeros(n, int), self.vt, np.arange(n) * 4.0, np.full(n, 30.0), team, weight, margin, np.zeros(n))
        self.assertEqual(new[3], 2)
        self.assertEqual((new == 1).sum(), 11)

    def test_eleven_is_untouched(self):
        team = np.array([0] * 11 + [1] * 3)
        new, st = BC.clean(self.frame, self.vt, self.x, self.y, team, np.full(14, 0.8), np.ones(14), np.zeros(14))
        self.assertTrue((new == team).all())
        self.assertEqual(st["to_other"] + st["to_other_team"], 0)


class TestGaps(unittest.TestCase):
    def one_player(self, present):
        """one player standing still at (10, 30); `present` = the moments in which he is detected."""
        frame = np.array(present, int)
        n = int(frame.max()) + 1
        return frame, np.arange(n) / 3.0, np.full(len(frame), 10.0), np.full(len(frame), 30.0)

    def test_a_missed_moment_keeps_the_track(self):
        frame, vt, x, y = self.one_player([0, 1, 2, 4, 5, 8, 9])        # holes of 1 and 2 moments
        self.assertEqual(len(set(BC.link_tracks(frame, x, y, vt))), 1)

    def test_a_long_absence_starts_a_new_track(self):
        frame, vt, x, y = self.one_player([0, 1, 2, 20, 21])           # 6 s away
        self.assertEqual(len(set(BC.link_tracks(frame, x, y, vt))), 2)

    def test_the_gate_grows_with_the_absence(self):
        frame = np.array([0, 1, 5, 6])
        vt = np.arange(7) / 3.0
        x = np.array([10.0, 10.0, 14.0, 14.0])                        # 4 m further after 1.3 s: a runner, not a stranger
        self.assertEqual(len(set(BC.link_tracks(frame, x, np.full(4, 30.0), vt))), 1)
        x2 = np.array([10.0, 10.0, 18.0, 18.0])
        frame2 = np.array([0, 1, 3, 4])                                # 8 m in 0.7 s is too fast
        self.assertEqual(len(set(BC.link_tracks(frame2, x2, np.full(4, 30.0), np.arange(5) / 3.0))), 2)


class TestWeakDetections(unittest.TestCase):
    def test_a_steady_weak_track_is_kept_and_an_isolated_one_dropped(self):
        track = np.array([0] * 10 + [1])
        conf = np.array([0.22, 0.41, 0.25, 0.24, 0.33, 0.21, 0.27, 0.46, 0.23, 0.26, 0.25])
        keep = BC.confirm_low(track, conf, 0.3)
        self.assertTrue(keep[:10].all())
        self.assertFalse(keep[10])

    def test_a_short_weak_track_is_dropped(self):
        track = np.array([0, 0, 0])
        self.assertFalse(BC.confirm_low(track, np.array([0.25, 0.28, 0.22]), 0.3).any())

    def test_strong_detections_always_stay(self):
        self.assertTrue(BC.confirm_low(np.array([-1, -1]), np.array([0.9, 0.31]), 0.3).all())


class TestFillGaps(unittest.TestCase):
    def test_a_short_hole_is_bridged_with_the_same_label(self):
        frame = np.array([0, 1, 4, 5])
        vt = np.arange(6) / 3.0
        x = np.array([0.0, 1.0, 4.0, 5.0])
        f, fx, fy, ft = BC.fill_gaps(frame, vt, x, np.full(4, 30.0), np.array([1, 1, 1, 1]), np.zeros(4, int))
        self.assertEqual(list(f), [2, 3])
        self.assertTrue(np.allclose(fx, [2.0, 3.0]))
        self.assertEqual(list(ft), [1, 1])

    def test_a_long_hole_or_a_changed_label_is_not_bridged(self):
        vt = np.arange(30) / 3.0
        f, *_ = BC.fill_gaps(np.array([0, 20]), vt, np.zeros(2), np.zeros(2), np.array([0, 0]), np.zeros(2, int))
        self.assertEqual(len(f), 0)
        f, *_ = BC.fill_gaps(np.array([0, 3]), vt, np.zeros(2), np.zeros(2), np.array([0, 1]), np.zeros(2, int))
        self.assertEqual(len(f), 0)


if __name__ == "__main__":
    unittest.main()

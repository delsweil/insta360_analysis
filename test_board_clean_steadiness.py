import unittest
import numpy as np
import board_clean as BC


def game(n_moments=90, noise=0.25, seed=3):
    """Two teams of 4 players walking apart; each detection's colour label is wrong with probability `noise`."""
    rng = np.random.default_rng(seed)
    frame, x, y, team = [], [], [], []
    for f in range(n_moments):
        for p in range(8):
            frame.append(f)
            x.append(-20 + 2.5 * p + 0.05 * f * (1 if p < 4 else -1))
            y.append(30 + 6 * (p % 4))
            true = 0 if p < 4 else 1
            team.append(1 - true if rng.random() < noise else true)
    vt = np.arange(n_moments) / 3.0
    return np.array(frame), vt, np.array(x), np.array(y), np.array(team)


class TestSteadiness(unittest.TestCase):
    def test_disagreement_counts_minority_labels_of_long_tracks(self):
        track = np.array([0] * 12 + [1] * 12 + [2] * 3)
        team = np.array([0] * 10 + [1] * 2 + [1] * 12 + [0] * 3)
        bad, tot = BC.track_disagreement(track, team)
        self.assertEqual((bad, tot), (2, 24))                  # the short track (3) is not counted

    def test_clean_reports_steadiness_before_and_after_the_vote(self):
        frame, vt, x, y, team = game()
        n = len(frame)
        new, st = BC.clean(frame, vt, x, y, team, np.ones(n), np.ones(n), np.zeros(n))
        self.assertGreater(st["tracked"], 0)
        self.assertGreater(st["disagree_start"], st["disagree_vote"])         # the vote makes labels steadier
        self.assertLessEqual(st["disagree_final"], st["disagree_start"])

    def test_whole_track_vote_is_at_least_as_steady(self):
        frame, vt, x, y, team = game()
        n = len(frame)
        _, short = BC.clean(frame, vt, x, y, team, np.ones(n), np.ones(n), np.zeros(n), window_s=0.5)
        _, whole = BC.clean(frame, vt, x, y, team, np.ones(n), np.ones(n), np.zeros(n), window_s=1e9)
        self.assertLessEqual(whole["disagree_vote"], short["disagree_vote"])


if __name__ == "__main__":
    unittest.main()

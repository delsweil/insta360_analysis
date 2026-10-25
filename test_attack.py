"""Which way team A attacks: told by the coach, read from the goalkeepers, else guessed."""
import unittest

import numpy as np

import team_stats as TS


class Roles:
    has_roles = True

    def role_of(self, bgr):                      # first channel marks the keeper: 1 = GK A, 2 = GK B
        return np.where(bgr[:, 0] == 1, 4, np.where(bgr[:, 0] == 2, 5, -1))


def game(xa, xb, n=40):
    x = np.r_[np.full(n, xa, float), np.full(n, xb, float), np.zeros(n)]
    bgr = np.r_[np.tile([1, 0, 0], (n, 1)), np.tile([2, 0, 0], (n, 1)), np.tile([0, 0, 0], (n, 1))].astype(float)
    return {"x": x, "t": np.arange(3 * n, dtype=float), "bgr": bgr}


class TestAttack(unittest.TestCase):
    def test_keepers_tell_the_direction(self):
        self.assertTrue(TS.keeper_attack(game(-40, 40), Roles()))      # A's keeper at the left goal: A attacks right
        self.assertFalse(TS.keeper_attack(game(40, -40), Roles()))

    def test_unseparated_keepers_give_no_answer(self):
        self.assertIsNone(TS.keeper_attack(game(5, 8), Roles()))       # both keepers in one place / one colour group
        self.assertIsNone(TS.keeper_attack(game(-40, 40, n=5), Roles()))   # too few sightings

    def test_no_roles_no_answer(self):
        class Plain:
            pass
        self.assertIsNone(TS.keeper_attack(game(-40, 40), Plain()))

    def test_territory_follows_the_told_direction(self):
        n = 200
        g = {"frame": np.repeat(np.arange(n // 10), 10), "t": np.repeat(np.arange(n // 10) * 5.0, 10) + 1.0,
             "x": np.r_[np.full(n // 2, -10.0), np.full(n // 2, -10.0)], "duration": 100.0}
        team = np.tile([0, 1], n // 2)
        box = (-50.0, 50.0, 0.0, 70.0)
        _, v_right, a_right = TS.territory(g, team, box, attack=True)
        _, v_left, a_left = TS.territory(g, team, box, attack=False)
        self.assertTrue(a_right)
        self.assertFalse(a_left)
        self.assertTrue((np.sign(v_right[5:-5]) == -np.sign(v_left[5:-5])).all())    # same play, the other way round


if __name__ == "__main__":
    unittest.main()

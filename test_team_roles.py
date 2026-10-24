"""team_labels.py roles: referee, goalkeepers of A and B, 'other' (not counted)."""
import json
import os
import tempfile
import unittest

import numpy as np

import scene_check as S
import team_labels as TL


class Args:
    a = b = ref = gk_a = gk_b = None
    name_a = name_b = colour_a = colour_b = None


def write_meta(path, k=7):
    bgr = np.array([[230, 230, 230], [30, 30, 220], [20, 20, 20], [200, 10, 200], [10, 200, 10], [200, 200, 60], [120, 120, 120]], float)[:k]
    meta = {"centres": S.colour_features(bgr).tolist(), "sizes": [50, 40, 10, 3, 3, 8, 6][:k], "mean_bgr": bgr.tolist(),
            "assign": {"A": [], "B": []}, "names": {"A": "A", "B": "B"}, "colours": {"A": "#fff", "B": "#f00"}}
    json.dump(meta, open(path, "w"))
    return bgr


class TestRoles(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "l.json")
        self.bgr = write_meta(self.path)

    def assign(self, **kw):
        a = Args()
        for k, v in kw.items():
            setattr(a, k, v)
        a.labels = self.path
        return TL.assign(a)

    def test_roles_are_stored_and_read(self):
        self.assign(a=[0], b=[1], ref=[2], gk_a=[3], gk_b=[4])
        L = TL.Labelled(self.path)
        self.assertTrue(L.has_roles)
        self.assertEqual(list(L.role_of(self.bgr)), [-1, -1, 3, 4, 5, -1, -1])
        self.assertEqual(list(L.team_of(self.bgr)), [0, 1, -1, -1, -1, -1, -1])   # statistics: only the two teams

    def test_old_files_without_roles_still_load(self):
        m = json.load(open(self.path)); m["assign"] = {"A": [0], "B": [1]}; json.dump(m, open(self.path, "w"))
        L = TL.Labelled(self.path)
        self.assertFalse(L.has_roles)
        self.assertTrue((L.role_of(self.bgr) == -1).all())

    def test_a_group_cannot_have_two_roles(self):
        with self.assertRaises(SystemExit):
            self.assign(a=[0], b=[1], ref=[2], gk_a=[2])

    def test_quality_does_not_count_roles_as_other(self):
        self.assign(a=[0, 5], b=[1], ref=[2], gk_a=[3], gk_b=[4])
        q = TL.quality(json.load(open(self.path)))
        self.assertAlmostEqual(q["roles"], 16 / 120, places=3)
        self.assertAlmostEqual(q["a"] + q["b"] + q["roles"] + q["other"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

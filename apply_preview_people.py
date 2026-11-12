#!/usr/bin/env python3
"""apply_preview_people.py - adds the quick look at people before the detection run to this folder.

Put preview_people.py, test_preview_people.py and this file in the insta360_analysis folder, then:  python3 apply_preview_people.py
It changes pipeline.py, app.py, app.html and test_app.py by exact text replacement (a patch failed on this copy). A change that
is already there is skipped; one whose old text is not found is reported and nothing else in that file is touched.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHANGES = [
    [
     "pipeline.py",
     "FIXED_SECONDS = {\"check\": 15, \"pitch\": 20,",
     "FIXED_SECONDS = {\"check\": 15, \"preview\": 40, \"pitch\": 20,"
    ],
    [
     "pipeline.py",
     "        st = [{\"id\": \"check\", \"label\": \"Look at the videos\"}, {\"id\": \"teams\", \"label\": \"Find the shirt colours\"}]\n        for i in range(n):\n            st.append({\"id\": f\"extract{i}\",",
     "        st = [{\"id\": \"check\", \"label\": \"Look at the videos\"}]\n        for i in range(n):\n            st.append({\"id\": f\"preview{i}\", \"label\": f\"Look for people near the pitch{self._of(i, n)}\", \"video\": i})\n        st.append({\"id\": \"teams\", \"label\": \"Find the shirt colours\"})\n        for i in range(n):\n            st.append({\"id\": f\"extract{i}\","
    ],
    [
     "pipeline.py",
     "res = fn(s) if kind in (\"extract\", \"pitch\", \"keyframes\", \"render\") else fn()",
     "res = fn(s) if kind in (\"extract\", \"preview\", \"pitch\", \"keyframes\", \"render\") else fn()"
    ],
    [
     "pipeline.py",
     "    def _frames_re(self):",
     "    def s_preview(self, st: dict) -> None:\n        \"\"\"A quick look at about a dozen moments before the long detection run: where do people stand? The outline editor\n        shows them (green = kept, red = outside) so the line can be drawn before the players are found.\"\"\"\n        i = st[\"video\"]\n        cam = self.info()[i]\n        s0, d0 = self.span(i)\n        try:\n            self.ensure_frame(i)                                 # the picture the outline is drawn on\n        except Exception as e:\n            self.log(f\"picture for the outline not made: {e}\")\n        pat = re.compile(r\"frame (\\d+)/(\\d+)\")\n        parse = lambda line: (int(pat.search(line).group(1)) / max(1, int(pat.search(line).group(2)))) if pat.search(line) else None  # noqa: E731\n        try:\n            self.exec(self.py(\"preview_people.py\", \"--video\", self.videos()[i][\"path\"], \"--start\", f\"{s0:.2f}\",\n                              \"--duration\", f\"{d0:.2f}\", \"--raw\", \"1\" if cam[\"layout\"] == \"raw\" else \"0\",\n                              \"--out\", self.p(f\"pitch{i}.preview.json\")), st[\"id\"], parse)\n        except Cancelled:\n            raise\n        except Exception as e:                                   # only a help for drawing: never stop the game for it\n            self.log(f\"preview of the people not made: {e}\")\n\n    def preview_feet(self, i: int) -> List[List[int]]:\n        \"\"\"The feet seen by the quick look (see s_preview), [] if it has not run or found nothing.\"\"\"\n        d = read_json(self.p(f\"pitch{i}.preview.json\"))\n        return [[int(x), int(y)] for x, y in (d or {}).get(\"feet\", [])]\n\n    def _frames_re(self):"
    ],
    [
     "app.py",
     "\"feet\": job.feet(i) if os.path.isfile(job.p(f\"v{i}.detections.jsonl\")) else []})",
     "\"feet\": job.feet(i) if os.path.isfile(job.p(f\"v{i}.detections.jsonl\")) else job.preview_feet(i)})"
    ],
    [
     "app.html",
     "const SL={check:[\"Look at the videos\",\"Videos ansehen\"],",
     "const SL={check:[\"Look at the videos\",\"Videos ansehen\"],preview:[\"Look for people near the pitch\",\"Personen am Spielfeldrand suchen\"],"
    ],
    [
     "test_app.py",
     "    elif script == \"calibrate_auto.py\":\n        json.dump({\"pixel_polygon\"",
     "    elif script == \"preview_people.py\":\n        json.dump({\"times\": [10.0], \"feet\": [[500, 700], [100, 300]]}, open(arg(cmd, \"--out\"), \"w\"))\n    elif script == \"calibrate_auto.py\":\n        json.dump({\"pixel_polygon\""
    ],
    [
     "test_app.py",
     "        self.assertEqual(ids[:2], [\"check\", \"teams\"])\n        self.assertIn(\"render1\", ids)",
     "        self.assertEqual(ids[:4], [\"check\", \"preview0\", \"preview1\", \"teams\"])     # a quick look for people comes first\n        self.assertIn(\"render1\", ids)"
    ],
    [
     "test_app.py",
     "    def test_only_second_half(self):",
     "    def test_quick_look_for_people_before_the_detection_run(self):\n        j = self.job(1)\n        self.assertEqual(j.run(), \"needs_input\")                       # waits for the teams, as in the other tests\n        self.assertEqual(j.preview_feet(0), [[500, 700], [100, 300]])\n        ids = [s[\"id\"] for s in j.stages()]\n        self.assertLess(ids.index(\"preview0\"), ids.index(\"extract0\"))\n\n    def test_quick_look_failing_does_not_stop_the_game(self):\n        def broken(self, cmd, sid, parse=None):\n            if \"preview_people.py\" in cmd[2]:\n                raise RuntimeError(\"no model\")\n            return fake_exec(self, cmd, sid, parse)\n        P.Job.exec = broken\n        j = self.job(1)\n        self.assertEqual(j.run(), \"needs_input\")\n        self.assertEqual(j.preview_feet(0), [])\n\n    def test_only_second_half(self):"
    ]
]


def main() -> int:
    bad = 0
    for fn, old, new in CHANGES:
        path = os.path.join(HERE, fn)
        text = open(path, encoding="utf-8").read()
        if new in text:
            print(f"already there  {fn}: {old.strip().splitlines()[0][:60]}")
            continue
        if text.count(old) != 1:
            print(f"NOT FOUND ({text.count(old)}x)  {fn}: {old.strip().splitlines()[0][:60]}")
            bad += 1
            continue
        open(path, "w", encoding="utf-8").write(text.replace(old, new))
        print(f"changed        {fn}: {old.strip().splitlines()[0][:60]}")
    for need in ("preview_people.py", "test_preview_people.py"):
        if not os.path.isfile(os.path.join(HERE, need)):
            print(f"MISSING FILE {need}: copy it into this folder")
            bad += 1
    print("done" if not bad else f"{bad} problem(s): tell Claude which lines were not found")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

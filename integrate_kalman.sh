#!/usr/bin/env bash
# integrate_kalman.sh
# Integrates KalmanBallTracker into autopan_infer.py
# Run from: ~/insta360_analysis
# Usage: bash integrate_kalman.sh

set -e
SCRIPT="autopan_infer.py"
echo "Backing up ${SCRIPT}..."
cp "${SCRIPT}" "${SCRIPT}.bak2"

# ── CHANGE 1: Add Kalman constants after BALL_LONG_FALLBACK ──────
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = "BALL_LONG_FALLBACK   = 90   # frames before falling back to centre"
new = ("BALL_LONG_FALLBACK   = 90   # frames before falling back to centre\n"
       "\n"
       "# ── Kalman ball tracker ──────────────────────────────────────────────\n"
       "KALMAN_PROCESS_NOISE  = 50.0   # expected ball acceleration (px/frame²)\n"
       "KALMAN_MEASURE_NOISE  = 15.0   # pixel uncertainty in detection\n"
       "KALMAN_MAX_PREDICT    = 45     # max frames to trust prediction (~1.5s)\n"
       "KALMAN_REINIT_DIST    = 200    # px — jump distance to trigger reinit")
assert old in src, "BALL_LONG_FALLBACK not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 1 OK: Kalman constants added")
PYEOF

# ── CHANGE 2: Replace BallState dataclass with KalmanBallTracker ─
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = ("@dataclass\n"
       "class BallState:\n"
       "    trusted:      bool  = False\n"
       "    confirm:      int   = 0\n"
       "    frames_since: int   = 999\n"
       "    last_pos:     Optional[Tuple[float,float]] = None")

new = ("@dataclass\n"
       "class BallState:\n"
       "    trusted:      bool  = False\n"
       "    confirm:      int   = 0\n"
       "    frames_since: int   = 999\n"
       "    last_pos:     Optional[Tuple[float,float]] = None\n"
       "\n"
       "\n"
       "class KalmanBallTracker:\n"
       "    \"\"\"Constant-velocity Kalman filter for ball tracking.\n"
       "    State: [x, y, vx, vy]. Predicts ball position when detection is lost.\"\"\"\n"
       "\n"
       "    def __init__(self):\n"
       "        self._initialised = False\n"
       "        self.frames_since_detection = 999\n"
       "        self.frames_tracked = 0\n"
       "        self.last_conf = 0.0\n"
       "        # Compatibility with existing BallState references\n"
       "        self.last_pos = None\n"
       "        self.trusted = False\n"
       "        self.frames_since = 999\n"
       "        self._x = np.zeros((4, 1), dtype=np.float64)\n"
       "        self._P = np.eye(4, dtype=np.float64) * 500.0\n"
       "        self._F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float64)\n"
       "        self._H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)\n"
       "        q = KALMAN_PROCESS_NOISE\n"
       "        self._Q = np.array([[q/4,0,q/2,0],[0,q/4,0,q/2],[q/2,0,q,0],[0,q/2,0,q]], dtype=np.float64)\n"
       "        r = KALMAN_MEASURE_NOISE ** 2\n"
       "        self._R = np.array([[r,0],[0,r]], dtype=np.float64)\n"
       "\n"
       "    def _init_state(self, cx, cy):\n"
       "        self._x = np.array([[cx],[cy],[0.0],[0.0]], dtype=np.float64)\n"
       "        self._P = np.eye(4, dtype=np.float64) * 500.0\n"
       "        self._initialised = True\n"
       "        self.frames_tracked = 1\n"
       "        self.frames_since_detection = 0\n"
       "        self.frames_since = 0\n"
       "        self.last_pos = (cx, cy)\n"
       "        self.trusted = True\n"
       "\n"
       "    def update(self, cx, cy, conf=1.0):\n"
       "        if not self._initialised:\n"
       "            self._init_state(cx, cy)\n"
       "            self.last_conf = conf\n"
       "            return\n"
       "        pred_x, pred_y = float(self._x[0]), float(self._x[1])\n"
       "        dist = ((cx-pred_x)**2 + (cy-pred_y)**2)**0.5\n"
       "        if dist > KALMAN_REINIT_DIST and self.frames_since_detection == 0:\n"
       "            return  # consecutive jump — likely false positive\n"
       "        if dist > KALMAN_REINIT_DIST * 2:\n"
       "            self._init_state(cx, cy)  # reinitialise on extreme jump\n"
       "            self.last_conf = conf\n"
       "            return\n"
       "        self._x = self._F @ self._x\n"
       "        self._P = self._F @ self._P @ self._F.T + self._Q\n"
       "        z = np.array([[cx],[cy]], dtype=np.float64)\n"
       "        y = z - self._H @ self._x\n"
       "        S = self._H @ self._P @ self._H.T + self._R\n"
       "        K = self._P @ self._H.T @ np.linalg.inv(S)\n"
       "        self._x = self._x + K @ y\n"
       "        self._P = (np.eye(4) - K @ self._H) @ self._P\n"
       "        self.frames_since_detection = 0\n"
       "        self.frames_since = 0\n"
       "        self.frames_tracked += 1\n"
       "        self.last_conf = conf\n"
       "        self.last_pos = (float(self._x[0]), float(self._x[1]))\n"
       "        self.trusted = True\n"
       "\n"
       "    def predict_only(self):\n"
       "        if not self._initialised:\n"
       "            self.frames_since_detection += 1\n"
       "            self.frames_since += 1\n"
       "            return\n"
       "        self._x = self._F @ self._x\n"
       "        self._P = self._F @ self._P @ self._F.T + self._Q\n"
       "        self.frames_since_detection += 1\n"
       "        self.frames_since += 1\n"
       "        decay = 0.92 ** min(self.frames_since_detection, 30)\n"
       "        self._x[2] *= decay\n"
       "        self._x[3] *= decay\n"
       "        if self._initialised:\n"
       "            self.last_pos = (float(self._x[0]), float(self._x[1]))\n"
       "\n"
       "    def predicted_pos(self):\n"
       "        if not self._initialised: return None\n"
       "        return float(self._x[0]), float(self._x[1])\n"
       "\n"
       "    def velocity(self):\n"
       "        if not self._initialised: return 0.0, 0.0\n"
       "        return float(self._x[2]), float(self._x[3])\n"
       "\n"
       "    def is_valid(self):\n"
       "        return self._initialised and self.frames_since_detection < KALMAN_MAX_PREDICT\n"
       "\n"
       "    def is_fresh(self):\n"
       "        return self._initialised and self.frames_since_detection < BALL_SHORT_FALLBACK\n"
       "\n"
       "    def reset(self):\n"
       "        self.__init__()")

assert old in src, "BallState not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 2 OK: KalmanBallTracker added after BallState")
PYEOF

# ── CHANGE 3: Replace choose_target body with Kalman version ─────
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = ('def choose_target(\n'
       '    players: List[Tuple[float,float]],\n'
       '    ball: Optional[Tuple[float,float,float]],\n'
       '    ball_state: BallState,\n'
       ') -> Tuple[float, float, str]:\n'
       '    """Choose pan target. Priority: trusted ball > players centroid > centre."""\n'
       '\n'
       '    # Update ball state\n'
       '    if ball is not None:\n'
       '        ball_state.confirm += 1\n'
       '        ball_conf = ball[2]\n'
       '        # High confidence ball: trust immediately, no confirmation needed\n'
       '        if ball_conf >= 0.50 or ball_state.confirm >= BALL_CONFIRM_FRAMES:\n'
       '            ball_state.trusted = True\n'
       '        ball_state.last_pos = (ball[0], ball[1])\n'
       '        ball_state.frames_since = 0\n'
       '    else:\n'
       '        ball_state.frames_since += 1\n'
       '        ball_state.confirm = 0\n'
       '\n'
       '    # Use trusted ball — high conf ball always wins immediately\n'
       '    if ball is not None and ball[2] >= 0.50:\n'
       '        return ball[0], ball[1], \'ball_highconf\'\n'
       '    if ball_state.trusted and ball_state.frames_since < BALL_SHORT_FALLBACK:\n'
       '        return ball_state.last_pos[0], ball_state.last_pos[1], \'ball\'\n'
       '\n'
       '    # Recently saw ball — use last known position briefly\n'
       '    if ball_state.last_pos and ball_state.frames_since < BALL_LONG_FALLBACK:\n'
       '        if players:\n'
       '            cx = float(np.mean([p[0] for p in players]))\n'
       '            cy = float(np.mean([p[1] for p in players]))\n'
       '            # Blend toward last ball position\n'
       '            alpha = 1.0 - ball_state.frames_since / BALL_LONG_FALLBACK\n'
       '            tx = alpha * ball_state.last_pos[0] + (1-alpha) * cx\n'
       '            ty = alpha * ball_state.last_pos[1] + (1-alpha) * cy\n'
       '            return tx, ty, \'blend\'\n'
       '        return ball_state.last_pos[0], ball_state.last_pos[1], \'last_ball\'\n'
       '\n'
       '    # Players centroid — weighted by proximity to median position\n'
       '    # This down-weights outliers (e.g. goalkeeper on far side)\n'
       '    if len(players) >= 2:\n'
       '        xs = np.array([p[0] for p in players])\n'
       '        ys = np.array([p[1] for p in players])\n'
       '        # Median as action centre\n'
       '        med_x = float(np.median(xs))\n'
       '        med_y = float(np.median(ys))\n'
       '        # Gaussian weights: sigma = 25% of frame width (~320px)\n'
       '        sigma = OUT_W * 0.25\n'
       '        dists_sq = (xs - med_x)**2 + (ys - med_y)**2\n'
       '        weights = np.exp(-dists_sq / (2 * sigma**2))\n'
       '        weights /= weights.sum()\n'
       '        cx = float(np.sum(weights * xs))\n'
       '        cy = float(np.sum(weights * ys))\n'
       '        if len(players) >= 4:\n'
       '            return cx, cy, \'players\'\n'
       '        else:\n'
       '            # Few players — weak pull, mostly stay put\n'
       '            tx = 0.2 * cx + 0.8 * (OUT_W/2)\n'
       '            ty = 0.2 * cy + 0.8 * (OUT_H/2)\n'
       '            return tx, ty, \'few_players\'\n'
       '    elif len(players) == 1:\n'
       '        # Single player — almost entirely ignore, drift to centre\n'
       '        tx = 0.05 * players[0][0] + 0.95 * (OUT_W/2)\n'
       '        ty = 0.05 * players[0][1] + 0.95 * (OUT_H/2)\n'
       '        return tx, ty, \'single_player\'\n'
       '\n'
       '    # Centre fallback\n'
       '    return OUT_W / 2, OUT_H / 2, \'centre\'')

new = ('def choose_target(\n'
       '    players: List[Tuple[float,float]],\n'
       '    ball: Optional[Tuple[float,float,float]],\n'
       '    ball_state,  # KalmanBallTracker\n'
       ') -> Tuple[float, float, str]:\n'
       '    """Choose pan target using Kalman ball tracker."""\n'
       '\n'
       '    # Update Kalman tracker\n'
       '    if ball is not None:\n'
       '        ball_state.update(ball[0], ball[1], ball[2])\n'
       '    else:\n'
       '        ball_state.predict_only()\n'
       '\n'
       '    # 1. Fresh high-confidence detection — trust immediately\n'
       '    if ball is not None and ball[2] >= 0.50:\n'
       '        return ball[0], ball[1], \'ball_highconf\'\n'
       '\n'
       '    # 2. Fresh detection (recently confirmed)\n'
       '    if ball_state.is_fresh() and ball_state.predicted_pos() is not None:\n'
       '        px, py = ball_state.predicted_pos()\n'
       '        return px, py, \'ball\'\n'
       '\n'
       '    # 3. Kalman prediction still valid — follow predicted trajectory\n'
       '    if ball_state.is_valid() and ball_state.predicted_pos() is not None:\n'
       '        px, py = ball_state.predicted_pos()\n'
       '        age_ratio = ball_state.frames_since_detection / KALMAN_MAX_PREDICT\n'
       '        kalman_weight = max(0.3, 0.9 * (1.0 - age_ratio))\n'
       '        if len(players) >= 4:\n'
       '            xs = np.array([p[0] for p in players])\n'
       '            ys = np.array([p[1] for p in players])\n'
       '            med_x, med_y = float(np.median(xs)), float(np.median(ys))\n'
       '            sigma = OUT_W * 0.25\n'
       '            w = np.exp(-((xs-med_x)**2+(ys-med_y)**2)/(2*sigma**2))\n'
       '            w /= w.sum()\n'
       '            pcx, pcy = float(np.sum(w*xs)), float(np.sum(w*ys))\n'
       '            tx = kalman_weight * px + (1-kalman_weight) * pcx\n'
       '            ty = kalman_weight * py + (1-kalman_weight) * pcy\n'
       '            return tx, ty, \'kalman_blend\'\n'
       '        return px, py, \'kalman\'\n'
       '\n'
       '    # 4. No valid prediction — weighted player centroid\n'
       '    if len(players) >= 2:\n'
       '        xs = np.array([p[0] for p in players])\n'
       '        ys = np.array([p[1] for p in players])\n'
       '        med_x, med_y = float(np.median(xs)), float(np.median(ys))\n'
       '        sigma = OUT_W * 0.25\n'
       '        dists_sq = (xs-med_x)**2 + (ys-med_y)**2\n'
       '        weights = np.exp(-dists_sq/(2*sigma**2))\n'
       '        weights /= weights.sum()\n'
       '        cx = float(np.sum(weights*xs))\n'
       '        cy = float(np.sum(weights*ys))\n'
       '        if len(players) >= 4:\n'
       '            return cx, cy, \'players\'\n'
       '        else:\n'
       '            tx = 0.2 * cx + 0.8 * (OUT_W/2)\n'
       '            ty = 0.2 * cy + 0.8 * (OUT_H/2)\n'
       '            return tx, ty, \'few_players\'\n'
       '    elif len(players) == 1:\n'
       '        tx = 0.05 * players[0][0] + 0.95 * (OUT_W/2)\n'
       '        ty = 0.05 * players[0][1] + 0.95 * (OUT_H/2)\n'
       '        return tx, ty, \'single_player\'\n'
       '\n'
       '    return OUT_W / 2, OUT_H / 2, \'centre\'')

assert old in src, "choose_target body not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 3 OK: choose_target updated to use KalmanBallTracker")
PYEOF

# ── CHANGE 4: Replace BallState() instantiation with KalmanBallTracker() ──
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = "    ball_state = BallState()"
new = "    ball_state = KalmanBallTracker()"
assert old in src, "BallState() instantiation not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 4 OK: BallState() replaced with KalmanBallTracker()")
PYEOF

# ── CHANGE 5: Update mode_counts to include kalman modes ─────────
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = "    mode_counts = {'ball_highconf':0, 'ball':0, 'blend':0, 'last_ball':0, 'players':0, 'few_players':0, 'single_player':0, 'centre':0}"
new = "    mode_counts = {'ball_highconf':0, 'ball':0, 'kalman':0, 'kalman_blend':0, 'blend':0, 'last_ball':0, 'players':0, 'few_players':0, 'single_player':0, 'centre':0}"
assert old in src, "mode_counts not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 5 OK: kalman modes added to mode_counts")
PYEOF

# ── CHANGE 6: Update detect_ball call — remove ball_state reference ──
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = ("                # Use last known position as spatial reference if seen recently (within long fallback)\n"
       "                _last_trusted = ball_state.last_pos if ball_state.last_pos and ball_state.frames_since < BALL_LONG_FALLBACK else None\n"
       "                last_ball = detect_ball(persp, ball_model, device, mask, _last_trusted, mask_eroded)")
new = ("                # Use Kalman predicted position as spatial reference\n"
       "                _last_trusted = ball_state.predicted_pos() if ball_state.is_valid() else None\n"
       "                last_ball = detect_ball(persp, ball_model, device, mask, _last_trusted, mask_eroded)")
assert old in src, "detect_ball call not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 6 OK: detect_ball uses Kalman predicted_pos as spatial reference")
PYEOF

echo ""
echo "Verifying syntax..."
python3 -c "import ast; ast.parse(open('autopan_infer.py').read()); print('Syntax OK')"

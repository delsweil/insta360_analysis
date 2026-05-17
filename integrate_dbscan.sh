#!/usr/bin/env bash
# integrate_dbscan.sh
# Integrates DBSCAN player clustering into autopan_infer.py
# Run from: ~/insta360_analysis
# Usage: bash integrate_dbscan.sh

set -e
SCRIPT="autopan_infer.py"
echo "Backing up ${SCRIPT}..."
cp "${SCRIPT}" "${SCRIPT}.bak3"

# ── CHANGE 1: Add sklearn import at top of file ──────────────────
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = "import cv2"
new = "import cv2\nfrom sklearn.cluster import DBSCAN"
assert old in src, "cv2 import not found"
# Only replace first occurrence
p.write_text(src.replace(old, new, 1))
print("CHANGE 1 OK: DBSCAN import added")
PYEOF

# ── CHANGE 2: Add DBSCAN constants near other constants ──────────
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = "KALMAN_REINIT_DIST    = 200    # px — jump distance to trigger reinit"
new = ("KALMAN_REINIT_DIST    = 200    # px — jump distance to trigger reinit\n"
       "\n"
       "# ── DBSCAN player clustering ─────────────────────────────────────────\n"
       "DBSCAN_EPS_DEG        = 8.0    # cluster radius in equirect degrees\n"
       "DBSCAN_MIN_SAMPLES    = 2      # minimum players to form a cluster\n"
       "DBSCAN_MIN_SIZE       = 3      # min cluster size to use as target")
assert old in src, "KALMAN_REINIT_DIST not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 2 OK: DBSCAN constants added")
PYEOF

# ── CHANGE 3: Add find_action_cluster function before choose_target ──
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = "def choose_target("
new = (
    "def pixel_to_lon(px: float, yaw_deg: float,\n"
    "                 w: int = OUT_W, fov_deg: float = 100.0) -> float:\n"
    "    \"\"\"Convert perspective frame pixel x to equirectangular longitude.\"\"\"\n"
    "    nx = (px - w / 2) / (w / 2)\n"
    "    import math\n"
    "    half_fov = math.radians(fov_deg / 2)\n"
    "    angle_x = math.degrees(math.atan(nx * math.tan(half_fov)))\n"
    "    return angle_x + yaw_deg\n"
    "\n"
    "\n"
    "def find_action_cluster(players: List[Tuple[float, float]],\n"
    "                        cam_yaw: float,\n"
    "                        e2p_fov: float) -> Optional[Tuple[float, float]]:\n"
    "    \"\"\"Find the densest player cluster in equirectangular space.\n"
    "    Returns (cluster_cx_px, cluster_cy_px) in perspective frame coords,\n"
    "    or None if no clear cluster found.\"\"\"\n"
    "    if len(players) < DBSCAN_MIN_SAMPLES:\n"
    "        return None\n"
    "\n"
    "    # Convert player x positions to equirect longitudes\n"
    "    lons = np.array([pixel_to_lon(p[0], cam_yaw, OUT_W, e2p_fov)\n"
    "                     for p in players])\n"
    "    heights = np.array([p[1] for p in players])  # foot_y as proxy\n"
    "\n"
    "    # DBSCAN clustering on longitudes\n"
    "    X = lons.reshape(-1, 1)\n"
    "    db = DBSCAN(eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_SAMPLES).fit(X)\n"
    "    labels = db.labels_\n"
    "\n"
    "    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)\n"
    "    if n_clusters == 0:\n"
    "        return None\n"
    "\n"
    "    # Score each cluster: size * mean_player_height\n"
    "    best_score = -1\n"
    "    best_cx = None\n"
    "    best_cy = None\n"
    "\n"
    "    for label in set(labels):\n"
    "        if label == -1:\n"
    "            continue\n"
    "        mask = labels == label\n"
    "        if mask.sum() < DBSCAN_MIN_SIZE:\n"
    "            continue\n"
    "        # Use mean pixel position of cluster players as target\n"
    "        cluster_players = [players[i] for i in range(len(players)) if mask[i]]\n"
    "        cx = float(np.mean([p[0] for p in cluster_players]))\n"
    "        cy = float(np.mean([p[1] for p in cluster_players]))\n"
    "        # Score by cluster size (larger = more likely ball location)\n"
    "        score = mask.sum()\n"
    "        if score > best_score:\n"
    "            best_score = score\n"
    "            best_cx = cx\n"
    "            best_cy = cy\n"
    "\n"
    "    if best_cx is None:\n"
    "        return None\n"
    "    return best_cx, best_cy\n"
    "\n"
    "\n"
    "def choose_target("
)
assert "def choose_target(" in src, "choose_target not found"
p.write_text(src.replace("def choose_target(", new, 1))
print("CHANGE 3 OK: find_action_cluster and pixel_to_lon added")
PYEOF

# ── CHANGE 4: Update choose_target signature to accept cam_yaw and e2p_fov ──
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = ("def choose_target(\n"
       "    players: List[Tuple[float,float]],\n"
       "    ball: Optional[Tuple[float,float,float]],\n"
       "    ball_state,  # KalmanBallTracker\n"
       ") -> Tuple[float, float, str]:")
new = ("def choose_target(\n"
       "    players: List[Tuple[float,float]],\n"
       "    ball: Optional[Tuple[float,float,float]],\n"
       "    ball_state,  # KalmanBallTracker\n"
       "    cam_yaw: float = 0.0,\n"
       "    e2p_fov: float = 100.0,\n"
       ") -> Tuple[float, float, str]:")
assert old in src, "choose_target signature not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 4 OK: cam_yaw and e2p_fov added to choose_target signature")
PYEOF

# ── CHANGE 5: Replace player centroid fallback with DBSCAN cluster ──
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = (
    "    # 4. No valid prediction — weighted player centroid\n"
    "    if len(players) >= 2:\n"
    "        xs = np.array([p[0] for p in players])\n"
    "        ys = np.array([p[1] for p in players])\n"
    "        med_x, med_y = float(np.median(xs)), float(np.median(ys))\n"
    "        sigma = OUT_W * 0.25\n"
    "        dists_sq = (xs-med_x)**2 + (ys-med_y)**2\n"
    "        weights = np.exp(-dists_sq/(2*sigma**2))\n"
    "        weights /= weights.sum()\n"
    "        cx = float(np.sum(weights*xs))\n"
    "        cy = float(np.sum(weights*ys))\n"
    "        if len(players) >= 4:\n"
    "            return cx, cy, 'players'\n"
    "        else:\n"
    "            tx = 0.2 * cx + 0.8 * (OUT_W/2)\n"
    "            ty = 0.2 * cy + 0.8 * (OUT_H/2)\n"
    "            return tx, ty, 'few_players'\n"
    "    elif len(players) == 1:\n"
    "        tx = 0.05 * players[0][0] + 0.95 * (OUT_W/2)\n"
    "        ty = 0.05 * players[0][1] + 0.95 * (OUT_H/2)\n"
    "        return tx, ty, 'single_player'\n"
    "\n"
    "    return OUT_W / 2, OUT_H / 2, 'centre'"
)
new = (
    "    # 4. No valid prediction — try DBSCAN cluster first, fall back to weighted centroid\n"
    "    if len(players) >= DBSCAN_MIN_SAMPLES:\n"
    "        cluster = find_action_cluster(players, cam_yaw, e2p_fov)\n"
    "        if cluster is not None:\n"
    "            return cluster[0], cluster[1], 'players'\n"
    "\n"
    "    # Fallback: weighted centroid\n"
    "    if len(players) >= 2:\n"
    "        xs = np.array([p[0] for p in players])\n"
    "        ys = np.array([p[1] for p in players])\n"
    "        med_x, med_y = float(np.median(xs)), float(np.median(ys))\n"
    "        sigma = OUT_W * 0.25\n"
    "        dists_sq = (xs-med_x)**2 + (ys-med_y)**2\n"
    "        weights = np.exp(-dists_sq/(2*sigma**2))\n"
    "        weights /= weights.sum()\n"
    "        cx = float(np.sum(weights*xs))\n"
    "        cy = float(np.sum(weights*ys))\n"
    "        if len(players) >= 4:\n"
    "            return cx, cy, 'players'\n"
    "        else:\n"
    "            tx = 0.2 * cx + 0.8 * (OUT_W/2)\n"
    "            ty = 0.2 * cy + 0.8 * (OUT_H/2)\n"
    "            return tx, ty, 'few_players'\n"
    "    elif len(players) == 1:\n"
    "        tx = 0.05 * players[0][0] + 0.95 * (OUT_W/2)\n"
    "        ty = 0.05 * players[0][1] + 0.95 * (OUT_H/2)\n"
    "        return tx, ty, 'single_player'\n"
    "\n"
    "    return OUT_W / 2, OUT_H / 2, 'centre'"
)
assert old in src, "player centroid fallback not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 5 OK: DBSCAN cluster replaces weighted centroid as primary")
PYEOF

# ── CHANGE 6: Also update kalman_blend section to use DBSCAN ─────
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = (
    "        if len(players) >= 4:\n"
    "            xs = np.array([p[0] for p in players])\n"
    "            ys = np.array([p[1] for p in players])\n"
    "            med_x, med_y = float(np.median(xs)), float(np.median(ys))\n"
    "            sigma = OUT_W * 0.25\n"
    "            w = np.exp(-((xs-med_x)**2+(ys-med_y)**2)/(2*sigma**2))\n"
    "            w /= w.sum()\n"
    "            pcx, pcy = float(np.sum(w*xs)), float(np.sum(w*ys))\n"
    "            tx = kalman_weight * px + (1-kalman_weight) * pcx\n"
    "            ty = kalman_weight * py + (1-kalman_weight) * pcy\n"
    "            return tx, ty, 'kalman_blend'\n"
    "        return px, py, 'kalman'"
)
new = (
    "        # Blend Kalman prediction with DBSCAN cluster (or centroid)\n"
    "        cluster = find_action_cluster(players, cam_yaw, e2p_fov) if len(players) >= DBSCAN_MIN_SAMPLES else None\n"
    "        if cluster is not None:\n"
    "            tx = kalman_weight * px + (1-kalman_weight) * cluster[0]\n"
    "            ty = kalman_weight * py + (1-kalman_weight) * cluster[1]\n"
    "            return tx, ty, 'kalman_blend'\n"
    "        elif len(players) >= 4:\n"
    "            xs = np.array([p[0] for p in players])\n"
    "            ys = np.array([p[1] for p in players])\n"
    "            med_x, med_y = float(np.median(xs)), float(np.median(ys))\n"
    "            sigma = OUT_W * 0.25\n"
    "            w = np.exp(-((xs-med_x)**2+(ys-med_y)**2)/(2*sigma**2))\n"
    "            w /= w.sum()\n"
    "            pcx, pcy = float(np.sum(w*xs)), float(np.sum(w*ys))\n"
    "            tx = kalman_weight * px + (1-kalman_weight) * pcx\n"
    "            ty = kalman_weight * py + (1-kalman_weight) * pcy\n"
    "            return tx, ty, 'kalman_blend'\n"
    "        return px, py, 'kalman'"
)
assert old in src, "kalman_blend section not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 6 OK: kalman_blend uses DBSCAN cluster")
PYEOF

# ── CHANGE 7: Pass cam_yaw and e2p_fov to choose_target call ─────
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("autopan_infer.py")
src = p.read_text()

old = "        tx, ty, mode = choose_target(last_players, last_ball, ball_state)"
new = "        tx, ty, mode = choose_target(last_players, last_ball, ball_state, cam.yaw, e2p_fov)"
assert old in src, "choose_target call not found"
p.write_text(src.replace(old, new, 1))
print("CHANGE 7 OK: cam.yaw and e2p_fov passed to choose_target")
PYEOF

echo ""
echo "Verifying syntax..."
python3 -c "import ast; ast.parse(open('autopan_infer.py').read()); print('Syntax OK')"
echo ""
echo "Done! Test with:"
echo "  python3 autopan_infer.py --insv ... --device mps --debug"

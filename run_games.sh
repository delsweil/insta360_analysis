#!/bin/bash
# run_games.sh — the tracking checks on several games, results saved to text files.
#   cd ~/Library/CloudStorage/Dropbox/development/insta360_analysis
#   bash run_games.sh
# Edit BASE and GAMES if your folders are elsewhere. The game already tested keeps the colour-group fix it needs.
BASE=/Users/davidelsweiler/FootballAutopan
GAMES=("A-Jugend_vs_DJK_Eibach" "TSV_Johannes_83_Herren_1" "SG_Weinberg-Moosbach_Mädels" "TSV_Buch_A-Jugend")
OUT=./games_out
mkdir -p "$OUT"
for g in "${GAMES[@]}"; do
  D="$BASE/$g"
  if [ ! -d "$D" ]; then echo "!! $D not found, skipping"; continue; fi
  EXTRA=""
  if [ "$g" = "TSV_Buch_A-Jugend" ]; then EXTRA="--assign 6=B 5=REF"; fi
  echo "=== $g"
  python3 track_stitch.py "$D" $EXTRA --sweep 2>&1 | tee "$OUT/$g.sweep.txt"
  python3 track_stitch.py "$D" $EXTRA 2>&1 | tee "$OUT/$g.parts.txt"
  python3 diagnose_breaks.py "$D" $EXTRA 2>&1 | tee "$OUT/$g.breaks.txt"
done
echo; echo "Saved in $OUT/ . Paste the *.sweep.txt, *.parts.txt and *.breaks.txt files, or just: cat $OUT/*.txt"

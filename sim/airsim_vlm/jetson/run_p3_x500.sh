#!/usr/bin/env bash
# Runs ON THIS PC (step 8, P2 + P3, 10 Nov 2026). Needs the X500 rig booked (robotq) and BOTH locks held by the caller:
# ~/vlm_research/.x500.lock and ~/.pixhawk.lock. No motion, no PX4, no Pixhawk access: Python on the Jetson only.
#  1. rsync the bundle to x500:~/vlm_damage_p3/  2. setup venv (offline)  3. start the VLM bench (val, test, city, load)
#  4. run the 360-episode netsim batch on 5 cores while the VLM loops on city frames  5. STOP the VLM, fetch, md5-check.
set -uo pipefail
BUNDLE="${BUNDLE:?set BUNDLE to the local bundle folder}"; HOST="${HOST:-x500}"; REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"; LOCAL="$REPO/results/vlm_damage/jetson/$TS"; mkdir -p "$LOCAL"
exec > >(tee -a "$LOCAL/run_p3_x500.log") 2>&1
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST")
say() { echo "$(date -Is) [p3] $*"; }; fail() { say "FAILED: $*"; exit 1; }
"${SSH[@]}" 'uname -a; nvpmodel -q 2>/dev/null | head -2; free -m | head -2; df -h ~ | tail -1' || fail "ssh"
rsync -a --partial "$BUNDLE/" "$HOST:vlm_damage_p3/" || fail "rsync bundle"
"${SSH[@]}" 'cd ~/vlm_damage_p3 && md5sum -c --quiet model.md5' || fail "model md5 on the Jetson"
"${SSH[@]}" 'bash ~/vlm_damage_p3/code/setup_jetson.sh' || fail "setup"
R="vlm_damage_p3/out/$TS"
# Only the python command is put in the background ({ ... & }); backgrounding the whole && list kept ssh open until the
# bench ended, so the planner batch never started (found in the 10 Nov 2026 run; the batch was started by hand then).
"${SSH[@]}" "mkdir -p ~/$R && cd ~/vlm_damage_p3 && { setsid nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python code/bench_vlm.py \
  --model model --data data --frames frames --out ~/$R > ~/$R/bench.log 2>&1 < /dev/null & }" || fail "start bench"
say "bench started; waiting for the city phase before starting the planner batch"
until "${SSH[@]}" "grep -q '^city ' ~/$R/bench.log || grep -q Traceback ~/$R/bench.log"; do sleep 60; done
"${SSH[@]}" "grep -q Traceback ~/$R/bench.log" && { "${SSH[@]}" "tail -30 ~/$R/bench.log"; fail "bench crashed"; }
say "VLM now in the load loop; starting netsim batch (5 workers)"
"${SSH[@]}" "cd ~/vlm_damage_p3 && venv/bin/python code/step8_netsim_batch.py --out ~/$R/step8_netsim.jsonl --workers 5 --netsim ~/vlm_damage_p3/netsim > ~/$R/batch.log 2>&1" \
  || say "batch exited non-zero (see batch.log)"
"${SSH[@]}" "touch ~/$R/STOP"
until "${SSH[@]}" "test -f ~/$R/summary.json || grep -q Traceback ~/$R/bench.log"; do sleep 20; done
"${SSH[@]}" "cd ~/$R && find . -type f ! -name md5.txt -exec md5sum {} + > md5.txt"
rsync -a "$HOST:$R/" "$LOCAL/" || fail "fetch"
(cd "$LOCAL" && md5sum -c --quiet md5.txt) && say "md5 OK $(wc -l < "$LOCAL/md5.txt") files" || fail "md5 mismatch"
cp "$LOCAL/step8_netsim.jsonl" "$REPO/results/vlm_damage/step8_netsim.jsonl"
say "DONE: $LOCAL"

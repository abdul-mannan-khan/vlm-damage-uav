#!/bin/bash
# Batch of closed-loop missions on the PX4 VM (inside the 'px4' container), resumable after a spot reclaim.
# Plan file: one mission per line, "NAME N ARGS..." (scripts/airsim_fleet_plan.py writes it). For every mission whose
# runs/NAME/result.json does not exist: stop PX4, reset AirSim (users removed, vehicles to spawn), start N PX4
# instances, fly the mission in free-run (sim-clock pacing), retry once on failure. A fleet-size change needs the
# map restarted with N vehicles first (vm_test.sh map N from the laptop); the batch stops and says so.
#   bash run_fleet_batch.sh plan_city_n3.txt
set -u
PLAN=$1; WIN_IP=${WIN_IP:-10.142.0.5}; HERE=$(cd "$(dirname "$0")" && pwd); RUNS=/work/maqac/runs
STACK="bash $HERE/px4_stack.sh"
while read -r NAME N ARGS; do
  [ -z "${NAME:-}" ] && continue; case "$NAME" in \#*) continue;; esac
  [ -f $RUNS/$NAME/result.json ] && { echo "skip $NAME (done)"; continue; }
  for try in 1 2; do
    echo "$(date +%T) $NAME try $try"
    $STACK stop >/dev/null
    python3 $HERE/airsim_reset.py --ip $WIN_IP || { echo "AirSim not reachable"; exit 1; }
    WIN_IP=$WIN_IP $STACK start $N || { echo "PX4 start failed (map started with $N vehicles?)"; exit 2; }
    mkdir -p $RUNS/$NAME
    (cd $HERE && timeout 3600 python3 run_maqac_airsim.py --ip $WIN_IP --n $N --free-run --out $RUNS/$NAME $ARGS \
        > $RUNS/$NAME/stdout.log 2>&1)
    [ -f $RUNS/$NAME/result.json ] && break
    echo "$(date +%T) $NAME failed: $(tail -2 $RUNS/$NAME/stdout.log)"
  done
  date +%s > $RUNS/heartbeat
done < "$PLAN"
$STACK stop >/dev/null
echo "batch $PLAN finished"

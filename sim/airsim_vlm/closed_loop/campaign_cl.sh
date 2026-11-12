#!/bin/bash
# Step 14, runs ON the Vast box (image abdulmannan617/airsim-city-wine:1.8.1-wine11-dxvk3.1.1). Adapted from
# sim/airsim_vlm/box/campaign.sh (unchanged there): CityEnviron (image's city-start) + PX4 SITL (snapshot px4_stack.sh) +
# the closed-loop driver run_closed_loop_airsim.py; per mission: map (re)start when needed, PX4 stop, scene reset, PX4
# start, mission with a time limit, up to 3 tries; resumable (skips missions with result.json). Map restart every
# RESTART_EVERY missions (step 5: AirSim 'bad allocation' in simGetImages after 7 missions without a restart).
# No live VLM: verdicts come from the shipped table (no model, no images, no torch on the box).
#   bash campaign_cl.sh plan.txt [plan2.txt ...]      log: /work/vlm/driver.log ; writes /work/vlm/runs/CAMPAIGN_DONE
set -u
V=/work/vlm; M=$V/airsim/maqac_px4; CL=$V/closed_loop; RUNS=$V/runs; LOG=$V/driver.log; CLOCK=${CLOCK:-0.7}
RESTART_EVERY=${RESTART_EVERY:-6}
mkdir -p $RUNS
export PX4_DIR=/opt/PX4-Autopilot RUN=/work/px4_run WIN_IP=127.0.0.1 CITY_LOG=$V/city.log
log() { echo "$(date -u +%F_%T) $*" >> $LOG; }
alive() { timeout 25 python3 -c "import airsim; c=airsim.MultirotorClient(ip='127.0.0.1',timeout_value=15); c.confirmConnection(); c.getMultirotorState(vehicle_name='UAV0')" >/dev/null 2>&1; }
map_start() {
  local N=$1
  python3 $V/airsim/make_settings_city.py --n $N --win-ip 127.0.0.1 --px4-ip 127.0.0.1 --clock-speed $CLOCK --fov 60 > $V/settings_$N.json
  city-start $V/settings_$N.json >> $LOG 2>&1
}
[ -f $V/buildings.json ] || { log "no buildings.json"; exit 4; }
[ -f $V/verdicts_step14.json ] || { log "no verdict table"; exit 4; }
N_MAP=0; FAILS=0; SINCE=0
for PLAN in "$@"; do
  log "plan $PLAN"
  while read -r NAME N ARGS <&3; do
    case "$NAME" in ''|'#'*) continue;; esac
    [ -f $RUNS/$NAME/result.json ] && { log "skip $NAME"; continue; }
    LIM=$(python3 -c "print(int(2*300/$CLOCK+900))")
    for try in 1 2 3; do
      [ $SINCE -ge $RESTART_EVERY ] && { log "planned map restart after $SINCE missions"; N_MAP=0; SINCE=0; }
      if [ "$N_MAP" != "$N" ] || ! alive; then
        log "starting map with $N vehicles (clock $CLOCK)"; bash $M/px4_stack.sh stop >/dev/null; city-stop >/dev/null 2>&1
        if ! map_start $N; then log "AirSim did not come up"; FAILS=$((FAILS+1)); N_MAP=0
          [ $FAILS -ge 4 ] && { log "4 map failures in a row, giving up"; touch $RUNS/CAMPAIGN_DONE; exit 3; }; continue; fi
        FAILS=0; N_MAP=$N; SINCE=0
      fi
      bash $M/px4_stack.sh stop >/dev/null
      python3 $M/airsim_reset.py --ip 127.0.0.1 >/dev/null 2>&1 && bash $M/px4_stack.sh start $N > $V/px4_start.log 2>&1 < /dev/null
      log "$NAME try $try (limit ${LIM}s): $(tail -1 $V/px4_start.log)"
      rm -rf $RUNS/$NAME; mkdir -p $RUNS/$NAME
      (cd $CL && timeout $LIM python3 run_closed_loop_airsim.py --ip 127.0.0.1 --n $N --free-run --out $RUNS/$NAME $ARGS > $RUNS/$NAME/stdout.log 2>&1 < /dev/null)
      SINCE=$((SINCE+1))
      if [ -f $RUNS/$NAME/result.json ]; then log "$NAME done"; break; fi
      log "$NAME failed: $(tail -1 $RUNS/$NAME/stdout.log)"; N_MAP=0
    done
  done 3< <(grep -v "^#" "$PLAN")
  log "plan $PLAN finished"
done
bash $M/px4_stack.sh stop >/dev/null; city-stop >/dev/null
touch $RUNS/CAMPAIGN_DONE; log "campaign finished"

#!/bin/bash
# MAQAC city HIL with AirSim in the loop (4 Nov 2026, robotq run maqac-x500-hil-city), on the Jetson (~/maqac_hil).
# UAV 0: Jetson stack -> W2's p27_router.py -> Pixhawk 6C (PX4 HITL) <- physics + sensors from AirSim CityEnviron on the
# laptop (TCP 4560; laptop side: sim/hil/airsim_city/run_city_hitl.sh under resman). UAVs 1..n-1 kinematic.
# Protocol as hil_campaign.sh (full backup, HITL set-up, shakedown + CONTROL, missions, restore, read-back to 0 differing)
# with the AirSim set-up from W2/ThinkFly (~/lit_results/x500_airsim_hitl_bridge/README.md):
#   SYS_HITL=1, COM_RC_IN_MODE=4, reboot, HIL_ACT_FUNC1..4=101..104, reboot; SYS_AUTOSTART stays at the board's value;
#   by-id serial port (can come back as ttyACM1); wait 8 s after the last parameter write before any reboot;
#   restore: HIL_ACT_FUNC* = 0 first, then every value that differs from the backup (covers CAL_ACC2_ID, CAL_GYRO2_ID,
#   EKF2_MAG_DECL, GPS_1_CONFIG, which entering HITL changes).
# Hand-over (user rule 5 Nov 2026): handover.py secures (force-disarm if armed, Hold) before every reboot, checks that
# time_boot_ms reset, and at the end checks disarmed + system_status != 8 and writes HANDOVER_REPORT.txt.
# Start (only through robotq, both locks; AirSim must already be up on the laptop):
#   flock -n ~/.pixhawk.lock bash hil_campaign_city.sh
# CONTROL file after the shakedown: GO / REDO / ABORT; no answer in 3 h -> restore and stop.
cd ~/maqac_hil || exit 1
export PX4_DEV=/dev/serial/by-id/usb-Auterion_PX4_FMU_v6C.x_0-if00
DEV="--dev $PX4_DEV"
OUTDIR=${OUTDIR:-results/hil_airsim}
mkdir -p params $OUTDIR
L=~/maqac_hil/campaign_city.log
log(){ echo "$(date -u +%F_%T) $*" | tee -a $L; }
TAG="MAQAC city HIL, AirSim (maqac-vehcom-a5)"
B=params/params_before_maqac_city_$(date -u +%m%d_%H%M).params
REF=params/reference_paperA_2026-11-02.params
SEEDS=${SEEDS:-"20000 20001 20002 20003 20004"}
METHODS="maqac_seq maqac rag"
RUN="--router --L 600 --city survey/heightmap.npz"
RESTORED=0; ROUTER=

for f in ~/.pixhawk.owner ~/vlm_research/.x500.owner; do
  if grep -qi '^busy' $f 2>/dev/null; then log "owner file $f says busy: $(cat $f); not starting"; exit 2; fi
done
[ -e $PX4_DEV ] || { log "no Pixhawk at $PX4_DEV; not starting"; exit 2; }
if fuser $PX4_DEV >/dev/null 2>&1; then log "serial port in use; not starting"; exit 2; fi
busy="busy: $TAG $(date -u)"; echo "$busy" > ~/.pixhawk.owner; echo "$busy" > ~/vlm_research/.x500.owner
log "start; owner files set busy"

pset(){ timeout 300 python3 param_set.py $DEV "$@" >> $L 2>&1; local rc=$?; sleep 8; return $rc; }   # 8 s: let the write land
# Reboot with the hand-over checks (user rule 5 Nov; handover.py): force-disarm if armed (PX4 refuses reboots while armed),
# Hold, reboot, then require SYSTEM_TIME time_boot_ms to have reset (a returning heartbeat alone is not proof).
# Falls back to the plain reboot if handover.py gets no link.
reboot_fc(){ local w=${1:-45} r rc
             r=$(timeout $((w + 150)) python3 handover.py --dev $PX4_DEV --wait $w reboot 2>>$L); rc=$?
             log "reboot check rc $rc: $(echo "$r" | grep HANDOVER | head -c 400)"
             if [ $rc = 2 ] || [ -z "$r" ]; then
               timeout 60 python3 param_set.py $DEV --reboot >> $L 2>&1; sleep $w
             fi
             for i in $(seq 1 30); do [ -e $PX4_DEV ] && break; sleep 2; done
             return $rc; }
router_on(){ python3 -u p27_router.py run --host 192.168.55.100 --port 4560 --stats 10 >> router.log 2>&1 & ROUTER=$!; sleep 5; }
router_off(){ [ -n "$ROUTER" ] && { kill $ROUTER 2>/dev/null; wait $ROUTER 2>/dev/null; }; ROUTER=; sleep 2; }

restore(){
  [ $RESTORED = 1 ] && return; RESTORED=1
  [ -n "$MPID" ] && kill $MPID 2>/dev/null; sleep 2
  router_off
  if [ "${HAVE_BACKUP:-0}" = 1 ]; then
    log "restore"
    log "secure before restore: $(timeout 60 python3 handover.py --dev $PX4_DEV secure 2>>$L | head -c 300)"
    reboot_fc 40
    pset HIL_ACT_FUNC1=0 HIL_ACT_FUNC2=0 HIL_ACT_FUNC3=0 HIL_ACT_FUNC4=0
    timeout 900 python3 param_set.py $DEV --restore $B >> $L 2>&1; sleep 8
    reboot_fc 60
    for pass in 1 2 3 4; do
      A=params/params_after_maqac_city_$pass.params
      timeout 1500 python3 backup_by_name.py $B $A >> $L 2>&1
      fix=$(python3 - "$B" "$A" <<'PY'
import sys
def rd(f):
    d = {}
    for l in open(f):
        p = l.split()
        if len(p) >= 4 and not l.startswith('#'):
            d[p[2]] = p[3]
    return d
a, b = rd(sys.argv[1]), rd(sys.argv[2])
def eq(x, y):
    try: return abs(float(x) - float(y)) < 1e-6 * max(1, abs(float(x)))
    except ValueError: return x == y
COUNTERS = {'_HASH_CHECK', 'COM_FLIGHT_UUID', 'LND_FLIGHT_T_LO', 'LND_FLIGHT_T_HI'}
miss = [k for k in a if k not in COUNTERS and k not in b]
dif = [f'{k}={a[k]}' for k in a if k not in COUNTERS and k in b and not eq(a[k], b[k])]
open(sys.argv[2] + '.diff', 'w').write(f'differing {len(dif)}: {" ".join(dif)}\nmissing {len(miss)}: {" ".join(miss)}\n')
print(' '.join(dif))
PY
)
      log "read-back pass $pass: $(head -c 400 $A.diff | tr '\n' ' ')"
      [ -z "$fix" ] && { log "read-back pass $pass: 0 differing parameters (flight counters excluded)"; break; }
      pset $fix; reboot_fc 60
    done
  else
    log "no backup was taken; nothing was changed"
  fi
  # final hand-over check: disarmed, system_status != 8 (flight termination); report what is left non-default
  fin=$(timeout 60 python3 handover.py --dev $PX4_DEV final 2>>$L); frc=$?
  last=$(ls -t params/params_after_maqac_city_*.params.diff 2>/dev/null | head -1)
  { echo "MAQAC city HIL hand-over $(date -u +%F_%T)"; echo "final check rc $frc: $fin"
    echo "last read-back ($last): $( [ -n "$last" ] && tr '\n' ' ' < $last)"
    echo "excluded from the read-back (flight counters): COM_FLIGHT_UUID LND_FLIGHT_T_LO LND_FLIGHT_T_HI"
    echo "geofence: left as found (this script writes no GF_*)"; } > HANDOVER_REPORT.txt
  log "HANDOVER $( [ $frc = 0 ] && echo OK || echo "WARNING: check failed, tell the next session and it-services-9b" ): $(tr '\n' ' ' < HANDOVER_REPORT.txt | head -c 600)"
  idle="idle (last: MAQAC city HIL $(date -u))"; echo "$idle" > ~/.pixhawk.owner; echo "$idle" > ~/vlm_research/.x500.owner
  log "done; owner files idle"
}
trap restore EXIT
trap 'log "signal: stopping"; exit 1' INT TERM HUP

# ---- backup (full list, else by name)
timeout 300 python3 param_dump.py $B $DEV >> $L 2>&1
n=$(grep -c "^1" $B 2>/dev/null); log "backup lines $n"
if [ "${n:-0}" -le 1000 ]; then
  log "full list failed: backup by name"; timeout 1500 python3 backup_by_name.py $REF $B >> $L 2>&1
  n=$(grep -c "^1" $B 2>/dev/null); log "backup lines $n"
fi
[ "${n:-0}" -gt 1000 ] || { log "NO BACKUP: stopping before any change"; exit 1; }
HAVE_BACKUP=1
grep -E " (SYS_HITL|SYS_AUTOSTART|PWM_MAIN_MIN1|PWM_MAIN_MAX1|COM_RC_IN_MODE) " $B | awk '{print $3"="$4}' | tr '\n' ' ' | sed 's/^/baseline: /' | tee -a $L; echo
cp $B params/params_before_maqac_city_$(date -u +%F_%H%M).params

# ---- HITL set-up for AirSim (W2 P27 steps) + the same speed limits as the first HIL
# Geofence: NONE (user's rule, 4 Nov ~19:30 UTC: "NO fences" for X500 HIL; W2 turned the board's fence off once and robotq job
# x500-fence-restore restores 100 m / 60 m / RTL after all HIL sessions). This script writes NO GF_* parameter; restore()
# goes back to this run's own backup $B, so the fence stays as found. The values found are logged (stated in the paper).
grep -E " (GF_ACTION|GF_MAX_HOR_DIST|GF_MAX_VER_DIST) " $B | awk '{print $3"="$4}' | tr '\n' ' ' | sed 's/^/geofence found: /' | tee -a $L; echo
pset SYS_HITL=1 COM_RC_IN_MODE=4 MPC_XY_VEL_MAX=20 MPC_Z_VEL_MAX_DN=2.0 MPC_Z_VEL_MAX_UP=3.0
reboot_fc 45
pset HIL_ACT_FUNC1=101 HIL_ACT_FUNC2=102 HIL_ACT_FUNC3=103 HIL_ACT_FUNC4=104

fly(){   # reboot, router up, run, router down; one retry if the HIL link was not ready
  for try in 1 2; do
    log "reboot before: $*"; reboot_fc 45
    router_on
    python3 run_maqac_hil.py "$@" $RUN --out $OUTDIR >> ~/maqac_hil/hil_city_run.log 2>&1 & MPID=$!; wait $MPID; rc=$?; MPID=
    router_off
    log "run exit $rc: $* (router: $(tail -1 router.log | head -c 200))"
    [ $rc = 0 ] && return 0
    grep -q "HIL not ready" <(tail -5 ~/maqac_hil/hil_city_run.log) || return $rc
  done
}

dec=GO
while [ "${SKIP_SHAKEDOWN:-0}" != 1 ]; do
  rm -f CONTROL
  fly --T 60 --methods maqac_seq --seeds 19999 --tag shakedown_
  log "shakedown finished; waiting for CONTROL (GO / REDO / ABORT)"
  dec=""
  for i in $(seq 1 1080); do [ -f CONTROL ] && { dec=$(cat CONTROL); break; }; sleep 10; done
  log "CONTROL: ${dec:-timeout}"
  [ "$dec" = "REDO" ] && continue
  break
done
[ "$dec" = "GO" ] || exit 0

# ---- design: urban, 3 UAVs, 600 m CityEnviron area, T = 300 s, seeds 20000-20004 x (maqac_seq, maqac, rag)
# One call for all seeds: the runner lands UAV 0 on its AirSim pad only at the end, and the board must never be
# rebooted with the AirSim vehicle in the air (its motors would stop and it would fall in the simulation).
fly --T 300 --methods $METHODS --seeds $SEEDS
log "missions finished: $(grep -c RESULT ~/maqac_hil/hil_city_run.log) RESULT lines"
exit 0

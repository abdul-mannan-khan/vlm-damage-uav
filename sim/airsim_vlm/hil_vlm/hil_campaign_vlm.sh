#!/bin/bash
# Closed-loop VLM HIL campaign (VLM paper, IEEE TVT), on the Jetson in ~/vlm_hil. PREPARED 12 Nov 2026, NOT YET RUN.
# Adapted from maqac-vehcom sim/hil/hil_campaign_city.sh (VehCom set-up 2, 15 missions on 4-5 Nov), same safety protocol:
#   both locks + owner files -> full parameter backup (else by name) -> HITL set-up for AirSim (SYS_HITL=1, COM_RC_IN_MODE=4,
#   reboot, HIL_ACT_FUNC1..4=101..104) + limits for this experiment -> reboot -> 60 s shakedown -> wait for CONTROL GO ->
#   missions -> restore (HIL_ACT_FUNC*=0 first, then the backup) -> reboot -> read back by name until 0 differences ->
#   standard limits GF_MAX_HOR_DIST 100 / GF_MAX_VER_DIST 60 / GF_ACTION 3 and SYS_HITL 0 checked (written if needed) ->
#   hand-over checks -> owner files idle.
# Changes from VehCom: own folder/logs/backups (~/vlm_hil), the VLM runner (run_vlm_hil.py) and its memory pre-check,
# tegrastats logged for the whole campaign, geofence handled by the 11 Nov 2026 standard practice
# (~/lit_results/x500_airsim_hitl_bridge/FENCE_STATE.md: set the limits the experiment needs, restore the same-day backup,
# make sure 100/60/3 is left, read back), final SYS_HITL 0 check.
#
# HUMAN DECISIONS (marked "DECIDE" below; nothing here runs until they are settled):
#   D1 book the rig in robotq (run vlm-x500-hil-closedloop) and start AirSim on the laptop first (laptop is reserved for HIL)
#   D2 VLM coupling to the planner (VLM_PLANNER: "off prior" = both arms, PLAN.md) and the seeds / mission count
#   D3 single seed int8 (default) vs 4-seed ensemble (needs a 4-seed onboard calibration that does not exist yet)
#   D4 the shakedown verdict: echo GO / REDO / ABORT > ~/vlm_hil/CONTROL
#   D5 geofence during the missions (default: off, as the 11 Nov standard practice allows; restored at the end)
# Start (only through robotq, both locks; AirSim already up on the laptop with settings_hitl_city_vlm.json):
#   cd ~/vlm_hil && setsid nohup flock -n ~/.pixhawk.lock bash hil_campaign_vlm.sh > campaign.out 2>&1 &
cd ~/vlm_hil || exit 1
export PX4_DEV=/dev/serial/by-id/usb-Auterion_PX4_FMU_v6C.x_0-if00
DEV="--dev $PX4_DEV"
OUTDIR=${OUTDIR:-results/hil_vlm}
mkdir -p params $OUTDIR
L=~/vlm_hil/campaign_vlm.log
log(){ echo "$(date -u +%F_%T) $*" | tee -a $L; }
TAG="VLM closed-loop HIL, AirSim city (ras-multi-agent-6g-uav VLM paper)"
B=params/params_before_vlm_hil_$(date -u +%m%d_%H%M).params
REF=${REF:-params/reference_paperA_2026-11-02.params}     # names/types for a by-name backup (copy from ~/maqac_hil/params)
SEEDS=${SEEDS:-"20000 20001 20002 20003 20004 20005"}     # DECIDE D2 (PLAN.md: 6 seeds x 2 arms = 12 missions ~ 1.6 h)
VLM_PLANNER=${VLM_PLANNER:-"off prior"}                   # DECIDE D2
VLM_MODEL=${VLM_MODEL:-int8}                              # DECIDE D3 (int8x4 needs CALIB=<4-seed onboard calibration>)
CALIB=${CALIB:-int8_v2_calibration.npz}
FRAMES=${FRAMES:-airsim}                                  # airsim (RPC 192.168.55.100:41451) or relay (frame_relay.py on the laptop)
FENCE_OFF=${FENCE_OFF:-1}                                 # DECIDE D5
MIN_FREE_MB=${MIN_FREE_MB:-6200}                          # VLM ~4.7 GB + runner + HIL server; below this: do not start
RUN="--router --L 600 --city heightmap.npz --vlm-model $VLM_MODEL --vlm-frames $FRAMES --calib $CALIB --model-dir $HOME/vlm_damage_p3/model --ens-adapters $HOME/vlm_damage_p3/adapters --vlm-python $HOME/vlm_damage_p3/venv/bin/python --save-frames"
RESTORED=0; ROUTER=; TEGRA=

for f in ~/.pixhawk.owner ~/vlm_research/.x500.owner; do
  if grep -qi '^busy' $f 2>/dev/null; then log "owner file $f says busy: $(cat $f); not starting"; exit 2; fi
done
[ -e $PX4_DEV ] || { log "no Pixhawk at $PX4_DEV; not starting"; exit 2; }
if fuser $PX4_DEV >/dev/null 2>&1; then log "serial port in use; not starting"; exit 2; fi
free_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
  log "only $free_mb MB available (< $MIN_FREE_MB): another process holds memory (LLM server? ps below); not starting"
  ps -eo pid,rss,comm --sort=-rss | head -8 | tee -a $L; exit 2
fi
for f in run_vlm_hil.py vlm_worker.py bench_vlm.py hil_link.py city.py heightmap.npz buildings.json $CALIB $HOME/vlm_damage_p3/model/config.json $HOME/vlm_damage_p3/venv/bin/python; do
  [ -e "$f" ] || { log "missing $f; not starting"; exit 2; }
done
busy="busy: $TAG $(date -u)"; echo "$busy" > ~/.pixhawk.owner; echo "$busy" > ~/vlm_research/.x500.owner
log "start; owner files set busy; MemAvailable $free_mb MB"
( command -v tegrastats >/dev/null && exec tegrastats --interval 1000 --logfile $OUTDIR/tegrastats_campaign.log ) & TEGRA=$!

pset(){ timeout 300 python3 param_set.py $DEV "$@" >> $L 2>&1; local rc=$?; sleep 8; return $rc; }   # 8 s: let the write land
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

readback(){   # $1 backup, $2 after-file; writes $2.diff; prints NAME=VALUE pairs (backup values) that differ
python3 - "$1" "$2" <<'PY'
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
STD = {'GF_MAX_HOR_DIST': 100, 'GF_MAX_VER_DIST': 60, 'GF_ACTION': 3}     # standard limits left at hand-over (11 Nov rule)
for k, v in STD.items():                                                  # expected = backup, except the standard limits
    a[k] = str(v)
miss = [k for k in a if k not in COUNTERS and k not in b]
dif = [f'{k}={a[k]}' for k in a if k not in COUNTERS and k in b and not eq(a[k], b[k])]
open(sys.argv[2] + '.diff', 'w').write(f'differing {len(dif)}: {" ".join(dif)}\nmissing {len(miss)}: {" ".join(miss)}\n')
print(' '.join(dif))
PY
}

restore(){
  [ $RESTORED = 1 ] && return; RESTORED=1
  [ -n "$MPID" ] && kill $MPID 2>/dev/null; sleep 2
  pkill -f run_vlm_hil.py 2>/dev/null; sleep 2           # the VLM worker is a child of the runner; make sure both are gone
  router_off
  if [ "${HAVE_BACKUP:-0}" = 1 ]; then
    log "restore"
    log "secure before restore: $(timeout 60 python3 handover.py --dev $PX4_DEV secure 2>>$L | head -c 300)"
    reboot_fc 40
    pset HIL_ACT_FUNC1=0 HIL_ACT_FUNC2=0 HIL_ACT_FUNC3=0 HIL_ACT_FUNC4=0
    timeout 900 python3 param_set.py $DEV --restore $B >> $L 2>&1; sleep 8
    # standard limits at hand-over (FENCE_STATE.md, 11 Nov 21:00): write them even if the same-day backup had 0/0/0
    pset GF_MAX_HOR_DIST=100 GF_MAX_VER_DIST=60 GF_ACTION=3 SYS_HITL=0
    reboot_fc 60
    for pass in 1 2 3 4; do
      A=params/params_after_vlm_hil_$pass.params
      timeout 1500 python3 backup_by_name.py $B $A >> $L 2>&1
      fix=$(readback $B $A)
      log "read-back pass $pass: $(head -c 400 $A.diff | tr '\n' ' ')"
      [ -z "$fix" ] && { log "read-back pass $pass: 0 differing parameters (flight counters excluded; GF_* = 100/60/3 expected)"; break; }
      pset $fix; reboot_fc 60
    done
    # explicit final check of the hand-over values (SYS_HITL 0, GF 100/60/3) from the last read-back file
    chk=$(grep -P "\t(SYS_HITL|GF_MAX_HOR_DIST|GF_MAX_VER_DIST|GF_ACTION|SYS_AUTOSTART|HIL_ACT_FUNC1)\t" $A | awk -F"\t" '{print $3"="$4}' | tr '\n' ' ')   # 12 Nov: files are tab-separated
    log "hand-over values: $chk"
    echo "$chk" | grep -q "SYS_HITL=0 " || log "WARNING: SYS_HITL is not 0 after restore: tell the user and it-services-75"
  else
    log "no backup was taken; nothing was changed"
  fi
  fin=$(timeout 60 python3 handover.py --dev $PX4_DEV final 2>>$L); frc=$?
  last=$(ls -t params/params_after_vlm_hil_*.params.diff 2>/dev/null | head -1)
  { echo "VLM closed-loop HIL hand-over $(date -u +%F_%T)"; echo "final check rc $frc: $fin"
    echo "last read-back ($last): $( [ -n "$last" ] && tr '\n' ' ' < $last)"
    echo "excluded from the read-back (flight counters): COM_FLIGHT_UUID LND_FLIGHT_T_LO LND_FLIGHT_T_HI"
    echo "limits during the missions: $( [ "$FENCE_OFF" = 1 ] && echo 'GF_ACTION 0, GF_MAX_HOR_DIST 0, GF_MAX_VER_DIST 0 (off)' || echo 'as found')"
    echo "limits left: GF_MAX_HOR_DIST 100, GF_MAX_VER_DIST 60, GF_ACTION 3 (standard set), SYS_HITL 0"; } > HANDOVER_REPORT.txt
  log "HANDOVER $( [ $frc = 0 ] && echo OK || echo "WARNING: check failed, tell the next session and it-services-75" ): $(tr '\n' ' ' < HANDOVER_REPORT.txt | head -c 700)"
  [ -n "$TEGRA" ] && kill $TEGRA 2>/dev/null
  idle="idle (last: VLM closed-loop HIL $(date -u))"; echo "$idle" > ~/.pixhawk.owner; echo "$idle" > ~/vlm_research/.x500.owner
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
grep -E " (SYS_HITL|SYS_AUTOSTART|PWM_MAIN_MIN1|PWM_MAIN_MAX1|COM_RC_IN_MODE|GF_ACTION|GF_MAX_HOR_DIST|GF_MAX_VER_DIST) " $B | awk '{print $3"="$4}' | tr '\n' ' ' | sed 's/^/baseline: /' | tee -a $L; echo
cp $B params/params_before_vlm_hil_$(date -u +%F_%H%M).params

# ---- HITL set-up for AirSim (W2 P27 steps) + the VehCom speed limits; limits for this experiment (DECIDE D5)
LIM=""; [ "$FENCE_OFF" = 1 ] && LIM="GF_ACTION=0 GF_MAX_HOR_DIST=0 GF_MAX_VER_DIST=0"
pset SYS_HITL=1 COM_RC_IN_MODE=4 MPC_XY_VEL_MAX=20 MPC_Z_VEL_MAX_DN=2.0 MPC_Z_VEL_MAX_UP=3.0 $LIM
reboot_fc 45
pset HIL_ACT_FUNC1=101 HIL_ACT_FUNC2=102 HIL_ACT_FUNC3=103 HIL_ACT_FUNC4=104

fly(){   # reboot, router up, run, router down; one retry if the HIL link was not ready
  for try in 1 2; do
    log "reboot before: $*"; reboot_fc 45
    router_on
    python3 -u run_vlm_hil.py "$@" $RUN --out $OUTDIR >> ~/vlm_hil/hil_vlm_run.log 2>&1 & MPID=$!; wait $MPID; rc=$?; MPID=
    router_off
    log "run exit $rc: $* (router: $(tail -1 router.log | head -c 200))"
    [ $rc = 0 ] && return 0
    grep -q "HIL not ready" <(tail -5 ~/vlm_hil/hil_vlm_run.log) || return $rc
  done
}

dec=GO
while [ "${SKIP_SHAKEDOWN:-0}" != 1 ]; do
  rm -f CONTROL
  fly --T 60 --seeds 19999 --vlm-planner prior --tag shakedown_
  log "shakedown finished: $(grep RESULT ~/vlm_hil/hil_vlm_run.log | tail -1 | head -c 900)"
  log "waiting for CONTROL (GO / REDO / ABORT)   <- DECIDE D4 (PLAN.md shakedown checklist)"
  dec=""
  for i in $(seq 1 1080); do [ -f CONTROL ] && { dec=$(cat CONTROL); break; }; sleep 10; done
  log "CONTROL: ${dec:-timeout}"
  [ "$dec" = "REDO" ] && continue
  break
done
[ "$dec" = "GO" ] || exit 0

# ---- design (PLAN.md): disaster, 2 UAVs, 600 m CityEnviron area, T = 300 s, MAQAC-S + chain rule,
# seeds x VLM_PLANNER arms, complete seeds first. One call for all missions: UAV 0 lands on its AirSim pad only at the end,
# and the board must never be rebooted with the AirSim vehicle in the air.
fly --T 300 --seeds $SEEDS --vlm-planner $VLM_PLANNER
log "missions finished: $(grep -c RESULT ~/vlm_hil/hil_vlm_run.log) RESULT lines"
exit 0

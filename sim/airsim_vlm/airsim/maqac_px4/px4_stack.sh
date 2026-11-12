#!/bin/bash
# Start / stop N PX4 SITL instances (v1.14.3, none_iris, lockstep) that connect to AirSim on the Windows VM.
# Runs inside the 'px4' container on maqac-px4 (docker run --network host), e.g.
#   docker exec px4 bash /work/maqac/sim/airsim/maqac_px4/px4_stack.sh start 4
#   docker exec px4 bash /work/maqac/sim/airsim/maqac_px4/px4_stack.sh status 4
#   docker exec px4 bash /work/maqac/sim/airsim/maqac_px4/px4_stack.sh stop
# Start the city map on the Windows VM first (AirSim is the TCP server on WIN_IP:4560+i; PX4 instance i connects).
# Instance i: working dir $RUN/px4_$i (fresh parameters every start), parameters from PX4_PARAMS, and an extra MAVLink
# instance for the bridge (custom mode with selected streams, UDP 14700+i -> 127.0.0.1:14800+i).
set -u
PX4_DIR=${PX4_DIR:-/opt/PX4-Autopilot}
BUILD=$PX4_DIR/build/px4_sitl_default
RUN=${RUN:-/work/px4_run}
WIN_IP=${WIN_IP:-10.142.0.5}
# NAV/COM: no RC, no data-link failsafe, offboard loss after 5 s (the bridge re-sends setpoints at 5 Hz of sim time).
# MPC_XY_VEL_MAX 15 > the network model's 14 m/s cruise speed, so PX4 does not cap the reference.
# MPC_Z_VEL_MAX_UP/DN 5/2.5 m/s: faster climbs over towers taller than the flight altitude (bridge climb rule).
# SIM_BAT_DRAIN: the network model keeps its own energy budget (battery swap logic); PX4's simulated battery must not
# trigger its own failsafe first.
PX4_PARAMS="
NAV_RCL_ACT 0
NAV_DLL_ACT 0
COM_RC_IN_MODE 4
COM_RCL_EXCEPT 4
COM_OF_LOSS_T 5.0
COM_OBL_RC_ACT 5
COM_LOW_BAT_ACT 0
COM_DISARM_PRFLT 60.0
MPC_XY_VEL_MAX 15.0
MPC_XY_CRUISE 14.0
MPC_Z_VEL_MAX_UP 5.0
MPC_Z_VEL_MAX_DN 2.5
SIM_BAT_DRAIN 100000.0
SIM_BAT_MIN_PCT 50.0
"

stop_all() {
    pkill -f '[b]in/px4 -i' 2>/dev/null
    sleep 2
    pkill -9 -f '[b]in/px4 -i' 2>/dev/null
    rm -f /tmp/px4_lock-* /tmp/px4-sock-* 2>/dev/null
    true
}

start_px4() {
    local i=$1 dir=$RUN/px4_$1
    rm -rf "$dir"; mkdir -p "$dir"
    (cd "$dir" && PX4_SIM_MODEL=none_iris PX4_SIM_HOSTNAME=$WIN_IP \
        setsid nohup $BUILD/bin/px4 -i $i -d $BUILD/etc > out.log 2>&1 < /dev/null &)
}

configure_px4() {
    local i=$1
    for s in $(seq 1 300); do
        grep -q 'Startup script returned successfully' $RUN/px4_$i/out.log 2>/dev/null && break
        sleep 1
    done
    grep -q 'Startup script returned successfully' $RUN/px4_$i/out.log || \
        { echo "PX4 $i did not boot"; tail -20 $RUN/px4_$i/out.log; return 1; }
    echo "$PX4_PARAMS" | while read -r name val; do
        [ -z "$name" ] && continue
        (cd $RUN/px4_$i && timeout 20 $BUILD/bin/px4-param --instance $i set $name $val) >> $RUN/px4_$i/params.log 2>&1
    done
    (cd $RUN/px4_$i && timeout 20 $BUILD/bin/px4-mavlink --instance $i start -u $((14700 + i)) -o $((14800 + i)) \
        -t 127.0.0.1 -m custom -r 200000) >> $RUN/px4_$i/params.log 2>&1
    for st in "HEARTBEAT 2" "BATTERY_STATUS 1" "EXTENDED_SYS_STATE 2" "LOCAL_POSITION_NED 20" "SYS_STATUS 1" \
              "STATUSTEXT 5"; do
        set -- $st
        (cd $RUN/px4_$i && timeout 20 $BUILD/bin/px4-mavlink --instance $i stream -u $((14700 + i)) -s $1 -r $2) \
            >> $RUN/px4_$i/params.log 2>&1
    done
    echo "PX4 $i configured (bridge link udp $((14700 + i)) -> $((14800 + i)))"
}

case "${1:-}" in
    start)
        N=$2
        mkdir -p $RUN
        stop_all
        [ -x $BUILD/bin/px4 ] || { echo "no PX4 build at $BUILD"; exit 1; }
        (echo > /dev/tcp/$WIN_IP/41451) 2>/dev/null || { echo "AirSim API $WIN_IP:41451 not reachable: start the map first"; exit 1; }
        for i in $(seq 0 $((N - 1))); do start_px4 $i; sleep 1; done
        for i in $(seq 0 $((N - 1))); do configure_px4 $i || exit 1; done
        echo "started $N PX4 instances"
        ;;
    status)
        N=${2:-1}
        for i in $(seq 0 $((N - 1))); do
            if pgrep -f "[b]in/px4 -i $i " >/dev/null; then st=running; else st=stopped; fi
            echo "px4 $i: $st; $(grep -c 'Simulator connected\|Got.*simulator\|lockstep' $RUN/px4_$i/out.log 2>/dev/null) sim lines;" \
                 "last: $(tail -1 $RUN/px4_$i/out.log 2>/dev/null)"
        done
        ;;
    stop)
        stop_all; echo stopped ;;
    *)
        echo "usage: $0 start N | status N | stop"; exit 2 ;;
esac

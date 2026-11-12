#!/bin/bash
# Laptop-side helper for the AirSim (Windows VM) + PX4 (Linux VM) runs. Nothing starts unless called explicitly.
#   vm_test.sh up                 start both VMs                        (~$1.6/h together while running)
#   vm_test.sh push               copy src/netsim + sim/airsim code into the px4 container (/work/maqac)
#   vm_test.sh deps               install the bridge's Python packages in the container
#   vm_test.sh map N              write settings for N PX4 vehicles and start the map (MAP=city default, MAP=mountains)
#   vm_test.sh px4 N              start N PX4 instances in the container (after 'map')
#   vm_test.sh survey             height-map survey (+ orientation check) -> /work/maqac/survey
#   vm_test.sh mission NAME ARGS  run_maqac_airsim.py ARGS in the background -> /work/maqac/runs/NAME
#   vm_test.sh render NAME ARGS   vla_render_pilot.py (map started with SETTINGS_ARGS=--render)
#   vm_test.sh logs NAME          tail the bridge log of a run
#   vm_test.sh pull DIR           copy /work/maqac/{survey,runs} to results/airsim/DIR on the laptop
#   vm_test.sh down               stop PX4, stop both VMs
# Windows VM in another zone: WIN=maqac-airsim-win2 WIN_ZONE=us-east1-d WIN_IP=10.142.0.x vm_test.sh ...
set -eu
export PATH=$HOME/google-cloud-sdk/bin:$PATH
PROJECT=project-ee36cf33-b5c7-49f1-ac0
ZONE=${LNX_ZONE:-us-east1-c}          # zone of the PX4 VM
WIN=${WIN:-maqac-airsim-win}
WIN_ZONE=${WIN_ZONE:-us-east1-c}
LNX=${LNX:-maqac-px4}
WIN_IP=${WIN_IP:-10.142.0.5}
PX4_IP=${PX4_IP:-10.142.0.6}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PS1_TEMPLATE=$HOME/phd_students/swarm-airsim/windows_vm/start_map.ps1
G="--project $PROJECT --zone $ZONE"
GW="--project $PROJECT --zone $WIN_ZONE"
lx() { gcloud compute ssh $LNX $G --command "$*"; }
dx() { lx "sudo docker exec px4 bash -lc '$*'"; }

case "${1:-}" in
    up)
        gcloud compute instances start $WIN $GW
        gcloud compute instances start $LNX $G
        lx "sudo docker start px4 >/dev/null && echo px4 container up" ;;
    push)
        tmp=$(mktemp -d)
        (cd "$REPO" && tar -czf $tmp/maqac.tgz src/netsim/*.py sim/airsim/make_settings_city.py sim/airsim/maqac_px4)
        gcloud compute scp $tmp/maqac.tgz $LNX:/tmp/maqac.tgz $G
        lx "sudo docker exec px4 mkdir -p /work/maqac && sudo docker cp /tmp/maqac.tgz px4:/work/maqac.tgz && \
            sudo docker exec px4 tar -C /work/maqac -xzf /work/maqac.tgz && echo pushed"
        rm -rf $tmp ;;
    deps)
        dx "pip3 install -q numpy scipy pillow pymavlink msgpack-rpc-python && pip3 install -q --no-deps airsim && \
            python3 -c \"import airsim, pymavlink, scipy, PIL; print(\\\"deps ok\\\")\"" ;;
    map)
        N=$2
        python3 "$REPO/sim/airsim/make_settings_city.py" --n $N --win-ip $WIN_IP --px4-ip $PX4_IP ${SETTINGS_ARGS:-} \
            > "$REPO/sim/airsim/settings_city_px4_$N.json"
        B64=$(base64 -w0 "$REPO/sim/airsim/settings_city_px4_$N.json")
        sed -e "s|__MAP__|${MAP:-city}|" -e "s|__SETTINGS_B64__|$B64|" "$PS1_TEMPLATE" | \
            gcloud compute ssh $WIN $GW --command "powershell -Command -" ;;
    px4)
        dx "WIN_IP=$WIN_IP bash /work/maqac/sim/airsim/maqac_px4/px4_stack.sh start $2" ;;
    survey)
        dx "cd /work/maqac/sim/airsim/maqac_px4 && python3 survey_heightmap.py --ip $WIN_IP --out /work/maqac/survey" ;;
    mission)
        NAME=$2; shift 2
        dx "cd /work/maqac/sim/airsim/maqac_px4 && mkdir -p /work/maqac/runs/$NAME && \
            nohup python3 run_maqac_airsim.py --ip $WIN_IP --out /work/maqac/runs/$NAME $* \
            > /work/maqac/runs/$NAME/stdout.log 2>&1 < /dev/null & echo started $NAME" ;;
    render)
        NAME=$2; shift 2
        dx "cd /work/maqac/sim/airsim/maqac_px4 && mkdir -p /work/maqac/runs/$NAME && \
            nohup python3 vla_render_pilot.py --ip $WIN_IP --out /work/maqac/runs/$NAME $* \
            > /work/maqac/runs/$NAME/stdout.log 2>&1 < /dev/null & echo started $NAME" ;;
    logs)
        dx "tail -20 /work/maqac/runs/$2/bridge.log; tail -5 /work/maqac/runs/$2/stdout.log" ;;
    pull)
        DST="$REPO/results/airsim/$2"; mkdir -p "$DST"
        lx "sudo docker exec px4 tar -C /work/maqac -czf /tmp/out.tgz survey runs 2>/dev/null; sudo docker cp px4:/tmp/out.tgz /tmp/out.tgz"
        gcloud compute scp $LNX:/tmp/out.tgz "$DST/out.tgz" $G
        tar -C "$DST" -xzf "$DST/out.tgz" && rm "$DST/out.tgz" && echo "pulled to $DST" ;;
    down)
        lx "sudo docker exec px4 bash /work/maqac/sim/airsim/maqac_px4/px4_stack.sh stop" || true
        gcloud compute instances stop $WIN $GW
        gcloud compute instances stop $LNX $G ;;
    *)
        sed -n 2,13p "$0"; exit 2 ;;
esac

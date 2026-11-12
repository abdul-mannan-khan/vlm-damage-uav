#!/bin/bash
# Step 14 closed-loop flights on ONE Vast GPU box. NOT RUN: needs the user's OK on cost first (cloud rule).
# Adapted from scripts/vast_vlm_replay.sh (create/ready/destroy), sim/airsim_vlm/box/campaign.sh (box side, here
# closed_loop/campaign_cl.sh) and sim/airsim_vlm/box/monitor6.sh (20-min pulls, md5 copy-back, destroy). All unchanged.
# Image abdulmannan617/airsim-city-wine:1.8.1-wine11-dxvk3.1.1 (AirSim 1.8.1 CityEnviron under Wine + PX4 SITL), NVIDIA
# driver >= 595 required (steps 5/6). No VLM on the box (verdict table replay), so ~3 GB GPU memory for AirSim suffices.
# Ships ONLY: netsim, the step-5/6 driver (unchanged), closed_loop/, the AirSim survey heightmap.npz, buildings.json,
# visibility_model_step5.json and the verdict table JSON. No city .npz, no lidar, no RescueNet images, no model, no keys.
# Safety: hard stop (box destroyed after HARD_H h whatever happens), cost cap (destroy when elapsed x $/h > CAP),
# stall (driver.log idle 60 min) -> pull + destroy, md5 copy-back before every destroy.
#   DRY=1 bash vast_step14.sh            # build + check the bundle only, print what would run (no vastai calls)
#   bash vast_step14.sh <offer> [<offer> ...]      # after the user's OK
set -u
cd ~/kfupm_ws/ras-multi-agent-6g-uav
SEEDS="21000 21001 21002 21003 21004 21005 21006 21007"; PILOT_SEED=21900
LABEL=vlmdmg-step14; OUT=results/vlm_damage/step14/flights; LOG=$OUT/vast.log
HARD_H=${HARD_H:-8}; CAP=${CAP:-2.50}; mkdir -p $OUT
CL=sim/airsim_vlm/closed_loop; STAGE=$(mktemp -d)
say() { echo "$(date -u '+%F_%T') $*" | tee -a $LOG; }

# ---------------------------------------------------------------- bundle (laptop)
build_bundle() {
  python3 $CL/damage_loop.py --seeds $SEEDS $PILOT_SEED --layout ${LAYOUT:-indep} --out $OUT/verdicts_step14.json > /dev/null || return 1
  mkdir -p $STAGE/vlm/survey
  tar -cf - --exclude=__pycache__ --exclude='*.pid' -C sim/airsim_vlm netsim airsim closed_loop | tar -xf - -C $STAGE/vlm
  cp sim/airsim_vlm/survey/heightmap.npz $STAGE/vlm/survey/
  cp results/vlm_damage/city/buildings.json results/vlm_damage/visibility_model_step5.json $OUT/verdicts_step14.json $STAGE/vlm/
  # restricted-data guard: the only .npz allowed is the AirSim survey height map
  BAD=$(cd $STAGE && find . -name '*.npz' ! -path './vlm/survey/heightmap.npz'; cd $STAGE && find . -iname '*lidar*' -o -iname '*manhattan_nyc*' -o -iname '*cities*' -o -name 'keys.env' -o -name 'pat')
  [ -n "$BAD" ] && { say "restricted file in bundle: $BAD"; return 1; }
  (cd $STAGE/vlm && find . -type f | sort | xargs md5sum > BUNDLE_MD5.txt)
  tar -czf $OUT/bundle_step14.tgz -C $STAGE vlm
  say "bundle $(du -h $OUT/bundle_step14.tgz | cut -f1), $(wc -l < $STAGE/vlm/BUNDLE_MD5.txt) files, md5 $(md5sum $OUT/bundle_step14.tgz | cut -c1-12)"
}
build_bundle || { say "bundle failed"; exit 1; }
if [ "${DRY:-0}" = 1 ]; then
  say "DRY: would create --image abdulmannan617/airsim-city-wine:1.8.1-wine11-dxvk3.1.1 --disk 60, ship the bundle,"
  say "DRY: run plan_step14_pilot.txt, check it, then plan_step14.txt (32 missions); hard stop ${HARD_H} h, cap \$${CAP}"
  rm -rf $STAGE; exit 0
fi

# ---------------------------------------------------------------- one attempt per offer
attempt() {
  local OFFER=$1 ID S st ip port DPH T0
  ID=$(vastai create instance $OFFER --image abdulmannan617/airsim-city-wine:1.8.1-wine11-dxvk3.1.1 --disk 60 --ssh --direct --label $LABEL \
    --env '-e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all' \
    --onstart-cmd 'ssh-keygen -A; mkdir -p /run/sshd; /usr/sbin/sshd; touch /root/READY' --raw 2>&1 \
    | tee -a $OUT/vast_create.txt | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$')
  if [ -z "$ID" ]; then sleep 20
    for x in $(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys;[print(d['id']) for d in json.load(sys.stdin) if d.get('label')=='$LABEL']"); do
      vastai destroy instance $x -y >> $LOG 2>&1; say "destroyed stray $x"; done
    say "create failed on $OFFER"; return 1; fi
  echo $ID > $OUT/vast.id; T0=$(date +%s); say "created $ID on offer $OFFER"
  ( sleep $((HARD_H*3600)); vastai destroy instance $ID -y >/dev/null 2>&1; echo "$(date -u '+%F_%T') hard stop destroyed $ID" >> $LOG ) &
  local HARD=$!; echo $HARD > $OUT/hardstop.pid
  for t in $(seq 1 90); do
    read st ip port < <(vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('actual_status'),d.get('public_ipaddr'),((d.get('ports') or {}).get('22/tcp') or [{}])[0].get('HostPort'))" 2>/dev/null)
    if [ "$st" = running ] && [ -n "$port" ] && [ "$port" != None ]; then
      S="ssh -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30 root@$ip"
      $S 'test -f /root/READY' 2>/dev/null && break; fi
    sleep 20; done
  kill_box() { vastai destroy instance $ID -y >> $LOG 2>&1; kill $HARD 2>/dev/null; say "destroyed $ID ($1)"; }
  if ! $S 'test -f /root/READY' 2>/dev/null; then say "never ready (status $st)"; vastai logs $ID --tail 30 >> $LOG 2>&1; kill_box "not ready"; return 1; fi
  echo "$S" > $OUT/vast.ssh
  DPH=$(vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('dph_total') or 0.5)")
  local GPU; GPU=$($S 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader' 2>/dev/null)
  say "ready $ip:$port $GPU, \$$DPH/h"
  local DRV; DRV=$(echo "$GPU" | awk -F', ' '{print int($2)}')
  [ "${DRV:-0}" -ge 595 ] || { kill_box "driver $DRV < 595"; return 1; }
  # ship + verify bundle
  $S 'mkdir -p /work' && cat $OUT/bundle_step14.tgz | $S 'cd /work && tar -xzf -'
  $S 'cd /work/vlm && md5sum -c BUNDLE_MD5.txt --quiet' >> $LOG 2>&1 || { kill_box "bundle md5 failed"; return 1; }
  $S 'cd /work && find . -name "*.npz" ! -path "./vlm/survey/heightmap.npz" | grep -q . && echo BAD' | grep -q BAD && { kill_box "restricted file on box"; return 1; }
  $S 'python3 -c "import airsim, PIL, scipy, numpy; print(\"deps ok\", numpy.__version__)"' >> $LOG 2>&1 || { kill_box "missing python deps"; return 1; }
  pull() { rsync -a -e "ssh -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20" \
             root@$ip:/work/vlm/runs root@$ip:/work/vlm/driver.log root@$ip:/work/vlm/city.log $OUT/ 2>/dev/null; }
  verify() { $S 'cd /work/vlm && find runs -type f | sort | xargs md5sum' > $OUT/.remote_md5 2>/dev/null
             (cd $OUT && md5sum -c .remote_md5 --quiet > .md5_check 2>&1); local rc=$?; say "md5 rc=$rc ($(wc -l < $OUT/.remote_md5) files)"; return $rc; }
  over_cap() { python3 -c "import sys,time; sys.exit(0 if (time.time()-$T0)/3600*$DPH > $CAP else 1)"; }
  # pilot (1 mission, not a test seed): must finish with frames, >= 1 seen building and the weight hook active
  $S "cd /work/vlm && nohup bash closed_loop/campaign_cl.sh closed_loop/plan_step14_pilot.txt > /work/vlm/campaign.out 2>&1 &"
  for t in $(seq 1 12); do sleep 300; $S 'test -f /work/vlm/runs/CAMPAIGN_DONE' 2>/dev/null && break; over_cap && break; done
  pull
  P=$OUT/runs/s14pilot_C_set_$PILOT_SEED/result.json
  if ! python3 -c "import json,sys; r=json.load(open('$P')); ok=r['seen']['n']>0 and not r.get('collisions_in_flight'); print('pilot seen', r['seen'], 'J_true', r['true']['J_true'], 'kin', r['kinematic']['J_true']); sys.exit(0 if ok else 1)" >> $LOG 2>&1; then
    verify; kill_box "pilot failed: kept logs in $OUT"; return 2; fi
  say "pilot OK; launching 32 missions"
  $S 'rm -f /work/vlm/runs/CAMPAIGN_DONE'
  $S "cd /work/vlm && nohup bash closed_loop/campaign_cl.sh closed_loop/plan_step14.txt > /work/vlm/campaign.out 2>&1 &"
  while true; do
    sleep 1200; pull; say "pulled: $(ls $OUT/runs/s14_*/result.json 2>/dev/null | wc -l)/32 results"
    if $S 'test -f /work/vlm/runs/CAMPAIGN_DONE' 2>/dev/null; then
      pull; if verify || { pull; verify; }; then kill_box "campaign done, md5 OK"; return 0; fi
      say "md5 FAILED twice: box kept for a manual check (hard stop still armed)"; return 3; fi
    if over_cap; then pull; verify; kill_box "cost cap \$$CAP reached"; return 4; fi
    age=$($S 'echo $(( $(date +%s) - $(stat -c %Y /work/vlm/driver.log) ))' 2>/dev/null)
    [ -z "$age" ] && { say "box unreachable"; continue; }
    if [ "$age" -gt 3600 ]; then say "driver.log idle ${age}s: stall"; pull; verify; kill_box stall; return 5; fi
  done
}
for OFFER in "$@"; do attempt $OFFER; rc=$?; [ $rc = 0 ] && { say SUCCESS; rm -rf $STAGE; exit 0; }; [ $rc -ge 2 ] && break; done
rm -rf $STAGE; say "stopped (rc $rc)"; exit 1

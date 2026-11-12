#!/bin/bash
# VLM-damage step-6 replay video on one Vast GPU box (user OK in this chat, 10 Nov 2026: "props off, USB power, yes video").
# Copied from an earlier MAQAC codebase vast_hil_replay.sh (unchanged there) and adapted: our run, 2 UAVs, our replay script.
# Prebuilt image abdulmannan617/airsim-city-wine (AirSim 1.8.1 CityEnviron under Wine). Copies only the replay script,
# one step-6 run's logs and frames and the AirSim survey height map (no restricted city files). Tries offers in turn; every box is
# destroyed after its attempt; 3 h hard stop per box. Usage: cloud/vast_hil_replay.sh <offer> [<offer> ...]
set -u
cd "$(dirname "$0")/.."
RUN=results/vlm_damage/city_step6/runs/s6_n2_maqac_seq_20000; HM=results/vlm_damage/video/heightmap.npz
LABEL=vlmdmg-replay; LOG=results/vlm_damage/video/vast_replay.log; OUT=results/vlm_damage/video; mkdir -p $OUT
say() { echo "$(date '+%d %H:%M') $*" >> $LOG; }
attempt() {
  local OFFER=$1 ID S st ip port
  ID=$(vastai create instance $OFFER --image abdulmannan617/airsim-city-wine:1.8.1-wine11-dxvk3.1.1 --disk 60 --ssh --direct --label $LABEL \
    --env '-e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all' \
    --onstart-cmd 'ssh-keygen -A; mkdir -p /run/sshd; /usr/sbin/sshd; touch /root/READY' --raw 2>&1 \
    | tee -a $OUT/vast_create.txt | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$')
  if [ -z "$ID" ]; then sleep 20
    for x in $(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys;[print(d['id']) for d in json.load(sys.stdin) if d.get('label')=='$LABEL']"); do
      vastai destroy instance $x -y >> $LOG 2>&1; say "destroyed stray $x"; done
    say "create failed on $OFFER"; return 1; fi
  echo $ID > $OUT/vast.id; say "created $ID on offer $OFFER"
  ( sleep 10800; vastai destroy instance $ID -y >/dev/null 2>&1; echo "$(date '+%d %H:%M') hard stop destroyed $ID" >> $LOG ) & local HARD=$!
  for t in $(seq 1 90); do
    read st ip port < <(vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('actual_status'),d.get('public_ipaddr'),((d.get('ports') or {}).get('22/tcp') or [{}])[0].get('HostPort'))" 2>/dev/null)
    if [ "$st" = running ] && [ -n "$port" ] && [ "$port" != None ]; then
      S="ssh -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30 root@$ip"
      $S 'test -f /root/READY' 2>/dev/null && break; fi
    sleep 20; done
  if ! $S 'test -f /root/READY' 2>/dev/null; then say "never ready (status $st port $port)"; vastai logs $ID --tail 30 >> $LOG 2>&1
    vastai destroy instance $ID -y >> $LOG 2>&1; kill $HARD 2>/dev/null; say "destroyed $ID"; return 1; fi
  say "ready $ip:$port $($S 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader' 2>/dev/null)"
  $S 'mkdir -p /work/rp/run /work/rp/out'
  tar -cf - scripts/vlm_replay_airsim.py $RUN/tracks.json $RUN/result.json $RUN/frames/vlm.jsonl $RUN/frames/labels.jsonl $RUN/frames/*_uav0_rgb.jpg $HM | $S 'cd /work/rp && tar xf -'
  $S 'cd /work/rp && find . -name "*lidar*" -o -name "*manhattan_nyc*" | grep -q . && echo BAD' | grep -q BAD && { say "restricted file on box"; vastai destroy instance $ID -y >> $LOG 2>&1; kill $HARD; return 1; }
  $S "cd /work/rp && (command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null)); python3 -c 'import matplotlib' 2>/dev/null || pip install -q matplotlib pillow; \
      city-settings --mode simpleflight --n 2 --overview > /work/s.json && python3 -c \"import json;f='/work/s.json';d=json.load(open(f));d['PhysicsEngineName']='ExternalPhysicsEngine';json.dump(d,open(f,'w'),indent=1)\" && city-start /work/s.json > /work/rp/start.log 2>&1; echo START_EXIT \$?" >> $LOG 2>&1
  if [ "${KEEP:-0}" = 1 ]; then echo "$S" > $OUT/vast.ssh; say "KEEP: box $ID up, ssh in cloud/vast_hil_replay.ssh (hard stop 3 h)"; return 0; fi
  $S "cd /work/rp && timeout 7200 python3 scripts/vlm_replay_airsim.py render --speed 2 --run $RUN --hm $HM --out out > render.log 2>&1; echo RENDER_EXIT \$?; tail -3 render.log" >> $LOG 2>&1
  $S "cd /work/rp && timeout 5400 python3 scripts/vlm_replay_airsim.py compose --speed 2 --run $RUN --hm $HM --out out > compose.log 2>&1; echo COMPOSE_EXIT \$?; tail -3 compose.log" >> $LOG 2>&1
  local ok=1
  for k in 1 2 3; do
    RM=$($S 'md5sum /work/rp/out/vlm_replay.mp4' 2>/dev/null | awk '{print $1}')
    [ -z "$RM" ] && break
    $S 'cat /work/rp/out/vlm_replay.mp4' > $OUT/vlmdmg_airsim_replay.mp4
    [ "$RM" = "$(md5sum $OUT/vlmdmg_airsim_replay.mp4 | awk '{print $1}')" ] && { ok=0; say "pulled video, md5 ok"; break; }
  done
  $S 'cd /work/rp/out && tar cf - airsim/00000.jpg airsim/00600.jpg airsim/01200.jpg 2>/dev/null' | tar xf - -C $OUT 2>/dev/null
  $S 'cat /work/rp/start.log /work/rp/render.log' > $OUT/box.log 2>/dev/null
  $S 'city-stop' >/dev/null 2>&1
  vastai destroy instance $ID -y >> $LOG 2>&1; kill $HARD 2>/dev/null; say "destroyed $ID"
  return $ok
}
for OFFER in "$@"; do attempt $OFFER && { say SUCCESS; [ "${KEEP:-0}" = 1 ] && disown -a; exit 0; }; done
say "all offers failed"; exit 1

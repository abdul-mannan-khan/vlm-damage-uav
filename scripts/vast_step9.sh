#!/bin/bash
# Step 9 (pre-registered 10 Nov 2026): chain-connectivity rule, 180 netsim episodes + 180 plain MAQAC-S episodes as a
# cross-machine check against the Jetson step-8 rows. One Vast box (user OK in chat, 10 Nov: "Vast CPU box").
# Copies only the netsim code and the batch script (no city files). 2 h hard stop; destroyed after md5 copy-back.
# Usage: scripts/vast_step9.sh <offer> [<offer> ...]
set -u
cd "$(dirname "$0")/.."
LABEL=vlmdmg-step9; OUT=results/vlm_damage/step9; LOG=$OUT/vast.log; mkdir -p $OUT
say() { echo "$(date -u '+%d %H:%M') $*" | tee -a $LOG; }
attempt() {
  local OFFER=$1 ID S st ip port
  ID=$(vastai create instance $OFFER --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime --disk 20 --ssh --direct --label $LABEL --raw 2>&1 \
    | tee -a $OUT/vast_create.txt | grep -o '"new_contract": *[0-9]*' | grep -o '[0-9]*$')
  if [ -z "$ID" ]; then sleep 20
    for x in $(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys;[print(d['id']) for d in json.load(sys.stdin) if d.get('label')=='$LABEL']"); do
      vastai destroy instance $x -y >> $LOG 2>&1; say "destroyed stray $x"; done
    say "create failed on $OFFER"; return 1; fi
  echo $ID > $OUT/vast.id; say "created $ID on offer $OFFER"
  ( sleep 7200; vastai destroy instance $ID -y >/dev/null 2>&1; echo "$(date -u '+%d %H:%M') hard stop destroyed $ID" >> $LOG ) & local HARD=$!
  for t in $(seq 1 60); do
    read st ip port < <(vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('actual_status'),d.get('public_ipaddr'),((d.get('ports') or {}).get('22/tcp') or [{}])[0].get('HostPort'))" 2>/dev/null)
    if [ "$st" = running ] && [ -n "$port" ] && [ "$port" != None ]; then
      S="ssh -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30 root@$ip"
      $S true 2>/dev/null && break; fi
    sleep 20; done
  if ! $S true 2>/dev/null; then say "never ready (status $st)"; vastai destroy instance $ID -y >> $LOG 2>&1; kill $HARD 2>/dev/null; say "destroyed $ID"; return 1; fi
  say "ready $ip:$port cores $($S nproc)"
  tar -cf - --exclude=__pycache__ sim/airsim_vlm/netsim scripts/step8_netsim_batch.py | $S 'mkdir -p /work && cd /work && tar xf -'
  $S 'cd /work && find . -name "*.npz" | grep -q . && echo BAD' | grep -q BAD && { say "data file on box"; vastai destroy instance $ID -y >> $LOG; kill $HARD; return 1; }
  $S 'pip install -q "numpy<2" scipy 2>&1 | tail -1; python3 -c "import numpy,scipy;print(numpy.__version__,scipy.__version__)"' >> $LOG 2>&1
  W=$($S 'echo $(( $(nproc) - 2 ))')
  $S "cd /work && python3 scripts/step8_netsim_batch.py --out /work/step9_chain.jsonl --methods maqac_seq_chain --workers $W > /work/b1.log 2>&1; tail -1 /work/b1.log; \
      python3 scripts/step8_netsim_batch.py --out /work/step9_plain_check.jsonl --methods maqac_seq --workers $W > /work/b2.log 2>&1; tail -1 /work/b2.log" >> $LOG 2>&1
  local ok=1
  for f in step9_chain.jsonl step9_plain_check.jsonl b1.log b2.log; do
    RM=$($S "md5sum /work/$f" 2>/dev/null | awk '{print $1}'); [ -z "$RM" ] && { say "missing $f"; continue; }
    $S "cat /work/$f" > $OUT/$f
    [ "$RM" = "$(md5sum $OUT/$f | awk '{print $1}')" ] && say "pulled $f md5 ok" || say "md5 MISMATCH $f"
  done
  [ "$(wc -l < $OUT/step9_chain.jsonl)" = 180 ] && [ "$(wc -l < $OUT/step9_plain_check.jsonl)" = 180 ] && ok=0
  vastai destroy instance $ID -y >> $LOG 2>&1; kill $HARD 2>/dev/null; say "destroyed $ID"
  return $ok
}
for OFFER in "$@"; do attempt $OFFER && { say SUCCESS; exit 0; }; done
say "all offers failed"; exit 1

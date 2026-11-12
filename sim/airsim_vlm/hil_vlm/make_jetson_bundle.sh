#!/usr/bin/env bash
# Build the LOCAL folder that is copied to the Jetson as ~/vlm_hil (copying is a separate, booked step: DESIGN.md).
# Contents: this folder's code + the VLM snapshot's netsim (with third_party) + survey/heightmap.npz (AirSim CityEnviron
# height survey, md5 2d4eed4f..., the same file VehCom's set-up 2 used; allowed) + buildings.json + the int8 calibration.
# NOT included (restricted, never copied): data/cities/*.npz, london_lidar.npz, manhattan_nyc.npz; no keys or tokens.
# The int8 VLM itself is already on the Jetson (~/vlm_damage_p3/model, step 8; checked by md5 there, model.md5).
#   bash make_jetson_bundle.sh OUT_DIR
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SNAP=$(cd "$HERE/.." && pwd)
O=${1:?OUT_DIR}; mkdir -p "$O/params" "$O/netsim"
cp "$HERE"/{run_vlm_hil.py,run_maqac_hil.py,vlm_worker.py,hil_link.py,city.py,heightmap.py,buildings.py,seg_rgbs.txt} "$O/"
cp "$HERE"/{bench_vlm.py,bench_vlm_ens.py,p27_router.py,param_dump.py,param_set.py,backup_by_name.py,fc_state.py,handover.py} "$O/"
cp "$HERE"/{hil_campaign_vlm.sh,summarize_vlm_hil.py,buildings.json,int8_v2_calibration.npz,DESIGN.md} "$O/"
cp "$SNAP/survey/heightmap.npz" "$O/heightmap.npz"
cp "$SNAP"/netsim/*.py "$O/netsim/"; mkdir -p "$O/netsim/third_party"; cp "$SNAP"/netsim/third_party/*.py "$SNAP"/netsim/third_party/LICENSE* "$O/netsim/third_party/"
if find "$O" -name '*.npz' | grep -E -i 'cities|london|manhattan' ; then echo "restricted file in bundle: stop"; exit 1; fi
( cd "$O" && find . -type f ! -name MD5SUMS -exec md5sum {} + | sort -k2 > MD5SUMS )
echo "bundle: $O ($(du -sh "$O" | cut -f1)); on the Jetson: cd ~/vlm_hil && md5sum -c --quiet MD5SUMS"
echo "also on the Jetson (by hand): cp ~/maqac_hil/params/reference_paperA_2026-11-02.params ~/vlm_hil/params/"

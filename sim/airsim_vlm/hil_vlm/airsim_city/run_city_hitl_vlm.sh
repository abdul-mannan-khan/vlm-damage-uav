#!/bin/bash
# [VLM paper copy 12 Nov 2026: settings_hitl_city_vlm.json = VehCom settings + downward camera "down" 640x480, 60 deg, Scene+Segmentation, as step 6 (make_settings_city.camera)]
# MAQAC city HIL, laptop side: AirSim 1.8.1 CityEnviron (Windows build, Wine 9 + DXVK 2.3.1) with ONE PX4 HITL vehicle,
# UAV 0 on the pad used by the 52-mission campaign (X 0, Y -4). Settings follow W2's bridge
# (~/lit_results/x500_airsim_hitl_bridge/README.md): no "Sensors" block (AirSim 1.8.1 then sends no HIL_SENSOR/HIL_GPS),
# no "Parameters" block (AirSim would write PX4 parameters to the Pixhawk), TCP 4560 on the USB-network address.
# Run ONLY under resman, one AirSim on the laptop at a time:
#   resman run -m 8G -g 4G -n vlm-city-hitl -- bash sim/airsim_vlm/hil_vlm/airsim_city/run_city_hitl_vlm.sh OUTDIR [TMAX]
# Stops when OUTDIR/STOP exists, after TMAX s (default 240 = boot test), if free RAM < 1.5 GB or swap grows > 1 GB;
# a background backstop kills it at TMAX+60 s regardless.
# Kills only its own process group and its own Wine prefix.
HERE=$(cd "$(dirname "$0")" && pwd)
O=${1:?OUTDIR}; TMAX=${2:-240}; mkdir -p "$O"; rm -f "$O/STOP"
export WINEPREFIX=~/sim_bins/wine_city WINEDEBUG=${CITY_WINEDEBUG:--all} DXVK_HUD=0 DXVK_FRAME_RATE=20
export DXVK_FILTER_DEVICE_NAME="NVIDIA" DXVK_LOG_LEVEL=info DXVK_LOG_PATH="$O"   # Vulkan lists the Intel iGPU first
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
# Diagnosis 4 Nov (diag_1104_1759): the bootstrap CityEnviron.exe exits in ~20 s under Wine; the real binary in
# CityEnviron/Binaries/Win64 with DX11 (DXVK) on the NVIDIA GPU boots, AirSim API up after 21 s, ~2.9 GB VRAM.
EXE=~/sim_bins/airsim181_win/CityEnviron/WindowsNoEditor/CityEnviron/Binaries/Win64/CityEnviron.exe
WSET=$(winepath -w "$HERE/settings_hitl_city_vlm.json" 2>/dev/null)
PY=~/sim_bins/airsim_venv/bin/python
setsid wine "$EXE" -dx11 -windowed -ResX=480 -ResY=270 -nosound -nosplash -log -stdout -FullStdOutLogOutput \
    -ExecCmds="sg.ViewDistanceQuality 0, sg.ShadowQuality 0, sg.TextureQuality 0, sg.EffectsQuality 0, sg.PostProcessQuality 0, sg.FoliageQuality 0, sg.AntiAliasingQuality 0, t.MaxFPS 20" \
    -settings="$WSET" > "$O/airsim.log" 2>&1 &
AP=$!
stop(){ kill -- -$AP 2>/dev/null; sleep 5; kill -9 -- -$AP 2>/dev/null; WINEPREFIX=~/sim_bins/wine_city wineserver -k 2>/dev/null; }
trap stop EXIT
# hard backstop: whatever happens in the loop, kill this AirSim TMAX+60 s after start
( sleep $((TMAX+60)); echo "hard timeout"; kill -- -$AP 2>/dev/null; sleep 5; kill -9 -- -$AP 2>/dev/null
  WINEPREFIX=~/sim_bins/wine_city wineserver -k 2>/dev/null ) &
WD=$!
trap 'stop; kill $WD 2>/dev/null' EXIT
sw0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
t0=$(date +%s); up=0
while :; do
  now=$(($(date +%s)-t0))
  [ -f "$O/STOP" ] && { echo "STOP file"; break; }
  [ $now -ge $TMAX ] && { echo "TMAX reached"; break; }
  kill -0 $AP 2>/dev/null || { echo "AirSim exited"; break; }
  fr=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$fr" -lt 1536 ] && { echo "free RAM ${fr} MB < 1.5 GB: stopping"; break; }
  sw=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
  [ $((sw-sw0)) -gt 1024 ] && { echo "swap grew by $((sw-sw0)) MB: stopping"; break; }
  if [ $up = 0 ] && $PY -c "import airsim; c=airsim.MultirotorClient(timeout_value=5); c.confirmConnection()" >/dev/null 2>&1; then
    up=1; echo "AirSim API up after ${now} s"
  fi
  echo "$(date -u +%T) t=${now}s free_MB=$fr swap_MB=$sw gpu_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" >> "$O/health.log"
  sleep 10
done
[ $up = 1 ] || echo "AirSim API never answered"

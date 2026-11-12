# Closed-loop VLM HIL: plan (draft, 12 Nov 2026; nothing flown yet)

**Goal.** Close perception → calibrated belief → conformal set → planner weights → decision → PX4 on flight hardware.
UAV 0 (of 2) flies in PX4 HITL on the Pixhawk 6C. AirSim CityEnviron on the laptop supplies the physics, the sensors and
UAV 0's downward camera. The Jetson Orin Nano runs the network model, both UAVs' decisions (1 s rounds) and the int8 VLM
on UAV 0's frames (one every 5 s).
This is VehCom set-up 2 (30 HIL missions, 4–5 Nov) with the VLM added. It is a systems check and does not test a
performance claim.

## What step 6 actually did, and why there are two arms
In step 6 (sec:flights) the VLM read every frame live and its output was **logged**. It never reached the planner:
MAQAC-S planned on the network model's onboard user perception, and the label-set coverage on the map was computed
afterwards. So "feed it to the planner as step 6 did" means "do not feed it". The runner therefore has two arms:
- `off` = step 6 exactly. The VLM runs live on the Jetson and is logged; the decisions are plain MAQAC-S.
- `prior` = the new closed loop. Each frame's set midpoint ŵ (Lemma 2, g = 1/1.5/2.5/3) goes to the buildings that fill
  ≥ 2 % of the frame (segmentation). MAQAC's prior density of unseen users (rho0 and rho_arr) is multiplied by ŵ in the
  cells within 25 m of each building. This is applied after round t and used from round t+1. **DECIDE D2**: whether
  this coupling is acceptable as "planner weights", and `--vlm-share all` (UAV 0's map goes to both planners) vs `own`.
- Checks: with n = 2 and full links the chain rule never binds. Kinematic J is identical with and without it, and it
  reproduces step 6's J_kin exactly (seeds 20000–20002). The coupling does change decisions: on 120 s kinematic runs
  with weights forced on, J moved by −2.6 to +0.4 %.
- Caveat: CityEnviron is intact. The VLM mostly answers {B} or {B, C}, so ŵ is 1.0–1.25. The `prior` arm exercises the
  data path. It does not test whether damage-aware weighting is useful.

## Mission design (DECIDE D2/D3)
- Scenario as step 6 (`box/plan_step6.txt`): disaster, 2 UAVs, 600 m area at origin N 40 / E −10, T = 300 s,
  100 m + 3 m layer, keep-out box (−80, 36, −339, −235, top 240), survey height map, onboard perception. The method is
  MAQAC-S with the relayed list and the chain rule (the planner of record).
- Arms: `off` and `prior`. Seeds 20000–20005, with both arms of a seed flown back to back: **12 missions**, plus a 60 s
  shakedown (seed 19999, `prior`).
- Time: about 7 min per mission (300 s flight, transfer and climb, settle, kinematic twin and replay twin ≈ 10 s,
  drain ≤ 8 s). 12 missions take about 85 min. The protocol takes about 60–75 min more: backup (≤ 25 min if done by
  name), set-up reboots, shakedown and GO, restore, and 1–2 read-back passes. **Total ≈ 2.6 h**, inside 3 h. Eight
  seeds (16 missions) would take about 3.3 h.
- **Use the single int8 seed (default), not the 4-seed ensemble.** The ensemble alone peaked at 7.5 of 7.6 GB, and the
  HIL adds the runner, the HIL server and the router (≈ 0.8–1.0 GB), so it would swap or be OOM-killed. No onboard
  4-seed calibration exists either: the step-11 run has only city and test logits, no val halves. It would also take
  3.6 s of the 5 s period. Cost of the single seed: ECE 0.060 vs 0.046; coverage 0.902 is kept (Table "Onboard, int8,
  one seed").

## Jetson budget (7.6 GB shared CPU/GPU, 6 cores, 25 W)
| process | memory | CPU |
|---|---|---|
| VLM worker (int8, 1 seed) | 4.6–4.75 GB CUDA + ~0.4 GB host (step 8: system peak 7.0 GB together with 5 netsim workers) | 1 core preprocessing, GPU 1.8 s/frame |
| runner (netsim + decisions + twins) | ~0.3–0.7 GB (dry run VmHWM 0.73 GB on x86) | 1 core; MAQAC-S 17–33 ms per round (VehCom) |
| HIL server (setpoints 50 Hz) + p27_router (MAVLink AirSim↔Pixhawk, ~333 HIL_SENSOR/s) | ~0.15 GB | ~1 core |

Expected peak ≈ 6.0–6.6 GB. The campaign refuses to start below 6.2 GB MemAvailable, which also catches another
session's LLM server left resident. Memory, overruns and latency are logged every round.

## What goes to the Jetson (`make_jetson_bundle.sh OUT`, 0.8 MB)
- This folder's code and the VLM snapshot's netsim (with third_party).
- `heightmap.npz` = `sim/airsim_vlm/survey/heightmap.npz` (AirSim CityEnviron height survey, md5 2d4eed4f). This is the
  same file VehCom's set-up 2 used. It is allowed.
- `buildings.json` (step-6 inventory) and `int8_v2_calibration.npz`. The calibration was re-fitted from the Jetson
  val logits and reproduces step 8 P3 V2 exactly: F1 0.663, ECE 0.060, coverage 0.902, 1.59 labels, q̂ 0.848.
- **No** `data/cities/*.npz`, `london_lidar.npz` or `manhattan_nyc.npz` (the script refuses them), and no keys.
- Already on the Jetson: `~/vlm_damage_p3/model` (int8, check against model.md5) and `~/vlm_damage_p3/venv` (torch; used
  by the worker through `--vlm-python`). The runner uses the system python3 with pymavlink, as in VehCom.

## Commands (all after a robotq booking, both locks; **DECIDE D1**)
```
# laptop (reserved for HIL), one AirSim at a time:
resman run -m 8G -g 4G -n vlm-city-hitl -- bash sim/airsim_vlm/hil_vlm/airsim_city/run_city_hitl_vlm.sh /tmp/vlm_city 14400
# only if the Jetson has no 'airsim' client: ~/sim_bins/airsim_venv/bin/python sim/airsim_vlm/hil_vlm/frame_relay.py --buildings sim/airsim_vlm/hil_vlm/buildings.json   (then FRAMES=relay)
bash sim/airsim_vlm/hil_vlm/make_jetson_bundle.sh /tmp/vlm_hil_bundle   # then copy it to Jetson ~/vlm_hil (booked step)
# Jetson:
cd ~/vlm_hil && md5sum -c --quiet MD5SUMS && cp ~/maqac_hil/params/reference_paperA_2026-11-02.params params/
setsid nohup flock -n ~/.pixhawk.lock bash hil_campaign_vlm.sh > campaign.out 2>&1 &
echo GO > ~/vlm_hil/CONTROL           # DECIDE D4, after the shakedown check below
python3 summarize_vlm_hil.py results/hil_vlm --json results/hil_vlm/summary.json
# dry run anywhere (no hardware): python3 run_vlm_hil.py --dry-run --T 30 --seeds 20002 --vlm-planner off prior --out /tmp/d
```
**Protocol** (as VehCom): owner files busy → full backup (else by name) → SYS_HITL 1, COM_RC_IN_MODE 4, speed limits,
geofence off (**D5**, allowed by the 11 Nov standard practice) → reboot → HIL_ACT_FUNC 101–104 → shakedown → GO → 12
missions in one call (never reboot with the AirSim vehicle in the air) → HIL_ACT_FUNC 0 → restore backup → write
GF 100/60/3 and SYS_HITL 0 → reboot → read back by name until 0 differences (GF expected 100/60/3) → hand-over checks
→ owner files idle.

**Shakedown check before GO**:
- RESULT line: 0 failsafe, mode drops and overruns;
- VLM answered 12/12, p90 ≤ 5 s, age ≤ 10 s;
- mem_peak ≤ 7.3 GB;
- tracking error < 10 m;
- `loop_late_4ms`/router stats normal;
- frames in `<mission>/frames/` show the city from 100 m.

## Risks
1. **The AirSim RPC (41451) has to be reachable from the Jetson over usb0.** No top-level LocalHostIp is set, so
   AirSim should listen on all interfaces, but the laptop firewall could block it. The `airsim` pip client
   (msgpack-rpc, old tornado) may not install offline on the Jetson; the fallback is `frame_relay.py`.
2. **Frame transfers share the USB link with the HITL MAVLink stream.** Each transfer is about 1.8 MB raw per 5 s.
   Watch router RTT and loop lateness. If they degrade, switch to compressed PNG.
3. **Rendering a camera under Wine/DXVK** with `ViewMode NoDisplay` needs a 20–30 s capture test first. The first
   segmentation capture is stale and is discarded.
4. **GPU memory-bandwidth contention could slow the HIL server or router** (on this board the CPU and GPU share
   memory). This is logged and is bar P2.
5. **Building attribution is coarse.** The frame-level answer ("most severe damage visible") is given to every building
   that fills ≥ 2 % of the frame.
6. **The `prior` coupling is new** and not covered by Theorem 1 as stated. Lemma 1 still holds for whatever weights
   the planner used.
7. **Exchangeability.** Rendered city frames are not RescueNet, so the set coverage on them is not guaranteed. This is
   already stated in sec:disc.

## Pre-registration draft (fix before the shakedown; per mission unless stated)
| bar | criterion |
|---|---|
| P1 | no failsafe, mode drop, disarm in the air or abort in any mission (report k/n with a Clopper–Pearson upper bound) |
| P2 | 0 decision-round overruns (round not finished inside its 1 s) and router/HIL loop stats like VehCom |
| P3 | in-loop VLM compute p90 ≤ 5 s (the frame period); capture + VLM p90 ≤ 5 s |
| P4 | ≥ 95 % of frame requests answered, 0 worker errors, belief age at first use ≤ 10 s for every answered frame |
| P5 | MAQAC-S fleet decision p95 ≤ 100 ms per round with the VLM running (descriptive: change vs VehCom's ≤ 41 ms) |
| P6 | system memory in use ≤ 7.3 GB at every round; no OOM kill, no swap growth > 256 MB |
| P7 | `off` arm: J within ±5 % of the step-6 SITL mission with the same seed, and of the kinematic twin |
| P8 | `prior` arm: J within ±5 % of its replay twin (same weights at the same rounds, no HIL), so the HIL flight does not change the outcome; ΔJ(replay vs kinematic) reported as the VLM's effect, with no claim |
| P9 | UAV 0 horizontal tracking error mean ≤ 5 m; minimum clearance above the surveyed map > 0 m |
| — | all 12 missions reported, including failures |

# Trustworthy onboard VLM damage assessment for multi-UAV networks

Code, per-run results and videos for the manuscript *Trustworthy Onboard Vision-Language Damage Assessment for Multi-UAV
Networks with Perception-Aware Coordination Guarantees* (submitted to IEEE Transactions on Vehicular Technology).

A compact vision-language model (Qwen2.5-VL-3B with LoRA) grades building damage in post-hurricane UAV images, is calibrated
and wrapped in split-conformal label sets, and feeds a multi-UAV planner whose guarantee accounts for perception errors.

## Contents

| Folder | What |
|---|---|
| `scripts/vlm_damage_step2.py` | untuned baselines (Qwen2.5-VL 3B/7B, CLIP) and the label rule from RescueNet masks |
| `scripts/vlm_damage_tune.py` | LoRA fine-tuning (4 seeds), validation-plateau stopping, GRPO live-group gate |
| `scripts/vlm_damage_posthoc.py`, `vlm_damage_export_calib.py` | vector/temperature scaling and split-conformal sets |
| `scripts/vlm_damage_trust_eval.py` | trust and belief evaluation: Brier (Murphy decomposition), risk-coverage/AURC, aleatoric vs epistemic split, per-class (Mondrian) coverage, false alarms with Clopper-Pearson bounds |
| `scripts/step7_guarantees.py` | exact-optimum test bench (served-error bound, relayed vs one-hop) |
| `scripts/step8_*.py`, `step9_analyse.py`, `sim/airsim_vlm/netsim/` | network simulator with range-limited backhaul, chain rule |
| `scripts/vlm_damage_live.py`, `vlm_damage_city_*.py`, `scripts/step6/`, `sim/airsim_vlm/airsim/` | AirSim CityEnviron flights with the live VLM (PX4 SITL; outputs logged, not fed to the planner), visibility model |
| `scripts/jetson_vlm/`, `sim/airsim_vlm/jetson/` | int8 merge and timing on a Jetson Orin Nano; `bench_vlm_ens.py` runs all four seeds on one int8 model |
| `sim/airsim_vlm/hil_vlm/` | closed-loop hardware-in-the-loop: Pixhawk 6C (PX4 HITL) + AirSim; the Jetson runs the int8 VLM and the planner (`run_vlm_hil.py`, `hil_campaign_vlm.sh`, `PLAN.md`); results in `results/vlm_damage/hil_vlm_20261112/` |
| `scripts/step8_netsim_batch.py` (`STEP12=1`), `scripts/step12_analyse.py`, `scripts/vast_step12.sh` | best-first relayed planner vs fixed order vs RAG, 6-16 UAVs, 1,350 episodes |
| `scripts/vlm_tvt_figs.py`, `vlm_setup_fig.py`, `vlm_graphical_abstract.py`, `vlm_demo_video.py` | figures, graphical abstract, demo video |
| `results/vlm_damage/` | every number in the paper (JSON/JSONL), learning curves (`tune/sft_s*/curve.json`), test predictions |
| `media/` | AirSim replay video, narrated demo with subtitles, graphical abstract, bench photo |
| `paper/` | figure PDFs as in the manuscript |

Hypotheses and pass/fail bars were written before each run; the manuscript's Table "Pre-registered criteria and outcomes"
lists all of them, including the ones that failed.

## Not included

- RescueNet images: download from the dataset authors (Rahnemoonfar et al., Scientific Data 2023, CC0).
- LoRA adapter weights (120 MB per seed): available from the authors on request.
- Raw AirSim camera frames of the city flights (about 1 GB).

## Closed-loop hardware in the loop

12 missions with a Pixhawk 6C flight controller (PX4 HITL, propellers off) flying UAV 0 through AirSim CityEnviron, and the
int8 VLM plus the fleet decisions on a Jetson Orin Nano: code in `sim/airsim_vlm/hil_vlm/` (`DESIGN.md`), results in
`results/vlm_damage/hil_vlm_20261112/` and `results/vlm_damage/step13_summary.json`.

## Licence

Code: MIT (see `LICENSE`). Results and media: CC BY 4.0.

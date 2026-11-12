# CityEnviron ground-contact check (2026-10-29)

VM: `maqac-airsim-win` (GCP spot, us-east1-c, n1-standard-8 + T4 vWS, image `airsim181-city-landscape-win2022-t4`).
Start/stop: `gcloud compute instances start|stop maqac-airsim-win --zone us-east1-c --project project-ee36cf33-b5c7-49f1-ac0`.
Map launch: `~/phd_students/swarm-airsim/windows_vm/start_map.ps1` with our settings (see the commands in this session; `VM_NAME` differs from vm.sh's default).
Images and JSON in `checks/`.

## Findings
| Item | Result | Evidence |
|---|---|---|
| Drones | All settle on the ground **except** a drone spawned at (X 0, Y -8) relative to PlayerStart, which rests on the park's stone railing. | `checks/01`, `checks/02` |
| Drone spawn spots verified on the plaza paving | (0,-4) (0,0) (0,4) (4,-4) (4,0) (4,4) (-4,-4) (-4,0) (-4,4). (0,8) is on the edge of a traffic lane: do not use. A 10th plaza spot still to test. | `checks/05`, `report5.json` |
| City traffic and pedestrians | Moving and on the ground (shadows at wheels/feet). Pedestrians are cyclists (`BicycleMan` mesh). | `checks/03`, `checks/04` |
| API access to native traffic/pedestrians | **Not available**: they are spawned by `CityEnviron_NPCVehicleManager` / `CityEnv_PedManager` and do not appear in `simListSceneObjects` or `simGetDetections`; the `Vehicle_*` objects are parked static cars. | `report3.json` |
| Segmentation | Pedestrians get their own segmentation colour: usable as per-pixel labels for the VLA. | `checks/04` |
| Spawned users | `simSpawnObject(name, "BicycleMan", pose, scale, False)` works; objects can be moved with `simSetObjectPose`. Spawned users are the ground truth for MAQAC (positions known exactly). | `report4.json`, `checks/04` |
| Ground-height probe (downward distance sensor on a teleported drone) | **Too noisy** (±0.3-0.7 m at a fixed point). Replace with a DepthPlanar image from a downward external camera for snapping spawned users to the ground. | `report5.json` |

## Consequences for work package D
- Users = our own spawned `BicycleMan` objects driven by the network model's mobility; native traffic/pedestrians remain as visual clutter.
- Ground snapping of users: depth-camera height map (to implement and verify visually before experiments).
- Drone spawn list: the 9 verified spots above.

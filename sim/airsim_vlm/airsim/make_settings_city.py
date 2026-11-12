#!/usr/bin/env python3
"""AirSim 1.8.1 settings for CityEnviron (Windows VM) with N PX4 SITL vehicles running on a separate Linux VM.

AirSim is the lockstep TCP server: vehicle i listens on LocalHostIp:4560+i and PX4 instance i (on the Linux VM,
PX4_SIM_HOSTNAME=<windows ip>) connects to it. MAQAC talks to PX4 directly over MAVLink on the Linux VM, so AirSim's
own offboard UDP link is configured but not used.
Spawn spots: the plaza positions verified in GROUND_CHECK.md (metres, NED, relative to PlayerStart).
Each vehicle carries a downward camera 'down' (scene, depth-planar, segmentation) for VLA frames.
External cameras: 'mapcam' (downward, depth-planar + segmentation, for the height survey and ground snapping of users,
maqac_px4/survey_heightmap.py) and 'overview' (scene, for the mission video).

    python3 make_settings_city.py --n 6 --win-ip 10.142.0.5 --px4-ip 10.142.0.6 > settings_city_px4.json
"""
import argparse
import json

SPOTS = [(0, -4), (0, 0), (0, 4), (4, -4), (4, 0), (4, 4), (-4, -4), (-4, 0), (-4, 4)]


def camera(w, h, fov):
    return {'CaptureSettings': [{'ImageType': t, 'Width': w, 'Height': h, 'FOV_Degrees': fov} for t in (0, 1, 5)],
            'X': 0.0, 'Y': 0.0, 'Z': 0.3, 'Pitch': -90.0, 'Roll': 0.0, 'Yaw': 0.0}


def external(a):
    return {'mapcam': {'CaptureSettings': [{'ImageType': t, 'Width': a.map_px, 'Height': a.map_px, 'FOV_Degrees': 90}
                                           for t in (0, 1, 5)],
                       'X': 0.0, 'Y': 0.0, 'Z': -400.0, 'Pitch': -90.0, 'Roll': 0.0, 'Yaw': 0.0},
            'overview': {'CaptureSettings': [{'ImageType': 0, 'Width': 1280, 'Height': 720, 'FOV_Degrees': 70}],
                         'X': -650.0, 'Y': 0.0, 'Z': -450.0, 'Pitch': -35.0, 'Roll': 0.0, 'Yaw': 0.0}}


def render_cams(a):
    """Nadir cameras for VLA frame rendering (posed per frame with simSetCameraPose): the proposed 'vla' camera and
    the current on-board 'down' camera ('vla_old') for comparison."""
    def cam(w, h, fov):
        return {'CaptureSettings': [{'ImageType': t, 'Width': w, 'Height': h, 'FOV_Degrees': fov} for t in (0, 5)],
                'X': 0.0, 'Y': 0.0, 'Z': -200.0, 'Pitch': -90.0, 'Roll': 0.0, 'Yaw': 0.0}
    return {'vla': cam(a.vla_px, a.vla_px, a.vla_fov), 'vla_old': cam(a.w, a.h, a.fov)}


def vehicle(i, x, y, a):
    return {
        'VehicleType': 'PX4Multirotor', 'X': x, 'Y': y, 'Z': 0, 'Yaw': 0,
        'UseSerial': False, 'LockStep': True, 'UseTcp': True, 'TcpPort': 4560 + i,
        'LocalHostIp': a.win_ip, 'ControlIp': a.px4_ip,
        'ControlPortLocal': 14540 + i, 'ControlPortRemote': 14580 + i,
        'Sensors': {'Barometer': {'SensorType': 1, 'Enabled': True, 'PressureFactorSigma': 0.0001825},
                    'Imu': {'SensorType': 2, 'Enabled': True}, 'Gps': {'SensorType': 3, 'Enabled': True},
                    'Magnetometer': {'SensorType': 4, 'Enabled': True}},
        'Parameters': {'NAV_RCL_ACT': 0, 'NAV_DLL_ACT': 0, 'COM_OBL_ACT': 1, 'LPE_LAT': 47.641468, 'LPE_LON': -122.140165},
        'Cameras': {'down': camera(a.w, a.h, a.fov)},
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--win-ip', required=True)
    ap.add_argument('--px4-ip', required=True)
    ap.add_argument('--clock-speed', type=float, default=1.0)
    ap.add_argument('--w', type=int, default=640)
    ap.add_argument('--h', type=int, default=480)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--map-px', type=int, default=1024)
    ap.add_argument('--render', action='store_true', help='frame rendering: one SimpleFlight vehicle, no PX4, VLA cameras')
    ap.add_argument('--vla-px', type=int, default=1536)
    ap.add_argument('--vla-fov', type=float, default=50.0)
    a = ap.parse_args()
    if a.render:
        s = {'SettingsVersion': 1.2, 'SimMode': 'Multirotor', 'ClockSpeed': 1.0, 'ApiServerPort': 41451,
             'ViewMode': 'NoDisplay', 'ExternalCameras': {**external(a), **render_cams(a)},
             'Vehicles': {'UAV0': {'VehicleType': 'SimpleFlight', 'X': SPOTS[0][0], 'Y': SPOTS[0][1], 'Z': 0, 'Yaw': 0}}}
        print(json.dumps(s, indent=1))
        raise SystemExit
    assert a.n <= len(SPOTS), f'only {len(SPOTS)} verified spawn spots'
    s = {'SettingsVersion': 1.2, 'SimMode': 'Multirotor', 'ClockSpeed': a.clock_speed, 'ApiServerPort': 41451,
         'ViewMode': 'NoDisplay' if a.n > 1 else 'SpringArmChase',
         'ExternalCameras': external(a),
         'Vehicles': {f'UAV{i}': vehicle(i, *SPOTS[i], a) for i in range(a.n)}}
    print(json.dumps(s, indent=1))

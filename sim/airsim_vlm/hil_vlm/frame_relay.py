#!/usr/bin/env python3
"""Fallback frame path for the closed-loop VLM HIL (use only if the Jetson cannot run the airsim RPC client itself).

Runs on the LAPTOP next to AirSim (python of ~/sim_bins/airsim_venv), binds to the USB-network address only, and answers
GET /frame?vehicle=UAV0&camera=down with an .npz holding rgb and seg (uint8 H x W x 3, the same two images and byte order
as vlm_worker.AirSimFrames). Light: one capture per request (one every 5 s); no state, no writes to AirSim.

    ~/sim_bins/airsim_venv/bin/python frame_relay.py --bind 192.168.55.100 --port 41460
The Jetson then uses run_vlm_hil.py --vlm-frames relay --relay-port 41460.
"""
import argparse
import io
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bind', default='192.168.55.100')
    ap.add_argument('--port', type=int, default=41460)
    ap.add_argument('--airsim-ip', default='127.0.0.1')
    ap.add_argument('--buildings', default=None, help='buildings.json: set the step-6 segmentation IDs once at start')
    a = ap.parse_args()
    import airsim
    c = airsim.MultirotorClient(ip=a.airsim_ip, timeout_value=10)
    c.confirmConnection()
    if a.buildings:
        import json
        from buildings import apply_ids
        print('segmentation IDs set for', apply_ids(c, json.load(open(a.buildings))), 'buildings', flush=True)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path != '/frame':
                self.send_error(404); return
            q = dict(urllib.parse.parse_qsl(u.query))
            veh, cam = q.get('vehicle', 'UAV0'), q.get('camera', 'down')
            rs = c.simGetImages([airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False),
                                 airsim.ImageRequest(cam, airsim.ImageType.Segmentation, False, False)], vehicle_name=veh)
            f = [np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)[:, :, ::-1] for r in rs]
            b = io.BytesIO(); np.savez(b, rgb=f[0], seg=f[1]); body = b.getvalue()
            self.send_response(200); self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *x):
            pass
    ThreadingHTTPServer((a.bind, a.port), H).serve_forever()


if __name__ == '__main__':
    main()

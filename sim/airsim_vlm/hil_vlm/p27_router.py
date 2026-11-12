"""MAQAC copy (4 Nov 2026, maqac-vehcom-a5) of W2's router with ONE change: only a whitelist of Pixhawk messages goes to
AirSim. The Windows AirSim 1.8.1 CityEnviron build rejects any decoded message longer than its old MAVLink definition
("Message length 37 doesn't match expected length 36" after the first heartbeat), then drops the vehicle. AirSim needs
only HEARTBEAT and HIL_ACTUATOR_CONTROLS; fixed-length state messages are passed as well. Everything else still goes to
the UDP side as before. Dropped types are logged once each.
Second change (same evening): PX4 starts the USB link in MAVLink 1 (MAV_PROTO_VER 0) and AirSim then falls back to v1 and
cannot encode its 37-byte HIL_GPS. So the router first sends a few MAVLink 2 GCS heartbeats to the Pixhawk, waits until
PX4 answers in v2, and forwards only v2 frames to AirSim.

P27 (docs/paper/PREREG_P27.md): MAVLink router on the Jetson for PX4 HITL with AirSim on the laptop.

  Pixhawk USB (by-id path)  <->  AirSim on the laptop (TCP client to HOST:4560; HIL_SENSOR/HIL_GPS in, HIL_ACTUATOR_CONTROLS out)
                              <->  the live mission loop (local UDP: the loop sends to 127.0.0.1:14560, gets replies back)

Whole MAVLink frames are forwarded (parsed with pymavlink), so frames from two sources never interleave on the serial port.
  python -m stlagent.p27_router run --host 192.168.55.100 [--port 4560] [--stats 10]
  python -m stlagent.p27_router selftest       # no hardware: framing and forwarding over loopback sockets
"""
import argparse
import socket
import subprocess
import threading
import time

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2

SERIAL, BAUD, UDP_PORT = "/dev/serial/by-id/usb-Auterion_PX4_FMU_v6C.x_0-if00", 921600, 14560   # may be ttyACM1 after a reboot


class Framer:
    """Splits a byte stream into whole MAVLink frames (v1 or v2)."""

    def __init__(self):
        self.mav = mavlink2.MAVLink(None); self.mav.robust_parsing = True

    def frames(self, data):
        out = []
        for m in self.mav.parse_buffer(data) or []:
            if m.get_type() != "BAD_DATA":
                out.append((m.get_type(), bytes(m.get_msgbuf())))
        return out


TO_AIRSIM = {"HEARTBEAT", "HIL_ACTUATOR_CONTROLS", "HIL_CONTROLS", "ATTITUDE", "LOCAL_POSITION_NED",
             "GLOBAL_POSITION_INT", "VFR_HUD", "EXTENDED_SYS_STATE"}


class Router:
    def __init__(self, ser_write, tcp_sock, udp_sock):
        self.ser_write, self.tcp, self.udp = ser_write, tcp_sock, udp_sock
        self.lock = threading.Lock(); self.udp_peer = None; self.stop = threading.Event()
        self.n = dict(ser_in=0, to_tcp=0, to_udp=0, tcp_in=0, udp_in=0, act=0, sens=0, dropped=0)
        self.dropped_types = set()

    def to_serial(self, frames):
        with self.lock:
            for _, f in frames:
                self.ser_write(f)

    def from_serial(self, msg):
        b = bytes(msg.get_msgbuf()); t = msg.get_type(); self.n["ser_in"] += 1
        if t == "HIL_ACTUATOR_CONTROLS":
            self.n["act"] += 1
        if t in TO_AIRSIM and b[:1] == b"\xfd":
            try:
                self.tcp.sendall(b); self.n["to_tcp"] += 1
            except OSError:
                self.stop.set()
        else:
            self.n["dropped"] += 1
            if t not in self.dropped_types:
                self.dropped_types.add(t); print(f"not sent to AirSim: {t} (len {len(b)})", flush=True)
        if self.udp_peer is not None and not t.startswith("HIL_"):
            self.udp.sendto(b, self.udp_peer); self.n["to_udp"] += 1

    def tcp_loop(self):
        fr = Framer()
        while not self.stop.is_set():
            try:
                d = self.tcp.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not d:
                break
            F = fr.frames(d); self.n["tcp_in"] += len(F); self.n["sens"] += sum(1 for t, _ in F if t == "HIL_SENSOR")
            self.to_serial(F)
        self.stop.set()

    def udp_loop(self):
        fr = Framer()
        while not self.stop.is_set():
            try:
                d, peer = self.udp.recvfrom(65536)
            except socket.timeout:
                continue
            self.udp_peer = peer; F = fr.frames(d); self.n["udp_in"] += len(F); self.to_serial(F)


def cmd_run(a):
    busy = subprocess.run(["fuser", SERIAL], capture_output=True, text=True).stdout.strip()
    if busy:
        raise SystemExit(f"{SERIAL} in use by {busy}")
    ser = mavutil.mavlink_connection(SERIAL, baud=BAUD, robust_parsing=True)
    gcs = mavlink2.MAVLink(None, srcSystem=255, srcComponent=191); gcs.WIRE_PROTOCOL_VERSION = "2.0"
    hb = gcs.heartbeat_encode(mavlink2.MAV_TYPE_GCS, mavlink2.MAV_AUTOPILOT_INVALID, 0, 0, 0).pack(gcs, force_mavlink1=False)
    v2, t0, t_hb = False, time.time(), 0.0
    while time.time() - t0 < 20 and not v2:                # make PX4 switch this link to MAVLink 2 before AirSim sees it
        if time.time() - t_hb > 1.0:
            ser.write(hb); t_hb = time.time()
        m = ser.recv_msg()
        if m is None:
            time.sleep(0.002)
        elif m.get_type() == "HEARTBEAT" and bytes(m.get_msgbuf())[:1] == b"\xfd":
            v2 = True
    print(f"PX4 link in MAVLink 2: {v2} after {time.time() - t0:.1f} s", flush=True)
    tcp = None
    for _ in range(600):                                   # AirSim listens once its vehicle is created; wait up to 10 min
        try:
            tcp = socket.create_connection((a.host, a.port), timeout=2); break
        except OSError:
            time.sleep(1)
    if tcp is None:
        raise SystemExit("AirSim did not accept the connection")
    tcp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); tcp.settimeout(0.5)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); udp.bind(("127.0.0.1", UDP_PORT)); udp.settimeout(0.5)
    R = Router(lambda b: ser.write(b), tcp, udp)
    threading.Thread(target=R.tcp_loop, daemon=True).start(); threading.Thread(target=R.udp_loop, daemon=True).start()
    print(f"router: {SERIAL} <-> tcp {a.host}:{a.port} <-> udp 127.0.0.1:{UDP_PORT}", flush=True)
    t_last = time.time()
    try:
        while not R.stop.is_set():
            m = ser.recv_msg()
            if m is None:
                time.sleep(0.0005)
            elif m.get_type() != "BAD_DATA":
                R.from_serial(m)
            if time.time() - t_last >= a.stats:
                print(time.strftime("%T"), R.n, flush=True); t_last = time.time()
    finally:
        R.stop.set(); ser.close(); tcp.close(); udp.close()
        print("router stopped", R.n, flush=True)


def cmd_selftest(a):
    """Two loopback TCP/UDP pairs stand in for AirSim and the loop; a fake serial records writes."""
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    cli = socket.create_connection(srv.getsockname()); acc, _ = srv.accept(); cli.settimeout(0.5); acc.settimeout(2)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); udp.bind(("127.0.0.1", 0)); udp.settimeout(0.5)
    loop = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); loop.settimeout(2)
    wrote = []
    R = Router(wrote.append, cli, udp)
    threading.Thread(target=R.tcp_loop, daemon=True).start(); threading.Thread(target=R.udp_loop, daemon=True).start()
    enc = mavlink2.MAVLink(None, srcSystem=1, srcComponent=1)
    hb = enc.heartbeat_encode(2, 12, 0, 0, 0); hb_b = hb.pack(enc)
    # AirSim -> serial: two frames split across three TCP writes must arrive whole and in order
    sens = enc.hil_sensor_encode(1, 0, 0, 9.8, 0, 0, 0, 0, 0, 0, 1013, 0, 0, 20, 0, 0).pack(enc)
    blob = sens + hb_b; acc.sendall(blob[:7]); time.sleep(0.05); acc.sendall(blob[7:30]); time.sleep(0.05); acc.sendall(blob[30:])
    loop.sendto(hb_b, udp.getsockname())                  # loop -> serial, and registers the loop's address
    time.sleep(0.5)
    ok1 = wrote[:2] == [sens, hb_b] and wrote[2:] == [hb_b]
    # serial -> AirSim (all) and loop (not HIL_*)
    act = mavlink2.MAVLink(None, 1, 1).hil_actuator_controls_encode(0, [0.0] * 16, 0, 0)
    act.pack(mavlink2.MAVLink(None, 1, 1)); m_hb = mavlink2.MAVLink(None, 1, 1).parse_char(hb_b)
    fa = mavlink2.MAVLink(None, 1, 1); a_b = act.pack(fa); m_act = mavlink2.MAVLink(None, 1, 1).parse_char(a_b)
    R.from_serial(m_act); R.from_serial(m_hb)
    got_tcp = acc.recv(4096); got_udp = loop.recv(4096)
    ok2 = got_tcp == a_b + hb_b and got_udp == hb_b
    R.stop.set()
    print("selftest framing", ok1, "forwarding", ok2, R.n)
    raise SystemExit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--host", default="192.168.55.100"); ap.add_argument("--port", type=int, default=4560)
    ap.add_argument("--stats", type=float, default=10.0)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)

#!/usr/bin/env python3
"""
control_robstride_ros.py
========================
Bilateral teleop tay kẹp 06 bám vô lăng 07 + ROS publish state cho recorder.

Topics publish:
    /gripper/state  (Float32MultiArray)  [pos_master, pos_slave, pos_norm, torque, contact, mode]
      [0] pos_master: vị trí vô lăng 07 (rad) — INPUT người dùng (action)
      [1] pos_slave:  vị trí tay kẹp 06 (rad) — STATE robot
      [2] pos_norm:   tay kẹp normalized [0=mở, 1=đóng] — chuẩn pi0.5
      [3] torque:     lực kẹp (N·m, moving avg)
      [4] contact:    1 nếu chạm vật, 0 nếu không
      [5] mode:       0=idle, 1=auto, 2=mimic

Phím tắt: GIỐNG control_robstride.py
    Enter   → Start/Stop auto mode
    +/-     → ±0.5 rad/s
    r       → Đổi chiều
    0       → 0 rad/s
    m       → Bật/Tắt MIMIC (bilateral teleop)
    t0.12   → Đổi threshold
    q       → Thoát
"""

import argparse
import can
import csv
import math
import signal
import struct
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def _send(bus, arb, data):
    bus.send(can.Message(arbitration_id=arb, data=data, is_extended_id=True))

def enable(bus, mid):
    _send(bus, (3 << 24) | (0xFD << 8) | mid, [0]*8)
    time.sleep(0.1)

def disable(bus, mid):
    _send(bus, (4 << 24) | (0xFD << 8) | mid, [0]*8)

def set_mode(bus, mid, mode):
    payload = struct.pack('<H', 0x7005) + b'\x00\x00' + struct.pack('<I', mode)
    _send(bus, (18 << 24) | (0xFD << 8) | mid, list(payload))
    time.sleep(0.05)

def send_spd(bus, mid, spd):
    payload = struct.pack('<H', 0x700A) + b'\x00\x00' + struct.pack('<f', float(spd))
    _send(bus, (18 << 24) | (0xFD << 8) | mid, list(payload))

def read_both(bus, mid1, mid2):
    _send(bus, (2 << 24) | (0xFD << 8) | mid1, [0]*8)
    _send(bus, (2 << 24) | (0xFD << 8) | mid2, [0]*8)
    results = {}
    deadline = time.time() + 0.030
    while time.time() < deadline:
        if len(results) >= 2:
            break
        r = bus.recv(timeout=0.005)
        if r is None: break
        if r.arbitration_id == 0x7FE: continue
        resp_mid = (r.arbitration_id >> 8) & 0xFF
        if resp_mid in (mid1, mid2) and resp_mid not in results:
            d = bytes(r.data)
            pos = struct.unpack('>H', d[0:2])[0] / 65535.0 * 8 * math.pi - 4 * math.pi
            vel = struct.unpack('>H', d[2:4])[0] / 65535.0 * 88.0 - 44.0
            tor = struct.unpack('>H', d[4:6])[0] / 65535.0 * 11.0 - 5.5
            results[resp_mid] = (pos, vel, tor)
    p1, v1, t1 = results.get(mid1, (None, None, None))
    p2, v2, t2 = results.get(mid2, (None, None, None))
    return p1, v1, t1, p2, v2, t2


class MovingAvg:
    def __init__(self, n=8):
        self.buf = deque(maxlen=n); self.n = n
    def update(self, val):
        if val is not None: self.buf.append(val)
        return sum(self.buf) / len(self.buf) if self.buf else 0.0
    def ready(self):
        return len(self.buf) >= self.n
    def reset(self):
        self.buf.clear()


class PIDController:
    def __init__(self, kp, ki, kd, limit_out=15.0, limit_i=2.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit_out, self.limit_i = limit_out, limit_i
        self.prev_error = 0.0; self.integral = 0.0
        self.last_time = time.time()
    def compute(self, target, current):
        now = time.time(); dt = now - self.last_time
        if dt <= 0.001: dt = 0.02
        error = target - current
        P = self.kp * error
        self.integral += error * dt
        self.integral = max(-self.limit_i, min(self.limit_i, self.integral))
        I = self.ki * self.integral
        D = self.kd * (error - self.prev_error) / dt
        out = max(-self.limit_out, min(self.limit_out, P + I + D))
        self.prev_error = error; self.last_time = now
        return out, error
    def reset(self):
        self.prev_error = 0.0; self.integral = 0.0
        self.last_time = time.time()


# ── Mechanical limits ──
POS_OPEN_RAD  = 1.459   # mở hoàn toàn
POS_CLOSE_RAD = 8.509   # đóng hoàn toàn

def normalize_pos(p_rad):
    """Map rad → [0=mở, 1=đóng] cho pi0.5"""
    if p_rad is None: return 0.0
    norm = (p_rad - POS_OPEN_RAD) / (POS_CLOSE_RAD - POS_OPEN_RAD)
    return max(0.0, min(1.0, norm))


class GripperPublisher(Node):
    def __init__(self):
        super().__init__('gripper_publisher')
        self.pub = self.create_publisher(Float32MultiArray, '/gripper/state', 10)
        self.get_logger().info(
            "✅ /gripper/state [pos_master, pos_slave, pos_norm, torque, contact, mode]")

    def publish_state(self, p_master, p_slave, pos_norm, torque, contact, mode):
        msg = Float32MultiArray()
        msg.data = [
            float(p_master if p_master is not None else 0.0),
            float(p_slave  if p_slave  is not None else 0.0),
            float(pos_norm),
            float(torque),
            1.0 if contact else 0.0,
            float(mode),
        ]
        self.pub.publish(msg)


def run(args):
    REV = args.reverse
    SPD_MAX = 44.0
    DT = 0.02
    RAMP = 1.5 * DT
    WARMUP = 20
    CONFIRM_TICKS = 3
    FILTER_N = 8
    MASTER_ID = 7
    SLAVE_ID = 6

    s = {
        'running': False, 'speed': args.speed, 'actual': 0.0,
        'threshold': args.threshold, 'warmup': 0, 'quit': False,
        'mimic': False, 'mimic_init': False, 'offset': 0.0,
    }
    lock = threading.Lock()
    filt_slave = MovingAvg(FILTER_N)
    confirm_count = 0
    pid = PIDController(kp=12.3, ki=0, kd=2.23)

    rclpy.init()
    ros_node = GripperPublisher()

    def show_help():
        print('─' * 60)
        print(f'  Tay kẹp ID={SLAVE_ID} | Vô lăng ID={MASTER_ID} | reverse={REV}')
        print(f'  Speed: {s["speed"]:+.1f} rad/s | Threshold: {s["threshold"]}')
        print(f'  ROS: /gripper/state')
        print('─' * 60)
        print('  [Enter]=Start/Stop  [+/-]=Speed  [r]=Reverse')
        print('  [m]=Toggle MIMIC    [t0.12]=Threshold  [q]=Quit')
        print('─' * 60)

    def kb():
        show_help()
        while not s['quit']:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            with lock:
                if cmd == '':
                    if s['mimic']:
                        print('  [!] Đang ở Mimic, tắt trước.')
                        continue
                    if not s['running']:
                        s['running'] = True; s['warmup'] = WARMUP
                        filt_slave.reset()
                        print(f'\n  RUN {abs(s["speed"]):.1f} rad/s\n')
                    else:
                        s['running'] = False; s['actual'] = 0.0
                        print('\n  STOP\n')
                elif cmd == '+':
                    sign = -1 if s['speed'] < 0 else 1
                    s['speed'] = round(sign * min(abs(s['speed']) + 0.5, SPD_MAX), 1)
                elif cmd == '-':
                    sign = -1 if s['speed'] < 0 else 1
                    s['speed'] = round(sign * max(abs(s['speed']) - 0.5, 0.0), 1)
                elif cmd == 'r':
                    s['speed'] = -s['speed']
                elif cmd == '0':
                    s['speed'] = 0.0
                elif cmd == 'm':
                    if not s['mimic']:
                        s['running'] = False; s['actual'] = 0.0
                        s['mimic'] = True; s['mimic_init'] = True
                        print('\n  [+] MIMIC ON\n')
                    else:
                        s['mimic'] = False
                        print('\n  [-] MIMIC OFF\n')
                elif cmd == 'q':
                    s['quit'] = True; s['running'] = False
                elif cmd.startswith('t'):
                    try: s['threshold'] = float(cmd[1:])
                    except ValueError: pass
                else:
                    try:
                        s['speed'] = round(max(-SPD_MAX, min(SPD_MAX, float(cmd))), 2)
                    except ValueError:
                        show_help()

    threading.Thread(target=kb, daemon=True).start()

    log_file = open('pid_log.csv', mode='w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Time(s)', 'VoLang_07', 'Target', 'TayKep_06', 'PID_Out'])
    start_time = time.time()

    with can.Bus(interface=args.interface, channel=args.channel) as bus:
        print(f'Init {SLAVE_ID} Mode 2...')
        set_mode(bus, SLAVE_ID, 2); enable(bus, SLAVE_ID)
        print(f'Init {MASTER_ID} disabled...')
        disable(bus, MASTER_ID); set_mode(bus, MASTER_ID, 0)
        print('Ready.\n')

        signal.signal(signal.SIGINT,
                      lambda sg, f: s.update({'quit': True, 'running': False}))

        next_tick = time.time()
        last_print = 0.0

        while not s['quit']:
            with lock:
                running = s['running']
                target_spd = s['speed'] if running else 0.0
                threshold = s['threshold']
                warmup = s['warmup']
                mimic = s['mimic']
                mimic_init = s['mimic_init']

            p_m, v_m, t_m, p_s, v_s, t_s_raw = read_both(bus, MASTER_ID, SLAVE_ID)
            avg_s = filt_slave.update(abs(t_s_raw) if t_s_raw is not None else None)

            pos_norm = normalize_pos(p_s)
            contact = False
            mode = 0

            # ── MIMIC ──
            if mimic:
                mode = 2
                if mimic_init:
                    if p_m is not None and p_s is not None:
                        with lock:
                            s['offset'] = p_s - p_m
                            s['mimic_init'] = False
                        pid.reset()
                        start_time = time.time()
                        print(f'  OFFSET = {s["offset"]:.3f}')
                else:
                    if p_m is not None and p_s is not None:
                        target_p = p_m + s['offset']
                        safe_target = max(POS_OPEN_RAD, min(POS_CLOSE_RAD, target_p))
                        cmd_spd, error = pid.compute(target=safe_target, current=p_s)
                        if abs(error) < 0.015:
                            cmd_spd = 0.0
                        send_spd(bus, SLAVE_ID, cmd_spd)

                        if filt_slave.ready() and avg_s > threshold:
                            contact = True

                        log_writer.writerow([
                            time.time() - start_time, p_m, safe_target, p_s, cmd_spd])
                        log_file.flush()

                        t_now = time.time()
                        if t_now - last_print >= 0.2:
                            last_print = t_now
                            flg = ' CONTACT!' if contact else ''
                            print(f'  [MIMIC] m={p_m:+.3f} s={p_s:+.3f} '
                                  f'pos={pos_norm:.2f} tor={avg_s:.3f}{flg}')

            # ── AUTO ──
            else:
                if running: mode = 1
                diff = target_spd - s['actual']
                if abs(diff) <= RAMP:
                    s['actual'] = target_spd
                else:
                    s['actual'] += RAMP if diff > 0 else -RAMP
                send_spd(bus, SLAVE_ID, -s['actual'] if REV else s['actual'])

                if warmup > 0:
                    with lock:
                        s['warmup'] = warmup - 1
                    confirm_count = 0

                if running and warmup == 0 and abs(s['actual']) > 0.05 and filt_slave.ready():
                    if avg_s > threshold:
                        confirm_count += 1
                    else:
                        confirm_count = 0
                    if confirm_count >= CONFIRM_TICKS:
                        contact = True
                        send_spd(bus, SLAVE_ID, 0.0)
                        with lock:
                            s['running'] = False
                            s['actual'] = 0.0
                        confirm_count = 0
                        print(f'\n  CHẠM VẬT! tor={avg_s:.3f}\n')
                else:
                    if not running: confirm_count = 0

                t_now = time.time()
                if t_now - last_print >= 0.2:
                    last_print = t_now
                    st = 'RUN' if running else 'STP'
                    flg = ' CONTACT!' if contact else ''
                    print(f'  [{st}] spd={s["actual"]:+5.2f} pos={pos_norm:.2f} '
                          f'tor={avg_s:.3f}{flg}')

            # Publish ROS mỗi tick
            ros_node.publish_state(p_m, p_s, pos_norm, avg_s, contact, mode)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            next_tick += DT
            time.sleep(max(0, next_tick - time.time()))

        # Shutdown
        print('\nDừng...')
        send_spd(bus, SLAVE_ID, 0.0); time.sleep(0.2)
        for mid in (MASTER_ID, SLAVE_ID):
            disable(bus, mid); set_mode(bus, mid, 0)
        log_file.close()
        ros_node.destroy_node()
        rclpy.shutdown()
        print('Done.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed',     type=float, default=1.5)
    ap.add_argument('--threshold', type=float, default=0.12)
    ap.add_argument('--reverse',   action='store_true')
    ap.add_argument('--interface', default='socketcan')
    ap.add_argument('--channel',   default='can0')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
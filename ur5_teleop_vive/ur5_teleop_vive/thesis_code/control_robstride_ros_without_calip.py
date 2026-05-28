#!/usr/bin/env python3
"""
control_robstride_ros_without_calip.py
Tay kẹp (06) bám Vô lăng (07) với Full PID. Ghi log đồ thị PID.
Tích hợp:
1. KHÓA AN TOÀN (Ly hợp điện tử) khi kẹp trúng vật.
2. ROS 2 Publisher trạng thái.
(Đã fix siêu mượt: Thêm Stall Detection và Ly hợp 2 biên cơ khí)

pos_norm (data[2]) cho Pi0.5:
  - Dùng vị trí VÔ LĂNG (p_m) để tính, KHÔNG dùng vị trí tay kẹp
  - Smooth transition 0.0 → 1.0 theo range vô lăng thực tế
  - MASTER_POS_OPEN  = -4.7 rad  → pos_norm = 0.0 (mở hoàn toàn)
  - MASTER_POS_CLOSE = +5.59 rad → pos_norm = 1.0 (kẹp hoàn toàn)
  - Giữa: tuyến tính 0.0 → 1.0
"""

import argparse
import can
import signal
import struct
import threading
import time
import math
import csv
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
        if r is None:
            break
        if r.arbitration_id == 0x7FE:
            continue
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
        self.buf = deque(maxlen=n)
        self.n   = n

    def update(self, val):
        if val is not None:
            self.buf.append(val)
        return sum(self.buf) / len(self.buf) if self.buf else 0.0

    def ready(self):
        return len(self.buf) >= self.n

    def reset(self):
        self.buf.clear()


class PIDController:
    def __init__(self, kp, ki, kd, limit_out=15.0, limit_i=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit_out = limit_out
        self.limit_i   = limit_i
        self.prev_error = 0.0
        self.integral   = 0.0
        self.last_time  = time.time()

    def compute(self, target, current):
        now = time.time()
        dt  = now - self.last_time
        if dt <= 0.001:
            dt = 0.02

        error      = target - current
        P          = self.kp * error
        self.integral += error * dt
        self.integral  = max(-self.limit_i, min(self.limit_i, self.integral))
        I          = self.ki * self.integral
        D          = self.kd * (error - self.prev_error) / dt
        out        = max(-self.limit_out, min(self.limit_out, P + I + D))
        self.prev_error = error
        self.last_time  = now
        return out, error

    def reset(self):
        self.prev_error = 0.0
        self.integral   = 0.0
        self.last_time  = time.time()


# ══════════════════════════════════════════════════════════════════════
# pos_norm cho Pi0.5 — dùng VÔ LĂNG (p_m), smooth 0.0 → 1.0
#
# Giá trị đo thực tế:
#   MASTER_POS_OPEN  = -5.6 rad  →  pos_norm = 0.0  (kẹp ra hoàn toàn)
#   MASTER_POS_CLOSE = -1.3 rad  →  pos_norm = 1.0  (kẹp vào hoàn toàn)
#
# Nếu cần đo lại:
#   1. Xoay vô lăng mở hết → xem data[0] → sửa MASTER_POS_OPEN
#   2. Xoay vô lăng kẹp hết → xem data[0] → sửa MASTER_POS_CLOSE
# ══════════════════════════════════════════════════════════════════════
MASTER_POS_OPEN  = -5.6    # rad — vô lăng khi mở hoàn toàn (kẹp ra)
MASTER_POS_CLOSE = -1.3    # rad — vô lăng khi kẹp hoàn toàn (kẹp vào)

def gripper_norm(p_m):
    """
    Tính pos_norm từ vị trí vô lăng.
    0.0 = mở hoàn toàn (p_m = MASTER_POS_OPEN)
    1.0 = kẹp hoàn toàn (p_m = MASTER_POS_CLOSE)
    Smooth tuyến tính ở giữa.
    """
    if p_m is None:
        return 0.0
    span = MASTER_POS_CLOSE - MASTER_POS_OPEN   # ~10.29 rad
    norm = (p_m - MASTER_POS_OPEN) / max(span, 0.001)
    return float(max(0.0, min(1.0, norm)))


# ── Mechanical limits (cho tay kẹp, dùng để clamp) ──
POS_OPEN_RAD  = 2.634
POS_CLOSE_RAD = 8.795


class GripperPublisher(Node):
    def __init__(self):
        super().__init__('gripper_publisher')
        self.pub = self.create_publisher(Float32MultiArray, '/gripper/state', 10)
        self.get_logger().info(
            "✅ /gripper/state [pos_master, pos_slave, pos_norm, torque, contact, mode]"
        )
        self.get_logger().info(
            f"   pos_norm range: OPEN={MASTER_POS_OPEN} rad → 0.0 | "
            f"CLOSE={MASTER_POS_CLOSE} rad → 1.0"
        )

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
    REV     = args.reverse
    SPD_MAX = 44.0
    DT      = 0.01
    RAMP    = 1.5 * DT

    WARMUP        = 20
    CONFIRM_TICKS = 5
    FILTER_N      = 8

    MASTER_ID = 7
    SLAVE_ID  = 6

    s = {
        'running':       False,
        'speed':         args.speed,
        'actual':        0.0,
        'threshold':     args.threshold,
        'warmup':        0,
        'quit':          False,
        'mimic':         False,
        'mimic_init':    False,
        'offset':        0.0,
        'clamped':       False,
        'clamp_pos':     0.0,
        'clamp_dir':     1,
        'limit_min':     POS_OPEN_RAD,
        'limit_max':     POS_CLOSE_RAD,
        'unlock_ignore': 0,
    }
    lock = threading.Lock()

    filt_slave    = MovingAvg(FILTER_N)
    confirm_count = 0
    pid           = PIDController(kp=13.2, ki=0.5, kd=1.0)

    rclpy.init()
    ros_node = GripperPublisher()

    def show_help():
        print('─' * 65)
        print(f'  Tay kẹp: ID {SLAVE_ID} | Vô lăng: ID {MASTER_ID} | reverse={REV}')
        print(f'  Speed Tự Động: {s["speed"]:+.1f} rad/s | Ngưỡng Lực: {s["threshold"]} N·m')
        print(f'  Chốt an toàn hiện tại: [{s["limit_min"]:.3f} rad  <-->  {s["limit_max"]:.3f} rad]')
        print(f'  pos_norm: VÔ LĂNG {MASTER_POS_OPEN} rad → 0.0 (mở)  |  {MASTER_POS_CLOSE} rad → 1.0 (kẹp)')
        print('─' * 65)
        print('  [m]   [Enter]   -> Bật/Tắt MIMIC (Xoay 07 -> 06 bám theo)')
        print('  [Enter]         -> Bắt đầu / Dừng chế độ kẹp tự động')
        print('  [+/-] [Enter]   -> Tăng/giảm tốc độ kẹp tự động 0.5 rad/s')
        print('  [q]   [Enter]   -> Thoát chương trình')
        print('─' * 65)

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
                        print('\n  [!] Đang ở chế độ Mimic, vui lòng tắt trước khi chạy tự động.\n')
                        continue
                    if not s['running']:
                        s['running'] = True
                        s['warmup']  = WARMUP
                        filt_slave.reset()
                        print(f'\n  RUN TỰ ĐỘNG {abs(s["speed"]):.1f} rad/s\n')
                    else:
                        s['running'] = False
                        s['actual']  = 0.0
                        print('\n  STOP Chạy Tự Động\n')
                elif cmd == '+':
                    s['speed'] = round(min(abs(s['speed']) + 0.5, SPD_MAX), 1)
                elif cmd == '-':
                    s['speed'] = round(max(abs(s['speed']) - 0.5, 0.0), 1)
                elif cmd == 'm':
                    if not s['mimic']:
                        s['running']    = False
                        s['actual']     = 0.0
                        s['mimic']      = True
                        s['mimic_init'] = True
                        print(f'\n  [+] KÍCH HOẠT MIMIC: Đang lấy thông số...')
                    else:
                        s['mimic'] = False
                        print(f'\n  [-] TẮT MIMIC: Khôi phục lại trạng thái chờ.')
                elif cmd == 'q':
                    s['quit']    = True
                    s['running'] = False
                    print('\n  Thoát...')

    threading.Thread(target=kb, daemon=True).start()

    log_file   = open('pid_log.csv', mode='w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Time(s)', 'VoLang_07_Rad', 'Target_Rad', 'TayKep_06_Rad',
                         'PID_Speed_Out', 'pos_norm'])
    start_time = time.time()

    with can.Bus(interface=args.interface, channel=args.channel) as bus:
        print(f'Khởi tạo Tay Kẹp {SLAVE_ID} ở Mode 2 (Velocity)...')
        set_mode(bus, SLAVE_ID, 2)
        enable(bus, SLAVE_ID)

        print(f'Khởi tạo Vô Lăng {MASTER_ID} - Cắt điện để xoay tự do...')
        disable(bus, MASTER_ID)
        set_mode(bus, MASTER_ID, 0)

        print('Cả 2 motor sẵn sàng.\n')
        signal.signal(signal.SIGINT, lambda sg, f: s.update({'quit': True, 'running': False}))

        next_tick  = time.time()
        last_print = 0.0
        confirm_count = 0

        while not s['quit']:
            with lock:
                running     = s['running']
                target      = s['speed'] if running else 0.0
                threshold   = s['threshold']
                warmup      = s['warmup']
                mimic       = s['mimic']
                mimic_init  = s['mimic_init']
                limit_min   = s['limit_min']
                limit_max   = s['limit_max']

            p_m, v_m, t_m, p_s, v_s, t_s_raw = read_both(bus, MASTER_ID, SLAVE_ID)
            avg_s = filt_slave.update(abs(t_s_raw) if t_s_raw is not None else None)

            # ══════════════════════════════════════════════════════
            # 1. LOGIC BÁM VÔ LĂNG (MIMIC)
            # ══════════════════════════════════════════════════════
            if mimic:
                if mimic_init:
                    if p_m is not None and p_s is not None:
                        with lock:
                            s['offset']     = p_s - p_m
                            s['mimic_init'] = False
                            s['clamped']    = False
                            s['unlock_ignore'] = 0
                        pid.reset()
                        filt_slave.reset()
                        confirm_count = 0
                        start_time = time.time()
                        print(f'  -> Đã chốt OFFSET = {s["offset"]:.3f} rad. Vặn vô lăng đi nào!')

                    next_tick += DT
                    time.sleep(max(0, next_tick - time.time()))
                    continue

                if p_m is not None and p_s is not None:
                    virtual_target = p_m + s['offset']

                    if s['clamped']:
                        if s['clamp_dir'] == 1 and virtual_target > s['clamp_pos']:
                            with lock: s['offset'] = s['clamp_pos'] - p_m
                            virtual_target = s['clamp_pos']
                        elif s['clamp_dir'] == -1 and virtual_target < s['clamp_pos']:
                            with lock: s['offset'] = s['clamp_pos'] - p_m
                            virtual_target = s['clamp_pos']

                        target_p = s['clamp_pos']

                        if s['clamp_dir'] == 1 and virtual_target < (s['clamp_pos'] - 0.05):
                            with lock:
                                s['clamped'] = False
                                s['unlock_ignore'] = 50
                            print('\n  [MIMIC] Mở khóa! Tiếp tục bám vô lăng.\n')
                        elif s['clamp_dir'] == -1 and virtual_target > (s['clamp_pos'] + 0.05):
                            with lock:
                                s['clamped'] = False
                                s['unlock_ignore'] = 50
                            print('\n  [MIMIC] Mở khóa! Tiếp tục bám vô lăng.\n')

                    if not s['clamped']:
                        if virtual_target > limit_max:
                            with lock: s['offset'] = limit_max - p_m
                            virtual_target = limit_max
                        elif virtual_target < limit_min:
                            with lock: s['offset'] = limit_min - p_m
                            virtual_target = limit_min

                        if s['unlock_ignore'] > 0:
                            with lock: s['unlock_ignore'] -= 1
                            confirm_count = 0
                        elif filt_slave.ready() and avg_s > threshold and (v_s is None or abs(v_s) < 1.5):
                            confirm_count += 1
                            if confirm_count >= CONFIRM_TICKS:
                                with lock:
                                    s['clamped']   = True
                                    s['clamp_pos'] = p_s
                                    s['clamp_dir'] = 1 if (virtual_target - p_s) > 0 else -1
                                confirm_count = 0
                                print(f'\n  [MIMIC] KHÓA AN TOÀN! Chạm vật tại {p_s:.3f} rad (Lực: {avg_s:.2f} N.m)')
                                print(f'  -> Vặn vô lăng ngược lại để nhả kẹp.\n')
                        else:
                            confirm_count = 0

                        target_p = virtual_target

                    safe_target = max(limit_min, min(limit_max, target_p))
                    cmd_spd, error = pid.compute(target=safe_target, current=p_s)
                    if abs(error) < 0.015:
                        cmd_spd = 0.0
                    send_spd(bus, SLAVE_ID, cmd_spd)

                    t_now = time.time()
                    current_time_s = t_now - start_time
                    pn = gripper_norm(p_m)
                    log_writer.writerow([current_time_s, p_m, safe_target, p_s, cmd_spd, pn])
                    log_file.flush()

                    if t_now - last_print >= 0.2:
                        last_print = t_now
                        mode_str = "CLAMPED" if s['clamped'] else "MIMIC"
                        pn_display = gripper_norm(p_m)
                        print(f'  [{mode_str}] Vô lăng={p_m:+.3f} | '
                              f'Tay kẹp={p_s:+.3f} | '
                              f'Lực={avg_s:.2f} | '
                              f'norm={pn_display:.3f}')
                        if target_p <= limit_min or target_p >= limit_max:
                            print(f'          [CẢNH BÁO] Chạm Giới hạn cơ khí ({limit_min:.2f} - {limit_max:.2f})!')

            # ══════════════════════════════════════════════════════
            # 2. LOGIC KẸP TỰ ĐỘNG (AUTO SPEED)
            # ══════════════════════════════════════════════════════
            else:
                diff = target - s['actual']
                if abs(diff) <= RAMP:
                    s['actual'] = target
                else:
                    s['actual'] += RAMP if diff > 0 else -RAMP

                send_spd(bus, SLAVE_ID, -s['actual'] if REV else s['actual'])

                if warmup > 0:
                    with lock:
                        s['warmup'] = warmup - 1
                    confirm_count = 0

                contact = False
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
                            s['actual']  = 0.0
                        confirm_count = 0
                        print(f'\n  CHẠM VẬT! Lực tay kẹp = {avg_s:.3f} N·m')
                        print(f'  TỰ DỪNG!')
                else:
                    if not running:
                        confirm_count = 0

                t_now = time.time()
                if t_now - last_print >= 0.2:
                    last_print = t_now
                    st  = 'RUN' if running else 'STP'
                    r_s = f'{t_s_raw:+.3f}' if t_s_raw is not None else ' ---'
                    wm  = f' (warm {warmup})' if warmup > 0 else ''
                    cf  = f' [{confirm_count}/{CONFIRM_TICKS}]' if confirm_count > 0 else ''
                    pn_display = gripper_norm(p_m)
                    print(f'  [{st}] spd={s["actual"]:+5.2f}  raw={r_s}  '
                          f'avg={avg_s:.3f} N·m  norm={pn_display:.3f}{wm}{cf}')

            # ── Publish ROS gripper state mỗi tick ──
            with lock:
                clamped      = s['clamped']
                mimic_active = s['mimic']
                running_now  = s['running']

            # ── pos_norm: smooth theo vô lăng ──
            pos_norm_val = gripper_norm(p_m)

            contact_now = clamped

            if mimic_active:
                mode_val = 2
            elif running_now:
                mode_val = 1
            else:
                mode_val = 0

            ros_node.publish_state(p_m, p_s, pos_norm_val, avg_s, contact_now, mode_val)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            next_tick += DT
            time.sleep(max(0, next_tick - time.time()))

        print('\nDừng tay kẹp...')
        send_spd(bus, SLAVE_ID, 0.0)
        time.sleep(0.2)

        for mid in (MASTER_ID, SLAVE_ID):
            disable(bus, mid)
            set_mode(bus, mid, 0)

        log_file.close()
        print('Đã lưu file pid_log.csv thành công!')
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
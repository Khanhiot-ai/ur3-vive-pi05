#!/usr/bin/env python3
"""
control_robstride.py
Tay kẹp (06) bám Vô lăng (07) với Full PID. Ghi log đồ thị PID.
Tích hợp:
1. KHÓA AN TOÀN (Ly hợp điện tử) khi kẹp trúng vật.
2. AUTO-CALIBRATION: Đã fix lỗi "Ghost Collision" bằng Warmup.
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
        self.limit_i = limit_i
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, target, current):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.001: 
            dt = 0.02 

        error = target - current
        
        P = self.kp * error
        
        self.integral += error * dt
        self.integral = max(-self.limit_i, min(self.limit_i, self.integral))
        I = self.ki * self.integral
        
        D = self.kd * (error - self.prev_error) / dt
        
        out = P + I + D
        out = max(-self.limit_out, min(self.limit_out, out))
        
        self.prev_error = error
        self.last_time = now
        
        return out, error

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


# ── Mechanical limits ──
POS_OPEN_RAD  = 2.876   # limit_min mặc định (mở)
POS_CLOSE_RAD = 9.802   # limit_max mặc định (đóng)

def normalize_pos(p_rad, limit_min=POS_OPEN_RAD, limit_max=POS_CLOSE_RAD):
    """Map rad → [0=mở, 1=đóng] cho Pi0.5"""
    if p_rad is None: return 0.0
    norm = (p_rad - limit_min) / max(limit_max - limit_min, 0.001)
    return max(0.0, min(1.0, norm))


class GripperPublisher(Node):
    """ROS node publish gripper state cho record_all.py"""
    def __init__(self):
        super().__init__('gripper_publisher')
        self.pub = self.create_publisher(Float32MultiArray, '/gripper/state', 10)
        self.get_logger().info(
            "✅ /gripper/state [pos_master, pos_slave, pos_norm, torque, contact, mode]")

    def publish_state(self, p_master, p_slave, pos_norm, torque, contact, mode):
        """
        Publish 6 values:
          [0] pos_master: vị trí vô lăng 07 (rad) — ACTION input
          [1] pos_slave:  vị trí tay kẹp 06 (rad) — STATE robot
          [2] pos_norm:   tay kẹp normalized [0=mở, 1=đóng]
          [3] torque:     lực kẹp (N·m, moving avg)
          [4] contact:    1=chạm vật, 0=không
          [5] mode:       0=idle, 1=auto, 2=mimic, 3=calib
        """
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
        'running':   False,
        'speed':     args.speed,
        'actual':    0.0,
        'threshold': args.threshold,
        'warmup':    0,
        'quit':      False,
        'mimic':     False,        
        'mimic_init':False,        
        'offset':    0.0,
        'clamped':   False,
        'clamp_pos': 0.0,
        'clamp_dir': 1,
        'calibrating': False,
        'calib_phase': 0,
        'calib_warmup': 0, # [CẬP NHẬT] Biến chống chạm ma cho Calib
        'limit_1':     0.0,
        'limit_min':   2.876,
        'limit_max':   9.802
    }
    lock = threading.Lock()

    filt_slave = MovingAvg(FILTER_N)
    confirm_count = 0
    
    pid = PIDController(kp=13.2, ki=0.5, kd=1.0)

    # ── ROS init ──
    rclpy.init()
    ros_node = GripperPublisher()

    def show_help():
        print('─' * 65)
        print(f'  Tay kẹp: ID {SLAVE_ID} | Vô lăng: ID {MASTER_ID} | reverse={REV}')
        print(f'  Speed Tự Động: {s["speed"]:+.1f} rad/s | Ngưỡng Lực: {s["threshold"]} N·m')
        print(f'  Chốt an toàn hiện tại: [{s["limit_min"]:.3f} rad  <-->  {s["limit_max"]:.3f} rad]')
        print('─' * 65)
        print('  [c]   [Enter]   -> TỰ ĐỘNG DÒ TÌM GIỚI HẠN CƠ KHÍ (CALIB)')
        print('  [m]   [Enter]   -> Bật/Tắt MIMIC (Xoay 07 -> 06 bám theo)')
        print('  [Enter]         -> Bắt đầu / Dừng chế độ kẹp tự động')
        print('  [+/-] [Enter]   -> Tăng/giảm tốc độ kẹp tự động 0.5 rad/s')
        print('  [r]   [Enter]   -> Đổi chiều quay chế độ tự động')
        print('  [0]   [Enter]   -> Về vận tốc 0 rad/s')
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
                if cmd == 'c':
                    if s['mimic'] or s['running']:
                        print('\n  [LỖI] Phải tắt chế độ MIMIC hoặc Tự động trước khi Calib!\n')
                    elif s['calibrating']:
                        print('\n  [!] Hệ thống đang Calib rồi, vui lòng đợi...\n')
                    else:
                        s['calibrating'] = True
                        s['calib_phase'] = 1
                        s['calib_warmup']= 50 # Khởi động phase 1 bỏ qua 0.5s đầu
                        s['actual']      = 0.0
                        filt_slave.reset()
                        print('\n  [CALIB] TIẾN HÀNH DÒ TÌM GIỚI HẠN CƠ KHÍ...')
                        print('  -> Đang chạy ra để tìm Biên 1...')
                
                elif cmd == '':
                    if s['calibrating']: continue
                    if s['mimic']:
                        print('\n  [!] Đang ở chế độ Mimic, vui lòng tắt trước khi chạy tự động.\n')
                        continue
                    if not s['running']:
                        s['running'] = True
                        s['warmup']  = WARMUP
                        filt_slave.reset()
                        arrow = '<<' if s['speed'] < 0 else '>>'
                        print(f'\n  RUN TỰ ĐỘNG {arrow} {abs(s["speed"]):.1f} rad/s\n')
                    else:
                        s['running'] = False
                        s['actual']  = 0.0
                        print('\n  STOP Chạy Tự Động\n')

                elif cmd == '+':
                    sign = -1 if s['speed'] < 0 else 1
                    s['speed'] = round(sign * min(abs(s['speed']) + 0.5, SPD_MAX), 1)
                    print(f'  Speed -> {s["speed"]:+.1f} rad/s')

                elif cmd == '-':
                    sign = -1 if s['speed'] < 0 else 1
                    s['speed'] = round(sign * max(abs(s['speed']) - 0.5, 0.0), 1)
                    print(f'  Speed -> {s["speed"]:+.1f} rad/s')

                elif cmd == 'r':
                    s['speed'] = -s['speed']
                    arrow = '<<' if s['speed'] < 0 else '>>'
                    print(f'  Đổi chiều {arrow} {s["speed"]:+.1f} rad/s')

                elif cmd == '0':
                    s['speed'] = 0.0
                    print('  Speed -> 0.0')

                elif cmd == 'm':
                    if s['calibrating']: continue
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
                    s['calibrating'] = False
                    print('\n  Thoát...')

                elif cmd.startswith('t'):
                    try:
                        s['threshold'] = float(cmd[1:])
                        print(f'  Threshold -> {s["threshold"]:.4f} N·m')
                    except ValueError:
                        pass
                else:
                    try:
                        s['speed'] = round(max(-SPD_MAX, min(SPD_MAX, float(cmd))), 2)
                        print(f'  Speed -> {s["speed"]:+.1f} rad/s')
                    except ValueError:
                        show_help()

    threading.Thread(target=kb, daemon=True).start()

    log_file = open('pid_log.csv', mode='w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Time(s)', 'VoLang_07_Rad', 'Target_Rad', 'TayKep_06_Rad', 'PID_Speed_Out'])
    start_time = time.time()

    with can.Bus(interface=args.interface, channel=args.channel) as bus:
        print(f'Khởi tạo Tay Kẹp {SLAVE_ID} ở Mode 2 (Velocity)...')
        set_mode(bus, SLAVE_ID, 2)
        enable(bus, SLAVE_ID)
        
        print(f'Khởi tạo Vô Lăng {MASTER_ID} - Cắt điện để xoay tự do...')
        disable(bus, MASTER_ID)
        set_mode(bus, MASTER_ID, 0)
        
        print('Cả 2 motor sẵn sàng.\n')

        signal.signal(signal.SIGINT,
                      lambda sg, f: s.update({'quit': True, 'running': False, 'calibrating': False}))

        next_tick     = time.time()
        last_print    = 0.0
        confirm_count = 0

        while not s['quit']:
            with lock:
                running     = s['running']
                target      = s['speed'] if running else 0.0
                threshold   = s['threshold']
                warmup      = s['warmup']
                mimic       = s['mimic']
                mimic_init  = s['mimic_init']
                
                calibrating  = s['calibrating']
                calib_phase  = s['calib_phase']
                calib_warmup = s['calib_warmup']
                limit_min    = s['limit_min']
                limit_max    = s['limit_max']

            p_m, v_m, t_m, p_s, v_s, t_s_raw = read_both(bus, MASTER_ID, SLAVE_ID)
            
            avg_s = filt_slave.update(abs(t_s_raw) if t_s_raw is not None else None)

            # ==========================================
            # 1. LOGIC AUTO CALIBRATION
            # ==========================================
            if calibrating and p_s is not None:
                calib_spd_target = 1.5 if calib_phase == 1 else -1.5
                
                diff = calib_spd_target - s['actual']
                if abs(diff) <= RAMP:
                    s['actual'] = calib_spd_target
                else:
                    s['actual'] += RAMP if diff > 0 else -RAMP

                send_spd(bus, SLAVE_ID, s['actual'])

                # [CẬP NHẬT] Đếm ngược thời gian miễn nhiễm
                if calib_warmup > 0:
                    with lock:
                        s['calib_warmup'] -= 1
                    confirm_count = 0

                # Chỉ cho phép đo lực khi đã hết thời gian miễn nhiễm
                if calib_warmup == 0 and abs(s['actual']) > 0.05 and filt_slave.ready():
                    if avg_s > threshold:
                        confirm_count += 1
                    else:
                        confirm_count = 0

                    if confirm_count >= CONFIRM_TICKS:
                        send_spd(bus, SLAVE_ID, 0.0) 
                        confirm_count = 0
                        
                        if calib_phase == 1:
                            with lock:
                                s['limit_1'] = p_s
                                s['calib_phase'] = 2
                                # [CẬP NHẬT] Cấp 100 ticks (1 giây) để động cơ lùi khỏi vách kẹt
                                s['calib_warmup'] = 100 
                                s['actual'] = 0.0
                            print(f'\n  [CALIB] Đã chạm cứng Biên 1 tại: {p_s:.3f} rad.')
                            print('  -> Đang đảo chiều để tìm Biên 2 (Bỏ qua đo lực 1s đầu)...')
                            filt_slave.reset()
                        
                        elif calib_phase == 2:
                            limit_2 = p_s
                            print(f'\n  [CALIB] Đã chạm cứng Biên 2 tại: {limit_2:.3f} rad.')
                            
                            SAFE_BUFFER = 0.05
                            new_min = min(s['limit_1'], limit_2) + SAFE_BUFFER
                            new_max = max(s['limit_1'], limit_2) - SAFE_BUFFER
                            
                            with lock:
                                s['limit_min'] = new_min
                                s['limit_max'] = new_max
                                s['calibrating'] = False
                                s['calib_phase'] = 0
                                s['actual'] = 0.0
                                
                            print(f'  [CALIB] HOÀN TẤT THÀNH CÔNG!')
                            print(f'  => CHỐT AN TOÀN ĐƯỢC CẬP NHẬT: Min = {new_min:.3f} | Max = {new_max:.3f}\n')

                t_now = time.time()
                if t_now - last_print >= 0.2:
                    last_print = t_now
                    # Cập nhật print để bạn thấy lúc nào nó đang Warmup
                    wm_str = f" (Warmup {calib_warmup})" if calib_warmup > 0 else ""
                    print(f'  [CALIB] Đang tìm biên {calib_phase}... pos={p_s:.3f} | Lực={avg_s:.2f} N.m{wm_str}')

            # ==========================================
            # 2. LOGIC BÁM VÔ LĂNG (MIMIC)
            # ==========================================
            elif mimic:
                if mimic_init:
                    if p_m is not None and p_s is not None:
                        with lock:
                            s['offset'] = p_s - p_m
                            s['mimic_init'] = False
                            s['clamped'] = False
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

                    if not s['clamped']:
                        if filt_slave.ready() and avg_s > threshold:
                            confirm_count += 1
                            if confirm_count >= CONFIRM_TICKS:
                                with lock:
                                    s['clamped'] = True
                                    s['clamp_pos'] = p_s
                                    s['clamp_dir'] = 1 if (virtual_target - p_s) > 0 else -1
                                confirm_count = 0
                                print(f'\n  [MIMIC] KHÓA AN TOÀN! Chạm vật tại {p_s:.3f} rad (Lực: {avg_s:.2f} N.m)')
                                print(f'  -> Vặn vô lăng ngược lại để nhả kẹp.\n')
                        else:
                            confirm_count = 0

                    if s['clamped']:
                        target_p = s['clamp_pos']
                        if s['clamp_dir'] == 1 and virtual_target < (s['clamp_pos'] - 0.05):
                            with lock: s['clamped'] = False
                            print('\n  [MIMIC] Mở khóa! Tiếp tục bám vô lăng.\n')
                        elif s['clamp_dir'] == -1 and virtual_target > (s['clamp_pos'] + 0.05):
                            with lock: s['clamped'] = False
                            print('\n  [MIMIC] Mở khóa! Tiếp tục bám vô lăng.\n')
                    else:
                        target_p = virtual_target

                    safe_target = max(limit_min, min(limit_max, target_p)) 
                    
                    cmd_spd, error = pid.compute(target=safe_target, current=p_s)
                    
                    if abs(error) < 0.015: 
                        cmd_spd = 0.0
                        
                    send_spd(bus, SLAVE_ID, cmd_spd)
                    
                    t_now = time.time()
                    current_time_s = t_now - start_time
                    log_writer.writerow([current_time_s, p_m, safe_target, p_s, cmd_spd])
                    log_file.flush() 

                    if t_now - last_print >= 0.2:
                        last_print = t_now
                        mode_str = "CLAMPED" if s['clamped'] else "MIMIC"
                        print(f'  [{mode_str}] Vô lăng={p_m:+.3f} | Tay kẹp={p_s:+.3f} | Lực={avg_s:.2f}')
                        
                        if target_p < limit_min or target_p > limit_max:
                            print(f'          [CẢNH BÁO] Chạm Giới hạn cơ khí ({limit_min:.2f} - {limit_max:.2f})!')

            # ==========================================
            # 3. LOGIC KẸP TỰ ĐỘNG (AUTO SPEED)
            # ==========================================
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
                        print(f'\n  CHẠM VẬT!'
                              f'  Lực tay kẹp = {avg_s:.3f} N·m'
                              f'  (ngưỡng={threshold:.3f})')
                        print(f'  TỰ DỪNG!')
                        print(f'  [r]+Enter = mở ra  |  [Enter] = kẹp lại\n')
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
                    flg = '  CONTACT!' if contact else ''
                    print(f'  [{st}] spd={s["actual"]:+5.2f}'
                          f'  raw={r_s}'
                          f'  avg={avg_s:.3f} N·m'
                          f'{wm}{cf}{flg}')

            # ── Publish ROS gripper state mỗi tick ──
            with lock:
                lmin = s['limit_min']
                lmax = s['limit_max']
                clamped = s['clamped']
                calib = s['calibrating']
                mimic_active = s['mimic']
                running_now = s['running']

            pos_norm_val = normalize_pos(p_s, lmin, lmax)
            contact_now = clamped
            if calib:
                mode_val = 3
            elif mimic_active:
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
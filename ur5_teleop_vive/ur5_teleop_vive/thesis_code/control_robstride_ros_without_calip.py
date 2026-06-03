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
  - TƯƠNG ĐỐI: mốc 0.0 chốt khi bật MIMIC, xoay MASTER_SPAN rad → 1.0
  - Calip làm pose_master đổi cũng KHÔNG sao (mỗi lần MIMIC chốt mốc mới)
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

def set_zero(bus, mid):
    """Set vi tri hien tai lam moc 0 rad (comm type 6). Motor phai DISABLED."""
    _send(bus, (6 << 24) | (0xFD << 8) | mid, [1, 0, 0, 0, 0, 0, 0, 0])
    time.sleep(0.1)

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
# CƠ CHẾ 2 MỐC (chốt thủ công, chính xác nhất):
#   - Mở vô lăng hết cỡ, gõ 'o' → chốt master_open  = vị trí này (norm 0.0)
#   - Kẹp vô lăng hết cỡ, gõ 'p' → chốt master_close = vị trí này (norm 1.0)
#   - pos_norm = (p_m - master_open) / (master_close - master_open)
#
# Ưu điểm: KHÔNG cần đoán, KHÔNG phụ thuộc MASTER_SPAN cố định.
#          Calip làm pose_master đổi bao nhiêu cũng chốt lại được.
#
# Fallback: nếu chưa chốt 'o'/'p', dùng master_zero (lúc bật MIMIC) + MASTER_SPAN.
# ══════════════════════════════════════════════════════════════════════
MASTER_SPAN = 3.72     # rad — fallback span nếu chưa chốt mốc 'o'/'p'

def gripper_norm(p_m, master_open, master_close, master_zero):
    """
    pos_norm từ vị trí vô lăng.
    Ưu tiên dùng 2 mốc open/close (chốt bằng phím 'o'/'p').
    Nếu chưa chốt đủ → fallback master_zero + MASTER_SPAN.
    """
    if p_m is None:
        return 0.0
    # Cách 1: có cả 2 mốc open + close
    if master_open is not None and master_close is not None:
        span = master_close - master_open
        if abs(span) < 0.01:
            return 0.0
        norm = (p_m - master_open) / span
        return float(max(0.0, min(1.0, norm)))
    # Cách 2 (fallback): mốc zero + span cố định
    if master_zero is not None:
        norm = (p_m - master_zero) / MASTER_SPAN
        return float(max(0.0, min(1.0, norm)))
    return 0.0


# ── Mechanical limits (vi tri SLAVE / tay kep) ──
# KHONG con gia tri mac dinh cung. Gioi han hanh trinh DUOC XAC DINH
# qua tu dong calip (phim 'c'): tim 2 bien -> ve giua -> zero tai giua.
# Truoc khi calip, limit_min/max = None va KHONG cho phep MIMIC / kep tu dong.


class GripperPublisher(Node):
    def __init__(self):
        super().__init__('gripper_publisher')
        self.pub = self.create_publisher(Float32MultiArray, '/gripper/state', 10)
        self.get_logger().info(
            "✅ /gripper/state [pos_master, pos_slave, pos_norm, torque, contact, mode]"
        )
        self.get_logger().info(
            f"   pos_norm: TƯƠNG ĐỐI — mốc 0.0 chốt lúc bật MIMIC, "
            f"xoay {MASTER_SPAN} rad → 1.0"
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

    # ── Tu dong calip hanh trinh (phim 'c') ──
    CALIB_SPD       = 1.0    # rad/s — toc do quay khi calip (cham de an toan)
    CALIB_THRESHOLD = 0.2    # N·m — torque bao da cham bien cung
    CALIB_CONFIRM   = 5       # so tick lien tiep xac nhan cham bien
    CALIB_MARGIN    = 0.15    # rad — lui vao tu bien lam gioi han mem

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
        'master_zero':   None,    # mốc vô lăng lúc bật MIMIC (fallback)
        'master_open':   None,    # mốc vô lăng MỞ hết (chốt bằng phím 'o') → norm 0.0
        'master_close':  None,    # mốc vô lăng KẸP hết (chốt bằng phím 'p') → norm 1.0
        'last_p_m':      None,    # vị trí vô lăng mới nhất (cho phím o/p đọc)
        'clamped':       False,
        'clamp_pos':     0.0,      # vi tri slave luc cham vat
        'clamp_m_pos':   0.0,      # vi tri master luc cham vat (de do chuyen dong that)
        'clamp_dir':     1,
        'limit_min':     None,     # se duoc dat sau khi CALIP
        'limit_max':     None,     # se duoc dat sau khi CALIP
        'unlock_ignore': 0,
        'calib':         False,    # dang chay tu dong calip
        'calibrated':    False,    # da calip xong it nhat 1 lan chua
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
        if s['calibrated']:
            print(f'  Hành trình tay kẹp: [{s["limit_min"]:.3f} <--> {s["limit_max"]:.3f}] rad (đã calip)')
        else:
            print(f'  Hành trình tay kẹp: CHƯA CALIP — bấm [c] trước khi dùng')
        if s['master_open'] is not None and s['master_close'] is not None:
            sp = s['master_close'] - s['master_open']
            print(f'  pos_norm: ĐÃ CHỐT 2 mốc — open={s["master_open"]:+.2f} close={s["master_close"]:+.2f} (span={sp:+.2f})')
        else:
            print(f'  pos_norm: CHƯA chốt mốc — mở hết bấm [o], kẹp hết bấm [p]')
        print('─' * 65)
        print('  [m]   [Enter]   -> Bật/Tắt MIMIC (Xoay 07 -> 06 bám theo)')
        print('  [o]   [Enter]   -> Chốt mốc MỞ hết (pos_norm = 0.0)')
        print('  [p]   [Enter]   -> Chốt mốc KẸP hết (pos_norm = 1.0)')
        print('  [Enter]         -> Bắt đầu / Dừng chế độ kẹp tự động')
        print('  [+/-] [Enter]   -> Tăng/giảm tốc độ kẹp tự động 0.5 rad/s')
        print('  [c]   [Enter]   -> Tự động calip hành trình (tìm 2 biên, về giữa)')
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
                    if not s['calibrated']:
                        print('\n  [!] Chưa calip! Bấm [c] để calip hành trình trước.\n')
                        continue
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
                    if not s['calibrated']:
                        print('\n  [!] Chưa calip! Bấm [c] để calip hành trình trước.\n')
                    elif not s['mimic']:
                        s['running']    = False
                        s['actual']     = 0.0
                        s['mimic']      = True
                        s['mimic_init'] = True
                        print(f'\n  [+] KÍCH HOẠT MIMIC: Đang lấy thông số...')
                    else:
                        s['mimic'] = False
                        print(f'\n  [-] TẮT MIMIC: Khôi phục lại trạng thái chờ.')
                elif cmd == 'c':
                    if s['mimic'] or s['running']:
                        print('\n  [!] Tắt MIMIC / kẹp tự động trước khi calip.\n')
                    elif s['calib']:
                        print('\n  [!] Đang calip rồi.\n')
                    else:
                        s['calib'] = True
                        print('\n  [+] BẮT ĐẦU CALIP HÀNH TRÌNH...')
                elif cmd == 'o':
                    # Chốt mốc MỞ (norm 0.0) tại vị trí vô lăng hiện tại
                    pm = s['last_p_m']
                    if pm is None:
                        print('\n  [!] Chưa đọc được vị trí vô lăng. Bật MIMIC (m) trước.\n')
                    else:
                        s['master_open'] = pm
                        print(f'\n  [O] Đã chốt mốc MỞ = {pm:+.3f} rad → norm 0.0')
                        if s['master_close'] is not None:
                            print(f'      Span = {s["master_close"] - pm:+.3f} rad. pos_norm sẵn sàng!\n')
                        else:
                            print(f'      Giờ kẹp hết cỡ rồi bấm [p].\n')
                elif cmd == 'p':
                    # Chốt mốc KẸP (norm 1.0) tại vị trí vô lăng hiện tại
                    pm = s['last_p_m']
                    if pm is None:
                        print('\n  [!] Chưa đọc được vị trí vô lăng. Bật MIMIC (m) trước.\n')
                    else:
                        s['master_close'] = pm
                        print(f'\n  [P] Đã chốt mốc KẸP = {pm:+.3f} rad → norm 1.0')
                        if s['master_open'] is not None:
                            print(f'      Span = {pm - s["master_open"]:+.3f} rad. pos_norm sẵn sàng!\n')
                        else:
                            print(f'      Giờ mở hết cỡ rồi bấm [o].\n')
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

        # ── Tu dong calip hanh trinh ──
        def calib_find_edge(direction):
            """Quay slave theo 'direction' (+1/-1) toi khi cham bien cung
            (torque vuot nguong + slave khung lai). Tra ve pos luc cham, hoac
            None neu bi huy."""
            filt = MovingAvg(FILTER_N)
            cc   = 0
            warm = WARMUP
            last_pos = None
            while not s['quit'] and s['calib']:
                send_spd(bus, SLAVE_ID, CALIB_SPD * direction)
                _, _, _, p_s_l, v_s_l, t_s_l = read_both(bus, MASTER_ID, SLAVE_ID)
                avg = filt.update(abs(t_s_l) if t_s_l is not None else None)
                if p_s_l is not None:
                    last_pos = p_s_l
                if warm > 0:
                    warm -= 1
                elif filt.ready() and avg > CALIB_THRESHOLD \
                        and (v_s_l is None or abs(v_s_l) < 1.0):
                    cc += 1
                    if cc >= CALIB_CONFIRM:
                        send_spd(bus, SLAVE_ID, 0.0)
                        return last_pos
                else:
                    cc = 0
                time.sleep(DT)
            send_spd(bus, SLAVE_ID, 0.0)
            return None

        def do_calib():
            """Tim 2 bien, ve giua, dat giua = 0 rad, cap nhat limit_min/max."""
            print('  -> Đang tìm biên thứ nhất (chiều +)...')
            edge_a = calib_find_edge(+1)
            if edge_a is None:
                print('  [!] Calip bị hủy.\n')
                with lock: s['calib'] = False
                return
            print(f'  -> Biên + tại {edge_a:+.3f} rad. Tìm biên thứ hai (chiều -)...')
            time.sleep(0.3)
            edge_b = calib_find_edge(-1)
            if edge_b is None:
                print('  [!] Calip bị hủy.\n')
                with lock: s['calib'] = False
                return
            print(f'  -> Biên - tại {edge_b:+.3f} rad.')

            lo, hi = sorted([edge_a, edge_b])
            mid    = (lo + hi) / 2.0
            travel = hi - lo
            print(f'  -> Hành trình = {travel:.3f} rad | Giữa = {mid:.3f} rad')

            print('  -> Đưa slave về giữa hành trình...')
            t_end = time.time() + 8.0
            while not s['quit'] and time.time() < t_end:
                _, _, _, p_s_l, _, _ = read_both(bus, MASTER_ID, SLAVE_ID)
                if p_s_l is None:
                    time.sleep(DT); continue
                err = mid - p_s_l
                if abs(err) < 0.02:
                    break
                v = max(-CALIB_SPD, min(CALIB_SPD, err * 3.0))
                send_spd(bus, SLAVE_ID, v)
                time.sleep(DT)
            send_spd(bus, SLAVE_ID, 0.0)
            time.sleep(0.2)

            # Dat giua = 0 rad: zero encoder tai day -> hanh trinh doi xung quanh 0.
            disable(bus, SLAVE_ID)
            set_mode(bus, SLAVE_ID, 0)
            set_zero(bus, SLAVE_ID)
            set_mode(bus, SLAVE_ID, 2)
            enable(bus, SLAVE_ID)

            half = travel / 2.0
            with lock:
                s['limit_min']  = -half + CALIB_MARGIN
                s['limit_max']  = +half - CALIB_MARGIN
                s['calibrated'] = True
                s['calib']      = False
            print(f'  ✅ CALIP XONG! Hành trình mới: '
                  f'[{-half + CALIB_MARGIN:+.3f} <-> {+half - CALIB_MARGIN:+.3f}] rad')
            print(f'     (Giữa = 0 rad, đã zero encoder tại giữa.)\n')

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

            # ── TU DONG CALIP (blocking) — uu tien cao nhat ──
            if s['calib']:
                do_calib()
                next_tick = time.time()
                continue

            p_m, v_m, t_m, p_s, v_s, t_s_raw = read_both(bus, MASTER_ID, SLAVE_ID)
            avg_s = filt_slave.update(abs(t_s_raw) if t_s_raw is not None else None)

            # Lưu vị trí vô lăng mới nhất cho phím 'o'/'p' đọc
            if p_m is not None:
                with lock:
                    s['last_p_m'] = p_m

            # ══════════════════════════════════════════════════════
            # 1. LOGIC BÁM VÔ LĂNG (MIMIC)
            # ══════════════════════════════════════════════════════
            if mimic:
                if mimic_init:
                    if p_m is not None and p_s is not None:
                        with lock:
                            s['offset']     = p_s - p_m
                            s['master_zero']= p_m    # chốt mốc vô lăng = mở (0.0)
                            s['mimic_init'] = False
                            s['clamped']    = False
                            s['unlock_ignore'] = 0
                        pid.reset()
                        filt_slave.reset()
                        confirm_count = 0
                        start_time = time.time()
                        print(f'  -> Đã chốt OFFSET = {s["offset"]:.3f} rad.')
                        if s['master_open'] is None or s['master_close'] is None:
                            print(f'  -> CHỐT MỐC pos_norm: mở hết bấm [o], kẹp hết bấm [p].')
                        else:
                            sp = s['master_close'] - s['master_open']
                            print(f'  -> pos_norm đã có mốc (span={sp:+.2f} rad). Vặn đi nào!')

                    next_tick += DT
                    time.sleep(max(0, next_tick - time.time()))
                    continue

                if p_m is not None and p_s is not None:
                    virtual_target = p_m + s['offset']

                    PI_8 = 8.0 * math.pi
                    PI_4 = 4.0 * math.pi

                    if s['clamped']:
                        # Slave dung im tai vi tri da khoa
                        target_p = s['clamp_pos']

                        # Do chuyen dong THAT cua master so voi luc cham vat.
                        # Khong dung virtual_target (bi ghim) -> on dinh, khong
                        # con tinh trang "luc nha duoc luc khong".
                        raw = p_m - s['clamp_m_pos']
                        while raw >  PI_4: raw -= PI_8
                        while raw < -PI_4: raw += PI_8
                        penetration = raw * s['clamp_dir']

                        # Van nguoc ra (chieu mo) qua 0.05 rad -> mo khoa.
                        if penetration < -0.05:
                            with lock:
                                s['clamped'] = False
                                s['unlock_ignore'] = 50
                                # Chot lai offset tai vi tri hien tai de slave
                                # bam tiep tu day, tranh nhay.
                                s['offset'] = p_s - p_m
                            print('\n  [MIMIC] Mở khóa! Tiếp tục bám vô lăng.\n')

                    if not s['clamped']:
                        # Gioi han hanh trinh: ghim virtual_target trong bien.
                        # Neu vuot, cap nhat offset de slave dung tai bien,
                        # khong tong cung vao gioi han co khi.
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
                                # Gripper CHI kep vat khi DONG (p_s tang = chieu +).
                                # Nen chieu kep luon la +1, khong can doan tu chuyen
                                # dong (von bi nhieu khi overshoot). Mo khoa = van
                                # master nguoc lai (penetration am).
                                with lock:
                                    s['clamped']     = True
                                    s['clamp_pos']   = p_s
                                    s['clamp_m_pos'] = p_m   # luu vi tri master luc cham
                                    s['clamp_dir']   = 1     # co dinh: dong = chieu +
                                confirm_count = 0
                                print(f'\n  [MIMIC] KHÓA AN TOÀN! Chạm vật tại {p_s:.3f} rad '
                                      f'(Lực: {avg_s:.2f} N.m)')
                                print(f'  -> Vặn vô lăng ngược lại để nhả kẹp.\n')
                        else:
                            confirm_count = 0

                        target_p = virtual_target

                    # Chot bao ve cuoi: dam bao lenh xuong slave luon trong bien
                    safe_target = max(limit_min, min(limit_max, target_p))
                    cmd_spd, error = pid.compute(target=safe_target, current=p_s)
                    if abs(error) < 0.015:
                        cmd_spd = 0.0
                    send_spd(bus, SLAVE_ID, cmd_spd)

                    t_now = time.time()
                    current_time_s = t_now - start_time
                    pn = gripper_norm(p_m, s['master_open'], s['master_close'], s['master_zero'])
                    log_writer.writerow([current_time_s, p_m, safe_target, p_s, cmd_spd, pn])
                    log_file.flush()

                    if t_now - last_print >= 0.2:
                        last_print = t_now
                        mode_str = "CLAMPED" if s['clamped'] else "MIMIC  "
                        pn_disp = gripper_norm(p_m, s['master_open'], s['master_close'], s['master_zero'])
                        if s['clamped']:
                            # Tinh penetration de thay tien trinh mo khoa
                            raw_d = p_m - s['clamp_m_pos']
                            while raw_d >  PI_4: raw_d -= PI_8
                            while raw_d < -PI_4: raw_d += PI_8
                            pen_disp = raw_d * s['clamp_dir']
                            print(f'  [{mode_str}] pose_master={p_m:+.3f} | '
                                  f'pose_slave={p_s:+.3f} | '
                                  f'force_slave={avg_s:.2f} | '
                                  f'norm={pn_disp:.3f} | '
                                  f'penetration={pen_disp:+.3f}')
                        else:
                            print(f'  [{mode_str}] pose_master={p_m:+.3f} | '
                                  f'pose_slave={p_s:+.3f} | '
                                  f'force_slave={avg_s:.2f} | '
                                  f'norm={pn_disp:.3f} | '
                                  f'cmd_spd={cmd_spd:+.2f}')
                        if not s['clamped'] and (target_p <= limit_min or target_p >= limit_max):
                            print(f'          [GIỚI HẠN] Chạm biên hành trình ({limit_min:.2f} - {limit_max:.2f})')

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
                    print(f'  [{st}] spd={s["actual"]:+5.2f}  raw={r_s}  '
                          f'force_slave={avg_s:.3f} N·m{wm}{cf}')

            # ── Publish ROS gripper state mỗi tick ──
            with lock:
                clamped      = s['clamped']
                mimic_active = s['mimic']
                running_now  = s['running']

            # ── pos_norm: smooth theo vô lăng (tương đối với mốc lúc bật MIMIC) ──
            pos_norm_val = gripper_norm(p_m, s['master_open'], s['master_close'], s['master_zero'])

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
    ap.add_argument('--threshold', type=float, default=0.18)
    ap.add_argument('--reverse',   action='store_true')
    ap.add_argument('--interface', default='socketcan')
    ap.add_argument('--channel',   default='can0')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
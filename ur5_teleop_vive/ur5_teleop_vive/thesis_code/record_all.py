#!/usr/bin/env python3
"""
DATASET RECORDER — HDF5 cho Pi0.5 / LeRobot
Format giống lerobot/berkeley_autolab_ur5

Schema HDF5 (giống lerobot/berkeley_autolab_ur5):
    dataset/<task>.hdf5
    └── data/
        ├── demo_0/
        │   ├── obs/
        │   │   ├── image         (T, 480, 640, 3) uint8   ← Realsense front
        │   │   ├── wrist_image   (T, 480, 640, 3) uint8   ← Realsense wrist
        │   │   ├── digit_left     (T, 240, 320, 3) uint8   ← DIGIT tactile trái
        │   │   ├── digit_right    (T, 240, 320, 3) uint8   ← DIGIT tactile phải
        │   │   └── state         (T, 8)  float32
        │   │         [ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper]
        │   │         ← Cartesian TCP pose + gripper norm
        │   ├── actions           (T, 7)  float32
        │   │     [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
        │   │     ← DELTA Cartesian + gripper norm
        │   └── attrs: success, n_frames, fps, task, recorded_at
        └── demo_1/, demo_2/, ...

Chạy:
    python3 record_all.py --task pick_cube --fps 20
GUI:
    SPACE = rec / stop
    S     = lưu SUCCESS
    F     = lưu FAIL
    Q     = quit
"""

import os, sys, time, argparse, threading
import numpy as np
import cv2
import h5py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

try:
    from ur5_teleop_vive.msg import Xyzrpy
    HAS_XYZRPY = True
except ImportError:
    HAS_XYZRPY = False

# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
# Image: 640×480 (giống lerobot/berkeley_autolab_ur5)
# cv2.resize lấy (W, H) → output shape là (H, W, 3) = (480, 640, 3)
IMAGE_SIZE   = (640, 480)   # (width, height) cho cv2.resize
IMAGE_SHAPE  = (480, 640, 3)   # (H, W, C) — shape trong HDF5
DISPLAY_SIZE = (640, 360)

# State: [ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper] — giống berkeley
STATE_DIM    = 8

# Action: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper] — delta Cartesian
ACTION_DIM   = 7

# DIGIT tactile images @60Hz — PORTRAIT (320 cao × 240 rộng)
# Publisher đã xoay 90° ra portrait → recorder lưu NGUYÊN XI, KHÔNG resize.
# V-JEPA cần cửa sổ 4 frame @60fps ở (H=320, W=240) RGB.
DIGIT_SHAPE  = (320, 240, 3)   # (H, W, C) portrait — shape trong HDF5

TOPIC_FRONT   = '/camera_front/camera/color/image_raw'
TOPIC_WRIST   = '/camera_wrist/camera/color/image_raw'
TOPIC_JOINTS  = '/ur_joint_states'
TOPIC_ACTUAL  = '/ur_actual_pose'      # Xyzrpy: x,y,z,roll,pitch,yaw
TOPIC_TARGET  = '/ur_target_pose'      # PoseStamped: xyz + quat
TOPIC_GRIPPER = '/gripper/state'
TOPIC_DIGIT_L = '/digit_left/image_raw'    # DIGIT ngón trái
TOPIC_DIGIT_R = '/digit_right/image_raw'   # DIGIT ngón phải
# ════════════════════════════════════════════════════════


class HDF5Recorder(Node):
    def __init__(self, task_name, fps):
        super().__init__('hdf5_recorder')
        self.bridge    = CvBridge()
        self.task      = task_name
        self.fps       = fps
        self.dt        = 1.0 / fps

        # Latest sensor data
        self._lock       = threading.Lock()
        self.front_img   = None
        self.wrist_img   = None
        self.digit_l_img = None
        self.digit_r_img = None
        self.actual_pose = None   # dict: x,y,z,roll,pitch,yaw
        self.target_pose = None   # dict: x,y,z,qx,qy,qz,qw
        self.gripper     = None   # dict: pos_norm, torque, contact, mode
        self.front_count = 0
        self.wrist_count = 0
        self.digit_l_count = 0
        self.digit_r_count = 0

        # Recording buffers
        self.recording   = False
        self.frame_idx   = 0
        self.t_start     = 0.0
        self.buf_front   = []
        self.buf_wrist   = []
        self.buf_digit_l = []        # @60Hz (append trong callback)
        self.buf_digit_r = []        # @60Hz
        self.buf_digit_l_ts = []     # timestamp 60Hz cho digit trái
        self.buf_digit_r_ts = []     # timestamp 60Hz cho digit phải
        self.buf_state   = []
        self.buf_tick_ts = []        # timestamp 20Hz mỗi tick
        # Forward delta: lưu pose mỗi tick để tính action[t]=pose(t+1)-pose(t)
        self.buf_pose    = []        # actual pose mỗi tick (cho forward delta)

        self.ep_total    = 0
        self.ep_success  = 0

        # HDF5
        self.h5_path = os.path.join("dataset", f"{task_name}.hdf5")
        os.makedirs("dataset", exist_ok=True)
        self._open_or_init_hdf5()

        # ── QoS ──
        qos_cam = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_robot = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # ── Subscribers ──
        self.create_subscription(Image, TOPIC_FRONT, self._cb_front, qos_cam)
        self.create_subscription(Image, TOPIC_WRIST, self._cb_wrist, qos_cam)
        self.create_subscription(Image, TOPIC_DIGIT_L, self._cb_digit_l, qos_cam)
        self.create_subscription(Image, TOPIC_DIGIT_R, self._cb_digit_r, qos_cam)
        if HAS_XYZRPY:
            self.create_subscription(Xyzrpy,      TOPIC_ACTUAL,  self._cb_actual,  qos_robot)
        self.create_subscription(PoseStamped,      TOPIC_TARGET,  self._cb_target,  qos_robot)
        self.create_subscription(Float32MultiArray, TOPIC_GRIPPER, self._cb_gripper, qos_robot)

        # ── Publishers điều khiển tự động ──
        self.pub_teleop = self.create_publisher(Bool, '/teleop_enable', 10)
        self.pub_home   = self.create_publisher(Bool, '/auto_home', 10)

        self.tick_timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info("══════════════════════════════════════════════")
        self.get_logger().info(f"  HDF5 RECORDER  task={task_name}  fps={fps}")
        self.get_logger().info(f"  File : {self.h5_path}")
        self.get_logger().info(f"  Demo : {self.next_demo_id} đã có")
        self.get_logger().info(f"  State  ({STATE_DIM}): [ee_x, ee_y, ee_z, qx, qy, qz, qw, grip]")
        self.get_logger().info(f"  Action ({ACTION_DIM}): [dx, dy, dz, d_roll, d_pitch, d_yaw, grip]")
        self.get_logger().info("  SPACE=rec  S=success  F=fail  Q=quit")
        self.get_logger().info("══════════════════════════════════════════════")

    # ──────────────────────────────────────────────────────
    # HDF5 init
    # ──────────────────────────────────────────────────────
    def _open_or_init_hdf5(self):
        need_init = True
        if os.path.exists(self.h5_path):
            try:
                with h5py.File(self.h5_path, "r") as f:
                    if "data" in f:
                        nums = [int(k.split("_")[1]) for k in f["data"]
                                if k.startswith("demo_")]
                        self.next_demo_id = max(nums, default=-1) + 1
                        need_init = False
                    else:
                        # File tồn tại nhưng thiếu group 'data' → hỏng
                        self.get_logger().warn(
                            f"File {self.h5_path} thiếu group 'data' — tạo lại group.")
            except Exception as e:
                # File hỏng hoàn toàn → backup rồi tạo mới
                self.get_logger().warn(f"File hỏng ({e}) — backup .corrupt rồi tạo mới.")
                try:
                    os.rename(self.h5_path, self.h5_path + ".corrupt")
                except Exception:
                    os.remove(self.h5_path)

        if need_init:
            # Tạo group 'data' (mode 'a' để không xóa demo cũ nếu file chỉ thiếu group)
            mode = "a" if os.path.exists(self.h5_path) else "w"
            with h5py.File(self.h5_path, mode) as f:
                if "data" not in f:
                    g = f.create_group("data")
                    g.attrs["task"]       = self.task
                    g.attrs["fps"]        = self.fps
                    g.attrs["state_dim"]  = STATE_DIM
                    g.attrs["action_dim"] = ACTION_DIM
                    g.attrs["image_shape"]= list(IMAGE_SHAPE)
                    g.attrs["robot"]      = "UR3"
                    g.attrs["state_keys"] = "ee_x,ee_y,ee_z,qx,qy,qz,qw,gripper"
                    g.attrs["action_keys"]= "dx,dy,dz,d_roll,d_pitch,d_yaw,gripper"
            self.next_demo_id = 0

    # ──────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────
    def _cb_front(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self.front_img = img
                self.front_count += 1
        except Exception as e:
            self.get_logger().warn(f"front: {e}", throttle_duration_sec=2.0)

    def _cb_wrist(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self.wrist_img = img
                self.wrist_count += 1
        except Exception as e:
            self.get_logger().warn(f"wrist: {e}", throttle_duration_sec=2.0)

    def _cb_digit_l(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # BGR→RGB, lưu nguyên xi (publisher đã ra portrait 320×240)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ts  = self.get_clock().now().nanoseconds * 1e-9
            with self._lock:
                self.digit_l_img = img        # cho GUI (BGR)
                self.digit_l_count += 1
                # DUAL-RATE: nếu đang record → append frame 60Hz vào buffer riêng
                if self.recording:
                    self.buf_digit_l.append(rgb)
                    self.buf_digit_l_ts.append(ts)
        except Exception as e:
            self.get_logger().warn(f"digit_l: {e}", throttle_duration_sec=2.0)

    def _cb_digit_r(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ts  = self.get_clock().now().nanoseconds * 1e-9
            with self._lock:
                self.digit_r_img = img        # cho GUI (BGR)
                self.digit_r_count += 1
                if self.recording:
                    self.buf_digit_r.append(rgb)
                    self.buf_digit_r_ts.append(ts)
        except Exception as e:
            self.get_logger().warn(f"digit_r: {e}", throttle_duration_sec=2.0)

    def _cb_actual(self, msg):
        """Nhận TCP pose thực tế (Xyzrpy): x,y,z,roll,pitch,yaw"""
        with self._lock:
            self.actual_pose = {
                "x":     float(msg.x),
                "y":     float(msg.y),
                "z":     float(msg.z),
                "roll":  float(msg.roll),
                "pitch": float(msg.pitch),
                "yaw":   float(msg.yaw),
            }

    def _cb_target(self, msg):
        """Nhận TCP target (PoseStamped): xyz + quaternion"""
        with self._lock:
            self.target_pose = {
                "x":  float(msg.pose.position.x),
                "y":  float(msg.pose.position.y),
                "z":  float(msg.pose.position.z),
                "qx": float(msg.pose.orientation.x),
                "qy": float(msg.pose.orientation.y),
                "qz": float(msg.pose.orientation.z),
                "qw": float(msg.pose.orientation.w),
            }

    def _cb_gripper(self, msg):
        """[pos_master, pos_slave, pos_norm, torque, contact, mode]"""
        if len(msg.data) >= 6:
            with self._lock:
                self.gripper = {
                    "pos_master": float(msg.data[0]),
                    "pos_slave":  float(msg.data[1]),
                    "pos_norm":   float(msg.data[2]),   # 0=mở, 1=kẹp
                    "torque":     float(msg.data[3]),
                    "contact":    int(msg.data[4]),
                    "mode":       int(msg.data[5]),
                }

    # ──────────────────────────────────────────────────────
    # Tick: append frame vào buffer
    # ──────────────────────────────────────────────────────
    def _tick(self):
        if not self.recording:
            return

        with self._lock:
            f       = self.front_img.copy()  if self.front_img  is not None else None
            w       = self.wrist_img.copy()  if self.wrist_img  is not None else None
            actual  = dict(self.actual_pose) if self.actual_pose is not None else None
            target  = dict(self.target_pose) if self.target_pose is not None else None
            gripper = dict(self.gripper)     if self.gripper     is not None else None

        # Cần 2 camera + actual pose để ghi
        if f is None or w is None or actual is None:
            self.get_logger().warn(
                "Thiếu data (cần front/wrist cam + actual pose)",
                throttle_duration_sec=1.0)
            return

        # ── Images → RGB (480×640) ──
        front_rgb = cv2.cvtColor(cv2.resize(f, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(cv2.resize(w, IMAGE_SIZE), cv2.COLOR_BGR2RGB)

        # ── Gripper norm ──
        grip_norm = gripper["pos_norm"] if gripper else 0.0

        # ── STATE (8 dim): [ee_x,ee_y,ee_z, qx,qy,qz,qw, gripper] ──
        if target is not None:
            qx, qy, qz, qw = (target["qx"], target["qy"],
                               target["qz"], target["qw"])
        else:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

        state = np.array([
            actual["x"], actual["y"], actual["z"],
            qx, qy, qz, qw,
            grip_norm,
        ], dtype=np.float32)

        # ── FORWARD DELTA: lưu pose thô + gripper, action tính khi save ──
        # action[t] = pose(t+1) - pose(t). Không tính ở đây vì cần frame sau.
        pose_now = np.array([
            actual["x"], actual["y"], actual["z"],
            actual["roll"], actual["pitch"], actual["yaw"],
            grip_norm,
        ], dtype=np.float32)

        ts = self.get_clock().now().nanoseconds * 1e-9

        with self._lock:
            self.buf_front.append(front_rgb)
            self.buf_wrist.append(wrist_rgb)
            self.buf_state.append(state)
            self.buf_pose.append(pose_now)
            self.buf_tick_ts.append(ts)
        self.frame_idx += 1

    # ──────────────────────────────────────────────────────
    # Episode control
    # ──────────────────────────────────────────────────────
    def start_episode(self):
        if self.recording:
            return
        if self.front_img is None or self.wrist_img is None:
            self.get_logger().warn("Chưa có camera!")
            return
        if self.actual_pose is None:
            self.get_logger().warn("Chưa có actual_pose từ robot!")
            return

        with self._lock:
            self.buf_front      = []
            self.buf_wrist      = []
            self.buf_digit_l    = []
            self.buf_digit_r    = []
            self.buf_digit_l_ts = []
            self.buf_digit_r_ts = []
            self.buf_state      = []
            self.buf_pose       = []
            self.buf_tick_ts    = []
            self.frame_idx      = 0
            self.recording      = True
        self.t_start     = time.time()
        self.get_logger().info(f"🔴 START demo_{self.next_demo_id}")

    def stop_recording(self):
        self.recording = False

    def teleop_on(self):
        """Bật robot bám vive."""
        m = Bool(); m.data = True
        self.pub_teleop.publish(m)
        self.get_logger().info("🟢 teleop ON (robot bám vive)")

    def teleop_off(self):
        """Tắt robot bám vive."""
        m = Bool(); m.data = False
        self.pub_teleop.publish(m)
        self.get_logger().info("🔴 teleop OFF")

    def go_home(self):
        """Ra lệnh robot về home."""
        m = Bool(); m.data = True
        self.pub_home.publish(m)
        self.get_logger().info("🏠 lệnh về home đã gửi")

    def save_episode(self, success):
        # Snapshot buffer dưới lock (callback DIGIT vẫn chạy)
        with self._lock:
            self.recording = False   # dừng append DIGIT trước khi snapshot
            buf_front   = list(self.buf_front)
            buf_wrist   = list(self.buf_wrist)
            buf_state   = list(self.buf_state)
            buf_pose    = list(self.buf_pose)
            buf_tick_ts = list(self.buf_tick_ts)
            buf_dl      = list(self.buf_digit_l)
            buf_dr      = list(self.buf_digit_r)
            buf_dl_ts   = list(self.buf_digit_l_ts)
            buf_dr_ts   = list(self.buf_digit_r_ts)

        T20 = len(buf_state)
        if T20 == 0:
            self.get_logger().warn("Buffer rỗng — chưa có frame nào!")
            return

        # Cảnh báo nếu thiếu DIGIT (đừng lưu rác)
        T60_l = len(buf_dl)
        T60_r = len(buf_dr)
        if T60_l == 0 or T60_r == 0:
            self.get_logger().warn(
                f"⚠️  DIGIT trống (L={T60_l}, R={T60_r})! "
                f"Kiểm tra digit_publisher. KHÔNG lưu demo này.")
            self._reset_buffers()
            return

        elapsed  = time.time() - self.t_start
        fps_real = T20 / max(elapsed, 0.001)

        # ── FORWARD DELTA: action[t] = pose(t+1) - pose(t) ──
        # pose = [x,y,z, roll,pitch,yaw, gripper]. Frame cuối không có t+1 → lặp delta=0 cho gripper giữ.
        poses = np.stack(buf_pose, axis=0)   # (T20, 7)
        actions = np.zeros((T20, ACTION_DIM), dtype=np.float32)
        for t in range(T20 - 1):
            actions[t, 0:6] = poses[t+1, 0:6] - poses[t, 0:6]   # delta xyz+rpy (forward)
            actions[t, 6]   = poses[t+1, 6]                      # gripper = lệnh kế tiếp
        # Frame cuối: không có t+1 → delta xyz/rpy = 0, gripper giữ nguyên
        actions[T20-1, 0:6] = 0.0
        actions[T20-1, 6]   = poses[T20-1, 6]

        arr_front   = np.stack(buf_front, axis=0)
        arr_wrist   = np.stack(buf_wrist, axis=0)
        arr_state   = np.stack(buf_state, axis=0)
        arr_digit_l = np.stack(buf_dl, axis=0)        # (T60, 320, 240, 3)
        arr_digit_r = np.stack(buf_dr, axis=0)
        arr_dl_ts   = np.array(buf_dl_ts, dtype=np.float64)
        arr_dr_ts   = np.array(buf_dr_ts, dtype=np.float64)
        arr_tick_ts = np.array(buf_tick_ts, dtype=np.float64)

        demo_name = f"demo_{self.next_demo_id}"
        with h5py.File(self.h5_path, "a") as f:
            if "data" not in f:
                g = f.create_group("data")
                g.attrs["task"]       = self.task
                g.attrs["fps"]        = self.fps
                g.attrs["digit_fps"]  = 60
                g.attrs["state_dim"]  = STATE_DIM
                g.attrs["action_dim"] = ACTION_DIM
                g.attrs["image_shape"]= list(IMAGE_SHAPE)
                g.attrs["digit_shape"]= list(DIGIT_SHAPE)
                g.attrs["robot"]      = "UR3"
                g.attrs["action_convention"] = "forward_delta: action[t]=pose(t+1)-pose(t)"
                g.attrs["state_keys"] = "ee_x,ee_y,ee_z,qx,qy,qz,qw,gripper"
                g.attrs["action_keys"]= "dx,dy,dz,d_roll,d_pitch,d_yaw,gripper"
            grp = f["data"].create_group(demo_name)
            obs = grp.create_group("obs")
            # 20Hz
            obs.create_dataset("image",       data=arr_front,
                               compression="gzip", compression_opts=4)
            obs.create_dataset("wrist_image", data=arr_wrist,
                               compression="gzip", compression_opts=4)
            obs.create_dataset("state",       data=arr_state)
            obs.create_dataset("timestamp",   data=arr_tick_ts)
            # 60Hz (dual-rate)
            obs.create_dataset("digit_left",     data=arr_digit_l,
                               compression="gzip", compression_opts=4)
            obs.create_dataset("digit_right",    data=arr_digit_r,
                               compression="gzip", compression_opts=4)
            obs.create_dataset("digit_left_ts",  data=arr_dl_ts)
            obs.create_dataset("digit_right_ts", data=arr_dr_ts)
            # action 20Hz
            grp.create_dataset("actions",     data=actions)

            grp.attrs["success"]      = bool(success)
            grp.attrs["n_frames"]     = T20
            grp.attrs["n_digit_l"]    = T60_l
            grp.attrs["n_digit_r"]    = T60_r
            grp.attrs["fps_actual"]   = float(fps_real)
            grp.attrs["duration_s"]   = float(elapsed)
            grp.attrs["task"]         = self.task
            grp.attrs["recorded_at"]  = time.strftime("%Y-%m-%d %H:%M:%S")

        icon = "✅" if success else "❌"
        self.get_logger().info(
            f"{icon} {demo_name}  20Hz={T20}f  60Hz=L{T60_l}/R{T60_r}  "
            f"{fps_real:.1f}fps  {elapsed:.1f}s → {self.h5_path}")

        self._reset_buffers()
        self.next_demo_id += 1
        self.ep_total += 1
        if success:
            self.ep_success += 1

    def _reset_buffers(self):
        with self._lock:
            self.buf_front      = []
            self.buf_wrist      = []
            self.buf_digit_l    = []
            self.buf_digit_r    = []
            self.buf_digit_l_ts = []
            self.buf_digit_r_ts = []
            self.buf_state      = []
            self.buf_pose       = []
            self.buf_tick_ts    = []

    # ──────────────────────────────────────────────────────
    # GUI
    # ──────────────────────────────────────────────────────
    def make_gui_frame(self):
        with self._lock:
            f           = self.front_img.copy()   if self.front_img   is not None else None
            w           = self.wrist_img.copy()   if self.wrist_img   is not None else None
            dl          = self.digit_l_img.copy() if self.digit_l_img is not None else None
            dr          = self.digit_r_img.copy() if self.digit_r_img is not None else None
            fc, wc      = self.front_count, self.wrist_count
            dlc, drc    = self.digit_l_count, self.digit_r_count
            has_actual  = self.actual_pose  is not None
            has_target  = self.target_pose  is not None
            has_gripper = self.gripper      is not None
            grip_data   = dict(self.gripper) if self.gripper else None
            actual      = dict(self.actual_pose) if self.actual_pose else None

        # Mỗi panel (4 cam → lưới 2x2, panel nhỏ hơn cho vừa màn hình)
        PW, PH = 480, 270

        def panel(img, label, count):
            if img is not None:
                p   = cv2.resize(img, (PW, PH))
                col = (0, 220, 0)
                txt = f"{label}  #{count}"
            else:
                p   = np.zeros((PH, PW, 3), np.uint8)
                col = (0, 0, 220)
                txt = f"{label}  NO DATA"
                cv2.putText(p, "NO DATA", (PW//2-90, PH//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            cv2.rectangle(p, (0, 0), (PW-1, PH-1), col, 3)
            cv2.putText(p, txt, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            return p

        # Lưới 2×2:
        #   [ FRONT ] [ WRIST ]
        #   [ DIGIT_L] [ DIGIT_R]
        row_top = np.hstack([panel(f,  "FRONT",   fc),
                             panel(w,  "WRIST",   wc)])
        row_bot = np.hstack([panel(dl, "DIGIT_L", dlc),
                             panel(dr, "DIGIT_R", drc)])
        cams = np.vstack([row_top, row_bot])

        full_w = PW * 2

        bar = np.zeros((100, full_w, 3), np.uint8)

        # Line 1: REC status
        if self.recording:
            txt = (f"REC  demo_{self.next_demo_id}  "
                   f"frame={self.frame_idx}  t={time.time()-self.t_start:.1f}s")
            col = (0, 0, 255)
        else:
            txt = (f"IDLE  next=demo_{self.next_demo_id}  "
                   f"ok={self.ep_success}/{self.ep_total}")
            col = (180, 180, 180)
        cv2.putText(bar, txt, (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        # Line 2: Topic status (gồm cả 2 DIGIT)
        dot = lambda ok: "*" if ok else "-"
        info = (f"front {dot(f is not None)}#{fc}  "
                f"wrist {dot(w is not None)}#{wc}  "
                f"digL {dot(dl is not None)}#{dlc}  "
                f"digR {dot(dr is not None)}#{drc}  "
                f"actual {dot(has_actual)}  "
                f"target {dot(has_target)}  "
                f"grip {dot(has_gripper)}")
        cv2.putText(bar, info, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Line 3: TCP pose
        if actual:
            pose_txt = (f"TCP  X={actual['x']:+.3f}  Y={actual['y']:+.3f}  "
                        f"Z={actual['z']:+.3f}  "
                        f"R={actual['roll']:+.2f}  P={actual['pitch']:+.2f}  "
                        f"Yaw={actual['yaw']:+.2f}")
            cv2.putText(bar, pose_txt, (12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)

        # Line 4: Gripper
        if grip_data:
            g        = grip_data
            mode_str = ['IDLE', 'AUTO', 'MIMIC'][g['mode']] \
                       if g['mode'] in (0, 1, 2) else '?'
            col_g    = (0, 0, 255) if g['contact'] else (100, 220, 100)
            grip_txt = (f"GRIP [{mode_str}]  norm={g['pos_norm']:.2f}  "
                        f"tor={g['torque']:.2f}  "
                        f"{'CONTACT!' if g['contact'] else ''}")
            cv2.putText(bar, grip_txt, (12, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_g, 1)

        return np.vstack([cams, bar])


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,  default='pick_cube')
    parser.add_argument('--fps',  type=int,  default=20)
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = HDF5Recorder(args.task, args.fps)

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    # ── Terminal status ──
    def print_status():
        while rclpy.ok():
            try:
                with node._lock:
                    has_front   = node.front_img  is not None
                    has_wrist   = node.wrist_img  is not None
                    has_dl      = node.digit_l_img is not None
                    has_dr      = node.digit_r_img is not None
                    has_actual  = node.actual_pose is not None
                    has_target  = node.target_pose is not None
                    has_grip    = node.gripper     is not None
                    fc, wc      = node.front_count, node.wrist_count
                    dlc, drc    = node.digit_l_count, node.digit_r_count

                dot = lambda ok: "●" if ok else "○"
                if node.recording:
                    elapsed = time.time() - node.t_start
                    s = (f"\r🔴 REC demo_{node.next_demo_id} "
                         f"{node.frame_idx}f {elapsed:.1f}s  "
                         f"| front {dot(has_front)}#{fc} "
                         f"wrist {dot(has_wrist)}#{wc} "
                         f"digL {dot(has_dl)}#{dlc} "
                         f"digR {dot(has_dr)}#{drc} "
                         f"actual {dot(has_actual)} "
                         f"target {dot(has_target)} "
                         f"grip {dot(has_grip)}   ")
                else:
                    s = (f"\r⏸  IDLE next=demo_{node.next_demo_id} "
                         f"ok={node.ep_success}/{node.ep_total}  "
                         f"| front {dot(has_front)}#{fc} "
                         f"wrist {dot(has_wrist)}#{wc} "
                         f"digL {dot(has_dl)}#{dlc} "
                         f"digR {dot(has_dr)}#{drc} "
                         f"actual {dot(has_actual)} "
                         f"target {dot(has_target)} "
                         f"grip {dot(has_grip)}   ")
                print(s, end="", flush=True)
            except Exception:
                pass
            time.sleep(0.5)

    threading.Thread(target=print_status, daemon=True).start()

    # ── GUI cv2 thread ──
    def gui_loop():
        win = "Record All — Front|Wrist + DIGIT L|R"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # Lưới 2x2 panel 480x270 = 960x540, + 100 status bar
        cv2.resizeWindow(win, 960, 640)
        while rclpy.ok():
            try:
                frame = node.make_gui_frame()
                cv2.imshow(win, frame)
                cv2.waitKey(30)
            except Exception:
                time.sleep(0.1)

    threading.Thread(target=gui_loop, daemon=True).start()

    # ── Keyboard input ──
    import tty, termios

    def getch():
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("\n" + "═"*55)
    print("  DATASET RECORDER")
    print(f"  State  (8): [ee_x, ee_y, ee_z, qx, qy, qz, qw, grip]")
    print(f"  Action (7): [dx, dy, dz, d_roll, d_pitch, d_yaw, grip]")
    print("  SPACE = bắt đầu/dừng  |  S = success  |  F = fail  |  Q = quit")
    print("═"*55 + "\n")

    try:
        while rclpy.ok():
            k = getch()
            if k == ' ':
                if not node.recording:
                    # Robot ON + record NGAY (origin đã chốt bằng Home tay ở T3)
                    node.teleop_on()
                    node.start_episode()
                    print(f"\n🟢🔴 SPACE — robot ON + REC demo_{node.next_demo_id}")
                else:
                    node.stop_recording()
                    print("\n⏹  Dừng — bấm S (success) hoặc F (fail)")
            elif k in ('s', 'S'):
                if node.recording:
                    node.stop_recording()
                node.save_episode(success=True)
                # Tự động: tắt teleop + về home
                node.teleop_off()
                node.go_home()
                print("✅ Đã lưu SUCCESS → robot OFF + về home. Bấm SPACE cho demo tiếp.")
            elif k in ('f', 'F'):
                if node.recording:
                    node.stop_recording()
                node.save_episode(success=False)
                node.teleop_off()
                node.go_home()
                print("❌ Đã lưu FAIL → robot OFF + về home. Bấm SPACE cho demo tiếp.")
            elif k in ('q', 'Q', '\x03', '\x1b'):
                node.teleop_off()
                print(f"\n👋 Thoát  success={node.ep_success}/{node.ep_total}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node.stop_recording()
        print()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
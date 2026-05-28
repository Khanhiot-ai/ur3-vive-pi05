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
        │   │   ├── state         (T, 8)  float32
        │   │   │     [ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper]
        │   │   │     ← Cartesian TCP pose + gripper norm
        │   │   └── tactile_state (T, 1)  float32          ← gripper torque
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

TACTILE_DIM  = 1   # gripper torque

TOPIC_FRONT   = '/camera_front/camera/color/image_raw'
TOPIC_WRIST   = '/camera_wrist/camera/color/image_raw'
TOPIC_JOINTS  = '/ur_joint_states'
TOPIC_ACTUAL  = '/ur_actual_pose'      # Xyzrpy: x,y,z,roll,pitch,yaw
TOPIC_TARGET  = '/ur_target_pose'      # PoseStamped: xyz + quat
TOPIC_GRIPPER = '/gripper/state'
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
        self.actual_pose = None   # dict: x,y,z,roll,pitch,yaw
        self.target_pose = None   # dict: x,y,z,qx,qy,qz,qw
        self.gripper     = None   # dict: pos_norm, torque, contact, mode
        self.front_count = 0
        self.wrist_count = 0

        # ── Prev actual pose để tính delta ──
        self._prev_actual = None

        # Recording buffers
        self.recording   = False
        self.frame_idx   = 0
        self.t_start     = 0.0
        self.buf_front   = []
        self.buf_wrist   = []
        self.buf_state   = []
        self.buf_action  = []
        self.buf_tactile = []

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
        if HAS_XYZRPY:
            self.create_subscription(Xyzrpy,      TOPIC_ACTUAL,  self._cb_actual,  qos_robot)
        self.create_subscription(PoseStamped,      TOPIC_TARGET,  self._cb_target,  qos_robot)
        self.create_subscription(Float32MultiArray, TOPIC_GRIPPER, self._cb_gripper, qos_robot)

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
        if os.path.exists(self.h5_path):
            with h5py.File(self.h5_path, "r") as f:
                if "data" in f:
                    nums = [int(k.split("_")[1]) for k in f["data"]
                            if k.startswith("demo_")]
                    self.next_demo_id = max(nums, default=-1) + 1
                else:
                    self.next_demo_id = 0
        else:
            with h5py.File(self.h5_path, "w") as f:
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

        # Cần ít nhất 2 camera + actual pose để ghi
        if f is None or w is None or actual is None:
            self.get_logger().warn(
                "Thiếu data (cần front/wrist cam + actual pose)",
                throttle_duration_sec=1.0)
            return

        # ── Images → RGB 224×224 ──
        front_rgb = cv2.cvtColor(cv2.resize(f, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(cv2.resize(w, IMAGE_SIZE), cv2.COLOR_BGR2RGB)

        # ── Gripper norm ──
        grip_norm = gripper["pos_norm"] if gripper else 0.0

        # ── STATE (8 dim) ──
        # [ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper]
        # Lấy quaternion từ target (vì actual_pose là Xyzrpy không có quat trực tiếp)
        # Nếu target chưa có → dùng identity quat
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

        # ── ACTION (7 dim) ──
        # [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
        # delta = actual_current - actual_prev (robot đã di chuyển bao nhiêu)
        if self._prev_actual is not None:
            prev = self._prev_actual
            action = np.array([
                actual["x"]     - prev["x"],
                actual["y"]     - prev["y"],
                actual["z"]     - prev["z"],
                actual["roll"]  - prev["roll"],
                actual["pitch"] - prev["pitch"],
                actual["yaw"]   - prev["yaw"],
                grip_norm,
            ], dtype=np.float32)
        else:
            # Frame đầu tiên: delta = 0
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip_norm],
                               dtype=np.float32)

        # Cập nhật prev
        self._prev_actual = dict(actual)

        # ── Tactile ──
        tactile = np.array([gripper["torque"] if gripper else 0.0],
                            dtype=np.float32)

        self.buf_front.append(front_rgb)
        self.buf_wrist.append(wrist_rgb)
        self.buf_state.append(state)
        self.buf_action.append(action)
        self.buf_tactile.append(tactile)
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

        self.buf_front   = []
        self.buf_wrist   = []
        self.buf_state   = []
        self.buf_action  = []
        self.buf_tactile = []
        self.frame_idx   = 0
        self.t_start     = time.time()
        self._prev_actual = None   # Reset delta
        self.recording   = True
        self.get_logger().info(f"🔴 START demo_{self.next_demo_id}")

    def stop_recording(self):
        self.recording = False

    def save_episode(self, success):
        if not self.buf_state:
            self.get_logger().warn("Buffer rỗng — chưa có frame nào!")
            return

        T       = len(self.buf_state)
        elapsed = time.time() - self.t_start
        fps_real = T / max(elapsed, 0.001)

        arr_front   = np.stack(self.buf_front,   axis=0)
        arr_wrist   = np.stack(self.buf_wrist,   axis=0)
        arr_state   = np.stack(self.buf_state,   axis=0)
        arr_action  = np.stack(self.buf_action,  axis=0)
        arr_tactile = np.stack(self.buf_tactile, axis=0)

        demo_name = f"demo_{self.next_demo_id}"
        with h5py.File(self.h5_path, "a") as f:
            grp = f["data"].create_group(demo_name)
            obs = grp.create_group("obs")
            obs.create_dataset("image",         data=arr_front,
                               compression="gzip", compression_opts=4)
            obs.create_dataset("wrist_image",   data=arr_wrist,
                               compression="gzip", compression_opts=4)
            obs.create_dataset("state",         data=arr_state)
            obs.create_dataset("tactile_state", data=arr_tactile)
            grp.create_dataset("actions",       data=arr_action)

            grp.attrs["success"]     = bool(success)
            grp.attrs["n_frames"]    = T
            grp.attrs["fps_actual"]  = float(fps_real)
            grp.attrs["duration_s"]  = float(elapsed)
            grp.attrs["task"]        = self.task
            grp.attrs["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        icon = "✅" if success else "❌"
        self.get_logger().info(
            f"{icon} {demo_name}  {T}f  {fps_real:.1f}fps  {elapsed:.1f}s"
            f"  state8  action7  → {self.h5_path}")

        # Reset buffers
        self.buf_front = []; self.buf_wrist   = []
        self.buf_state = []; self.buf_action  = []
        self.buf_tactile = []
        self.next_demo_id += 1
        self.ep_total += 1
        if success:
            self.ep_success += 1

    # ──────────────────────────────────────────────────────
    # GUI
    # ──────────────────────────────────────────────────────
    def make_gui_frame(self):
        with self._lock:
            f           = self.front_img.copy()  if self.front_img  is not None else None
            w           = self.wrist_img.copy()  if self.wrist_img  is not None else None
            fc, wc      = self.front_count, self.wrist_count
            has_actual  = self.actual_pose  is not None
            has_target  = self.target_pose  is not None
            has_gripper = self.gripper      is not None
            grip_data   = dict(self.gripper) if self.gripper else None
            actual      = dict(self.actual_pose) if self.actual_pose else None

        W, H = DISPLAY_SIZE

        def panel(img, label, count):
            if img is not None:
                p   = cv2.resize(img, (W, H))
                col = (0, 220, 0)
                txt = f"{label}  #{count}"
            else:
                p   = np.zeros((H, W, 3), np.uint8)
                col = (0, 0, 220)
                txt = f"{label}  NO DATA"
                cv2.putText(p, "NO DATA", (W//2-100, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            cv2.rectangle(p, (0, 0), (W-1, H-1), col, 4)
            cv2.putText(p, txt, (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
            return p

        cams = np.hstack([panel(f, "FRONT", fc), panel(w, "WRIST", wc)])

        bar = np.zeros((100, W*2, 3), np.uint8)

        # Line 1: REC status
        if self.recording:
            txt = (f"🔴 REC  demo_{self.next_demo_id}  "
                   f"frame={self.frame_idx}  t={time.time()-self.t_start:.1f}s")
            col = (0, 0, 255)
        else:
            txt = (f"⏸  IDLE  next=demo_{self.next_demo_id}  "
                   f"ok={self.ep_success}/{self.ep_total}")
            col = (180, 180, 180)
        cv2.putText(bar, txt, (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        # Line 2: Topic status
        dot = lambda ok: "●" if ok else "○"
        info = (f"front {dot(f is not None)}#{fc}  "
                f"wrist {dot(w is not None)}#{wc}  "
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
                    has_actual  = node.actual_pose is not None
                    has_target  = node.target_pose is not None
                    has_grip    = node.gripper     is not None
                    fc, wc      = node.front_count, node.wrist_count

                dot = lambda ok: "●" if ok else "○"
                if node.recording:
                    elapsed = time.time() - node.t_start
                    s = (f"\r🔴 REC demo_{node.next_demo_id} "
                         f"{node.frame_idx}f {elapsed:.1f}s  "
                         f"| front {dot(has_front)}#{fc} "
                         f"wrist {dot(has_wrist)}#{wc} "
                         f"actual {dot(has_actual)} "
                         f"target {dot(has_target)} "
                         f"grip {dot(has_grip)}   ")
                else:
                    s = (f"\r⏸  IDLE next=demo_{node.next_demo_id} "
                         f"ok={node.ep_success}/{node.ep_total}  "
                         f"| front {dot(has_front)}#{fc} "
                         f"wrist {dot(has_wrist)}#{wc} "
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
        win = "Record All — Front | Wrist"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, DISPLAY_SIZE[0]*2, DISPLAY_SIZE[1]+100)
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
                    node.start_episode()
                    print(f"\n🔴 START demo_{node.next_demo_id - 1 if node.next_demo_id > 0 else 0}")
                else:
                    node.stop_recording()
                    print("\n⏹  Dừng — bấm S (success) hoặc F (fail)")
            elif k in ('s', 'S'):
                if node.recording:
                    node.stop_recording()
                node.save_episode(success=True)
            elif k in ('f', 'F'):
                if node.recording:
                    node.stop_recording()
                node.save_episode(success=False)
            elif k in ('q', 'Q', '\x03', '\x1b'):
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
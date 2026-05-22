#!/usr/bin/env python3
"""
DATASET RECORDER — HDF5 cho Pi0.5 / LeRobot

Schema HDF5:
    dataset/<task>.hdf5
    └── data/
        ├── demo_0/
        │   ├── obs/
        │   │   ├── image         (T, 224, 224, 3) uint8   ← Realsense
        │   │   ├── wrist_image   (T, 224, 224, 3) uint8   ← C922
        │   │   ├── state         (T, 7)  float32          ← joints[6]+grip[1]
        │   │   └── tactile_state (T, 1)  float32          ← torque
        │   ├── actions           (T, 8)  float32          ← xyz[3]+quat[4]+grip[1]
        │   └── attrs: success, n_frames, fps, task, recorded_at
        └── demo_1/, demo_2/, ...

Chạy:
    python3 record_all.py --task pick_cube --fps 10
GUI:
    SPACE = rec / stop
    S = save SUCCESS
    F = save FAIL
    Q = quit
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

# ════════════ CONFIG ════════════
IMAGE_SIZE   = (224, 224)
DISPLAY_SIZE = (640, 360)
STATE_DIM    = 7   # joints[6] + gripper_pos[1]
ACTION_DIM   = 8   # xyz[3] + quat[4] + gripper_cmd[1]
TACTILE_DIM  = 1   # torque

TOPIC_FRONT   = '/camera/camera/color/image_raw'
TOPIC_WRIST   = '/camera_wrist/image_raw'
TOPIC_JOINTS  = '/ur_joint_states'
TOPIC_ACTUAL  = '/ur_actual_pose'
TOPIC_TARGET  = '/ur_target_pose'
TOPIC_GRIPPER = '/gripper/state'
# ════════════════════════════════


class HDF5Recorder(Node):
    def __init__(self, task_name, fps):
        super().__init__('hdf5_recorder')
        self.bridge = CvBridge()
        self.task = task_name
        self.fps = fps
        self.dt = 1.0 / fps

        # State
        self._lock = threading.Lock()
        self.front_img = None
        self.wrist_img = None
        self.joints = None
        self.actual_pose = None
        self.target_pose = None
        self.gripper = None
        self.front_count = 0
        self.wrist_count = 0

        # Recording
        self.recording = False
        self.frame_idx = 0
        self.t_start = 0.0
        self.buf_front = []
        self.buf_wrist = []
        self.buf_state = []
        self.buf_action = []
        self.buf_tactile = []

        self.ep_total = 0
        self.ep_success = 0

        # HDF5 file
        self.h5_path = os.path.join("dataset", f"{task_name}.hdf5")
        os.makedirs("dataset", exist_ok=True)
        self._open_or_init_hdf5()

        # ROS subs
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image,       TOPIC_FRONT,  self._cb_front,  qos)
        self.create_subscription(Image,       TOPIC_WRIST,  self._cb_wrist,  qos)
        self.create_subscription(JointState,  TOPIC_JOINTS, self._cb_joints, qos)
        self.create_subscription(PoseStamped, TOPIC_TARGET, self._cb_target, qos)
        if HAS_XYZRPY:
            self.create_subscription(Xyzrpy, TOPIC_ACTUAL, self._cb_actual, qos)
        self.create_subscription(Float32MultiArray, TOPIC_GRIPPER,
                                 self._cb_gripper, qos)

        self.tick_timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info("══════════════════════════════════════════")
        self.get_logger().info(f"  HDF5 RECORDER  task={task_name}")
        self.get_logger().info(f"  File: {self.h5_path}")
        self.get_logger().info(f"  Demo có sẵn: {self.next_demo_id}")
        self.get_logger().info(f"  state_dim={STATE_DIM}  action_dim={ACTION_DIM}")
        self.get_logger().info("  SPACE=rec  S=success  F=fail  Q=quit")
        self.get_logger().info("══════════════════════════════════════════")

    def _open_or_init_hdf5(self):
        if os.path.exists(self.h5_path):
            with h5py.File(self.h5_path, "r") as f:
                if "data" in f:
                    existing = list(f["data"].keys())
                    nums = [int(k.split("_")[1]) for k in existing
                            if k.startswith("demo_")]
                    self.next_demo_id = max(nums, default=-1) + 1
                else:
                    self.next_demo_id = 0
        else:
            with h5py.File(self.h5_path, "w") as f:
                f.create_group("data")
                f["data"].attrs["task"] = self.task
                f["data"].attrs["fps"] = self.fps
                f["data"].attrs["state_dim"] = STATE_DIM
                f["data"].attrs["action_dim"] = ACTION_DIM
                f["data"].attrs["image_shape"] = list(IMAGE_SIZE) + [3]
                f["data"].attrs["robot"] = "UR3"
            self.next_demo_id = 0

    # ── Callbacks ──
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

    def _cb_joints(self, msg):
        with self._lock:
            self.joints = list(msg.position[:6])

    def _cb_actual(self, msg):
        with self._lock:
            self.actual_pose = {
                "x": float(msg.x), "y": float(msg.y), "z": float(msg.z),
                "roll": float(msg.roll), "pitch": float(msg.pitch),
                "yaw": float(msg.yaw),
            }

    def _cb_target(self, msg):
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
        # [pos_master, pos_slave, pos_norm, torque, contact, mode]
        if len(msg.data) >= 6:
            with self._lock:
                self.gripper = {
                    "pos_master": float(msg.data[0]),
                    "pos_slave":  float(msg.data[1]),
                    "pos_norm":   float(msg.data[2]),
                    "torque":     float(msg.data[3]),
                    "contact":    int(msg.data[4]),
                    "mode":       int(msg.data[5]),
                }

    # ── Tick: append to buffer ──
    def _tick(self):
        if not self.recording:
            return

        with self._lock:
            f = self.front_img.copy() if self.front_img is not None else None
            w = self.wrist_img.copy() if self.wrist_img is not None else None
            joints = list(self.joints) if self.joints is not None else None
            target = dict(self.target_pose) if self.target_pose is not None else None
            gripper = dict(self.gripper) if self.gripper is not None else None

        if f is None or w is None or joints is None:
            self.get_logger().warn("Thiếu data", throttle_duration_sec=1.0)
            return

        # Images BGR → RGB
        front_rgb = cv2.cvtColor(cv2.resize(f, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(cv2.resize(w, IMAGE_SIZE), cv2.COLOR_BGR2RGB)

        # State = joints[6] + gripper_pos[1]
        grip_pos = gripper["pos_norm"] if gripper else 0.0
        state = np.array(joints + [grip_pos], dtype=np.float32)

        # Action = target_xyz + target_quat + gripper_cmd
        if target is not None:
            action = np.array([
                target["x"], target["y"], target["z"],
                target["qx"], target["qy"], target["qz"], target["qw"],
                grip_pos,  # gripper command = current normalized pos
            ], dtype=np.float32)
        else:
            action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Tactile = [torque]
        tactile = np.array([gripper["torque"] if gripper else 0.0],
                           dtype=np.float32)

        self.buf_front.append(front_rgb)
        self.buf_wrist.append(wrist_rgb)
        self.buf_state.append(state)
        self.buf_action.append(action)
        self.buf_tactile.append(tactile)
        self.frame_idx += 1

        if self.frame_idx % 10 == 0:
            elapsed = time.time() - self.t_start
            fps = self.frame_idx / max(elapsed, 0.001)
            self.get_logger().info(
                f"🔴 {self.frame_idx}f  {elapsed:.1f}s  {fps:.1f}fps",
                throttle_duration_sec=1.0)

    # ── Episode control ──
    def start_episode(self):
        if self.recording:
            return
        if self.front_img is None or self.wrist_img is None:
            self.get_logger().warn("Chưa có camera")
            return

        self.buf_front = []
        self.buf_wrist = []
        self.buf_state = []
        self.buf_action = []
        self.buf_tactile = []
        self.frame_idx = 0
        self.t_start = time.time()
        self.recording = True
        self.get_logger().info(f"🔴 START demo_{self.next_demo_id}")

    def stop_recording(self):
        self.recording = False

    def save_episode(self, success):
        if not self.buf_state:
            self.get_logger().warn("Buffer rỗng")
            return

        T = len(self.buf_state)
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
            f"{icon}  {demo_name}  {T}f  {fps_real:.1f}fps  "
            f"{elapsed:.1f}s  → {self.h5_path}")

        self.buf_front = []
        self.buf_wrist = []
        self.buf_state = []
        self.buf_action = []
        self.buf_tactile = []

        self.next_demo_id += 1
        self.ep_total += 1
        if success:
            self.ep_success += 1

    # ── GUI ──
    def make_gui_frame(self):
        with self._lock:
            f = self.front_img
            w = self.wrist_img
            fc, wc = self.front_count, self.wrist_count
            has_joints = self.joints is not None
            has_target = self.target_pose is not None
            has_actual = self.actual_pose is not None
            has_gripper = self.gripper is not None
            grip_data = dict(self.gripper) if self.gripper else None

        W, H = DISPLAY_SIZE

        def panel(img, label, count):
            if img is not None:
                p = cv2.resize(img, (W, H))
                col = (0, 220, 0)
                txt = f"{label}  #{count}"
            else:
                p = np.zeros((H, W, 3), np.uint8)
                col = (0, 0, 220)
                txt = f"{label}  NO DATA"
                cv2.putText(p, "NO DATA", (W//2-100, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            cv2.rectangle(p, (0,0), (W-1,H-1), col, 4)
            cv2.putText(p, txt, (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
            return p

        cams = np.hstack([panel(f, "FRONT", fc), panel(w, "WRIST", wc)])

        bar = np.zeros((90, W*2, 3), np.uint8)
        if self.recording:
            txt = (f"🔴 REC  demo_{self.next_demo_id}  "
                   f"frame={self.frame_idx}  t={time.time()-self.t_start:.1f}s")
            col = (0, 0, 255)
        else:
            txt = (f"⏸  IDLE  next=demo_{self.next_demo_id}  "
                   f"total={self.ep_success}/{self.ep_total}")
            col = (180, 180, 180)
        cv2.putText(bar, txt, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

        dot = lambda ok: "●" if ok else "○"
        info = (f"joints {dot(has_joints)}   "
                f"actual {dot(has_actual)}   "
                f"target {dot(has_target)}   "
                f"gripper {dot(has_gripper)}   "
                f"file: {os.path.basename(self.h5_path)}")
        cv2.putText(bar, info, (12, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if has_actual:
            with self._lock: p = self.actual_pose
            pose_txt = (f"TCP X={p['x']:+.3f} Y={p['y']:+.3f} Z={p['z']:+.3f}  "
                        f"R={p['roll']:+.2f} P={p['pitch']:+.2f} Y={p['yaw']:+.2f}")
            cv2.putText(bar, pose_txt, (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)

        if grip_data is not None:
            g = grip_data
            grip_col = (0, 0, 255) if g['contact'] else (180, 220, 180)
            mode_str = ['IDLE', 'AUTO', 'MIMIC'][g['mode']] if g['mode'] in (0,1,2) else '?'
            grip_txt = (f"GRIP [{mode_str}] pos={g['pos_norm']:.2f}  "
                        f"tor={g['torque']:.2f}  "
                        f"{'CONTACT!' if g['contact'] else ''}")
            bar2 = np.zeros((30, bar.shape[1], 3), np.uint8)
            cv2.putText(bar2, grip_txt, (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_col, 1)
            return np.vstack([cams, bar, bar2])

        return np.vstack([cams, bar])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='task_1')
    parser.add_argument('--fps', type=int, default=10)
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = HDF5Recorder(args.task, args.fps)

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    win = "Dataset Recorder (HDF5) — Q=quit SPACE=rec S=success F=fail"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_SIZE[0]*2, DISPLAY_SIZE[1]+120)

    try:
        while rclpy.ok():
            frame = node.make_gui_frame()
            cv2.imshow(win, frame)
            k = cv2.waitKey(10) & 0xFF
            if k == ord(' '):
                if not node.recording:
                    node.start_episode()
                else:
                    node.stop_recording()
                    node.get_logger().info("⏹ Dừng — S/F để lưu")
            elif k in (ord('s'), ord('S')):
                if node.recording: node.stop_recording()
                node.save_episode(success=True)
            elif k in (ord('f'), ord('F')):
                if node.recording: node.stop_recording()
                node.save_episode(success=False)
            elif k in (ord('q'), ord('Q'), 27):
                node.get_logger().info(
                    f"👋 success={node.ep_success}/{node.ep_total}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording: node.stop_recording()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
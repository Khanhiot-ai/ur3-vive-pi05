#!/usr/bin/env python3
"""
UR3/UR5 Teleop Follower — ur_rtde version

Khác biệt chính so với bản URBasic:
[1] Dùng ur_rtde (official) thay vì URBasic
    → servoL có lookahead_time + gain thật sự, không bị wipe
[2] Không cần custom URScript hack
[3] Không có "Realtime control not initialized" spam
[4] Latency thấp hơn (~5ms vs ~30ms)

Cài đặt:
    pip install ur_rtde

ROS interface giữ NGUYÊN 100%:
  - Subscribe: /ur_target_pose, /vive_right, /robot_origin_cmd
  - Publish:   /joint_states, /ee_pose, /ur_actual_pose, /ur_joint_states
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Header
from ur5_teleop_vive.msg import Xyzrpy

import rtde_control
import rtde_receive

import math
import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# OneEuroFilter
# ==============================================================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    @staticmethod
    def _alpha(t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 1e-6 or t_e > 0.5:
            self.t_prev = t
            self.x_prev = x
            return x
        dx = (x - self.x_prev) / t_e
        a_d = self._alpha(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def velocity(self):
        return self.dx_prev


# ==============================================================================
# MAIN NODE
# ==============================================================================
class URFollowViveSmooth(Node):
    def __init__(self):
        super().__init__('ur_follow_vive_smooth')

        # ════════════════════ KHU VỰC TUNE ════════════════════
        self.ROBOT_IP = '192.168.1.1'
        self.sim_mode = 'true'

        # Control loop
        self.control_dt = 0.010      # 100Hz
        self.state_pub_dt = 0.033    # 30Hz publish state

        # OneEuroFilter (position)
        self.use_filter = True
        self.filter_min_cutoff = 0.4   # rất mượt — chống jitter Vive (cũ: 0.7)
        self.filter_beta = 0.03        # ít sensitive hơn (cũ: 0.05)
        self.filter_d_cutoff = 1.0

        # Velocity / Acceleration / Jerk
        self.max_vel = 0.5      # giảm 50% (cũ: 1.0)
        self.max_accel = 2.0    # giảm 50% (cũ: 4.0)
        self.max_jerk = 30.0    # giảm 50% (cũ: 60.0)

        # Feed-forward
        self.lookahead_time_ff = 0.04
        self.use_feedforward = True

        # Safety
        self.glitch_threshold = 0.15   # 15cm — tay người vung 1m/s = 10mm/cycle nên OK
        self.min_z = 0.05
        self.TCP_OFFSET = 0.175

        # Orientation safety (chặn singularity → IK fail)
        # UR3 dễ singularity khi xoay nhiều quanh top-down. Clamp 45°.
        self.MAX_TILT_FROM_TOPDOWN_DEG = 90.0   # tilt tối đa từ top-down
        self.TOP_DOWN_RV = np.array([np.pi, 0.0, 0.0])

        # Deadzone
        self.deadzone_enter = 0.010   # 10mm — ra khỏi deadzone khi vượt (cũ: 3mm)
        self.deadzone_exit = 0.005    # 5mm — vào deadzone khi dưới (cũ: 1mm)

        # ★ SERVOL params (RTDE — KHÔNG bị wipe như URBasic) ★
        # lookahead_time: 0.03-0.2 (s). 0.1 sweet spot mượt + responsive.
        # gain: 100-2000. 300 cho UR3 mượt; 600+ cho responsive nhưng cứng.
        self.SERVO_LOOKAHEAD = 0.15  # smoothing nhiều hơn (cũ: 0.1)
        self.SERVO_GAIN = 300

        # moveJ/moveL params (cho home, origin, wrist rotation)
        self.MOVE_VELOCITY = 0.8
        self.MOVE_ACCELERATION = 0.8
        self.WRIST_ROT_STEP_DEG = 5

        # Orientation filter
        self._orient_filter_alpha = 0.3   # smooth orientation (cũ: 0.5)
        self.orient_deadzone_deg = 2.0    # ★ xoay < 2° thì coi như đứng yên (chống jitter)
        self._orient_filtered_rv = None

        self.debug_mode = False
        # ══════════════════════════════════════════════════════

        self._log_config()

        # --- Initialize position filter ---
        if self.use_filter:
            t0 = time.time()
            self.fx = OneEuroFilter(t0, 0.0, min_cutoff=self.filter_min_cutoff,
                                    beta=self.filter_beta, d_cutoff=self.filter_d_cutoff)
            self.fy = OneEuroFilter(t0, 0.0, min_cutoff=self.filter_min_cutoff,
                                    beta=self.filter_beta, d_cutoff=self.filter_d_cutoff)
            self.fz = OneEuroFilter(t0, 0.0, min_cutoff=self.filter_min_cutoff,
                                    beta=self.filter_beta, d_cutoff=self.filter_d_cutoff)
            self._filter_initialized = False

        # --- State variables ---
        self.target_pos_raw = None
        self.target_pos_filtered = None
        self.target_vel_filtered = np.zeros(3)
        self.target_quat = None
        self.joy_data = None

        self._cached_tcp_pose = None
        self._cached_joint_pos = None
        self._state_lock = threading.Lock()

        self.cmd_vel = np.zeros(3)
        self.cmd_accel = np.zeros(3)
        self.cmd_pos = None

        # movej/moveL async state
        self.move_in_progress = False
        self.move_lock = threading.Lock()

        # Servo control lock
        self.servo_lock = threading.Lock()
        self.servo_active = False

        self.in_deadzone = False
        self.is_moving_to_origin = False
        self.origin_target = None
        self.glitch_count = 0

        # Baseline cho relative orientation
        # Tại Ctrl_R PRESS: ghi cả Vive và TCP orientation
        # → TCP target = TCP_baseline * delta(Vive_now, Vive_baseline)
        self._baseline_vive_rot = None
        self._baseline_tcp_rot = None
        self._was_button_pressed = False

        # --- QoS ---
        qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- Subscribers ---
        self.create_subscription(PoseStamped, '/ur_target_pose',
                                 self.vive_target_cb, qos_be)
        self.create_subscription(Joy, '/vive_right',
                                 self.vive_joy_cb, qos_be)
        self.create_subscription(geometry_msgs.msg.Pose, '/robot_origin_cmd',
                                 self.origin_cmd_callback, 10)

        # --- Publishers ---
        self.joint_states_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(Xyzrpy, '/ee_pose', 10)
        self.actual_pose_pub = self.create_publisher(Xyzrpy, '/ur_actual_pose', 10)
        self.joint_pub = self.create_publisher(JointState, '/ur_joint_states', 10)

        # --- Connect via RTDE ---
        try:
            self.get_logger().info(f"🔌 Connecting RTDE @ {self.ROBOT_IP}...")
            self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
            self.get_logger().info(
                f"✓ RTDE connected — lookahead={self.SERVO_LOOKAHEAD}s gain={self.SERVO_GAIN}")
        except Exception as e:
            self.get_logger().error(f"✗ RTDE connection FAILED: {e}")
            self.get_logger().error("   pip install ur_rtde + robot trong remote control mode")
            raise

        # --- Home position ---
        self.robot_startposition = [-0.1834, -1.4779, 1.6630,
                                     -1.7602, -1.5327, 2.9670]
        self.get_logger().info("Moving to home position...")
        self.rtde_c.moveJ(self.robot_startposition,
                          self.MOVE_VELOCITY, self.MOVE_ACCELERATION)
        time.sleep(0.5)

        initial_pose = np.array(self.rtde_r.getActualTCPPose())
        self.cmd_pos = initial_pose[:3].copy()
        with self._state_lock:
            self._cached_tcp_pose = initial_pose
            self._cached_joint_pos = np.array(self.rtde_r.getActualQ())

        # --- Message templates ---
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ee_pose_msg = Xyzrpy()
        self.ee_pose_msg.header = Header()
        self.ee_pose_msg.header.frame_id = 'base'
        self.actual_pose_msg = Xyzrpy()
        self.actual_pose_msg.header = Header()
        self.actual_pose_msg.header.frame_id = 'base'

        # --- Timers ---
        self.timer = self.create_timer(self.control_dt, self.control_action)
        self.state_timer = self.create_timer(self.state_pub_dt, self.update_robot_state)

        self.get_logger().info("╔═══════════════════════════════════════════╗")
        self.get_logger().info(f"║  SYSTEM READY — RTDE @ {1/self.control_dt:.0f}Hz, state {1/self.state_pub_dt:.0f}Hz  ║")
        self.get_logger().info("╚═══════════════════════════════════════════╝")

    # ----------------------------------------------------------------------
    def _log_config(self):
        self.get_logger().info("┌──────── UR Teleop ur_rtde ────────┐")
        self.get_logger().info(f"│ Robot IP:    {self.ROBOT_IP}")
        self.get_logger().info(f"│ Control:     {1/self.control_dt:.0f}Hz (RTDE)")
        self.get_logger().info(f"│ Max vel:     {self.max_vel} m/s")
        self.get_logger().info(f"│ Max accel:   {self.max_accel} m/s²")
        self.get_logger().info(f"│ Max jerk:    {self.max_jerk} m/s³")
        self.get_logger().info(f"│ Filter:      ON (min_cut={self.filter_min_cutoff}, β={self.filter_beta})")
        self.get_logger().info(f"│ Feed-fwd:    {'ON' if self.use_feedforward else 'OFF'} ({self.lookahead_time_ff*1000:.0f}ms)")
        self.get_logger().info(f"│ ★ Servo:     lookahead={self.SERVO_LOOKAHEAD}s gain={self.SERVO_GAIN}")
        self.get_logger().info("└────────────────────────────────────┘")

    # ----------------------------------------------------------------------
    def update_robot_state(self):
        try:
            tcp = np.array(self.rtde_r.getActualTCPPose())
            jq = np.array(self.rtde_r.getActualQ())
        except Exception as e:
            self.get_logger().warn(f"RTDE read failed: {e}", throttle_duration_sec=2.0)
            return

        with self._state_lock:
            self._cached_tcp_pose = tcp
            self._cached_joint_pos = jq

        now = self.get_clock().now().to_msg()
        self.actual_pose_msg.header.stamp = now
        self.actual_pose_msg.x = float(tcp[0]); self.actual_pose_msg.y = float(tcp[1])
        self.actual_pose_msg.z = float(tcp[2]); self.actual_pose_msg.roll = float(tcp[3])
        self.actual_pose_msg.pitch = float(tcp[4]); self.actual_pose_msg.yaw = float(tcp[5])
        self.actual_pose_pub.publish(self.actual_pose_msg)

        js = JointState()
        js.header.stamp = now
        js.name = self.joint_names
        js.position = [float(x) for x in jq]
        self.joint_pub.publish(js)
        self.joint_states_pub.publish(js)

    def _get_cached_tcp(self):
        with self._state_lock:
            return None if self._cached_tcp_pose is None else self._cached_tcp_pose.copy()

    # ----------------------------------------------------------------------
    def vive_target_cb(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0:
            t = time.time()

        raw = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        if self.use_filter:
            if not self._filter_initialized:
                self.fx.x_prev = raw[0]; self.fx.t_prev = t
                self.fy.x_prev = raw[1]; self.fy.t_prev = t
                self.fz.x_prev = raw[2]; self.fz.t_prev = t
                self._filter_initialized = True
            fx = self.fx(t, raw[0])
            fy = self.fy(t, raw[1])
            fz = self.fz(t, raw[2])
            filtered = np.array([fx, fy, fz])
            vel = np.array([self.fx.velocity(), self.fy.velocity(), self.fz.velocity()])
        else:
            filtered = raw
            vel = np.zeros(3)

        self.target_pos_raw = raw
        self.target_pos_filtered = filtered
        self.target_vel_filtered = vel
        self.target_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])

    def vive_joy_cb(self, msg):
        self.joy_data = msg

    def origin_cmd_callback(self, msg):
        self.origin_target = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.origin_quat = np.array([
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w])
        self.is_moving_to_origin = True
        self.get_logger().info(f"🎯 Origin cmd: {self.origin_target}")
        threading.Thread(target=self._move_to_origin_thread, daemon=True).start()

    def _move_to_origin_thread(self):
        try:
            with self.servo_lock:
                if self.servo_active:
                    self.rtde_c.servoStop()
                    self.servo_active = False

            # Origin: chỉ di chuyển position, orientation = TOP-DOWN
            rv = self.TOP_DOWN_RV

            target_pose = [float(self.origin_target[0]),
                           float(self.origin_target[1]),
                           float(self.origin_target[2]),
                           float(rv[0]), float(rv[1]), float(rv[2])]

            self.rtde_c.moveL(target_pose, self.MOVE_VELOCITY, self.MOVE_ACCELERATION)
            time.sleep(0.2)

            self.cmd_pos = np.array(target_pose[:3])
            self.cmd_vel = np.zeros(3)
            self.cmd_accel = np.zeros(3)
            self._orient_filtered_rv = np.array(target_pose[3:6])

            self.is_moving_to_origin = False
            self.get_logger().info("✅ Origin reached — servo unlocked")
        except Exception as e:
            self.get_logger().error(f"❌ Origin failed: {e}")
            self.is_moving_to_origin = False

    # ----------------------------------------------------------------------
    def publish_ee_pose_msg(self, pose):
        self.ee_pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.ee_pose_msg.x = float(pose[0]); self.ee_pose_msg.y = float(pose[1])
        self.ee_pose_msg.z = float(pose[2]); self.ee_pose_msg.roll = float(pose[3])
        self.ee_pose_msg.pitch = float(pose[4]); self.ee_pose_msg.yaw = float(pose[5])
        self.ee_pose_pub.publish(self.ee_pose_msg)

    # ----------------------------------------------------------------------
    def _start_wrist_rotation_async(self, direction):
        with self.move_lock:
            if self.move_in_progress:
                return
            self.move_in_progress = True

        def _do_movej():
            try:
                with self.servo_lock:
                    if self.servo_active:
                        self.rtde_c.servoStop()
                        self.servo_active = False

                with self._state_lock:
                    q = self._cached_joint_pos.copy()
                q[-1] += math.radians(self.WRIST_ROT_STEP_DEG) * direction

                self.rtde_c.moveJ(list(q), self.MOVE_VELOCITY, self.MOVE_ACCELERATION)
                time.sleep(0.1)

                new_pose = np.array(self.rtde_r.getActualTCPPose())
                self.cmd_pos = new_pose[:3].copy()
                self.cmd_vel = np.zeros(3)
                self.cmd_accel = np.zeros(3)
                self._orient_filtered_rv = new_pose[3:6].copy()
            except Exception as e:
                self.get_logger().error(f"Movej failed: {e}")
            finally:
                with self.move_lock:
                    self.move_in_progress = False

        threading.Thread(target=_do_movej, daemon=True).start()

    # ----------------------------------------------------------------------
    def _plan_smooth_step(self, target_pos, target_vel, dt):
        error = target_pos - self.cmd_pos
        dist = float(np.linalg.norm(error))

        if self.in_deadzone:
            if dist > self.deadzone_enter:
                self.in_deadzone = False
                self.get_logger().debug(
                    f"🔓 Out of deadzone (dist={dist*1000:.1f}mm)")
        else:
            if dist < self.deadzone_exit:
                self.in_deadzone = True
                # Reset cmd_vel để robot dừng hẳn
                self.cmd_vel = np.zeros(3)
                self.cmd_accel = np.zeros(3)

        if self.in_deadzone:
            desired_vel = np.zeros(3)
        else:
            kp = min(1.0 / dt, 10.0)
            desired_vel = kp * error
            if self.use_feedforward:
                desired_vel = desired_vel + target_vel * 0.5

        v_norm = float(np.linalg.norm(desired_vel))
        if v_norm > self.max_vel:
            desired_vel = desired_vel * (self.max_vel / v_norm)

        desired_accel = (desired_vel - self.cmd_vel) / dt
        a_norm = float(np.linalg.norm(desired_accel))
        if a_norm > self.max_accel:
            desired_accel = desired_accel * (self.max_accel / a_norm)

        desired_jerk = (desired_accel - self.cmd_accel) / dt
        j_norm = float(np.linalg.norm(desired_jerk))
        if j_norm > self.max_jerk:
            desired_jerk = desired_jerk * (self.max_jerk / j_norm)
            desired_accel = self.cmd_accel + desired_jerk * dt

        new_vel = self.cmd_vel + desired_accel * dt
        nv = float(np.linalg.norm(new_vel))
        if nv > self.max_vel:
            new_vel = new_vel * (self.max_vel / nv)

        new_pos = self.cmd_pos + new_vel * dt
        self.cmd_accel = desired_accel
        self.cmd_vel = new_vel
        self.cmd_pos = new_pos
        return new_pos

    # ----------------------------------------------------------------------
    def _filter_orient(self, rv):
        """SLERP filter + DEADZONE: nếu xoay < 2° thì giữ nguyên (chống jitter)"""
        from scipy.spatial.transform import Slerp
        if self._orient_filtered_rv is None:
            self._orient_filtered_rv = rv.copy()
            return rv.copy()
        try:
            R_prev = R.from_rotvec(self._orient_filtered_rv)
            R_new = R.from_rotvec(rv)

            # ★ Orientation deadzone: nếu góc lệch < ngưỡng → giữ nguyên ★
            angle_diff_rad = (R_prev.inv() * R_new).magnitude()
            angle_diff_deg = math.degrees(angle_diff_rad)
            if angle_diff_deg < self.orient_deadzone_deg:
                return self._orient_filtered_rv.copy()

            key_rots = R.concatenate([R_prev, R_new])
            slerp = Slerp([0, 1], key_rots)
            out = slerp(self._orient_filter_alpha).as_rotvec()
            self._orient_filtered_rv = out.copy()
            return out
        except Exception:
            return self._orient_filtered_rv.copy()

    def _safe_orient(self, new_rv, current_rv):
        from scipy.spatial.transform import Slerp
        MAX_DEG_PER_FRAME = 6.0
        # Lớp 1: rate limit
        try:
            R_new = R.from_rotvec(new_rv)
            R_cur = R.from_rotvec(current_rv)
            angle_deg = np.degrees((R_new * R_cur.inv()).magnitude())
            if angle_deg > MAX_DEG_PER_FRAME and angle_deg > 0:
                t = MAX_DEG_PER_FRAME / angle_deg
                slerp = Slerp([0, 1], R.concatenate([R_cur, R_new]))
                new_rv = slerp(t).as_rotvec()
        except Exception:
            return current_rv.copy()
        # Lớp 2: tilt clamp khỏi top-down (chống singularity UR3)
        try:
            R_target = R.from_rotvec(new_rv)
            R_topdown = R.from_rotvec(self.TOP_DOWN_RV)
            tilt_deg = np.degrees((R_target * R_topdown.inv()).magnitude())
            if tilt_deg > self.MAX_TILT_FROM_TOPDOWN_DEG:
                self.get_logger().warn(
                    f"⚠ Tilt {tilt_deg:.0f}° > {self.MAX_TILT_FROM_TOPDOWN_DEG}° clamped",
                    throttle_duration_sec=1.0)
                t = self.MAX_TILT_FROM_TOPDOWN_DEG / tilt_deg
                slerp = Slerp([0, 1], R.concatenate([R_topdown, R_target]))
                new_rv = slerp(t).as_rotvec()
        except Exception:
            pass
        return new_rv

    # ----------------------------------------------------------------------
    # SERVO L wrapper — dùng lookahead THẬT
    # ----------------------------------------------------------------------
    def _servo_to_pose(self, pose_6d):
        with self.servo_lock:
            try:
                ok = self.rtde_c.servoL(
                    list(pose_6d),
                    0.0, 0.0,               # speed, accel (ignored in servo)
                    self.control_dt,        # time
                    self.SERVO_LOOKAHEAD,   # lookahead_time
                    self.SERVO_GAIN         # gain
                )
                # Một số bản ur_rtde return bool — nếu False, script stopped
                if ok is False:
                    self.get_logger().warn(
                        "⚠ servoL returned False — RTDE script stopped, đang reconnect...",
                        throttle_duration_sec=2.0)
                    self._reconnect_rtde()
                self.servo_active = True
            except Exception as e:
                self.get_logger().warn(f"servoL failed: {e}", throttle_duration_sec=2.0)
                self._reconnect_rtde()

    def _reconnect_rtde(self):
        """Reconnect khi RTDE script bị dừng (vd: do IK fail trên pendant)."""
        try:
            self.rtde_c.disconnect()
        except Exception:
            pass
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
            self.servo_active = False
            # Re-seed cmd_pos từ pose thực tế để tránh glitch giả
            try:
                actual = np.array(self.rtde_r.getActualTCPPose())
                self.cmd_pos = actual[:3].copy()
                self.cmd_vel = np.zeros(3)
                self.cmd_accel = np.zeros(3)
                self._orient_filtered_rv = actual[3:6].copy()
            except Exception:
                pass
            self.get_logger().info("✓ RTDE reconnected — cmd_pos re-seeded")
        except Exception as e:
            self.get_logger().error(
                f"Reconnect failed: {e}", throttle_duration_sec=5.0)

    # ----------------------------------------------------------------------
    # CONTROL LOOP (100Hz)
    # ----------------------------------------------------------------------
    def control_action(self):
        current_pose = self._get_cached_tcp()
        if current_pose is None:
            return

        if self.is_moving_to_origin:
            return
        with self.move_lock:
            if self.move_in_progress:
                return

        if self.target_pos_filtered is None or self.joy_data is None:
            self.get_logger().warn(
                f"⚠ Waiting data: target={self.target_pos_filtered is not None}, "
                f"joy={self.joy_data is not None}",
                throttle_duration_sec=2.0)
            return

        # Glitch detection trong control_action tự xử lý jump.
        # Sau Home, cmd_pos được seed bởi origin_target → target nhỏ hơn glitch_threshold.

        target_pos = self.target_pos_filtered.copy()
        target_vel = self.target_vel_filtered

        if self.sim_mode == 'false':
            target_pos[2] -= self.TCP_OFFSET

        is_button = (self.joy_data.buttons[0] == 1)

        # ★ Button transition detection cho baseline orientation ★
        if is_button and not self._was_button_pressed:
            # FALSE → TRUE: ghi baseline Vive + TCP orientation tại thời điểm này
            if self.target_quat is not None:
                self._baseline_vive_rot = R.from_quat(self.target_quat)
                # TCP baseline = orientation hiện tại của TCP (KHÔNG phải top-down)
                # → Sau Home, TCP đang ở Vive orientation, dùng nó làm gốc
                self._baseline_tcp_rot = R.from_rotvec(current_pose[3:6])
                self.get_logger().info(
                    "🎯 Baseline captured — TCP follow Vive rotation from here")
            # Reset orient filter về TCP hiện tại
            self._orient_filtered_rv = current_pose[3:6].copy()
        elif not is_button and self._was_button_pressed:
            # TRUE → FALSE: nhả button, không clear baseline (giữ để lần sau)
            pass
        self._was_button_pressed = is_button

        self.get_logger().info(
            f"📊 button={is_button} target=[{target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:+.3f}]",
            throttle_duration_sec=2.0)

        if self.use_feedforward:
            target_pos = target_pos + target_vel * self.lookahead_time_ff

        if is_button:
            if target_pos[2] < self.min_z:
                self.get_logger().warn(
                    f"⛔ Z too low: {target_pos[2]:.3f}m < {self.min_z}m",
                    throttle_duration_sec=1.0)
                self._servo_to_pose(current_pose)
                return

            dist_to_target = float(np.linalg.norm(target_pos - self.cmd_pos))
            if dist_to_target > self.glitch_threshold:
                # Thay vì hold đứng im, drag cmd_pos về target từng bước nhỏ
                # → tay người di chuyển xa, robot từ từ bắt kịp
                self.get_logger().warn(
                    f"⚠ Large jump {dist_to_target*100:.1f}cm — dragging slowly",
                    throttle_duration_sec=1.0)
                direction = (target_pos - self.cmd_pos) / dist_to_target
                # Bước tối đa = glitch_threshold mỗi frame, đảm bảo dưới limit
                step = direction * min(self.glitch_threshold * 0.5, dist_to_target)
                target_pos = self.cmd_pos + step
                target_vel = np.zeros(3)  # không feed-forward

            next_pos = self._plan_smooth_step(target_pos, target_vel, self.control_dt)

            # ★ RELATIVE ORIENTATION (full 6DOF: roll, pitch, yaw, wrist3) ★
            # Baseline ghi lại lúc Ctrl_R PRESS (False→True).
            # TCP_orient = delta(Vive_now, Vive_baseline) * TCP_baseline
            # → Sau Home, TCP_baseline = Vive_at_home → TCP và Vive trùng trục
            if (self._baseline_vive_rot is not None
                    and self._baseline_tcp_rot is not None
                    and self.target_quat is not None):
                try:
                    R_current = R.from_quat(self.target_quat)
                    R_delta = R_current * self._baseline_vive_rot.inv()
                    R_target = R_delta * self._baseline_tcp_rot
                    orient_rv = R_target.as_rotvec()
                except Exception:
                    orient_rv = current_pose[3:6]
            else:
                orient_rv = current_pose[3:6]

            orient_rv = self._filter_orient(orient_rv)
            orient_rv = self._safe_orient(orient_rv, current_pose[3:6])

            cmd = np.concatenate([next_pos, orient_rv])
            self._servo_to_pose(cmd)
            self.publish_ee_pose_msg(cmd)

            if self.debug_mode:
                err = np.linalg.norm(target_pos - next_pos) * 1000
                v = np.linalg.norm(self.cmd_vel)
                self.get_logger().info(
                    f"err={err:.1f}mm v={v:.2f}m/s",
                    throttle_duration_sec=0.2)

        else:
            self.cmd_vel = self.cmd_vel * 0.85
            self.cmd_accel = self.cmd_accel * 0.85
            if np.linalg.norm(self.cmd_vel) < 0.001:
                self.cmd_vel = np.zeros(3)
                self.cmd_accel = np.zeros(3)
            else:
                self.cmd_pos = self.cmd_pos + self.cmd_vel * self.control_dt
                hold = np.concatenate([self.cmd_pos, current_pose[3:6]])
                self._servo_to_pose(hold)

    # ----------------------------------------------------------------------
    def shutdown(self):
        try:
            with self.servo_lock:
                if self.servo_active:
                    self.rtde_c.servoStop()
                    self.servo_active = False
            self.rtde_c.stopScript()
        except Exception:
            pass
        try: self.rtde_c.disconnect()
        except Exception: pass
        try: self.rtde_r.disconnect()
        except Exception: pass


def main(args=None):
    rclpy.init(args=args)
    node = URFollowViveSmooth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
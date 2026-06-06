#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker
from ur5_teleop_vive.msg import Xyzrpy
import math
import numpy as np
import copy
import time

try:
    import rtde_control
    import rtde_receive
    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False
    raise ImportError("Cài ur_rtde: pip install ur_rtde")

# ==============================================================================
# CLASS: OneEuroFilter
# ==============================================================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev
        
        j_dx = (x - self.x_prev) / t_e
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = self.exponential_smoothing(a_d, j_dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * abs(dx)
        a = self.smoothing_factor(t_e, cutoff)
        x_filtered = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_filtered
        self.dx_prev = dx
        self.t_prev = t
        return x_filtered

# ==============================================================================
# MAIN NODE: URFollowVive
# ==============================================================================
class URFollowVive(Node):
    def __init__(self):
        super().__init__('ur_follow_vive')
        
        # ########################################################################
        # KHU VỰC TUNE - CHỈNH THÔNG SỐ TẠI ĐÂY
        # ########################################################################
        
        # 1. Robot Connection
        self.sim_mode = 'true'
        self.ROBOT_IP = '192.168.1.1'  # ← ĐIỀN IP ROBOT
        
        # 2. Preset (Uncomment 1 trong 3)
        # --- PRESET 1: AN TOÀN (Người mới) ---
        # self.normal_max_speed = 0.5
        # self.filter_min_cutoff = 0.5
        # self.filter_beta = 0.005
        # self.deadzone = 0.003
        
        # --- PRESET 2: CÂN BẰNG (Khuyến nghị) ← ĐANG DÙNG ---
        self.normal_max_speed = 5
        self.filter_min_cutoff = 1.0
        self.filter_beta = 0.075
        self.deadzone = 0.002
        
        # --- PRESET 3: NHANH (Người có kinh nghiệm) ---
        # self.normal_max_speed = 1.2
        # self.filter_min_cutoff = 2.0
        # self.filter_beta = 0.01
        # self.deadzone = 0.001
        
        # 3. Advanced Settings (RTDE)
        self.use_filter = False
        self.control_dt = 0.010         # 100Hz servoL
        self.SERVO_LOOKAHEAD   = 0.15   # s - smoothing ur_rtde
        self.SERVO_GAIN        = 300    # stiffness
        self.glitch_threshold  = 0.15   # 15cm
        self.TCP_OFFSET        = 0.175
        self.min_z             = 0.05
        self.tcp_orient_thresh = 0.6
        self.ACCELERATION      = 0.5
        self.VELOCITY          = 0.5
        self.ACCELERATION_GRIPPER = 1.0
        self.VELOCITY_GRIPPER     = 1.0
        
        # Debug mode
        self.debug_mode = False  # True = Xem log chi tiết
        
        # ########################################################################
        
        self.get_logger().info(f'╔══════════════════════════════════════════╗')
        self.get_logger().info(f'║  UR3 RTDE TELEOPERATION - RTDE VERSION  ║')
        self.get_logger().info(f'╠══════════════════════════════════════════╣')
        self.get_logger().info(f'║  Robot IP: {self.ROBOT_IP:27} ║')
        self.get_logger().info(f'║  Mode: {self.sim_mode:33} ║')
        self.get_logger().info(f'║  Max Speed: {self.normal_max_speed:.2f} m/s{" "*24}║')
        self.get_logger().info(f'║  Filter: {"ON" if self.use_filter else "OFF":34} ║')
        self.get_logger().info(f'╚══════════════════════════════════════════╝')
        
        # Initialize Filter
        if self.use_filter:
            t0 = time.time()
            self.fx = OneEuroFilter(t0, 0, min_cutoff=self.filter_min_cutoff, beta=self.filter_beta)
            self.fy = OneEuroFilter(t0, 0, min_cutoff=self.filter_min_cutoff, beta=self.filter_beta)
            self.fz = OneEuroFilter(t0, 0, min_cutoff=self.filter_min_cutoff, beta=self.filter_beta)
            self.get_logger().info("✓ OneEuroFilter initialized")
        
        # Initialize variables
        self.vive_gripper_data = None
        self.joy_data = None
        self.used_movej = 0
        
        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        # self.vive_gripper_sub = self.create_subscription(
        #     Marker, '/vive_gripper', self.vive_gripper_cb, qos_profile
        # )
        self.vive_gripper_sub = self.create_subscription(
        PoseStamped,          # Đổi kiểu tin nhắn
        '/ur_target_pose',    # Đổi tên Topic khớp với Node 3
        self.vive_gripper_cb, # Giữ nguyên tên hàm callback cho tiện
        qos_profile
        )

        self.vive_joy_sub = self.create_subscription(
            Joy, '/vive_right', self.vive_joy_cb, qos_profile
        )
        
        # ✅ [NEW] Subscriber cho origin command từ Node 3
        self.origin_cmd_sub = self.create_subscription(
            geometry_msgs.msg.Pose,
            '/robot_origin_cmd',
            self.origin_cmd_callback,
            10
        )
        self.get_logger().info("✅ Subscribed to /robot_origin_cmd")

        # ✅ [NEW] Subscriber /auto_home — record_all gửi khi bấm S
        from std_msgs.msg import Bool as _Bool
        self.auto_home_sub = self.create_subscription(
            _Bool,
            '/auto_home',
            self.auto_home_callback,
            10
        )
        self.get_logger().info("✅ Subscribed to /auto_home")
        self.is_going_home = False
        
        # ✅ [NEW] Flag để track origin movement
        self.is_moving_to_origin = False
        self.origin_target = None
        
        # Publishers
        # self.joint_states_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_states_pub = self.create_publisher(JointState, '/joint_states', qos_profile)
        # self.target_ee_marker_pub = self.create_publisher(Marker, 'target_gripper', 10)
        
        # Topic cũ (Command)
        self.ee_pose_pub = self.create_publisher(Xyzrpy, '/ee_pose', 10)
        # ---> [ADD] Topic mới (Actual Feedback)
        self.actual_pose_pub = self.create_publisher(Xyzrpy, '/ur_actual_pose', 10)

        # ── Publishers cho RoboMIND-style dataset ──
        self.ur_joint_pub = self.create_publisher(JointState, '/ur_joint_states', 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, '/ur_wrench', 10)
        self.joint_torque_pub = self.create_publisher(
            Float32MultiArray, '/ur_joint_torque', 10)
        self.get_logger().info("✅ Publishers: /ur_joint_states /ur_wrench /ur_joint_torque")
        
        # Initialize Robot via ur_rtde
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.ROBOT_IP)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ROBOT_IP)
            self.get_logger().info(f"✓ RTDE connected: {self.ROBOT_IP}")
        except Exception as e:
            self.get_logger().error(f"✗ RTDE CONNECTION FAILED: {e}")
            raise
        
        # Move to home position
        self.robot_startposition = (
            -0.1834, -1.4779, 1.6630,
            -1.7602, -1.5327, 4.5698
        )
        self.get_logger().info("Moving to home position...")
        self.rtde_c.moveJ(
            self.robot_startposition,
            self.VELOCITY,
            self.ACCELERATION
        )
        # servoJ warmup để RTDE sẵn sàng
        time.sleep(0.2)
        
        # Setup messages
        self.joint_states_msg = JointState()
        self.joint_states_msg.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.ee_pose_msg = Xyzrpy()
        self.ee_pose_msg.header = Header()
        self.ee_pose_msg.header.frame_id = 'base'
        
        # Message cho Topic mới
        self.actual_pose_msg = Xyzrpy()
        self.actual_pose_msg.header = Header()
        self.actual_pose_msg.header.frame_id = 'base'
        
        # Create control timer
        self.timer = self.create_timer(self.control_dt, self.control_action)
        freq = 1.0 / self.control_dt
        self.get_logger().info(f"✓ Control loop: {freq:.0f} Hz ({self.control_dt*1000:.1f}ms)")
        self.get_logger().info("╔══════════════════════════════════════════╗")
        self.get_logger().info("║         SYSTEM READY TO CONTROL         ║")
        self.get_logger().info("╚══════════════════════════════════════════╝")

    def vive_gripper_cb(self, msg):
        """Receive Vive data and filter immediately"""
        if self.use_filter and hasattr(self, 'fx'):
            t = time.time()
            x = self.fx(t, msg.pose.position.x)
            y = self.fy(t, msg.pose.position.y)
            z = self.fz(t, msg.pose.position.z)
            
            self.vive_gripper_data = copy.deepcopy(msg)
            self.vive_gripper_data.pose.position.x = x
            self.vive_gripper_data.pose.position.y = y
            self.vive_gripper_data.pose.position.z = z
        else:
            self.vive_gripper_data = msg

    def vive_joy_cb(self, msg):
        self.joy_data = msg

    def origin_cmd_callback(self, msg):
        """Nhận lệnh di chuyển đến origin từ Node 3"""
        self.origin_target = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])
        self.is_moving_to_origin = True
        
        self.get_logger().info(f"🎯 [ORIGIN CMD] Received: {self.origin_target}")
        
        # Di chuyển robot bằng movel (blocking)
        try:
            # Lấy pose hiện tại
            current_pose = self.rtde_r.getActualTCPPose()
            
            # Tạo target pose (giữ nguyên orientation)
            target_pose = [
                float(self.origin_target[0]),
                float(self.origin_target[1]),
                float(self.origin_target[2]),
                current_pose[3],  # rx
                current_pose[4],  # ry
                current_pose[5]   # rz
            ]
            
            # Di chuyển bằng movel (smooth)
            self.get_logger().info(f"🤖 Moving to origin: {target_pose[:3]}")
            self.rtde_c.servoStop()
            self.rtde_c.moveL(
                target_pose,
                self.VELOCITY,
                self.ACCELERATION
            )
            
            # Đợi robot đến vị trí
            time.sleep(2)
            
            # Reset flag
            self.is_moving_to_origin = False
            self.get_logger().info("✅ Origin reached!")
            
            # Khởi động lại realtime control
            # RTDE servoL không cần reinit - tự động tiếp tục
            
        except Exception as e:
            self.get_logger().error(f"❌ Origin movement failed: {e}")
            self.is_moving_to_origin = False

    def auto_home_callback(self, msg):
        """Nhận lệnh về HOME từ record_all (khi bấm S)."""
        if not msg.data:
            return
        if self.is_going_home:
            return
        self.is_going_home = True
        self.get_logger().info("🏠 [AUTO HOME] Đưa robot về home position...")
        try:
            # Dừng servo realtime trước khi moveJ
            self.rtde_c.servoStop()
            time.sleep(0.1)
            # moveJ về home joint config
            self.rtde_c.moveJ(
                self.robot_startposition,
                self.VELOCITY,
                self.ACCELERATION
            )
            time.sleep(0.2)
            self.get_logger().info("✅ Đã về home!")
        except Exception as e:
            self.get_logger().error(f"❌ Auto home failed: {e}")
        finally:
            self.is_going_home = False

    def publish_ee_pose_msg(self, current_pose):
        self.ee_pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.ee_pose_msg.x = float(current_pose[0])
        self.ee_pose_msg.y = float(current_pose[1])
        self.ee_pose_msg.z = float(current_pose[2])
        self.ee_pose_msg.roll = float(current_pose[3])
        self.ee_pose_msg.pitch = float(current_pose[4])
        self.ee_pose_msg.yaw = float(current_pose[5])
        self.ee_pose_pub.publish(self.ee_pose_msg)

    # ---> Ham moi cho topic moi
    def publish_actual_pose_msg(self, current_pose):
        self.actual_pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.actual_pose_msg.x = float(current_pose[0])
        self.actual_pose_msg.y = float(current_pose[1])
        self.actual_pose_msg.z = float(current_pose[2])
        self.actual_pose_msg.roll = float(current_pose[3])
        self.actual_pose_msg.pitch = float(current_pose[4])
        self.actual_pose_msg.yaw = float(current_pose[5])
        self.actual_pose_pub.publish(self.actual_pose_msg)

    def control_action(self):
        """Main control loop (Non-blocking) - DEBUG VERSION"""
        
        # ✅ [NEW] Nếu đang di chuyển đến origin → Skip control loop
        if self.is_moving_to_origin:
            return

        # ✅ [NEW] Nếu đang về home → Skip control loop
        if self.is_going_home:
            return
        
        # --- DEBUG 1: KIỂM TRA DỮ LIỆU ĐẦU VÀO ---
        if self.vive_gripper_data is None or self.joy_data is None:
            self.get_logger().warn(
                f"⚠ WAITING DATA: Gripper={self.vive_gripper_data is not None}, Joy={self.joy_data is not None}", 
                throttle_duration_sec=2.0
            )
            return

        # Lấy Pose thực ngay đầu vòng lặp
        current_pose = np.array(self.rtde_r.getActualTCPPose())
        
        # ---> [ADD] Publish Pose thực ra topic mới ngay lập tức
        self.publish_actual_pose_msg(current_pose)
        
        # Publish joint states
        self.joint_states_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_states_msg.position = [
            float(x) for x in self.rtde_r.getActualQ()
        ]
        self.joint_states_pub.publish(self.joint_states_msg)
        # Alias topic cho record_all.py
        self.ur_joint_pub.publish(self.joint_states_msg)

        # ── Publish joint torque (current × torque constant) ──
        try:
            currents = self.rtde_r.getActualCurrent()
            torque_consts = [0.075, 0.075, 0.075, 0.06, 0.06, 0.06]
            torques = [c * k for c, k in zip(currents, torque_consts)]
            tor_msg = Float32MultiArray()
            tor_msg.data = list(map(float, torques))
            self.joint_torque_pub.publish(tor_msg)
        except Exception:
            pass

        # ── Publish 6-DoF wrench tại TCP ──
        try:
            tcp_force = self.rtde_r.getActualTCPForce()
            w_msg = WrenchStamped()
            w_msg.header.stamp = self.get_clock().now().to_msg()
            w_msg.header.frame_id = "tool0"
            w_msg.wrench.force.x  = float(tcp_force[0])
            w_msg.wrench.force.y  = float(tcp_force[1])
            w_msg.wrench.force.z  = float(tcp_force[2])
            w_msg.wrench.torque.x = float(tcp_force[3])
            w_msg.wrench.torque.y = float(tcp_force[4])
            w_msg.wrench.torque.z = float(tcp_force[5])
            self.wrench_pub.publish(w_msg)
        except Exception:
            pass

        # Get target position
        target_pos = np.array([
            self.vive_gripper_data.pose.position.x,
            self.vive_gripper_data.pose.position.y,
            self.vive_gripper_data.pose.position.z
        ])
        
        if self.sim_mode == 'false':
            target_pos[2] -= self.TCP_OFFSET
            
        # Visualization marker
        # target_ee_marker = copy.deepcopy(self.vive_gripper_data)
        # target_ee_marker.color.r = 1.0; target_ee_marker.color.g = 0.64; target_ee_marker.color.b = 0.0; target_ee_marker.color.a = 0.3
        # target_ee_marker.header.stamp = self.get_clock().now().to_msg()

        # ========== CONTROL LOGIC (DEBUG ADDED) ==========
        
        # Tách điều kiện để log
        is_button_pressed = (self.joy_data.buttons[0] == 1)
        current_target_z = target_pos[2]
        is_height_safe = (current_target_z > self.min_z)

        if is_button_pressed:
            # --- DEBUG 2: KIỂM TRA ĐỘ CAO AN TOÀN ---
            if not is_height_safe:
                self.get_logger().warn(
                    f"⛔ BLOCKED: Too Low! Target Z={current_target_z:.3f} < Min Z={self.min_z}", 
                    throttle_duration_sec=1.0
                )
                # Giữ nguyên vị trí để robot không trôi
                self.rtde_c.servoL(
                current_pose.tolist(),
                self.VELOCITY,
                self.ACCELERATION,
                self.control_dt,
                self.SERVO_LOOKAHEAD,
                self.SERVO_GAIN
            )
                return # Thoát luôn

            # Nếu đã vào đây tức là ĐÃ ẤN NÚT và ĐỘ CAO OK
            if self.used_movej == 1:
                self.robot.init_realtime_control()
                time.sleep(0.5)
                self.used_movej = 0
            
            # Calculate movement
            curr_pos_xyz = current_pose[:3]
            goal_pos_xyz = target_pos
            diff_vector = goal_pos_xyz - curr_pos_xyz
            distance = np.linalg.norm(diff_vector)
            
            # --- DEBUG 3: KIỂM TRA KHOẢNG CÁCH (CHỐNG GIẬT) ---
            if distance > self.glitch_threshold:
                self.get_logger().error(
                    f'⚠ GLITCH PROTECTION: Jump detected ({distance:.3f}m > {self.glitch_threshold}m). Robot Halted.',
                    throttle_duration_sec=1.0
                )
                realtime_cmd = current_pose
                
            # Velocity Clamping
            else:
                max_step = self.normal_max_speed * self.control_dt
                
                if distance < self.deadzone:
                    # Log nhẹ nếu ở trong deadzone mà người dùng đang cố di chuyển
                    if distance > 0.0001: 
                        # self.get_logger().info(f"Deadzone: {distance:.4f}", throttle_duration_sec=2.0)
                        pass
                    realtime_cmd = current_pose
                elif distance > max_step:
                    scale = max_step / distance
                    next_pos_xyz = curr_pos_xyz + (diff_vector * scale)
                    realtime_cmd = np.concatenate([next_pos_xyz, current_pose[3:6]])
                else:
                    realtime_cmd = np.concatenate([goal_pos_xyz, current_pose[3:6]])
            
            self.rtde_c.servoL(
                realtime_cmd.tolist(),
                self.VELOCITY,
                self.ACCELERATION,
                self.control_dt,
                self.SERVO_LOOKAHEAD,
                self.SERVO_GAIN
            )
            # self.target_ee_marker_pub.publish(target_ee_marker)
            self.publish_ee_pose_msg(realtime_cmd)
            self.used_movej = 0
            
        elif abs(self.joy_data.axes[1]) > self.tcp_orient_thresh:
            # ... (Giữ nguyên phần xoay gripper) ...
            self.used_movej = 1
            q_rotated = list(self.joint_states_msg.position)
            rotation_step = math.radians(5)
            
            if self.joy_data.axes[1] > 0:
                q_rotated[-1] += rotation_step
            else:
                q_rotated[-1] -= rotation_step
                
            self.rtde_c.servoStop()
            self.rtde_c.moveJ(q_rotated, self.VELOCITY_GRIPPER, self.ACCELERATION_GRIPPER)
            self.publish_ee_pose_msg(current_pose)
            
        else:
            # --- DEBUG 4: LOG TRẠNG THÁI CHỜ (Optional) ---
            # Nếu bạn muốn biết joy có nhận không nhưng ko ấn nút
            # self.get_logger().info("Idle (Button 0 released)", throttle_duration_sec=5.0)
            pass

def main(args=None):
    rclpy.init(args=args)
    node = URFollowVive()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.rtde_c.servoStop()
        node.rtde_c.stopScript()
        node.rtde_c.disconnect()
        node.rtde_r.disconnect()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
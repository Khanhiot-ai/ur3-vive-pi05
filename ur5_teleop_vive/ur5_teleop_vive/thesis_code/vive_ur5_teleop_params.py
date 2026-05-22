#!/usr/bin/env python3
"""
Teleoperation với World Alignment Matrix

[FIX 1] Xóa DEBUG log spam 100Hz → throttle 1Hz
[FIX 2] Hoàn thiện apply_world_alignment: thêm BƯỚC 1-3 translation còn thiếu
[FIX 3] workspace_check tạo 1 lần trong __init__
[FIX 4] get_clock().now() gọi 1 lần dùng lại
[FIX 5] Thêm orientation thật từ tracker vào target (xoay tay = xoay TCP)
        Home → reset orientation về top-down (1,0,0,0)
"""

import sys
import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster, TransformListener, Buffer

try:
    from ur5_teleop_vive.msg import Xyzrpy
except ImportError:
    Xyzrpy = None

from pynput import keyboard

# Quaternion top-down cố định (gripper nhìn xuống) — dùng khi Home
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # [x, y, z, w]


# ============================================
# Relative Control Class — GIỮ NGUYÊN LOGIC, thêm orientation
# ============================================
class RelativeControlWithKeyboard:
    def __init__(self, logger):
        self.logger = logger
        self.tracker_origin = None
        self.robot_origin = None
        self.origin_rot = None   # R_tracker lúc Home (để tính delta rotation)
        self.tcp_origin_rot = None  # R_tcp lúc Home (top-down)
        self.origin_set = False
        self.home_pressed = False
        self.last_home_state = False

        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()

        self.logger.info("🎮 Relative Control initialized")
        self.logger.info("   Home: Set origin (press to capture)")

    def _on_key_press(self, key):
        try:
            self.logger.debug(f"Key pressed: {key}")
            if key == keyboard.Key.home:
                self.logger.info("✅ HOME KEY DETECTED!")
                self.home_pressed = True
        except AttributeError:
            pass

    def _on_key_release(self, key):
        try:
            if key == keyboard.Key.home:
                self.home_pressed = False
        except AttributeError:
            pass

    def update(self, tracker_pose_matrix, robot_current_pose, workspace_check=None, robot_current_quat=None):
        """
        Returns:
            (target_pos, target_quat, status, is_origin_cmd)
            target_quat: np.array [x,y,z,w] — None nếu chưa set origin
        """
        # Lưu để _set_origin dùng
        self._robot_current_quat = robot_current_quat

        # AUTO-ORIGIN (kiểu Pika): pose đầu tiên ổn định tự làm origin
        # Đợi 30 frames stable (~0.3s @ 100Hz) để tránh dùng pose lúc tracker chưa hội tụ
        if not self.origin_set:
            if not hasattr(self, "_frame_count"):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count >= 30:
                self.logger.info("🎯 AUTO-ORIGIN triggered after stable tracking")
                target_pos, target_quat = self._set_origin(
                    tracker_pose_matrix, robot_current_pose,
                    workspace_check=workspace_check,
                    robot_current_quat=robot_current_quat)
                if target_pos is not None:
                    return target_pos, target_quat, "Auto-origin set", True
                return None, None, "Auto-origin: tracker outside workspace", False
        home_just_pressed = self.home_pressed and not self.last_home_state
        self.last_home_state = self.home_pressed

        if home_just_pressed:
            target_pos, target_quat = self._set_origin(
                tracker_pose_matrix, robot_current_pose,
                workspace_check=workspace_check,
                robot_current_quat=getattr(self, '_robot_current_quat', None))
            if target_pos is not None:
                return target_pos, target_quat, "Origin set - Moving robot", True
            else:
                return None, None, "Origin set - Robot stays", False

        if not self.origin_set:
            return None, None, "Press Home to set origin", False

        target_pos, target_quat = self._calculate_target(tracker_pose_matrix)
        return target_pos, target_quat, "Active", False

    def _set_origin(self, tracker_pose_matrix, robot_current_pose, workspace_check=None, robot_current_quat=None):
        tracker_pos = tracker_pose_matrix[:3, 3].copy()
        is_safe = workspace_check(tracker_pos) if workspace_check is not None else True

        # Lưu Vive orientation tại Home để tính delta
        self.origin_rot = R.from_matrix(tracker_pose_matrix[:3, :3])

        # TCP base orientation = orientation HIỆN TẠI của TCP (không snap top-down)
        # → marker tại Home trùng khít với TCP ngay lập tức
        if robot_current_quat is not None:
            self.tcp_origin_rot = R.from_quat(robot_current_quat)
            current_quat = robot_current_quat.copy()
        else:
            self.tcp_origin_rot = R.from_quat(TOP_DOWN_QUAT)
            current_quat = TOP_DOWN_QUAT.copy()

        if is_safe:
            self.tracker_origin = tracker_pos
            self.robot_origin   = tracker_pos
            self.origin_set     = True
            self.logger.info("📍 [Home] Marker = TCP current orientation")
            self.logger.info(f"   TRACKER: X={tracker_pos[0]:.4f}, Y={tracker_pos[1]:.4f}, Z={tracker_pos[2]:.4f}")
            return tracker_pos, current_quat
        else:
            self.tracker_origin = tracker_pos
            self.robot_origin   = robot_current_pose.copy()
            self.origin_set     = True
            self.logger.warn("⚠️ [Home] Tracker outside workspace!")
            self.logger.info(f"   TRACKER (UNSAFE): X={tracker_pos[0]:.4f}, Y={tracker_pos[1]:.4f}, Z={tracker_pos[2]:.4f}")
            return None, None

    def _calculate_target(self, tracker_current_matrix):
        """
        Position: follow Vive delta from Home.
        Orientation: TOP-DOWN base + FULL delta rotation từ Vive (roll, pitch, yaw).
        """
        tracker_current = tracker_current_matrix[:3, 3]
        delta      = tracker_current - self.tracker_origin
        target_pos = self.robot_origin + delta

        # Full delta rotation từ lúc Home
        R_current   = R.from_matrix(tracker_current_matrix[:3, :3])
        R_delta     = R_current * self.origin_rot.inv()
        # Áp delta lên TOP_DOWN
        R_tcp_target = R_delta * self.tcp_origin_rot
        target_quat  = R_tcp_target.as_quat()

        return target_pos, target_quat

    def reset_origin(self):
        self.origin_set = False
        self.logger.info("🔄 Origin reset")

    def shutdown(self):
        self.listener.stop()


# -------------------------------------------------------

class ViveMarkerPub(Node):

    def __init__(self):
        super().__init__('vive_ur5_teleop')
        self._declare_parameters()
        self.cfg = self._load_config()

        self.base_frame = self.cfg.get('frames', {}).get('base', 'base')
        self.ee_frame   = self.cfg.get('frames', {}).get('ee', 'tool0')

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.world_alignment = None
        self.use_world_alignment = self.load_world_alignment()

        self.control_mode = self.cfg.get('control_mode', 'relative')

        if self.control_mode == 'relative':
            self.rel_control = RelativeControlWithKeyboard(self.get_logger())
            self.robot_current_pose = np.array([0.3, 0.2, 0.4])
            self.robot_current_quat = TOP_DOWN_QUAT.copy()  # default

            if Xyzrpy is not None:
                self.actual_pose_sub = self.create_subscription(
                    Xyzrpy, '/ur_actual_pose', self.actual_pose_callback, 10)
                self.get_logger().info("✅ Subscribed to /ur_actual_pose")
            else:
                self.get_logger().warn("⚠️ Xyzrpy message not available")

            self.origin_cmd_pub = self.create_publisher(Pose, '/robot_origin_cmd', 10)
            self.get_logger().info("✅ Publisher /robot_origin_cmd created")
        else:
            self.rel_control = None
            self.robot_current_pose = None
            self.origin_cmd_pub = None

        self.ee_marker_pub  = self.create_publisher(Marker, 'vive_gripper', 10)
        self.control_pub    = self.create_publisher(
            PoseStamped, '/ur_target_pose', rclpy.qos.qos_profile_sensor_data)
        self.robot_ws_pub   = self.create_publisher(Marker, '/robot_ws_marker', 10)

        self.setup_markers()

        # workspace_check tạo 1 lần
        ws_cfg = self.cfg['robot_ws']
        self._ws_center = [float(ws_cfg['x_mid']), float(ws_cfg['y_mid']), float(ws_cfg['z_mid'])]
        self._ws_len    = [float(ws_cfg['x_len']), float(ws_cfg['y_len']), float(ws_cfg['z_len'])]

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/right_controller_as_posestamped',
            self.pose_callback, qos_best_effort)

        self.viz_counter = 0

        alignment_mode   = "WORLD ALIGNMENT" if self.use_world_alignment else "HARDCODED ROTATION"
        control_mode_str = f" + {self.control_mode.upper()} MODE"
        self.get_logger().info(f"######### INIT done - {alignment_mode}{control_mode_str} #########")
        self.get_logger().info(f"Base frame: {self.base_frame}")

    # ------------------------------------------------------------------
    def load_world_alignment(self, filepath="world_alignment_matrix.txt"):
        try:
            self.world_alignment = np.loadtxt(filepath)
            if self.world_alignment.shape != (4, 4):
                self.get_logger().error(f"❌ Invalid matrix shape: {self.world_alignment.shape}")
                return False
            self.get_logger().info(f"✅ Loaded World Alignment Matrix from {filepath}")
            return True
        except FileNotFoundError:
            self.get_logger().warn(f"⚠️ {filepath} not found — fallback to hardcoded")
            return False
        except Exception as e:
            self.get_logger().error(f"❌ Error loading matrix: {e}")
            return False

    def _declare_parameters(self):
        self.declare_parameter('manipulation.offsets_real.trans_x', 0.60)
        self.declare_parameter('manipulation.offsets_real.trans_y', 1.0)
        self.declare_parameter('manipulation.offsets_real.trans_z', 0.848)
        self.declare_parameter('manipulation.offsets_real.rot_x', 0.0)
        self.declare_parameter('manipulation.offsets_real.rot_y', 0.0)
        self.declare_parameter('manipulation.offsets_real.rot_z', 0.0)
        self.declare_parameter('manipulation.offsets_sim.trans_x', -0.2)
        self.declare_parameter('manipulation.offsets_sim.trans_y', 2.5)
        self.declare_parameter('manipulation.offsets_sim.trans_z', 1.65)
        self.declare_parameter('manipulation.offsets_sim.rot_x', 0.0)
        self.declare_parameter('manipulation.offsets_sim.rot_y', 0.0)
        self.declare_parameter('manipulation.offsets_sim.rot_z', 95.5)
        self.declare_parameter('robot_ws.x_mid', -0.32)
        self.declare_parameter('robot_ws.x_len', 0.4)
        self.declare_parameter('robot_ws.y_mid', -0.005)
        self.declare_parameter('robot_ws.y_len', 0.4)
        self.declare_parameter('robot_ws.z_mid', 0.31)
        self.declare_parameter('robot_ws.z_len', 0.3)
        self.declare_parameter('frames.base', 'base')
        self.declare_parameter('frames.ee', 'tool0')
        self.declare_parameter('sim', 'true')
        self.declare_parameter('control_mode', 'relative')

    def _load_config(self):
        return {
            'manipulation': {
                'offsets_real': {k: self.get_parameter(f'manipulation.offsets_real.{k}').value
                                 for k in ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']},
                'offsets_sim':  {k: self.get_parameter(f'manipulation.offsets_sim.{k}').value
                                 for k in ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']},
            },
            'robot_ws':     {k: self.get_parameter(f'robot_ws.{k}').value
                             for k in ['x_mid','x_len','y_mid','y_len','z_mid','z_len']},
            'frames':       {'base': self.get_parameter('frames.base').value,
                             'ee':   self.get_parameter('frames.ee').value},
            'sim':          self.get_parameter('sim').value,
            'control_mode': self.get_parameter('control_mode').value,
        }

    def setup_markers(self):
        mesh_path = "file:///home/khanh/ur5_teleop_vive/ur5_teleop_vive/mesh/hand.dae"
        self.ee_marker = self.create_ee_marker(mesh_path, [0.0, 1.0, 0.0, 0.6])

        self.robot_ws_marker = Marker()
        self.robot_ws_marker.header.frame_id = self.base_frame
        self.robot_ws_marker.type  = Marker.CUBE
        self.robot_ws_marker.action = Marker.ADD
        self.robot_ws_marker.pose.orientation.w = 1.0
        self.robot_ws_marker.color.r = 0.0
        self.robot_ws_marker.color.g = 1.0
        self.robot_ws_marker.color.b = 0.0
        self.robot_ws_marker.color.a = 0.3
        ws = self.cfg['robot_ws']
        self.robot_ws_marker.pose.position.x = float(ws['x_mid'])
        self.robot_ws_marker.pose.position.y = float(ws['y_mid'])
        self.robot_ws_marker.pose.position.z = float(ws['z_mid'])
        self.robot_ws_marker.scale.x = float(ws['x_len'])
        self.robot_ws_marker.scale.y = float(ws['y_len'])
        self.robot_ws_marker.scale.z = float(ws['z_len'])

    def create_ee_marker(self, mesh_resource, color):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.ns = "robot"; marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = mesh_resource
        marker.action = Marker.ADD
        marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0
        marker.color.r = color[0]; marker.color.g = color[1]
        marker.color.b = color[2]; marker.color.a = color[3]
        return marker

    def is_point_inside_cuboid(self, point, center, lengths):
        return all(abs(point[i] - center[i]) <= lengths[i] / 2 for i in range(3))

    # ------------------------------------------------------------------
    # apply_world_alignment — đầy đủ BƯỚC 1-4
    def apply_world_alignment(self, pose_msg):
        p = copy.deepcopy(pose_msg)

        # BƯỚC 1-3: Transform position
        pos_h = np.array([pose_msg.position.x, pose_msg.position.y,
                          pose_msg.position.z, 1.0])
        pos_r = self.world_alignment @ pos_h
        p.position.x = float(pos_r[0])
        p.position.y = float(pos_r[1])
        p.position.z = float(pos_r[2])

        # BƯỚC 4: Transform orientation
        q_vr      = [pose_msg.orientation.x, pose_msg.orientation.y,
                     pose_msg.orientation.z, pose_msg.orientation.w]
        rot_final = R.from_matrix(self.world_alignment[:3, :3]) * R.from_quat(q_vr)
        q_f = rot_final.as_quat()
        p.orientation.x = float(q_f[0]); p.orientation.y = float(q_f[1])
        p.orientation.z = float(q_f[2]); p.orientation.w = float(q_f[3])
        return p

    def apply_offset_hardcoded(self, pose_msg):
        p = copy.deepcopy(pose_msg)
        offsets = self.cfg['manipulation']['offsets_sim' if self.cfg.get('sim') == 'true' else 'offsets_real']
        vr_x = -pose_msg.position.x
        vr_y = -pose_msg.position.y
        vr_z =  pose_msg.position.z
        theta = np.deg2rad(float(offsets['rot_z']))
        p.position.x = vr_x * math.cos(theta) - vr_y * math.sin(theta) + float(offsets['trans_x'])
        p.position.y = vr_x * math.sin(theta) + vr_y * math.cos(theta) + float(offsets['trans_y'])
        p.position.z = vr_z + float(offsets['trans_z'])
        p.orientation.x = 1.0; p.orientation.y = 0.0
        p.orientation.z = 0.0; p.orientation.w = 0.0
        return p

    def apply_offset(self, pose_msg):
        return self.apply_world_alignment(pose_msg) if self.use_world_alignment \
               else self.apply_offset_hardcoded(pose_msg)

    def actual_pose_callback(self, msg):
        if self.control_mode == 'relative':
            self.robot_current_pose = np.array([msg.x, msg.y, msg.z])
            # Lưu orientation hiện tại của TCP (rotvec → quaternion)
            try:
                rotvec = np.array([msg.roll, msg.pitch, msg.yaw])
                self.robot_current_quat = R.from_rotvec(rotvec).as_quat()
            except Exception:
                self.robot_current_quat = TOP_DOWN_QUAT.copy()

    def pose_to_matrix(self, pose):
        T = np.eye(4)
        T[:3, :3] = R.from_quat([pose.orientation.x, pose.orientation.y,
                                  pose.orientation.z, pose.orientation.w]).as_matrix()
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    # ------------------------------------------------------------------
    def pose_callback(self, pose_msg):
        if self.control_mode == 'relative':
            tracker_matrix = self.pose_to_matrix(pose_msg.pose)
            if self.use_world_alignment:
                tracker_matrix = self.world_alignment @ tracker_matrix

            def workspace_check(pt):
                return self.is_point_inside_cuboid(pt, self._ws_center, self._ws_len)

            target_pos, target_quat, status, is_origin_cmd = self.rel_control.update(
                tracker_matrix, self.robot_current_pose,
                workspace_check=workspace_check,
                robot_current_quat=self.robot_current_quat)

            # Log: chỉ khi có sự kiện hoặc throttle 1Hz
            if is_origin_cmd:
                self.get_logger().info(f"Status: {status} | target={target_pos}")
            else:
                self.get_logger().info(f"Status: {status}", throttle_duration_sec=1.0)

            if target_pos is not None:
                marker_pose = Pose()
                marker_pose.position.x = float(target_pos[0])
                marker_pose.position.y = float(target_pos[1])
                marker_pose.position.z = float(target_pos[2])
                # Gán orientation thật từ tracker (hoặc top-down nếu vừa Home)
                marker_pose.orientation.x = float(target_quat[0])
                marker_pose.orientation.y = float(target_quat[1])
                marker_pose.orientation.z = float(target_quat[2])
                marker_pose.orientation.w = float(target_quat[3])

                if workspace_check(target_pos):
                    marker_color = [0.0, 1.0, 0.0]  # Green
                    if is_origin_cmd:
                        self.get_logger().info(f"🤖 [ORIGIN CMD] {target_pos}")
                        origin_cmd = Pose()
                        origin_cmd.position.x = float(target_pos[0])
                        origin_cmd.position.y = float(target_pos[1])
                        origin_cmd.position.z = float(target_pos[2])
                        # Origin cmd: giữ nguyên orientation TCP hiện tại
                        origin_cmd.orientation.x = float(target_quat[0])
                        origin_cmd.orientation.y = float(target_quat[1])
                        origin_cmd.orientation.z = float(target_quat[2])
                        origin_cmd.orientation.w = float(target_quat[3])
                        self.origin_cmd_pub.publish(origin_cmd)
                        self.robot_current_pose = target_pos
                else:
                    marker_color = [1.0, 0.0, 0.0]  # Red
            else:
                marker_pose = Pose()
                marker_pose.position.x = float(self.robot_current_pose[0])
                marker_pose.position.y = float(self.robot_current_pose[1])
                marker_pose.position.z = float(self.robot_current_pose[2])
                marker_pose.orientation.x = float(TOP_DOWN_QUAT[0])
                marker_pose.orientation.y = float(TOP_DOWN_QUAT[1])
                marker_pose.orientation.z = float(TOP_DOWN_QUAT[2])
                marker_pose.orientation.w = float(TOP_DOWN_QUAT[3])
                marker_color = [0.5, 0.5, 0.5]  # Gray

        else:
            # ABSOLUTE MODE
            processed = self.apply_offset(pose_msg.pose)
            marker_pose = processed
            pt = [marker_pose.position.x, marker_pose.position.y, marker_pose.position.z]
            marker_color = [0.0, 1.0, 0.0] if self.is_point_inside_cuboid(
                pt, self._ws_center, self._ws_len) else [1.0, 0.0, 0.0]

        # Publish — dùng chung 1 timestamp
        now = self.get_clock().now().to_msg()

        self.ee_marker.pose = marker_pose
        self.ee_marker.header.stamp = now
        self.ee_marker.header.frame_id = self.base_frame
        self.ee_marker.color.r = marker_color[0]
        self.ee_marker.color.g = marker_color[1]
        self.ee_marker.color.b = marker_color[2]
        self.ee_marker_pub.publish(self.ee_marker)

        control_msg = PoseStamped()
        control_msg.header.stamp = now
        control_msg.header.frame_id = self.base_frame
        control_msg.pose = marker_pose
        self.control_pub.publish(control_msg)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.base_frame
        t.child_frame_id = "marker"
        t.transform.translation.x = marker_pose.position.x
        t.transform.translation.y = marker_pose.position.y
        t.transform.translation.z = marker_pose.position.z
        t.transform.rotation = marker_pose.orientation
        self.tf_broadcaster.sendTransform(t)

        self.viz_counter += 1
        if self.viz_counter % 50 == 0:
            self.robot_ws_marker.header.stamp = now
            self.robot_ws_pub.publish(self.robot_ws_marker)
            if self.viz_counter > 10000:
                self.viz_counter = 0

    def destroy_node(self):
        if self.rel_control is not None:
            self.rel_control.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ViveMarkerPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
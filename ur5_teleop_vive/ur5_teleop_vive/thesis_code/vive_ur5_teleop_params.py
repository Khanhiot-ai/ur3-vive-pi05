#!/usr/bin/env python3
"""
Teleoperation với World Alignment Matrix
Thay thế hardcoded rotation bằng ma trận từ calibration
"""

import sys
import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
# from sensor_msgs.msg import Joy
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster, TransformListener, Buffer

# Custom message for robot pose
try:
    from ur5_teleop_vive.msg import Xyzrpy
except ImportError:
    Xyzrpy = None  # Fallback if message not available

# Keyboard Input for Relative Control — dùng evdev (chạy được trên Wayland)
# pynput không bắt được phím trên Wayland + map sai phím Home (KP7).
# → đọc thẳng device bàn phím qua evdev, bắt cả KEY_HOME lẫn KEY_KP7.
import threading as _threading
try:
    import evdev
    from evdev import ecodes as _ecodes
    HAS_EVDEV = True
except Exception:
    HAS_EVDEV = False

# Mã phím Home: KEY_HOME (102) HOẶC KEY_KP7 (71 — numpad 7, Home khi NumLock off)
_HOME_KEYCODES = set()
if HAS_EVDEV:
    _HOME_KEYCODES = {_ecodes.KEY_HOME, _ecodes.KEY_KP7}

def _find_keyboard_device():
    """Tìm device bàn phím thật (có nhiều phím chữ, không phải chuột)."""
    if not HAS_EVDEV:
        return None
    best = None
    for path in evdev.list_devices():
        try:
            d = evdev.InputDevice(path)
            caps = d.capabilities()
            keys = caps.get(_ecodes.EV_KEY, [])
            # Bàn phím thật: có phím chữ A và Home/KP7, nhiều phím
            if (_ecodes.KEY_A in keys and len(keys) > 80
                    and (_ecodes.KEY_HOME in keys or _ecodes.KEY_KP7 in keys)):
                # Ưu tiên "AT Translated" (bàn phím laptop) hơn chuột-kèm-phím
                if "mouse" not in d.name.lower():
                    return d
                best = best or d
        except Exception:
            continue
    return best

# ============================================
# Relative Control Class
# ============================================
class RelativeControlWithKeyboard:
    """
    Relative Control:
    - Home key (evdev): set origin — bắt KEY_HOME + KEY_KP7, chạy trên Wayland
    - /set_origin (từ record_all): set origin trực tiếp (dự phòng)
    """
    def __init__(self, logger):
        self.logger = logger
        
        # State
        self.tracker_origin = None
        self.robot_origin = None
        self.origin_set = False
        
        # Keyboard state
        self.home_pressed = False
        self.last_home_state = False
        
        # ── Đọc bàn phím qua evdev (Wayland-compatible) ──
        self.listener = None
        self._evdev_running = False
        kbd = _find_keyboard_device()
        if kbd is not None:
            self._evdev_running = True
            self._evdev_dev = kbd
            t = _threading.Thread(target=self._evdev_loop, daemon=True)
            t.start()
            self.logger.info(f"🎮 Relative Control — đọc Home từ: {kbd.name}")
            self.logger.info("   Bấm Home (hoặc numpad 7) để set origin")
        else:
            self.logger.warn("⚠️ Không tìm thấy bàn phím evdev — dùng /set_origin từ record_all")
        self.logger.info("   /set_origin: set origin từ record_all (dự phòng)")

    def _evdev_loop(self):
        """Đọc phím từ evdev, bắt Home (KEY_HOME / KEY_KP7)."""
        try:
            for ev in self._evdev_dev.read_loop():
                if not self._evdev_running:
                    break
                if ev.type == _ecodes.EV_KEY and ev.code in _HOME_KEYCODES:
                    if ev.value == 1:      # nhấn xuống
                        self.home_pressed = True
                    elif ev.value == 0:    # nhả
                        self.home_pressed = False
        except Exception as e:
            self.logger.warn(f"evdev loop lỗi: {e}")

    def _on_key_press(self, key):
        """(Giữ cho tương thích — không dùng với evdev)"""
        pass

    def _on_key_release(self, key):
        pass

    def update(self, tracker_pose_matrix, robot_current_pose, workspace_check=None):
        """
        Update mỗi frame
        
        Args:
            tracker_pose_matrix: 4x4 matrix (đã qua World Alignment)
            robot_current_pose: np.array [x, y, z]
            workspace_check: Function(point) -> bool
        
        Returns:
            (target_pose, status_msg, is_origin_command)
        """
        # Phát hiện nhấn Home (edge detection)
        home_just_pressed = self.home_pressed and not self.last_home_state
        self.last_home_state = self.home_pressed

        # Nếu nhấn Home → Set origin
        if home_just_pressed:
            target = self._set_origin(
                tracker_pose_matrix, 
                robot_current_pose,
                workspace_check=workspace_check
            )
            if target is not None:
                return target, "Origin set - Moving robot", True
            else:
                return None, "Origin set - Robot stays", False
        
        # Nếu chưa set origin
        if not self.origin_set:
            return None, "Press Home to set origin", False
        
        # Tính target (luôn active sau khi set origin)
        target = self._calculate_target(tracker_pose_matrix)
        return target, "Active", False
    
    def _set_origin(self, tracker_pose_matrix, robot_current_pose, workspace_check=None):
        """Set origin position"""
        tracker_pos = tracker_pose_matrix[:3, 3].copy()
        
        # Kiểm tra workspace
        if workspace_check is not None:
            is_safe = workspace_check(tracker_pos)
        else:
            is_safe = True
        
        if is_safe:
            # AN TOÀN → Di chuyển robot đến vị trí tracker
            self.tracker_origin = tracker_pos
            self.robot_origin = tracker_pos
            self.origin_set = True
            
            self.logger.info("📍 [Home] Origin set + Robot moving to tracker:")
            self.logger.info(f"   >>> TRACKER COORDS: X={tracker_pos[0]:.4f}, Y={tracker_pos[1]:.4f}, Z={tracker_pos[2]:.4f}")
            self.logger.info(f"   Target: {tracker_pos}")
            
            return tracker_pos
        else:
            # NGOÀI WORKSPACE → Chỉ lưu origin
            self.tracker_origin = tracker_pos
            self.robot_origin = robot_current_pose.copy()
            self.origin_set = True
            
            self.logger.warn("⚠️ [Home] Tracker outside workspace!")
            self.logger.info(f"   >>> TRACKER (UNSAFE): X={tracker_pos[0]:.4f}, Y={tracker_pos[1]:.4f}, Z={tracker_pos[2]:.4f}")
            self.logger.info("📍 Origin set (robot stays at current position)")
            
            return None
    
    def _calculate_target(self, tracker_current_matrix):
        """Tính vị trí target từ delta"""
        tracker_current = tracker_current_matrix[:3, 3]
        delta = tracker_current - self.tracker_origin
        target = self.robot_origin + delta
        return target
    
    def reset_origin(self):
        """Reset origin"""
        self.origin_set = False
        self.logger.info("🔄 Origin reset")
    
    def shutdown(self):
        """Cleanup"""
        self._evdev_running = False
        if self.listener is not None:
            self.listener.stop()

# -------------------------------------------------------

class ViveMarkerPub(Node):

    def __init__(self):
        super().__init__('vive_ur5_teleop')
        self._declare_parameters()
        self.cfg = self._load_config()
        
        self.base_frame = self.cfg.get('frames', {}).get('base', 'base')
        self.ee_frame = self.cfg.get('frames', {}).get('ee', 'tool0')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # self.gripper_state = 'close'
        # self.last_trigger_state = 0

        # ✅ THÊM: Load World Alignment Matrix
        self.world_alignment = None
        self.use_world_alignment = self.load_world_alignment()

        # ✅ THÊM: Control Mode (absolute hoặc relative)
        self.control_mode = self.cfg.get('control_mode', 'relative')
        
        # ✅ THÊM: Relative Control (nếu cần)
        if self.control_mode == 'relative':
            self.rel_control = RelativeControlWithKeyboard(self.get_logger())
            # Robot current pose (sẽ được cập nhật từ /ur_actual_pose)
            self.robot_current_pose = np.array([0.3, 0.2, 0.4])  # Giá trị khởi tạo
            self._latest_tracker_matrix = None   # matrix tracker mới nhất (cho /set_origin)

            # Subscribe actual pose từ Node 4
            if Xyzrpy is not None:
                self.actual_pose_sub = self.create_subscription(
                    Xyzrpy,
                    '/ur_actual_pose',
                    self.actual_pose_callback,
                    10
                )
                self.get_logger().info("✅ Subscribed to /ur_actual_pose")
            else:
                self.get_logger().warn("⚠️ Xyzrpy message not available, using default pose")
            
            # Publisher cho lệnh di chuyển đến origin
            self.origin_cmd_pub = self.create_publisher(Pose, '/robot_origin_cmd', 10)
            self.get_logger().info("✅ Publisher /robot_origin_cmd created")

            # ✅ [FIX Wayland] Subscriber /set_origin — set origin TRỰC TIẾP từ
            #    tracker matrix mới nhất (KHÔNG cần pynput, chạy được trên Wayland)
            from std_msgs.msg import Bool as _Bool
            self.set_origin_sub = self.create_subscription(
                _Bool, '/set_origin', self._set_origin_direct_cb, 10)
            self.get_logger().info("✅ Subscribed to /set_origin (set origin trực tiếp, không cần Home tay)")
        else:
            self.rel_control = None
            self.robot_current_pose = None
            self.origin_cmd_pub = None

        self.ee_marker_pub = self.create_publisher(Marker, 'vive_gripper', 10)
        self.control_pub = self.create_publisher(PoseStamped, '/ur_target_pose', rclpy.qos.qos_profile_sensor_data)
        self.robot_ws_pub = self.create_publisher(Marker, '/robot_ws_marker', 10)

        self.setup_markers()

        # 1. Cấu hình QoS Best Effort (Ưu tiên tốc độ, chấp nhận mất gói tin cũ)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. Subscribe trực tiếp (Không qua Synchronizer, Không chờ Joy)
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            '/right_controller_as_posestamped', 
            self.pose_callback,  # Lưu ý: Tên hàm callback đổi từ synced_callback -> pose_callback
            qos_best_effort
        )

        # 3. Biến đếm để giảm tải hiển thị Workspace (Downsampling)
        self.viz_counter = 0

        alignment_mode = "WORLD ALIGNMENT" if self.use_world_alignment else "HARDCODED ROTATION"
        control_mode_str = f" + {self.control_mode.upper()} MODE"
        self.get_logger().info(f"######### INIT done - {alignment_mode}{control_mode_str} #########")
        self.get_logger().info(f"Base frame: {self.base_frame}")

    def load_world_alignment(self, filepath="world_alignment_matrix.txt"):
        """
        Load World Alignment Matrix từ file
        
        Returns:
            bool: True nếu load thành công, False nếu fallback về hardcoded
        """
        try:
            self.world_alignment = np.loadtxt(filepath)
            
            if self.world_alignment.shape != (4, 4):
                self.get_logger().error(f"❌ Invalid matrix shape: {self.world_alignment.shape}")
                return False
            
            self.get_logger().info(f"✅ Loaded World Alignment Matrix from {filepath}")
            self.get_logger().info(f"Matrix:\n{self.world_alignment}")
            return True
            
        except FileNotFoundError:
            self.get_logger().warn(f"⚠️ File not found: {filepath}")
            self.get_logger().warn("   Falling back to hardcoded rotation")
            return False
        except Exception as e:
            self.get_logger().error(f"❌ Error loading matrix: {e}")
            return False

    def _declare_parameters(self):
        # Giá trị offset của Robot thật
        self.declare_parameter('manipulation.offsets_real.trans_x', 0.60)
        self.declare_parameter('manipulation.offsets_real.trans_y', 1.0)
        self.declare_parameter('manipulation.offsets_real.trans_z', 0.848)
        self.declare_parameter('manipulation.offsets_real.rot_x', 0.0)
        self.declare_parameter('manipulation.offsets_real.rot_y', 0.0)
        self.declare_parameter('manipulation.offsets_real.rot_z', 0.0)  # Fallback

        # Giá trị offset của mô phỏng
        self.declare_parameter('manipulation.offsets_sim.trans_x', -0.2)
        self.declare_parameter('manipulation.offsets_sim.trans_y', 2.5)
        self.declare_parameter('manipulation.offsets_sim.trans_z', 1.65)
        self.declare_parameter('manipulation.offsets_sim.rot_x', 0.0)
        self.declare_parameter('manipulation.offsets_sim.rot_y', 0.0)
        self.declare_parameter('manipulation.offsets_sim.rot_z', 95.5)  # Fallback
        
        # Workspace Cuboid
        self.declare_parameter('robot_ws.x_mid', -0.32)
        self.declare_parameter('robot_ws.x_len', 0.4)
        self.declare_parameter('robot_ws.y_mid', -0.005)
        self.declare_parameter('robot_ws.y_len', 0.4)
        self.declare_parameter('robot_ws.z_mid', 0.31)
        self.declare_parameter('robot_ws.z_len', 0.3)
        
        # Frames
        self.declare_parameter('frames.base', 'base')
        self.declare_parameter('frames.ee', 'tool0')
        self.declare_parameter('sim', 'true')
        
        # Control Mode Chọn chế độ tại đây
        self.declare_parameter('control_mode', 'relative')  # 'absolute' hoặc 'relative'

    def _load_config(self):
        config = {
            'manipulation': {
                'offsets_real': {
                    'trans_x': self.get_parameter('manipulation.offsets_real.trans_x').value,
                    'trans_y': self.get_parameter('manipulation.offsets_real.trans_y').value,
                    'trans_z': self.get_parameter('manipulation.offsets_real.trans_z').value,
                    'rot_x': self.get_parameter('manipulation.offsets_real.rot_x').value,
                    'rot_y': self.get_parameter('manipulation.offsets_real.rot_y').value,
                    'rot_z': self.get_parameter('manipulation.offsets_real.rot_z').value,
                },
                'offsets_sim': {
                    'trans_x': self.get_parameter('manipulation.offsets_sim.trans_x').value,
                    'trans_y': self.get_parameter('manipulation.offsets_sim.trans_y').value,
                    'trans_z': self.get_parameter('manipulation.offsets_sim.trans_z').value,
                    'rot_x': self.get_parameter('manipulation.offsets_sim.rot_x').value,
                    'rot_y': self.get_parameter('manipulation.offsets_sim.rot_y').value,
                    'rot_z': self.get_parameter('manipulation.offsets_sim.rot_z').value,
                }
            },
            'robot_ws': {
                'x_mid': self.get_parameter('robot_ws.x_mid').value,
                'x_len': self.get_parameter('robot_ws.x_len').value,
                'y_mid': self.get_parameter('robot_ws.y_mid').value,
                'y_len': self.get_parameter('robot_ws.y_len').value,
                'z_mid': self.get_parameter('robot_ws.z_mid').value,
                'z_len': self.get_parameter('robot_ws.z_len').value,
            },
            'frames': {
                'base': self.get_parameter('frames.base').value,
                'ee': self.get_parameter('frames.ee').value,
            },
            'sim': self.get_parameter('sim').value,
            'control_mode': self.get_parameter('control_mode').value
        }
        return config

    def setup_markers(self):
        mesh_path = "file:///home/xuanlinh/Desktop/Sample/UR5_VR_teleop-master/ur5_teleop_vive/mesh/hand.dae"
        self.ee_marker = self.create_ee_marker(mesh_path, [0.0, 1.0, 0.0, 0.6])
        
        self.robot_ws_marker = Marker()
        self.robot_ws_marker.header.frame_id = self.base_frame
        self.robot_ws_marker.type = Marker.CUBE
        self.robot_ws_marker.action = Marker.ADD
        self.robot_ws_marker.pose.orientation.w = 1.0
        self.robot_ws_marker.color.r = 0.0; self.robot_ws_marker.color.g = 1.0; self.robot_ws_marker.color.b = 0.0; self.robot_ws_marker.color.a = 0.3
        
        ws_cfg = self.cfg['robot_ws']
        self.robot_ws_marker.pose.position.x = float(ws_cfg['x_mid'])
        self.robot_ws_marker.pose.position.y = float(ws_cfg['y_mid'])
        self.robot_ws_marker.pose.position.z = float(ws_cfg['z_mid'])
        self.robot_ws_marker.scale.x = float(ws_cfg['x_len'])
        self.robot_ws_marker.scale.y = float(ws_cfg['y_len'])
        self.robot_ws_marker.scale.z = float(ws_cfg['z_len'])

    def create_ee_marker(self, mesh_resource, color):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = mesh_resource
        marker.action = Marker.ADD
        marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0
        marker.color.r = color[0]; marker.color.g = color[1]; marker.color.b = color[2]; marker.color.a = color[3]
        return marker

    def apply_world_alignment(self, pose_msg):
        """
        Áp dụng World Alignment Matrix
        - Chỉ xoay trục Z để đồng bộ hướng
        - KHÔNG có offset translation trong ma trận
        """
        p = copy.deepcopy(pose_msg)
        
        # Lấy offset translation
        if self.cfg.get('sim') == 'true':
            offsets = self.cfg['manipulation']['offsets_sim']
        else:
            offsets = self.cfg['manipulation']['offsets_real']

        # --- BƯỚC 1: Lấy vị trí VR (đảo trục) ---
        vr_pos = np.array([
            -pose_msg.position.x,  # Đảo X
            -pose_msg.position.y,  # Đảo Y
            pose_msg.position.z,   # Giữ nguyên Z
            1.0  # Homogeneous coordinate
        ])

        # --- BƯỚC 2: Áp dụng World Alignment Matrix ---
        aligned_pos = self.world_alignment @ vr_pos

        # --- BƯỚC 3: Cộng offset translation ---
        p.position.x = aligned_pos[0] + float(offsets['trans_x'])
        p.position.y = aligned_pos[1] + float(offsets['trans_y'])
        p.position.z = aligned_pos[2] + float(offsets['trans_z'])

        # --- BƯỚC 4: Khóa orientation (top-down) ---
        p.orientation.x = 1.0
        p.orientation.y = 0.0
        p.orientation.z = 0.0
        p.orientation.w = 0.0

        return p

    #Hàm offset cứng
    def apply_offset_hardcoded(self, pose_msg):
        """
        Fallback: Hardcoded rotation matrix (CŨ)
        Dùng khi không có world_alignment_matrix.txt
        """
        p = copy.deepcopy(pose_msg)
        
        # Lấy offset từ config
        if self.cfg.get('sim') == 'true':
            offsets = self.cfg['manipulation']['offsets_sim']
        else:
            offsets = self.cfg['manipulation']['offsets_real']

        # --- BƯỚC 1: Lấy tọa độ thô từ VR (Có đảo trục để khớp hướng chung) ---
        vr_x = -pose_msg.position.x
        vr_y = -pose_msg.position.y
        vr_z = pose_msg.position.z

        # --- BƯỚC 2: Tính toán Xoay (Rotation Matrix) ---
        # Lấy góc xoay rot_z (độ) đổi sang radian
        theta = np.deg2rad(float(offsets['rot_z']))

        # Công thức xoay vector 2D:
        # x' = x*cos(theta) - y*sin(theta)
        # y' = x*sin(theta) + y*cos(theta)
        rotated_x = vr_x * math.cos(theta) - vr_y * math.sin(theta)
        rotated_y = vr_x * math.sin(theta) + vr_y * math.cos(theta)

        # --- BƯỚC 3: Cộng Offset vị trí (Translation) ---
        p.position.x = rotated_x + float(offsets['trans_x'])
        p.position.y = rotated_y + float(offsets['trans_y'])
        p.position.z = vr_z + float(offsets['trans_z'])

        # --- BƯỚC 4: Khóa hướng tay gắp (Top-down) ---
        p.orientation.x = 1.0
        p.orientation.y = 0.0
        p.orientation.z = 0.0
        p.orientation.w = 0.0

        return p

    def apply_offset(self, pose_msg):
        """
        Wrapper: Tự động chọn World Alignment hoặc Hardcoded
        """
        if self.use_world_alignment:
            return self.apply_world_alignment(pose_msg)
        else:
            return self.apply_offset_hardcoded(pose_msg)

    def actual_pose_callback(self, msg):
        """Nhận vị trí thật từ Node 4 (/ur_actual_pose)"""
        if self.control_mode == 'relative':
            self.robot_current_pose = np.array([msg.x, msg.y, msg.z])

    def _set_origin_direct_cb(self, msg):
        """
        [FIX Wayland] record_all gửi /set_origin=True → set origin NGAY từ
        tracker matrix mới nhất. Không qua pynput/edge-detection nên chạy
        được trên Wayland (bấm Home không cần nữa).
        """
        if not msg.data:
            return
        if self._latest_tracker_matrix is None:
            self.get_logger().warn(
                "⚠️ [/set_origin] Chưa có tracker pose! Tracker được track chưa?")
            return

        # Workspace check
        ws_cfg = self.cfg['robot_ws']
        ws_center = [float(ws_cfg['x_mid']), float(ws_cfg['y_mid']), float(ws_cfg['z_mid'])]
        ws_len    = [float(ws_cfg['x_len']), float(ws_cfg['y_len']), float(ws_cfg['z_len'])]
        def workspace_check(point):
            return self.is_point_inside_cuboid(point, ws_center, ws_len)

        # Set origin trực tiếp (gọi thẳng _set_origin, không qua update/edge)
        target = self.rel_control._set_origin(
            self._latest_tracker_matrix,
            self.robot_current_pose,
            workspace_check=workspace_check)

        self.get_logger().info("🏠 [/set_origin] Origin set TRỰC TIẾP (không cần Home tay)")

        # Nếu trong workspace → publish lệnh robot di chuyển đến tracker
        if target is not None and workspace_check(target):
            origin_cmd = Pose()
            origin_cmd.position.x = float(target[0])
            origin_cmd.position.y = float(target[1])
            origin_cmd.position.z = float(target[2])
            origin_cmd.orientation.x = 1.0
            origin_cmd.orientation.y = 0.0
            origin_cmd.orientation.z = 0.0
            origin_cmd.orientation.w = 0.0
            self.origin_cmd_pub.publish(origin_cmd)
            self.robot_current_pose = target
            self.get_logger().info(f"🤖 [ORIGIN CMD] Robot di chuyển đến: {target.round(3).tolist()}")

    def pose_to_matrix(self, pose):
        """Chuyển Pose → Matrix 4x4"""
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        rot_matrix = R.from_quat(q).as_matrix()  
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = t
        return T

    def is_point_inside_cuboid(self, point, center, lengths):
        x, y, z = point
        x_center, y_center, z_center = center
        length_x, length_y, length_z = lengths
        return (abs(x - x_center) <= length_x / 2) and \
               (abs(y - y_center) <= length_y / 2) and \
               (abs(z - z_center) <= length_z / 2)

    def pose_callback(self, pose_msg):
        # 2. Chọn chế độ điều khiển
        if self.control_mode == 'relative':
            # ========== RELATIVE MODE ==========
            # Chuyển pose → matrix
            tracker_pose_matrix = self.pose_to_matrix(pose_msg.pose)
            
            # Áp dụng World Alignment (nếu có)
            if self.use_world_alignment:
                tracker_pose_matrix = self.world_alignment @ tracker_pose_matrix

            # Lưu matrix mới nhất (cho /set_origin set trực tiếp, không cần pynput)
            self._latest_tracker_matrix = tracker_pose_matrix
            
            # Tạo workspace check function
            ws_cfg = self.cfg['robot_ws']
            ws_center = [float(ws_cfg['x_mid']), float(ws_cfg['y_mid']), float(ws_cfg['z_mid'])]
            ws_len = [float(ws_cfg['x_len']), float(ws_cfg['y_len']), float(ws_cfg['z_len'])]
            
            def workspace_check(point):
                return self.is_point_inside_cuboid(point, ws_center, ws_len)
            
            # Update relative control
            target, status, is_origin_cmd = self.rel_control.update(
                tracker_pose_matrix,
                self.robot_current_pose,
                workspace_check=workspace_check
            )
            
            # Xử lý target
            if target is not None:
                # Tạo marker pose
                marker_pose = Pose()
                marker_pose.position.x = float(target[0])
                marker_pose.position.y = float(target[1])
                marker_pose.position.z = float(target[2])
                marker_pose.orientation.x = 1.0
                marker_pose.orientation.y = 0.0
                marker_pose.orientation.z = 0.0
                marker_pose.orientation.w = 0.0
                
                # Kiểm tra workspace
                if workspace_check(target):
                    marker_color = [0.0, 1.0, 0.0]  # Green
                    
                    # Nếu là lệnh set origin → Publish lệnh di chuyển
                    if is_origin_cmd:
                        self.get_logger().info(f"🤖 [ORIGIN CMD] Publishing move command: {target}")
                        
                        # Tạo Pose message cho Node 4
                        origin_cmd = Pose()
                        origin_cmd.position.x = float(target[0])
                        origin_cmd.position.y = float(target[1])
                        origin_cmd.position.z = float(target[2])
                        origin_cmd.orientation.x = 1.0
                        origin_cmd.orientation.y = 0.0
                        origin_cmd.orientation.z = 0.0
                        origin_cmd.orientation.w = 0.0
                        
                        # Publish lệnh
                        self.origin_cmd_pub.publish(origin_cmd)
                        
                        # Cập nhật robot_current_pose (sẽ được cập nhật lại từ /ur_actual_pose)
                        self.robot_current_pose = target
                else:
                    marker_color = [1.0, 0.0, 0.0]  # Red
            else:
                # Không có target → Hiển thị vị trí hiện tại
                marker_pose = Pose()
                marker_pose.position.x = float(self.robot_current_pose[0])
                marker_pose.position.y = float(self.robot_current_pose[1])
                marker_pose.position.z = float(self.robot_current_pose[2])
                marker_pose.orientation.x = 1.0
                marker_pose.orientation.y = 0.0
                marker_pose.orientation.z = 0.0
                marker_pose.orientation.w = 0.0
                marker_color = [0.5, 0.5, 0.5]  # Gray
        
        else:
            # ========== ABSOLUTE MODE ==========
            # Process Pose (dùng code cũ)
            processed_pose = self.apply_offset(pose_msg.pose)
            marker_pose = processed_pose
            
            # Check Workspace
            point = [marker_pose.position.x, marker_pose.position.y, marker_pose.position.z]
            ws_cfg = self.cfg['robot_ws']
            ws_center = [float(ws_cfg['x_mid']), float(ws_cfg['y_mid']), float(ws_cfg['z_mid'])]
            ws_len = [float(ws_cfg['x_len']), float(ws_cfg['y_len']), float(ws_cfg['z_len'])]

            if self.is_point_inside_cuboid(point, ws_center, ws_len):
                marker_color = [0.0, 1.0, 0.0]  # Green
            else:
                marker_color = [1.0, 0.0, 0.0]  # Red
        
        # 3. Update Marker (chung cho cả 2 chế độ)
        self.ee_marker.pose = marker_pose
        self.ee_marker.header.stamp = self.get_clock().now().to_msg()
        self.ee_marker.header.frame_id = self.base_frame
        self.ee_marker.color.r = marker_color[0]
        self.ee_marker.color.g = marker_color[1]
        self.ee_marker.color.b = marker_color[2]

        # 4. Publish (ĐÃ TỐI ƯU)
        
        # A. Hand Marker: Gửi luôn (Real-time) để nhìn mượt
        self.ee_marker.header.stamp = self.get_clock().now().to_msg()
        self.ee_marker_pub.publish(self.ee_marker)

        control_msg = PoseStamped()
        control_msg.header.stamp = self.get_clock().now().to_msg()
        control_msg.header.frame_id = self.base_frame
        control_msg.pose = marker_pose  # Sử dụng lại biến marker_pose đã tính toán ở trên
        self.control_pub.publish(control_msg)
        
        # TF cũng gửi luôn (đi kèm Hand)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = "marker"
        t.transform.translation.x = self.ee_marker.pose.position.x
        t.transform.translation.y = self.ee_marker.pose.position.y
        t.transform.translation.z = self.ee_marker.pose.position.z
        t.transform.rotation = self.ee_marker.pose.orientation
        self.tf_broadcaster.sendTransform(t)

        # B. Workspace Marker: Chỉ gửi 1 lần mỗi 50 frame (Giảm tải băng thông)
        self.viz_counter += 1
        if self.viz_counter % 50 == 0:
            self.robot_ws_marker.header.stamp = self.get_clock().now().to_msg()
            self.robot_ws_pub.publish(self.robot_ws_marker)
            # Reset counter để tránh số quá lớn
            if self.viz_counter > 10000: self.viz_counter = 0

    def destroy_node(self):
        """Cleanup khi tắt node"""
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
#!/usr/bin/env python3
"""
HTC Vive Tracker to ROS2 Bridge

Publishes tracker pose as TF and Joy messages for robot teleoperation.
Uses keyboard SPACE key to simulate trigger button.
"""

import time
import openvr
from math import sqrt, copysign

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Joy
from tf2_ros import TransformBroadcaster
from pynput import keyboard

# --- TOGGLE MODE: tap Ctrl_R bật/tắt thay vì giữ ---
# Tap lần 1 → ON  (buttons[0]=1)
# Tap lần 2 → OFF (buttons[0]=0)
is_space_pressed = False
_last_key_state = False

def on_press(key):
    global is_space_pressed, _last_key_state
    if key == keyboard.Key.ctrl_r:
        # Edge detection: chỉ toggle khi vừa nhấn xuống
        if not _last_key_state:
            is_space_pressed = not is_space_pressed
            state = "ON 🟢" if is_space_pressed else "OFF 🔴"
            print(f"\n  [TOGGLE] Trigger {state}\n")
        _last_key_state = True

def on_release(key):
    global _last_key_state
    if key == keyboard.Key.ctrl_r:
        _last_key_state = False


# --- CÁC HÀM HỖ TRỢ ---
def get_lighthouse_ids(vrsys):
    lighthouse_ids = []
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vrsys.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_TrackingReference:
            lighthouse_ids.append(i)
    return lighthouse_ids

def get_generic_tracker_ids(vrsys):
    generic_tracker_ids = []
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vrsys.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker:
            generic_tracker_ids.append(i)
    return generic_tracker_ids

def from_matrix_to_transform(matrix, stamp, frame_id, child_frame_id):
    """Convert OpenVR 3x4 matrix to ROS2 TransformStamped"""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    
    # Extract translation
    t.transform.translation.x = matrix[0][3]
    t.transform.translation.y = matrix[1][3]
    t.transform.translation.z = matrix[2][3]
    
    # Extract quaternion from rotation matrix
    w = sqrt(max(0, 1 + matrix[0][0] + matrix[1][1] + matrix[2][2])) / 2.0
    x = sqrt(max(0, 1 + matrix[0][0] - matrix[1][1] - matrix[2][2])) / 2.0
    y = sqrt(max(0, 1 - matrix[0][0] + matrix[1][1] - matrix[2][2])) / 2.0
    z = sqrt(max(0, 1 - matrix[0][0] - matrix[1][1] + matrix[2][2])) / 2.0
    
    x = copysign(x, matrix[2][1] - matrix[1][2])
    y = copysign(y, matrix[0][2] - matrix[2][0])
    z = copysign(z, matrix[1][0] - matrix[0][1])

    t.transform.rotation.w = w
    t.transform.rotation.x = x
    t.transform.rotation.y = y
    t.transform.rotation.z = z

    # Convert from SteamVR (Y-up) to ROS (Z-up) coordinate system
    tr = t.transform.translation
    rot = t.transform.rotation
    tr.z, tr.y, tr.x = tr.y, -tr.x, -tr.z
    rot.z, rot.y, rot.x = rot.y, -rot.x, -rot.z
    
    return t


class ViveTrackerNode(Node):
    def __init__(self, vrsystem, tracker_id, lighthouse_ids):
        super().__init__('htc_vive_ros2')
        
        self.vrsystem = vrsystem
        self.tracker_id = tracker_id
        self.lighthouse_ids = lighthouse_ids
        
        # Setup poses array
        self.poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        self.poses = self.poses_t()
        
        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Joy Publisher (simulating right controller)
        self.joy_pub = self.create_publisher(Joy, '/vive_right', 10)
        
        # Timer for main loop (90 Hz)
        self.timer = self.create_timer(1.0 / 90.0, self.timer_callback)
        
        self.get_logger().info('Vive Tracker Node Started!')
        self.get_logger().info('--> TAP [Ctrl_R] TO TOGGLE TRIGGER ON/OFF')
        self.get_logger().info('    🟢 ON  = Robot bám tracker')
        self.get_logger().info('    🔴 OFF = Robot đứng yên')
        self.get_logger().info(f'--> Tracker ID {tracker_id} published as "right_controller"')

    def timer_callback(self):
        global is_space_pressed
        
        # 1. Get poses from OpenVR
        self.vrsystem.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, self.poses
        )
        now = self.get_clock().now().to_msg()
        transforms = []

        # 2. Process main tracker (as right_controller)
        if self.tracker_id is not None and self.poses[self.tracker_id].bPoseIsValid:
            matrix = self.poses[self.tracker_id].mDeviceToAbsoluteTracking
            
            # Send TF
            tf_msg = from_matrix_to_transform(matrix, now, "world", "right_controller")
            transforms.append(tf_msg)
            
            # Send Joy Message (based on SPACE key)
            joy_msg = Joy()
            joy_msg.header.stamp = now
            joy_msg.header.frame_id = "right_controller"
            
            # Configure Joy like HTC Vive controller
            # Buttons: [Trigger, Menu, Grip, ...]
            # Axes: [Trigger_Val, Trackpad_X, Trackpad_Y]
            
            if is_space_pressed:
                joy_msg.buttons = [1, 0, 0, 0]  # Button 0 (Trigger) = 1
                joy_msg.axes = [1.0, 0.0, 0.0]  # Axis 0 (Trigger) = 1.0
            else:
                joy_msg.buttons = [0, 0, 0, 0]
                joy_msg.axes = [0.0, 0.0, 0.0]
            
            self.joy_pub.publish(joy_msg)

        # 3. Process other devices (HMD, Lighthouse...)
        if self.poses[0].bPoseIsValid:
            hmd_tf = from_matrix_to_transform(
                self.poses[0].mDeviceToAbsoluteTracking, now, "world", "hmd"
            )
            transforms.append(hmd_tf)
            
        for idx, _id in enumerate(self.lighthouse_ids):
            if self.poses[_id].bPoseIsValid:
                lh_tf = from_matrix_to_transform(
                    self.poses[_id].mDeviceToAbsoluteTracking, 
                    now, "world", f"lighthouse_{idx}"
                )
                transforms.append(lh_tf)

        # Send all TFs
        if transforms:
            for tf in transforms:
                self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    print("===========================")
    print("Initializing OpenVR...")
    
    # Initialize OpenVR
    try:
        openvr.init(openvr.VRApplication_Other)
        vrsystem = openvr.VRSystem()
    except openvr.OpenVRError as e:
        print(f"Error: {e}")
        return

    # Wait for Tracker
    tracker_id = None
    print("Waiting for tracker...")
    try:
        while tracker_id is None:
            ids = get_generic_tracker_ids(vrsystem)
            if ids:
                tracker_id = ids[0]
            time.sleep(0.5)
    except KeyboardInterrupt:
        openvr.shutdown()
        return

    print(f"Tracker Found ID: {tracker_id}")

    # Get other device IDs
    lighthouse_ids = get_lighthouse_ids(vrsystem)
    print(f"Lighthouse IDs: {lighthouse_ids}")

    # Start keyboard listener (runs in separate thread)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        node = ViveTrackerNode(vrsystem, tracker_id, lighthouse_ids)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        listener.stop()
        openvr.shutdown()
        rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()
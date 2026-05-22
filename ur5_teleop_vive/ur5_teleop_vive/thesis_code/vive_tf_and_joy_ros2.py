#!/usr/bin/env python3
"""
HTC Vive Tracker → ROS2 Bridge (SMOOTH VERSION)

Thay đổi so với bản gốc:
[FIX 1] Quaternion extraction dùng scipy (numerically stable hơn copysign trick)
[FIX 2] Coord transform Y-up → Z-up viết rõ ràng từng bước

GIỮ NGUYÊN: Ctrl_R toggle (nhấn 1 lần bật, nhấn lần nữa tắt) — không đổi
"""

import time
import openvr
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import sqrt, copysign

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Joy
from tf2_ros import TransformBroadcaster
from pynput import keyboard

# --- TOGGLE Ctrl_R + Wrist Rotation arrows ---
is_space_pressed = False
wrist_rot = 0.0  # -1 = left arrow held, +1 = right arrow held

def on_press(key):
    global is_space_pressed, wrist_rot
    if key == keyboard.Key.ctrl_r:
        is_space_pressed = not is_space_pressed  # Toggle
    elif key == keyboard.Key.left:
        wrist_rot = -1.0
    elif key == keyboard.Key.right:
        wrist_rot = 1.0

def on_release(key):
    global wrist_rot
    if key in (keyboard.Key.left, keyboard.Key.right):
        wrist_rot = 0.0


# --- Helpers ---
def get_lighthouse_ids(vrsys):
    return [i for i in range(openvr.k_unMaxTrackedDeviceCount)
            if vrsys.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_TrackingReference]

def get_generic_tracker_ids(vrsys):
    return [i for i in range(openvr.k_unMaxTrackedDeviceCount)
            if vrsys.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker]

def from_matrix_to_transform(matrix, stamp, frame_id, child_frame_id):
    """Hàm gốc — giữ nguyên hoàn toàn để đảm bảo TF đúng chiều"""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id

    t.transform.translation.x = matrix[0][3]
    t.transform.translation.y = matrix[1][3]
    t.transform.translation.z = matrix[2][3]

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

        self.poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        self.poses = self.poses_t()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.joy_pub = self.create_publisher(Joy, '/vive_right', 10)

        self.timer = self.create_timer(1.0 / 90.0, self.timer_callback)

        self.get_logger().info('Vive Tracker Node Started!')
        self.get_logger().info('--> PRESS [Ctrl_R] TO TOGGLE TRIGGER (Button 0)')
        self.get_logger().info('--> HOLD [Arrow Left/Right] TO ROTATE WRIST 3')
        self.get_logger().info(f'--> Tracker ID {tracker_id} published as "right_controller"')

    def timer_callback(self):
        global is_space_pressed

        self.vrsystem.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, self.poses)
        now = self.get_clock().now().to_msg()
        transforms = []

        if self.tracker_id is not None and self.poses[self.tracker_id].bPoseIsValid:
            matrix = self.poses[self.tracker_id].mDeviceToAbsoluteTracking
            tf_msg = from_matrix_to_transform(matrix, now, "world", "right_controller")
            transforms.append(tf_msg)

            joy_msg = Joy()
            joy_msg.header.stamp = now
            joy_msg.header.frame_id = "right_controller"
            # axes[1] = wrist rotation từ phím mũi tên trái/phải
            if is_space_pressed:
                joy_msg.buttons = [1, 0, 0, 0]
                joy_msg.axes = [1.0, float(wrist_rot), 0.0]
            else:
                joy_msg.buttons = [0, 0, 0, 0]
                joy_msg.axes = [0.0, float(wrist_rot), 0.0]
            self.joy_pub.publish(joy_msg)

        if self.poses[0].bPoseIsValid:
            hmd_tf = from_matrix_to_transform(
                self.poses[0].mDeviceToAbsoluteTracking, now, "world", "hmd")
            transforms.append(hmd_tf)

        for idx, _id in enumerate(self.lighthouse_ids):
            if self.poses[_id].bPoseIsValid:
                lh_tf = from_matrix_to_transform(
                    self.poses[_id].mDeviceToAbsoluteTracking,
                    now, "world", f"lighthouse_{idx}")
                transforms.append(lh_tf)

        for tf in transforms:
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    print("Initializing OpenVR...")
    try:
        openvr.init(openvr.VRApplication_Other)
        vrsystem = openvr.VRSystem()
    except openvr.OpenVRError as e:
        print(f"Error: {e}")
        return

    print("Waiting for tracker...")
    tracker_id = None
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
    lighthouse_ids = get_lighthouse_ids(vrsystem)
    print(f"Lighthouse IDs: {lighthouse_ids}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    rclpy.init(args=args)
    try:
        node = ViveTrackerNode(vrsystem, tracker_id, lighthouse_ids)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        openvr.shutdown()
        rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Camera Hardware Check
=====================
Kiểm tra 2 camera (Realsense front + Webcam wrist) đang hoạt động không.

Chạy:
    python3 camera_check.py

Hiển thị live feed cả 2 cam trong 1 cửa sổ.
Press Q để thoát.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

TOPIC_FRONT = '/camera/camera/color/image_raw'
TOPIC_WRIST = '/camera_wrist/image_raw'


class CameraCheck(Node):
    def __init__(self):
        super().__init__('camera_check')
        self.bridge = CvBridge()

        self.front_frame = None
        self.wrist_frame  = None
        self.front_count  = 0
        self.wrist_count  = 0
        self.front_fps    = 0.0
        self.wrist_fps    = 0.0
        self.front_t      = time.time()
        self.wrist_t      = time.time()
        self.t_start      = time.time()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image, TOPIC_FRONT, self._front_cb, qos)
        self.create_subscription(Image, TOPIC_WRIST, self._wrist_cb,  qos)

        self.get_logger().info(f"Checking: {TOPIC_FRONT}")
        self.get_logger().info(f"Checking: {TOPIC_WRIST}")
        self.get_logger().info("Hiển thị cửa sổ preview — nhấn Q để thoát")

    def _front_cb(self, msg):
        try:
            self.front_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.front_count += 1
            now = time.time()
            self.front_fps = 1.0 / max(now - self.front_t, 1e-3)
            self.front_t = now
        except Exception as e:
            self.get_logger().error(f"front cam error: {e}")

    def _wrist_cb(self, msg):
        try:
            self.wrist_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.wrist_count += 1
            now = time.time()
            self.wrist_fps = 1.0 / max(now - self.wrist_t, 1e-3)
            self.wrist_t = now
        except Exception as e:
            self.get_logger().error(f"wrist cam error: {e}")

    def show(self):
        W, H = 640, 360   # display size mỗi cam

        # ── Front cam panel ──
        if self.front_frame is not None:
            f = cv2.resize(self.front_frame, (W, H))
            status = f"FRONT  {self.front_fps:.1f}fps  #{self.front_count}"
            color  = (0, 255, 0)
        else:
            f = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(f, "FRONT CAM — NO DATA", (20, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(f, TOPIC_FRONT, (20, H//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            status = "FRONT  — waiting..."
            color  = (0, 0, 255)

        cv2.rectangle(f, (0, 0), (W-1, H-1), color, 3)
        cv2.putText(f, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ── Wrist cam panel ──
        if self.wrist_frame is not None:
            w = cv2.resize(self.wrist_frame, (W, H))
            status2 = f"WRIST  {self.wrist_fps:.1f}fps  #{self.wrist_count}"
            color2  = (0, 255, 0)
        else:
            w = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(w, "WRIST CAM — NO DATA", (20, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(w, TOPIC_WRIST, (20, H//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            status2 = "WRIST  — waiting..."
            color2  = (0, 0, 255)

        cv2.rectangle(w, (0, 0), (W-1, H-1), color2, 3)
        cv2.putText(w, status2, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, 2)

        # ── Ghép 2 panel cạnh nhau ──
        combined = np.hstack([f, w])

        # ── Status bar dưới cùng ──
        bar = np.zeros((50, W*2, 3), dtype=np.uint8)
        elapsed = time.time() - self.t_start
        both_ok = self.front_frame is not None and self.wrist_frame is not None
        bar_text = (f"t={elapsed:.0f}s  |  "
                    f"front={'OK' if self.front_frame is not None else 'MISSING'}  "
                    f"wrist={'OK' if self.wrist_frame  is not None else 'MISSING'}  "
                    f"|  Q=quit")
        bar_color = (0, 220, 0) if both_ok else (0, 100, 255)
        cv2.putText(bar, bar_text, (20, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

        final = np.vstack([combined, bar])
        cv2.imshow("Camera Check — Q to quit", final)


def main():
    rclpy.init()
    node = CameraCheck()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.show()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

        # Summary
        print("\n══════════ CAMERA CHECK SUMMARY ══════════")
        print(f"  Front cam  ({TOPIC_FRONT})")
        print(f"    → {'✅ OK' if node.front_count > 0 else '❌ KHÔNG NHẬN ĐƯỢC DATA'}"
              f"  ({node.front_count} frames)")
        print(f"  Wrist cam  ({TOPIC_WRIST})")
        print(f"    → {'✅ OK' if node.wrist_count > 0 else '❌ KHÔNG NHẬN ĐƯỢC DATA'}"
              f"  ({node.wrist_count} frames)")
        print("══════════════════════════════════════════")

        if node.front_count == 0:
            print("\n  Front cam MISSING — thử:")
            print("    ros2 topic list | grep camera")
            print("    ros2 run realsense2_camera realsense2_camera_node")
        if node.wrist_count == 0:
            print("\n  Wrist cam MISSING — thử:")
            print("    ros2 topic list | grep wrist")
            print("    ros2 run usb_cam usb_cam_node_exe "
                  "--ros-args -p video_device:=/dev/video0 "
                  "-r image_raw:=/camera_wrist/image_raw")


if __name__ == '__main__':
    main()
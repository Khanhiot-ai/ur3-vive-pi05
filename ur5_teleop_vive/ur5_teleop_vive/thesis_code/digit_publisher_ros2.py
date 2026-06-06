#!/usr/bin/env python3
"""
digit_publisher_ros2.py
=======================
Node ROS2 đọc 2 cảm biến DIGIT v1 và publish ảnh qua topic.

Topics output:
    /digit_left/image_raw    (sensor_msgs/Image, BGR8, 320x240, ~60Hz)
    /digit_right/image_raw   (sensor_msgs/Image, BGR8, 320x240, ~60Hz)

Cài trước:
    pip install digit-interface
    sudo chmod 666 /dev/video*   # nếu chưa cấp quyền

Chạy:
    python3 digit_publisher_ros2.py

Verify:
    ros2 topic hz /digit_left/image_raw    # ~60Hz
    ros2 topic hz /digit_right/image_raw   # ~60Hz
    ros2 run rqt_image_view rqt_image_view /digit_left/image_raw
"""

import threading
import time
import sys

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    from digit_interface import Digit
except ImportError:
    print("❌ Chưa cài digit-interface. Chạy: pip install digit-interface")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
#  CONFIG — gán serial cho từng vai trò
# ═══════════════════════════════════════════════════════════
SERIAL_LEFT  = "D21383"   # ngón trái
SERIAL_RIGHT = "D21384"   # ngón phải

TOPIC_LEFT   = "/digit_left/image_raw"
TOPIC_RIGHT  = "/digit_right/image_raw"

FRAME_LEFT   = "digit_left"
FRAME_RIGHT  = "digit_right"

# QVGA 320x240 @ 60fps (DIGIT v1)
PUBLISH_RATE_HZ = 60

# ══════════════════════════════════════════════════════════════════
# CHIỀU XOAY DIGIT → PORTRAIT
# DIGIT của bạn trả frame ĐÃ portrait sẵn (320 cao, 240 rộng) — KHÔNG
# cần xoay. Để None để giữ nguyên.
#
# Nếu DIGIT khác trả landscape (240,320) → đặt cv2.ROTATE_90_CLOCKWISE
# (hoặc COUNTERCLOCKWISE nếu lệch chiều) để xoay thành portrait.
# ══════════════════════════════════════════════════════════════════
ROTATE_DIR = None   # DIGIT đã portrait (320,240) sẵn → không xoay
# ══════════════════════════════════════════════════════════════════


class DigitPublisher(Node):
    def __init__(self):
        super().__init__("digit_publisher")
        self.bridge = CvBridge()

        # QoS RELIABLE (giống Realsense) để record_all nhận đúng
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Connect 2 DIGIT ──
        self.digit_left  = self._connect(SERIAL_LEFT,  "LEFT")
        self.digit_right = self._connect(SERIAL_RIGHT, "RIGHT")

        if self.digit_left is None and self.digit_right is None:
            self.get_logger().error("❌ Không kết nối được DIGIT nào. Thoát.")
            raise RuntimeError("No DIGIT connected")

        # ── Publishers ──
        self.pub_left  = self.create_publisher(Image, TOPIC_LEFT,  qos)
        self.pub_right = self.create_publisher(Image, TOPIC_RIGHT, qos)

        # ── Counter ──
        self.count_left  = 0
        self.count_right = 0
        self.last_left   = None   # frame mới nhất (cho GUI)
        self.last_right  = None
        self.t_start     = time.time()

        # ── Threads đọc 2 cam song song (mỗi cam 1 thread) ──
        self._running = True
        if self.digit_left:
            threading.Thread(target=self._loop_left,  daemon=True).start()
        if self.digit_right:
            threading.Thread(target=self._loop_right, daemon=True).start()

        # ── Timer in stats mỗi 2 giây ──
        self.create_timer(2.0, self._print_stats)

        self.get_logger().info("══════════════════════════════════════════")
        self.get_logger().info("  DIGIT PUBLISHER ready")
        self.get_logger().info(f"  LEFT  ({SERIAL_LEFT}) → {TOPIC_LEFT}")
        self.get_logger().info(f"  RIGHT ({SERIAL_RIGHT}) → {TOPIC_RIGHT}")
        self.get_logger().info("══════════════════════════════════════════")

    def _connect(self, serial, label):
        self.get_logger().info(f"🔌 Kết nối DIGIT {label} ({serial})...")
        try:
            d = Digit(serial)
            d.connect()
            d.set_resolution(Digit.STREAMS["QVGA"])
            d.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])
            d.set_intensity(Digit.LIGHTING_MAX)
            self.get_logger().info(f"   ✅ {label} OK")
            return d
        except Exception as e:
            self.get_logger().error(f"   ❌ {label} lỗi: {e}")
            return None

    def _loop_left(self):
        period = 1.0 / PUBLISH_RATE_HZ
        next_t = time.time()
        while self._running and rclpy.ok():
            try:
                frame = self.digit_left.get_frame()
                if frame is not None:
                    # Xoay nếu cần (DIGIT này đã portrait sẵn → ROTATE_DIR=None)
                    if ROTATE_DIR is not None:
                        frame = cv2.rotate(frame, ROTATE_DIR)
                    self.last_left = frame   # cho GUI
                    msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = FRAME_LEFT
                    self.pub_left.publish(msg)
                    self.count_left += 1
            except Exception as e:
                self.get_logger().warn(f"LEFT loop: {e}", throttle_duration_sec=2.0)
            next_t += period
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.time()

    def _loop_right(self):
        period = 1.0 / PUBLISH_RATE_HZ
        next_t = time.time()
        while self._running and rclpy.ok():
            try:
                frame = self.digit_right.get_frame()
                if frame is not None:
                    # Xoay nếu cần (DIGIT này đã portrait sẵn → ROTATE_DIR=None)
                    if ROTATE_DIR is not None:
                        frame = cv2.rotate(frame, ROTATE_DIR)
                    self.last_right = frame   # cho GUI
                    msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = FRAME_RIGHT
                    self.pub_right.publish(msg)
                    self.count_right += 1
            except Exception as e:
                self.get_logger().warn(f"RIGHT loop: {e}", throttle_duration_sec=2.0)
            next_t += period
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.time()

    def _print_stats(self):
        elapsed = max(time.time() - self.t_start, 0.001)
        fps_l = self.count_left  / elapsed
        fps_r = self.count_right / elapsed
        self.get_logger().info(
            f"  LEFT: {self.count_left}f ({fps_l:.1f}fps)  |  "
            f"RIGHT: {self.count_right}f ({fps_r:.1f}fps)"
        )

    def shutdown(self):
        self._running = False
        time.sleep(0.2)
        if self.digit_left:
            try: self.digit_left.disconnect()
            except: pass
        if self.digit_right:
            try: self.digit_right.disconnect()
            except: pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true",\
                    help="Hiện cửa sổ xem 2 DIGIT live")
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = None
    try:
        node = DigitPublisher()

        # ── GUI thread (nếu --gui) ──
        if args.gui:
            def gui_loop():
                win = "DIGIT — LEFT | RIGHT (portrait 320x240)"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                while rclpy.ok():
                    l = node.last_left
                    r = node.last_right
                    panels = []
                    for img, label in [(l, "LEFT D21383"), (r, "RIGHT D21384")]:
                        if img is None:
                            # placeholder portrait (H=320, W=240)
                            p = np.zeros((320, 240, 3), np.uint8)
                            cv2.putText(p, "NO DATA", (40, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            p = img.copy()
                        cv2.putText(p, label, (8, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        panels.append(p)
                    h = min(p.shape[0] for p in panels)
                    w = min(p.shape[1] for p in panels)
                    panels = [cv2.resize(p, (w, h)) for p in panels]
                    combined = np.hstack(panels)
                    cv2.imshow(win, combined)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                cv2.destroyAllWindows()

            threading.Thread(target=gui_loop, daemon=True).start()

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
    finally:
        try:
            node.shutdown()
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        print("✅ Đã thoát.")


if __name__ == "__main__":
    main()
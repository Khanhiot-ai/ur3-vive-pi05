#!/usr/bin/env python3
import cv2, threading, subprocess, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

WIDTH   = 640
HEIGHT  = 480
PUB_FPS = 30

def find_c922():
    """Tự tìm device path của C922"""
    try:
        out = subprocess.check_output(['v4l2-ctl', '--list-devices'],
                                       stderr=subprocess.DEVNULL).decode()
        lines = out.split('\n')
        for i, line in enumerate(lines):
            if 'C922' in line or 'c922' in line:
                # Lấy /dev/videoX dòng tiếp theo
                for j in range(i+1, min(i+5, len(lines))):
                    if '/dev/video' in lines[j]:
                        return lines[j].strip()
    except Exception:
        pass
    return None

class WristCam(Node):
    def __init__(self):
        super().__init__('wrist_cam')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera_wrist/image_raw', 10)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True

        device = find_c922()
        if not device:
            self.get_logger().error("❌ Không tìm thấy C922!")
            return
        self.get_logger().info(f"Found C922 at: {device}")

        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          PUB_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Không mở được {device}!")
            return

        threading.Thread(target=self._read_loop, daemon=True).start()
        self.create_timer(1.0 / PUB_FPS, self._publish)
        self.get_logger().info(f"✅ Wrist cam {device} {WIDTH}x{HEIGHT} @{PUB_FPS}fps")

    def _read_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def _publish(self):
        with self._lock:
            frame = self._frame
        if frame is None: return
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(str(e), throttle_duration_sec=2.0)

    def destroy_node(self):
        self._running = False
        self.cap.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = WristCam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

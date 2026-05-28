#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class FrameToPoseStamped(Node):

    def __init__(self):
        # 1. Khởi tạo Node với tên 'frame_to_posestamped'
        super().__init__('frame_to_posestamped')

        # 2. Khai báo tham số (Thay thế cho sys.argv)
        # Giá trị mặc định là 'right_controller', 'world', 30Hz
        self.declare_parameter('target_frame', 'right_controller')
        self.declare_parameter('reference_frame', 'world')
        self.declare_parameter('rate', 60)
        # self.declare_parameter('rate', 30)

        # Lấy giá trị tham số
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.ref_frame = self.get_parameter('reference_frame').get_parameter_value().string_value
        self.rate_hz = self.get_parameter('rate').get_parameter_value().integer_value

        # TF2 không cho phép frame_id bắt đầu với '/'
        self.target_frame = self.target_frame.lstrip('/')
        self.ref_frame = self.ref_frame.lstrip('/')

        # 3. Tạo Publisher
        topic_name = self.target_frame.replace('/', '_') + '_as_posestamped'
        # QoS = 10 thay cho queue_size=1 (Cấu hình mặc định)
        self.publisher_ = self.create_publisher(PoseStamped, topic_name, 10)

        # 4. Khởi tạo TF2 Buffer & Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 5. Tạo Timer (Thay thế cho vòng lặp while True)
        timer_period = 1.0 / self.rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(f'Started converting {self.target_frame} -> {self.ref_frame} at {self.rate_hz}Hz')

    def timer_callback(self):
        try:
            # 6. Lookup Transform (Hiện đại hơn tf cũ)
            # Lấy transform mới nhất có thể
            t = self.tf_buffer.lookup_transform(
                self.ref_frame,      # Khung đích (to_frame)
                self.target_frame,   # Khung nguồn (from_frame)
                rclpy.time.Time()    # Lấy thời gian mới nhất (0)
            )
        except TransformException as ex:
            # Không cần sleep, timer sẽ tự gọi lại sau
            self.get_logger().debug(f'Could not transform: {ex}')
            return

        # 7. Đóng gói tin nhắn
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg() # Lấy giờ hiện tại của Node
        msg.header.frame_id = self.ref_frame
        
        # Gán tọa độ từ transform (t) sang pose
        msg.pose.position.x = t.transform.translation.x
        msg.pose.position.y = t.transform.translation.y
        msg.pose.position.z = t.transform.translation.z
        msg.pose.orientation.x = t.transform.rotation.x
        msg.pose.orientation.y = t.transform.rotation.y
        msg.pose.orientation.z = t.transform.rotation.z
        msg.pose.orientation.w = t.transform.rotation.w

        # 8. Publish
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = FrameToPoseStamped()

    try:
        rclpy.spin(node) # Giữ cho node chạy liên tục
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

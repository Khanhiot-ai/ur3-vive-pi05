#!/usr/bin/env python3
"""
Script tính góc xoay World Alignment - MANUAL MODE
Phương pháp:
1. Gắn Tracker lên Robot
2. ĐIỀU KHIỂN THỦ CÔNG robot bằng Teach Pendant
3. Ghi lại chuyển động Tracker từ ROS2 topic
4. Tính góc giữa 2 vector
"""

import numpy as np
import URBasic
import time
import rclpy
import sys
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

class ManualCalibrator(Node):
    def __init__(self, robot_ip="192.168.1.1"):
        super().__init__('manual_calibrator')
        
        print("="*70)
        print("MANUAL WORLD ALIGNMENT CALIBRATION")
        print("="*70)
        
        # 1. KẾT NỐI ROBOT (Direct Connection)
        print(f"🤖 Connecting to robot at {robot_ip}...")
        self.robot = None
        try:
            # --- KHỞI TẠO ĐÚNG CÁCH CHO URBASIC ---
            self.robotModel = URBasic.robotModel.RobotModel()
            self.robot = URBasic.urScriptExt.UrScriptExt(
                host=robot_ip,
                robotModel=self.robotModel
            )
            self.robot.reset_error()
            time.sleep(1)
            print(f"✅ Robot Connected Successfully!")
            
        except Exception as e:
            print(f"❌ KẾT NỐI THẤT BẠI: {e}")
            print("👉 Hãy đảm bảo không có driver nào khác đang chạy (như ur_robot_driver)!")
            sys.exit(1)
        
        # Subscribe ROS2 topic
        print("\n🔌 Subscribing to /right_controller_as_posestamped...")
        self.tracker_pose = None
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/right_controller_as_posestamped',
            self.pose_callback,
            10
        )
        print("✅ Subscribed to topic")
    
    def pose_callback(self, msg):
        """Callback nhận pose từ topic"""
        self.tracker_pose = msg.pose
    
    def get_tracker_position_averaged(self, num_samples=50, sample_rate=50):
        """Lấy trung bình nhiều mẫu để giảm nhiễu"""
        samples = []
        sample_interval = 1.0 / sample_rate
        
        print(f"      Collecting {num_samples} samples...", end="", flush=True)
        
        for i in range(num_samples):
            rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.tracker_pose is not None:
                pos = np.array([
                    self.tracker_pose.position.x,
                    self.tracker_pose.position.y,
                    self.tracker_pose.position.z
                ])
                samples.append(pos)
            
            time.sleep(sample_interval)
        
        if len(samples) == 0:
            print(" ❌ No samples!")
            return None
        
        avg_pos = np.mean(samples, axis=0)
        std_dev = np.std(samples, axis=0)
        max_std = np.max(std_dev)
        
        print(f" ✅ ({len(samples)} samples, noise: {max_std*1000:.2f}mm)")
        
        if max_std > 0.005:
            print(f"      ⚠️ WARNING: High noise!")
        
        return avg_pos
    
    def get_robot_position(self):
        """Lấy vị trí Robot (TCP)"""
        pose = self.robot.get_actual_tcp_pose()
        return np.array(pose[:3])
    
    def calibrate_manual(self, num_points=6):
        """
        Calibration thủ công với Teach Pendant
        
        Args:
            num_points: Số điểm đo (bao gồm start và end)
        
        Returns:
            yaw_avg: Góc xoay trung bình
            alignment_matrix: Ma trận alignment
        """
        print("\n" + "="*70)
        print("MANUAL CALIBRATION PROCEDURE")
        print("="*70)
        print("Hướng dẫn:")
        print("1. GẮN Tracker lên Robot (flange hoặc end-effector)")
        print("2. Đảm bảo Tracker tracking (màu xanh trong SteamVR)")
        print(f"3. Dùng TEACH PENDANT di chuyển robot theo trục X")
        print(f"4. Script sẽ ghi lại vị trí tại {num_points} điểm")
        print()
        print("⚠️ QUAN TRỌNG:")
        print("   - Di chuyển THEO TRỤC X của robot")
        print("   - Mỗi lần di chuyển khoảng 5-10cm")
        print("   - Tổng quãng đường nên ~50cm")
        print()
        
        input("Sẵn sàng? Nhấn Enter để bắt đầu...")
        
        positions_tracker = []
        positions_robot = []
        
        # Ghi lại từng điểm
        for i in range(num_points):
            print(f"\n{'='*70}")
            print(f"📍 Point {i+1}/{num_points}")
            print(f"{'='*70}")
            
            if i == 0:
                print("Vị trí BAN ĐẦU:")
            elif i == num_points - 1:
                print("Vị trí CUỐI CÙNG:")
            else:
                print(f"Vị trí TRUNG GIAN {i}:")
            
            print()
            print("🎮 Dùng Teach Pendant:")
            if i == 0:
                print("   - Đặt robot ở vị trí bắt đầu")
            else:
                print(f"   - Di chuyển robot theo +X khoảng 5-10cm")
            print("   - Nhấn Enter khi đã đến vị trí")
            
            input("\n   Nhấn Enter khi sẵn sàng ghi lại vị trí...")
            
            # Ghi lại vị trí
            print("\n📊 Recording position:")
            print("   Tracker (averaged):")
            tracker_pos = self.get_tracker_position_averaged(num_samples=50)
            
            if tracker_pos is None:
                print("❌ Tracker not valid! Skipping this point.")
                continue
            
            print("   Robot:")
            robot_pos = self.get_robot_position()
            
            print(f"\n   Tracker: [{tracker_pos[0]:.4f}, {tracker_pos[1]:.4f}, {tracker_pos[2]:.4f}]")
            print(f"   Robot:   [{robot_pos[0]:.4f}, {robot_pos[1]:.4f}, {robot_pos[2]:.4f}]")
            
            positions_tracker.append(tracker_pos)
            positions_robot.append(robot_pos)
            
            print(f"\n   ✅ Point {i+1} recorded!")
        
        # Tính góc từ các cặp điểm liên tiếp
        print("\n" + "="*70)
        print("📊 CALCULATING YAW ANGLES")
        print("="*70)
        
        yaw_angles = []
        
        for i in range(len(positions_tracker) - 1):
            tracker_delta = positions_tracker[i+1] - positions_tracker[i]
            robot_delta = positions_robot[i+1] - positions_robot[i]
            
            print(f"\n🔢 Segment {i+1} → {i+2}:")
            print(f"   Tracker delta: [{tracker_delta[0]:+.4f}, {tracker_delta[1]:+.4f}, {tracker_delta[2]:+.4f}] m")
            print(f"   Robot delta:   [{robot_delta[0]:+.4f}, {robot_delta[1]:+.4f}, {robot_delta[2]:+.4f}] m")
            
            # Tính magnitude
            tracker_magnitude = np.linalg.norm(tracker_delta[:2])
            robot_magnitude = np.linalg.norm(robot_delta[:2])
            
            print(f"   Tracker magnitude: {tracker_magnitude*1000:.2f} mm")
            print(f"   Robot magnitude:   {robot_magnitude*1000:.2f} mm")
            
            # Tính góc
            tracker_xy = tracker_delta[:2]
            robot_xy = robot_delta[:2]
            
            tracker_angle = np.arctan2(tracker_xy[1], tracker_xy[0])
            robot_angle = np.arctan2(robot_xy[1], robot_xy[0])
            
            yaw_rad = robot_angle - tracker_angle
            yaw_deg = np.rad2deg(yaw_rad)
            
            # Normalize
            if yaw_deg > 180:
                yaw_deg -= 360
            elif yaw_deg < -180:
                yaw_deg += 360
            
            print(f"   Tracker angle: {np.rad2deg(tracker_angle):.2f}°")
            print(f"   Robot angle:   {np.rad2deg(robot_angle):.2f}°")
            print(f"   Yaw: {yaw_deg:.2f}°")
            
            yaw_angles.append(yaw_deg)
        
        # Thống kê
        print("\n" + "="*70)
        print("📊 STATISTICAL ANALYSIS")
        print("="*70)
        
        yaw_avg = np.mean(yaw_angles)
        yaw_std = np.std(yaw_angles)
        yaw_min = np.min(yaw_angles)
        yaw_max = np.max(yaw_angles)
        
        print(f"\nYaw angles measured:")
        for i, yaw in enumerate(yaw_angles, 1):
            deviation = yaw - yaw_avg
            print(f"   Segment {i}: {yaw:+7.2f}° (deviation: {deviation:+.2f}°)")
        
        print(f"\nStatistics:")
        print(f"   Average:  {yaw_avg:.2f}°")
        print(f"   Std dev:  {yaw_std:.2f}°")
        print(f"   Min:      {yaw_min:.2f}°")
        print(f"   Max:      {yaw_max:.2f}°")
        print(f"   Range:    {yaw_max - yaw_min:.2f}°")
        
        # Đánh giá
        print(f"\nQuality assessment:")
        if yaw_std < 0.5:
            print(f"   ✅ EXCELLENT! Very low variance (< 0.5°)")
        elif yaw_std < 1.0:
            print(f"   ✅ GOOD! Low variance (< 1.0°)")
        elif yaw_std < 2.0:
            print(f"   ⚠️ ACCEPTABLE. Moderate variance (< 2.0°)")
        else:
            print(f"   ❌ WARNING! High variance (> 2.0°)")
        
        # Tạo ma trận
        print("\n✅ Creating World Alignment Matrix...")
        alignment_matrix = self.create_alignment_matrix(yaw_avg)
        
        print(f"\nWorld Alignment Matrix (yaw = {yaw_avg:.2f}°):")
        print(alignment_matrix)
        
        # Lưu file
        output_file = "world_alignment_matrix.txt"
        np.savetxt(output_file, alignment_matrix)
        print(f"\n💾 Saved to: {output_file}")
        
        angle_file = "world_alignment_angle.txt"
        with open(angle_file, 'w') as f:
            f.write(f"{yaw_avg:.6f}\n")
            f.write(f"# Statistics:\n")
            f.write(f"# Std dev: {yaw_std:.6f}\n")
            f.write(f"# Measurements: {yaw_angles}\n")
        print(f"💾 Saved angle to: {angle_file}")
        
        return yaw_avg, alignment_matrix
    
    def create_alignment_matrix(self, yaw_degrees):
        """Tạo ma trận xoay quanh Z"""
        theta = np.deg2rad(yaw_degrees)
        
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0,              0,             1, 0],
            [0,              0,             0, 1]
        ])
        
        return R_z
    
    def shutdown(self):
        """Cleanup"""
        self.robot.close()
        self.destroy_node()

def main():
    print("\n" + "="*70)
    print("MANUAL WORLD ALIGNMENT CALIBRATION TOOL")
    print("="*70)
    print()
    print("Công cụ này sẽ:")
    print("1. Bạn dùng TEACH PENDANT di chuyển robot")
    print("2. Script ghi lại vị trí tại mỗi điểm")
    print("3. Tính góc xoay giữa 2 hệ tọa độ")
    print("4. Tạo file world_alignment_matrix.txt")
    print()
    
    # Cấu hình
    robot_ip = input("Robot IP [172.17.0.2]: ").strip() or "172.17.0.2"
    num_points = int(input("Number of points [6]: ").strip() or "6")
    
    # Khởi tạo ROS2
    rclpy.init()
    
    try:
        calibrator = ManualCalibrator(robot_ip)
        
        # Đợi tracker data
        print("\n⏳ Waiting for tracker data...")
        for i in range(10):
            rclpy.spin_once(calibrator, timeout_sec=0.5)
            if calibrator.tracker_pose is not None:
                print("✅ Receiving tracker data")
                break
        else:
            print("❌ No tracker data received!")
            return
        
        # Chạy calibration
        yaw, matrix = calibrator.calibrate_manual(num_points=num_points)
        
        print("\n" + "="*70)
        print("CALIBRATION COMPLETE!")
        print("="*70)
        print(f"Yaw angle: {yaw:.2f}°")
        print(f"Files created:")
        print("  - world_alignment_matrix.txt")
        print("  - world_alignment_angle.txt")
        print()
        print("Bạn có thể dùng ma trận này trong teleoperation!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            calibrator.shutdown()
        except:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    main()

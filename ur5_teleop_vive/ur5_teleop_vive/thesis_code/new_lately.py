#!/usr/bin/env python3
"""
Trajectory Analyzer - EUCLIDEAN PATH LENGTH (All Metrics)
----------------------------------------------------------
1. Visualization:
   - 3D Plot (Target vs Actual)
   - 3x 2D Plots (X, Y, Z vs Time)
2. Metrics (Euclidean-based):
   - Path Length: Euclidean distance cho cả per-axis và total
   - RMSE: Root Mean Square Error
   - Accuracy: Clamped về 0 nếu âm
3. Data: Raw (không lọc)
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from ur5_teleop_vive.msg import Xyzrpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import correlate, correlation_lags
import time
import sys
import threading

class TrajectoryStatsEuclidean(Node):
    def __init__(self):
        super().__init__('trajectory_stats_euclidean')
        
        self.target_data = []
        self.actual_data = []
        
        self.WAIT_TIME = 2.0
        self.start_time = None
        self.is_recording = False
        self.should_stop = False
        self.init_time = time.time()
        
        # Subscribers
        self.create_subscription(Marker, '/vive_gripper', self.target_cb, 10)
        self.create_subscription(Xyzrpy, '/ur_actual_pose', self.actual_cb, 10)
        
        self.get_logger().info('='*60)
        self.get_logger().info('📊 EUCLIDEAN PATH LENGTH ANALYSIS')
        self.get_logger().info('='*60)
        self.get_logger().info(f'⏳ Waiting {self.WAIT_TIME}s before recording...')
        self.get_logger().info('💡 Press ENTER to stop recording')
        
        # Keyboard listener
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def keyboard_listener(self):
        """Listen for Enter key to stop recording"""
        while not self.should_stop:
            try:
                input()
                if self.is_recording:
                    self.get_logger().info('⏸️  Enter pressed - stopping...')
                    self.should_stop = True
                    break
            except:
                break

    def target_cb(self, msg):
        if self.is_recording:
            t = time.time() - self.start_time
            self.target_data.append((t, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z))

    def actual_cb(self, msg):
        if self.is_recording:
            t = time.time() - self.start_time
            self.actual_data.append((t, msg.x, msg.y, msg.z))

    def control_loop(self):
        now = time.time()
        if not self.is_recording and (now - self.init_time >= self.WAIT_TIME):
            self.is_recording = True
            self.start_time = now
            self.get_logger().info('🔴 REC: START MOVING! Press ENTER when done.')
            return

        if self.is_recording and self.should_stop:
            self.is_recording = False
            duration = now - self.start_time
            self.get_logger().info(f'⚫ STOP: Recorded {duration:.1f}s. Processing...')
            self.process_metrics()
            sys.exit()

    def process_metrics(self):
        if len(self.target_data) < 50 or len(self.actual_data) < 50:
            self.get_logger().error('❌ Not enough data!')
            return

        # 1. Convert to Numpy
        target = np.array(self.target_data)
        actual = np.array(self.actual_data)
        
        t_target = target[:, 0]
        t_actual = actual[:, 0]
        xyz_target = target[:, 1:4]
        xyz_actual = actual[:, 1:4]

        # 2. Data Quality Check
        dt_actual = np.diff(t_actual)
        dt_target = np.diff(t_target)
        print("\n" + "="*60)
        print("📊 DATA QUALITY")
        print("="*60)
        print(f"Target: {1/np.mean(dt_target):.1f} Hz (std: {np.std(dt_target)*1000:.2f} ms)")
        print(f"Actual: {1/np.mean(dt_actual):.1f} Hz (std: {np.std(dt_actual)*1000:.2f} ms)")
        print(f"Samples: Target={len(t_target)}, Actual={len(t_actual)}")
        if np.max(dt_actual) > 0.1:
            print(f"⚠️ WARNING: Data gap detected! Max gap = {np.max(dt_actual)*1000:.0f} ms")

        # 3. Resampling Target to Actual time
        xyz_target_interp = np.zeros_like(xyz_actual)
        for i in range(3):
            xyz_target_interp[:, i] = np.interp(t_actual, t_target, xyz_target[:, i])

        # 4. Calculate Metrics
        diff = xyz_target_interp - xyz_actual
        
        # A. RMSE (mm)
        rmse_axis = np.sqrt(np.mean(diff**2, axis=0)) * 1000 
        rmse_total = np.sqrt(np.mean(np.linalg.norm(diff, axis=1)**2)) * 1000
        
        # B. Path Length - EUCLIDEAN METHOD
        diffs = np.diff(xyz_target_interp, axis=0)  # (N-1) × 3
        segment_lengths = np.linalg.norm(diffs, axis=1)  # Euclidean distance per segment
        
        # Total 3D path length
        path_length_total = np.sum(segment_lengths)
        
        # Per-axis path length (Manhattan distance)
        # Tổng quãng đường di chuyển tuyệt đối trên từng trục
        path_length_axis = np.sum(np.abs(diffs), axis=0)
        
        # Prevent division by zero
        path_length_axis[path_length_axis == 0] = 1e-6
        if path_length_total == 0: path_length_total = 1e-6
        
        # C. Accuracy per Axis (%) - CLAMPED TO 0
        accuracy_axis = np.maximum(0, (1 - (rmse_axis / (path_length_axis * 1000))) * 100)
        accuracy_total = max(0, (1 - (rmse_total / (path_length_total * 1000))) * 100)

        # D. Latency Estimate
        mag_target = np.linalg.norm(xyz_target_interp - np.mean(xyz_target_interp, axis=0), axis=1)
        mag_actual = np.linalg.norm(xyz_actual - np.mean(xyz_actual, axis=0), axis=1)
        corr = correlate(mag_target - np.mean(mag_target), mag_actual - np.mean(mag_actual), mode='full')
        lags = correlation_lags(len(mag_target), len(mag_actual), mode='full')
        latency_ms = lags[np.argmax(corr)] * np.mean(dt_actual) * 1000

        print("\n" + "="*60)
        print("📏 PATH LENGTH")
        print("="*60)
        print(f"Total 3D (Euclidean): {path_length_total:.4f} m")
        print(f"Per-axis (Manhattan): X={path_length_axis[0]:.4f}, Y={path_length_axis[1]:.4f}, Z={path_length_axis[2]:.4f} m")

        # --- VISUALIZATION ---
        fig = plt.figure(figsize=(18, 10))
        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.2, hspace=0.3)
        
        fig.suptitle(f"TRAJECTORY ANALYSIS", fontsize=16, fontweight='bold', color='darkblue')

        # --- LEFT COLUMN ---
        
        # 1. Stats Table Area (Top Left)
        ax_stats = plt.subplot2grid((3, 2), (0, 0))
        ax_stats.axis('off')

        # 2. 3D Plot Area (Middle & Bottom Left)
        ax3d = plt.subplot2grid((3, 2), (1, 0), rowspan=2, projection='3d')
        
        ax3d.plot(xyz_target_interp[:,0], xyz_target_interp[:,1], xyz_target_interp[:,2], 
                 'k--', lw=1.8, alpha=0.7, label='Target (Vive Tracker)')
        ax3d.plot(xyz_actual[:,0], xyz_actual[:,1], xyz_actual[:,2], 
                 'r-', lw=2.0, alpha=0.9, label='Actual (Robot)')
        
        # REMOVED START/END MARKERS as requested
        # ax3d.scatter(xyz_actual[0,0], xyz_actual[0,1], xyz_actual[0,2], c='g', s=60, label='Start')
        # ax3d.scatter(xyz_actual[-1,0], xyz_actual[-1,1], xyz_actual[-1,2], c='b', marker='X', s=60, label='End')

        # Equal Aspect Ratio
        all_coords = np.vstack([xyz_target_interp, xyz_actual])
        max_range = np.array([all_coords[:,0].ptp(), all_coords[:,1].ptp(), all_coords[:,2].ptp()]).max() / 2.0
        mid_x, mid_y, mid_z = np.mean(all_coords, axis=0)
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title("3D Trajectory", fontweight='bold')
        # Moved legend to upper left to avoid overlap
        ax3d.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize='small', framealpha=0.6)

        # [INFO BOX] STATS TABLE
        stats_text = (
            f"PERFORMANCE REPORT\n"
            f"--------------------------------------------------\n"
            f" AXIS | PATH (m) | RMSE (mm) | ACCURACY (%)\n"
            f"--------------------------------------------------\n"
            f"  X   |  {path_length_axis[0]:6.4f}  |  {rmse_axis[0]:6.2f}   |  {accuracy_axis[0]:6.2f} %\n"
            f"  Y   |  {path_length_axis[1]:6.4f}  |  {rmse_axis[1]:6.2f}   |  {accuracy_axis[1]:6.2f} %\n"
            f"  Z   |  {path_length_axis[2]:6.4f}  |  {rmse_axis[2]:6.2f}   |  {accuracy_axis[2]:6.2f} %\n"
            f"--------------------------------------------------\n"
            f" TOTAL|  {path_length_total:6.4f}  |  {rmse_total:6.2f}   |  {accuracy_total:6.2f} %"
        )
        
        ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes, 
                   ha='center', va='center', fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.95, edgecolor='#333'))

        # --- RIGHT: X, Y, Z PLOTS ---
        axes_xyz = [
            plt.subplot2grid((3, 2), (0, 1)),
            plt.subplot2grid((3, 2), (1, 1)),
            plt.subplot2grid((3, 2), (2, 1))
        ]
        labels = ['X', 'Y', 'Z']

        for i, ax in enumerate(axes_xyz):
            ax.plot(t_actual, xyz_target_interp[:, i], 'k--', alpha=0.6, label='Target')
            ax.plot(t_actual, xyz_actual[:, i], 'r-', label='Actual')
            ax.fill_between(t_actual, xyz_target_interp[:, i], xyz_actual[:, i], color='red', alpha=0.15)
            
            ax.set_ylabel(f'{labels[i]} (m)', fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            ax.set_title(f"Axis {labels[i]}: Acc {accuracy_axis[i]:.1f}% | Dist {path_length_axis[i]:.4f}m", 
                         fontsize=10, loc='left', color='#333')
            
            if i == 0: ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize='small')
            if i == 2: ax.set_xlabel('Time (s)', fontweight='bold')

        print("\n" + stats_text)
        print("\n✅ VISUALIZATION READY.")
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryStatsEuclidean()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Trajectory Analyzer (Detailed Axis Statistics) - Optimized
--------------------------------------------------------
Features:
1. Separate X, Y, Z axis plots.
2. Detailed statistics:
   - Total Path Length (Reference: Target)
   - RMSE per axis (mm)
   - Error Ratio (%) = RMSE / Target Path Length
3. Slider for latency adjustment.
4. Optimized 3D visualization for small scale movements (~300mm).
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from ur5_teleop_vive.msg import Xyzrpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# from mpl_toolkits.mplot3d import Axes3D # Not strictly needed in newer mpl but good for compat
from scipy import signal
import time
import sys

class TrajectoryAnalyzerStats(Node):
    def __init__(self):
        super().__init__('trajectory_analyzer_stats')
        
        self.target_data = []
        self.actual_data = []
        
        self.WAIT_TIME = 2.0
        self.RECORD_DURATION = 20.0 # Increased for better data
        self.TRIM_SECONDS = 3.0
        
        self.start_time = None
        self.is_recording = False
        self.init_time = time.time()
        
        self.create_subscription(Marker, '/vive_gripper', self.target_cb, 10)
        self.create_subscription(Xyzrpy, '/ur_actual_pose', self.actual_cb, 10)
        
        self.get_logger().info('='*60)
        self.get_logger().info('📊 ANALYZER: OPTIMIZED STATISTICS MODE')
        self.get_logger().info('='*60)
        self.get_logger().info(f'⏳ Waiting {self.WAIT_TIME}s before recording...')
        
        self.timer = self.create_timer(0.1, self.control_loop)

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
            self.get_logger().info('🔴 REC: START MOVING! (Test ~300mm range)')
            return

        if self.is_recording and (now - self.start_time > self.RECORD_DURATION):
            self.is_recording = False
            self.get_logger().info('⚫ STOP: Processing data...')
            self.process_and_visualize()
            sys.exit()

    def process_and_visualize(self):
        if len(self.target_data) < 50 or len(self.actual_data) < 50:
            self.get_logger().error('❌ Not enough data received!')
            return

        # --- PRE-PROCESSING ---
        target = np.array(self.target_data)
        actual = np.array(self.actual_data)
        
        # Trim Data (remove start shock) & Reset Time
        target = target[target[:, 0] > self.TRIM_SECONDS]
        actual = actual[actual[:, 0] > self.TRIM_SECONDS]
        
        if len(target) == 0 or len(actual) == 0:
             self.get_logger().error('❌ Data empty after trimming!')
             return

        target[:, 0] -= target[0, 0]
        actual[:, 0] -= actual[0, 0]
        
        t_target, xyz_target = target[:, 0], target[:, 1:4]
        t_actual, xyz_actual = actual[:, 0], actual[:, 1:4]

        # Auto-Latency Estimate (Cross Correlation of Magnitude)
        # Interpolate Target to Actual time grid
        xyz_target_interp = np.zeros_like(xyz_actual)
        for i in range(3):
            xyz_target_interp[:, i] = np.interp(t_actual, t_target, xyz_target[:, i])

        # Centered Magnitude (Shape only)
        mag_target = np.linalg.norm(xyz_target_interp - np.mean(xyz_target_interp, axis=0), axis=1)
        mag_actual = np.linalg.norm(xyz_actual - np.mean(xyz_actual, axis=0), axis=1)
        mag_target -= np.mean(mag_target)
        mag_actual -= np.mean(mag_actual)
        
        corr = signal.correlate(mag_target, mag_actual, mode='full')
        lags = signal.correlation_lags(len(mag_target), len(mag_actual), mode='full')
        suggested_latency = lags[np.argmax(corr)] * np.mean(np.diff(t_actual))
        
        print(f"🤖 Suggested Latency: {suggested_latency*1000:.2f} ms")

        # --- METRICS CALCULATION BASE ---
        # 1. Calculate Target Path Length (smoother, good reference for denominator)
        dist_target = np.sum(np.abs(np.diff(xyz_target, axis=0)), axis=0)
        dist_target[dist_target == 0] = 1e-6 
        
        # 2. Actual Path Length (for reference)
        dist_actual = np.sum(np.abs(np.diff(xyz_actual, axis=0)), axis=0)

        # Centering Data for Visuals (Remove Offset)
        centroid_actual = np.mean(xyz_actual, axis=0)
        xyz_actual_centered = xyz_actual - centroid_actual
        
        # --- VISUALIZATION SETUP ---
        fig = plt.figure(figsize=(16, 10))
        plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.4, top=0.88, left=0.05, right=0.95)

        # TEXT BOX STATS
        stats_text = fig.text(0.5, 0.94, '', ha='center', va='center', 
                              fontsize=11, family='monospace', weight='bold',
                              bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ccc'))

        # 1. 3D Plot (Left) - OPTIMIZED FOR 300mm scale
        ax3d = plt.subplot2grid((3, 2), (0, 0), rowspan=3, projection='3d')
        line_target_3d, = ax3d.plot([], [], [], 'g--', label='Target (Shifted)', lw=1)
        ax3d.plot(xyz_actual_centered[:,0], xyz_actual_centered[:,1], xyz_actual_centered[:,2], 
                 'r-', label='Actual', lw=2, alpha=0.8)
        
        # Add Start/End markers
        ax3d.scatter(xyz_actual_centered[0,0], xyz_actual_centered[0,1], xyz_actual_centered[0,2], c='k', marker='o', s=30, label='Start')
        ax3d.scatter(xyz_actual_centered[-1,0], xyz_actual_centered[-1,1], xyz_actual_centered[-1,2], c='k', marker='x', s=50, label='End')

        ax3d.set_title('Spatial Trajectory (Centered) - Equal Aspect')
        ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
        ax3d.legend()

        # Enforce Equal Aspect Ratio for 3D
        # Calculate max range to center the plot
        max_range = np.array([xyz_actual_centered[:,0].ptp(), xyz_actual_centered[:,1].ptp(), xyz_actual_centered[:,2].ptp()]).max() / 2.0
        mid_x, mid_y, mid_z = np.mean(xyz_actual_centered, axis=0) # Centered around 0
        
        # Little padding
        max_range *= 1.1 
        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)


        # 2. XYZ Subplots (Right)
        axes_xyz = [
            plt.subplot2grid((3, 2), (0, 1)),
            plt.subplot2grid((3, 2), (1, 1)),
            plt.subplot2grid((3, 2), (2, 1))
        ]
        labels = ['X Axis', 'Y Axis', 'Z Axis']
        lines_target_xyz = []

        for i, ax in enumerate(axes_xyz):
            ax.plot(t_actual, xyz_actual_centered[:, i], 'r-', label='Actual', alpha=0.7)
            ln, = ax.plot([], [], 'g--', label='Target (Shifted)', lw=1.5)
            lines_target_xyz.append(ln)
            ax.set_ylabel(f'{labels[i]} (m)')
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
            if i == 0: ax.legend(loc='upper right', fontsize='small')

        axes_xyz[0].set_title('Axis Separation Analysis (Time Domain)')
        axes_xyz[2].set_xlabel('Time (s)')

        # --- SLIDER SETUP ---
        range_width = 3.0
        ax_slider = plt.axes([0.15, 0.02, 0.70, 0.04])
        slider = Slider(
            ax_slider, 'Latency Shift (s)', 
            valmin=suggested_latency - range_width, 
            valmax=suggested_latency + range_width, 
            valinit=suggested_latency,
            valfmt='%.4f s'
        )

        def update(val):
            lag = slider.val
            
            # 1. Time Shift & Interpolate Target to Actual grid
            xyz_target_shifted = np.zeros_like(xyz_actual)
            for i in range(3):
                # Interpolate using (t_actual - lag) to shift target in time
                xyz_target_shifted[:, i] = np.interp(t_actual - lag, t_target, xyz_target[:, i])

            # 2. Centering Target (Dynamic centering based on shifted view)
            centroid_tgt = np.mean(xyz_target_shifted, axis=0)
            xyz_target_centered = xyz_target_shifted - centroid_tgt 
            
            # 3. CALCULATE ERROR METRICS (Per Axis)
            # Vector Error (Actual - Target)
            diff = xyz_actual_centered - xyz_target_centered
            
            # RMSE per axis (mm)
            rmse_axis = np.sqrt(np.mean(diff**2, axis=0)) * 1000 
            
            # Error Percentage = (RMSE / Target Path Length) * 100
            # Dividing by Target Path Length avoids "Actual Jitter" inflating the denominator
            err_percent = (rmse_axis / (dist_target * 1000)) * 100 
            
            # Total RMSE (3D)
            rmse_total = np.sqrt(np.mean(np.linalg.norm(diff, axis=1)**2)) * 1000

            # 4. Update Plots
            line_target_3d.set_data(xyz_target_centered[:,0], xyz_target_centered[:,1])
            line_target_3d.set_3d_properties(xyz_target_centered[:,2])
            
            for i, ln in enumerate(lines_target_xyz):
                ln.set_data(t_actual, xyz_target_centered[:, i])

            # 5. Update Stats Table
            table_str = (
                f"LATENCY: {lag*1000:6.1f} ms  |  TOTAL RMSE (3D): {rmse_total:6.2f} mm\n"
                f"{'-'*75}\n"
                f"AXIS | DIST (Target) |   RMSE (Avg)  |  ERROR RATIO (%)\n"
                f"  X  |   {dist_target[0]:6.3f} m   |   {rmse_axis[0]:6.2f} mm  |     {err_percent[0]:6.2f} %\n"
                f"  Y  |   {dist_target[1]:6.3f} m   |   {rmse_axis[1]:6.2f} mm  |     {err_percent[1]:6.2f} %\n"
                f"  Z  |   {dist_target[2]:6.3f} m   |   {rmse_axis[2]:6.2f} mm  |     {err_percent[2]:6.2f} %"
            )
            stats_text.set_text(table_str)
            
            fig.canvas.draw_idle()

        slider.on_changed(update)
        update(suggested_latency)
        
        plt.show()
        print(f"\n✅ FINAL LATENCY: {slider.val*1000:.2f} ms")

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryAnalyzerStats()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

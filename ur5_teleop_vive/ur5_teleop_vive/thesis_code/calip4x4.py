#!/usr/bin/env python3
"""
Full 4x4 World Alignment Calibration (Kabsch/SVD)

Khác với calib_manual.py (chỉ tính yaw quanh Z), script này tính:
  - Rotation matrix 3x3 đầy đủ (yaw + pitch + roll)
  - Translation vector
  - Output: ma trận 4x4 hoàn chỉnh

Cách thu data:
  1. Gắn Vive tracker lên flange của UR3
  2. Dùng Teach Pendant di chuyển robot đến N điểm RẢI ĐỀU trong không gian
     (KHÔNG phải dọc 1 trục — phải 3D thực sự, ít nhất 8 điểm)
  3. Ở mỗi điểm: nhấn Enter để ghi lại cả Vive pos và TCP pos
  4. Script tự động tính ma trận tối ưu bằng SVD (Kabsch algorithm)

Lý thuyết:
  Tìm T (4x4) sao cho: T @ vive_homogeneous ≈ robot_position cho tất cả điểm.
  Bước 1: Tính centroid hai cloud, dịch về gốc.
  Bước 2: H = V^T @ R (covariance matrix).
  Bước 3: U, S, V^T = SVD(H).
  Bước 4: Rotation = V^T^T @ U^T (kèm fix lật nếu det < 0).
  Bước 5: Translation = centroid_robot - Rotation @ centroid_vive.

Lỗi RMSE sau alignment được in ra. Mục tiêu: < 5mm. Nếu > 10mm,
có nghĩa là tracking Vive bị nhiễu hoặc lighthouse chưa được calibrate kỹ.
"""

import numpy as np
import sys
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

try:
    import URBasic
except ImportError:
    print("❌ URBasic chưa cài. Chạy: pip install URBasic"); sys.exit(1)


def kabsch_4x4(vive_points, robot_points):
    """
    Tìm 4x4 transform tối ưu mapping Vive → Robot bằng Kabsch algorithm.

    Args:
        vive_points:  (N, 3) array
        robot_points: (N, 3) array (cùng index = corresponding)
    Returns:
        T: (4, 4) homogeneous transformation matrix
        rmse: float — root mean square error (m)
    """
    assert vive_points.shape == robot_points.shape
    assert vive_points.shape[0] >= 3, "Cần ít nhất 3 điểm"

    # 1. Tính centroid
    centroid_v = vive_points.mean(axis=0)
    centroid_r = robot_points.mean(axis=0)

    # 2. Dịch về gốc
    V = vive_points - centroid_v
    R = robot_points - centroid_r

    # 3. Covariance matrix
    H = V.T @ R

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Rotation (fix reflection nếu det < 0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    Rot = Vt.T @ D @ U.T

    # 6. Translation
    t = centroid_r - Rot @ centroid_v

    # 7. Build 4x4
    T = np.eye(4)
    T[:3, :3] = Rot
    T[:3, 3] = t

    # 8. Tính RMSE
    transformed = (Rot @ vive_points.T).T + t
    errors = np.linalg.norm(transformed - robot_points, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return T, rmse


class Calib4x4(Node):
    def __init__(self, robot_ip="172.17.0.2"):
        super().__init__('calib_4x4')

        print("="*70)
        print("FULL 4x4 WORLD ALIGNMENT CALIBRATION (Kabsch/SVD)")
        print("="*70)

        # 1. Kết nối robot
        print(f"\n🤖 Kết nối robot {robot_ip}...")
        try:
            self.robotModel = URBasic.robotModel.RobotModel()
            self.robot = URBasic.urScriptExt.UrScriptExt(
                host=robot_ip, robotModel=self.robotModel)
            self.robot.reset_error()
            time.sleep(1)
            print(f"✅ Robot connected")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            sys.exit(1)

        # 2. Subscribe Vive
        print("\n🔌 Subscribe /right_controller_as_posestamped...")
        self.tracker_pose = None
        self.create_subscription(
            PoseStamped, '/right_controller_as_posestamped',
            self._pose_cb, 10)
        print("✅ Subscribed")

    def _pose_cb(self, msg):
        self.tracker_pose = msg.pose

    def get_vive_pos_averaged(self, n=50):
        """Lấy trung bình 50 mẫu để giảm noise"""
        samples = []
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tracker_pose is not None:
                samples.append([
                    self.tracker_pose.position.x,
                    self.tracker_pose.position.y,
                    self.tracker_pose.position.z,
                ])
            time.sleep(0.02)
        if len(samples) == 0:
            return None
        arr = np.array(samples)
        return arr.mean(axis=0), arr.std(axis=0).max()

    def get_robot_pos(self):
        pose = self.robot.get_actual_tcp_pose()
        return np.array(pose[:3])

    def run(self, n_points=8):
        print("\n" + "="*70)
        print("HƯỚNG DẪN")
        print("="*70)
        print(f"1. Gắn TRACKER lên flange của UR3")
        print(f"2. Đảm bảo tracker xanh trong SteamVR")
        print(f"3. Dùng TEACH PENDANT di chuyển robot đến {n_points} điểm")
        print(f"   ⚠ QUAN TRỌNG: các điểm phải RẢI ĐỀU trong không gian 3D")
        print(f"   ⚠ KHÔNG được nằm trên 1 đường thẳng hay 1 mặt phẳng")
        print(f"   ⚠ Cách nhau ít nhất 10cm")
        print(f"4. Ở mỗi điểm: nhấn Enter")
        print()
        input("Sẵn sàng? Nhấn Enter để bắt đầu...")

        # Đợi tracker data
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.tracker_pose is not None:
                break
        else:
            print("❌ Không nhận được Vive data"); return

        # Thu N điểm
        vive_pts = []
        robot_pts = []

        for i in range(n_points):
            print(f"\n{'─'*70}")
            print(f"📍 Điểm {i+1}/{n_points}")
            print(f"{'─'*70}")
            print("Di chuyển robot đến tư thế mới (khác xa các điểm trước)")
            input("Nhấn Enter để ghi lại...")

            print("Đang lấy mẫu Vive...", end="", flush=True)
            result = self.get_vive_pos_averaged(50)
            if result is None:
                print(" ❌"); continue
            vive_pos, noise = result
            print(f" ✓ noise={noise*1000:.2f}mm")

            robot_pos = self.get_robot_pos()

            vive_pts.append(vive_pos)
            robot_pts.append(robot_pos)

            print(f"  Vive  : [{vive_pos[0]:+.4f}, {vive_pos[1]:+.4f}, {vive_pos[2]:+.4f}]")
            print(f"  Robot : [{robot_pos[0]:+.4f}, {robot_pos[1]:+.4f}, {robot_pos[2]:+.4f}]")

        # Tính alignment
        print("\n" + "="*70)
        print("📊 COMPUTING 4x4 ALIGNMENT (Kabsch/SVD)")
        print("="*70)

        vive_arr = np.array(vive_pts)
        robot_arr = np.array(robot_pts)

        T, rmse = kabsch_4x4(vive_arr, robot_arr)

        print(f"\nMa trận alignment 4x4:")
        print(T)
        print(f"\nRMSE: {rmse*1000:.2f} mm")

        if rmse < 0.005:
            print("✅ EXCELLENT (< 5mm)")
        elif rmse < 0.010:
            print("✓ GOOD (< 10mm)")
        elif rmse < 0.020:
            print("⚠ ACCEPTABLE (< 20mm) — có thể cần thu lại")
        else:
            print("❌ POOR (> 20mm) — TRACKING KÉM, kiểm tra lighthouse")

        # In errors từng điểm
        Rot = T[:3, :3]; t = T[:3, 3]
        transformed = (Rot @ vive_arr.T).T + t
        errors = np.linalg.norm(transformed - robot_arr, axis=1)
        print(f"\nLỗi từng điểm:")
        for i, e in enumerate(errors):
            print(f"  Điểm {i+1}: {e*1000:.2f} mm")

        # Lưu
        np.savetxt("world_alignment_matrix.txt", T)
        print(f"\n💾 Đã lưu world_alignment_matrix.txt (4x4 đầy đủ)")
        print("→ Restart vive_ur5_teleop_params.py để áp dụng")

    def shutdown(self):
        try: self.robot.close()
        except Exception: pass
        self.destroy_node()


def main():
    print("\n" + "="*70)
    print("FULL 4x4 CALIBRATION TOOL (Kabsch/SVD)")
    print("="*70)
    print("So với calib_manual.py:")
    print("  calib_manual: chỉ tính yaw Z (1 DOF)")
    print("  calib_4x4:    tính full 3x3 rotation + translation (6 DOF)")
    print("→ Orientation tracker và TCP sẽ thẳng hàng đúng trong RViz")
    print()

    ip = input("Robot IP [172.17.0.2]: ").strip() or "172.17.0.2"
    n  = int(input("Số điểm [8]: ").strip() or "8")

    rclpy.init()
    node = Calib4x4(ip)
    try:
        node.run(n_points=n)
    except KeyboardInterrupt:
        print("\nCancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
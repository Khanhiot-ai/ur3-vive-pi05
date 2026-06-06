# UR3 + HTC Vive Teleop — Thu data cho Pi0.5 VLA với cảm biến xúc giác DIGIT

Hệ thống teleoperation điều khiển robot **UR3** bằng **HTC Vive Tracker**, thu thập demonstrations để train mô hình **Pi0.5 (Vision-Language-Action)** có tích hợp cảm biến xúc giác **DIGIT v1**.

**Task:** Gắp bánh răng thả vào cột (gear pick-and-place, contact-rich, 2 loại bánh răng to/nhỏ).

🔗 **Repo:** https://github.com/Khanhiot-ai/ur3-vive-pi05
📦 **Dataset:** https://huggingface.co/datasets/qkhanh1/ur3_pick_cube
📚 **Tham khảo:** https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5

---

## 1. Tổng quan hệ thống

```
HTC Vive Tracker  ──►  UR3 (RTDE servoL @100Hz)
       │
       ├──►  2× Realsense D435i (front + wrist)
       ├──►  2× DIGIT v1 tactile (left + right, 60Hz)
       └──►  Robstride gripper (master/slave qua CAN)
                         │
                         ▼
              record_all.py  ──►  HDF5 dataset
                         │
                         ▼
        Convert_hdf5_to_lerobot.py  ──►  LeRobot v2 (train Pi0.5)
```

## 2. Phần cứng

| Thiết bị | Chi tiết |
|---|---|
| Robot | UR3 @ 192.168.1.1 (Remote Control mode), RTDE servoL |
| Camera | 2× Realsense D435i: front + wrist (480×640 RGB) |
| Tracker | HTC Vive Tracker 3.0 + 2 Lighthouse |
| Tactile | 2× DIGIT v1: D21383 (LEFT), D21384 (RIGHT) — QVGA 320×240 @60fps |
| Gripper | Robstride: ID 7 (master/vô lăng), ID 6 (slave/tay kẹp), CANable2 |

## 3. Mô hình Pi0.5 — lý do thiết kế data

Pi0.5 có 2 backbone xúc giác đông cứng, mỗi cái cần input khác:

- **DINO** (ước lượng lực): ăn ảnh DIGIT, model tự resize về 224×224.
- **V-JEPA** (phát hiện trượt/slip): ăn cửa sổ 4 frame liên tiếp @60fps (~100ms), độ phân giải **(320 cao × 240 rộng) PORTRAIT** RGB.

→ DIGIT bắt buộc: **60Hz**, **portrait (320,240)**, **forward delta action**.

## 4. Cấu trúc HDF5 (dual-rate)

DIGIT thu ở 60Hz (callback), camera/state/action thu ở 20Hz (tick). Timestamp dùng để align sau.

```
dataset/<task>.hdf5
└── data/
    ├── demo_0/
    │   ├── obs/
    │   │   ├── image           (T20, 480, 640, 3)  uint8   Realsense front @20Hz
    │   │   ├── wrist_image     (T20, 480, 640, 3)  uint8   Realsense wrist @20Hz
    │   │   ├── digit_left      (T60, 320, 240, 3)  uint8   DIGIT trái @60Hz
    │   │   ├── digit_right     (T60, 320, 240, 3)  uint8   DIGIT phải @60Hz
    │   │   ├── digit_left_ts   (T60,)              float64 timestamp @60Hz
    │   │   ├── digit_right_ts  (T60,)              float64
    │   │   ├── state           (T20, 8)            float32 [ee_xyz, quat, gripper]
    │   │   └── timestamp       (T20,)              float64 tick @20Hz
    │   ├── actions             (T20, 7)            float32 forward delta
    │   └── attrs: success, n_frames, fps, robot=UR3, action_convention
    └── demo_1/, demo_2/, ...
```

- **state (8):** `[ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper]`
- **action (7):** `[dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]` — forward delta: `action[t] = pose(t+1) - pose(t)`
- Tỉ lệ T60/T20 ≈ 3 (DIGIT 60Hz / tick 20Hz)

Định dạng tương thích `lerobot/berkeley_autolab_ur5` (khác: gripper liên tục thay vì 0/1, quaternion cố định cho task tịnh tiến).

## 5. Cài đặt

```bash
# ROS2 Humble + workspace
source /opt/ros/humble/setup.bash
source ~/ur5_teleop_vive/install/setup.bash

# Dependencies
pip install digit-interface evdev h5py opencv-python numpy
pip install ur-rtde            # RTDE robot control

# Quyền DIGIT (mỗi lần khởi động)
sudo chmod 666 /dev/video*

# CAN cho gripper
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up

# Bàn phím evdev (cho phím Home set origin trên Wayland)
sudo usermod -a -G input $USER   # rồi logout/login lại
```

## 6. Chạy pipeline (8 terminal)

Mỗi terminal source workspace trước (`source ~/ur5_teleop_vive/install/setup.bash`).

| Terminal | Lệnh | Chức năng |
|---|---|---|
| T1 | `python3 vive_tf_and_joy_ros2.py` | Đọc Vive tracker → TF |
| T2 | `python3 frame_as_posestamped_ros2.py` | TF → PoseStamped @60Hz |
| T3 | `python3 vive_ur5_teleop_params.py` | Teleop logic + set origin (Home key) |
| T4 | `python3 ur_follow_using_class_ros2.py` | Điều khiển UR3 qua RTDE |
| T5 | `./launch_realsense_all.sh` | 2 Realsense camera |
| T6 | `python3.10 control_robstride_ros_without_calip.py --channel can0` | Gripper |
| T7 | `python3 digit_publisher_ros2.py` | 2 DIGIT @60Hz |
| T8 | `python3 record_all.py --task pick_cube --fps 20` | Recorder |

**T6 gripper** — gõ lần lượt:
```
c   → auto-calip hành trình tay kẹp
m   → bật MIMIC (tay kẹp bám vô lăng)
o   → (mở vô lăng hết) chốt mốc pos_norm = 0.0
p   → (kẹp vô lăng hết) chốt mốc pos_norm = 1.0
```

## 7. Quy trình thu demo

```
1. Đặt bánh răng ở vị trí mới
2. Cầm tracker (đảm bảo lighthouse track — đèn tracker xanh)
3. Bấm phím HOME (hoặc numpad 7) → chốt origin, robot di chuyển đến tracker
4. SPACE (ở T8) → robot ON + bắt đầu record
5. Di chuyển tracker → kẹp bánh răng → đưa đến cột → thả vào
6. S (thành công) hoặc F (thất bại) → lưu + robot OFF + tự về home
7. Đợi robot về home (~2s), lặp lại từ bước 1
```

**Phím tắt recorder (T8):** `SPACE` = rec/stop, `S` = save success, `F` = save fail, `Q` = quit.

## 8. Kiểm tra dataset

```bash
# Kiểm tra HDF5 (schema, dual-rate, portrait, timestamp)
python3 check_hdf5.py

# Xem ảnh demo
python3 check_hdf5.py --demo 0 --save
```

Output cần thấy:
```
✅ obs/digit_left: (T60, 320, 240, 3)
✅ Dual-rate OK: ~3× (DIGIT 60Hz)
✅ Portrait (320,240) — đúng cho V-JEPA
✅ timestamp: digit (60Hz), tick (20Hz)
```

## 9. Convert sang LeRobot + push HuggingFace

```bash
# Convert HDF5 → LeRobot v2 (gồm DIGIT)
python3.10 Convert_hdf5_to_lerobot.py \
  --src dataset/pick_cube.hdf5 \
  --task "put the gear onto the peg" \
  --fps 20 --skip-failed --overwrite

# Push lên HuggingFace
python3.10 push_to_huggingface.py --repo-id <user>/ur3_pick_cube
```

Sau convert, LeRobot có: `observation.image`, `observation.wrist_image`, `observation.digit_left`, `observation.digit_right`, `observation.state`, `action`.

## 10. Files

| File | Mô tả |
|---|---|
| `vive_tf_and_joy_ros2.py` | Đọc Vive tracker, publish TF + teleop enable |
| `frame_as_posestamped_ros2.py` | Convert TF → PoseStamped @60Hz |
| `vive_ur5_teleop_params.py` | Teleop logic, world alignment, set origin (evdev Home key) |
| `ur_follow_using_class_ros2.py` | Điều khiển UR3 qua RTDE, auto home |
| `control_robstride_ros_without_calip.py` | Gripper master/slave qua CAN |
| `digit_publisher_ros2.py` | Publish 2 DIGIT @60Hz |
| `record_all.py` | Recorder HDF5 dual-rate |
| `Convert_hdf5_to_lerobot.py` | Convert HDF5 → LeRobot v2 |
| `check_hdf5.py` | Kiểm tra dataset |
| `launch_realsense_all.sh` | Khởi động 2 Realsense |
| `view_ur5.launch.py` | Xem robot UR3 trong RViz |

## 11. Ghi chú kỹ thuật

- **Xem RViz:** `ros2 launch view_ur5.launch.py` (cần T4 chạy publish `/joint_states`). RViz hoạt động với RTDE, không cần URBasic.
- **Phím Home:** đọc qua `evdev` (bắt cả `KEY_HOME` và `KEY_KP7` vì phím Home laptop = numpad 7), chạy được trên Wayland. Cần user thuộc group `input`.
- **World alignment:** `world_alignment_matrix.txt` (góc xoay ~-30.49°). Không recalib trừ khi lighthouse/robot dời.
- **Folder "ur5":** robot thật là UR3, tên folder để "ur5" cho tiện (không đổi tên).
- **DIGIT node:** `/dev/video*` đổi số mỗi lần cắm — chạy `sudo chmod 666 /dev/video*` và rút/cắm lại nếu lỗi.
- **DIGIT portrait:** DIGIT v1 trả frame (320,240) portrait sẵn → KHÔNG xoay (`ROTATE_DIR = None`). Nếu DIGIT khác trả landscape thì đặt `cv2.ROTATE_90_CLOCKWISE`.

## 12. Troubleshooting

| Lỗi | Fix |
|---|---|
| DIGIT "Cannot open video" | `sudo chmod 666 /dev/video*`, rút/cắm lại |
| Gripper "Transmit buffer full" | Reset CAN: `ip link set can0 down/up`, tăng `txqueuelen 1000` |
| Home không set origin | Cần T1-T4 chạy + tracker được track (pose_callback cần tracker pose) |
| `actual ○` ở recorder | T4 ur_follow chưa publish, hoặc tracker chưa track |
| Participant index full | Đóng bớt `ros2 topic echo`, hoặc tăng MaxAutoParticipantIndex |
| bashrc "not found" workspace cũ | Xóa dòng source workspace cũ trong `~/.bashrc` |

## 13. Changelog (các cải tiến chính)

- **DIGIT dual-rate 60Hz/20Hz:** DIGIT thu ở callback 60Hz, camera/state/action ở tick 20Hz, lưu timestamp để align — đáp ứng V-JEPA cần 60fps.
- **DIGIT portrait (320,240):** giữ nguyên frame portrait gốc cho V-JEPA.
- **Forward delta action:** `action[t] = pose(t+1) - pose(t)` (convention BC chuẩn).
- **Bỏ tactile_state:** không lưu torque thô vào HDF5 (gripper vẫn dùng torque để kẹp).
- **Phím Home qua evdev:** chạy trên Wayland, bắt KEY_HOME + KEY_KP7.
- **Tự động hóa:** SPACE = origin + robot ON + record, S/F = save + off + về home.
- **Converter có DIGIT:** đưa digit_left/right vào LeRobot (map 60→20Hz theo timestamp).

# UR3 Vive Teleop + Pi0.5 Dataset System

Hệ thống điều khiển **UR3** bằng **HTC Vive Tracker** + gripper **Robstride**, thu data đa-modal HDF5 để fine-tune **Pi0.5 VLA** cho task pick & place.

🔗 **Repo**: https://github.com/Khanhiot-ai/ur3-vive-pi05
📦 **Dataset**: https://huggingface.co/datasets/qkhanh1/ur3_pick_cube

---

## 📋 Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Changelog — Các Fix Đã Làm](#2-changelog--các-fix-đã-làm)
3. [Phần Cứng](#3-phần-cứng)
4. [Phần Mềm](#4-phần-mềm)
5. [Cấu Trúc Dự Án](#5-cấu-trúc-dự-án)
6. [Cài Đặt và Build](#6-cài-đặt-và-build)
7. [Setup Phần Cứng](#7-setup-phần-cứng)
8. [Calibration (góc yaw)](#8-calibration-góc-yaw)
9. [Workflow Thu Data](#9-workflow-thu-data)
10. [Check Dataset HDF5](#10-check-dataset-hdf5)
11. [Format Dataset HDF5](#11-format-dataset-hdf5)
12. [Convert + Push HuggingFace](#12-convert--push-huggingface)
13. [Inference Pi0.5 trên Robot](#13-inference-pi05-trên-robot)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Tổng Quan

### Mục tiêu

Thu dataset demonstration để fine-tune **Pi0.5 VLA** cho task gắp vật trên UR3.

### Pipeline tổng

```
┌─ THU DATA ─────────┐    ┌─ TRAIN PI0.5 ──┐    ┌─ INFERENCE ─────┐
│ Vive Tracker       │    │ Pi0.5 base     │    │ Camera + state  │
│ → UR3 + Gripper    │ →  │ + ur3 dataset  │ →  │ → Pi0.5         │
│ → HDF5             │    │ → Checkpoint   │    │ → Robot tự làm  │
└──────────────┬─────┘    └────────┬───────┘    └─────────────────┘
               │                   │                      ▲
               └── HuggingFace ────┴──────────────────────┘
                   (qkhanh1/ur3_pick_cube)
```

### State + Action (cho Pi0.5)

```
State  (7 dim) = [joint1..joint6, gripper_pos_norm]
Action (8 dim) = [x, y, z, qx, qy, qz, qw, gripper_cmd]
```

### Pipeline 8 terminal

| T | Node | Vai trò |
|---|---|---|
| 1 | `vive_tf_and_joy_ros2.py` | OpenVR → /tf + Joy (Ctrl_R toggle) |
| 2 | `frame_as_posestamped_ros2.py` | TF → PoseStamped @60Hz |
| 3 | `vive_ur5_teleop_params.py` | Apply yaw alignment → `/ur_target_pose` |
| 4 | `ur_follow_using_class_ros2.py` | RTDE servoL @100Hz |
| 5 | `./launch_realsense_all.sh` | 2 Realsense D435I đồng thời |
| 6 | `control_robstride_ros_without_calip.py` | Gripper Mimic PID + `/gripper/state` |
| 7 | `record_all.py` | HDF5 recorder |

---

## 2. Changelog — Các Fix Đã Làm

### 2.1 Driver UR: URBasic → ur_rtde

**Lý do**: URBasic hay treo, lỗi `get_inverse_kin failed`, latency cao.
**Mới**: `RTDEControlInterface` + `RTDEReceiveInterface`, servoL @100Hz, force/torque feedback.

### 2.2 Calibration: 4×4 Kabsch → góc yaw đơn giản

**Lý do**: Kabsch 8 điểm 3D phức tạp, dễ sai khi điểm gần đồng phẳng.
**Mới** (`calib_manual.py`): chỉ đo **góc yaw** giữa trục X tracker và robot.

Output:
```
world_alignment_angle.txt    ← -30.490862°  (giá trị thực tế đã đo)
world_alignment_matrix.txt   ← Ma trận 4×4 tự sinh từ góc
```

### 2.3 Camera: C922 webcam → 2× Realsense D435I

**Lý do**: `usb_cam` crash với mọi `pixel_format`. C922 chỉ USB 2.0.
**Mới**: 2 Realsense cùng loại, cùng USB 3.0 (Bus 04 Thunderbolt 4).

Cách launch 2 cam đồng thời:
```bash
./launch_realsense_all.sh
```

Topics:
```
/camera_front/camera/color/image_raw    ← serial 243322073847
/camera_wrist/camera/color/image_raw    ← serial 027422070272
```

Vấn đề đã gặp khi setup:
- Serial dạng số thuần → ROS2 parser tự cast sang integer → **fix**: bọc trong `\"...\"` trong bash
- Namespace + node name → topic thêm `camera/` → **fix**: bỏ `-r __node:=...`, topic ra `/<ns>/camera/color/image_raw`
- `rs-enumerate-devices -s` có ERROR log trước output → `awk` lấy sai cột → **fix**: dùng `grep -oE "[0-9]{12}"`
- Realsense #2 cắm vào Bus 03 USB 2.0 (dù cổng nhìn xanh) → **fix**: tìm cổng Bus 04 Port 4 + Port 8

### 2.4 Ctrl_R Toggle (tap thay vì hold)

**Lý do**: Giữ phím liên tục mỏi tay, nhả lỡ là robot dừng giữa demo.
**Mới**: Tap Ctrl_R 1 lần = ON, tap lần 2 = OFF. Edge detection chống OS autorepeat.

```python
if not _last_key_state:           # chỉ toggle khi vừa nhấn
    is_space_pressed = not is_space_pressed
_last_key_state = True
```

### 2.5 QoS Mismatch fix (`front ○` trong recorder)

**Lý do**: Realsense publish **RELIABLE**, recorder cũ subscribe **BEST_EFFORT** → không nhận data.
**Fix** trong `record_all.py`:

```python
qos_cam   = QoSProfile(reliability=RELIABLE,    depth=1)   # Realsense
qos_robot = QoSProfile(reliability=BEST_EFFORT, depth=1)   # ur_follow, gripper
```

### 2.6 Realsense USB 3.0 (story thực tế)

Lỗi `Frames didn't arrive within 5 seconds` dù cổng màu xanh.

**Nguyên nhân**: Cổng USB-A đi qua hub USB 2.0 nội bộ → Realsense thấy Bus 03 (480M).

**Verify**:
```bash
lsusb -t | grep uvcvideo
# Phải thấy 5000M+ (USB 3.0)
```

Trên Tiger Lake-H: Bus 04 (Thunderbolt 4, 20000M) là USB 3.0 thật. Bus 03 (480M) = USB 2.0.

### 2.7 pos_norm Gripper: dùng vô lăng, smooth 0→1

**Cũ**: Normalize theo vị trí tay kẹp (pos_slave) dựa trên limit cơ học hardcode → không chính xác.
**Mới**: Dùng **vị trí vô lăng (p_m)** để tính, smooth tuyến tính:

```python
MASTER_POS_OPEN  = -5.6    # rad → pos_norm = 0.0 (kẹp ra)
MASTER_POS_CLOSE = -1.3    # rad → pos_norm = 1.0 (kẹp vào)

pos_norm = (p_m - MASTER_POS_OPEN) / (MASTER_POS_CLOSE - MASTER_POS_OPEN)
```

Lý do smooth tốt hơn 0/1 cứng: Pi0.5 học trajectory liên tục → cần thấy gripper đóng dần từng bước.

### 2.8 record_all.py: 2 QoS + GUI thread riêng

- **2 QoS riêng**: `qos_cam` RELIABLE (Realsense), `qos_robot` BEST_EFFORT (robot/gripper)
- **GUI thread**: cv2.imshow ở thread riêng ~15fps, không block ROS spin
- **Default fps=20**: 10fps quá thưa cho Pi0.5

### 2.9 convert_hdf5_to_lerobot.py: bỏ lerobot.common

LeRobot v0.4.4+ xóa `lerobot.common.datasets`. Converter mới ghi parquet + MP4 + JSON trực tiếp.

### 2.10 ROS_LOCALHOST_ONLY + ROS_DOMAIN_ID

Thêm vào `~/.bashrc`:
```bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
```

Fix: DDS spam network errors + lỗi `Failed to find a free participant index`.

---

## 3. Phần Cứng

### Robot
- **Universal Robots UR3**, IP `192.168.1.1`
- Polyscope: Remote Control mode (không cần URCap)

### VR Tracking
- **2× HTC Vive Base Station 2.0** gắn tường
- **1× HTC Vive Tracker 3.0** cầm tay
- Không cần Headset — config null driver (xem 7.4)

### Camera
- **2× Intel RealSense D435I**
  - Front cam: serial `243322073847` → top-down nhìn xuống bàn
  - Wrist cam: serial `027422070272` → gắn cạnh gripper
  - Cả 2 cắm USB 3.0 Bus 04 (Port 4 + Port 8) trên máy

### Gripper Robstride
- **2× Robstride 06**:
  - ID=7 MASTER = vô lăng (tay người cầm xoay)
  - ID=6 SLAVE = tay kẹp (bám vô lăng qua PID Mimic)
- PID Mimic: kp=13.2, ki=0.5, kd=1.0
- **pos_norm range**: OPEN=-5.6 rad → 0.0 | CLOSE=-1.3 rad → 1.0

### USB-CAN
- **CANable2** slcan, bitrate 1Mbps (`-s8`), tự động detect `/dev/ttyACM*`

### Máy tính
- Ubuntu 22.04 LTS, ROS2 Humble
- Tiger Lake-H: Bus 04 (Thunderbolt 4, 20Gbps) — cổng USB 3.0 thật

---

## 4. Phần Mềm

### Python

```bash
# python3.10 (lerobot/ur_rtde cài ở đây, không phải python3.12 default)
python3.10 -m pip install \
  ur_rtde pynput python-can h5py \
  opencv-python pandas pyarrow \
  huggingface_hub lerobot openvr \
  "numpy<2"   # cv_bridge không tương thích NumPy 2.x
```

### ROS2

```bash
sudo apt install ros-humble-desktop \
  ros-humble-realsense2-camera \
  ros-humble-cv-bridge \
  ros-humble-tf-transformations \
  can-utils
```

---

## 5. Cấu Trúc Dự Án

```
~/ur5_teleop_vive/
├── README.md
└── ur5_teleop_vive/
    ├── CMakeLists.txt
    ├── package.xml
    ├── msg/Xyzrpy.msg
    ├── launch/view_ur5.launch.py
    ├── mesh/hand.dae
    └── ur5_teleop_vive/thesis_code/
        │
        ├── vive_tf_and_joy_ros2.py         ← OpenVR → TF + /vive_right Joy @90Hz
        │                                     Ctrl_R = TOGGLE (tap, không hold)
        │
        ├── frame_as_posestamped_ros2.py    ← TF → /right_controller_as_posestamped @60Hz
        │
        ├── vive_ur5_teleop_params.py       ← Apply world_alignment_matrix.txt
        │                                     → /ur_target_pose + /robot_origin_cmd
        │                                     Phím Home = robot moveL đến vị trí tracker
        │
        ├── ur_follow_using_class_ros2.py   ← UR3 RTDE controller
        │                                     ✅ ur_rtde (thay URBasic)
        │                                     ✅ servoL @100Hz (control_dt=0.010)
        │                                     ✅ 3 PRESET speed (đang dùng CÂN BẰNG: 5 m/s)
        │                                     ✅ IK precheck + workspace clamp (min_z=0.05)
        │                                     ✅ Glitch threshold 0.15m
        │                                     ✅ TCP_OFFSET=0.175m
        │                                     ✅ Publish: /joint_states /ur_joint_states
        │                                               /ur_actual_pose /ee_pose
        │                                               /ur_wrench /ur_joint_torque
        │
        ├── control_robstride_ros_without_calip.py  ← Gripper bilateral
        │                                     ✅ PID Mimic + Safety Lock
        │                                     ✅ pos_norm smooth theo vô lăng (p_m)
        │                                       OPEN=-5.6 rad → 0.0
        │                                       CLOSE=-1.3 rad → 1.0
        │                                     ✅ Publish /gripper/state (6 fields)
        │                                     ✅ CLI: --channel can0 --speed 1.5 --threshold 0.12
        │
        ├── launch_realsense_all.sh         ← Launch 2 Realsense đồng thời
        │                                     serial_front=243322073847
        │                                     serial_wrist=027422070272
        │                                     Profile: 640x480x30 (USB 3.0)
        │                                     Topics: /camera_{front,wrist}/camera/color/image_raw
        │
        ├── record_all.py                   ← HDF5 recorder
        │                                     ✅ 2 QoS: qos_cam RELIABLE, qos_robot BEST_EFFORT
        │                                     ✅ GUI cv2 thread riêng
        │                                     ✅ Default --fps 20
        │                                     TOPIC_FRONT = /camera_front/camera/color/image_raw
        │                                     TOPIC_WRIST = /camera_wrist/camera/color/image_raw
        │
        ├── camera_check.py                 ← Live preview 2 cam (cần sửa topic)
        │                                     TOPIC_FRONT = /camera_front/camera/color/image_raw
        │                                     TOPIC_WRIST = /camera_wrist/camera/color/image_raw
        │
        ├── check_hdf5.py                   ← Inspect dataset
        │                                     python3.10 check_hdf5.py dataset/pick_cube.hdf5
        │                                     python3.10 check_hdf5.py --demo 0 --save
        │
        ├── Convert_hdf5_to_lerobot.py      ← HDF5 → LeRobot v2 (không cần lerobot.common)
        ├── push_to_huggingface.py          ← Auto-detect + upload_folder
        ├── inference.py                    ← Pi0.5 trên robot (post-train)
        ├── calib_manual.py                 ← Calibration yaw (ghi world_alignment_*.txt)
        │
        ├── world_alignment_angle.txt       ← -30.490862° (std=0.043°)
        ├── world_alignment_matrix.txt      ← Ma trận 4×4 tự sinh
        │
        └── dataset/                        ← .gitignore
            └── pick_cube.hdf5
```

### ROS Topics — Luồng dữ liệu

```
vive_tf ──/tf──→ frame_as_posestamped ──/right_controller_as_posestamped──→ vive_teleop
vive_tf ──/vive_right (Joy)──────────────────────────────────────────────→ ur_follow

vive_teleop ──/ur_target_pose──→ ur_follow ──/ur_joint_states──→ record_all
vive_teleop ──/robot_origin_cmd─→ ur_follow ──/ur_actual_pose──→ record_all
                                             ──/ur_wrench───────→ (debug)

realsense (front) ──/camera_front/camera/color/image_raw──→ record_all
realsense (wrist) ──/camera_wrist/camera/color/image_raw──→ record_all

gripper ──/gripper/state──→ record_all
```

### /gripper/state fields

| Index | Field | Mô tả |
|---|---|---|
| 0 | pos_master | Vị trí vô lăng (rad) |
| 1 | pos_slave | Vị trí tay kẹp (rad) |
| 2 | **pos_norm** | 0.0=mở, 1.0=kẹp (Pi0.5 dùng) |
| 3 | torque | Torque tay kẹp (N·m) |
| 4 | contact | 0/1 — chạm vật |
| 5 | mode | 0=IDLE, 1=AUTO, 2=MIMIC |

---

## 6. Cài Đặt và Build

```bash
# Clone
git clone https://github.com/Khanhiot-ai/ur3-vive-pi05.git ~/ur5_teleop_vive
cd ~/ur5_teleop_vive

# Build
colcon build --packages-select ur5_teleop_vive --symlink-install
source install/setup.bash

# Auto-source
echo "source /opt/ros/humble/setup.bash"            >> ~/.bashrc
echo "source ~/ur5_teleop_vive/install/setup.bash"  >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1"                  >> ~/.bashrc
echo "export ROS_DOMAIN_ID=0"                       >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
ros2 interface show ur5_teleop_vive/msg/Xyzrpy
ping -c 3 192.168.1.1
ls /dev/ttyACM*
```

---

## 7. Setup Phần Cứng

### 7.1 Setup CAN cho Robstride

**Thủ công** (mỗi lần cắm CANable2):
```bash
# Tìm port
ls /dev/ttyACM*

# Bring up CAN interface
sudo slcand -o -c -s8 /dev/ttyACM0 can0   # ← đổi ttyACMX theo ls ở trên
sudo ip link set can0 up
sudo ip link set can0 txqueuelen 1000

# Kiểm tra
ip link show can0
# Phải thấy: state UP
```

**Script tự động** (khuyến nghị):
```bash
cat > ~/setup_can.sh << 'EOF'
#!/bin/bash
CANABLE=$(ls -t /dev/ttyACM* 2>/dev/null | head -1)
[ -z "$CANABLE" ] && echo "❌ Không thấy CANable2" && exit 1
echo "Found: $CANABLE"
sudo pkill slcand 2>/dev/null; sleep 0.5
sudo ip link set can0 down 2>/dev/null
sudo slcand -o -c -s8 $CANABLE can0
sleep 0.5
sudo ip link set can0 up
sudo ip link set can0 txqueuelen 1000
ip link show can0 | grep -q "state UP" && echo "✅ can0 UP!" || echo "❌ Fail"
EOF
chmod +x ~/setup_can.sh
```

### 7.2 Test motor Robstride

```bash
python3.10 check_robstride.py --scan
# → ✅ FOUND ID=6 và ID=7
```

### 7.3 Verify 2 Realsense USB 3.0

```bash
lsusb -t | grep uvcvideo
# Phải thấy CẢ 2 thiết bị ở 5000M trên Bus 04

rs-enumerate-devices | grep -E "Serial Number|Usb Type"
# Cả 2 phải: Usb Type Descriptor: 3.2
```

Cổng đúng trên máy này: **Bus 04 Port 4** và **Bus 04 Port 8** (Thunderbolt 4).

### 7.4 SteamVR không cần Headset

**File 1**:
```bash
nano ~/.local/share/Steam/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings
# Đổi "enable": false → true
```

**File 2**:
```bash
nano ~/.local/share/Steam/steamapps/common/SteamVR/resources/settings/default.vrsettings
# Sửa:
# "requireHmd": false,
# "forcedDriver": "null",
# "activateMultipleDrivers": true,
```

### 7.5 Verify gripper pos_norm

```bash
python3.10 control_robstride_ros_without_calip.py --channel can0
# Gõ 'm' → MIMIC ON
```

Terminal khác:
```bash
ros2 topic echo /gripper/state --field data
```

Xoay vô lăng từ mở hết → kẹp hết:
- `data[0]` (vô lăng) phải đi từ ~ **-5.6 → -1.3 rad**
- `data[2]` (pos_norm) phải đi từ **0.0 → 1.0** mượt mà

Nếu range khác → sửa 2 dòng trong file:
```python
MASTER_POS_OPEN  = -5.6    # ← đổi thành giá trị đo được khi mở hết
MASTER_POS_CLOSE = -1.3    # ← đổi thành giá trị đo được khi kẹp hết
```

---

## 8. Calibration (góc yaw)

### Tại sao chỉ cần yaw?

Vive lighthouse + UR3 đều đứng thẳng → trục Z song song → chỉ khác góc yaw quanh Z. Không cần Kabsch 4×4.

### Kết quả đã có

```
world_alignment_angle.txt: -30.490862°
Std dev: 0.043° → EXCELLENT (< 0.5°)
```

File đã có, không cần calib lại trừ khi di chuyển lighthouse hoặc robot.

### Calib lại khi cần

```bash
python3.10 calib_manual.py
# 1. Gắn tracker lên flange UR3
# 2. Dùng Teach Pendant di chuyển theo trục X+ (~10cm, 5 đoạn)
# 3. Script tính góc + lưu world_alignment_*.txt
```

| Std dev | Đánh giá |
|---|---|
| < 0.5° | Excellent |
| < 1.0° | Good |
| < 2.0° | Acceptable |
| > 2.0° | Calib lại |

---

## 9. Workflow Thu Data

### 9.1 Chuẩn bị

```bash
~/setup_can.sh
python3.10 check_robstride.py --scan   # → FOUND ID=6 và ID=7
```

### 9.2 Mở 7 terminals

**T1 — Vive bridge** (Ctrl_R TOGGLE):
```bash
python3.10 vive_tf_and_joy_ros2.py
```

**T2 — TF converter**:
```bash
python3.10 frame_as_posestamped_ros2.py
```

**T3 — Teleop logic**:
```bash
python3.10 vive_ur5_teleop_params.py
```

**T4 — UR3 RTDE**:
```bash
python3.10 ur_follow_using_class_ros2.py
```
Phải thấy:
```
║  Max Speed: 5.00 m/s                     ║
✓ RTDE connected: 192.168.1.1
✓ Control loop: 100 Hz (10.0ms)
SYSTEM READY TO CONTROL
```

**T5 — 2 Realsense**:
```bash
./launch_realsense_all.sh
```

**T6 — Gripper**:
```bash
python3.10 control_robstride_ros_without_calip.py --channel can0 --speed 1.5 --threshold 0.12
# Gõ 'm' + Enter → MIMIC ON
```

**T7 — Recorder**:
```bash
python3.10 record_all.py --task pick_cube --fps 20
```

### 9.3 Verify trước khi ghi

Status bar `record_all.py` phải hiện đủ 5 chấm xanh:
```
front ●#xxx  wrist ●#xxx  joints ●  target ●  grip ●
```

Check FPS (terminal khác):
```bash
ros2 topic hz /camera_front/camera/color/image_raw    # ~30Hz
ros2 topic hz /camera_wrist/camera/color/image_raw    # ~30Hz
ros2 topic hz /ur_actual_pose                         # ~100Hz
ros2 topic hz /gripper/state                          # ~100Hz
```

### 9.4 Phím điều khiển

| Phím | Tác dụng | Terminal focus |
|---|---|---|
| `Ctrl_R` (tap) | Toggle robot bám tracker ON/OFF | T1 |
| `Home` | Robot moveL đến vị trí tracker | T3 |
| `m` + Enter | Bật MIMIC gripper | T6 |
| `+`/`-` + Enter | Tăng/giảm speed gripper | T6 |
| `SPACE` | Bắt đầu / Dừng ghi | T7 |
| `S` | Lưu SUCCESS ✅ | T7 |
| `F` | Lưu FAIL ❌ | T7 |
| `Q` | Thoát | T7 |

### 9.5 Quy trình 1 episode (5-15s)

```
1. Đặt vật ở vị trí mới (đa dạng hóa mỗi demo)
2. Tap Ctrl_R OFF (robot đứng yên)
3. Bấm Home (T3 focus) → robot về vị trí tracker
4. T7: SPACE → 🔴 REC bắt đầu
5. Tap Ctrl_R ON → robot bám tracker
6. Di chuyển tracker đến vật
7. Xoay vô lăng → gripper đóng kẹp (data[2] → 1.0)
8. Di chuyển đến vị trí thả
9. Xoay vô lăng ngược → gripper mở (data[2] → 0.0)
10. Tap Ctrl_R OFF
11. T7: S (success) ✅ hoặc F (fail) ❌
```

### 9.6 Mục tiêu

| Demos | Pi0.5 success rate |
|---|---|
| 10 | Test pipeline |
| 50 | 30-50% |
| 100 | 60-75% |
| 200+ | 80-90% (thesis quality) |

---

## 10. Check Dataset HDF5

### Kiểm tra tổng quan

```bash
cd ~/ur5_teleop_vive/ur5_teleop_vive/ur5_teleop_vive/thesis_code

# Auto-detect file hdf5 trong dataset/
python3.10 check_hdf5.py

# Hoặc chỉ rõ file
python3.10 check_hdf5.py dataset/pick_cube.hdf5
```

Output hiện: số demo, success/fail, frames/duration, schema check, action sanity.

### Xem ảnh demo

```bash
# Lưu ảnh ra /tmp/ (5 frame đại diện)
python3.10 check_hdf5.py --demo 0 --save
xdg-open /tmp/demo0_frame0000_front.jpg
xdg-open /tmp/demo0_frame0000_wrist.jpg

# Xem trực tiếp cửa sổ cv2
python3.10 check_hdf5.py --demo 0
# Nhấn phím bất kỳ để xem frame tiếp
```

### Cảnh báo thường gặp

| Cảnh báo | Nguyên nhân | Fix |
|---|---|---|
| `XYZ không thay đổi` | Robot không di chuyển | Ctrl_R chưa ON |
| `Gripper không đóng` | MIMIC chưa bật | Gõ `m` trong T6 trước khi SPACE |
| `Episode > 60s` | Demo quá dài | Nên 5-15s/demo |
| `0 success demos` | Quên bấm S | Bấm S sau mỗi demo |
| `wrist_image MISSING` | Topic wrist sai | Check TOPIC_WRIST trong record_all.py |

---

## 11. Format Dataset HDF5

```
dataset/pick_cube.hdf5
└── data/                              attrs: task, fps=20, state_dim=7, action_dim=8
    ├── demo_0/                        attrs: success, n_frames, fps_actual, duration_s
    │   ├── obs/
    │   │   ├── image          uint8  (T, 224, 224, 3)   ← Realsense front
    │   │   ├── wrist_image    uint8  (T, 224, 224, 3)   ← Realsense wrist
    │   │   ├── state          f32    (T, 7)             ← joints[6] + gripper_norm[1]
    │   │   └── tactile_state  f32    (T, 1)             ← gripper torque
    │   └── actions            f32    (T, 8)             ← xyz + quat + gripper_cmd
    ├── demo_1/, demo_2/, ...
```

### Vai trò field cho Pi0.5

| Field | Mục đích | Quan trọng |
|---|---|---|
| `image` | Vision encoder — scene tổng thể | ★★★ |
| `wrist_image` | Vision encoder — close-up gripper | ★★★ |
| `state` | Proprioception robot | ★★★ |
| `actions` | **Ground truth** Pi0.5 predict | ★★★ |
| `tactile_state` | Force feedback | ★★ |

---

## 12. Convert + Push HuggingFace

### Convert HDF5 → LeRobot v2

```bash
python3.10 Convert_hdf5_to_lerobot.py \
  --src dataset/pick_cube.hdf5 \
  --task "pick up the red cube" \
  --fps 20 \
  --image-size 224 \
  --skip-failed \
  --overwrite
```

Output `dataset/pick_cube_lerobot/`:
```
meta/{info.json, episodes.jsonl, tasks.jsonl, stats.json}
data/chunk-000/episode_NNNNNN.parquet
videos/chunk-000/observation.{image,wrist_image}_episode_NNNNNN.mp4
```

### Push lên HuggingFace

```bash
# Login 1 lần
huggingface-cli login
huggingface-cli whoami    # → qkhanh1

# Push
python3.10 push_to_huggingface.py --repo-id qkhanh1/ur3_pick_cube
```

---

## 13. Inference Pi0.5 trên Robot

### Train trên Colab/GPU

```python
!pip install lerobot openpi
from huggingface_hub import login
login(token="hf_xxx")

!python -m openpi.train \
  --dataset qkhanh1/ur3_pick_cube \
  --model pi0.5-base \
  --epochs 50

from huggingface_hub import upload_folder
upload_folder(folder_path="./checkpoints/final",
              repo_id="qkhanh1/pi0.5_ur3_pickcube",
              repo_type="model")
```

### Inference trên robot thật

```bash
# Download checkpoint
huggingface-cli download qkhanh1/pi0.5_ur3_pickcube \
  --local-dir ~/checkpoints/pi0.5_ur3_pickcube

# Chạy pipeline T1-T6 như thu data (bỏ T3 vive_teleop)
# T7: inference thay record_all
python3.10 inference.py \
  --checkpoint ~/checkpoints/pi0.5_ur3_pickcube \
  --task "pick up the red cube"
```

⚠️ Đổi sang PRESET AN TOÀN trước khi test Pi0.5 (sửa `ur_follow_using_class_ros2.py` dòng 80-83):
```python
self.normal_max_speed = 0.5    # PRESET AN TOÀN
```

---

## 14. Troubleshooting

### Robot UR3

| Lỗi | Fix |
|---|---|
| `get_inverse_kin failed` | IK precheck + workspace clamp đã có trong file mới |
| `RTDE connection failed` | Robot chưa ở Remote Control mode |
| `Protective stop` | Restart trên teach pendant |
| Robot quá nhanh/mạnh | Đổi sang PRESET AN TOÀN (max_speed=0.5) |
| Robot lao 1 hướng | Tap Ctrl_R OFF, kiểm tra calib |

### Camera Realsense

| Lỗi | Fix |
|---|---|
| `Frames didn't arrive` | USB 2.0 — cắm Bus 04 Port 4/8 |
| `Device busy (errno=16)` | Có process khác giữ camera — `pkill -9 -f realsense2_camera_node` |
| FPS chỉ 0.5Hz | Profile không support hoặc bandwidth — thử `1280x720x30` |
| Serial not found | `rs-enumerate-devices -s` xem serial đúng |
| Serial type mismatch | Bọc serial trong `\"...\"` trong bash script |
| Topic `/camera_front/camera_front_node/...` | Bỏ `-r __node:=...` trong launch script |

### Camera C922 (nếu còn dùng)

| Lỗi | Fix |
|---|---|
| `pixel_format unsupported` | Không dùng usb_cam — dùng `wrist_cam.py` (OpenCV) |
| Device không tìm thấy | `wrist_cam.py` có auto-detect qua v4l2-ctl |

### QoS / Topic

| Lỗi | Fix |
|---|---|
| `front ○` trong recorder | Realsense RELIABLE, recorder cũ BEST_EFFORT — dùng file mới có 2 QoS |
| `RCLError: rmw handle is invalid` | Terminal chưa source ROS2 — `source ~/.bashrc` |
| `Failed to find a free participant index` | Quá nhiều ROS2 node — `pkill -f ros2`, thêm `ROS_DOMAIN_ID=0` |

### Vive / SteamVR

| Lỗi | Fix |
|---|---|
| Tracker xám | Giữ nút tracker 3s bật lại, check pin |
| SteamVR đòi headset | Sửa 2 file vrsettings (7.4) |
| Robot không bám tracker | Click vào terminal T1, xem log `[TOGGLE] Trigger ON 🟢` |

### Gripper Robstride

| Lỗi | Fix |
|---|---|
| `can3: No such device` | Sai interface — dùng `--channel can0` |
| `can0 not found` | CAN chưa setup — chạy `~/setup_can.sh` |
| Motor 6 không respond | Check dây CAN daisy-chain |
| `/gripper/state` không publish | Dùng đúng file `control_robstride_ros_without_calip.py` |
| `pos_norm` không về 1.0 khi kẹp | Sửa `MASTER_POS_CLOSE` theo giá trị đo thực |
| `pos_norm` stuck ở 0.0 | Sửa `MASTER_POS_OPEN` theo giá trị đo thực |

### CAN bus

| Lỗi | Fix |
|---|---|
| `slcand: device busy` | `sudo pkill slcand` rồi setup lại |
| `Permission denied /dev/ttyACM*` | `sudo usermod -aG dialout $USER` rồi logout/login |
| CANable ở `/dev/ttyACM3` không phải `ttyACM0` | Script `setup_can.sh` tự tìm `ttyACM*` đầu tiên |

### HDF5 / Recorder

| Lỗi | Fix |
|---|---|
| GUI đơ | GUI thread riêng đã có trong file mới |
| `gripper = 0` toàn bộ demo | MIMIC chưa bật — gõ `m` trong T6 trước SPACE |
| `XYZ range = 0` | Ctrl_R chưa ON khi ghi |
| Demo 60s+ | Mỗi demo 5-15s — 1 hành động pick+place duy nhất |

### Convert / Push

| Lỗi | Fix |
|---|---|
| `ModuleNotFoundError: lerobot.common` | Dùng `Convert_hdf5_to_lerobot.py` mới |
| Push "Auth" fail | `huggingface-cli login` lại với Write token |
| `python3` không có lerobot | Dùng `python3.10` |

### Git

| Lỗi | Fix |
|---|---|
| File `ur_log/*.log` (1.4GB) reject | `.gitignore` add `ur_log/`, `dataset/`, `*.log` |
| Push reject > 50MB | `git filter-branch --force --index-filter "git rm --cached path"` |

---

## 📝 References

- **Pi0.5**: https://www.physicalintelligence.company
- **openpi**: https://github.com/Physical-Intelligence/openpi
- **LeRobot**: https://github.com/huggingface/lerobot
- **ur_rtde**: https://sdurobotics.gitlab.io/ur_rtde/
- **SteamVR no-headset**: https://github.com/username223/SteamVRNoHeadset

---

*README v7 — May 2026 — qkhanh1 (UR3 thesis)*

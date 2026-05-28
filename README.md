# UR3 Vive Teleop + Pi0.5 Dataset System

Hệ thống điều khiển **UR3** bằng **HTC Vive Tracker** + gripper **Robstride**, thu data đa-modal HDF5 để fine-tune **Pi0.5 VLA** cho task pick & place.

Format dataset **tương thích 100%** với `lerobot/berkeley_autolab_ur5` — sẵn sàng fine-tune Pi0.5.

🔗 **Repo**: https://github.com/Khanhiot-ai/ur3-vive-pi05
📦 **Dataset**: https://huggingface.co/datasets/qkhanh1/ur3_pick_cube
📚 **Tham khảo**: https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5

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
│ → 1 file HDF5      │    │ → Checkpoint   │    │ → Robot tự làm  │
└──────────────┬─────┘    └────────┬───────┘    └─────────────────┘
               │                   │                      ▲
               └── HuggingFace ────┴──────────────────────┘
                   (qkhanh1/ur3_pick_cube)
```

### Schema Dataset — Giống Berkeley AutoLab UR5

```
observation.state  (8 dim) = [ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper]
action             (7 dim) = [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
observation.image       (480, 640, 3) — Front cam (top-down)
observation.wrist_image (480, 640, 3) — Wrist cam (cạnh gripper)
```

- **state**: Cartesian TCP pose (xyz + quaternion) + gripper [0=mở, 1=kẹp]
- **action**: **DELTA Cartesian** (xyz delta + RPY delta) + gripper
- **image**: 480×640 RGB (giống berkeley 100%)

### Pipeline 7 terminal

| T | Node | Vai trò |
|---|---|---|
| 1 | `vive_tf_and_joy_ros2.py` | OpenVR → /tf + Joy (Ctrl_R toggle) |
| 2 | `frame_as_posestamped_ros2.py` | TF → PoseStamped @60Hz |
| 3 | `vive_ur5_teleop_params.py` | Apply yaw alignment → `/ur_target_pose` |
| 4 | `ur_follow_using_class_ros2.py` | RTDE servoL @100Hz |
| 5 | `./launch_realsense_all.sh` | 2 Realsense D435I đồng thời |
| 6 | `control_robstride_ros_without_calip.py` | Gripper Mimic PID + `/gripper/state` |
| 7 | `record_all.py` | HDF5 recorder (1 file cho tất cả demo) |

---

## 2. Changelog — Các Fix Đã Làm

### 2.1 Driver UR: URBasic → ur_rtde

URBasic hay treo, lỗi `get_inverse_kin failed`, latency cao. ur_rtde servoL @100Hz mượt hơn, force/torque feedback dễ lấy.

### 2.2 Calibration: 4×4 Kabsch → góc yaw đơn giản

Kabsch 8 điểm 3D phức tạp, dễ sai khi điểm gần đồng phẳng. Vive lighthouse + UR3 đều đứng thẳng → chỉ khác góc yaw quanh Z → đo 1 số duy nhất (vd: -30.49°) là đủ.

### 2.3 Camera: C922 webcam → 2× Realsense D435I

`usb_cam` crash với mọi `pixel_format`. C922 chỉ USB 2.0. Đổi sang 2 Realsense:
- Front: serial `243322073847`
- Wrist: serial `027422070272`
- Cả 2 cắm USB 3.0 Bus 04 (Thunderbolt 4)

Vấn đề đã gặp khi setup:
- Serial dạng số thuần → ROS2 cast sang integer → **fix**: bọc `\"...\"` trong bash
- `rs-enumerate-devices -s` có ERROR log → `awk` parse sai → **fix**: `grep -oE "[0-9]{12}"`
- Profile `640x480x30` đôi khi báo invalid → fallback `1280x720x30`
- Realsense #2 cắm cổng USB-A xanh nhưng vẫn ở Bus 03 (USB 2.0) → **fix**: thử cổng vật lý khác cho đến khi vào Bus 04

### 2.4 Ctrl_R Toggle (tap thay vì hold)

Giữ phím liên tục mỏi tay, nhả lỡ là robot dừng giữa demo. Edge detection chống OS autorepeat:

```python
if not _last_key_state:
    is_space_pressed = not is_space_pressed
_last_key_state = True
```

### 2.5 QoS Mismatch fix (`front ○` trong recorder)

Realsense publish **RELIABLE**, recorder cũ subscribe **BEST_EFFORT** → không nhận data. Fix 2 QoS riêng:

```python
qos_cam   = QoSProfile(reliability=RELIABLE,    depth=1)   # Realsense
qos_robot = QoSProfile(reliability=BEST_EFFORT, depth=1)   # ur_follow, gripper
```

### 2.6 Realsense USB 3.0

Lỗi `Frames didn't arrive within 5 seconds` dù cổng nhìn xanh. Nguyên nhân: cổng USB-A đi qua hub USB 2.0 nội bộ → Realsense thấy Bus 03 (480M).

**Verify**:
```bash
lsusb -t | grep uvcvideo   # Phải thấy 5000M+
```

Trên Tiger Lake-H: **Bus 04** (Thunderbolt 4, 20000M) là USB 3.0 thật.

### 2.7 pos_norm Gripper: dùng vô lăng, smooth 0→1

Cũ: normalize theo vị trí tay kẹp (pos_slave) — không chính xác khi kẹp vật.
Mới: dùng **vị trí vô lăng (p_m)** — smooth tuyến tính 0→1:

```python
MASTER_POS_OPEN  = -5.6    # rad → pos_norm = 0.0 (kẹp ra)
MASTER_POS_CLOSE = -1.3    # rad → pos_norm = 1.0 (kẹp vào)
```

Pi0.5 học trajectory liên tục → smooth tốt hơn 0/1 cứng.

### 2.8 Dataset format: chuyển sang Berkeley AutoLab UR5

**Lý do**: berkeley_autolab_ur5 là dataset chuẩn cho UR robot. Pi0.5 đã train trên đó → fine-tune dễ hơn.

| | Cũ | Berkeley (mới) |
|---|---|---|
| state | 7 dim joints | **8 dim** Cartesian + grip |
| action | 8 dim xyz+quat (abs) | **7 dim** delta xyz+rpy (delta) |
| image | 224×224 | **480×640** |

**Delta action** = `actual_current - actual_prev`. Frame đầu mỗi episode = 0.

### 2.9 record_all.py: GUI thread riêng + default fps=20

- **3 thread riêng**: ROS spin, GUI cv2, keyboard input — không block nhau
- **Default fps=20**: 10fps quá thưa cho Pi0.5
- **1 file HDF5**: tất cả demo nối vào `dataset/<task>.hdf5`, không tách file

### 2.10 convert_hdf5_to_lerobot.py: bỏ lerobot.common

LeRobot v0.4.4+ xóa `lerobot.common.datasets`. Converter ghi parquet + MP4 + JSON trực tiếp qua `huggingface_hub` + `pandas` + `pyarrow`.

### 2.11 ROS_LOCALHOST_ONLY + ROS_DOMAIN_ID

Thêm vào `~/.bashrc`:
```bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
```

Fix DDS spam network errors + lỗi `Failed to find a free participant index`.

---

## 3. Phần Cứng

### Robot
- **Universal Robots UR3**, IP `192.168.1.1`
- Polyscope: Remote Control mode (không cần URCap)

### VR Tracking
- **2× HTC Vive Base Station 2.0**
- **1× HTC Vive Tracker 3.0**
- Không cần Headset — config null driver (xem 7.4)

### Camera (2× Intel RealSense D435I)
- Front: `243322073847` → top-down (treo cao nhìn xuống bàn)
- Wrist: `027422070272` → gắn cạnh gripper
- Cả 2 cắm USB 3.0 Bus 04 Port 4 + Port 8

### Gripper Robstride
- **2× Robstride 06**:
  - ID=7 MASTER = vô lăng
  - ID=6 SLAVE = tay kẹp (bám vô lăng qua PID Mimic)
- pos_norm: OPEN=-5.6 rad → 0.0 | CLOSE=-1.3 rad → 1.0

### USB-CAN
- **CANable2** slcan, bitrate 1Mbps

### Máy tính
- Ubuntu 22.04 LTS, ROS2 Humble
- Tiger Lake-H: Bus 04 (Thunderbolt 4, 20Gbps) — USB 3.0 thật

---

## 4. Phần Mềm

### Python (cài ở python3.10)

```bash
python3.10 -m pip install \
  ur_rtde pynput python-can h5py \
  opencv-python pandas pyarrow \
  huggingface_hub lerobot openvr \
  "numpy<2"
```

### ROS2 + tools

```bash
sudo apt install ros-humble-desktop \
  ros-humble-realsense2-camera \
  ros-humble-cv-bridge \
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
    └── ur5_teleop_vive/thesis_code/
        ├── vive_tf_and_joy_ros2.py             ← OpenVR → /vive_right Joy
        ├── frame_as_posestamped_ros2.py        ← TF → PoseStamped @60Hz
        ├── vive_ur5_teleop_params.py           ← /ur_target_pose + /robot_origin_cmd
        ├── ur_follow_using_class_ros2.py       ← RTDE servoL @100Hz
        ├── control_robstride_ros_without_calip.py  ← Gripper Mimic + /gripper/state
        ├── launch_realsense_all.sh             ← 2 Realsense đồng thời
        │
        ├── record_all.py                       ← HDF5 recorder
        │                                         ✅ State 8 dim, Action 7 dim (delta)
        │                                         ✅ Image 480×640 (berkeley)
        │                                         ✅ 2 QoS riêng
        │
        ├── camera_check.py                     ← Live preview 2 cam
        ├── check_hdf5.py                       ← Inspect dataset
        ├── Convert_hdf5_to_lerobot.py          ← HDF5 → LeRobot v2 (berkeley schema)
        ├── push_to_huggingface.py              ← Upload Hub
        ├── inference.py                        ← Pi0.5 post-train
        ├── calib_manual.py                     ← Yaw calibration
        │
        ├── world_alignment_angle.txt           ← -30.490862° (std=0.043°)
        ├── world_alignment_matrix.txt          ← Tự sinh từ góc
        │
        └── dataset/                            ← .gitignore
            └── pick_cube.hdf5                  ← 1 file chứa TẤT CẢ demo
```

### ROS Topics

| Topic | Type | Publisher | Subscriber |
|---|---|---|---|
| `/right_controller_as_posestamped` | PoseStamped | frame_as_posestamped | vive_teleop |
| `/vive_right` | Joy | vive_tf | ur_follow |
| `/ur_target_pose` | PoseStamped | vive_teleop | ur_follow + record_all |
| `/robot_origin_cmd` | Pose | vive_teleop | ur_follow |
| `/ur_actual_pose` | Xyzrpy | ur_follow | **record_all** (cho state + delta action) |
| `/ur_joint_states` | JointState | ur_follow | record_all |
| `/gripper/state` | Float32MultiArray[6] | control_robstride | record_all |
| `/camera_front/camera/color/image_raw` | Image | realsense | record_all |
| `/camera_wrist/camera/color/image_raw` | Image | realsense | record_all |

### /gripper/state fields

| Index | Field | Mô tả |
|---|---|---|
| 0 | pos_master | Vị trí vô lăng (rad) |
| 1 | pos_slave | Vị trí tay kẹp (rad) |
| 2 | **pos_norm** | 0.0=mở, 1.0=kẹp (Pi0.5 dùng) |
| 3 | torque | Torque tay kẹp (N·m) |
| 4 | contact | 0/1 |
| 5 | mode | 0=IDLE, 1=AUTO, 2=MIMIC |

---

## 6. Cài Đặt và Build

```bash
git clone https://github.com/Khanhiot-ai/ur3-vive-pi05.git ~/ur5_teleop_vive
cd ~/ur5_teleop_vive

colcon build --packages-select ur5_teleop_vive --symlink-install
source install/setup.bash

# Auto-source mỗi terminal
cat >> ~/.bashrc << 'EOF'
source /opt/ros/humble/setup.bash
source ~/ur5_teleop_vive/install/setup.bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
EOF
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

```bash
# Tự động — script tìm /dev/ttyACMX
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
~/setup_can.sh
```

### 7.2 Test motor Robstride

```bash
python3.10 check_robstride.py --scan
# → ✅ FOUND ID=6 và ID=7
```

### 7.3 Verify 2 Realsense USB 3.0

```bash
rs-enumerate-devices | grep -E "Serial Number|Usb Type"
# Cả 2 phải: Usb Type Descriptor: 3.2

lsusb -t | grep uvcvideo
# Cả 2 phải 5000M trên Bus 04
```

### 7.4 SteamVR không cần Headset

Sửa 2 file vrsettings:

```bash
# File 1: bật null driver
nano ~/.local/share/Steam/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings
# Đổi "enable": false → true

# File 2: tắt requireHmd
nano ~/.local/share/Steam/steamapps/common/SteamVR/resources/settings/default.vrsettings
# "requireHmd": false, "forcedDriver": "null", "activateMultipleDrivers": true,
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

Xoay vô lăng mở hết → kẹp hết:
- `data[0]` (vô lăng): từ ~ **-5.6 → -1.3 rad**
- `data[2]` (pos_norm): từ **0.0 → 1.0** mượt

Nếu range khác → sửa 2 dòng trong file:
```python
MASTER_POS_OPEN  = -5.6    # giá trị đo được khi mở hết
MASTER_POS_CLOSE = -1.3    # giá trị đo được khi kẹp hết
```

---

## 8. Calibration (góc yaw)

### Kết quả đã có

```
world_alignment_angle.txt: -30.490862°
Std dev: 0.043° → EXCELLENT
```

Không cần calib lại trừ khi di chuyển lighthouse/robot.

### Calib lại

```bash
python3.10 calib_manual.py
# 1. Gắn tracker lên flange UR3
# 2. Di chuyển robot theo trục X+ (~10cm, 5 đoạn)
# 3. Script lưu world_alignment_*.txt
```

---

## 9. Workflow Thu Data

### 9.1 Bật pipeline (7 terminals)

```bash
# T0: CAN setup (1 lần)
~/setup_can.sh

# T1: Vive bridge
python3.10 vive_tf_and_joy_ros2.py

# T2: TF converter
python3.10 frame_as_posestamped_ros2.py

# T3: Teleop logic
python3.10 vive_ur5_teleop_params.py

# T4: UR3 RTDE
python3.10 ur_follow_using_class_ros2.py

# T5: 2 Realsense
./launch_realsense_all.sh

# T6: Gripper (gõ 'm' + Enter sau khi khởi động)
python3.10 control_robstride_ros_without_calip.py --channel can0

# T7: Recorder
python3.10 record_all.py --task pick_cube --fps 20
```

### 9.2 Verify 5 chấm xanh

Status bar trong recorder:
```
front ●  wrist ●  actual ●  target ●  grip ●
```

Cả 5 phải `●` (xanh) — nếu còn `○` thì topic đó chưa publish.

Check FPS:
```bash
ros2 topic hz /camera_front/camera/color/image_raw    # ~30Hz
ros2 topic hz /camera_wrist/camera/color/image_raw    # ~30Hz
ros2 topic hz /ur_actual_pose                         # ~100Hz
ros2 topic hz /gripper/state                          # ~100Hz
```

### 9.3 Phím điều khiển

| Phím | Tác dụng | Terminal |
|---|---|---|
| `Ctrl_R` (tap) | Toggle robot bám tracker ON/OFF | T1 |
| `Home` | Robot moveL đến vị trí tracker | T3 |
| `m` + Enter | Bật MIMIC gripper | T6 |
| `+`/`-` + Enter | Tăng/giảm speed gripper | T6 |
| `SPACE` | Bắt đầu / Dừng ghi | T7 |
| `S` | Lưu SUCCESS ✅ | T7 |
| `F` | Lưu FAIL ❌ | T7 |
| `Q` | Thoát | T7 |

### 9.4 Thu 50 demo — quy trình lặp lại

**Cứ lặp đi lặp lại trên 1 terminal T7**:

```
SPACE → làm demo (5-15s) → SPACE → S    ← demo_0
SPACE → làm demo            → SPACE → S    ← demo_1
SPACE → làm demo            → SPACE → S    ← demo_2
...
SPACE → làm demo            → SPACE → S    ← demo_49
Q                                          ← Thoát
```

**Tất cả demo nối vào CÙNG 1 file** `dataset/pick_cube.hdf5`:

```
pick_cube.hdf5
├── demo_0   (success)
├── demo_1   (success)
├── demo_2   (fail)      ← tự skip khi convert
├── demo_3   (success)
└── ...
```

**Tip**: Có thể bấm `S` trực tiếp (không cần SPACE trước) — recorder tự dừng + lưu luôn.

### 9.5 Mỗi episode (5-15s, 1 hành động pick+place)

```
1. Đặt vật ở vị trí mới (đa dạng hóa!)
2. Tap Ctrl_R OFF (robot đứng yên)
3. Bấm Home → robot về vị trí tracker
4. T7: SPACE → 🔴 REC
5. Tap Ctrl_R ON → robot bám tracker
6. Di chuyển tracker đến vật
7. Xoay vô lăng → gripper đóng (data[2] → 1.0)
8. Di chuyển đến vị trí thả
9. Xoay vô lăng ngược → gripper mở (data[2] → 0.0)
10. Tap Ctrl_R OFF
11. T7: S (success) hoặc F (fail)
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

Sau mỗi 10 demo nên check:

```bash
cd ~/ur5_teleop_vive/ur5_teleop_vive/ur5_teleop_vive/thesis_code

python3.10 check_hdf5.py
# Hoặc:
python3.10 check_hdf5.py dataset/pick_cube.hdf5
```

Output mong đợi:
```
✅ obs/image:       (T, 480, 640, 3)  uint8
✅ obs/wrist_image: (T, 480, 640, 3)  uint8
✅ obs/state:       (T, 8)            float32
✅ actions:         (T, 7)            float32
   XYZ range: [0.15, 0.20, 0.10] m   ← robot có di chuyển
   Quaternion norm OK
   Gripper: min=0.0  max=1.0          ← kẹp hết hành trình
```

Xem ảnh demo cụ thể:
```bash
python3.10 check_hdf5.py --demo 0 --save
xdg-open /tmp/demo0_frame0000_front.jpg
```

### Cảnh báo thường gặp

| Cảnh báo | Nguyên nhân | Fix |
|---|---|---|
| `XYZ không thay đổi` | Robot đứng yên | Ctrl_R chưa ON |
| `Gripper max < 1.0` | Vô lăng chưa kẹp đến -1.3 rad | Xoay vô lăng đến hết |
| `Episode > 30s` | Demo quá dài | Mỗi demo 5-15s |
| `wrist_image MISSING` | Topic wrist sai | Sửa TOPIC_WRIST trong record_all.py |

---

## 11. Format Dataset HDF5

```
dataset/pick_cube.hdf5
└── data/                                attrs: task, fps=20, state_dim=8, action_dim=7
    ├── demo_0/                          attrs: success, n_frames, fps_actual, duration_s
    │   ├── obs/
    │   │   ├── image          uint8  (T, 480, 640, 3)   ← Front cam
    │   │   ├── wrist_image    uint8  (T, 480, 640, 3)   ← Wrist cam
    │   │   ├── state          f32    (T, 8)             ← TCP + grip
    │   │   └── tactile_state  f32    (T, 1)             ← torque
    │   └── actions            f32    (T, 7)             ← DELTA Cartesian
    └── demo_N/...
```

### State (8 dim) — Cartesian TCP

```
[ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper]
 ─── m ────────── │ ─ quaternion ─ │ 0-1
```

### Action (7 dim) — DELTA Cartesian

```
[dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
 ─── Δm ─── │ ──── Δrad ─────────│ 0-1
```

Tính delta: `action[t] = actual[t] - actual[t-1]`. Frame đầu mỗi episode = 0.

### So với berkeley_autolab_ur5

| Field | Bạn | Berkeley | ✓ |
|---|---|---|---|
| state shape | (8,) | (8,) | ✅ |
| state names | ee_xyz+quat+grip | motor_0..7 | (giống số chiều) |
| action shape | (7,) | (7,) | ✅ |
| action names | dx,dy,dz,drpy,grip | motor_0..6 | (giống số chiều) |
| image shape | (480, 640, 3) | (480, 640, 3) | ✅ |
| wrist_image | có | có | ✅ |
| fps | 20 | 5 | bạn cao hơn |

---

## 12. Convert + Push HuggingFace

### Workflow đơn giản (3 lệnh)

Sau khi thu xong **1 file HDF5** chứa tất cả demo:

```bash
cd ~/ur5_teleop_vive/ur5_teleop_vive/ur5_teleop_vive/thesis_code

# 1. Check lần cuối
python3.10 check_hdf5.py dataset/pick_cube.hdf5

# 2. Convert → LeRobot v2
python3.10 Convert_hdf5_to_lerobot.py \
  --src dataset/pick_cube.hdf5 \
  --task "pick up the red cube" \
  --fps 20 \
  --skip-failed \
  --overwrite

# 3. Push lên HuggingFace
python3.10 push_to_huggingface.py --repo-id qkhanh1/ur3_pick_cube
```

### Output sau Convert

```
dataset/pick_cube_lerobot/
├── meta/
│   ├── info.json       ← schema giống berkeley
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── stats.json      ← mean/std normalization
├── data/chunk-000/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
└── videos/chunk-000/
    ├── observation.image_episode_000000.mp4
    ├── observation.wrist_image_episode_000000.mp4
    └── ...
```

### Setup HuggingFace (1 lần)

```bash
huggingface-cli login   # paste Write token
huggingface-cli whoami  # → qkhanh1
```

### Dung lượng dự kiến

Với 480×640 (giống berkeley):

| Demos | Frames | Size |
|---|---|---|
| 50 demos (×10s) | 10k | ~2.2 GB |
| 100 demos | 20k | ~4.5 GB |

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

# Chạy T1, T2, T4, T5, T6 như thu data (KHÔNG cần T3 vive_teleop, T7 record)
# Thêm T7: inference
python3.10 inference.py \
  --checkpoint ~/checkpoints/pi0.5_ur3_pickcube \
  --task "pick up the red cube"
```

⚠️ **An toàn**: Đổi sang PRESET AN TOÀN trước khi test:
```python
# Trong ur_follow_using_class_ros2.py dòng 80-83
self.normal_max_speed = 0.5    # ← giảm từ 5 xuống 0.5 m/s
```

Tay ở E-Stop pendant UR3.

---

## 14. Troubleshooting

### Robot UR3

| Lỗi | Fix |
|---|---|
| `get_inverse_kin failed` | IK precheck + workspace clamp đã có |
| `RTDE connection failed` | Robot chưa Remote Control mode |
| `Protective stop` | Restart trên teach pendant |
| Robot quá nhanh | Đổi PRESET AN TOÀN (max_speed=0.5) |

### Camera Realsense

| Lỗi | Fix |
|---|---|
| `Frames didn't arrive` | USB 2.0 — cắm Bus 04 Port 4/8 |
| `Device busy (errno=16)` | `pkill -9 -f realsense2_camera_node` |
| FPS chỉ 0.5Hz | Profile không support — thử `1280x720x30` |
| Serial not found | `rs-enumerate-devices -s` xem serial đúng |
| Serial integer error | Bọc serial trong `\"...\"` trong bash |
| Topic có `camera_front_node` | Bỏ `-r __node:=...` trong launch script |

### QoS / Topic

| Lỗi | Fix |
|---|---|
| `front ○` trong recorder | Recorder mới đã có 2 QoS riêng |
| `RCLError: rmw handle invalid` | Terminal chưa source ROS2 |
| `Failed to find a free participant index` | `pkill -f ros2` + `ROS_DOMAIN_ID=0` |

### Vive / SteamVR

| Lỗi | Fix |
|---|---|
| Tracker xám | Giữ nút tracker 3s, check pin |
| SteamVR đòi headset | Sửa 2 file vrsettings (7.4) |
| Robot không bám tracker | Click vào T1 trước khi tap Ctrl_R |

### Gripper Robstride

| Lỗi | Fix |
|---|---|
| `can3: No such device` | Dùng `--channel can0` |
| `can0 not found` | Chạy `~/setup_can.sh` |
| `/gripper/state` không publish | Dùng `control_robstride_ros_without_calip.py` |
| `pos_norm` không về 1.0 | Sửa `MASTER_POS_CLOSE` theo giá trị đo thực |
| `pos_norm` stuck 0.0 | Sửa `MASTER_POS_OPEN` theo giá trị đo thực |

### CAN bus

| Lỗi | Fix |
|---|---|
| `slcand: device busy` | `sudo pkill slcand` rồi setup lại |
| `Permission denied /dev/ttyACM*` | `sudo usermod -aG dialout $USER` rồi logout/login |

### HDF5 / Recorder

| Lỗi | Fix |
|---|---|
| GUI đơ | GUI thread riêng đã có trong file mới |
| Buffer rỗng | Camera/actual_pose chưa publish |
| `gripper = 0` toàn bộ demo | MIMIC chưa bật — gõ `m` trong T6 |
| `XYZ range = 0` | Ctrl_R chưa ON |
| Demo 60s+ | 1 hành động pick+place chỉ 5-15s |
| Action toàn 0 | `_prev_actual` chưa reset (đã fix trong file mới) |

### Convert / Push

| Lỗi | Fix |
|---|---|
| `ModuleNotFoundError: lerobot.common` | Dùng Converter mới (bỏ lerobot.common) |
| Push "Auth" fail | `huggingface-cli login` lại |
| `python3` không có lerobot | Dùng `python3.10` |

### Git

| Lỗi | Fix |
|---|---|
| File `*.log` (1.4GB) reject | `.gitignore` add `ur_log/`, `dataset/`, `*.log` |
| Push reject >50MB | `git filter-branch --force --index-filter "git rm --cached path"` |

---

## 📝 References

- **Pi0.5**: https://www.physicalintelligence.company
- **openpi**: https://github.com/Physical-Intelligence/openpi
- **LeRobot**: https://github.com/huggingface/lerobot
- **Berkeley AutoLab UR5 dataset**: https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5
- **ur_rtde**: https://sdurobotics.gitlab.io/ur_rtde/
- **SteamVR no-headset**: https://github.com/username223/SteamVRNoHeadset

---

*README v8 — May 2026 — qkhanh1 (UR3 thesis project)*

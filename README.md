# UR3 Vive Teleop + Pi0.5 Dataset System

Hệ thống điều khiển UR3 bằng HTC Vive Tracker, thu data demo cho fine-tune **Pi0.5** (Vision-Language-Action model).

🔗 **Repo**: https://github.com/Khanhiot-ai/ur3-vive-pi05

---

## 📋 Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Phần Cứng](#2-phần-cứng)
3. [Phần Mềm](#3-phần-mềm)
4. [Cấu Trúc Dự Án](#4-cấu-trúc-dự-án)
5. [Tạo ROS2 Package Từ Đầu](#5-tạo-ros2-package-từ-đầu)
6. [Cài Đặt và Build](#6-cài-đặt-và-build)
7. [Setup Phần Cứng](#7-setup-phần-cứng)
8. [Calibration](#8-calibration)
9. [Workflow Thu Data](#9-workflow-thu-data)
10. [Format Dataset HDF5](#10-format-dataset-hdf5)
11. [Convert sang LeRobot](#11-convert-sang-lerobot)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Tổng Quan

### Mục tiêu

Thu thập dữ liệu demonstration cho UR3 thực hiện task gắp vật, để fine-tune Pi0.5.

### Cách hoạt động

```
HTC Vive Tracker (tay cầm)
        ↓
    OpenVR → ROS2 TF
        ↓
    Calibration → Robot target pose
        ↓
    ur_rtde → UR3 di chuyển
        ↓
    Robstride gripper (vô lăng + tay kẹp, bilateral)
        ↓
    Record: 2 camera + state + action → HDF5
        ↓
    Convert HDF5 → LeRobot dataset
        ↓
    Fine-tune Pi0.5
```

### Pipeline 8 terminal

| Terminal | Node | Vai trò |
|---|---|---|
| T1 | `vive_tf_and_joy_ros2.py` | OpenVR → /tf + button |
| T2 | `frame_as_posestamped_ros2.py` | TF → PoseStamped |
| T3 | `vive_ur5_teleop_params.py` | Apply calibration → target pose |
| T4 | `ur_follow_using_class_ros2.py` | UR3 follow target qua ur_rtde |
| T5 | `realsense2_camera_node` | Camera front (top-down) |
| T6 | `usb_cam_node_exe` | Camera wrist (gripper) |
| T7 | `control_robstride_ros.py` | Gripper bilateral + ROS publish |
| T8 | `record_all.py` | GUI record + HDF5 output |

---

## 2. Phần Cứng

### Robot
- **Universal Robots UR3** với network IP `192.168.1.1`
- Polyscope đã enable External Control / RTDE

### VR Tracking
- **2 Lighthouses** SteamVR gắn tường
- **1 Vive Tracker 3.0** cầm tay
- **Headset** chỉ để init SteamVR

### Camera (BẮT BUỘC)
- **Intel RealSense D435I** — front cam (top-down)
- **Logitech C922 Pro** — wrist cam (gắn gripper)
- ⚠️ **USB 3.0 cho Realsense** (cáp USB-C → USB-A 3.0)
- 2 camera nên cắm vào 2 USB controller khác nhau

### Gripper Robstride
- **2 motor Robstride 06** (ID=7 master/vô lăng, ID=6 slave/tay kẹp)
- Bilateral teleop với PID Mimic mode
- Auto-stop khi chạm vật (torque detection)
- Limits: `POS_OPEN_RAD=1.459`, `POS_CLOSE_RAD=8.509`

### USB-CAN Adapter
- **CANable2** với firmware slcan, bitrate 1Mbps
- Termination resistor 120Ω giữa CAN_H/CAN_L

### Máy tính
- Ubuntu 22.04 LTS
- ROS2 Humble

---

## 3. Phần Mềm

### ROS2 Humble
```bash
sudo apt update
sudo apt install ros-humble-desktop \
  ros-humble-realsense2-camera \
  ros-humble-usb-cam \
  ros-humble-tf-transformations \
  ros-humble-cv-bridge
```

### Python packages
```bash
pip install ur_rtde scipy numpy opencv-python openvr pynput \
            h5py python-can lerobot
```

### CAN tools
```bash
sudo apt install can-utils
```

### SteamVR
- Cài qua Steam
- Mở SteamVR TRƯỚC khi chạy ROS node

---

## 4. Cấu Trúc Dự Án

```
~/ur5_teleop_vive/                          ← outer ROS workspace
├── build/                                  ← gitignore
├── install/                                ← gitignore
├── log/                                    ← gitignore
├── .gitignore
├── README.md
└── ur5_teleop_vive/                        ← ROS package
    ├── CMakeLists.txt
    ├── package.xml
    ├── msg/
    │   └── Xyzrpy.msg                      ← Custom msg
    ├── launch/
    │   └── view_ur5.launch.py              ← RViz visualization
    ├── config/
    ├── mesh/
    │   └── hand.dae                        ← 3D model gripper
    ├── resource/
    └── ur5_teleop_vive/
        └── thesis_code/                    ← Code chính
            ├── vive_tf_and_joy_ros2.py
            ├── frame_as_posestamped_ros2.py
            ├── vive_ur5_teleop_params.py
            ├── ur_follow_using_class_ros2.py
            ├── control_robstride_ros.py    ← Gripper + ROS
            ├── check_robstride.py          ← Test motor
            ├── record_all.py               ← HDF5 recorder
            ├── convert_hdf5_to_lerobot.py
            ├── calib_4x4.py
            └── dataset/                    ← gitignore (HDF5)
```

### Custom Message

```
# msg/Xyzrpy.msg
std_msgs/Header header
float64 x
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
```

### ROS Topics

| Topic | Type | Vai trò |
|---|---|---|
| `/right_controller_as_posestamped` | PoseStamped | Vive pose |
| `/vive_right` | Joy | Button trigger |
| `/ur_target_pose` | PoseStamped | Target cho robot |
| `/ur_actual_pose` | Xyzrpy | TCP thực tế |
| `/ur_joint_states` | JointState | 6 joints |
| `/gripper/state` | Float32MultiArray | Gripper state (6 fields) |
| `/camera/camera/color/image_raw` | Image | Front cam |
| `/camera_wrist/image_raw` | Image | Wrist cam |

---

## 5. Tạo ROS2 Package Từ Đầu

Nếu dựng lại workspace trên máy mới:

### 5.1 Tạo folder workspace

```bash
mkdir -p ~/ur5_teleop_vive/src
cd ~/ur5_teleop_vive/src
```

### 5.2 Tạo ROS2 package

```bash
ros2 pkg create --build-type ament_cmake ur5_teleop_vive \
    --dependencies rclcpp std_msgs geometry_msgs sensor_msgs
```

Output:
```
going to create a new package
package name: ur5_teleop_vive
destination directory: /home/khanh/ur5_teleop_vive/src
package format: 3
version: 0.0.0
description: TODO: Package description
maintainer: ['khanh <khanh@example.com>']
licenses: ['TODO: License declaration']
build type: ament_cmake
dependencies: ['rclcpp', 'std_msgs', 'geometry_msgs', 'sensor_msgs']
creating folder ./ur5_teleop_vive
creating ./ur5_teleop_vive/package.xml
creating source and include folder
creating folder ./ur5_teleop_vive/src
creating folder ./ur5_teleop_vive/include/ur5_teleop_vive
creating ./ur5_teleop_vive/CMakeLists.txt
```

### 5.3 Tạo folder cần thiết

```bash
cd ~/ur5_teleop_vive/src/ur5_teleop_vive
mkdir -p msg launch config mesh resource ur5_teleop_vive/thesis_code
```

### 5.4 Tạo custom message `msg/Xyzrpy.msg`

```bash
cat > msg/Xyzrpy.msg << 'EOF'
std_msgs/Header header
float64 x
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
EOF
```

### 5.5 Sửa `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.8)
project(ur5_teleop_vive)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Xyzrpy.msg"
  DEPENDENCIES std_msgs
)

# Install Python scripts
install(PROGRAMS
  ur5_teleop_vive/thesis_code/vive_ur5_teleop_params.py
  ur5_teleop_vive/thesis_code/ur_follow_using_class_ros2.py
  ur5_teleop_vive/thesis_code/frame_as_posestamped_ros2.py
  ur5_teleop_vive/thesis_code/vive_tf_and_joy_ros2.py
  ur5_teleop_vive/thesis_code/control_robstride_ros.py
  ur5_teleop_vive/thesis_code/record_all.py
  ur5_teleop_vive/thesis_code/check_robstride.py
  ur5_teleop_vive/thesis_code/calib_4x4.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch / config / mesh
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY mesh   DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
```

### 5.6 Sửa `package.xml`

```xml
<?xml version="1.0"?>
<package format="3">
  <name>ur5_teleop_vive</name>
  <version>0.0.1</version>
  <description>UR3 teleop with HTC Vive for Pi0.5 dataset</description>
  <maintainer email="khanh@email.com">Khanh</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### 5.7 Build workspace

```bash
cd ~/ur5_teleop_vive
colcon build --packages-select ur5_teleop_vive
source install/setup.bash

# Verify custom msg
ros2 interface show ur5_teleop_vive/msg/Xyzrpy
```

Phải in ra:
```
std_msgs/Header header
        builtin_interfaces/Time stamp
                int32 sec
                uint32 nanosec
        string frame_id
float64 x
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
```

### 5.8 Copy code vào thesis_code

```bash
cd ~/ur5_teleop_vive/src/ur5_teleop_vive/ur5_teleop_vive/thesis_code
# Copy tất cả .py từ repo vào đây
```

### 5.9 Build lại sau khi thêm code

```bash
cd ~/ur5_teleop_vive
colcon build --packages-select ur5_teleop_vive
source install/setup.bash
```

---

## 6. Cài Đặt và Build

### Lần đầu trên máy mới

```bash
# 1. Clone repo
git clone https://github.com/Khanhiot-ai/ur3-vive-pi05.git ~/ur5_teleop_vive
cd ~/ur5_teleop_vive

# 2. Build
colcon build --packages-select ur5_teleop_vive
source install/setup.bash

# 3. Auto-source mỗi terminal
echo "source ~/ur5_teleop_vive/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Verify

```bash
ros2 interface show ur5_teleop_vive/msg/Xyzrpy   # phải có
ping 192.168.1.1                                  # UR3
ls /dev/ttyACM*                                   # CANable2
```

---

## 7. Setup Phần Cứng

### 7.1 Setup CAN cho Robstride

Script auto-setup:

```bash
cat > ~/setup_can.sh << 'EOF'
#!/bin/bash
CANABLE=$(ls -t /dev/ttyACM* 2>/dev/null | head -1)
if [ -z "$CANABLE" ]; then
    echo "❌ Không tìm thấy CANable2 — cắm USB chưa?"
    exit 1
fi
echo "🔍 CANable2 ở: $CANABLE"
sudo pkill slcand 2>/dev/null
sleep 0.5
sudo ip link set can0 down 2>/dev/null
sudo slcand -o -c -s8 $CANABLE can0
sleep 0.5
sudo ip link set can0 up
sudo ip link set can0 txqueuelen 1000
ip link show can0 | grep -q UP && echo "✅ can0 UP!" || echo "❌ Setup fail"
EOF
chmod +x ~/setup_can.sh
```

Mỗi lần cắm CANable2:
```bash
~/setup_can.sh
```

### 7.2 Test motor

```bash
python3 check_robstride.py --scan
python3 check_robstride.py --read --id 6 --id 7
python3 check_robstride.py --test --id 7 --speed 0.5
```

### 7.3 Test camera

```bash
v4l2-ctl --list-devices
realsense-viewer

# USB speed (Realsense phải 5000M)
ls /sys/bus/usb/devices/ | xargs -I{} sh -c \
  'echo -n "{}: "; cat /sys/bus/usb/devices/{}/speed 2>/dev/null'
```

### 7.4 Test SteamVR

- Mở SteamVR
- Tracker xanh + 2 lighthouse xanh

---

## 8. Calibration

```bash
python3 calib_4x4.py
```

1. Gắn Vive tracker lên flange UR3
2. Di chuyển robot đến **8 điểm** rải đều 3D
3. Mỗi điểm ≥ 10cm, KHÔNG đồng phẳng
4. Output: `world_alignment_matrix.txt`

| RMSE | Đánh giá |
|---|---|
| < 5mm | Excellent |
| < 10mm | Good |
| < 20mm | Acceptable |
| > 20mm | Làm lại |

---

## 9. Workflow Thu Data

### 9.1 Setup CAN + verify motor

```bash
~/setup_can.sh
python3 check_robstride.py --scan
```

### 9.2 Mở 8 terminal (source workspace mỗi terminal)

```bash
source ~/ur5_teleop_vive/install/setup.bash
```

**T1 — Vive bridge:**
```bash
python3 vive_tf_and_joy_ros2.py
```

**T2 — TF converter:**
```bash
python3 frame_as_posestamped_ros2.py
```

**T3 — Teleop logic:**
```bash
python3 vive_ur5_teleop_params.py
```

**T4 — UR3 controller:**
```bash
python3 ur_follow_using_class_ros2.py
```
Đợi: `SYSTEM READY — RTDE @ 100Hz`

**T5 — Realsense front:**
```bash
ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -p enable_depth:=false -p enable_infra1:=false -p enable_infra2:=false \
  -p enable_gyro:=false -p enable_accel:=false \
  -p rgb_camera.color_profile:="640x480x30"
```

**T6 — Wrist cam:**
```bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video8 \
  -p pixel_format:=mjpeg2rgb \
  -p image_width:=640 -p image_height:=480 -p framerate:=30.0 \
  -r image_raw:=/camera_wrist/image_raw
```

**T7 — Gripper:**
```bash
python3 control_robstride_ros.py --channel can0
```

**T8 — Recorder:**
```bash
python3 record_all.py --task pick_cube --fps 10
```

### 9.3 Phím tắt

**Robot teleop (T1):**
- **Home** — Set origin
- **Ctrl_R** — Toggle ON/OFF teleop

**Gripper (T7):**
- **Enter** — Auto start/stop
- **m** — Bật/tắt MIMIC (bilateral)
- **+/-** — Speed
- **r** — Đổi chiều

**Recorder GUI (T8):**
- **SPACE** — Bắt đầu / dừng
- **S** — Save SUCCESS
- **F** — Save FAIL
- **Q** — Thoát

### 9.4 Quy trình 1 episode

```
1. GUI: 2 cam viền xanh + 4 indicator ● (joints, actual, target, gripper)

2. Robot về home, đặt vật lên bàn

3. Ctrl_R → robot follow Vive
   'm' trong T7 → MIMIC gripper

4. SPACE trong recorder → bắt đầu ghi

5. Demo:
   - Vive di chuyển → UR3 tới vị trí gắp
   - Vặn vô lăng → tay kẹp đóng
   - Vive di chuyển → đến vị trí thả
   - Vặn ngược → tay kẹp mở

6. S (success) / F (fail)

7. Lặp lại
```

### 9.5 Số lượng episode khuyến nghị

| Mục đích | Số episode |
|---|---|
| Test pipeline | 1-5 |
| Pi0.5 cơ bản | 50-100 |
| Pi0.5 quality | 200-500 |
| Pi0.5 robust | 1000+ |

Đa dạng hóa: vị trí vật, hướng, ánh sáng. **Giữ nguyên camera.**

---

## 10. Format Dataset HDF5

### Schema

```
dataset/<task>.hdf5
└── data/                                  attrs: task, fps, robot, state_dim, action_dim
    ├── demo_0/                            attrs: success, n_frames, fps_actual, duration_s
    │   ├── obs/
    │   │   ├── image           uint8  (T, 224, 224, 3)
    │   │   ├── wrist_image     uint8  (T, 224, 224, 3)
    │   │   ├── state           f32    (T, 7)
    │   │   └── tactile_state   f32    (T, 1)
    │   └── actions             f32    (T, 8)
    └── demo_1/, demo_2/, ...
```

**State (7 dim):**
- `[0:6]` — 6 joint angles (rad)
- `[6]` — Gripper position [0=mở, 1=đóng]

**Action (8 dim):**
- `[0:3]` — TCP target XYZ (m)
- `[3:7]` — TCP target quaternion
- `[7]` — Gripper command [0,1]

**Tactile (1 dim):**
- `[0]` — Gripper torque (N·m)

### Verify

```bash
python3 << 'EOF'
import h5py
with h5py.File('dataset/pick_cube.hdf5', 'r') as f:
    print('Demos:', list(f['data'].keys()))
    for name in f['data']:
        g = f['data'][name]
        print(f'  {name}: {g.attrs["n_frames"]}f  success={g.attrs["success"]}')
EOF
```

---

## 11. Convert sang LeRobot

```bash
pip install lerobot

python3 convert_hdf5_to_lerobot.py \
  --src dataset/pick_cube.hdf5 \
  --repo-id khanh/ur3_pick_cube \
  --task "pick up the red cube" \
  --fps 10 \
  --skip-failed
```

Output: `~/.cache/huggingface/lerobot/khanh/ur3_pick_cube/`

### Push lên HuggingFace Hub

```bash
huggingface-cli login
python3 -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset('khanh/ur3_pick_cube').push_to_hub()
"
```

---



## 13. Troubleshooting

### Camera không hiện
- `v4l2-ctl --list-devices` để tìm device mới
- USB speed phải 5000M cho Realsense

### `actual: null`
- Chưa source workspace:
  ```bash
  source ~/ur5_teleop_vive/install/setup.bash
  ros2 topic echo /ur_actual_pose --once
  ```

### Robot không di chuyển
- Kiểm tra `SYSTEM READY` ở T4
- Đã nhấn **Home** rồi **Ctrl_R** chưa?
- SteamVR thấy tracker xanh không?

### Gripper không phản hồi
- CAN UP? `ip link show can0`
- Motor sống? `python3 check_robstride.py --scan`
- Bitrate đúng 1Mbps? `slcan -s8`

### Robot rung khi Vive đứng yên
Tăng deadzone trong `ur_follow_using_class_ros2.py`:
```python
self.deadzone_enter = 0.015      # 15mm
self.deadzone_exit = 0.008       # 8mm
self.orient_deadzone_deg = 3.0
```

### DDS spam errors
```bash
export ROS_LOCALHOST_ONLY=1
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc
```

### Realsense delay 1-2s
Driver issue, không fix được hoàn toàn. Không ảnh hưởng dataset (recorder dùng `time.time()`).

### `ros2 interface show Xyzrpy` not found
```bash
cd ~/ur5_teleop_vive
colcon build --packages-select ur5_teleop_vive
source install/setup.bash
```

### File > 100MB khi push
```bash
find . -type f -size +50M -not -path "./.git/*"

git filter-branch --force --index-filter \
  "git rm -r --cached --ignore-unmatch path/to/big_file" \
  --prune-empty --tag-name-filter cat -- --all
git push --force
```

---

## 📝 References

- **Pi0.5**: https://www.physicalintelligence.company
- **openpi**: https://github.com/Physical-Intelligence/openpi
- **LeRobot**: https://github.com/huggingface/lerobot
- **ur_rtde**: https://sdurobotics.gitlab.io/ur_rtde/
- **Robstride**: https://www.robstride.com/

---

## 📄 License

MIT — Tự do sử dụng, sửa đổi.

---

*Updated: May 2026 — Khanh (UR3 thesis project)*

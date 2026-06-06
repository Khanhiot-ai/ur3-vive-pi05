# UR3 + HTC Vive Teleop — Thu data cho Pi0.5 VLA (có cảm biến xúc giác DIGIT)

> **Hướng dẫn đầy đủ A→Z** để dựng lại dự án từ đầu. Đọc tuần tự từ mục 1.

Hệ thống teleoperation: điều khiển robot **UR3** bằng **HTC Vive Tracker**, gripper **Robstride**, 2 camera **RealSense** + 2 cảm biến xúc giác **DIGIT v1**, thu dataset HDF5 đa-modal để fine-tune **Pi0.5 VLA** cho task gắp bánh răng thả vào cột.

🔗 **Repo:** https://github.com/Khanhiot-ai/ur3-vive-pi05
📦 **Dataset:** https://huggingface.co/datasets/qkhanh1/ur3_pick_cube
📚 **Tham khảo:** https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5

---

## 📋 Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Phần cứng cần có](#2-phần-cứng-cần-có)
3. [Mô hình Pi0.5 — vì sao data phải thế này](#3-mô-hình-pi05--vì-sao-data-phải-thế-này)
4. [Cài đặt hệ điều hành + phần mềm](#4-cài-đặt-hệ-điều-hành--phần-mềm)
5. [Clone & build project](#5-clone--build-project)
6. [Setup phần cứng từng bước](#6-setup-phần-cứng-từng-bước)
7. [Calibration (góc yaw)](#7-calibration-góc-yaw)
8. [Cấu trúc dự án & ROS topics](#8-cấu-trúc-dự-án--ros-topics)
9. [Chạy pipeline (9 terminal)](#9-chạy-pipeline-9-terminal)
10. [Workflow thu data](#10-workflow-thu-data)
11. [Format HDF5 (dual-rate)](#11-format-hdf5-dual-rate)
12. [Check dataset](#12-check-dataset)
13. [Convert + push HuggingFace](#13-convert--push-huggingface)
14. [Train + inference Pi0.5](#14-train--inference-pi05)
15. [Changelog — các fix đã làm](#15-changelog--các-fix-đã-làm)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Tổng quan

### Pipeline tổng

```
┌─ THU DATA ──────────┐   ┌─ TRAIN PI0.5 ──┐   ┌─ INFERENCE ─────┐
│ Vive Tracker        │   │ Pi0.5 base     │   │ Camera + state  │
│ → UR3 + Gripper     │ → │ + ur3 dataset  │ → │ + DIGIT tactile │
│ → 2 cam + 2 DIGIT   │   │ → Checkpoint   │   │ → Robot tự làm  │
│ → 1 file HDF5       │   └────────┬───────┘   └─────────────────┘
└──────────────┬──────┘            │                     ▲
               └── HuggingFace ─────┴─────────────────────┘
```

### Task

Gắp bánh răng thả vào cột (gear pick-and-place) — contact-rich, 2 loại bánh răng to/nhỏ. Đây là task đòi hỏi xúc giác (biết lực kẹp, phát hiện trượt) nên cần DIGIT.

---

## 2. Phần cứng cần có

| Thiết bị | Số lượng | Chi tiết |
|---|---|---|
| Robot UR3 | 1 | IP `192.168.1.1`, Polyscope Remote Control mode |
| HTC Vive Base Station 2.0 (lighthouse) | 2 | IDs [2, 3] |
| HTC Vive Tracker 3.0 | 1 | KHÔNG cần kính VR (dùng null driver) |
| Intel RealSense D435i | 2 | Front (top-down) + Wrist (cạnh gripper) |
| DIGIT v1 (cảm biến xúc giác) | 2 | D21383 (LEFT) + D21384 (RIGHT) |
| Robstride 06 motor | 2 | ID 7 (master/vô lăng) + ID 6 (slave/tay kẹp) |
| CANable2 (USB-CAN) | 1 | slcan, bitrate 1Mbps |
| Máy tính | 1 | Ubuntu 22.04, ROS2 Humble, USB 3.0 (Thunderbolt) |

**RealSense serial (ví dụ — đổi theo máy bạn):**
- Front: `243322073847`
- Wrist: `027422070272`

---

## 3. Mô hình Pi0.5 — vì sao data phải thế này

Pi0.5 có 2 backbone xúc giác đông cứng, mỗi cái cần input riêng:

- **DINO** (ước lượng lực): ăn ảnh DIGIT, model tự resize về 224×224.
- **V-JEPA** (phát hiện trượt/slip): ăn cửa sổ **4 frame liên tiếp @60fps** (mốc t, t-2, t-4, t-6 ≈ 100ms), độ phân giải **(320 cao × 240 rộng) PORTRAIT** RGB.

→ Vì vậy DIGIT BẮT BUỘC: **60Hz**, **portrait (320,240)**, và action dùng **forward delta**.

Đây là lý do recorder dùng kiến trúc **dual-rate** (DIGIT 60Hz, camera/state 20Hz) — xem mục 11.

---

## 4. Cài đặt hệ điều hành + phần mềm

### 4.1 Hệ điều hành

Ubuntu 22.04 LTS + ROS2 Humble.

```bash
# Cài ROS2 Humble (nếu chưa có) — theo docs.ros.org
sudo apt install ros-humble-desktop \
  ros-humble-realsense2-camera \
  ros-humble-cv-bridge \
  can-utils
```

### 4.2 Python packages (cài ở python3.10)

```bash
python3.10 -m pip install \
  ur_rtde python-can h5py evdev \
  opencv-python pandas pyarrow \
  huggingface_hub openvr \
  digit-interface \
  "numpy<2"
```

| Package | Dùng cho |
|---|---|
| `ur_rtde` | Điều khiển UR3 (RTDE servoL) |
| `python-can` | Gripper Robstride qua CAN |
| `h5py` | Ghi/đọc HDF5 dataset |
| `evdev` | Đọc phím Home (chạy trên Wayland) |
| `openvr` | Đọc Vive Tracker qua SteamVR |
| `digit-interface` | Đọc cảm biến DIGIT |

### 4.3 ROS environment

Thêm vào `~/.bashrc`:
```bash
cat >> ~/.bashrc << 'EOF'
source /opt/ros/humble/setup.bash
source ~/ur5_teleop_vive/install/setup.bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
alias ur='source ~/ur5_teleop_vive/install/setup.bash'
EOF
source ~/.bashrc
```

`ROS_LOCALHOST_ONLY=1` + `ROS_DOMAIN_ID=0` fix lỗi DDS spam + "Failed to find a free participant index".

---

## 5. Clone & build project

```bash
git clone https://github.com/Khanhiot-ai/ur3-vive-pi05.git ~/ur5_teleop_vive
cd ~/ur5_teleop_vive

colcon build --packages-select ur5_teleop_vive --symlink-install
source install/setup.bash
```

Verify:
```bash
ros2 interface show ur5_teleop_vive/msg/Xyzrpy   # custom msg OK
ping -c 3 192.168.1.1                            # robot reachable
ls /dev/ttyACM*                                  # CANable2 có
```

> **Lưu ý tên folder:** robot thật là **UR3**, nhưng folder/package tên "ur5_teleop_vive" (giữ tên cũ cho tiện, không đổi).

---

## 6. Setup phần cứng từng bước

### 6.1 CAN cho gripper (CANable2 / slcand)

```bash
# Tạo script setup CAN
cat > ~/setup_can.sh << 'EOF'
#!/bin/bash
CANABLE=$(ls -t /dev/ttyACM* 2>/dev/null | head -1)
[ -z "$CANABLE" ] && echo "❌ Không thấy CANable2" && exit 1
echo "Found: $CANABLE"
sudo pkill slcand 2>/dev/null; sleep 0.5
sudo ip link set can0 down 2>/dev/null
sudo slcand -o -c -s8 $CANABLE can0   # -s8 = 1Mbps
sleep 0.5
sudo ip link set can0 up
sudo ip link set can0 txqueuelen 1000
ip link show can0 | grep -q "state UP" && echo "✅ can0 UP!" || echo "❌ Fail"
EOF
chmod +x ~/setup_can.sh
~/setup_can.sh
```

Bảng bitrate slcand: s3=100k, s4=125k, s5=250k, s6=500k, s7=800k, **s8=1M** (Robstride dùng s8).

### 6.2 Quyền DIGIT (camera USB)

```bash
sudo chmod 666 /dev/video*    # mỗi lần khởi động máy

# Verify 2 DIGIT nhận
python3 -c "from digit_interface import DigitHandler; print(DigitHandler.list_digits())"
# Phải thấy D21383 + D21384
```

### 6.3 Quyền bàn phím evdev (cho phím Home)

```bash
sudo usermod -a -G input $USER
# LOGOUT/LOGIN lại (hoặc reboot) để vào group input
groups | grep input    # verify thấy "input"
```

### 6.4 Verify 2 RealSense USB 3.0

```bash
rs-enumerate-devices | grep -E "Serial Number|Usb Type"
# Cả 2 phải: Usb Type Descriptor: 3.2

lsusb -t | grep uvcvideo
# Cả 2 phải 5000M+ (trên Thunderbolt Bus 04, KHÔNG phải Bus 03 USB 2.0)
```

⚠️ Nếu RealSense ở Bus 03 (480M) → cắm cổng USB khác (Thunderbolt 4) cho đến khi vào Bus 04.

### 6.5 SteamVR không cần kính (null HMD)

Dự án track Vive Tracker không có kính VR → bật null driver.

```bash
# Cài SteamVR: Steam → Library → SteamVR → Install

# File 1: bật null driver
nano ~/.local/share/Steam/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings
# Đổi "enable": false → true

# File 2: tắt requireHmd
nano ~/.local/share/Steam/steamapps/common/SteamVR/resources/settings/default.vrsettings
# Thêm/sửa trong block "steamvr":
#   "requireHmd": false,
#   "forcedDriver": "null",
#   "activateMultipleDrivers": true,
```

**Chạy SteamVR** (qua Steam UI, bỏ qua cảnh báo không HMD). Verify: SteamVR status hiện 1 tracker (xanh) + 2 lighthouse (xanh).

Tracker lần đầu cần pair: SteamVR → Devices → Pair Controller (giữ nút tracker đến khi đèn nhấp nháy).

### 6.6 Verify DIGIT portrait

```bash
python3 -c "
import cv2
from digit_interface import Digit
d = Digit('D21383'); d.connect()
d.set_resolution(Digit.STREAMS['QVGA'])
print('DIGIT shape:', d.get_frame().shape)
d.disconnect()
"
# Phải ra (320, 240, 3) — portrait sẵn, KHÔNG cần xoay
```

> DIGIT v1 (máy này) trả frame portrait (320,240) sẵn → `digit_publisher_ros2.py` đặt `ROTATE_DIR = None`. Nếu DIGIT khác trả landscape (240,320) → đặt `cv2.ROTATE_90_CLOCKWISE`.

### 6.7 Verify gripper pos_norm

```bash
python3.10 control_robstride_ros_without_calip.py --channel can0
# Gõ: c (calip) → m (MIMIC) → o (mở hết, chốt 0.0) → p (kẹp hết, chốt 1.0)
```

Terminal khác:
```bash
ros2 topic echo /gripper/state --field data
```
Xoay vô lăng: `data[2]` (pos_norm) phải chạy mượt 0.0 → 1.0.

---

## 7. Calibration (góc yaw)

Vive lighthouse + UR3 đều đứng thẳng → chỉ khác góc yaw quanh trục Z → đo 1 góc duy nhất là đủ (không cần Kabsch 3D phức tạp).

### Kết quả có sẵn

```
world_alignment_angle.txt:  -30.490862°  (std 0.043° — excellent)
world_alignment_matrix.txt: tự sinh từ góc
```

Không cần calib lại trừ khi di chuyển lighthouse/robot.

### Calib lại (nếu cần)

```bash
python3.10 calib_manual.py
# 1. Gắn tracker lên flange UR3
# 2. Di chuyển robot theo trục X+ (~10cm, 5 đoạn)
# 3. Script lưu world_alignment_*.txt
```

---

## 8. Cấu trúc dự án & ROS topics

```
~/ur5_teleop_vive/                          ← GIT REPO GỐC
├── README.md
└── ur5_teleop_vive/
    ├── CMakeLists.txt, package.xml
    ├── msg/Xyzrpy.msg
    └── ur5_teleop_vive/thesis_code/
        ├── vive_tf_and_joy_ros2.py              ← OpenVR → /tf + teleop enable
        ├── frame_as_posestamped_ros2.py         ← TF → PoseStamped @60Hz
        ├── vive_ur5_teleop_params.py            ← yaw align → /ur_target_pose, set origin (evdev Home)
        ├── ur_follow_using_class_ros2.py        ← RTDE servoL @100Hz, auto home
        ├── control_robstride_ros_without_calip.py ← Gripper Mimic + /gripper/state
        ├── digit_publisher_ros2.py              ← 2 DIGIT @60Hz portrait
        ├── launch_realsense_all.sh              ← 2 RealSense
        ├── record_all.py                        ← HDF5 recorder DUAL-RATE
        ├── check_hdf5.py                         ← inspect dataset
        ├── Convert_hdf5_to_lerobot.py           ← HDF5 → LeRobot v2 (gồm DIGIT)
        ├── push_to_huggingface.py               ← upload Hub
        ├── view_ur5.launch.py                    ← xem robot trong RViz
        ├── world_alignment_*.txt                 ← calibration
        └── dataset/                              ← .gitignore (file nặng)
            └── pick_cube.hdf5                     ← 1 file chứa tất cả demo
```

### ROS Topics

| Topic | Type | Publisher | Subscriber |
|---|---|---|---|
| `/right_controller_as_posestamped` | PoseStamped | frame_as_posestamped | vive_teleop |
| `/teleop_enable` | Bool | record_all | vive_tf |
| `/set_origin` | Bool | record_all | vive_teleop (dự phòng) |
| `/auto_home` | Bool | record_all | ur_follow |
| `/ur_target_pose` | PoseStamped | vive_teleop | ur_follow + record_all |
| `/robot_origin_cmd` | Pose | vive_teleop | ur_follow |
| `/ur_actual_pose` | Xyzrpy | ur_follow | record_all (state + delta) |
| `/gripper/state` | Float32MultiArray[6] | control_robstride | record_all |
| `/digit_left/image_raw` | Image | digit_publisher | record_all |
| `/digit_right/image_raw` | Image | digit_publisher | record_all |
| `/camera_front/.../image_raw` | Image | realsense | record_all |
| `/camera_wrist/.../image_raw` | Image | realsense | record_all |

### /gripper/state fields

| Index | Field | Mô tả |
|---|---|---|
| 0 | pos_master | Vị trí vô lăng (rad) |
| 1 | pos_slave | Vị trí tay kẹp (rad) |
| 2 | **pos_norm** | 0.0=mở, 1.0=kẹp (Pi0.5 dùng) |
| 3 | torque | Torque (N·m) — kẹp vật, không lưu HDF5 |
| 4 | contact | 0/1 |
| 5 | mode | 0=IDLE, 1=AUTO, 2=MIMIC |

---

## 9. Chạy pipeline (9 terminal)

⚠️ **Bật SteamVR (mục 6.5) + chạy `~/setup_can.sh` (mục 6.1) + `sudo chmod 666 /dev/video*` TRƯỚC.**

Mỗi terminal `cd` vào code + source workspace (`ur`):
```bash
cd ~/ur5_teleop_vive/ur5_teleop_vive/ur5_teleop_vive/thesis_code
ur    # alias source workspace
```

### Giai đoạn A — Tracker (T1, T2, T3)

```bash
# T1 — Vive tracker → TF
python3 vive_tf_and_joy_ros2.py
# Đợi: "Lighthouse IDs: [2,3]" + "Tracker ID published"

# T2 — TF → PoseStamped @60Hz
python3 frame_as_posestamped_ros2.py
# Đợi: "Started converting right_controller -> world at 60Hz"

# T3 — Teleop logic + set origin (phím Home)
python3 vive_ur5_teleop_params.py
# Đợi: "đọc Home từ: AT Translated Set 2 keyboard" + "INIT done"
```

### Giai đoạn B — Robot + Camera (T4, T5)

```bash
# T4 — UR3 RTDE (⚠️ robot tự về home, đứng xa robot)
python3 ur_follow_using_class_ros2.py
# Đợi: "RTDE connected" + "SYSTEM READY TO CONTROL"

# T5 — 2 RealSense
./launch_realsense_all.sh
# Đợi: "CẢ 2 CAMERA ĐANG CHẠY"
```

### Giai đoạn C — Gripper + DIGIT (T6, T7)

```bash
# T6 — Gripper Robstride
python3.10 control_robstride_ros_without_calip.py --channel can0
# Gõ: c → m → o (mở hết) → p (kẹp hết)

# T7 — 2 DIGIT @60Hz
python3 digit_publisher_ros2.py
# Đợi: "LEFT: ~60fps | RIGHT: ~60fps"
```

### Giai đoạn D — Recorder (T8)

```bash
# T8 — Recorder HDF5
python3 record_all.py --task pick_cube --fps 20
# GUI lưới 2x2: FRONT | WRIST | DIGIT_L | DIGIT_R + status bar 7 chấm
```

### (Tùy chọn) RViz xem robot

```bash
# T9 — xem robot UR3 3D (cần T4 chạy)
ros2 launch view_ur5.launch.py
```

---

## 10. Workflow thu data

### 10.1 Verify status bar (7 chấm)

GUI recorder phải đủ 7 chấm `*` (sáng):
```
front *  wrist *  digL *  digR *  actual *  target *  grip *
```
Nếu `actual ○` → chưa set origin (bước tiếp theo).

### 10.2 Phím điều khiển

| Phím | Tác dụng | Terminal |
|---|---|---|
| `Home` (hoặc numpad 7) | Set origin, robot tới tracker | T3 |
| `c` / `m` / `o` / `p` | Calip / MIMIC / chốt mở / chốt kẹp | T6 |
| `SPACE` | Bắt đầu/dừng ghi (kèm robot ON) | T8 |
| `S` | Lưu SUCCESS ✅ + off + về home | T8 |
| `F` | Lưu FAIL ❌ + off + về home | T8 |
| `Q` | Thoát | T8 |

### 10.3 Mỗi demo (5-15s, 1 hành động pick+place)

```
1. Đặt bánh răng ở vị trí mới (đa dạng hóa!)
2. Cầm tracker (đảm bảo lighthouse track — đèn tracker xanh)
3. Bấm HOME → robot di chuyển đến tracker, origin chốt
4. SPACE → robot ON + bắt đầu record
5. Di chuyển tracker → gắp bánh răng (xoay vô lăng kẹp)
6. Đưa đến cột → thả vào (xoay vô lăng mở)
7. S (thành công) hoặc F (thất bại) → tự lưu + off + về home
8. Đợi robot về home (~2s), lặp lại bước 1
```

### 10.4 Thu nhiều demo — tất cả vào 1 file

```
HOME → SPACE → pick+place → S    ← demo_0
HOME → SPACE → pick+place → S    ← demo_1
... lặp ...
Q                                ← thoát
```

Tất cả nối vào `dataset/pick_cube.hdf5`.

### 10.5 Mục tiêu

| Demos | Pi0.5 success rate |
|---|---|
| 10 | Test pipeline |
| 50 | 30-50% |
| 100 | 60-75% |
| 200+ | 80-90% (thesis quality) |

---

## 11. Format HDF5 (dual-rate)

DIGIT thu ở **60Hz** (callback), camera/state/action thu ở **20Hz** (tick). Timestamp dùng để converter align sau.

```
dataset/pick_cube.hdf5
└── data/                       attrs: task, fps=20, robot=UR3, action_convention
    ├── demo_0/                 attrs: success, n_frames, duration_s
    │   ├── obs/
    │   │   ├── image           uint8  (T20, 480, 640, 3)  Front cam @20Hz
    │   │   ├── wrist_image     uint8  (T20, 480, 640, 3)  Wrist cam @20Hz
    │   │   ├── digit_left      uint8  (T60, 320, 240, 3)  DIGIT trái @60Hz
    │   │   ├── digit_right     uint8  (T60, 320, 240, 3)  DIGIT phải @60Hz
    │   │   ├── digit_left_ts   f64    (T60,)              timestamp @60Hz
    │   │   ├── digit_right_ts  f64    (T60,)
    │   │   ├── state           f32    (T20, 8)            TCP + grip
    │   │   └── timestamp       f64    (T20,)              tick @20Hz
    │   └── actions             f32    (T20, 7)            forward delta
    └── demo_N/...
```

- **state (8):** `[ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper]` — Cartesian TCP + gripper [0,1]
- **action (7):** `[dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]` — forward delta: `action[t] = pose(t+1) - pose(t)`
- Tỉ lệ T60/T20 ≈ 3 (DIGIT 60Hz / tick 20Hz)

### So với berkeley_autolab_ur5

| Field | Dự án này | Berkeley | Ghi chú |
|---|---|---|---|
| state | (8,) Cartesian+grip | (8,) | ✅ giống chiều |
| action | (7,) delta | (7,) | ✅ giống chiều |
| image | 480×640 | 480×640 | ✅ |
| gripper | liên tục [0,1] | 0/1 nhị phân | khác (phù hợp 2 loại bánh răng) |
| quaternion | cố định | thay đổi | khác (task chỉ tịnh tiến) |
| DIGIT | có (60Hz) | không | thêm cho tactile |

---

## 12. Check dataset

Sau mỗi ~10 demo:
```bash
python3 check_hdf5.py
python3 check_hdf5.py --demo 0 --save    # lưu ảnh ra /tmp/
```

Output mong đợi:
```
✅ obs/image:       (T20, 480, 640, 3)
✅ obs/digit_left:  (T60, 320, 240, 3)
✅ Dual-rate OK: ~3× (DIGIT 60Hz)
✅ Portrait (320,240) — đúng cho V-JEPA
✅ obs/state:       (T20, 8)
✅ actions:         (T20, 7)
   Gripper: min=0.0  max=1.0
```

### Cảnh báo thường gặp

| Cảnh báo | Nguyên nhân | Fix |
|---|---|---|
| `XYZ không đổi` | Robot đứng yên | Origin chưa set (bấm Home) |
| `Landscape (240,320)` | DIGIT xoay sai | `ROTATE_DIR = None` trong publisher |
| `Gripper max < 1.0` | Vô lăng chưa kẹp hết | Chốt lại mốc p |
| `Episode > 30s` | Demo quá dài | Mỗi demo 5-15s |

---

## 13. Convert + push HuggingFace

```bash
cd ~/ur5_teleop_vive/ur5_teleop_vive/ur5_teleop_vive/thesis_code

# 1. Check lần cuối
python3 check_hdf5.py dataset/pick_cube.hdf5

# 2. Convert → LeRobot v2 (gồm DIGIT, map 60→20Hz theo timestamp)
python3.10 Convert_hdf5_to_lerobot.py \
  --src dataset/pick_cube.hdf5 \
  --task "put the gear onto the peg" \
  --fps 20 --skip-failed --overwrite

# 3. Push HuggingFace
huggingface-cli login    # paste Write token (1 lần)
python3.10 push_to_huggingface.py --repo-id qkhanh1/ur3_pick_cube
```

LeRobot output có: `observation.image`, `observation.wrist_image`, `observation.digit_left`, `observation.digit_right`, `observation.state`, `action`.

---

## 14. Train + inference Pi0.5

### Train (Colab/GPU)

```python
!pip install lerobot openpi
from huggingface_hub import login; login(token="hf_xxx")

!python -m openpi.train \
  --dataset qkhanh1/ur3_pick_cube \
  --model pi0.5-base --epochs 50
```

### Inference trên robot thật

```bash
huggingface-cli download qkhanh1/pi0.5_ur3_pickcube \
  --local-dir ~/checkpoints/pi0.5_ur3

# Chạy T1,T2,T4,T5,T6,T7 (KHÔNG cần T3 teleop, T8 record)
# Thêm node inference đọc camera+DIGIT+state → Pi0.5 → action
```

⚠️ **An toàn:** giảm `normal_max_speed = 0.5` trong ur_follow, tay ở E-Stop.

---

## 15. Changelog — các fix đã làm

Ghi lại toàn bộ hành trình debug để học trò hiểu vì sao code như hiện tại.

### 15.1 Driver UR: URBasic → ur_rtde
URBasic hay treo (`get_inverse_kin failed`), latency cao. ur_rtde servoL @100Hz mượt, dễ lấy force/torque. RViz vẫn xem được với RTDE (qua `/joint_states`).

### 15.2 Calibration: Kabsch 3D → góc yaw đơn giản
Lighthouse + robot đều đứng thẳng → chỉ khác yaw quanh Z → đo 1 góc (-30.49°) thay vì Kabsch 8 điểm phức tạp.

### 15.3 Camera: webcam → 2× RealSense D435i
usb_cam crash, C922 chỉ USB 2.0. Đổi RealSense. Lỗi đã gặp: serial bị cast integer (bọc `"..."`), profile invalid (fallback 1280×720), USB 2.0 nhầm (cắm Bus 04 Thunderbolt).

### 15.4 QoS mismatch (`front ○`)
RealSense publish RELIABLE, recorder cũ subscribe BEST_EFFORT → không nhận. Fix 2 QoS riêng: `qos_cam` (RELIABLE) cho camera, `qos_robot` (BEST_EFFORT) cho robot/gripper.

### 15.5 Gripper pos_norm: dùng vô lăng smooth 0→1
Cũ normalize theo tay kẹp (sai khi kẹp vật). Mới dùng vị trí vô lăng + chốt mốc thủ công (`o`/`p`) → pos_norm liên tục 0→1. Pi0.5 học trajectory liên tục tốt hơn 0/1 cứng.

### 15.6 Dataset format → Berkeley AutoLab UR5
state 8 dim Cartesian, action 7 dim delta, image 480×640. Pi0.5 đã train trên berkeley → fine-tune dễ.

### 15.7 DIGIT 20Hz → DUAL-RATE 60Hz (chí tử cho V-JEPA)
Recorder cũ thu DIGIT ở tick 20Hz → mất 2/3 frame, hỏng nhánh V-JEPA. Fix: DIGIT thu ở callback 60Hz (buffer riêng + timestamp), camera/state/action ở tick 20Hz. HDF5 lưu 2 nhịp, converter map 60→20Hz theo timestamp.

### 15.8 DIGIT chiều: phát hiện đã portrait sẵn
Tưởng DIGIT trả landscape nên thêm `cv2.rotate(ROTATE_90_CLOCKWISE)`. Test thực tế: DIGIT máy này trả **portrait (320,240) sẵn** → xoay lại làm thành landscape (sai)! Fix: `ROTATE_DIR = None`, giữ nguyên frame. Bài học: luôn kiểm tra `get_frame().shape` thật.

### 15.9 Action: backward → forward delta
Cũ `action[t] = actual[t] - actual[t-1]` (backward, lệch off-by-one). Convention BC chuẩn là forward: `action[t] = pose(t+1) - pose(t)`. Lưu pose mỗi tick, tính delta khi save. Frame cuối delta=0.

### 15.10 Bỏ tactile_state (torque thô)
Pi0.5 không dùng torque 1D (tactile thật từ ảnh DIGIT). Bỏ ghi vào HDF5, nhưng torque vẫn đọc trong gripper để kẹp vật.

### 15.11 Phím Home (Wayland + pynput → evdev KEY_KP7) — fix khó nhất
Bấm Home không tác dụng, terminal hiện `^[[H`. Quá trình debug:
1. Wayland chặn pynput đọc phím toàn cục.
2. Thử đổi X11 → desktop không vào được. Bỏ.
3. Cài evdev + vào group `input` → pynput vẫn fail.
4. Liệt kê `/dev/input/event*` → pynput đọc nhầm chuột Logitech G102.
5. Bàn phím thật = `/dev/input/event4` (AT Translated Set 2 keyboard).
6. Phát hiện: phím Home laptop = **KEY_KP7** (numpad 7), không phải KEY_HOME! pynput map sai.

Fix triệt để: thay pynput bằng **evdev** đọc thẳng bàn phím, bắt cả KEY_HOME + KEY_KP7, chạy trên Wayland. (`^[[H` lọt terminal là bình thường, không ảnh hưởng.)

### 15.12 Converter thêm DIGIT
Converter cũ không đưa DIGIT vào LeRobot. Fix: đọc digit_left/right + timestamp, map 60→20Hz (nearest), encode 2 video MP4, thêm vào parquet + info.json.

### 15.13 ROS participant index full
Quá nhiều node + `ros2 topic echo` → cạn participant. Fix: `ROS_LOCALHOST_ONLY=1`, `ROS_DOMAIN_ID=0`, đóng echo thừa.

---

## 16. Troubleshooting

### Robot UR3
| Lỗi | Fix |
|---|---|
| `RTDE connection failed` | Robot chưa Remote Control mode |
| `Protective stop` | Restart trên teach pendant |
| Robot quá nhanh | Giảm `normal_max_speed` |

### RealSense
| Lỗi | Fix |
|---|---|
| `Frames didn't arrive` | USB 2.0 — cắm Bus 04 Thunderbolt |
| `Device busy` | `pkill -9 -f realsense2_camera_node` |
| `front ○` recorder | QoS — recorder mới đã fix 2 QoS riêng |

### DIGIT
| Lỗi | Fix |
|---|---|
| `Cannot open video` | `sudo chmod 666 /dev/video*`, rút/cắm lại |
| Landscape (240,320) | DIGIT portrait sẵn → `ROTATE_DIR = None` |
| Node nhảy /dev/video số cao | rút/cắm lại + chmod |

### Gripper / CAN
| Lỗi | Fix |
|---|---|
| `Transmit buffer full` | Reset CAN: `~/setup_can.sh`, `txqueuelen 1000` |
| `slcand: device busy` | `sudo pkill slcand` rồi setup lại |
| `can0 not found` | Chạy `~/setup_can.sh` |
| pos_norm không về 1.0 | Chốt lại mốc `p` (kẹp hết) |
| `Permission denied /dev/ttyACM*` | `sudo usermod -aG dialout $USER` + logout |

### Vive / SteamVR
| Lỗi | Fix |
|---|---|
| Tracker xám | Giữ nút tracker 3s, check pin |
| SteamVR đòi headset | Set `requireHmd: false` + null driver enable (6.5) |
| Tracker không track (T1) | Bật SteamVR trước; đèn tracker xanh + 2 lighthouse |

### Phím Home
| Lỗi | Fix |
|---|---|
| Bấm Home không tác dụng | Chạy đủ T1-T4 + tracker track (pose_callback cần tracker pose) |
| `^[[H` ở terminal | Bình thường với evdev — kiểm tra T3 in "Origin set" |
| evdev đọc nhầm device | Tự dò bàn phím (bỏ qua chuột); phím Home = KEY_KP7 |
| Chưa vào group input | `sudo usermod -aG input $USER` + logout/login |

### ROS
| Lỗi | Fix |
|---|---|
| `Failed to find a free participant index` | `ROS_DOMAIN_ID=0`, đóng echo thừa |
| `RCLError: rmw handle invalid` | Terminal chưa source ROS2 (`ur`) |
| bashrc "not found" workspace cũ | Xóa dòng source cũ, hoặc tạo workspace rỗng |

### HDF5 / Recorder
| Lỗi | Fix |
|---|---|
| `actual ○` | Origin chưa set (Home), hoặc T4 chưa publish |
| Gripper toàn 0 | MIMIC chưa bật (gõ `m` ở T6) |
| Demo 60s+ | 1 pick+place chỉ 5-15s |
| DIGIT trống → skip demo | Kiểm tra T7 digit_publisher chạy |

### Convert / Push / Git
| Lỗi | Fix |
|---|---|
| `ModuleNotFoundError: lerobot.common` | Dùng converter mới |
| Push "Auth" fail | `huggingface-cli login` lại (Write token) |
| Git push reject (fetch first) | `git pull origin main --no-rebase` trước |
| File nặng reject | `.gitignore` add `dataset/`, `*.hdf5`, `*.mp4` |

---

## 📝 References

- **Pi0.5:** https://www.physicalintelligence.company
- **openpi:** https://github.com/Physical-Intelligence/openpi
- **LeRobot:** https://github.com/huggingface/lerobot
- **Berkeley AutoLab UR5:** https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5
- **ur_rtde:** https://sdurobotics.gitlab.io/ur_rtde/
- **DIGIT:** https://digit.ml

---

*README — UR3 Vive Pi0.5 + DIGIT thesis project — qkhanh1*

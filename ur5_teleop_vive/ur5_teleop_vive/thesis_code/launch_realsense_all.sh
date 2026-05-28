#!/bin/bash
# ════════════════════════════════════════════════════════════════════
#  Launch 2 Realsense D435I cùng lúc với namespace riêng
# ════════════════════════════════════════════════════════════════════
#
#  Topics output:
#    /camera_front/camera/color/image_raw
#    /camera_wrist/camera/color/image_raw
#
#  Nếu sai vai trò → đổi chỗ 2 dòng SERIAL_* dưới đây.
#  Ctrl+C để dừng cả 2 cam.
# ════════════════════════════════════════════════════════════════════

SERIAL_FRONT="243322073847"
SERIAL_WRIST="027422070272"

PROFILE="640x480x30"

# ── Cleanup ────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "🛑 Stopping cameras..."
    [[ -n "$FRONT_PID" ]] && kill -SIGINT "$FRONT_PID" 2>/dev/null
    [[ -n "$WRIST_PID" ]] && kill -SIGINT "$WRIST_PID" 2>/dev/null
    sleep 1
    [[ -n "$FRONT_PID" ]] && kill -9 "$FRONT_PID" 2>/dev/null
    [[ -n "$WRIST_PID" ]] && kill -9 "$WRIST_PID" 2>/dev/null
    pkill -f "realsense2_camera_node" 2>/dev/null
    echo "✅ Both cameras stopped."
    exit 0
}
trap cleanup INT TERM

# ════════════════════════════════════════════════════════════════════
#  Verify — dùng grep thay vì awk để tránh lỗi parse khi có ERROR log
# ════════════════════════════════════════════════════════════════════
echo "🔍 Detecting Realsense devices..."

# Lấy output, lọc bỏ dòng ERROR/WARNING, chỉ giữ dòng có serial
FOUND_SERIALS=$(rs-enumerate-devices -s 2>/dev/null \
    | grep -v "^[[:space:]]*$" \
    | grep -v "ERROR\|WARNING\|was previously" \
    | grep -oE "[0-9]{12}")

echo "   Detected serials: $(echo $FOUND_SERIALS | tr '\n' ' ')"

for SERIAL in "$SERIAL_FRONT" "$SERIAL_WRIST"; do
    if ! echo "$FOUND_SERIALS" | grep -q "$SERIAL"; then
        echo "❌ Serial $SERIAL không tìm thấy!"
        echo ""
        echo "Các serial đang có:"
        rs-enumerate-devices -s 2>/dev/null | grep -E "RealSense|D4|L5|T2" || echo "(không detect được)"
        exit 1
    fi
done
echo "✅ Cả 2 serial đều OK"

# ════════════════════════════════════════════════════════════════════
#  FRONT camera
# ════════════════════════════════════════════════════════════════════
echo ""
echo "▶️  Starting FRONT camera (serial: $SERIAL_FRONT)..."
ros2 run realsense2_camera realsense2_camera_node --ros-args \
    -r __ns:=/camera_front \
    -p serial_no:="\"$SERIAL_FRONT\"" \
    -p camera_name:="camera_front" \
    -p enable_color:=true \
    -p enable_depth:=false \
    -p enable_infra1:=false \
    -p enable_infra2:=false \
    -p enable_gyro:=false \
    -p enable_accel:=false \
    -p rgb_camera.color_profile:="$PROFILE" \
    > /tmp/camera_front.log 2>&1 &
FRONT_PID=$!
echo "   PID = $FRONT_PID"

sleep 4

if ! kill -0 "$FRONT_PID" 2>/dev/null; then
    echo "❌ FRONT camera crashed! Log:"
    tail -20 /tmp/camera_front.log
    exit 1
fi
echo "   ✅ FRONT OK"

# ════════════════════════════════════════════════════════════════════
#  WRIST camera
# ════════════════════════════════════════════════════════════════════
echo ""
echo "▶️  Starting WRIST camera (serial: $SERIAL_WRIST)..."
ros2 run realsense2_camera realsense2_camera_node --ros-args \
    -r __ns:=/camera_wrist \
    -p serial_no:="\"$SERIAL_WRIST\"" \
    -p camera_name:="camera_wrist" \
    -p enable_color:=true \
    -p enable_depth:=false \
    -p enable_infra1:=false \
    -p enable_infra2:=false \
    -p enable_gyro:=false \
    -p enable_accel:=false \
    -p rgb_camera.color_profile:="$PROFILE" \
    > /tmp/camera_wrist.log 2>&1 &
WRIST_PID=$!
echo "   PID = $WRIST_PID"

sleep 4

if ! kill -0 "$WRIST_PID" 2>/dev/null; then
    echo "❌ WRIST camera crashed! Log:"
    tail -20 /tmp/camera_wrist.log
    kill "$FRONT_PID" 2>/dev/null
    exit 1
fi
echo "   ✅ WRIST OK"

# ════════════════════════════════════════════════════════════════════
#  Summary
# ════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ CẢ 2 CAMERA ĐANG CHẠY"
echo "════════════════════════════════════════════════════════════════"
echo "  Profile: $PROFILE"
echo ""
echo "  Topics:"
echo "    /camera_front/camera/color/image_raw"
echo "    /camera_wrist/camera/color/image_raw"
echo ""
echo "  Verify:"
echo "    ros2 topic hz /camera_front/camera/color/image_raw"
echo "    ros2 topic hz /camera_wrist/camera/color/image_raw"
echo "    python3.10 camera_check.py"
echo ""
echo "  Press Ctrl+C để dừng cả 2."
echo "════════════════════════════════════════════════════════════════"

wait
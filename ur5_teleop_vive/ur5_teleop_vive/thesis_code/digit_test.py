#!/usr/bin/env python3
"""
digit_test.py
=============
Test đọc ảnh từ cảm biến DIGIT v1.

Cài trước:
    pip install digit-interface opencv-python

Chạy:
    # Liệt kê tất cả DIGIT đang cắm
    python3 digit_test.py --list

    # Xem ảnh live 1 DIGIT (thay serial của bạn)
    python3 digit_test.py --serial D20542

    # Xem ảnh live 2 DIGIT cùng lúc
    python3 digit_test.py --serial D20542 D20543

Phím:
    q = thoát
"""

import argparse
import sys

try:
    from digit_interface import Digit, DigitHandler
except ImportError:
    print("❌ Chưa cài digit-interface. Chạy:")
    print("   pip install digit-interface")
    sys.exit(1)

import cv2
import numpy as np


def list_digits():
    """Liệt kê tất cả DIGIT đang kết nối."""
    print("🔍 Đang tìm DIGIT...")
    digits = DigitHandler.list_digits()
    if not digits:
        print("❌ Không tìm thấy DIGIT nào!")
        print("   - Kiểm tra cáp USB")
        print("   - Thử: ls /dev/video*")
        print("   - Có thể cần: sudo chmod 666 /dev/video*")
        return
    print(f"✅ Tìm thấy {len(digits)} DIGIT:\n")
    for i, d in enumerate(digits):
        print(f"  [{i}] serial = {d.get('serial', '?')}")
        print(f"      dev    = {d.get('dev_name', '?')}")
        print(f"      manuf  = {d.get('manufacturer', '?')}")
        print()
    print("→ Dùng serial này cho --serial khi chạy script chính.")


def view_digits(serials):
    """Hiển thị ảnh live từ 1 hoặc nhiều DIGIT."""
    digits = []
    for s in serials:
        print(f"🔌 Kết nối DIGIT {s}...")
        try:
            d = Digit(s)
            d.connect()
            # Set resolution QVGA 320x240
            d.set_resolution(Digit.STREAMS["QVGA"])
            # Set fps 60 (QVGA hỗ trợ 60 hoặc 30)
            d.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])
            # Bật đèn LED max
            d.set_intensity(Digit.LIGHTING_MAX)
            digits.append((s, d))
            print(f"   ✅ {s} OK")
        except Exception as e:
            print(f"   ❌ {s} lỗi: {repr(e)}")
            import traceback
            traceback.print_exc()

    if not digits:
        print("❌ Không kết nối được DIGIT nào.")
        return

    print("\n▶️  Đang stream — nhấn 'q' để thoát\n")
    win = "DIGIT view"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = []
            for s, d in digits:
                frame = d.get_frame()   # numpy (H, W, 3) BGR
                if frame is None:
                    frame = np.zeros((240, 320, 3), np.uint8)
                # Label serial
                f = frame.copy()
                cv2.putText(f, s, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frames.append(f)

            # Ghép ngang nếu nhiều DIGIT
            combined = np.hstack(frames) if len(frames) > 1 else frames[0]
            cv2.imshow(win, combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for s, d in digits:
            d.disconnect()
        cv2.destroyAllWindows()
        print("✅ Đã ngắt kết nối.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="Liệt kê các DIGIT đang cắm")
    ap.add_argument("--serial", nargs="+", default=None,
                    help="Serial DIGIT (1 hoặc nhiều, cách nhau dấu cách)")
    args = ap.parse_args()

    if args.list or args.serial is None:
        list_digits()
        if args.serial is None:
            print("\nVí dụ xem ảnh: python3 digit_test.py --serial D20542")
        return

    view_digits(args.serial)


if __name__ == "__main__":
    main()
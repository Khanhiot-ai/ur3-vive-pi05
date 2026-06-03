#!/usr/bin/env python3
"""
check_hdf5.py
=============
Kiểm tra nhanh file HDF5 dataset thu được từ record_all.py

Chạy:
    python3 check_hdf5.py                          # Auto-detect dataset/
    python3 check_hdf5.py dataset/pick_cube.hdf5   # Chỉ rõ file
    python3 check_hdf5.py --demo 0                 # Xem ảnh demo_0
    python3 check_hdf5.py --demo 0 --save          # Lưu ảnh ra /tmp/
"""

import argparse
import os
import sys
import numpy as np

try:
    import h5py
except ImportError:
    print("pip install h5py")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

ok  = lambda s: f"{GREEN}✅ {s}{RESET}"
err = lambda s: f"{RED}❌ {s}{RESET}"
warn = lambda s: f"{YELLOW}⚠️  {s}{RESET}"
info = lambda s: f"{CYAN}{s}{RESET}"


def find_hdf5(path=None):
    """Auto-detect HDF5 file."""
    if path and os.path.exists(path):
        return path
    # Tìm trong dataset/
    for root, dirs, files in os.walk("dataset"):
        for f in files:
            if f.endswith(".hdf5"):
                return os.path.join(root, f)
    return None


def check_dataset(hdf5_path, show_demo=None, save_images=False):
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  HDF5 DATASET INSPECTOR{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"  File: {hdf5_path}")
    print(f"  Size: {os.path.getsize(hdf5_path)/1e6:.1f} MB")

    with h5py.File(hdf5_path, "r") as f:

        # ── Global attrs ──
        print(f"\n{BOLD}── Metadata ─────────────────────────────────{RESET}")
        attrs = dict(f["data"].attrs)
        for k, v in attrs.items():
            print(f"  {k}: {v}")

        # ── List demos ──
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        n_total   = len(demos)
        n_success = sum(1 for d in demos
                        if f["data"][d].attrs.get("success", False))
        n_fail    = n_total - n_success

        print(f"\n{BOLD}── Episodes ─────────────────────────────────{RESET}")
        print(f"  Total:   {n_total}")
        print(f"  {ok(f'Success: {n_success}')}")
        if n_fail > 0:
            print(f"  {warn(f'Fail: {n_fail}')}")

        total_frames = 0
        print(f"\n{BOLD}── Per-episode summary ──────────────────────{RESET}")
        print(f"  {'Demo':<10} {'Frames':<8} {'Duration':<10} {'Status'}")
        print(f"  {'-'*45}")
        for demo_name in demos:
            grp = f["data"][demo_name]
            n_frames  = int(grp.attrs.get("n_frames", 0))
            fps_real  = float(grp.attrs.get("fps_actual", 0))
            duration  = float(grp.attrs.get("duration_s", 0))
            success   = bool(grp.attrs.get("success", False))
            total_frames += n_frames

            status = f"{GREEN}SUCCESS{RESET}" if success else f"{RED}FAIL{RESET}"
            print(f"  {demo_name:<10} {n_frames:<8} {duration:.1f}s @ "
                  f"{fps_real:.1f}fps   {status}")

        fps_cfg = float(f["data"].attrs.get("fps", 10))
        avg_frames = total_frames / max(n_total, 1)
        avg_sec    = avg_frames / fps_cfg
        print(f"\n  Total frames: {total_frames}")
        print(f"  Avg episode:  {avg_frames:.0f} frames = {avg_sec:.1f}s")

        # ── Check schema của demo đầu tiên ──
        print(f"\n{BOLD}── Schema check (demo_0) ────────────────────{RESET}")
        if len(demos) == 0:
            print(f"  {err('Không có demo nào!')}")
            return

        grp0 = f["data"][demos[0]]

        # actions
        if "actions" in grp0:
            arr = np.asarray(grp0["actions"])
            print(f"  {ok(f'actions:        {arr.shape}  dtype={arr.dtype}')}")
            print(f"       min={arr.min():.4f}  max={arr.max():.4f}  "
                  f"mean={arr.mean():.4f}")
        else:
            print(f"  {err('actions: MISSING')}")

        obs = grp0["obs"]

        # state
        if "state" in obs:
            arr = np.asarray(obs["state"])
            print(f"  {ok(f'obs/state:       {arr.shape}  dtype={arr.dtype}')}")
            print(f"       TCP xyz: {arr[0, :3].round(3).tolist()} m")
            print(f"       quat:    {arr[0, 3:7].round(3).tolist()}")
            print(f"       gripper: {arr[0, 7]:.3f}")
        else:
            print(f"  {err('obs/state: MISSING')}")

        # image
        if "image" in obs:
            arr = np.asarray(obs["image"])
            print(f"  {ok(f'obs/image:       {arr.shape}  dtype={arr.dtype}')}")
            if arr.dtype != np.uint8:
                print(f"  {warn('image dtype không phải uint8!')}")
        else:
            print(f"  {err('obs/image: MISSING')}")

        # wrist_image
        if "wrist_image" in obs:
            arr = np.asarray(obs["wrist_image"])
            print(f"  {ok(f'obs/wrist_image: {arr.shape}  dtype={arr.dtype}')}")
        else:
            print(f"  {err('obs/wrist_image: MISSING')}")

        # tactile_state
        if "tactile_state" in obs:
            arr = np.asarray(obs["tactile_state"])
            print(f"  {ok(f'obs/tactile:     {arr.shape}  dtype={arr.dtype}')}")
        else:
            print(f"  {warn('obs/tactile_state: không có (OK nếu không dùng)')}")

        # ── Check action sanity (schema berkeley: 7 dim delta) ──
        print(f"\n{BOLD}── Action sanity check ──────────────────────{RESET}")
        actions = np.asarray(grp0["actions"])
        dxyz = actions[:, :3]    # delta x, y, z
        drpy = actions[:, 3:6]   # delta roll, pitch, yaw
        grip = actions[:, 6]     # gripper (index 6, không phải 7)

        # Delta XYZ range check
        dxyz_total = np.abs(dxyz).sum(axis=0)   # tổng quãng đường mỗi trục
        print(f"  Delta XYZ tổng: {dxyz_total.round(4).tolist()} m")
        if dxyz_total.max() < 0.001:
            print(f"  {err('Delta XYZ ~0 — robot không di chuyển?')}")
        elif dxyz_total.max() > 5.0:
            print(f"  {warn('Delta XYZ tổng lớn — kiểm tra lại')}")
        else:
            print(f"  {ok('Robot có di chuyển (delta Cartesian)')}")

        # Delta RPY range
        drpy_total = np.abs(drpy).sum(axis=0)
        print(f"  Delta RPY tổng: {drpy_total.round(4).tolist()} rad")

        # Gripper range
        print(f"  Gripper: min={grip.min():.3f}  max={grip.max():.3f}  "
              f"(0=mở, 1=kẹp)")
        if grip.max() < 0.1:
            print(f"  {warn('Gripper không đóng — có thu episode gắp vật không?')}")
        elif grip.max() < 0.7:
            print(f"  {warn(f'Gripper max chỉ {grip.max():.2f} — vô lăng chưa kẹp hết?')}")
        else:
            print(f"  {ok('Gripper có đóng mở đầy đủ')}")

        # ── Visualize ảnh nếu yêu cầu ──
        if show_demo is not None:
            demo_key = f"demo_{show_demo}"
            if demo_key not in f["data"]:
                print(f"\n{err(f'{demo_key} không tồn tại')}")
                return

            print(f"\n{BOLD}── Visualize {demo_key} ──────────────────────{RESET}")
            grp_v = f["data"][demo_key]
            obs_v = grp_v["obs"]

            front_frames = np.asarray(obs_v["image"])
            wrist_frames = np.asarray(obs_v["wrist_image"])
            T = len(front_frames)

            print(f"  {T} frames — hiển thị frame 0, T//4, T//2, 3T//4, T-1")
            idxs = sorted(set([0, T//4, T//2, 3*T//4, T-1]))

            for i, idx in enumerate(idxs):
                front = front_frames[idx]  # (H, W, 3) RGB
                wrist = wrist_frames[idx]

                # Convert RGB → BGR cho cv2
                if HAS_CV2:
                    front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
                    wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)

                    if save_images:
                        path_f = f"/tmp/demo{show_demo}_frame{idx:04d}_front.jpg"
                        path_w = f"/tmp/demo{show_demo}_frame{idx:04d}_wrist.jpg"
                        cv2.imwrite(path_f, front_bgr)
                        cv2.imwrite(path_w, wrist_bgr)
                        print(f"  Saved: {path_f}")
                        print(f"  Saved: {path_w}")
                    else:
                        # Hiển thị side-by-side
                        H, W = front_bgr.shape[:2]
                        combined = np.hstack([front_bgr, wrist_bgr])
                        label = f"demo_{show_demo} frame {idx}/{T-1}"
                        cv2.putText(combined, label, (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                        cv2.imshow("Front | Wrist", combined)
                        print(f"  Frame {idx} — nhấn bất kỳ phím để xem frame tiếp")
                        cv2.waitKey(0)

            if HAS_CV2 and not save_images:
                cv2.destroyAllWindows()
        else:
            print(f"\n{BOLD}── Xem ảnh ──────────────────────────────────{RESET}")
            print(f"  Thêm --demo 0 để xem ảnh demo_0")
            print(f"  Thêm --demo 0 --save để lưu ảnh vào /tmp/")

    # ── Final verdict ──
    print(f"\n{BOLD}{'═'*60}{RESET}")
    if n_total == 0:
        print(f"  {err('KHÔNG CÓ DATA — cần thu data trước!')}")
    elif n_success == 0:
        print(f"  {warn('Có data nhưng 0 success — kiểm tra lại pipeline')}")
    elif n_success < 10:
        print(f"  {warn(f'Chỉ có {n_success} success demos — cần thêm (mục tiêu 50+)')}")
    elif n_success < 50:
        print(f"  {warn(f'{n_success}/50 demos — tiếp tục thu thêm')}")
    else:
        print(f"  {ok(f'{n_success} success demos — đủ để convert + train Pi0.5!')}")
    print(f"{BOLD}{'═'*60}{RESET}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", nargs="?", default=None,
                        help="Path to HDF5 file (auto-detect nếu bỏ trống)")
    parser.add_argument("--demo", type=int, default=None,
                        help="Index demo để xem ảnh (vd: --demo 0)")
    parser.add_argument("--save", action="store_true",
                        help="Lưu ảnh vào /tmp/ thay vì hiện cửa sổ")
    args = parser.parse_args()

    path = find_hdf5(args.hdf5)
    if path is None:
        print(err("Không tìm thấy file HDF5!"))
        print("Chạy: python3 check_hdf5.py dataset/pick_cube.hdf5")
        sys.exit(1)

    check_dataset(path, show_demo=args.demo, save_images=args.save)


if __name__ == "__main__":
    main()
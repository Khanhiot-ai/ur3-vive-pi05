#!/usr/bin/env python3
"""
convert_hdf5_to_lerobot.py
==========================
Chuyển HDF5 dataset → LeRobot v2 format (Parquet + MP4).

KHÔNG dùng lerobot.common (API cũ đã bị xóa từ v0.4.x).
Tự tạo file theo đúng LeRobot v2 schema để upload lên HuggingFace.

LeRobot v2 format:
    dataset/
    ├── meta/
    │   ├── info.json           ← metadata tổng
    │   ├── episodes.jsonl      ← danh sách episodes
    │   ├── tasks.jsonl         ← language instructions
    │   └── stats.json          ← mean/std cho normalization
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── observation.image_episode_000000.mp4
            ├── observation.wrist_image_episode_000000.mp4
            └── ...

Chạy:
    python3 convert_hdf5_to_lerobot.py \\
        --src dataset/pick_cube.hdf5 \\
        --out output/ur3_pick_cube \\
        --repo-id qkhanh1/ur3_pick_cube \\
        --task "pick up the red cube" \\
        --fps 10

    # Push sau khi convert
    python3 push_to_huggingface.py --repo-id qkhanh1/ur3_pick_cube
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas chưa cài. Cài bằng: pip install pandas pyarrow")
    sys.exit(1)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    print("⚠️  pyarrow chưa cài. Cài bằng: pip install pyarrow")
    sys.exit(1)


# ── Helpers ──────────────────────────────────────────────────────────────────

def write_video(frames_rgb, output_path, fps=10):
    """Ghi list numpy frames (H,W,3) uint8 thành MP4."""
    if not frames_rgb:
        return False
    H, W = frames_rgb[0].shape[:2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for f in frames_rgb:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    return True


def compute_stats(all_values):
    """Tính mean/std/min/max từ list numpy arrays."""
    arr = np.concatenate([v.reshape(-1, v.shape[-1]) if v.ndim > 1 else v.reshape(-1, 1)
                          for v in all_values], axis=0)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).tolist(),
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
    }


# ── Main convert ──────────────────────────────────────────────────────────────

def convert(args):
    src_path = Path(args.src)
    out_path = Path(args.out)
    fps      = args.fps
    task_str = args.task
    img_height = args.image_height
    img_width  = args.image_width

    if not src_path.exists():
        print(f"❌ Không tìm thấy HDF5: {src_path}")
        sys.exit(1)

    if out_path.exists():
        if args.overwrite:
            shutil.rmtree(out_path)
            print(f"⚠️  Đã xóa: {out_path}")
        else:
            print(f"❌ {out_path} đã tồn tại. Dùng --overwrite để xóa.")
            sys.exit(1)

    # Tạo folders
    (out_path / "meta").mkdir(parents=True)
    (out_path / "data" / "chunk-000").mkdir(parents=True)
    (out_path / "videos" / "chunk-000").mkdir(parents=True)

    # ── Đọc HDF5 ──────────────────────────────────────────────────────────────
    print(f"\n📂 Đọc HDF5: {src_path}")
    with h5py.File(src_path, "r") as f:
        demos = f["data"]
        demo_names = sorted(demos.keys(), key=lambda k: int(k.split("_")[-1]))
        total = len(demo_names)
        print(f"   Tổng demos: {total}")

        # Accumulate stats
        all_states   = []
        all_actions  = []
        episode_list = []
        task_list    = [{"task_index": 0, "task": task_str}]

        frame_global = 0  # frame index toàn dataset

        for ep_idx, demo_name in enumerate(demo_names):
            grp     = demos[demo_name]
            obs     = grp["obs"]
            actions = grp["actions"]

            success  = bool(grp.attrs.get("success", True))
            n_frames = int(grp.attrs.get("n_frames", actions.shape[0]))

            if args.skip_failed and not success:
                print(f"  ⏭  {demo_name} SKIP (fail)")
                continue

            print(f"  {'✅' if success else '❌'} {demo_name}  {n_frames}f", end="")

            # ── Lấy arrays ──
            state_arr   = np.asarray(obs["state"],   dtype=np.float32)   # (T, 8): ee_xyz+quat+grip
            action_arr  = np.asarray(actions,        dtype=np.float32)   # (T, 7): delta_xyz+rpy+grip
            image_arr   = np.asarray(obs["image"])                       # (T,H,W,3)
            wrist_arr   = np.asarray(obs["wrist_image"])                 # (T,H,W,3)

            T = min(n_frames, len(state_arr))

            # ── Resize images về (H, W) = (480, 640) — giống berkeley ──
            target_h, target_w = img_height, img_width
            front_frames = []
            wrist_frames  = []
            for t in range(T):
                img = image_arr[t]
                wri = wrist_arr[t]
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h))   # cv2 lấy (W, H)
                    wri = cv2.resize(wri, (target_w, target_h))
                front_frames.append(img)
                wrist_frames.append(wri)

            # ── Write videos ──
            ep_str = f"episode_{ep_idx:06d}"
            front_mp4 = str(out_path / "videos" / "chunk-000" /
                            f"observation.image_{ep_str}.mp4")
            wrist_mp4 = str(out_path / "videos" / "chunk-000" /
                            f"observation.wrist_image_{ep_str}.mp4")
            write_video(front_frames, front_mp4, fps=fps)
            write_video(wrist_frames, wrist_mp4, fps=fps)

            # ── Build Parquet rows ──
            rows = []
            for t in range(T):
                row = {
                    "episode_index":     ep_idx,
                    "frame_index":       t,
                    "index":             frame_global + t,
                    "timestamp":         round(t / fps, 4),
                    "task_index":        0,
                    # state
                    "observation.state": state_arr[t].tolist(),
                    # action
                    "action":            action_arr[t].tolist(),
                    # video paths (relative)
                    "observation.image": f"videos/chunk-000/observation.image_{ep_str}.mp4",
                    "observation.wrist_image": f"videos/chunk-000/observation.wrist_image_{ep_str}.mp4",
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            parquet_path = out_path / "data" / "chunk-000" / f"{ep_str}.parquet"
            df.to_parquet(parquet_path, index=False)

            # ── Episode metadata ──
            episode_list.append({
                "episode_index": ep_idx,
                "tasks":         [task_str],
                "length":        T,
            })

            # ── Accumulate stats ──
            all_states.append(state_arr[:T])
            all_actions.append(action_arr[:T])
            frame_global += T
            print(f"  → {ep_str}")

    n_episodes = len(episode_list)
    n_frames_total = frame_global

    if n_episodes == 0:
        print("❌ Không có episode nào được convert!")
        sys.exit(1)

    # ── Tính stats ──
    print(f"\n📊 Tính normalization stats...")
    state_stats  = compute_stats(all_states)
    action_stats = compute_stats(all_actions)

    # ── info.json ──
    info = {
        "codebase_version":   "v2.0",
        "robot_type":         "ur3",
        "total_episodes":     n_episodes,
        "total_frames":       n_frames_total,
        "total_tasks":        1,
        "total_chunks":       1,
        "chunks_size":        n_episodes,
        "fps":                fps,
        "splits":             {"train": f"0:{n_episodes}"},
        "data_path":          "data/chunk-{episode_chunk:03d}/{episode_id:06d}.parquet",
        "video_path":         "videos/chunk-{episode_chunk:03d}/{video_key}_episode_{episode_id:06d}.mp4",
        "features": {
            "observation.image": {
                "dtype": "video",
                "shape": [img_height, img_width, 3],
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.wrist_image": {
                "dtype": "video",
                "shape": [img_height, img_width, 3],
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["ee_x", "ee_y", "ee_z",
                          "qx", "qy", "qz", "qw",
                          "gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {
                    "motors": [
                        "dx",
                        "dy",
                        "dz",
                        "d_roll",
                        "d_pitch",
                        "d_yaw",
                        "gripper",
                    ]
                },
            },
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }

    # ── stats.json ──
    stats = {
        "observation.state": state_stats,
        "action":            action_stats,
    }

    # ── Ghi meta files ──
    with open(out_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(out_path / "meta" / "episodes.jsonl", "w") as f:
        for ep in episode_list:
            f.write(json.dumps(ep) + "\n")

    with open(out_path / "meta" / "tasks.jsonl", "w") as f:
        for t in task_list:
            f.write(json.dumps(t) + "\n")

    with open(out_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ──
    print(f"\n{'═' * 55}")
    print(f"✅ HOÀN TẤT")
    print(f"   Episodes:  {n_episodes}")
    print(f"   Frames:    {n_frames_total}")
    print(f"   Image size: {img_height}×{img_width} (giống berkeley)")
    print(f"   Output:    {out_path}")
    print(f"\n   Cấu trúc:")
    for p in sorted(out_path.rglob("*"))[:15]:
        if p.is_file():
            size = p.stat().st_size
            print(f"     {p.relative_to(out_path)}  ({size/1e3:.0f} KB)")

    print(f"\n   Push lên HuggingFace:")
    print(f"   python3 push_to_huggingface.py --repo-id {args.repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",         type=str, required=True,
                        help="HDF5 file (vd: dataset/pick_cube.hdf5)")
    parser.add_argument("--out",         type=str, default=None,
                        help="Output folder (default: dataset/<task>_lerobot)")
    parser.add_argument("--repo-id",     type=str, default="qkhanh1/ur3_pick_cube",
                        help="HF repo id cho lệnh push sau")
    parser.add_argument("--task",        type=str,
                        default="pick up the red cube",
                        help="Language instruction")
    parser.add_argument("--fps",         type=int, default=20)
    parser.add_argument("--image-height",type=int, default=480,
                        help="Image height (default: 480, giống berkeley)")
    parser.add_argument("--image-width", type=int, default=640,
                        help="Image width (default: 640, giống berkeley)")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    if args.out is None:
        base = Path(args.src).stem
        args.out = str(Path(args.src).parent / f"{base}_lerobot")

    print("═" * 55)
    print(f"  HDF5 → LeRobot v2 Converter (berkeley format)")
    print(f"  src:   {args.src}")
    print(f"  out:   {args.out}")
    print(f"  task:  {args.task}")
    print(f"  fps:   {args.fps}")
    print(f"  size:  {args.image_height}×{args.image_width}")
    print("═" * 55)

    convert(args)


if __name__ == "__main__":
    main()
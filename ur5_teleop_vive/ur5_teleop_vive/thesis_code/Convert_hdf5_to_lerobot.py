#!/usr/bin/env python3
"""
convert_hdf5_to_lerobot.py
==========================
Chuyển dataset HDF5 (từ record_all.py) sang định dạng LeRobot cho Pi0.5.

HDF5 schema (input):
    data/
    ├── demo_0/
    │   ├── obs/
    │   │   ├── image         (T, 224, 224, 3) uint8
    │   │   ├── wrist_image   (T, 224, 224, 3) uint8
    │   │   ├── state         (T, 7)  float32  ← joints[6] + grip[1]
    │   │   └── tactile_state (T, 1)  float32  ← torque
    │   ├── actions           (T, 8)  float32  ← xyz[3]+quat[4]+grip[1]
    │   └── attrs: success, n_frames, ...
    └── ...

Chạy:
    pip install lerobot
    python3 convert_hdf5_to_lerobot.py \
        --src dataset/pick_cube.hdf5 \
        --repo-id khanh/ur3_pick_cube \
        --task "pick up the red cube" \
        --fps 10

    # Bỏ demo fail:
    python3 convert_hdf5_to_lerobot.py --src ... --skip-failed
"""

import argparse
import shutil

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def infer_shapes(h5_path):
    with h5py.File(h5_path, "r") as f:
        demos = f["data"]
        first_demo = sorted(demos.keys(), key=lambda k: int(k.split("_")[-1]))[0]
        grp = demos[first_demo]
        obs = grp["obs"]

        image_shape   = tuple(np.asarray(obs["image"][0]).shape)
        wrist_shape   = tuple(np.asarray(obs["wrist_image"][0]).shape)
        state_dim     = np.asarray(obs["state"][0]).shape[0]
        tactile_dim   = np.asarray(obs["tactile_state"][0]).shape[0]
        action_dim    = np.asarray(grp["actions"][0]).shape[0]

    return image_shape, wrist_shape, state_dim, tactile_dim, action_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",        type=str, required=True,
                        help="HDF5 file (vd: dataset/pick_cube.hdf5)")
    parser.add_argument("--repo-id",    type=str, required=True,
                        help="LeRobot repo id (vd: khanh/ur3_pick_cube)")
    parser.add_argument("--robot-type", type=str, default="ur3")
    parser.add_argument("--fps",        type=int, default=10)
    parser.add_argument("--task",       type=str, required=True,
                        help='Language instruction (vd: "pick up the red cube")')
    parser.add_argument("--skip-failed", action="store_true",
                        help="Bỏ qua demo có success=False")
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args()

    print(f"📂 Đọc HDF5: {args.src}")
    image_shape, wrist_shape, state_dim, tactile_dim, action_dim = \
        infer_shapes(args.src)

    print(f"   image_shape  = {image_shape}")
    print(f"   wrist_shape  = {wrist_shape}")
    print(f"   state_dim    = {state_dim}    (joints[6] + gripper[1])")
    print(f"   action_dim   = {action_dim}   (xyz[3] + quat[4] + gripper[1])")
    print(f"   tactile_dim  = {tactile_dim}")

    out_path = HF_LEROBOT_HOME / args.repo_id
    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out_path} đã tồn tại. Dùng --overwrite để xóa.")
        shutil.rmtree(out_path)
        print(f"⚠️  Đã xóa: {out_path}")

    print(f"\n📦 Tạo LeRobot dataset: {args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": wrist_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "tactile_state": {
                "dtype": "float32",
                "shape": (tactile_dim,),
                "names": ["tactile_state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    total_demos = 0
    total_frames = 0
    skipped_demos = 0

    with h5py.File(args.src, "r") as f:
        demos = f["data"]
        demo_names = sorted(demos.keys(), key=lambda k: int(k.split("_")[-1]))

        print(f"\n📊 Có {len(demo_names)} demo trong HDF5")

        for demo_name in demo_names:
            grp = demos[demo_name]
            obs = grp["obs"]
            actions = grp["actions"]

            success = bool(grp.attrs.get("success", True))
            n_frames = int(grp.attrs.get("n_frames", actions.shape[0]))

            # Lọc demo fail nếu chọn
            if args.skip_failed and not success:
                print(f"  ⏭  {demo_name}  SKIP (fail)")
                skipped_demos += 1
                continue

            # Add từng frame
            for t in range(n_frames):
                frame = {
                    "image":         np.asarray(obs["image"][t]),
                    "wrist_image":   np.asarray(obs["wrist_image"][t]),
                    "state":         np.asarray(obs["state"][t],
                                                dtype=np.float32),
                    "tactile_state": np.asarray(obs["tactile_state"][t],
                                                dtype=np.float32),
                    "actions":       np.asarray(actions[t], dtype=np.float32),
                    "task":          args.task,
                }
                dataset.add_frame(frame)

            dataset.save_episode()
            status = "✅" if success else "❌"
            print(f"  {status} {demo_name}  {n_frames} frames")
            total_demos += 1
            total_frames += n_frames

    print(f"\n{'═' * 50}")
    print(f"✅ HOÀN TẤT")
    print(f"   Lưu tại: {out_path}")
    print(f"   Total demos:  {total_demos}")
    print(f"   Total frames: {total_frames}")
    if skipped_demos > 0:
        print(f"   Skipped fail: {skipped_demos}")
    print(f"{'═' * 50}")
    print(f"\nDataset sẵn sàng cho Pi0.5 fine-tune!")
    print(f"\nPush lên HuggingFace Hub:")
    print(f"   huggingface-cli login")
    print(f"   python -c \"from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; "
          f"LeRobotDataset('{args.repo_id}').push_to_hub()\"")


if __name__ == "__main__":
    main()


    
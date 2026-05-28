#!/usr/bin/env python3
"""
push_to_huggingface.py
======================
Push LeRobot dataset lên HuggingFace Hub.

Yêu cầu:
    pip install huggingface_hub lerobot
    huggingface-cli login   ← chạy 1 lần

Chạy:
    python3 push_to_huggingface.py --repo-id qkhanh1/ur3_pick_cube
    python3 push_to_huggingface.py --repo-id qkhanh1/ur3_pick_cube --private
"""

import argparse
import os
import sys
from pathlib import Path


def get_lerobot_home():
    """Tìm thư mục cache của LeRobot."""
    # Thứ tự ưu tiên
    candidates = [
        os.environ.get("HF_LEROBOT_HOME"),
        os.environ.get("LEROBOT_HOME"),
        Path.home() / ".cache" / "huggingface" / "lerobot",
        Path.home() / ".lerobot",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    # Mặc định (chưa tồn tại cũng trả về)
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def push_dataset(repo_id, private=False, src=None):
    from huggingface_hub import HfApi, create_repo

    # ── Auto-detect dataset path ──
    if src:
        dataset_path = Path(src)
    else:
        # Thử các vị trí có thể có dataset
        candidates = []
        lerobot_home = get_lerobot_home()
        candidates.append(lerobot_home / repo_id)
        # Folder từ converter mới: dataset/<task>_lerobot/
        cwd = Path.cwd()
        for pat in ["dataset", "."]:
            base = cwd / pat
            if base.exists():
                for sub in base.iterdir():
                    if sub.is_dir() and sub.name.endswith("_lerobot"):
                        candidates.append(sub)
                    if (sub / "meta" / "info.json").exists():
                        candidates.append(sub)

        dataset_path = None
        for c in candidates:
            if c.exists() and (c / "meta" / "info.json").exists():
                dataset_path = c
                print(f"   Auto-detected: {dataset_path}")
                break

        if dataset_path is None:
            print(f"❌ Không tìm thấy dataset LeRobot.")
            print(f"   Đã tìm tại:")
            for c in candidates:
                print(f"     - {c}")
            print(f"\n   Dùng --src để chỉ rõ folder:")
            print(f"   python3 push_to_huggingface.py --repo-id {repo_id} --src dataset/pick_cube_lerobot/")
            sys.exit(1)

    # ── 1. Verify dataset tồn tại ──
    if not dataset_path.exists():
        print(f"❌ Không tìm thấy dataset: {dataset_path}")
        print(f"   Chạy convert_hdf5_to_lerobot.py trước!")
        sys.exit(1)

    # ── 2. Thống kê ──
    total_size = 0
    file_count = 0
    for f in dataset_path.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size
            file_count += 1

    print(f"📂 Dataset: {dataset_path}")
    print(f"   Files: {file_count}")
    print(f"   Size:  {total_size / 1e6:.1f} MB")

    # ── 3. Tạo repo trên Hub nếu chưa có ──
    api = HfApi()
    print(f"\n🔧 Tạo repo: {repo_id} ({'private' if private else 'public'})")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,   # Không lỗi nếu đã tồn tại
        )
        print(f"   ✅ Repo OK")
    except Exception as e:
        print(f"   ⚠️  {e}")

    # ── 4. Upload toàn bộ folder ──
    print(f"\n🚀 Uploading...")
    print(f"   Destination: https://huggingface.co/datasets/{repo_id}")

    try:
        api.upload_folder(
            folder_path=str(dataset_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload LeRobot dataset from {Path(dataset_path).name}",
        )
        print(f"\n✅ HOÀN TẤT!")
        print(f"   View: https://huggingface.co/datasets/{repo_id}")
        print(f"\n   Trên máy train:")
        print(f"   from lerobot.common.datasets.lerobot_dataset import LeRobotDataset")
        print(f"   dataset = LeRobotDataset('{repo_id}')")
        print(f"\n   Hoặc download thủ công:")
        print(f"   huggingface-cli download {repo_id} --repo-type dataset --local-dir ./dataset")

    except Exception as e:
        print(f"\n❌ Upload thất bại: {e}")
        print(f"\nFix thường gặp:")
        print(f"  1. Chưa login → huggingface-cli login")
        print(f"  2. Token thiếu Write permission → tạo token mới")
        print(f"  3. Tên repo sai format → chỉ dùng a-z, 0-9, _, -")
        print(f"  4. Dataset chưa convert → chạy convert_hdf5_to_lerobot.py")
        sys.exit(1)


def push_via_lerobot(repo_id, private=False):
    """Cách push dùng LeRobot API (backup nếu upload_folder không hoạt động)."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
        print(f"\n🚀 Push via LeRobot API...")
        dataset = LeRobotDataset(repo_id)
        dataset.push_to_hub(private=private)
        print(f"✅ Done: https://huggingface.co/datasets/{repo_id}")
    except ImportError:
        print("❌ LeRobot không tìm thấy — dùng upload_folder thay thế")
        raise
    except Exception as e:
        print(f"❌ LeRobot push thất bại: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Push LeRobot dataset lên HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HF repo id (vd: qkhanh1/ur3_pick_cube)")
    parser.add_argument("--src", type=str, default=None,
                        help="Folder dataset cần push (auto-detect nếu bỏ trống)")
    parser.add_argument("--private", action="store_true",
                        help="Repo private (default: public)")
    parser.add_argument("--method", choices=["folder", "lerobot"], default="folder",
                        help="Cách push: folder=upload_folder, lerobot=LeRobot API")
    args = parser.parse_args()

    print("═" * 55)
    print(f"  PUSH DATASET → HUGGINGFACE HUB")
    print(f"  Repo: {args.repo_id}")
    print(f"  Method: {args.method}")
    print("═" * 55)

    # Verify login
    try:
        from huggingface_hub import whoami
        info = whoami()
        print(f"  Logged in as: {info['name']}\n")
    except Exception:
        print("❌ Chưa login HuggingFace!")
        print("   Chạy: huggingface-cli login")
        sys.exit(1)

    if args.method == "lerobot":
        push_via_lerobot(args.repo_id, args.private)
    else:
        push_dataset(args.repo_id, args.private, args.src)


if __name__ == "__main__":
    main()
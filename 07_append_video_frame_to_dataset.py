# filename: 07_append_video_frame_to_dataset.py
# 从视频中每隔若干帧提取一帧
# 把提取到的各个帧按照一定的比例划分为训练集、验证集与测试集
# 假如文件夹结构为：
# dataset/
# ├── labels/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── images/
#     ├── train/
#     ├── val/
#     └── test/
# 可以append的形式，将这些数据集保存到dataset下的test|train|val三个目录下，
# 假设train目录下原本有0000.jpg ~n 0081.jpg共82张图片，
# 则append后train目录应新增有0082.jpg开始编号的若干张图片。
# 文件由ChatGPT-5.1生成。
# 使用示例：python 07_append_video_frame_to_dataset.py --video random_move.mp4 --dataset dataset/images --interval 10 --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1

import os
import re
import cv2
import random
import argparse

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png')

def ensure_dirs(root):
    for sub in ['train', 'val', 'test']:
        os.makedirs(os.path.join(root, sub), exist_ok=True)

def scan_split_numbers(dataset_root, default_width=4):
    """
    分别扫描 train/val/test 各自目录，找出各自的最大编号与零填充宽度。
    返回:
      next_num: dict(split -> 下一个编号，从最大编号+1开始；若没有，则为 1)
      width:    dict(split -> 零填充宽度；若无法推断，则使用 default_width)
    """
    next_num = {}
    width = {}

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_root, split)
        max_num = 0
        width_guess_for_max = None
        width_candidates = []

        if os.path.isdir(split_dir):
            for name in os.listdir(split_dir):
                if not name.lower().endswith(SUPPORTED_EXTS):
                    continue
                stem, _ = os.path.splitext(name)
                nums = re.findall(r'\d+', stem)
                if not nums:
                    continue
                for token in nums:
                    try:
                        num = int(token)
                    except ValueError:
                        continue
                    if num > max_num:
                        max_num = num
                        width_guess_for_max = len(token)
                    width_candidates.append(len(token))

        next_num[split] = max_num + 1 if max_num > 0 else 1
        if max_num > 0 and width_guess_for_max is not None:
            width[split] = width_guess_for_max
        elif width_candidates:
            width[split] = max(width_candidates)
        else:
            width[split] = default_width

    return next_num, width

def choose_split(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("train/val/test 比例之和必须大于 0")
    tr = train_ratio / total
    vr = val_ratio / total
    r = random.random()
    if r < tr:
        return 'train'
    elif r < tr + vr:
        return 'val'
    else:
        return 'test'

def format_number(n, width):
    return f"{n:0{width}d}"

def stem_exists_in_any_ext(dest_dir, stem):
    for ext in SUPPORTED_EXTS:
        if os.path.exists(os.path.join(dest_dir, stem + ext)):
            return True
    return False

def extract_and_split(video_path, dataset_root, interval,
                      train_ratio, val_ratio, test_ratio,
                      seed=42, default_width=4, output_ext='.png'):
    """
    按间隔抽帧，并按比例分配到 dataset/train|val|test。
    每个子目录独立延续编号与宽度（互不干扰）。
    """
    random.seed(seed)
    ensure_dirs(dataset_root)

    # 为各子目录分别确定自己的编号与宽度
    next_num, width = scan_split_numbers(dataset_root, default_width=default_width)
    print("编号初始化：")
    for sp in ['train', 'val', 'test']:
        print(f"  {sp}: 从 {format_number(next_num[sp], width[sp])} 开始，宽度 {width[sp]}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息 - 总帧数: {total_frames}, FPS: {fps:.2f} (若显示 -1 表示无法获取)")

    saved_count = {'train': 0, 'val': 0, 'test': 0}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 interval 帧取一帧
        take_this = True if interval <= 1 else (frame_idx % interval == 0)

        if take_this:
            split = choose_split(train_ratio, val_ratio, test_ratio)
            dest_dir = os.path.join(dataset_root, split)

            # 为该 split 使用它自己的编号与宽度
            candidate_stem = format_number(next_num[split], width[split])

            # 若同 stem 不论扩展名是否已存在，都顺延；避免 0021.jpg 与 0021.png 同时存在
            while stem_exists_in_any_ext(dest_dir, candidate_stem):
                next_num[split] += 1
                candidate_stem = format_number(next_num[split], width[split])

            filename = candidate_stem + output_ext
            out_path = os.path.join(dest_dir, filename)

            ok = cv2.imwrite(out_path, frame)
            if not ok:
                print(f"警告: 保存失败 -> {out_path}")
            else:
                saved_count[split] += 1
                next_num[split] += 1

                total_saved = sum(saved_count.values())
                if total_saved % 50 == 0:
                    print(f"已保存 {total_saved} 张图片。最近一张: [{split}] {out_path}")

        frame_idx += 1

    cap.release()
    print("完成。各 split 保存统计：")
    for sp in ['train', 'val', 'test']:
        last_num = next_num[sp] - 1
        print(f"  {sp}: 保存 {saved_count[sp]} 张，最后编号为 {format_number(last_num, width[sp])}")

def main():
    parser = argparse.ArgumentParser(description="从视频抽帧并按比例分配到数据集，每个目录独立延续编号")
    parser.add_argument("--video", type=str, default="random_move.mp4", help="输入视频路径")
    parser.add_argument("--dataset", type=str, default="dataset/images", help="数据集根目录（包含 train/val/test）")
    parser.add_argument("--interval", type=int, default=5, help="抽帧间隔（每隔多少帧取一帧）")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现的划分")
    parser.add_argument("--ext", type=str, default=".png", choices=[".jpg", ".jpeg", ".png"], help="输出图片扩展名")
    parser.add_argument("--default_width", type=int, default=4, help="无法推断编号宽度时的默认零填充宽度")
    args = parser.parse_args()

    extract_and_split(
        video_path=args.video,
        dataset_root=args.dataset,
        interval=args.interval,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        default_width=args.default_width,
        output_ext=args.ext
    )

if __name__ == "__main__":
    main()

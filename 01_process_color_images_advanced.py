# filename: 01_process_color_images_advanced.py
# DeepSeek自动生成的程序
# 识别原始文件夹中的所有命名带有color（不分大小写）的png文件，把它们打散后，按比例分配到dataset/image下的train/val/test文件夹中

import os
import shutil
import re
from pathlib import Path
import random

def process_color_images_advanced(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    批量处理color图像文件, 并自动分割为train/val/test集
    
    参数:
    train_ratio: 训练集比例 (默认0.7)
    val_ratio: 验证集比例 (默认0.2) 
    test_ratio: 测试集比例 (默认0.1)
    """
    # 检查比例总和是否为1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于1")
    
    dataset_raw_path = Path("dataset_raw/data_new") # 原始数据集根目录
    output_base_path = Path("dataset/images")  # 输出图像目录
    
    train_path = output_base_path / "train"
    val_path = output_base_path / "val" 
    test_path = output_base_path / "test"
    
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有包含color的PNG文件
    color_files = []
    
    # 遍历dataset_raw下的所有子目录
    for root, dirs, files in os.walk(dataset_raw_path):
        for file in files:
            if 'color' in file.lower() and file.lower().endswith('.png'): 
                color_files.append(Path(root) / file)
    
    print(f"找到 {len(color_files)} 个color图像文件")
    
    # 按文件名中的数字排序（确保顺序正确）
    def extract_number(filename):
        numbers = re.findall(r'\d+', str(filename))
        return int(numbers[0]) if numbers else 0
    
    color_files.sort(key=extract_number)
    
    # 随机打乱文件顺序（确保数据分布均匀）
    random.shuffle(color_files)
    
    # 计算各数据集的数量
    total_files = len(color_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count
    
    print(f"数据集分割: 训练集 {train_count}, 验证集 {val_count}, 测试集 {test_count}")
    
    # 分割数据集
    train_files = color_files[:train_count]
    val_files = color_files[train_count:train_count + val_count]
    test_files = color_files[train_count + val_count:]
    
    # 处理训练集
    print("\n处理训练集...")
    for i, file_path in enumerate(train_files):
        new_filename = f"{i:04d}.png"
        new_file_path = train_path / new_filename
        shutil.copy2(file_path, new_file_path)
        print(f"训练集: {file_path.name} -> {new_filename}")
    
    # 处理验证集
    print("\n处理验证集...")
    for i, file_path in enumerate(val_files):
        new_filename = f"{i:04d}.png"
        new_file_path = val_path / new_filename
        shutil.copy2(file_path, new_file_path)
        print(f"验证集: {file_path.name} -> {new_filename}")
    
    # 处理测试集
    print("\n处理测试集...")
    for i, file_path in enumerate(test_files):
        new_filename = f"{i:04d}.png"
        new_file_path = test_path / new_filename
        shutil.copy2(file_path, new_file_path)
        print(f"测试集: {file_path.name} -> {new_filename}")
    
    print(f"\n处理完成！")
    print(f"训练集: {len(train_files)} 个文件 -> {train_path.absolute()}")
    print(f"验证集: {len(val_files)} 个文件 -> {val_path.absolute()}")
    print(f"测试集: {len(test_files)} 个文件 -> {test_path.absolute()}")

if __name__ == "__main__":
    # 使用默认比例运行
    process_color_images_advanced()

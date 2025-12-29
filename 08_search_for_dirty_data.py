# filename: 08_search_for_dirty_data.py
# 由Qwen-3-Coder生成
# 功能：查找指定文件夹下超过1行的txt文件
# 使用方法：
# python 08_search_for_dirty_data.py --folder ./dataset/labels [--recursive]

import os
from pathlib import Path

def find_multi_line_txt_files(folder_path, recursive=False):
    """
    查找指定文件夹下超过1行的txt文件
    
    Args:
        folder_path (str): 要搜索的文件夹路径
        recursive (bool): 是否递归搜索子文件夹
    
    Returns:
        list: 包含超过1行的txt文件路径列表
    """
    folder_path = Path(folder_path)
    multi_line_files = []
    
    # 根据recursive参数决定搜索范围
    if recursive:
        txt_files = folder_path.rglob("*.txt")  # 递归搜索所有子目录
    else:
        txt_files = folder_path.glob("*.txt")   # 只搜索当前目录
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_count = len(lines)
                
                # 如果文件有超过1行，则添加到结果列表
                if line_count > 1:
                    multi_line_files.append((str(txt_file), line_count))
                    print(f"发现多行文件: {txt_file} (行数: {line_count})")
        
        except Exception as e:
            print(f"读取文件时出错 {txt_file}: {e}")
    
    return multi_line_files

def main():
    # 默认搜索当前目录下的dataset/labels文件夹
    default_folder = "./dataset/labels"
    
    # 如果用户提供了命令行参数，则使用该参数作为搜索路径
    import argparse
    parser = argparse.ArgumentParser(description='查找超过1行的txt文件')
    parser.add_argument('folder', nargs='?', default=default_folder, 
                        help='要搜索的文件夹路径 (默认: ./dataset/labels)')
    parser.add_argument('--recursive', '-r', action='store_true', 
                        help='递归搜索子文件夹')
    
    args = parser.parse_args()
    
    print(f"正在搜索文件夹: {args.folder}")
    if args.recursive:
        print("搜索模式: 递归搜索子文件夹")
    else:
        print("搜索模式: 仅搜索当前文件夹")
    
    # 执行搜索
    multi_line_files = find_multi_line_txt_files(args.folder, args.recursive)
    
    print(f"\n总共找到 {len(multi_line_files)} 个超过1行的txt文件")
    
    if multi_line_files:
        print("\n详细信息:")
        for file_path, line_count in multi_line_files:
            print(f"  {file_path}: {line_count} 行")
    else:
        print("未找到超过1行的txt文件")

if __name__ == "__main__":
    main()
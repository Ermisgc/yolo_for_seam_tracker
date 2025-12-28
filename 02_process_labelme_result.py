# filename: 02_process_labelme_result.py
# 由DeepSeek自动生成
# 功能：处理labelme标注结果，labelme这一步生成的应该是.json
# 本脚本用于将.json生成yolov8格式的.txt标注文件
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
# 那么可以这样调用本脚本：
# python 02_process_labelme_result.py -i ./dataset/labels --recursive

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class LabelMe2YOLOv8:
    def __init__(self, class_mapping: Dict[str, int] = None):
        """
        初始化转换器
        
        Args:
            class_mapping: 类别名称到ID的映射字典
                         例如: {"person": 0, "car": 1, "dog": 2}
                         如果为None，将自动从所有JSON文件中收集
        """
        self.class_mapping = class_mapping or {}
        self.reverse_mapping = {}
        
    def parse_labelme_json(self, json_path: str) -> Tuple[Tuple[int, int], List[Dict]]:
        """
        解析LabelMe JSON文件
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            (image_width, image_height), 标注列表
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        annotations = []
        for shape in data['shapes']:
            label = shape['label']
            shape_type = shape['shape_type']
            points = shape['points']
            
            if shape_type == 'rectangle':
                # 矩形: 两个点 [左上x, 左上y, 右下x, 右下y]
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 确保坐标顺序
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                
                bbox = [x_min, y_min, x_max, y_max]
                annotations.append({
                    'label': label,
                    'bbox': bbox,
                    'type': 'rectangle'
                })
                
            elif shape_type == 'polygon':
                # 多边形: 取所有点的最小外接矩形
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                bbox = [x_min, y_min, x_max, y_max]
                annotations.append({
                    'label': label,
                    'bbox': bbox,
                    'type': 'polygon'
                })
                
            elif shape_type == 'circle':
                # 圆形: 转换为外接矩形
                center_x, center_y = points[0]
                radius_x, radius_y = points[1]
                
                # 计算半径
                radius = ((radius_x - center_x)**2 + (radius_y - center_y)**2)**0.5
                
                x_min = center_x - radius
                y_min = center_y - radius
                x_max = center_x + radius
                y_max = center_y + radius
                
                bbox = [x_min, y_min, x_max, y_max]
                annotations.append({
                    'label': label,
                    'bbox': bbox,
                    'type': 'circle'
                })
                
            else:
                print(f"警告: 跳过不支持的形状类型: {shape_type}")
        
        return (image_width, image_height), annotations
    
    def convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        将边界框转换为YOLO格式(归一化的中心坐标和宽高)
        
        Args:
            bbox: [x_min, y_min, x_max, y_max]
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            [x_center, y_center, width, height] 归一化到0-1
        """
        x_min, y_min, x_max, y_max = bbox
        
        # 边界检查
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))
        
        # 计算中心点坐标
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        
        # 计算宽度和高度
        width = x_max - x_min
        height = y_max - y_min
        
        # 归一化
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # 确保值在0-1范围内
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        return [x_center_norm, y_center_norm, width_norm, height_norm]
    
    def get_class_id(self, label: str) -> int:
        """
        获取类别ID，如果不存在则自动添加
        
        Args:
            label: 类别名称
            
        Returns:
            类别ID
        """
        if label not in self.class_mapping:
            # 自动添加新类别
            new_id = len(self.class_mapping)
            self.class_mapping[label] = new_id
            print(f"添加新类别: '{label}' -> ID: {new_id}")
        
        return self.class_mapping[label]
    
    def convert_file(self, json_path: str, output_path: str = None) -> bool:
        """
        转换单个JSON文件
        
        Args:
            json_path: 输入JSON文件路径
            output_path: 输出TXT文件路径，如果为None则使用相同目录和名称
            
        Returns:
            是否成功转换
        """
        try:
            # 解析JSON文件
            (img_width, img_height), annotations = self.parse_labelme_json(json_path)
            
            if not annotations:
                print(f"警告: {json_path} 中没有找到标注")
                return False
            
            # 确定输出路径
            if output_path is None:
                output_path = str(Path(json_path).with_suffix('.txt'))
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 写入YOLO格式
            with open(output_path, 'w', encoding='utf-8') as f:
                for ann in annotations:
                    yolo_bbox = self.convert_bbox_to_yolo(ann['bbox'], img_width, img_height)
                    class_id = self.get_class_id(ann['label'])
                    
                    # 写入格式: class_id x_center y_center width height
                    line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                    f.write(line)
            
            print(f"✓ 转换完成: {json_path} -> {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ 转换失败 {json_path}: {e}")
            return False
    
    def convert_folder_recursive(self, input_dir: str, output_dir: str = None) -> Dict:
        """
        递归转换文件夹中的JSON文件
        
        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径，如果为None则使用输入文件夹
            
        Returns:
            统计信息字典
        """
        input_path = Path(input_dir).resolve()
        if output_dir is None:
            output_path = input_path
        else:
            output_path = Path(output_dir).resolve()
        
        # 递归查找所有JSON文件
        json_files = list(input_path.rglob("*.json"))
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 转换统计
        stats = {
            'total': len(json_files),
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        # 转换每个文件
        for json_file in json_files:
            # 计算相对路径
            try:
                relative_path = json_file.relative_to(input_path)
            except ValueError:
                # 如果无法计算相对路径，使用完整路径
                relative_path = Path(json_file.name)
            
            # 构建输出路径
            if output_dir is None:
                # 在原目录生成
                output_file = json_file.with_suffix('.txt')
            else:
                # 在输出目录中保持相同的目录结构
                output_file = output_path / relative_path.with_suffix('.txt')
            
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换文件
            if self.convert_file(str(json_file), str(output_file)):
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['failed_files'].append(str(json_file))
        
        return stats
    
    def convert_folder_non_recursive(self, input_dir: str, output_dir: str = None) -> Dict:
        """
        非递归转换文件夹中的JSON文件（仅顶层目录）
        
        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径，如果为None则使用输入文件夹
            
        Returns:
            统计信息字典
        """
        input_path = Path(input_dir).resolve()
        if output_dir is None:
            output_path = input_path
        else:
            output_path = Path(output_dir).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 查找顶层目录的JSON文件
        json_files = list(input_path.glob("*.json"))
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 转换统计
        stats = {
            'total': len(json_files),
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        # 转换每个文件
        for json_file in json_files:
            if output_dir is None:
                output_file = json_file.with_suffix('.txt')
            else:
                output_file = output_path / f"{json_file.stem}.txt"
            
            if self.convert_file(str(json_file), str(output_file)):
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['failed_files'].append(str(json_file))
        
        return stats
    
    def save_class_mapping(self, output_path: str):
        """
        保存类别映射到文件
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # 保存为YOLO格式的classes.txt
            # 首先按ID排序
            sorted_items = sorted(self.class_mapping.items(), key=lambda x: x[1])
            for label, class_id in sorted_items:
                f.write(f"{label}\n")
        
        print(f"✓ 类别映射已保存到: {output_path}")
        
        # 同时保存反向映射（可选）
        reverse_path = output_path.replace('.txt', '_reverse.txt')
        with open(reverse_path, 'w', encoding='utf-8') as f:
            for label, class_id in sorted_items:
                f.write(f"{class_id}: {label}\n")
        
        print(f"✓ 反向映射已保存到: {reverse_path}")


def main():
    parser = argparse.ArgumentParser(
        description='将LabelMe JSON格式转换为YOLOv8 TXT格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  1. 递归转换文件夹:
     python labelme2yolo.py --input data/annotations --recursive
  
  2. 递归转换并指定输出目录:
     python labelme2yolo.py --input data/annotations --output data/yolo_labels --recursive
  
  3. 使用预定义的类别列表:
     python labelme2yolo.py --input data/annotations --classes classes.txt --recursive
  
  4. 仅转换单个文件:
     python labelme2yolo.py --input image.json
  
  5. 仅转换顶层文件夹(非递归):
     python labelme2yolo.py --input data/annotations
        """
    )
    parser.add_argument('--input', '-i', required=True, 
                       help='输入路径（单个JSON文件或包含JSON文件的文件夹）')
    parser.add_argument('--output', '-o', 
                       help='输出路径（单个TXT文件或输出文件夹）')
    parser.add_argument('--classes', '-c', 
                       help='类别映射文件路径（可选，格式：每行一个类别名）')
    parser.add_argument('--save-classes', action='store_true', 
                       help='保存类别映射到文件')
    parser.add_argument('--class-map', 
                       help='手动指定类别映射（例如："person:0,car:1,dog:2"）')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='递归处理子文件夹（仅在输入为文件夹时有效）')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 初始化类别映射
    class_mapping = {}
    
    # 1. 从文件加载类别映射
    if args.classes and os.path.exists(args.classes):
        with open(args.classes, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                if class_name:
                    class_mapping[class_name] = idx
        print(f"从文件加载了 {len(class_mapping)} 个类别")
    
    # 2. 从命令行参数加载类别映射
    elif args.class_map:
        mappings = args.class_map.split(',')
        for mapping in mappings:
            if ':' in mapping:
                class_name, class_id = mapping.split(':')
                class_mapping[class_name.strip()] = int(class_id.strip())
        print(f"从命令行加载了 {len(class_mapping)} 个类别")
    
    # 初始化转换器
    converter = LabelMe2YOLOv8(class_mapping)
    
    # 判断输入类型
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.json':
        # 单个文件转换
        output_path = args.output
        converter.convert_file(str(input_path), output_path)
        
    elif input_path.is_dir():
        # 文件夹批量转换
        if args.recursive:
            stats = converter.convert_folder_recursive(str(input_path), args.output)
        else:
            stats = converter.convert_folder_non_recursive(str(input_path), args.output)
        
        # 打印统计信息
        print("\n" + "="*50)
        print(f"转换统计:")
        print(f"  总计: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")
        
        if stats['failed'] > 0 and args.verbose:
            print(f"  失败文件:")
            for file in stats['failed_files']:
                print(f"    - {file}")
    else:
        print(f"错误: 输入路径 '{args.input}' 不是有效的JSON文件或文件夹")
        return
    
    # 保存类别映射
    if args.save_classes:
        if args.output and Path(args.output).is_dir():
            classes_path = Path(args.output) / "classes.txt"
        elif input_path.is_dir():
            classes_path = input_path / "classes.txt"
        else:
            classes_path = input_path.parent / "classes.txt"
        
        converter.save_class_mapping(str(classes_path))
    
    # 打印最终的类别映射
    if converter.class_mapping:
        print("\n" + "="*50)
        print("类别映射:")
        sorted_items = sorted(converter.class_mapping.items(), key=lambda x: x[1])
        for label, class_id in sorted_items:
            print(f"  {class_id}: {label}")


if __name__ == "__main__":
    main()
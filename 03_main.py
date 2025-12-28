# filename: 03_main.py
# 由DeepSeek自动生成
# 功能：使用yolov8n.pt模型训练weld数据集
# 数据集weld.yaml已在02_process_labelme_result.py中处理过
# 使用示例：
# 基础训练：python 03_main.py --data weld.yaml --epochs 300 --batch 8
# 使用更多增强：python 03_main.py --mosaic 0.9 --mixup 0.2 --degrees 15

import torch, os, argparse
from ultralytics import YOLO


def setup_environment():
    """设置训练环境"""
    # 设置随机种子
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True  # 强制CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN基准测试，确保确定性
    
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def train_weld_detection(args):
    """训练焊缝检测模型"""
    device = setup_environment()
    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)
    args.name = args.name.split('.')[0]
    # 3. 训练参数
    train_args = {
        'data': args.data,          # 数据集配置文件路径
        'epochs': args.epochs,      # 增加训练轮次，小数据集需要更多迭代
        'patience': args.patience,  # 早停耐心值，若干个epoch无改善则停止
        'batch': args.batch,        # 批处理大小
        'imgsz': args.imgsz,        # 输入图像尺寸
        'device': device,           # 使用设置的设备
        
        # 优化器
        'lr0': args.lr0,            # 初始学习率
        'lrf': args.lrf,            # 最终学习率倍率(lr0 * lrf)
        'momentum': args.momentum,  # 动量
        'weight_decay': args.weight_decay,  # 权重衰减
        
        # 数据增强
        'hsv_h': args.hsv_h,        # HSV色调增强
        'hsv_s': args.hsv_s,        # HSV饱和度范围
        'hsv_v': args.hsv_v,        # HSV明度增强
        'degrees': args.degrees,    # 旋转角度范围(+/- deg)
        'translate': args.translate,# 平移比例
        'scale': args.scale,        # 缩放比例
        'fliplr': args.fliplr,      # 左右翻转概率
        'mosaic': args.mosaic,      # 马赛克增强概率
        'erasing': args.erasing,    # 随机擦除概率，建议为0.1，太大可能会把焊缝擦除了
        
        # 正则化
        'dropout': args.dropout,    # Dropout概率
        
        # 其他，项目配置
        'project': args.project,    # 项目名称，也对应设定的项目目录
        'name': args.name,          # 模型名称
        'exist_ok': True,           # 是否覆盖已有项目
        'verbose': True,            # 是否显示详细输出
        'cos_lr': args.cos_lr,      # 是否使用余弦退火学习率调度
        'amp': args.amp,            # 是否使用自动混合精度训练，节省显存
        'close_mosaic': args.close_mosaic,  # 是否在最后阶段关闭马赛克增强
    }
    
    # 打印配置
    print("\n训练配置:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(**train_args)
    
    # 验证最佳模型
    print("\n验证最佳模型...")
    best_model = YOLO(f"{args.project}/{args.name}/weights/best.pt")
    metrics = best_model.val()
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description='YOLOv8焊缝检测训练')
    
    # 路径参数
    parser.add_argument('--data', type=str, default='weld.yaml', help='数据配置文件')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='预训练模型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300, help='训练轮次')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--batch', type=int, default=8, help='批量大小')
    parser.add_argument('--imgsz', type=int, default=960, help='图像尺寸')
    parser.add_argument('--device', type=str, default='', help='设备')
    
    # 优化器参数
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率因子')
    parser.add_argument('--momentum', type=float, default=0.937, help='动量')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    
    # 数据增强参数
    parser.add_argument('--hsv_h', type=float, default=0.015, help='HSV色调增强')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='HSV饱和度增强')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='HSV明度增强')
    parser.add_argument('--degrees', type=float, default=5.0, help='旋转角度')
    parser.add_argument('--translate', type=float, default=0.05, help='平移比例')
    parser.add_argument('--scale', type=float, default=0.3, help='缩放比例')
    parser.add_argument('--fliplr', type=float, default=0.5, help='左右翻转概率')
    parser.add_argument('--mosaic', type=float, default=0.3, help='马赛克增强概率')
    parser.add_argument('--erasing', type=float, default=0.1, help='随机擦除概率，建议为0.1，太大可能会把焊缝擦除了')
    
    # 正则化参数
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout概率')
    
    # 其他参数
    parser.add_argument('--project', type=str, default='weld_seam_detection', help='项目名称')
    parser.add_argument('--name', type=str, default='yolov8s_v1', help='模型名称')
    parser.add_argument('--cos_lr', type=bool, default=True, help='使用余弦退火')
    parser.add_argument('--amp', type=bool, default=True, help='自动混合精度')
    parser.add_argument('--close_mosaic', type=int, default=10, help='最后N个epoch关闭马赛克')
    
    args = parser.parse_args()
    
    # 执行训练
    results, metrics = train_weld_detection(args)
    
    print("\n训练完成！")
    print(f"最佳模型保存在: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
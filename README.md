# YOLO for Seam Tracker

## 项目简介
本项目旨在使用深度学习技术对焊缝进行初步识别检测。项目基于 YOLOv8 框架，实现了从数据预处理到模型训练、验证和导出的完整流程。后续可以将本框架运用到更多的目标检测任务中。

## 功能特性

- **数据预处理**：将 LabelMe 标注的 JSON 文件转换为 YOLOv8 格式的标注文件
- **数据集处理**：自动将图像文件按比例分割为训练集、验证集和测试集
- **模型训练**：使用 YOLOv8 框架训练焊缝检测模型
- **模型导出**：将训练好的模型导出为 ONNX 格式
- **视频检测**：对视频文件进行焊缝检测
- **环境检查**：验证 PyTorch、CUDA 和 Ultralytics 库的可用性

## 后续部署工作
将导出的`TensorRT`的`engine`文件放到部署项目[seam_tracker](https://github.com/Ermisgc/seam_tracker)的根目录下，然后调用相关方法

## 文件结构
```
yolo_for_seam_tracker/
├── 00_hello.py # 环境检查脚本
├── 01_process_color_images_advanced.py # 图像数据预处理
├── 02_process_labelme_result.py # LabelMe 标注结果转换
├── 03_main.py # 模型训练主脚本
├── 04_export_to_onnx.py # 模型导出为 ONNX
├── 05_verify_onnx.py # ONNX 模型验证
├── 06_detect_video.py # 视频检测
├── 07_append_video_frame_to_dataset.py # 将视频帧添加到数据集
├── weld.yaml # 数据集配置文件
├── dataset/ # 数据集目录
│ └── images/
│ ├── train/
│ ├── val/
│ └── test/
└── README.md
```
## 安装依赖

在开始使用本项目前，请确保安装了以下依赖：
```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
```
或者直接运行 `requirements.txt` 安装依赖：

```bash
pip install -r requirements.txt
```
## 使用方法

### 1. 环境检查

运行以下命令检查环境：

```bash
python 00_hello.py
```

### 2. 数据预处理

处理包含 "color" 关键字的图像文件并分割为训练集、验证集和测试集：

```bash
python 01_process_color_images_advanced.py
```

### 3. 标注文件转换
首先需要用LabelMe标注数据，直接在命令行中执行：
```bash
labelme
```
标注时需要**注意**：
- 标注时需要将焊缝区域框起来，并为其分配类别 "weld"
- 标注完成后，将 JSON 文件保存到 `./dataset/labels` 目录下

将 LabelMe 格式的 JSON 标注文件转换为 YOLOv8 格式：
```bash
python 02_process_labelme_result.py -i ./dataset/labels --recursive
```

### 4. 模型训练

使用以下命令进行模型训练：

```bash
python 03_main.py --data weld.yaml --epochs 300 --batch 8
```

调参时可以灵活改变训练参数，示例如下，具体可以查看 `03_main.py` 中的参数说明：

```bash
python 03_main.py --mosaic 0.9 --mixup 0.2 --degrees 15
```

### 5. 模型导出

将训练好的模型导出为 ONNX 格式：

```bash
python 04_export_to_onnx.py
```

### 6. ONNX 模型验证

验证上一步导出的 ONNX 模型是否正确：

```bash
python 05_verify_onnx.py
```

### 7. 视频检测

对视频文件进行焊缝检测：

```bash
python 06_detect_video.py
```

### 8. 模型导出为TensorRT的`Engine`
把路径调整为模型的路径，例如：`weld_seam_detection/yolov8s_v1/weights/`
然后用`TensorRT`自带的`trtexec`工具导出`Engine`：
```bash
trtexec --onnx=weld_seam_detection/yolov8s_v1/weights/best.onnx --saveEngine=weld_seam_detection/yolov8s_v1/weights/yolov8s_fp32.engine
```
**注意**，如果是float16精度，需要在导出时添加参数`--fp16`：
```bash
trtexec --onnx=weld_seam_detection/yolov8s_v1/weights/best.onnx --saveEngine=weld_seam_detection/yolov8s_v1/weights/yolov8s_fp16.engine --fp16
```
如果要在C++中使用engine，更推荐在C++中直接获得加载engine，而不是在Python中加载。

## 训练参数说明

在 `03_main.py` 中提供了丰富的训练参数：

- `--data`: 数据集配置文件路径
- `--model`: 预训练模型
- `--epochs`: 训练轮次
- `--batch`: 批处理大小
- `--imgsz`: 输入图像尺寸
- `--lr0`: 初始学习率
- `--hsv_h`, `--hsv_s`, `--hsv_v`: HSV 颜色空间增强参数
- `--degrees`, `--translate`, `--scale`: 几何变换增强参数
- `--mosaic`, `--fliplr`: 数据增强参数

## 数据集配置

`weld.yaml` 文件定义了数据集的路径和类别信息：

```yaml
path: ./dataset    # 数据集根目录
train: images/train  
val: images/val
test: images/test

names:
  0: weld
```

## 项目特点

- **自动化流程**：从数据预处理到模型训练的完整自动化流程
- **灵活配置**：支持多种训练参数和数据增强选项
- **多格式支持**：支持多种标注格式到 YOLO 格式的转换
- **性能优化**：支持混合精度训练和 GPU 加速

## 注意事项

1. 确保有足够的 GPU 内存来训练模型
2. 根据你的数据集大小和硬件配置调整 `batch` 大小
3. 对于小数据集，可能需要增加训练轮次以获得更好的效果
4. 在数据增强参数中，特别是 `erasing` 参数不宜设置过大，以免影响焊缝特征
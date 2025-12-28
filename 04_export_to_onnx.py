# filename: 04_export_to_onnx.py
# DeepSeek生成的程序
# 将训练好的模型导出为ONNX格式，因为后续要部署在TensorRT上
# 这里要修改模型名称

from ultralytics import YOLO
import torch


model_path = 'weld_seam_detection/yolov8s_v1/weights/best.pt'  # TODO: 模型名称要修改
model = YOLO(model_path) 

# 导出为ONNX（推荐设置）
model.export(
    format='onnx',           # 导出格式
    imgsz=960,               # 输入尺寸（与训练一致）
    batch=1,                 # 批处理大小（部署时通常为1）
    half=False,              # 使用FP16精度量化  # TODO: 部署时是否需要FP16
    simplify=True,           # 简化模型
    opset=17,                # ONNX算子集版本
    dynamic=False,           # 固定输入尺寸（更简单）
    workspace=4,             # GPU内存限制(GB)
    device='cuda',           # 在GPU上导出
    verbose=True,
)

print(f"✅ ONNX导出完成: {model_path.split('/')[-2]}.onnx") 
print("输入尺寸:", (1, 3, 960, 960))
print("输出尺寸:", (1, 5, 18900))
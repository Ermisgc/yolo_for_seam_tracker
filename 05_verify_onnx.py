# filename: 05_verify_onnx.py
# 验证ONNX模型, 检查输入输出是否符合预期，检查模型精度与原模型是否一致
# 代码由Gemini-3-Pro生成
# 注意修改模型名称和输入的大小、是否要进行FP16量化等参数，可查找：TODO

import onnx
import onnxruntime as ort
import numpy as np
import torch
from ultralytics import YOLO
import sys
import os
import cv2


PT_MODEL_PATH = 'weld_seam_detection/yolov8s_v2/weights/best.pt'  # 原模型
ONNX_MODEL_PATH = 'weld_seam_detection/yolov8s_v2/weights/best.onnx'  # 导出的ONNX模型
IMAGE_PATH = 'test_img.png'  # 测试图像
IMG_SIZE = (960, 960)
IS_FP16 = False  # TODO: 部署时是否需要FP16，需要与onnx模型导出时一致


def preprocess_image(image_path, input_size):
    """
    读取图片并进行预处理：Resize -> RGB -> Normalize -> CHW -> Batch
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("图片读取失败，请检查格式")

    img_resized = cv2.resize(img_bgr, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0

    # (H, W, C) -> (C, H, W)
    img_transposed = img_norm.transpose(2, 0, 1)

    # 增加 Batch 维度 -> (1, C, H, W)
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    # 转换为连续内存数组 (防止内存不连续导致的报错)
    img_batch = np.ascontiguousarray(img_batch)

    return img_batch


def verify_onnx():
    print(f"🖼️  正在处理图片: {IMAGE_PATH}")
    
    # 准备输入数据 
    try:
        # 预处理得到 numpy (float32)
        input_numpy = preprocess_image(IMAGE_PATH, IMG_SIZE)
        print(f"✅ 图片预处理完成: Shape={input_numpy.shape}, Range=[{input_numpy.min():.2f}, {input_numpy.max():.2f}]")
    except Exception as e:
        print(f"❌ 图片处理错误: {e}")
        return

    # PyTorch 推理
    print("运行 PyTorch (原模型)...")
    try:
        pt_model = YOLO(PT_MODEL_PATH)
        # 切换到 GPU 和 FP16
        if IS_FP16:
            pt_model.model.to('cuda').half()
        else:
            pt_model.model.to('cuda')
            
        pt_model.model.eval()

        # 将 numpy 转为 tensor，并转为 fp16
        if IS_FP16:
            input_tensor = torch.from_numpy(input_numpy).cuda().half()
        else:
            input_tensor = torch.from_numpy(input_numpy).cuda()
        with torch.no_grad():
            # 获取原始输出
            pt_output = pt_model.model(input_tensor)[0]
        
        pt_result = pt_output.cpu().numpy()
        print(f"   PyTorch 输出尺寸: {pt_result.shape}")
        
    except Exception as e:
        print(f"❌ PyTorch 推理失败: {e}")
        return

    # === 3. ONNX 推理 ===
    print("运行 ONNX Runtime (导出模型)...")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # 转换为 FP16 的 numpy 数组
        if IS_FP16:
            onnx_input = input_numpy.astype(np.float16)
        else:
            onnx_input = input_numpy.astype(np.float32)

        onnx_result = session.run([output_name], {input_name: onnx_input})[0]
        print(f"   ONNX    输出尺寸: {onnx_result.shape}")

    except Exception as e:
        print(f"❌ ONNX 推理失败: {e}")
        print("提示: 确保安装了 onnxruntime-gpu，并且 CUDA 库路径配置正确。")
        return

    # === 4. 精度对比 ===
    print("开始对比精度...")
    
    # 检查 NaN
    if np.isnan(pt_result).any() or np.isnan(onnx_result).any():
        print("❌ 错误: 输出包含 NaN，验证失败。")
        return

    # 展平
    pt_flat = pt_result.flatten().astype(np.float32)
    onnx_flat = onnx_result.flatten().astype(np.float32)

    # 余弦相似度
    cos_sim = np.dot(pt_flat, onnx_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(onnx_flat))
    
    # 最大误差
    max_diff = np.max(np.abs(pt_flat - onnx_flat))
    
    # 平均误差
    mean_diff = np.mean(np.abs(pt_flat - onnx_flat))

    print("-" * 30)
    print(f"📊 验证报告:")
    print(f"   余弦相似度 (Cosine Similarity): {cos_sim:.6f}  (理想值 > 0.99)")
    print(f"   最大绝对误差 (Max Abs Diff)   : {max_diff:.6f}")
    print(f"   平均绝对误差 (Mean Abs Diff)  : {mean_diff:.6f}")
    print("-" * 30)

    if cos_sim > 0.99:
        print("✅ 成功: ONNX 模型与 PyTorch 原模型精度一致！")
    else:
        print("⚠️ 警告: 精度差异较大，请检查导出参数或 TensorRT 版本兼容性。")

if __name__ == "__main__":
    verify_onnx()
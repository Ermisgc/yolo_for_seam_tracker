# filename: 06_detect_vedio.py
# 由ChatGPT-5.1生成
# 视频检测, 输入视频，调用yolo模型，检测后输出结果视频
# 注意：
# 1. 修改模型路径
# 2. 修改输入视频路径
# 3. 在main函数输入时改变不同的preview参数，可选择是否实时预览
# 4. 模型推理时，置信度阈值为0.5，可根据需要调整

import cv2
from ultralytics import YOLO
import torch

def main(preview: bool = False):
    weights_path = "weld_seam_detection/yolov8s_v1/weights/best.pt"  # TODO:修改权重路径

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(weights_path)
    model.to(device)

    input_video = "./random_move.mp4" # TODO:修改输入视频路径
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"无法打开视频: {input_video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # 输出视频文件名，由视频文件自动生成
    output_video = input_video.replace(".mp4", "_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或 "XVID"
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 逐帧读取并检测
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读完了

        frame_idx += 1

        # Ultralytics YOLO 推理，单帧输入
        results = model.predict(
            source=frame,     # 直接传 ndarray
            device=device,    # GPU / CPU
            conf=0.8,        # TODO:置信度阈值，可根据需要调整
            verbose=False
        )

        # results 是一个列表，通常长度为 1（因为我们只传了一帧）
        result = results[0]

        # result.boxes 包含所有预测框 
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # box.xyxy 是 [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 置信度和类别
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # 类别名称（如果模型里有）
                if result.names and cls_id in result.names:
                    cls_name = result.names[cls_id]
                else:
                    cls_name = f"id_{cls_id}"

                label = f"{cls_name} {conf:.2f}"

                # 画框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 画标签
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # 把带框的这一帧写入输出视频
        out.write(frame)

        if preview:
            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                break

        if frame_idx % 50 == 0:
            print(f"已处理帧数: {frame_idx}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成，输出视频已保存为: {output_video}")

if __name__ == "__main__":
    main(True)  # TODO: 选择是否实时预览

# filename: 00_hello.py
# Hello World程序，检查torch、cuda、ultralytics是否可用

import torch
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    print(torch.__version__)

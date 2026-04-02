#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 实时目标检测 - Jetson Orin Nano 优化版
作者：霜叶
"""

import cv2
import torch
from ultralytics import YOLO
import time
from datetime import datetime

# ================= 配置参数 =================
WEIGHTS_PATH = "/home/jetson/workspace/yolo26/YOLO26_41/weights/best.pt"
CAMERA_ID = 0  # /dev/video0
CONF_THRESHOLD = 0.5
IMG_SIZE = 640  # 可根据性能调整，如 320/480/640
SKIP_FRAMES = 1  # 每 N 帧检测一次，提升 FPS
DISPLAY_FPS = True
SAVE_VIDEO = False  # 是否保存输出视频
OUTPUT_PATH = "output_{}.mp4".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

# ================= 初始化模型 =================
print("正在加载 YOLO 模型...")
start_time = time.time()
model = YOLO(WEIGHTS_PATH)
# 可选：启用 TensorRT 加速（需提前导出为 .engine 文件）
# model = YOLO("best.engine", task="detect")
print(f"模型加载完成，耗时：{time.time() - start_time:.2f} 秒")

# 设置模型参数
model.conf = CONF_THRESHOLD
model.imgsz = IMG_SIZE

# 检查是否使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前运行设备：{device}")

# ================= 初始化摄像头 =================
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"错误：无法打开摄像头 {CAMERA_ID}")
    exit(1)

# 设置摄像头参数（可根据实际调整）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# 可选：保存视频
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (1280, 720))

# ================= 实时检测主循环 =================
print("开始实时检测，按 'q' 退出...")
frame_count = 0
fps_start_time = time.time()
fps_frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        frame_count += 1
        fps_frame_count += 1

        # 每隔 SKIP_FRAMES 帧进行一次推理
        if frame_count % SKIP_FRAMES == 0:
            results = model(frame, verbose=False, device=device)
            result = results[0]

            # 绘制检测结果
            annotated_frame = result.plot()
        else:
            annotated_frame = frame

        # 显示 FPS
        if DISPLAY_FPS:
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                fps_start_time = time.time()
                fps_frame_count = 0

        # 显示画面
        cv2.imshow("YOLO Real-Time Detection", annotated_frame)

        # 保存视频
        if SAVE_VIDEO:
            out.write(annotated_frame)

        # 按键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n用户中断")
finally:
    cap.release()
    if SAVE_VIDEO:
        out.release()
    cv2.destroyAllWindows()
    print("程序结束")
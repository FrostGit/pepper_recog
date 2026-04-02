#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 实时目标检测 - Jetson Orin Nano 优化版
作者：霜叶
功能：获取目标中心坐标 + 区域过滤 (ROI)
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

# ================= 新增：有效范围 (ROI) 配置 =================
# 格式：左上角 (x1, y1), 右下角 (x2, y2)
# 请根据实际画面分辨率 (1280x720) 调整此处坐标
ROI_ENABLED = True       # 是否启用区域过滤
ROI_X1, ROI_Y1 = 400, 150  # 区域左上角
ROI_X2, ROI_Y2 = 900, 600 # 区域右下角
# ===========================================

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

        # 创建绘制画面副本
        annotated_frame = frame.copy()

        # 1. 绘制有效范围区域 (ROI) 背景框，方便调试查看
        if ROI_ENABLED:
            cv2.rectangle(annotated_frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, "ROI Zone", (ROI_X1, ROI_Y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 每隔 SKIP_FRAMES 帧进行一次推理
        if frame_count % SKIP_FRAMES == 0:
            results = model(frame, verbose=False, device=device)
            result = results[0]

            # 获取检测框数据
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    # 获取坐标 (tensor -> numpy)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[cls_id]

                    # ================= 获取中心坐标 =================
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    # ===============================================

                    # ================= 区域过滤逻辑 =================
                    if ROI_ENABLED:
                        # 判断中心点是否在 ROI 范围内
                        if not (ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2):
                            continue  # 如果不在范围内，跳过此目标，不绘制
                    # ===============================================

                    # 绘制符合条件的目标框
                    color = (45, 150, 232)  # 绿色表示有效目标
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # 绘制类别和置信度
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 打印中心坐标到控制台 (可选)
                    print(f"Target: {class_name}, Center: ({cx}, {cy})")
        else:
            # 跳帧时不检测，直接显示上一帧或原帧 (这里保持原帧加 ROI 框)
            pass

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
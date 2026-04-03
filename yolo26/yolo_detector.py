#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 实时目标检测 - 多线程封装版
作者：霜叶
功能：获取目标中心坐标 + 区域过滤 (ROI) + 多线程队列输出
"""

import cv2
import torch
import threading
import queue
import time
from datetime import datetime
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict, Any

class YOLODetector:
    def __init__(self, 
                 weights_path: str = "/home/jetson/workspace/yolo26/YOLO26_41/weights/best.pt",
                 camera_id: int = 0,
                 conf_threshold: float = 0.5,
                 img_size: int = 640,
                 skip_frames: int = 1,
                 roi_enabled: bool = True,
                 roi_coords: Tuple[int, int, int, int] = (400, 150, 900, 600),
                 queue_size: int = 5):
        """
        初始化检测器
        :param weights_path: 模型权重路径
        :param camera_id: 摄像头 ID
        :param conf_threshold: 置信度阈值
        :param img_size: 推理图像尺寸
        :param skip_frames: 跳帧检测 (每 N 帧检测一次)
        :param roi_enabled: 是否启用 ROI 过滤
        :param roi_coords: ROI 坐标 (x1, y1, x2, y2)
        :param queue_size: 结果队列最大长度 (防止内存溢出)
        """
        self.weights_path = weights_path
        self.camera_id = camera_id
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.skip_frames = skip_frames
        self.roi_enabled = roi_enabled
        self.roi_coords = roi_coords  # (x1, y1, x2, y2)
        
        self.model = None
        self.cap = None
        self.running = False
        self.thread = None
        self.result_queue = queue.Queue(maxsize=queue_size)
        
        # 统计信息
        self.fps = 0.0
        self.frame_count = 0

    def _load_model(self):
        """加载模型"""
        print("正在加载 YOLO 模型...")
        start_time = time.time()
        self.model = YOLO(self.weights_path)
        self.model.conf = self.conf_threshold
        self.model.imgsz = self.img_size
        print(f"模型加载完成，耗时：{time.time() - start_time:.2f} 秒")

    def _open_camera(self):
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"错误：无法打开摄像头 {self.camera_id}")
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def _inference_loop(self):
        """内部检测循环 (运行在独立线程)"""
        frame_count = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                time.sleep(0.1)
                continue

            frame_count += 1
            fps_frame_count += 1
            
            # 创建绘制画面副本
            annotated_frame = frame.copy()
            current_results = []

            # 1. 绘制 ROI 背景框 (调试用)
            if self.roi_enabled:
                x1, y1, x2, y2 = self.roi_coords
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "ROI Zone", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 2. 推理逻辑
            if frame_count % self.skip_frames == 0:
                results = self.model(frame, verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")
                result = results[0]
                boxes = result.boxes

                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[cls_id]

                        # 获取中心坐标
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # 区域过滤逻辑
                        if self.roi_enabled:
                            roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_coords
                            if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                                continue  # 跳过不在范围内的目标

                        # 绘制符合条件的目标框
                        color = (45, 150, 232)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # 保存结果数据
                        detection_data = {
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': (cx, cy),
                            'cls_id': cls_id
                        }
                        current_results.append(detection_data)
            
            # 3. 计算 FPS
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                self.fps = fps_frame_count / elapsed
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # # 在画面上写入 FPS (仅用于返回的画面，不影响逻辑)
            # cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 4. 将结果放入队列 (非阻塞，如果队列满则丢弃旧帧，保证实时性)
            try:
                # 如果队列满了，先获取一个旧的再放入新的，保持最新
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put_nowait((annotated_frame, current_results, self.fps))
            except queue.Full:
                pass

        # 循环结束清理
        if self.cap:
            self.cap.release()

    def start(self):
        """启动检测线程"""
        if self.running:
            return
        self._load_model()
        self._open_camera()
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print("检测线程已启动")

    def stop(self):
        """停止检测线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print("检测线程已停止")

    def get_result(self, timeout: float = 1.0) -> Optional[Tuple[Any, List[Dict], float]]:
        """
        获取最新的检测结果
        :param timeout: 等待队列数据的超时时间
        :return: (annotated_frame, results_list, fps) 或 None (如果超时)
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop()

# ================= 使用示例 (多线程演示) =================
if __name__ == "__main__":
    def display_thread(detector: YOLODetector):
        """模拟一个专门负责显示的线程"""
        print("显示线程启动...")
        while True:
            # 从检测器获取结果
            data = detector.get_result(timeout=1.0)
            if data is None:
                continue
            
            frame, results, fps = data
            
            # 在这里处理结果，例如打印或进一步逻辑
            if results:
                # 只打印第一个目标示例，避免刷屏
                # print(f"检测到：{results[0]['class_name']} 中心：{results[0]['center']}")
                pass
            
            # 显示画面 (注意：OpenCV 的 imshow 最好在主线程或专用 GUI 线程调用)
            cv2.imshow("YOLO Multi-Thread Display", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("显示线程退出")

    # 主程序
    try:
        # 1. 实例化检测器
        detector = YOLODetector(
            weights_path="/home/jetson/workspace/yolo26/YOLO26_41/weights/best.pt",
            roi_enabled=True,
            roi_coords=(400, 150, 900, 600)
        )

        # 2. 启动检测后台线程
        detector.start()

        # 3. 启动显示线程 (模拟多线程使用)
        # 实际使用中，你可以在这里启动多个业务逻辑线程消费 detector.get_result()
        display_t = threading.Thread(target=display_thread, args=(detector,))
        display_t.start()

        # 4. 主线程等待
        display_t.join()

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        detector.stop()
        print("程序结束")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辣椒分拣统计系统 - Flask 后端
"""

import os
import time
import threading
import logging
import queue
import signal
import sys
from collections import deque
from flask import Flask, send_from_directory, Response, request, jsonify
from functools import wraps
import psutil
import cv2
import numpy as np

# 导入您的 YOLO 检测器（根据实际路径调整）
from yolo26.yolo_detector import YOLODetector
# 导入机械臂库
from robot_arm_lib import RobotArmController

# ================= 配置区 =================
AUTH_KEY = "Tianjin HuaDa Tech"
VIDEO_SOURCE = 0  # 0=默认摄像头，也可改为 'test.mp4' 或 RTSP 地址
MODEL_PATH = "/home/jetson/workspace/yolo26/YOLO26_41/weights/best.pt"
ROI_ENABLED = True
ROI_COORDS = (400, 150, 900, 600)  # (x1, y1, x2, y2)
HOST = '0.0.0.0'
PORT = 5000

# 辣椒类别映射：模型输出 → 前端类别 key
CLASS_MAPPING = {
    "millet_dried_red": "millet_dr",
    "millet_fresh_red": "millet_fr", 
    "millet_fresh_green": "millet_fg",
    "lantern_round": "lantern",
    "zunyi_large": "zunyi_l",
    "zunyi_small": "zunyi_s",
}

# 各类辣椒绘制颜色 (BGR)
PEPPER_COLORS = {
    "millet_dr": (30, 30, 220),
    "millet_fr": (50, 80, 240),
    "millet_fg": (80, 200, 80),
    "lantern":   (50, 200, 220),
    "zunyi_l":   (180, 80, 220),
    "zunyi_s":   (220, 150, 240),
}

# 机械臂放置位置配置 (机械臂坐标系)
PLACING_POSITIONS = {
    "millet_dr": {"x": 250.0, "y": 50.0},   # 示例位置，需要根据实际调整
    "millet_fr": {"x": 250.0, "y": 0.0},
    "millet_fg": {"x": 250.0, "y": -50.0},
    "lantern":   {"x": 200.0, "y": 50.0},
    "zunyi_l":   {"x": 200.0, "y": 0.0},
    "zunyi_s":   {"x": 200.0, "y": -50.0},
}

# 机械臂姿态配置
GRIPPER_OPEN = 2.12  # 张开夹爪姿态
GRIPPER_CLOSED = 3.14   # 闭合夹爪姿态
GRIPPER_HOLD = 2.95      # 夹取姿态
GRIP_POSE = {
    "z": -119.8,
    "t": 0.0,
    "r": -0.20,
    "g": GRIPPER_OPEN
}

INITIAL_POSE = {
    "x": 91.24,
    "y": 15.54,
    "z": 112.5,
    "t": 0.0,
    "r": -0.145,
    "g": GRIPPER_CLOSED # 夹爪闭合姿态
}

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['JSON_AS_ASCII'] = False

# 全局退出标志
shutdown_event = threading.Event()

# 全局退出标志
shutdown_event = threading.Event()

# 机械臂控制器 (全局实例)
robot_arm = None

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ================= 优雅退出处理 =================
def signal_handler(signum, frame):
    """信号处理函数：退出"""
    logger.info(f"收到信号 {signum}，开始退出...")
    shutdown_event.set()
    
    # 停止检测器
    detector_mgr.stop()
    
    # 停止状态
    with state.lock:
        state.is_recognizing = False
        state.is_sorting = False
        state.is_stream_paused = False
    
    logger.info("所有资源已清理，程序即将退出")
    sys.exit(0)


# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ================= 坐标转换函数 =================
def convert_yolo_to_arm(x, y):
    """
    将YOLO检测的像素坐标转换为机械臂坐标
    X = 0.6242x - 195.16
    Y = 0.6242y - 195.16
    """
    X = 0.6242 * x - 195.16
    Y = -0.6085 * y + 270.42
    return X, Y


# ================= 状态管理 (线程安全) =================
class DetectionState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # 运行状态
        self.is_recognizing = False
        self.is_sorting = False
        self.is_stream_paused = False  # 视频暂停标志
        self.start_time = None
        
        # 系统资源 (由监控线程更新)
        self.cpu = 0.0
        self.memory = 0.0
        self.fps = 0.0  # 视频流 FPS (非画面绘制)
        self.signal = 95
        
        # 计数统计
        self.counts = {key: 0 for key in CLASS_MAPPING.values()}
        self.total_processed = 0
        self.error_count = 0
        
        # 置信度统计
        self.conf_stats = {"high": 0, "medium": 0, "low": 0, "total": 0}
        
        # 最近识别记录
        self.recent_detections = deque(maxlen=20)
        
        # 视频流相关
        self._frame_count = 0
        self._fps_last_time = time.time()
        self._last_frame = None  # 缓存最后一帧 (用于暂停时显示)
        
        # 结果队列
        self.result_queue = queue.Queue(maxsize=10)


state = DetectionState()


# ================= 鉴权中间件 =================
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"success": False, "error": "auth_failed", "message": "缺少鉴权 Token"}), 401
        token = auth_header.split('Bearer ', 1)[1]
        if token != AUTH_KEY:
            return jsonify({"success": False, "error": "auth_failed", "message": "鉴权失败"}), 403
        return f(*args, **kwargs)
    return decorated


# ================= YOLO 检测器管理 =================
class DetectorManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.detector = None
        self._running = False
        self._thread = None
        
    def start(self):
        if self._running:
            return
        try:
            logger.info(f"🔧 初始化 YOLODetector: {MODEL_PATH}")
            self.detector = YOLODetector(
                weights_path=MODEL_PATH,
                roi_enabled=ROI_ENABLED,
                roi_coords=ROI_COORDS
            )
            self.detector.start()
            self._running = True
            self._thread = threading.Thread(target=self._consume_results, daemon=True, name="YOLOConsumer")
            self._thread.start()
            logger.info("✅ YOLODetector 已启动")
        except Exception as e:
            logger.error(f"[ERROR]-启动 YOLODetector 失败: {e}")
            self._running = False
            
    def stop(self):
        if not self._running:
            return
        self._running = False
        if self.detector:
            self.detector.stop()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("YOLODetector 已停止")
        
    def _consume_results(self):
        while self._running and not shutdown_event.is_set():
            try:
                data = self.detector.get_result(timeout=0.5)
                if data is None:
                    continue
                frame, results, fps = data
                if state.result_queue.full():
                    try:
                        state.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                state.result_queue.put_nowait({"frame": frame, "results": results, "fps": fps})
            except Exception as e:
                logger.error(f"检测结果消费异常: {e}")
                time.sleep(0.1)
                
    def get_latest_result(self, timeout=0.1):
        try:
            return state.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def is_ready(self):
        return self._running and self.detector is not None


detector_mgr = DetectorManager()


# ================= 视频流生成器 (✅ 移除中文绘制 + ✅ 真正暂停) =================
def generate_frames():
    """MJPEG 视频流 仅统计不绘制"""
    
    if not detector_mgr.is_ready():
        logger.warning("⚠️ YOLODetector 未就绪，启用模拟画面")
        while not shutdown_event.is_set():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "YOLODetector Unavailable", (80, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)
        return

    while not shutdown_event.is_set():
        # 处理暂停逻辑：暂停时返回缓存的最后一帧
        if state.is_stream_paused:
            if state._last_frame is not None:
                ret, buffer = cv2.imencode('.jpg', state._last_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)  # 降低暂停时的资源占用
            continue
        
        result = detector_mgr.get_latest_result(timeout=0.1)
        
        if result is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        else:
            frame, results, fps = result["frame"], result["results"], result["fps"]
            
            # 更新视频流 FPS (仅统计，不绘制到画面)
            state._frame_count += 1
            now = time.time()
            if now - state._fps_last_time >= 1.0:
                state.fps = state._frame_count / (now - state._fps_last_time)
                state._frame_count = 0
                state._fps_last_time = now
            
            # 仅在识别/分拣模式下处理检测结果
            if state.is_recognizing or state.is_sorting:
                _process_detections(frame, results)
            
            # 缓存最后一帧 (用于暂停时显示)
            state._last_frame = frame.copy()
        
        # 添加简洁的状态指示 (仅右上角小圆点 + FPS 数值，无中文)
        _add_minimal_overlay(frame)
        
        # 编码推流
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def _process_detections(frame, results):
    """处理检测结果： 仅在识别模式下绘制彩色框 + 更新统计"""
    if not results:
        return
        
    for det in results:
        class_name = det.get("class_name", "")
        confidence = det.get("confidence", 0.0)
        bbox = det.get("bbox", {})
        
        # 映射到前端类别
        front_class = CLASS_MAPPING.get(class_name)
        if not front_class:
            continue
        
        # 仅在识别模式下绘制检测框
        if state.is_recognizing:
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            color = PEPPER_COLORS.get(front_class, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 更新统计 (线程安全)
        with state.lock:
            state.counts[front_class] += 1
            state.total_processed += 1
            
            if confidence >= 0.9:
                state.conf_stats["high"] += 1
            elif confidence >= 0.7:
                state.conf_stats["medium"] += 1
            else:
                state.conf_stats["low"] += 1
            state.conf_stats["total"] += 1
            
            state.recent_detections.appendleft({
                "class": front_class,
                "confidence": round(confidence, 3),
                "timestamp": time.time()
            })


def _add_minimal_overlay(frame):
    """ 最小化叠加层：仅右上角状态点 + FPS 数值 (无中文)"""
    h, w = frame.shape[:2]
    
    # 状态指示点 (左上角)
    status_color = (0, 255, 0) if state.is_recognizing else (200, 200, 200)
    cv2.circle(frame, (15, 15), 6, status_color, -1)
    
    # FPS 数值 (右上角，仅数字)
    cv2.putText(frame, f"{state.fps:.1f}", (w - 45, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)




def sorting_thread():
    """分拣执行线程：获取第一个识别结果并模拟机械臂抓取动作"""
    logger.info("🤖 分拣线程 [sorting_thread] 已启动，等待识别结果...")
    
    try:
        while not shutdown_event.is_set() and state.is_sorting:
            try:
                if robot_arm and robot_arm.is_connected:
                    # 1. 机械臂先移动到初始位置
                    x = INITIAL_POSE["x"]
                    y = INITIAL_POSE["y"]
                    logger.info(f"移动到初始位置: X={x:.2f}, Y={y:.2f}")
                    success = robot_arm.move_xyzt_goal(
                        x=INITIAL_POSE["x"], y=INITIAL_POSE["y"],
                        z=INITIAL_POSE["z"], t=INITIAL_POSE["t"],
                        r=INITIAL_POSE["r"], g=INITIAL_POSE["g"]
                    )
                    time.sleep(1.5)  # 等待移动完成
            except Exception as e:
                logger.error(f"❌ 机械臂控制异常: {e}")
                continue
            
            result = detector_mgr.get_latest_result(timeout=0.5)
            if result is None or not result.get("results"):
                continue
                
            frame, results, fps = result["frame"], result["results"], result["fps"]
            first_det = results[0]
            class_name = first_det.get("class_name", "")
            confidence = first_det.get("confidence", 0.0)
            bbox = first_det.get("bbox", {})
            
            front_class = CLASS_MAPPING.get(class_name)
            if not front_class and class_name in CLASS_MAPPING.values():
                front_class = class_name
            if not front_class:
                for orig, mapped in CLASS_MAPPING.items():
                    if class_name in orig or orig in class_name:
                        front_class = mapped
                        break
            
            if not front_class or front_class not in PEPPER_COLORS:
                logger.warning(f"⚠️ 无法识别的类别: '{class_name}'")
                with state.lock:
                    state.error_count += 1
                continue
            
            # 提取坐标
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                # 如果是字典格式（兼容旧格式）
                x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
                x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            logger.info(f"🎯 目标: {front_class} | 置信度: {confidence:.2f} | 像素坐标: ({center_x}, {center_y})")
            
            # 坐标转换：像素 -> 机械臂坐标
            arm_x, arm_y = convert_yolo_to_arm(center_x, center_y)
            logger.info(f"🔄 转换后机械臂坐标: ({arm_x:.2f}, {arm_y:.2f})")
            
            # 获取放置位置
            place_pos = PLACING_POSITIONS.get(front_class)
            if not place_pos:
                logger.error(f"❌ 未定义 {front_class} 的放置位置")
                continue
            
            # 机械臂抓取动作
            try:
                if robot_arm and robot_arm.is_connected:
                    logger.info("开始机械臂抓取流程...")
                    
                    # TODO:


                    # ---------------------------------------------------------------
                    # 1. 移动到抓取位置 (使用抓取姿态)
                    success = robot_arm.move_joints_rad(base=0.0583, 
                                                        shoulder=-0.8314, 
                                                        elbow=2.5295, 
                                                        wrist=0.1611, 
                                                        roll=-0.0522,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 移动到抓取位置失败")
                        continue
                    time.sleep(2)  # 等待移动完成
                    
                    # ---------------------------------------------------------------
                    # 2. 闭合夹爪 (模拟)
                    logger.info("🔧 闭合夹爪")
                    success = robot_arm.move_joints_rad(base=0.2684, 
                                                        shoulder=0.6719, 
                                                        elbow=1.8577, 
                                                        wrist=0.6443, 
                                                        roll=0.2178,
                                                        hand=1.5)
                    if not success:
                        logger.error("❌ 移动到抓取位置失败")
                        continue
                    time.sleep(2)  # 等待移动完成
                    
                    logger.info("🔧 闭合夹爪")
                    success = robot_arm.move_joints_rad(base=0.2684, 
                                                        shoulder=0.6719, 
                                                        elbow=1.8577, 
                                                        wrist=0.6443, 
                                                        roll=0.2178,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 移动到抓取位置失败")
                        continue
                    time.sleep(2)  # 等待移动完成
                    # 3. 抬起物品 (移动到安全高度)
                    logger.info("🔧 抬起物品")
                    success = robot_arm.move_joints_rad(base=0.4326, 
                                                        shoulder=-0.1227, 
                                                        elbow=1.6475, 
                                                        wrist=-0.1810, 
                                                        roll=-0.0169,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 抬起物品失败")
                        continue
                    time.sleep(2)
                    
                    
                    # 4. 移动到放置位置
                    success = robot_arm.move_joints_rad(base=1.2701, 
                                                        shoulder=-0.0583, 
                                                        elbow=2.4160, 
                                                        wrist=0.1672, 
                                                        roll=-0.0522,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 移动到放置位置失败")
                        continue
                    time.sleep(2)
                    
                    # 5. 释放物品 (张开夹爪)
                    success = robot_arm.move_joints_rad(base=1.2701, 
                                                        shoulder=-0.0583, 
                                                        elbow=2.4160, 
                                                        wrist=0.1672, 
                                                        roll=-0.0522,
                                                        hand=1.5)
                    logger.info("🔧 释放物品")
                    # 这里可以添加夹爪控制
                    time.sleep(0.5)
                    # ---------------------------------------------------------------
                    # 1. 移动到抓取位置 (使用抓取姿态)
                    success = robot_arm.move_joints_rad(base=0.0583, 
                                                        shoulder=-0.8314, 
                                                        elbow=2.5295, 
                                                        wrist=0.1611, 
                                                        roll=-0.0522,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 移动到抓取位置失败")
                        continue
                    time.sleep(2)  # 等待移动完成
                    
                    # ---------------------------------------------------------------
                    # 2. 闭合夹爪 (模拟)
                    logger.info("🔧 闭合夹爪")
                    success = robot_arm.move_joints_rad(base=-0.2117, 
                                                        shoulder=0.6581, 
                                                        elbow=1.8699, 
                                                        wrist=0.5507, 
                                                        roll=-0.2347,
                                                        hand=1.5)
                    if not success:
                        logger.error("❌ 移动到抓取位置失败")
                        continue
                    time.sleep(2)  # 等待移动完成
                    
                    logger.info("🔧 闭合夹爪")
                    success = robot_arm.move_joints_rad(base=-0.2102, 
                                                        shoulder=0.6581, 
                                                        elbow=1.8699, 
                                                        wrist=0.5507, 
                                                        roll=-0.2347,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 移动到抓取位置失败")
                        continue
                    time.sleep(2)  # 等待移动完成
                    # 3. 抬起物品 (移动到安全高度)
                    logger.info("🔧 抬起物品")
                    success = robot_arm.move_joints_rad(base=-0.2102, 
                                                        shoulder=-0.1365, 
                                                        elbow=1.9236, 
                                                        wrist=0.4786, 
                                                        roll=-0.2378,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 抬起物品失败")
                        continue
                    time.sleep(2)
                    
                    
                    # 4. 移动到放置位置
                    success = robot_arm.move_joints_rad(base=-0.2715, 
                                                        shoulder=0.8376, 
                                                        elbow=1.1137, 
                                                        wrist=0.1273, 
                                                        roll=-0.0491,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 移动到放置位置失败")
                        continue
                    time.sleep(2)
                    
                    # 5. 释放物品 (张开夹爪)
                    success = robot_arm.move_joints_rad(base=-0.2715, 
                                                        shoulder=0.8376, 
                                                        elbow=1.1137, 
                                                        wrist=0.1273, 
                                                        roll=-0.0491,
                                                        hand=1.5)
                    logger.info("🔧 释放物品")
                    # 这里可以添加夹爪控制
                    time.sleep(0.5)
                    # ---------------------------------------------------------------
                    # 1. 移动到抓取位置 (使用抓取姿态)
                    success = robot_arm.move_joints_rad(base=0.0583, 
                                                        shoulder=-0.8314, 
                                                        elbow=2.5295, 
                                                        wrist=0.1611, 
                                                        roll=-0.0522,
                                                        hand=3.14)
                    if not success:
                        logger.error("❌ 移动到抓取位置失败")
                        continue
                    time.sleep(2)  # 等待移动完成



                    
                    logger.info(f"✅ 分拣完成: {front_class} -> ({place_pos['x']:.2f}, {place_pos['y']:.2f})")
                    
                    
                  # FIXME:  
                else:
                    logger.warning("⚠️ 机械臂未连接，使用模拟动作")
                    # 模拟动作
                    for action in [
                        f"移动到 ({arm_x:.2f}, {arm_y:.2f})",
                        "下降至抓取高度",
                        f"闭合夹爪，抓取 [{front_class}]",
                        "抬起物品",
                        f"移动至分拣区 ({place_pos['x']:.2f}, {place_pos['y']:.2f})",
                        "释放物品 ✓"
                    ]:
                        logger.info(f"🔧 机械臂: {action}")
                        time.sleep(0.2)
            
            except Exception as e:
                logger.error(f"❌ 机械臂控制异常: {e}")
                continue
            
            # 更新统计（线程安全）
            with state.lock:
                state.counts[front_class] += 1
                state.total_processed += 1
                if confidence >= 0.9:
                    state.conf_stats["high"] += 1
                elif confidence >= 0.7:
                    state.conf_stats["medium"] += 1
                else:
                    state.conf_stats["low"] += 1
                state.conf_stats["total"] += 1
                state.recent_detections.appendleft({
                    "class": front_class,
                    "confidence": round(confidence, 3),
                    "timestamp": time.time()
                })
            
            logger.info(f"📊 统计: {front_class}+1 | 总计: {state.total_processed}")
            break  # 单次分拣完成，如需连续请注释此行
            
    except Exception as e:
        logger.error(f"❌ 分拣线程异常: {e}", exc_info=True)
        with state.lock:
            state.error_count += 1
    finally:
        logger.info("🏁 分拣线程执行完毕")


# ================= 后台资源监控线程 =================
def monitor_resources():
    """每 2 秒采集系统资源"""
    while not shutdown_event.is_set():
        try:
            state.cpu = psutil.cpu_percent(interval=1)
            state.memory = psutil.virtual_memory().percent
            state.signal = max(70, min(100, 95 + int((np.random.random() - 0.5) * 10)))
        except Exception as e:
            logger.error(f"资源监控异常: {e}")
        time.sleep(2)


# ================= API 路由 =================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/video_feed')
def video_feed():
    """ 支持暂停的视频流端点"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/system/status')
@require_auth
def get_system_status():
    """ 系统状态查询 (CPU/内存/FPS/信号)"""
    return jsonify({
        "success": True,
        "cpu": round(state.cpu, 1),
        "memory": round(state.memory, 1),
        "fps": round(state.fps, 1),  #  视频流 FPS
        "signal": state.signal,
        "recognizing": state.is_recognizing,
        "sorting": state.is_sorting,
        "stream_paused": state.is_stream_paused
    })


@app.route('/api/recognition/start', methods=['POST'])
@require_auth
def start_recognition():
    """ 启动识别模块"""
    data = request.get_json(silent=True) or {}
    with state.lock:
        if state.is_recognizing:
            return jsonify({"success": False, "message": "识别已在运行"})
        state.is_recognizing = True
        if not state.start_time:
            state.start_time = time.time()
    
    if not detector_mgr.is_ready():
        detector_mgr.start()
    
    logger.info(f"识别模块已启动 | 参数: {data}")
    return jsonify({"success": True, "message": "识别模块已启动"})


@app.route('/api/recognition/stop', methods=['POST'])
@require_auth
def stop_recognition():
    """ 停止识别模块"""
    with state.lock:
        state.is_recognizing = False
    logger.info("🛑 识别模块已停止")
    return jsonify({"success": True, "message": "识别模块已停止"})


@app.route('/api/emergency/stop', methods=['POST'])
@require_auth
def emergency_stop():
    """ 急停功能：关闭扭矩锁"""
    global robot_arm
    if robot_arm and robot_arm.is_connected:
        success = robot_arm.torque_control(0)  # 关闭扭矩锁
        if success:
            # 同时停止所有操作
            with state.lock:
                state.is_recognizing = False
                state.is_sorting = False
            logger.info("🚨 急停激活：已关闭扭矩锁并停止所有操作")
            return jsonify({"success": True, "message": "急停成功，已关闭扭矩锁"})
        else:
            logger.error("❌ 急停失败：无法关闭扭矩锁")
            return jsonify({"success": False, "message": "急停失败，无法关闭扭矩锁"})
    else:
        logger.warning("⚠️ 急停请求：机械臂未连接")
        return jsonify({"success": False, "message": "机械臂未连接"})


@app.route('/api/sorting/start', methods=['POST'])
@require_auth
def start_sorting():
    """ 启动分拣执行"""
    data = request.get_json(silent=True) or {}
    with state.lock:
        if state.is_sorting:
            return jsonify({"success": False, "message": "分拣已在运行"})
        state.is_sorting = True
        if not state.start_time:  # 避免重复重置开始时间
            state.start_time = time.time()
    
    if not detector_mgr.is_ready():
        detector_mgr.start()
    
    # 🚀 启动分拣执行线程（守护线程，主程序退出时自动清理）
    sorting_proc = threading.Thread(
        target=sorting_thread, 
        daemon=True, 
        name="SortingThread"
    )
    sorting_proc.start()
    
    logger.info(f"⚙️ 分拣执行已启动 | 参数: {data}")
    return jsonify({"success": True, "message": "分拣执行已启动", "thread": "sorting_thread"})

@app.route('/api/sorting/stop', methods=['POST'])
@require_auth
def stop_sorting():
    """停止分拣执行"""
    with state.lock:
        if not state.is_sorting:
            return jsonify({"success": False, "message": "分拣未在运行"})
        state.is_sorting = False  # 线程会检测此标志退出
    
    logger.info("[INFO] 分拣执行已停止，等待线程退出...")
    # 如需等待线程完全退出，可在此添加 threading.enumerate() 检查
    return jsonify({"success": True, "message": "分拣执行已停止"})


@app.route('/api/stream/pause', methods=['POST'])
@require_auth
def pause_stream():
    """真正暂停视频流"""
    with state.lock:
        state.is_stream_paused = True
    logger.info("⏸ 视频流已暂停")
    return jsonify({"success": True, "message": "视频流已暂停"})


@app.route('/api/stream/resume', methods=['POST'])
@require_auth
def resume_stream():
    """ 恢复视频流"""
    with state.lock:
        state.is_stream_paused = False
    logger.info("▶ 视频流已恢复")
    return jsonify({"success": True, "message": "视频流已恢复"})


@app.route('/api/detection/results')
@require_auth
def get_detection_results():
    """获取检测统计结果 (前端轮询更新)"""
    with state.lock:
        total = state.total_processed or 1
        rates = {k: round((v / total) * 100, 1) for k, v in state.counts.items()}
        
        conf_total = state.conf_stats["total"] or 1
        conf_rates = {
            "high": round((state.conf_stats["high"] / conf_total) * 100, 1),
            "medium": round((state.conf_stats["medium"] / conf_total) * 100, 1),
            "low": round((state.conf_stats["low"] / conf_total) * 100, 1),
            "avg": round(sum(d["confidence"] for d in state.recent_detections) / len(state.recent_detections), 3) 
                     if state.recent_detections else 0.0
        }
        
        recent = []
        for item in list(state.recent_detections)[:10]:
            recent.append({
                "class": item["class"],
                "confidence": item["confidence"],
                "time": time.strftime("%H:%M:%S", time.localtime(item["timestamp"])),
                "conf_level": "high" if item["confidence"] >= 0.9 else "medium" if item["confidence"] >= 0.7 else "low"
            })
        
        return jsonify({
            "success": True,
            "counts": state.counts.copy(),
            "rates": rates,
            "total": state.total_processed,
            "error_count": state.error_count,
            "confidence": conf_rates,
            "recent": recent,
            "running_time": _format_duration(state.start_time) if state.start_time else "00:00"
        })


@app.route('/api/detection/reset', methods=['POST'])
@require_auth
def reset_detection():
    """重置所有计数"""
    with state.lock:
        for key in state.counts:
            state.counts[key] = 0
        state.total_processed = 0
        state.error_count = 0
        state.conf_stats = {"high": 0, "medium": 0, "low": 0, "total": 0}
        state.recent_detections.clear()
    logger.info("🔄 检测统计已重置")
    return jsonify({"success": True, "message": "计数已重置"})


def _format_duration(start_time):
    """格式化运行时长"""
    if not start_time:
        return "00:00"
    seconds = int(time.time() - start_time)
    mins, secs = divmod(seconds, 60)
    return f"{mins:02d}:{secs:02d}"


# ================= 启动入口 =================
if __name__ == '__main__':
    try:
        # 1. 启动资源监控线程
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True, name="ResourceMonitor")
        monitor_thread.start()
        
        # 2. 预启动 YOLO 检测器
        logger.info("🔧 预启动 YOLODetector...")
        detector_mgr.start()
        
        # 3. 初始化机械臂控制器
        try:
            logger.info("🤖 初始化机械臂控制器...")
            robot_arm = RobotArmController(port="/dev/ttyUSB0")  # 根据实际端口调整
            robot_arm.connect()
            if robot_arm.is_connected:
                logger.info("✅ 机械臂控制器已连接")
            else:
                logger.warning("⚠️ 机械臂控制器连接失败，将使用模拟模式")
        except Exception as e:
            logger.error(f"❌ 机械臂控制器初始化失败: {e}")
            robot_arm = None
        
        # 4. 启动 Flask
        logger.info("🌶️ 辣椒分拣统计系统 后端已启动")
        logger.info(f"📡 服务地址: http://127.0.0.1:{PORT}")
        logger.info(f"🔐 鉴权密钥: {AUTH_KEY}")
        logger.info(f"🧠 模型路径: {MODEL_PATH}")
        logger.info("💡 按 Ctrl+C 停止服务")
        
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("收到键盘中断，正在退出...")
    except Exception as e:
        logger.error(f"程序异常退出: {e}")
    finally:
        # 确保所有资源被清理
        logger.info("正在清理资源...")
        shutdown_event.set()
        detector_mgr.stop()
        
        # 断开机械臂
        if robot_arm:
            robot_arm.disconnect()
        
        # 等待线程退出
        if 'monitor_thread' in locals() and monitor_thread.is_alive():
            monitor_thread.join(timeout=2.0)
        
        logger.info("程序已安全退出")
        
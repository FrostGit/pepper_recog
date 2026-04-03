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

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['JSON_AS_ASCII'] = False

# 全局退出标志
shutdown_event = threading.Event()

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
    """处理检测结果： 仅绘制彩色框 + 更新统计 (无中文)"""
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
        
        # 绘制检测框 (仅彩色框，无中文标签)
        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        color = PEPPER_COLORS.get(front_class, (255, 255, 255))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 不绘制任何中文文本，彻底解决乱码问题
        
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
    """ 支持真正暂停的视频流端点"""
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


@app.route('/api/sorting/start', methods=['POST'])
@require_auth
def start_sorting():
    """ 启动分拣执行"""
    data = request.get_json(silent=True) or {}
    with state.lock:
        if state.is_sorting:
            return jsonify({"success": False, "message": "分拣已在运行"})
        state.is_sorting = True
        state.start_time = time.time()
    
    if not detector_mgr.is_ready():
        detector_mgr.start()
    
    logger.info(f" 分拣执行已启动 | 参数: {data}")
    return jsonify({"success": True, "message": "分拣执行已启动"})


@app.route('/api/sorting/stop', methods=['POST'])
@require_auth
def stop_sorting():
    """停止分拣执行"""
    with state.lock:
        state.is_sorting = False
    logger.info("[INFO] 分拣执行已停止")
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
        
        # 3. 启动 Flask
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
        
        # 等待线程退出
        if 'monitor_thread' in locals() and monitor_thread.is_alive():
            monitor_thread.join(timeout=2.0)
        
        logger.info("程序已安全退出")
        
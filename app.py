#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌶️ 辣椒分拣统计系统 - Flask 后端
集成 YOLODetector 实现实时视频流推理 + 分类计数 + API 服务
"""

import os
import time
import threading
import logging
import queue
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
MODEL_PATH = "/home/jetson/workspace/yolo26/YOLO26_41/weights/best.pt"  # 您的模型路径
ROI_ENABLED = True
ROI_COORDS = (400, 150, 900, 600)  # (x1, y1, x2, y2) 感兴趣区域
HOST = '0.0.0.0'
PORT = 5000

# 辣椒类别映射：YOLO 模型输出类别名 → 前端类别 key
# 请根据您的模型实际类别名称修改此映射
CLASS_MAPPING = {
    # 小米辣系列
    "millet_dried_red": "millet_dr",    # 干的红色小米辣
    "millet_fresh_red": "millet_fr",    # 湿的红色小米辣
    "millet_fresh_green": "millet_fg",  # 湿的青色小米辣
    # 灯笼椒
    "lantern_round": "lantern",          # 圆形灯笼椒
    # 遵义辣椒
    "zunyi_large": "zunyi_l",            # 大的遵义辣椒
    "zunyi_small": "zunyi_s",            # 小的遵义辣椒
}

# 各类辣椒显示颜色 (BGR 格式，用于 OpenCV 绘制)
PEPPER_COLORS = {
    "millet_dr": (30, 30, 220),   # 干红 - 深红
    "millet_fr": (50, 80, 240),   # 鲜红 - 亮红
    "millet_fg": (80, 200, 80),   # 鲜青 - 绿色
    "lantern":   (50, 200, 220),  # 灯笼 - 青色
    "zunyi_l":   (180, 80, 220),  # 大遵义 - 紫色
    "zunyi_s":   (220, 150, 240), # 小遵义 - 浅紫
}

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['JSON_AS_ASCII'] = False

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ================= 状态管理 (线程安全) =================
class DetectionState:
    """检测状态与统计数据（线程安全）"""
    def __init__(self):
        self.lock = threading.Lock()
        self.is_recognizing = False
        self.is_sorting = False
        self.start_time = None
        
        # 系统资源
        self.cpu = 0.0
        self.memory = 0.0
        self.fps = 0.0
        self.signal = 95
        
        # 计数统计
        self.counts = {key: 0 for key in CLASS_MAPPING.values()}
        self.total_processed = 0
        self.error_count = 0
        
        # 置信度统计
        self.conf_stats = {"high": 0, "medium": 0, "low": 0, "total": 0}
        
        # 最近识别记录 (最多保留 20 条)
        self.recent_detections = deque(maxlen=20)
        
        # 视频流 FPS 计算
        self._frame_count = 0
        self._fps_last_time = time.time()
        
        # 检测结果队列 (用于视频流线程与 API 线程通信)
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
            return jsonify({"success": False, "error": "auth_failed", "message": "鉴权失败，密钥错误"}), 403
            
        return f(*args, **kwargs)
    return decorated


# ================= YOLO 检测器管理 =================
class DetectorManager:
    """YOLODetector 单例管理"""
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
        self._result_queue = queue.Queue(maxsize=5)
        
    def start(self):
        """启动检测器后台线程"""
        if self._running:
            logger.warning("⚠️ 检测器已在运行")
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
            
            # 启动结果消费线程
            self._thread = threading.Thread(target=self._consume_results, daemon=True, name="YOLOConsumer")
            self._thread.start()
            
            logger.info("✅ YOLODetector 已启动")
            
        except Exception as e:
            logger.error(f"❌ 启动 YOLODetector 失败: {e}")
            self._running = False
            
    def stop(self):
        """停止检测器"""
        if not self._running:
            return
        self._running = False
        if self.detector:
            self.detector.stop()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("🛑 YOLODetector 已停止")
        
    def _consume_results(self):
        """后台线程：持续从 detector 获取结果并放入共享队列"""
        while self._running:
            try:
                # 非阻塞获取结果，超时 0.5 秒
                data = self.detector.get_result(timeout=0.5)
                if data is None:
                    continue
                    
                frame, results, fps = data
                
                # 放入共享队列（如果队列满则丢弃最旧）
                if state.result_queue.full():
                    try:
                        state.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                state.result_queue.put_nowait({
                    "frame": frame,
                    "results": results,
                    "fps": fps
                })
                
            except Exception as e:
                logger.error(f"检测结果消费异常: {e}")
                time.sleep(0.1)
                
    def get_latest_result(self, timeout=0.1):
        """获取最新检测结果（供视频流使用）"""
        try:
            return state.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def is_ready(self):
        """检测器是否就绪"""
        return self._running and self.detector is not None


# 全局检测器实例
detector_mgr = DetectorManager()


# ================= 视频流生成器 (集成 YOLO) =================
def generate_frames():
    """MJPEG 视频流生成器 + YOLO 实时推理 + 绘制检测框"""
    
    # 如果检测器未启动，先尝试启动
    if not detector_mgr.is_ready():
        logger.warning("⚠️ YOLODetector 未就绪，尝试启动...")
        detector_mgr.start()
    
    # 降级方案：如果检测器启动失败，使用模拟画面
    if not detector_mgr.is_ready():
        logger.warning("⚠️ YOLODetector 启动失败，启用模拟画面模式")
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "YOLODetector Unavailable", (80, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, "Running in Simulation Mode", (60, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            _add_overlay_info(frame)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)
        return

    # 正常模式：从检测器获取推理结果
    while True:
        result = detector_mgr.get_latest_result(timeout=0.1)
        
        if result is None:
            # 无新结果时，生成一个空白帧保持连接
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for detection...", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        else:
            frame, results, fps = result["frame"], result["results"], result["fps"]
            
            # 更新视频流 FPS
            state._frame_count += 1
            now = time.time()
            if now - state._fps_last_time >= 1.0:
                state.fps = state._frame_count / (now - state._fps_last_time)
                state._frame_count = 0
                state._fps_last_time = now
            
            # 仅在识别/分拣模式下绘制检测结果并更新统计
            if state.is_recognizing or state.is_sorting:
                _process_detections(frame, results)
        
        # 添加系统信息叠加层
        _add_overlay_info(frame)
        
        # 编码并推流
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def _process_detections(frame, results):
    """处理检测结果：绘制检测框 + 更新统计"""
    if not results:
        return
        
    for det in results:
        class_name = det.get("class_name", "")
        confidence = det.get("confidence", 0.0)
        bbox = det.get("bbox", {})  # {"x1", "y1", "x2", "y2"}
        center = det.get("center", (0, 0))
        
        # 映射到前端类别
        front_class = CLASS_MAPPING.get(class_name)
        if not front_class:
            # 未知类别，跳过或记录
            continue
        
        # 1. 绘制检测框
        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        color = PEPPER_COLORS.get(front_class, (255, 255, 255))  # BGR
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{front_class} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制中心点
        cv2.circle(frame, tuple(map(int, center)), 4, color, -1)
        
        # 2. 更新统计（线程安全）
        with state.lock:
            state.counts[front_class] += 1
            state.total_processed += 1
            
            # 置信度统计
            if confidence >= 0.9:
                state.conf_stats["high"] += 1
            elif confidence >= 0.7:
                state.conf_stats["medium"] += 1
            else:
                state.conf_stats["low"] += 1
            state.conf_stats["total"] += 1
            
            # 添加到最近记录
            state.recent_detections.appendleft({
                "class": front_class,
                "name": _get_chinese_name(front_class),
                "confidence": round(confidence, 3),
                "timestamp": time.time(),
                "center": center
            })


def _add_overlay_info(frame):
    """在视频帧上叠加系统信息"""
    h, w = frame.shape[:2]
    
    # 状态指示
    status_text = "🟢 识别中" if state.is_recognizing else "🟡 待机"
    cv2.putText(frame, status_text, (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # FPS 显示
    cv2.putText(frame, f"FPS: {state.fps:.1f}", (w - 100, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ROI 区域指示（如果启用）
    if ROI_ENABLED and ROI_COORDS:
        x1, y1, x2, y2 = ROI_COORDS
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(frame, "ROI", (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def _get_chinese_name(front_class):
    """前端类别 key → 中文显示名"""
    names = {
        "millet_dr": "干红小米辣",
        "millet_fr": "鲜红小米辣", 
        "millet_fg": "鲜青小米辣",
        "lantern": "圆形灯笼椒",
        "zunyi_l": "大号遵义椒",
        "zunyi_s": "小号遵义椒"
    }
    return names.get(front_class, front_class)


# ================= 后台资源监控线程 =================
def monitor_resources():
    """每 2 秒采集一次系统资源"""
    while True:
        try:
            state.cpu = psutil.cpu_percent(interval=1)
            state.memory = psutil.virtual_memory().percent
            # 模拟信号强度波动 (70~100)
            state.signal = max(70, min(100, 95 + int((np.random.random() - 0.5) * 10)))
        except Exception as e:
            logger.error(f"资源监控异常: {e}")
        time.sleep(2)


# ================= API 路由 =================
@app.route('/')
def index():
    """返回前端页面（从 static 目录）"""
    return send_from_directory('static', 'index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG 视频流端点"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/system/status')
@require_auth
def get_system_status():
    """系统状态查询"""
    return jsonify({
        "success": True,
        "cpu": round(state.cpu, 1),
        "memory": round(state.memory, 1),
        "fps": round(state.fps, 1),
        "signal": state.signal,
        "recognizing": state.is_recognizing,
        "sorting": state.is_sorting
    })


@app.route('/api/recognition/start', methods=['POST'])
@require_auth
def start_recognition():
    """启动识别模块"""
    data = request.get_json(silent=True) or {}
    
    with state.lock:
        if state.is_recognizing:
            return jsonify({"success": False, "message": "识别已在运行"})
        
        state.is_recognizing = True
        if not state.start_time:
            state.start_time = time.time()
    
    # 确保检测器已启动
    if not detector_mgr.is_ready():
        detector_mgr.start()
    
    logger.info(f"✅ 识别模块已启动 | 参数: {data}")
    return jsonify({"success": True, "message": "识别模块已启动"})


@app.route('/api/recognition/stop', methods=['POST'])
@require_auth
def stop_recognition():
    """停止识别模块"""
    with state.lock:
        state.is_recognizing = False
    
    logger.info("🛑 识别模块已停止")
    return jsonify({"success": True, "message": "识别模块已停止"})


@app.route('/api/sorting/start', methods=['POST'])
@require_auth
def start_sorting():
    """启动分拣执行"""
    data = request.get_json(silent=True) or {}
    
    with state.lock:
        if state.is_sorting:
            return jsonify({"success": False, "message": "分拣已在运行"})
        
        state.is_sorting = True
        state.start_time = time.time()
    
    # 确保检测器已启动
    if not detector_mgr.is_ready():
        detector_mgr.start()
    
    logger.info(f"✅ 分拣执行已启动 | 参数: {data}")
    return jsonify({"success": True, "message": "分拣执行已启动"})


@app.route('/api/sorting/stop', methods=['POST'])
@require_auth
def stop_sorting():
    """停止分拣执行"""
    with state.lock:
        state.is_sorting = False
    
    logger.info("🛑 分拣执行已停止")
    return jsonify({"success": True, "message": "分拣执行已停止"})


@app.route('/api/detection/results')
@require_auth
def get_detection_results():
    """获取检测统计结果"""
    with state.lock:
        # 计算占比
        total = state.total_processed or 1
        rates = {k: round((v / total) * 100, 1) for k, v in state.counts.items()}
        
        # 置信度统计
        conf_total = state.conf_stats["total"] or 1
        conf_rates = {
            "high": round((state.conf_stats["high"] / conf_total) * 100, 1),
            "medium": round((state.conf_stats["medium"] / conf_total) * 100, 1),
            "low": round((state.conf_stats["low"] / conf_total) * 100, 1),
            "avg": round(sum(d["confidence"] for d in state.recent_detections) / len(state.recent_detections), 3) 
                     if state.recent_detections else 0.0
        }
        
        # 格式化最近记录
        recent = []
        for item in list(state.recent_detections)[:10]:
            recent.append({
                "class": item["class"],
                "name": item["name"],
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
    # 1. 启动资源监控线程
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True, name="ResourceMonitor")
    monitor_thread.start()
    
    # 2. 预启动 YOLO 检测器（可选：改为按需启动）
    logger.info("🔧 预启动 YOLODetector...")
    detector_mgr.start()
    
    # 3. 启动 Flask
    logger.info("🌶️ 辣椒分拣统计系统 后端已启动")
    logger.info(f"📡 服务地址: http://127.0.0.1:{PORT}")
    logger.info(f"📁 前端路径: {os.path.abspath('static/index.html')}")
    logger.info(f"🔐 鉴权密钥: {AUTH_KEY}")
    logger.info(f"🧠 模型路径: {MODEL_PATH}")
    logger.info("💡 按 Ctrl+C 停止服务")
    
    # threaded=True 允许并发处理视频流与 API 请求
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
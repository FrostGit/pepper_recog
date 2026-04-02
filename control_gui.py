#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoArm-M3 机械臂 GUI 调试程序
功能：三维坐标控制 + 夹爪调节 + 实时状态监控 + 阻塞/非阻塞模式

Author: 霜叶
Date: 2026
"""

import sys
import json
import time
import serial
import serial.tools.list_ports
# ✅ 修复：使用 PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QSlider,
    QGroupBox, QFormLayout, QTextEdit, QTabWidget, QDoubleSpinBox,
    QSpinBox, QMessageBox, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QTextCursor, QFont
from roarm_m3 import RoArmM3S, ArmState


# ============== 状态更新线程 ==============

class StateMonitorThread(QThread):
    state_updated = pyqtSignal(object)  # 建议传递对象或字典，避免依赖特定类
    
    def __init__(self, arm: RoArmM3S, interval_ms: int = 50):
        super().__init__()
        self.arm = arm
        self.interval = interval_ms / 1000.0
        self.running = True
        self.paused = False  # ✅ 新增：暂停标志
        
    def run(self):
        while self.running:
            if not self.paused and self.arm and self.arm.ser and self.arm.ser.is_open:
                try:
                    state = self.arm.current_state
                    if state:
                        self.state_updated.emit(state)
                except:
                    pass
            self.msleep(int(self.interval * 1000))
    
    def pause(self):
        self.paused = True
        
    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        self.wait()


# ============== 主窗口 ==============

class RoArmGUI(QMainWindow):
    """机械臂调试主界面"""
    
    # ✅ 修复：添加关节属性映射字典（ArmState 使用完整英文名）
    JOINT_ATTR_MAP = {
        'b': 'base',
        's': 'shoulder', 
        'e': 'elbow',
        't': 'wrist',
        'r': 'roll',
        'g': 'gripper'
    }
    
    TORQUE_ATTR_MAP = {
        'tB': 'torque_base',
        'tS': 'torque_shoulder',
        'tE': 'torque_elbow',
        'tT': 'torque_wrist',
        'tR': 'torque_roll'
    }
    
    def __init__(self):
        super().__init__()
        self.arm: RoArmM3S = None
        self.monitor_thread: StateMonitorThread = None
        self._latest_state = None
        self.init_ui()
        self.refresh_ports()
        
    def init_ui(self):
        """初始化界面布局"""
        self.setWindowTitle("🤖 RoArm-M3 调试控制台")
        self.setMinimumSize(900, 700)
        
        # 主布局
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # 左侧：控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
        
        # 右侧：状态显示
        status_panel = self.create_status_panel()
        main_layout.addWidget(status_panel, stretch=1)
        
        # 底部：日志和状态栏
        bottom_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        bottom_layout.addWidget(QLabel("📋 运行日志:"))
        bottom_layout.addWidget(self.log_text)
        main_layout.addLayout(bottom_layout, stretch=1)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("⭕ 未连接")
        self.status_bar.addWidget(self.status_label)
        
        # 定时器：更新界面状态
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_from_state)
        self.ui_timer.start(100)  # 10Hz UI 刷新
        
    def create_control_panel(self) -> QWidget:
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 1. 串口连接
        conn_group = QGroupBox("🔌 串口连接")
        conn_layout = QFormLayout()
        
        self.port_combo = QComboBox()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_combo.setCurrentText("115200")
        
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.setFixedWidth(60)
        refresh_btn.clicked.connect(self.refresh_ports)
        
        port_row = QHBoxLayout()
        port_row.addWidget(self.port_combo)
        port_row.addWidget(refresh_btn)
        
        conn_layout.addRow("端口:", port_row)
        conn_layout.addRow("波特率:", self.baud_combo)
        
        self.connect_btn = QPushButton("🔗 连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.connect_btn.setStyleSheet("background: #4CAF50; color: white; font-weight: bold;")
        conn_layout.addRow("", self.connect_btn)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        # 2. 运动参数
        param_group = QGroupBox("⚙️ 运动参数")
        param_layout = QFormLayout()
        
        self.blocking_check = QCheckBox("🎯 阻塞模式 (曲线插值)")
        self.blocking_check.setChecked(True)
        self.blocking_check.setToolTip("勾选=等待到达目标; 取消=立即返回适合连续轨迹")
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(25)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(f"{v/100:.2f}"))
        self.speed_label = QLabel("0.25")
        
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)
        
        param_layout.addRow(self.blocking_check, QWidget())
        param_layout.addRow("速度系数:", speed_row)
        param_layout.addRow("", QLabel("范围：0.01 ~ 1.0"))
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 3. 末端位置控制
        pos_group = QGroupBox("📍 末端位置控制 (mm)")
        pos_layout = QFormLayout()
        
        self.x_spin = QDoubleSpinBox()
        self.y_spin = QDoubleSpinBox()
        self.z_spin = QDoubleSpinBox()
        self.pitch_spin = QDoubleSpinBox()
        
        for spin in [self.x_spin, self.y_spin, self.z_spin]:
            spin.setRange(-300, 300)
            spin.setDecimals(1)
            spin.setSingleStep(5)
            spin.setValue(0)
        self.x_spin.setValue(200)  # 默认前伸
        
        self.pitch_spin.setRange(-1.57, 1.57)
        self.pitch_spin.setDecimals(3)
        self.pitch_spin.setSingleStep(0.1)
        self.pitch_spin.setValue(0)
        self.pitch_spin.setSuffix(" rad")
        
        pos_layout.addRow("X:", self.x_spin)
        pos_layout.addRow("Y:", self.y_spin)
        pos_layout.addRow("Z:", self.z_spin)
        pos_layout.addRow("Pitch:", self.pitch_spin)
        
        self.move_btn = QPushButton("🚀 移动到目标")
        self.move_btn.clicked.connect(self.execute_move)
        self.move_btn.setStyleSheet("background: #2196F3; color: white; font-weight: bold; padding: 8px;")
        pos_layout.addRow("", self.move_btn)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # 4. 夹爪控制
        grip_group = QGroupBox("🤚 夹爪控制")
        grip_layout = QVBoxLayout()
        
        self.grip_slider = QSlider(Qt.Horizontal)
        self.grip_slider.setRange(45, 180)
        self.grip_slider.setValue(180)
        self.grip_slider.setTickPosition(QSlider.TicksBelow)
        self.grip_slider.setTickInterval(45)
        self.grip_slider.valueChanged.connect(self.update_grip_display)
        
        self.grip_value = QLabel("180° (闭合)")
        self.grip_value.setAlignment(Qt.AlignCenter)
        
        grip_row = QHBoxLayout()
        grip_row.addWidget(QLabel("45°"))
        grip_row.addWidget(self.grip_slider)
        grip_row.addWidget(QLabel("180°"))
        
        grip_layout.addLayout(grip_row)
        grip_layout.addWidget(self.grip_value)
        
        self.grip_btn = QPushButton("✋ 执行夹爪")
        self.grip_btn.clicked.connect(self.execute_gripper)
        grip_layout.addWidget(self.grip_btn)
        
        # 预设按钮
        preset_row = QHBoxLayout()
        open_btn = QPushButton("🔓 张开")
        close_btn = QPushButton("🔒 闭合")
        open_btn.clicked.connect(lambda: self.set_gripper(60))
        close_btn.clicked.connect(lambda: self.set_gripper(180))
        preset_row.addWidget(open_btn)
        preset_row.addWidget(close_btn)
        grip_layout.addLayout(preset_row)
        
        grip_group.setLayout(grip_layout)
        layout.addWidget(grip_group)
        
        # 5. 快捷操作
        quick_group = QGroupBox("⚡ 快捷操作")
        quick_layout = QHBoxLayout()
        
        init_btn = QPushButton("🏠 复位")
        stop_btn = QPushButton("🛑 急停")
        feedback_btn = QPushButton("📡 请求反馈")
        
        init_btn.clicked.connect(lambda: self.arm.move_to_init() if self.arm else None)
        stop_btn.clicked.connect(self.emergency_stop)
        feedback_btn.clicked.connect(self.request_feedback)
        
        quick_layout.addWidget(init_btn)
        quick_layout.addWidget(stop_btn)
        quick_layout.addWidget(feedback_btn)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        layout.addStretch()
        return panel
    
    def create_status_panel(self) -> QWidget:
        """创建右侧状态显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 选项卡
        tabs = QTabWidget()
        
        # Tab 1: 末端位姿
        pose_tab = QWidget()
        pose_layout = QFormLayout()
        
        self.cur_x = QLabel("--")
        self.cur_y = QLabel("--")
        self.cur_z = QLabel("--")
        self.cur_pitch = QLabel("--")
        
        for lbl in [self.cur_x, self.cur_y, self.cur_z]:
            lbl.setStyleSheet("font-weight: bold; color: #1976D2;")
        
        pose_layout.addRow("X (mm):", self.cur_x)
        pose_layout.addRow("Y (mm):", self.cur_y)
        pose_layout.addRow("Z (mm):", self.cur_z)
        pose_layout.addRow("Pitch (rad):", self.cur_pitch)
        pose_tab.setLayout(pose_layout)
        tabs.addTab(pose_tab, "📍 末端位姿")
        
        # Tab 2: 关节角度
        joint_tab = QWidget()
        joint_layout = QFormLayout()
        
        self.joint_labels = {}
        # ✅ 修复：存储完整属性名 (base, shoulder 等)
        joint_names = [
            ("Base", "b", "base"), 
            ("Shoulder", "s", "shoulder"), 
            ("Elbow", "e", "elbow"), 
            ("Wrist", "t", "wrist"), 
            ("Roll", "r", "roll"), 
            ("Gripper", "g", "gripper")
        ]
        
        for name, key, attr in joint_names:
            deg_lbl = QLabel("--°")
            rad_lbl = QLabel("-- rad")
            deg_lbl.setStyleSheet("color: #388E3C;")
            self.joint_labels[key] = (deg_lbl, rad_lbl, attr)  # 存储属性名
            joint_layout.addRow(f"{name}:", deg_lbl)
        
        joint_tab.setLayout(joint_layout)
        tabs.addTab(joint_tab, "🦴 关节角度")
        
        # Tab 3: 负载/扭矩
        torque_tab = QWidget()
        torque_layout = QFormLayout()
        
        self.torque_labels = {}
        # ✅ 修复：存储完整属性名 (torque_base, torque_shoulder 等)
        torque_names = [
            ("Base", "tB", "torque_base"),
            ("Shoulder", "tS", "torque_shoulder"),
            ("Elbow", "tE", "torque_elbow"),
            ("Wrist", "tT", "torque_wrist"),
            ("Roll", "tR", "torque_roll")
        ]
        
        for name, key, attr in torque_names:
            lbl = QLabel("0")
            lbl.setStyleSheet("color: #F57C00;")
            self.torque_labels[key] = (lbl, attr)  # 存储属性名
            torque_layout.addRow(f"{name}:", lbl)
        
        torque_tab.setLayout(torque_layout)
        tabs.addTab(torque_tab, "⚡ 关节负载")
        
        # Tab 4: 原始数据
        raw_tab = QTextEdit()
        raw_tab.setReadOnly(True)
        raw_tab.setFont(QFont("Consolas", 8))
        self.raw_text = raw_tab
        tabs.addTab(raw_tab, "📄 原始 JSON")
        
        layout.addWidget(tabs)
        
        # 状态摘要
        summary = QGroupBox("📊 状态摘要")
        summary_layout = QHBoxLayout()
        
        self.grip_status = QLabel("🤚 夹爪：--")
        self.voltage_label = QLabel("🔋 电压：-- V")
        self.update_time = QLabel("🕐 更新：--")
        
        summary_layout.addWidget(self.grip_status)
        summary_layout.addWidget(self.voltage_label)
        summary_layout.addWidget(self.update_time)
        summary_layout.addStretch()
        summary.setLayout(summary_layout)
        layout.addWidget(summary)
        
        return panel
    
    # ============== 功能方法 ==============
    
    def refresh_ports(self):
        """刷新可用串口列表"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = f"{p.device}: {p.description}" if p.description else p.device
            self.port_combo.addItem(desc, userData=p.device)
        if ports:
            self.log(f"🔍 找到 {len(ports)} 个串口")
        else:
            self.log("⚠️ 未检测到串口，请检查连接")
    
    def toggle_connection(self):
        """连接/断开串口"""
        if self.arm and self.arm.ser and self.arm.ser.is_open:
            self.cleanup_connection()
            self.status_label.setText("⭕ 未连接")
            self.status_label.setStyleSheet("color: gray;")
            self.connect_btn.setText("🔗 连接")
            self.connect_btn.setStyleSheet("background: #4CAF50; color: white;")
            self.set_controls_enabled(False)
        else:
            port = self.port_combo.currentData()
            if not port:
                QMessageBox.warning(self, "提示", "请先选择串口端口")
                return
            
            try:
                self.arm = RoArmM3S(port, baudrate=int(self.baud_combo.currentText()))
                if self.arm.connect():
                    self.status_label.setText("🟢 已连接")
                    self.status_label.setStyleSheet("color: green; font-weight: bold;")
                    self.connect_btn.setText("🔌 断开")
                    self.connect_btn.setStyleSheet("background: #f44336; color: white;")
                    self.set_controls_enabled(True)
                    
                    self.monitor_thread = StateMonitorThread(self.arm)
                    self.monitor_thread.state_updated.connect(self.on_state_received)
                    self.monitor_thread.start()
                    
                    self.log(f"✅ 连接到 {port}")
                    # QTimer.singleShot(500, self.request_feedback)
                else:
                    raise Exception("连接失败")
            except Exception as e:
                self.log(f"❌ 连接错误：{e}")
                QMessageBox.critical(self, "连接失败", str(e))
                self.cleanup_connection()
    
    def cleanup_connection(self):
        """清理连接资源"""
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread = None
        if self.arm:
            self.arm.disconnect()
            self.arm = None
    
    def set_controls_enabled(self, enabled: bool):
        """批量启用/禁用控制控件"""
        for ctrl in [self.x_spin, self.y_spin, self.z_spin, self.pitch_spin,
                    self.grip_slider, self.move_btn, self.grip_btn,
                    self.blocking_check, self.speed_slider]:
            ctrl.setEnabled(enabled)
    
    def update_grip_display(self, value: int):
        """更新夹爪滑块显示"""
        status = "闭合" if value >= 150 else "半开" if value >= 100 else "张开"
        self.grip_value.setText(f"{value}° ({status})")
    
    def set_gripper(self, angle: int):
        """设置夹爪滑块值"""
        self.grip_slider.setValue(angle)
        self.update_grip_display(angle)
    
    def execute_move(self):
        """执行末端位置移动"""
        if not self.arm:
            return
        
        # ✅ 修复：暂停状态监控，防止串口冲突
        if self.monitor_thread:
            self.monitor_thread.pause()
        
        try:
            x = self.x_spin.value()
            y = self.y_spin.value()
            z = self.z_spin.value()
            pitch = self.pitch_spin.value()
            blocking = self.blocking_check.isChecked()
            spd = self.speed_slider.value() / 100.0
            
            # ⚠️ 警告：请确认 RoArmM3S 库的 move_to_xyz 参数名
            # 如果库期望角度，这里需要 pitch * 180 / pi
            # 如果库期望参数名是 rz 而不是 t，请修改
            kwargs = {'x': x, 'y': y, 'z': z, 'spd': spd}
            
            # 假设库支持 pitch 参数，否则可能需要计算逆解发送关节角度
            if hasattr(self.arm, 'move_to_xyz'):
                # 尝试调用，捕获参数错误
                try:
                    if blocking:
                        # 假设 t 是 pitch，如果报错请改为 rz 或 pitch
                        result = self.arm.move_to_xyz(x, y, z, pitch=pitch, spd=spd, blocking=True) 
                    else:
                        result = self.arm.move_to_xyz(x, y, z, pitch=pitch, spd=spd, blocking=False)
                except TypeError:
                    # 兼容旧版或不同参数名
                    result = self.arm.move_to_xyz(x, y, z, t=pitch, spd=spd, blocking=blocking)
                
                self.log(f"🎯 移动指令：XYZ=[{x},{y},{z}] P={pitch:.2f} → {'✓' if result else '✗'}")
            else:
                self.log("❌ 库不支持 move_to_xyz 方法")
                
        except Exception as e:
            self.log(f"❌ 移动错误：{e}")
        finally:
            # ✅ 修复：无论成功失败，恢复监控
            if self.monitor_thread:
                # 稍微延迟恢复，确保指令发送完毕
                QTimer.singleShot(200, self.monitor_thread.resume)
    
    def execute_gripper(self):
        """执行夹爪动作"""
        if not self.arm:
            return
        
        angle = self.grip_slider.value()
        try:
            result = self.arm.move_gripper_angle(angle)
            status = "闭合" if angle >= 150 else "张开"
            self.log(f"🤚 夹爪{status}({angle}°) → {'✓' if result else '✗'}")
        except Exception as e:
            self.log(f"❌ 夹爪错误：{e}")
    
    def emergency_stop(self):
        """急停：停止所有运动"""
        if self.arm:
            try:
                self.arm.stop_continuous_move()
                self.log("🛑 急停指令已发送")
            except:
                pass
    
    def request_feedback(self):
        """刷新状态显示（状态是持续发送的，无需主动请求）"""
        if self.arm and self._latest_state:
            self.log(f"📡 当前状态：XYZ=[{self._latest_state.x:.1f}, {self._latest_state.y:.1f}, {self._latest_state.z:.1f}]mm")
        else:
            self.log("📡 等待状态更新...")
    
    def on_state_received(self, state: ArmState):
        """接收并处理状态更新"""
        self._latest_state = state
        
        # ✅ 修复：直接使用传入的 state 对象记录日志，不再次读取串口
        # 假设 ArmState 可以被字典化，或者有 to_dict 方法
        # 如果 ArmState 是 dataclass，可以使用 asdict(state)
        try:
            from dataclasses import asdict
            raw_data = asdict(state) if hasattr(state, '__dataclass_fields__') else str(state)
            
            if not hasattr(self, '_last_raw_log') or time.time() - self._last_raw_log > 2:
                self._last_raw_log = time.time()
                self.raw_text.append(json.dumps(raw_data, ensure_ascii=False, indent=2))
                lines = self.raw_text.toPlainText().split('\n')
                if len(lines) > 200:
                    self.raw_text.setPlainText('\n'.join(lines[-200:]))
        except Exception:
            pass
    
    def update_ui_from_state(self):
        """定时器：从最新状态更新 UI 显示"""
        if not self._latest_state:
            return
        
        s = self._latest_state
        
        # 末端位姿
        self.cur_x.setText(f"{s.x:.1f}")
        self.cur_y.setText(f"{s.y:.1f}")
        self.cur_z.setText(f"{s.z:.1f}")
        self.cur_pitch.setText(f"{s.pitch:.3f}")
        
        # 关节角度
        for key, (deg_lbl, rad_lbl, attr_name) in self.joint_labels.items():
            try:
                angle_rad = getattr(s, attr_name)
                angle_deg = angle_rad * 180 / 3.1415926535
                deg_lbl.setText(f"{angle_deg:6.1f}°")
                rad_lbl.setText(f"{angle_rad:.3f}") # ✅ 修复：更新弧度标签
            except:
                pass
        
    
    def log(self, message: str):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.moveCursor(QTextCursor.End)
    
    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.ui_timer.stop()
        self.cleanup_connection()
        event.accept()


# ============== 程序入口 ==============

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    app.setStyleSheet("""
        QGroupBox { font-weight: bold; margin-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QSlider::handle:horizontal { background: #2196F3; border: 2px solid #1976D2; width: 18px; border-radius: 9px; }
        QPushButton:hover { opacity: 0.9; }
        QTextEdit { background: #1E1E1E; color: #D4D4D4; }
    """)
    
    window = RoArmGUI()
    window.show()
    
    # ✅ 修复：PyQt5 使用 exec_()
    sys.exit(app.exec_())
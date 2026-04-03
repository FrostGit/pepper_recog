#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoArm-M3-S 机械臂串口控制库 (增强版)
支持状态反馈实时解析与订阅

Author: 霜叶
Date: 2026
"""

import serial
import json
import time
import threading
import math
from typing import Optional, Dict, Callable, List
from enum import IntEnum
from dataclasses import dataclass, field


# ============== 常量定义 ==============

class Joint(IntEnum):
    """关节编号枚举"""
    BASE = 1
    SHOULDER = 2
    ELBOW = 3
    WRIST = 4
    ROLL = 5
    GRIPPER = 6


class Axis(IntEnum):
    """坐标轴枚举（逆运动学）"""
    X = 1
    Y = 2
    Z = 3
    PITCH = 4
    ROLL = 5
    GRIPPER = 6


class CommandCode(IntEnum):
    """JSON 指令类型码"""
    # 运动控制
    MOVE_INIT = 100
    SINGLE_JOINT_RAD = 101
    JOINTS_RAD = 102
    SINGLE_AXIS = 103
    XYZT_GOAL = 104          # 阻塞模式，带插值
    XYZT_DIRECT = 1041       # 非阻塞模式，直驱
    SERVO_FEEDBACK = 105     # 主动请求反馈
    EOAT_HAND_RAD = 106
    
    # 角度制控制
    SINGLE_JOINT_ANGLE = 121
    JOINTS_ANGLE = 122
    CONSTANT_CTRL = 123
    
    # 系统设置
    SET_JOINT_PID = 108
    RESET_PID = 109
    DYNAMIC_ADAPTATION = 112
    TORQUE_CTRL = 210
    
    # 反馈类型
    STATE_FEEDBACK = 1051    # 持续发送的状态反馈


# ============== 状态数据类 ==============

@dataclass
class ArmState:
    """机械臂实时状态数据类"""
    
    # 末端位姿
    x: float = 0.0          # X 坐标 (mm)
    y: float = 0.0          # Y 坐标 (mm)
    z: float = 0.0          # Z 坐标 (mm)
    pitch: float = 0.0      # Pitch 角度 (tit 字段，弧度)
    roll: float = 0.0       # Roll 角度 (r 字段，弧度)
    
    # 关节角度 (弧度)
    base: float = 0.0       # b
    shoulder: float = 0.0   # s
    elbow: float = 0.0      # e
    wrist: float = 0.0      # t (手腕关节 1)
    wrist2: float = 0.0     # r (手腕关节 2/ Roll)
    gripper: float = 0.0    # g
    
    # 关节负载 (单位：未知，参考值)
    torque_base: int = 0    # tB
    torque_shoulder: int = 0  # tS
    torque_elbow: int = 0   # tE
    torque_wrist: int = 0   # tT
    torque_wrist2: int = 0  # tR
    torque_gripper: int = 0 # tG
    
    # 扭矩开关状态 (0:关闭，1:开启)
    torque_switch: Dict[str, int] = field(default_factory=lambda: {
        'base': 0, 'shoulder': 0, 'elbow': 0,
        'wrist': 0, 'wrist2': 0, 'gripper': 0
    })
    
    # 系统信息
    voltage: Optional[float] = None  # 电压 (V)，v*0.01
    timestamp: float = field(default_factory=time.time)  # 最后更新时间
    
    @property
    def joint_angles_rad(self) -> Dict[str, float]:
        """获取所有关节角度 (弧度)"""
        return {
            'base': self.base,
            'shoulder': self.shoulder,
            'elbow': self.elbow,
            'wrist': self.wrist,
            'wrist2': self.wrist2,
            'gripper': self.gripper
        }
    
    @property
    def joint_angles_deg(self) -> Dict[str, float]:
        """获取所有关节角度 (角度制)"""
        return {k: round(v * 180 / math.pi, 2) for k, v in self.joint_angles_rad.items()}
    
    @property
    def end_effector_pose(self) -> Dict[str, float]:
        """获取末端位姿"""
        return {
            'position': (round(self.x, 2), round(self.y, 2), round(self.z, 2)),
            'pitch_rad': self.pitch,
            'pitch_deg': round(self.pitch * 180 / math.pi, 2),
            'roll_rad': self.roll,
            'roll_deg': round(self.roll * 180 / math.pi, 2)
        }
    
    @property
    def is_gripper_closed(self) -> bool:
        """判断夹爪是否闭合 (≥3.0 rad ≈ 172°)"""
        return self.gripper >= 3.0
    
    @property
    def position_accuracy_ok(self) -> bool:
        """位置精度是否满足±5mm 要求（用于业务判断）"""
        # 可根据实际需求实现，例如检查是否到达目标点附近
        return True
    
    def __str__(self) -> str:
        return (f"ArmState(pos=[{self.x:.1f}, {self.y:.1f}, {self.z:.1f}]mm, "
                f"pitch={math.degrees(self.pitch):.1f}°, "
                f"joints=[b:{self.base:.2f}, s:{self.shoulder:.2f}, e:{self.elbow:.2f}]rad)")


# ============== 状态解析器 ==============

class StateParser:
    """T:1051 状态反馈报文解析器"""
    
    @staticmethod
    def parse(raw: Dict) -> Optional[ArmState]:
        """
        解析原始 JSON 为 ArmState 对象
        
        Args:
            raw: 原始反馈字典 {"T": 1051, "x": ..., "y": ..., ...}
            
        Returns:
            ArmState 对象或 None（解析失败时）
        """
        if raw.get('T') != CommandCode.STATE_FEEDBACK:
            return None
        
        try:
            # 电压转换：v 单位是 0.01V
            voltage_raw = raw.get('v')
            voltage = (voltage_raw * 0.01) if voltage_raw is not None else None
            
            # 扭矩开关状态
            torque_switch = {
                'base': raw.get('torswitchB', 0),
                'shoulder': raw.get('torswitchS', 0),
                'elbow': raw.get('torswitchE', 0),
                'wrist': raw.get('torswitchT', 0),
                'wrist2': raw.get('torswitchR', 0),
                'gripper': raw.get('torswitchG', 0)
            }
            
            state = ArmState(
                # 末端位姿 (精度±5mm 足够，保留 2 位小数)
                x=round(raw.get('x', 0), 2),
                y=round(raw.get('y', 0), 2),
                z=round(raw.get('z', 0), 2),
                pitch=raw.get('tit', 0),  # ✅ 明确从 tit 读取 Pitch
                roll=raw.get('r', 0),
                
                # 关节角度 (弧度)
                base=raw.get('b', 0),
                shoulder=raw.get('s', 0),
                elbow=raw.get('e', 0),
                wrist=raw.get('t', 0),    # ✅ 明确从 t 读取 Wrist
                wrist2=raw.get('r', 0),
                gripper=raw.get('g', 0),
                
                # 关节负载
                torque_base=raw.get('tB', 0),
                torque_shoulder=raw.get('tS', 0),
                torque_elbow=raw.get('tE', 0),
                torque_wrist=raw.get('tT', 0),
                torque_wrist2=raw.get('tR', 0),
                torque_gripper=raw.get('tG', 0),
                
                # 扭矩开关状态
                torque_switch=torque_switch,
                
                # 系统信息
                voltage=voltage,
                timestamp=time.time()
            )
            return state
        except (TypeError, ValueError) as e:
            print(f"状态解析错误：{e}")
            return None


# ============== 主控制类（更新版） ==============

class RoArmM3S:
    """RoArm-M3-S 机械臂串口控制类（支持实时状态）"""
    
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        
        # 状态管理
        self._current_state: Optional[ArmState] = None
        self._state_lock = threading.Lock()
        self._state_callbacks: List[Callable[[ArmState], None]] = []
        
        # 通信线程
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 命令响应管理
        self._cmd_response_lock = threading.Lock()
        self._cmd_response: Optional[Dict] = None
        self._cmd_response_event = threading.Event()
        self._expected_cmd_type: Optional[int] = None
        
    def connect(self) -> bool:
        """连接串口"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                dsrdtr=False
            )
            self.ser.setRTS(False)
            self.ser.setDTR(False)
            time.sleep(2)  # 等待机械臂初始化
            self._running = True
            self._start_recv_thread()
            print(f"✓ 已连接到 {self.port}")
            return True
        except serial.SerialException as e:
            print(f"✗ 连接失败：{e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self._running = False
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"✓ 已断开连接")
    
    def _start_recv_thread(self):
        """启动接收线程（非阻塞读取持续反馈）"""
        def recv_loop():
            buffer = ""
            while self._running and self.ser and self.ser.is_open:
                try:
                    # 使用 readline 提高效率
                    line = self.ser.readline()
                    if not line:
                        time.sleep(0.001)
                        continue
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if not line_str or not line_str.startswith('{'):
                        continue
                    
                    self._handle_response(line_str)
                                
                except Exception as e:
                    if self._running:
                        print(f"接收线程错误：{e}")
                    break
        
        self._recv_thread = threading.Thread(target=recv_loop, daemon=True)
        self._recv_thread.start()
    
    def _handle_response(self, response: str):
        """处理返回的 JSON 响应"""
        try:
            data = json.loads(response)
            cmd_type = data.get('T')
            
            # 🎯 解析持续状态反馈 (T:1051)
            if cmd_type == CommandCode.STATE_FEEDBACK:
                state = StateParser.parse(data)
                if state:
                    with self._state_lock:
                        self._current_state = state
                    # 触发状态回调
                    for cb in self._state_callbacks:
                        try:
                            cb(state)
                        except Exception as e:
                            print(f"状态回调错误：{e}")
            
            # 🎯 处理命令响应 (T:104/105 等)
            elif self._expected_cmd_type and cmd_type == self._expected_cmd_type:
                with self._cmd_response_lock:
                    self._cmd_response = data
                    self._cmd_response_event.set()
                
        except json.JSONDecodeError:
            pass  # 忽略非 JSON 数据
    
    def _send_command(self, cmd: Dict, wait_response: bool = True, 
                     timeout: float = 2.0) -> Optional[Dict]:
        """
        发送 JSON 命令并等待响应
        
        Args:
            cmd: 命令字典
            wait_response: 是否等待响应
            timeout: 等待超时时间 (秒)
            
        Returns:
            响应字典或 None
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("串口未连接")
        
        # 清空上次响应
        with self._cmd_response_lock:
            self._cmd_response = None
            self._cmd_response_event.clear()
        
        # 设置期望的响应类型
        expected_type = cmd.get('T')
        with self._cmd_response_lock:
            self._expected_cmd_type = expected_type if wait_response else None
        
        # 发送命令
        cmd_str = json.dumps(cmd) + '\n'
        self.ser.write(cmd_str.encode('utf-8'))
        self.ser.flush()
        
        if not wait_response:
            return None
        
        # 等待响应
        if self._cmd_response_event.wait(timeout=timeout):
            with self._cmd_response_lock:
                result = self._cmd_response.copy() if self._cmd_response else None
                self._cmd_response = None
                return result
        return None
    
    # ============== 🎯 实时状态获取功能 ==============
    
    @property
    def current_state(self) -> Optional[ArmState]:
        """获取当前机械臂状态（线程安全）"""
        with self._state_lock:
            return self._current_state
    
    def request_state_feedback(self, timeout: float = 1.0) -> Optional[ArmState]:
        """
        主动请求一次状态反馈 (发送 T:105)
        
        Args:
            timeout: 等待响应的超时时间 (秒)
            
        Returns:
            ArmState 对象或 None
        """
        cmd = {"T": CommandCode.SERVO_FEEDBACK}
        response = self._send_command(cmd, wait_response=True, timeout=timeout)
        
        if response:
            state = StateParser.parse(response)
            if state:
                with self._state_lock:
                    self._current_state = state
                return state
        return None
    
    def wait_state_update(self, timeout: float = 1.0) -> Optional[ArmState]:
        """
        等待下一次状态更新（用于同步获取最新状态）
        
        Args:
            timeout: 最大等待时间 (秒)
        Returns:
            更新后的 ArmState 或 None
        """
        if not self._current_state:
            time.sleep(0.1)
        
        start_time = time.time()
        last_ts = self._current_state.timestamp if self._current_state else 0
        
        while time.time() - start_time < timeout:
            with self._state_lock:
                if self._current_state and self._current_state.timestamp > last_ts:
                    return self._current_state
            time.sleep(0.01)
        return self._current_state
    
    def register_state_callback(self, callback: Callable[[ArmState], None]):
        """注册状态更新回调（每次收到 T:1051 自动触发）"""
        if callback not in self._state_callbacks:
            self._state_callbacks.append(callback)
    
    def unregister_state_callback(self, callback: Callable[[ArmState], None]):
        """注销状态回调"""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)
    
    def get_state_snapshot(self) -> Optional[Dict]:
        """获取状态快照字典（便于序列化/日志）"""
        state = self.current_state
        if not state:
            return None
        return {
            'timestamp': state.timestamp,
            'end_effector': state.end_effector_pose,
            'joints_rad': state.joint_angles_rad,
            'joints_deg': state.joint_angles_deg,
            'torques': {
                'base': state.torque_base,
                'shoulder': state.torque_shoulder,
                'elbow': state.torque_elbow,
                'wrist': state.torque_wrist,
                'wrist2': state.torque_wrist2,
                'gripper': state.torque_gripper
            },
            'torque_switches': state.torque_switch,
            'gripper_closed': state.is_gripper_closed,
            'voltage': state.voltage
        }
    
    # ============== 运动控制指令 ==============
    
    def move_to_init(self, timeout: float = 5.0) -> bool:
        """复位到初始位置"""
        cmd = {"T": CommandCode.MOVE_INIT}
        return self._send_command(cmd, timeout=timeout) is not None

    def torque_control(self, cmd: int, max_retries: int = 1) -> bool:
        """TORQUE CTRL - 扭矩锁控制 (T:210)"""
        if cmd not in (0, 1):
            raise ValueError("cmd must be 0 (off) or 1 (on)")
        payload = {
            "T": CommandCode.TORQUE_CTRL,
            "cmd": cmd
        }
        # 该指令一般不会返回详细状态，直接发送即可
        for attempt in range(max_retries):
            try:
                result = self._send_command(payload, wait_response=False)
                if result is None:
                    # 非阻塞 send，返回 None 视为成功
                    return True
            except RuntimeError:
                pass
            if attempt < max_retries - 1:
                time.sleep(0.1)
        return False

    def move_joints_angle(self, b: float = 0, s: float = 0, e: float = 90,
                          t: float = 0, r: float = 0, h: float = 180,
                          spd: float = 0, acc: float = 0, blocking: bool = True) -> bool:
        """全部关节控制（角度制）"""
        cmd = {
            "T": CommandCode.JOINTS_ANGLE,
            "b": round(b, 2), "s": round(s, 2), "e": round(e, 2),
            "t": round(t, 2), "r": round(r, 2), "h": round(h, 2),
            "spd": spd, "acc": acc
        }
        return self._send_command(cmd, wait_response=blocking) is not None if blocking else True
    
    def move_to_xyz(self, x: float, y: float, z: float, 
                    pitch: float = 0, roll: float = 0, gripper: float = 3.14,
                    spd: float = 0.25, blocking: bool = True) -> bool:
        """
        末端位置控制（逆运动学）
        
        Args:
            x, y, z: 末端目标位置 (mm)，精度±5mm
            pitch: 末端俯仰角 (弧度，tit 轴)
            roll: 末端翻滚角 (弧度，r 轴)
            gripper: 夹爪角度 (弧度，3.14≈180°)
            spd: 运动速度系数 (仅 blocking=True 时生效，0-1)
            blocking: 
                True  - 使用 CMD_XYZT_GOAL(104)，有轨迹规划，等待执行完成
                False - 使用 CMD_XYZT_DIRECT(1041)，无插值直驱，发送即返回
                        ⚠️ 适合高频连续更新，相邻目标点位移建议 < 5mm
        
        Returns:
            blocking=True: 执行完成返回 True，超时返回 False
            blocking=False: 指令成功写入串口即返回 True
        """
        if blocking:
            # 阻塞模式：使用 104，带插值规划，等待执行完成
            cmd = {
                "T": CommandCode.XYZT_GOAL,
                "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
                "t": round(pitch, 4), "r": round(roll, 4), "g": round(gripper, 4),
                "spd": spd
            }
            response = self._send_command(cmd, wait_response=True, timeout=10.0)
            return response is not None
        else:
            # 非阻塞模式：使用 1041，直驱，发送即返回
            cmd = {
                "T": CommandCode.XYZT_DIRECT,
                "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
                "t": round(pitch, 4), "r": round(roll, 4), "g": round(gripper, 4)
                # 1041 不支持 spd 参数
            }
            try:
                self._send_command(cmd, wait_response=False)
                return True
            except RuntimeError:
                return False
    
    def move_gripper_angle(self, angle: float, spd: float = 0, acc: float = 0) -> bool:
        """夹爪控制（角度制，45°~180°）"""
        angle = max(45, min(180, angle))
        return self.move_joints_angle(h=angle, spd=spd, acc=acc, blocking=False)
    
    def get_current_position(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        获取当前末端位置（主动请求反馈）
        
        Args:
            timeout: 等待超时时间 (秒)
            
        Returns:
            {'x': float, 'y': float, 'z': float, 'pitch': float, 'roll': float} 或 None
        """
        state = self.request_state_feedback(timeout=timeout)
        if state:
            return {
                'x': state.x,
                'y': state.y,
                'z': state.z,
                'pitch': state.pitch,
                'roll': state.roll,
                'timestamp': state.timestamp
            }
        return None
    
    # ============== 上下文管理器 ==============
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
    
    def __del__(self):
        self.disconnect()


if __name__ == "__main__":
    # 简单使用示例：调用 move_to_xyz 的全部参数
    # sym:move_to_xyz
    port = "/dev/ttyUSB0"
    with RoArmM3S(port=port, baudrate=115200, timeout=1.0) as arm:
        if not arm.ser or not arm.ser.is_open:
            print("✗ 串口未连接，请检查端口")
        else:
            print("✓ 连接成功，执行 move_to_xyz 示例")
            success = arm.move_to_xyz(
                x=200.0,
                y=0.0,
                z=150.0,
                pitch=0.0,
                roll=0.0,
                gripper=3.14,
                spd=0.25,
                blocking=False
            )
            print(f"move_to_xyz 执行结果: {success}")

            # 等待一段时间以读取反馈状态
            time.sleep(1.0)
            state = arm.get_current_position(timeout=1.0)
            print(f"当前状态: {state}")
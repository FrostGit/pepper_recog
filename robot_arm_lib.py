import serial
import json
import threading
import time
import logging
from typing import Optional, Dict, Callable, Any
from collections import deque
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobotArmLib")


class StatusQueue:
    """线程安全的状态队列，用于解耦读取线程和回调处理"""
    def __init__(self, max_size: int = 100):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def put(self, status: Dict):
        with self.lock:
            self.queue.append(status)
    
    def get_all(self) -> list:
        with self.lock:
            data = list(self.queue)
            self.queue.clear()
            return data
    
    def get_latest(self) -> Optional[Dict]:
        with self.lock:
            if self.queue:
                return self.queue[-1]
            return None
    
    def clear(self):
        with self.lock:
            self.queue.clear()


class RobotArmController:
    def __init__(
        self, 
        port: str, 
        baudrate: int = 115200, 
        timeout: float = 1.0,
        max_buffer_size: int = 10240,
        status_queue_size: int = 100
    ):
        """
        初始化机械臂控制器
        :param port: 串口端口号，例如 'COM3' (Windows) 或 '/dev/ttyUSB0' (Linux)
        :param baudrate: 波特率，默认 115200
        :param timeout: 串口读取超时时间
        :param max_buffer_size: 串口缓冲区最大大小（字节）
        :param status_queue_size: 状态队列最大长度
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.max_buffer_size = max_buffer_size
        
        self.serial_conn: Optional[serial.Serial] = None
        self.is_running = False
        self.is_connected = False
        self.read_thread: Optional[threading.Thread] = None
        self.callback_thread: Optional[threading.Thread] = None
        
        # 状态存储与锁
        self.latest_status: Dict = {}
        self.status_lock = threading.Lock()
        self.status_timestamp: float = 0.0  # 上次状态更新的本地时间戳
        
        # 状态队列（用于回调处理）
        self.status_queue = StatusQueue(max_size=status_queue_size)
        
        # 状态回调函数 (可选)
        self.status_callback: Optional[Callable[[Dict], None]] = None
        
        # 串口读写锁
        self.send_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'bytes_received': 0,
            'messages_received': 0,
            'parse_errors': 0,
            'last_error': None,
            'connect_time': None
        }
        self.stats_lock = threading.Lock()

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
        return False  # 不抑制异常

    def connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """
        连接串口
        :param max_retries: 最大重试次数
        :param retry_delay: 重试间隔（秒）
        :return: 连接是否成功
        """
        for attempt in range(max_retries):
            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                
                self.is_running = True
                self.is_connected = True
                
                # 启动读取线程
                self.read_thread = threading.Thread(target=self._read_loop, daemon=True, name="ReadThread")
                self.read_thread.start()
                
                # 启动回调处理线程
                self.callback_thread = threading.Thread(target=self._callback_loop, daemon=True, name="CallbackThread")
                self.callback_thread.start()
                
                # 记录连接时间
                with self.stats_lock:
                    self.stats['connect_time'] = datetime.now().isoformat()
                
                logger.info(f"成功连接到串口 {self.port} (尝试 {attempt + 1}/{max_retries})")
                
                # 等待初始状态
                time.sleep(0.5)
                return True
                
            except Exception as e:
                logger.warning(f"连接尝试 {attempt + 1}/{max_retries} 失败：{e}")
                with self.stats_lock:
                    self.stats['last_error'] = str(e)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"连接失败，已尝试 {max_retries} 次")
                    self.is_connected = False
                    return False
        
        return False

    def disconnect(self):
        """断开连接"""
        logger.info("正在断开连接...")
        self.is_running = False
        self.is_connected = False
        
        # 强制关闭串口以唤醒阻塞的 read 调用
        if self.serial_conn:
            try:
                if self.serial_conn.is_open:
                    self.serial_conn.close()
            except Exception as e:
                logger.warning(f"关闭串口时出错：{e}")
            self.serial_conn = None
        
        # 等待线程退出
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        
        if self.callback_thread and self.callback_thread.is_alive():
            self.callback_thread.join(timeout=2.0)
        
        # 清空状态
        with self.status_lock:
            self.latest_status = {}
            self.status_timestamp = 0.0
        self.status_queue.clear()
        
        logger.info("已断开连接")

    def _read_loop(self):
        """后台读取线程，处理持续上报的状态"""
        buffer = ""
        
        while self.is_running:
            try:
                # 检查串口是否可用
                if not self.serial_conn or not self.serial_conn.is_open:
                    logger.warning("串口不可用，退出读取线程")
                    break
                
                if self.serial_conn.in_waiting > 0:
                    # 一次性读取所有可用数据
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    
                    with self.stats_lock:
                        self.stats['bytes_received'] += len(data)
                    
                    try:
                        # 尝试解码，如果失败则忽略该批次
                        text = data.decode('utf-8', errors='ignore')
                    except Exception as e:
                        logger.warning(f"数据解码失败：{e}")
                        continue
                        
                    buffer += text
                    
                    # 限制缓冲区大小
                    if len(buffer) > self.max_buffer_size:
                        logger.warning("串口缓冲区溢出，丢弃旧数据")
                        buffer = buffer[-self.max_buffer_size:]
                    
                    # 处理完整的数据包（兼容 \r\n 和 \n）
                    while True:
                        line = None
                        if '\r\n' in buffer:
                            line, buffer = buffer.split('\r\n', 1)
                        elif '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                        else:
                            break  # 没有完整行，等待更多数据
                            
                        line = line.strip()
                        if line:
                            self._parse_status(line)
                else:
                    # 没有数据时短暂休眠，降低 CPU 占用
                    time.sleep(0.01)
                    
            except serial.SerialException as e:
                logger.error(f"串口异常：{e}")
                with self.stats_lock:
                    self.stats['last_error'] = str(e)
                break
            except Exception as e:
                logger.error(f"读取线程未知错误：{e}")
                with self.stats_lock:
                    self.stats['last_error'] = str(e)
                break
        
        logger.info("读取线程已退出")

    def _parse_status(self, json_str: str):
        """解析状态 JSON 并更新缓存"""
        try:
            # 提取 JSON 对象（处理可能的脏数据）
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start == -1 or end == -1 or end <= start:
                return
            
            clean_json = json_str[start:end+1]
            data = json.loads(clean_json)
            
            with self.stats_lock:
                self.stats['messages_received'] += 1
            
            # 【关键修复】只更新 T:1051 的状态数据
            if data.get('T') == 1051:
                # 添加本地接收时间戳
                data['_recv_time'] = time.time()
                data['_recv_datetime'] = datetime.fromtimestamp(data['_recv_time']).isoformat()
                
                with self.status_lock:
                    self.latest_status = data
                    self.status_timestamp = data['_recv_time']
                
                # 放入队列供回调线程处理
                self.status_queue.put(data)
                
                logger.debug(f"状态更新：X={data.get('x'):.2f}, Y={data.get('y'):.2f}, Z={data.get('z'):.2f}")
            else:
                # 非状态数据（如命令响应）记录日志
                logger.debug(f"收到非状态报文：T={data.get('T')}")
                    
        except json.JSONDecodeError as e:
            with self.stats_lock:
                self.stats['parse_errors'] += 1
            # 静默忽略解析错误，避免日志刷屏
        except Exception as e:
            logger.error(f"解析状态出错：{e}")
            with self.stats_lock:
                self.stats['last_error'] = str(e)

    def _callback_loop(self):
        """后台回调处理线程，避免阻塞读取线程"""
        while self.is_running:
            try:
                status = self.status_queue.get_latest()
                if status and self.status_callback:
                    try:
                        self.status_callback(status)
                        self.status_queue.clear()  # 处理完后清空，避免重复回调
                    except Exception as e:
                        logger.error(f"状态回调执行错误：{e}")
                
                time.sleep(0.05)  # 回调频率限制，避免过于频繁
            except Exception as e:
                logger.error(f"回调线程错误：{e}")
                break
        
        logger.info("回调线程已退出")

    def _send_command(self, cmd_dict: dict, max_retries: int = 1) -> bool:
        """
        发送 JSON 命令
        :param cmd_dict: 命令字典
        :param max_retries: 发送失败重试次数
        :return: 发送是否成功
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            logger.error("串口未连接")
            return False
        
        for attempt in range(max_retries):
            try:
                json_str = json.dumps(cmd_dict) + "\n"
                with self.send_lock:
                    bytes_sent = self.serial_conn.write(json_str.encode('utf-8'))
                    self.serial_conn.flush()
                
                if bytes_sent > 0:
                    logger.debug(f"发送指令：{cmd_dict}")
                    return True
                else:
                    logger.warning(f"发送指令失败，尝试 {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"发送指令失败：{e}")
                with self.stats_lock:
                    self.stats['last_error'] = str(e)
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                else:
                    return False
        
        return False

    # ================= 状态获取方法 =================

    def get_status(self) -> dict:
        """获取最新的状态快照（包含时间戳）"""
        with self.status_lock:
            return self.latest_status.copy()

    def get_status_age(self) -> float:
        """
        获取状态数据的年龄（秒）
        :return: 距离上次状态更新的时间（秒），-1 表示无数据
        """
        with self.status_lock:
            if self.status_timestamp > 0:
                return time.time() - self.status_timestamp
            return -1

    def is_status_fresh(self, max_age: float = 1.0) -> bool:
        """
        检查状态是否新鲜
        :param max_age: 最大允许年龄（秒）
        :return: 状态是否在有效期内
        """
        return self.get_status_age() < max_age

    def get_stats(self) -> dict:
        """获取通信统计信息"""
        with self.stats_lock:
            return self.stats.copy()

    def get_position(self) -> Optional[Dict[str, float]]:
        """
        获取机械臂末端位置
        :return: {'x': float, 'y': float, 'z': float} 或 None
        """
        status = self.get_status()
        if status and 'x' in status and 'y' in status and 'z' in status:
            return {
                'x': status.get('x', 0.0),
                'y': status.get('y', 0.0),
                'z': status.get('z', 0.0)
            }
        return None

    def get_joints(self) -> Optional[Dict[str, float]]:
        """
        获取所有关节角度（弧度制）
        :return: 关节角度字典 或 None
        """
        status = self.get_status()
        if status and 'b' in status:
            return {
                'base': status.get('b', 0.0),
                'shoulder': status.get('s', 0.0),
                'elbow': status.get('e', 0.0),
                'wrist': status.get('t', 0.0),
                'roll': status.get('r', 0.0),
                'hand': status.get('g', 0.0)
            }
        return None

    # ================= 控制指令封装 =================

    def torque_control(self, cmd: int, max_retries: int = 1) -> bool:
        """
        TORQUE CTRL - 扭矩锁控制 (T:210)
        :param cmd: 0-关闭扭矩锁，1-开启扭矩锁
        :param max_retries: 重试次数
        """
        return self._send_command({"T": 210, "cmd": cmd}, max_retries=max_retries)

    def dynamic_adaptation(
        self, 
        mode: int, 
        b: int = 1000, s: int = 1000, e: int = 1000, 
        t: int = 1000, r: int = 1000, h: int = 1000,
        max_retries: int = 1
    ) -> bool:
        """
        DYNAMIC ADAPTATION - 动态外力自适应 (T:112)
        :param mode: 0-关闭，1-开启
        :param b,s,e,t,r,h: 各关节最大输出扭矩限制
        """
        return self._send_command({
            "T": 112, "mode": mode, "b": b, "s": s, "e": e, 
            "t": t, "r": r, "h": h
        }, max_retries=max_retries)

    def move_init(self, max_retries: int = 1) -> bool:
        """
        CMD_MOVE_INIT - 运动到初始位置 (T:100)
        ⚠️ 注意：该指令会引起机械臂内部阻塞
        """
        logger.warning("发送阻塞式运动指令 T:100")
        return self._send_command({"T": 100}, max_retries=max_retries)

    def move_single_joint_rad(
        self, joint: int, rad: float, spd: int = 0, acc: int = 10, 
        max_retries: int = 1
    ) -> bool:
        """
        CMD_SINGLE_JOINT_CTRL - 单独关节控制（弧度制）(T:101)
        :param joint: 1-6 (关节序号)
        :param rad: 目标角度 (弧度)
        :param spd: 速度 (步/秒), 0 为最大
        :param acc: 加速度 (100 步/秒^2)
        """
        return self._send_command(
            {"T": 101, "joint": joint, "rad": rad, "spd": spd, "acc": acc}, 
            max_retries=max_retries
        )

    def move_joints_rad(
        self, base: float, shoulder: float, elbow: float, wrist: float, 
        roll: float, hand: float, spd: int = 0, acc: int = 10,
        max_retries: int = 1
    ) -> bool:
        """
        CMD_JOINTS_RAD_CTRL - 全部关节控制（弧度制）(T:102)
        """
        return self._send_command({
            "T": 102, "base": base, "shoulder": shoulder, "elbow": elbow, 
            "wrist": wrist, "roll": roll, "hand": hand, "spd": spd, "acc": acc
        }, max_retries=max_retries)

    def move_eoat_rad(self, cmd: float, spd: int = 0, acc: int = 0, max_retries: int = 1) -> bool:
        """
        CMD_EOAT_HAND_CTRL - 末端关节控制（弧度制）(T:106)
        :param cmd: 夹爪角度 (弧度), 1.08-3.14
        """
        return self._send_command(
            {"T": 106, "cmd": cmd, "spd": spd, "acc": acc}, 
            max_retries=max_retries
        )

    def move_single_joint_angle(
        self, joint: int, angle: float, spd: int = 10, acc: int = 10, 
        max_retries: int = 1
    ) -> bool:
        """
        CMD_SINGLE_JOINT_ANGLE - 单独关节控制（角度制）(T:121)
        :param joint: 1-6
        :param angle: 目标角度 (度)
        """
        return self._send_command(
            {"T": 121, "joint": joint, "angle": angle, "spd": spd, "acc": acc}, 
            max_retries=max_retries
        )

    def move_joints_angle(
        self, b: float, s: float, e: float, t: float, r: float, h: float, 
        spd: int = 10, acc: int = 10, max_retries: int = 1
    ) -> bool:
        """
        CMD_JOINTS_ANGLE_CTRL - 全部关节控制（角度制）(T:122)
        :param b: Base, s: Shoulder, e: Elbow, t: Wrist, r: Roll, h: Hand
        """
        return self._send_command({
            "T": 122, "b": b, "s": s, "e": e, "t": t, "r": r, "h": h, 
            "spd": spd, "acc": acc
        }, max_retries=max_retries)

    def move_single_axis(
        self, axis: int, pos: float, spd: float = 0.25, max_retries: int = 1
    ) -> bool:
        """
        CMD_SINGLE_AXIS_CRTL - 机械臂末端点的单独轴位置控制 (T:103)
        :param axis: 1-X, 2-Y, 3-Z, 4-T, 5-R, 6-G
        :param pos: 位置 (mm 或 弧度)
        :param spd: 速度
        ⚠️ 注意：该指令会引起机械臂内部阻塞
        """
        logger.warning("发送阻塞式运动指令 T:103")
        return self._send_command(
            {"T": 103, "axis": axis, "pos": pos, "spd": spd}, 
            max_retries=max_retries
        )

    def move_xyzt_goal(
        self, x: float, y: float, z: float, t: float, r: float, g: float, 
        spd: float = 0.25, max_retries: int = 1
    ) -> bool:
        """
        CMD_XYZT_GOAL_CTRL - 机械臂末端点位置控制 (逆运动学) (T:104)
        ⚠️ 注意：该指令会引起机械臂内部阻塞
        """
        logger.warning("发送阻塞式运动指令 T:104")
        return self._send_command({
            "T": 104, "x": x, "y": y, "z": z, "t": t, "r": r, "g": g, "spd": spd
        }, max_retries=max_retries)

    def move_xyzt_direct(
        self, x: float, y: float, z: float, t: float, r: float, g: float, 
        max_retries: int = 1
    ) -> bool:
        """
        CMD_XYZT_DIRECT_CTRL - 机械臂末端点位置控制 (非阻塞) (T:1041)
        适合连续给新的目标点
        """
        return self._send_command({
            "T": 1041, "x": x, "y": y, "z": z, "t": t, "r": r, "g": g
        }, max_retries=max_retries)

    def request_feedback(self, max_retries: int = 1) -> bool:
        """
        CMD_SERVO_RAD_FEEDBACK - 主动请求机械臂反馈 (T:105)
        机械臂收到后会回复 T:1051 格式的数据
        """
        return self._send_command({"T": 105}, max_retries=max_retries)

    def constant_control(
        self, m: int, axis: int, cmd: int, spd: int = 0, max_retries: int = 1
    ) -> bool:
        """
        CMD_CONSTANT_CTRL - 连续运动控制 (T:123)
        :param m: 0-角度控制，1-坐标控制
        :param axis: 控制的关节或轴序号
        :param cmd: 0-STOP, 1-INCREASE, 2-DECREASE
        :param spd: 速度系数 0-20
        """
        return self._send_command(
            {"T": 123, "m": m, "axis": axis, "cmd": cmd, "spd": spd}, 
            max_retries=max_retries
        )

    def stop_all_motion(self) -> bool:
        """停止所有连续运动"""
        return self.constant_control(m=0, axis=0, cmd=0, spd=0)


# ================= 示例用法 =================

if __name__ == "__main__":
    # 请根据实际情况修改端口号
    PORT_NAME = "COM3"  # Windows: 'COM3', Linux: '/dev/ttyUSB0'
    
    # 定义状态回调函数
    def on_status_update(status: Dict):
        """状态更新回调（在独立线程中执行）"""
        recv_time = status.get('_recv_datetime', 'N/A')
        x = status.get('x', 0)
        y = status.get('y', 0)
        z = status.get('z', 0)
        print(f"[{recv_time}] 位置更新：X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
    
    # 使用上下文管理器（推荐）
    with RobotArmController(port=PORT_NAME) as arm:
        if not arm.is_connected:
            print("连接失败，请检查端口号")
            exit(1)
        
        # 设置状态回调
        arm.status_callback = on_status_update
        
        try:
            # 等待初始状态
            print("等待初始状态...")
            time.sleep(2)
            
            # 检查连接状态
            print(f"\n连接状态：{arm.is_connected}")
            print(f"状态年龄：{arm.get_status_age():.2f} 秒")
            print(f"状态新鲜度：{arm.is_status_fresh()}")
            
            # 获取当前状态
            status = arm.get_status()
            if status:
                print(f"\n当前状态 T 码：{status.get('T')}")
                print(f"当前坐标：X={status.get('x')}, Y={status.get('y')}, Z={status.get('z')}")
            
            # 获取位置
            pos = arm.get_position()
            if pos:
                print(f"\n末端位置：{pos}")
            
            # 获取关节角度
            joints = arm.get_joints()
            if joints:
                print(f"关节角度：{joints}")
            
            # 获取统计信息
            stats = arm.get_stats()
            print(f"\n通信统计：{stats}")
            
            # 示例：运动控制（取消注释以执行）
            # 1. 关闭扭矩锁
            # arm.torque_control(0)
            
            # 2. 运动到初始位置
            # arm.move_init()
            # time.sleep(3)
            
            # 3. 关节角度控制
            # arm.move_joints_angle(b=0, s=0, e=90, t=0, r=0, h=180, spd=10, acc=10)
            
            # 4. 笛卡尔坐标控制（非阻塞，适合连续运动）
            # arm.move_xyzt_direct(x=235, y=0, z=234, t=0, r=0, g=3.14)
            
            # 5. 主动请求反馈
            # arm.request_feedback()
            
            # 保持运行以接收状态上报
            print("\n开始实时监控（按 Ctrl+C 停止）...")
            while True:
                time.sleep(1)
                
                # 检查状态新鲜度
                if not arm.is_status_fresh(max_age=2.0):
                    logger.warning("状态数据过期，可能连接已断开")
                
                # 打印状态年龄
                age = arm.get_status_age()
                if age >= 0:
                    print(f"状态年龄：{age:.2f} 秒", end='\r')
                    
        except KeyboardInterrupt:
            print("\n程序停止")
        except Exception as e:
            logger.error(f"程序异常：{e}")
        finally:
            # 上下文管理器会自动调用 disconnect()
            print("\n清理完成")
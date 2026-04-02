#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂实时监控程序
启动后自动关闭扭矩锁，实时显示末端位置、姿态和夹爪开度
"""

import time
import sys
import os
from datetime import datetime
from robot_arm_lib import RobotArmController

# ================= 配置区 =================
PORT = "/dev/ttyUSB0"              # 修改为你的串口
BAUDRATE = 115200
REFRESH_RATE = 0.2         # 刷新频率 (秒)
AUTO_TORQUE_OFF = True     # 启动后自动关闭扭矩锁
SHOW_DEBUG = False         # 是否显示调试信息
# =========================================


class StatusMonitor:
    """状态监控器"""
    
    def __init__(self, arm: RobotArmController):
        self.arm = arm
        self.start_time = time.time()
        self.update_count = 0
        self.error_count = 0
        self.last_status = {}
        
    def rad_to_deg(self, rad: float) -> float:
        """弧度转角度"""
        return rad * 180.0 / 3.14159265359
    
    def get_gripper_opening(self, g_rad: float) -> dict:
        """
        计算夹爪开度
        协议：g 范围 1.08(全开) 至 3.14(全闭) 弧度
        """
        g_min = 1.08   # 全开
        g_max = 3.14   # 全闭
        
        # 计算开度百分比 (0%=全闭，100%=全开)
        opening = 100.0 * (g_max - g_rad) / (g_max - g_min)
        opening = max(0, min(100, opening))  # 限制在 0-100
        
        return {
            'angle': g_rad,
            'angle_deg': self.rad_to_deg(g_rad),
            'opening_percent': opening,
            'status': '🔴 闭合' if opening < 20 else ('🟡 半开' if opening < 80 else '🟢 全开')
        }
    
    def get_orientation(self, status: dict) -> dict:
        """
        获取机械臂末端姿态
        根据协议：
        - t: WRIST_JOINT (Pitch)
        - r: ROLL_JOINT (Roll)
        - tit: 末端关节姿态 (可用于计算 Yaw)
        """
        return {
            'roll': {
                'rad': status.get('r', 0),
                'deg': self.rad_to_deg(status.get('r', 0))
            },
            'pitch': {
                'rad': status.get('t', 0),
                'deg': self.rad_to_deg(status.get('t', 0))
            },
            'yaw': {
                'rad': status.get('tit', 0),
                'deg': self.rad_to_deg(status.get('tit', 0))
            }
        }
    
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """打印表头"""
        self.clear_screen()
        print("=" * 70)
        print("🤖 机械臂实时监控面板".center(70))
        print("=" * 70)
        print(f"串口：{PORT} | 波特率：{BAUDRATE} | 刷新率：{1/REFRESH_RATE:.1f} Hz")
        print(f"运行时间：{time.time() - self.start_time:.0f} 秒 | 更新次数：{self.update_count}")
        print("=" * 70)
    
    def print_position(self, status: dict):
        """打印位置信息"""
        print("\n📍 末端位置 (mm)")
        print("-" * 70)
        x = status.get('x', 0)
        y = status.get('y', 0)
        z = status.get('z', 0)
        print(f"  X 轴：{x:>10.3f} mm")
        print(f"  Y 轴：{y:>10.3f} mm")
        print(f"  Z 轴：{z:>10.3f} mm")
    
    def print_orientation(self, status: dict):
        """打印姿态信息"""
        orient = self.get_orientation(status)
        print("\n🎯 末端姿态 (角度)")
        print("-" * 70)
        print(f"  Roll  (翻滚): {orient['roll']['deg']:>10.3f}°  ({orient['roll']['rad']:.4f} rad)")
        print(f"  Pitch (俯仰): {orient['pitch']['deg']:>10.3f}°  ({orient['pitch']['rad']:.4f} rad)")
        print(f"  Yaw   (偏航): {orient['yaw']['deg']:>10.3f}°  ({orient['yaw']['rad']:.4f} rad)")
    
    def print_gripper(self, status: dict):
        """打印夹爪信息"""
        g = status.get('g', 3.14)
        gripper = self.get_gripper_opening(g)
        
        print("\n🔧 夹爪状态")
        print("-" * 70)
        print(f"  角度：{gripper['angle_deg']:>10.3f}°  ({gripper['angle']:.4f} rad)")
        print(f"  开度：{gripper['opening_percent']:>10.1f} %")
        print(f"  状态：{gripper['status']}")
    
    def print_joints(self, status: dict):
        """打印关节角度（可选）"""
        print("\n🦾 关节角度 (弧度)")
        print("-" * 70)
        print(f"  Base    (基础): {status.get('b', 0):>10.4f} rad  ({self.rad_to_deg(status.get('b', 0)):>6.1f}°)")
        print(f"  Shoulder(肩)  : {status.get('s', 0):>10.4f} rad  ({self.rad_to_deg(status.get('s', 0)):>6.1f}°)")
        print(f"  Elbow   (肘)  : {status.get('e', 0):>10.4f} rad  ({self.rad_to_deg(status.get('e', 0)):>6.1f}°)")
        print(f"  Wrist   (腕 1): {status.get('t', 0):>10.4f} rad  ({self.rad_to_deg(status.get('t', 0)):>6.1f}°)")
        print(f"  Roll    (腕 2): {status.get('r', 0):>10.4f} rad  ({self.rad_to_deg(status.get('r', 0)):>6.1f}°)")
        print(f"  Gripper (夹爪): {status.get('g', 0):>10.4f} rad  ({self.rad_to_deg(status.get('g', 0)):>6.1f}°)")
    
    def print_torque_status(self, status: dict):
        """打印扭矩锁状态"""
        print("\n🔒 扭矩锁状态")
        print("-" * 70)
        
        torque_keys = {
            'torswitchB': 'Base',
            'torswitchS': 'Shoulder',
            'torswitchE': 'Elbow',
            'torswitchT': 'Wrist',
            'torswitchR': 'Roll',
            'torswitchG': 'Gripper'
        }
        
        status_map = {0: '🔓 关闭', 1: '🔒 开启'}
        
        for key, name in torque_keys.items():
            val = status.get(key, -1)
            if val >= 0:
                print(f"  {name:>10}: {status_map.get(val, '未知')}")
    
    def print_load(self, status: dict):
        """打印负载信息"""
        print("\n⚡ 关节负载")
        print("-" * 70)
        print(f"  Base    (tB): {status.get('tB', 0):>6}")
        print(f"  Shoulder(tS): {status.get('tS', 0):>6}")
        print(f"  Elbow   (tE): {status.get('tE', 0):>6}")
        print(f"  Wrist   (tT): {status.get('tT', 0):>6}")
        print(f"  Roll    (tR): {status.get('tR', 0):>6}")
        print(f"  Gripper (tG): {status.get('tG', 0):>6}")
    
    def print_voltage(self, status: dict):
        """打印电压信息"""
        v = status.get('v', 0)
        voltage = v * 0.01  # 协议：单位 0.01V
        print("\n🔋 电源信息")
        print("-" * 70)
        print(f"  电压：{voltage:.2f} V")
    
    def print_status_bar(self, status: dict):
        """打印状态栏"""
        print("\n" + "=" * 70)
        
        # 状态新鲜度
        age = self.arm.get_status_age()
        if age >= 0:
            freshness = "✅ 正常" if age < 1.0 else ("⚠️  延迟" if age < 3.0 else "❌ 过期")
        else:
            freshness = "❌ 无数据"
        
        # 连接状态
        conn_status = "✅ 已连接" if self.arm.is_connected else "❌ 已断开"
        
        print(f"连接状态：{conn_status} | 数据新鲜度：{freshness} | 数据年龄：{age:.2f} 秒")
        print(f"上次更新：{datetime.fromtimestamp(status.get('_recv_time', 0)).strftime('%H:%M:%S.%f')[:-3]}")
        print("=" * 70)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n💡 操作提示：按 Ctrl+C 退出程序")
    
    def update(self):
        """更新显示"""
        status = self.arm.get_status()
        
        if not status or status.get('T') != 1051:
            self.error_count += 1
            if self.error_count > 10:
                print("⚠️  警告：连续未收到有效状态数据")
            return
        
        self.last_status = status
        self.update_count += 1
        self.error_count = 0
        
        # 打印所有信息
        self.print_header()
        self.print_position(status)
        self.print_orientation(status)
        self.print_gripper(status)
        # self.print_joints(status)
        self.print_torque_status(status)
        # self.print_load(status)
        # self.print_voltage(status)
        self.print_status_bar(status)
        self.print_help()
        
        if SHOW_DEBUG:
            stats = self.arm.get_stats()
            print(f"\n📊 调试信息：接收字节={stats['bytes_received']}, 消息数={stats['messages_received']}")


def main():
    """主函数"""
    print("=" * 70)
    print("🤖 机械臂实时监控程序".center(70))
    print("=" * 70)
    print(f"串口：{PORT}")
    print(f"波特率：{BAUDRATE}")
    print(f"自动关闭扭矩锁：{'是' if AUTO_TORQUE_OFF else '否'}")
    print("=" * 70)
    
    # 创建控制器
    arm = RobotArmController(port=PORT, baudrate=BAUDRATE)
    monitor = StatusMonitor(arm)
    
    try:
        # 连接
        print("\n🔌 正在连接机械臂...")
        if not arm.connect(max_retries=3):
            print("❌ 连接失败，请检查串口配置")
            return
        
        print("✅ 连接成功")
        
        # 等待初始状态
        print("⏳ 等待初始状态...")
        time.sleep(2)
        
        # 检查连接
        if not arm.is_status_fresh(max_age=5.0):
            print("⚠️  警告：未收到状态上报，但继续运行")
        else:
            print("✅ 状态数据正常")
        
        # 关闭扭矩锁
        if AUTO_TORQUE_OFF:
            print("\n🔓 正在关闭扭矩锁...")
            result = arm.torque_control(cmd=0)
            if result:
                print("✅ 扭矩锁已关闭")
            else:
                print("❌ 关闭扭矩锁失败")
            time.sleep(1)
        
        # 清屏
        time.sleep(1)
        monitor.clear_screen()
        
        # 主循环
        print("🔄 开始实时监控...")
        while True:
            monitor.update()
            time.sleep(REFRESH_RATE)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在退出...")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复扭矩锁（可选）
        if AUTO_TORQUE_OFF:
            print("\n🔒 正在恢复扭矩锁...")
            arm.torque_control(cmd=1)
            time.sleep(0.5)
        
        # 断开连接
        print("🔌 断开连接...")
        arm.disconnect()
        
        # 显示统计
        if monitor.update_count > 0:
            stats = arm.get_stats()
            print("\n📊 会话统计:")
            print(f"  运行时间：{time.time() - monitor.start_time:.1f} 秒")
            print(f"  状态更新：{monitor.update_count} 次")
            print(f"  接收字节：{stats['bytes_received']}")
            print(f"  接收消息：{stats['messages_received']}")
            print(f"  解析错误：{stats['parse_errors']}")
        
        print("\n✅ 程序退出完成")


if __name__ == "__main__":
    main()
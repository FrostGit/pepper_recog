#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoArm-M3-S 机械臂完整使用示例
功能：记录初始位置 → 移动到目标点 → 夹爪操作 → 返回初始位置

Author: 霜叶
Date: 2026
"""

import time
import sys
from roarm_m3 import RoArmM3S, ArmState  # 假设保存为 roarm_m3s.py


# ============== 配置参数 ==============

# 串口配置
SERIAL_PORT = "/dev/ttyUSB0"  # Linux: /dev/ttyUSB0, Windows: COM3/COM4
BAUDRATE = 115200

# 目标位置 (mm)
TARGET_X = 300
TARGET_Y = 0
TARGET_Z = 250
TARGET_PITCH = 0      # 弧度，0 表示水平
TARGET_ROLL = 0       # 弧度
TARGET_GRIPPER_OPEN = 0.5    # 夹爪张开角度 (弧度，约 28°)
TARGET_GRIPPER_CLOSE = 3.14  # 夹爪闭合角度 (弧度，约 180°)

# 运动参数
MOVE_SPEED = 0.3      # 速度系数 (0-1)
MOVE_TIMEOUT = 10.0   # 移动超时时间 (秒)


# ============== 状态回调函数 ==============

def on_state_update(state: ArmState):
    """状态更新回调（实时打印关键信息）"""
    print(f"  [状态] 位置: [{state.x:6.1f}, {state.y:6.1f}, {state.z:6.1f}]mm | "
          f"电压: {state.voltage:.1f}V | "
          f"夹爪: {'闭合' if state.is_gripper_closed else '张开'}")


# ============== 主程序 ==============

def main():
    print("=" * 60)
    print("🤖 RoArm-M3-S 机械臂控制示例")
    print("=" * 60)
    
    # 1️⃣ 连接机械臂
    print("\n[1/6] 连接机械臂...")
    arm = RoArmM3S(port=SERIAL_PORT, baudrate=BAUDRATE)
    
    if not arm.connect():
        print("❌ 连接失败，请检查串口配置")
        sys.exit(1)
    
    print("✅ 连接成功")
    
    # 注册状态回调（可选，用于实时监控）
    arm.register_state_callback(on_state_update)
    
    try:
        # 2️⃣ 等待初始状态稳定
        print("\n[2/6] 等待机械臂状态稳定...")
        time.sleep(1)
        
        # 主动请求一次反馈，确保获取到最新状态
        initial_state = arm.request_state_feedback(timeout=2.0)
        if not initial_state:
            print("⚠️  未能获取初始状态，继续执行...")
            initial_state = arm.current_state
        
        if initial_state:
            print(f"✅ 初始位置：X={initial_state.x:.1f}, Y={initial_state.y:.1f}, Z={initial_state.z:.1f} mm")
            print(f"   初始夹爪角度：{initial_state.gripper:.2f} rad ({initial_state.gripper*180/3.14:.1f}°)")
        else:
            print("⚠️  无法读取初始状态")
        
        # 记录初始位置（用于返回）
        init_x = initial_state.x if initial_state else 235
        init_y = initial_state.y if initial_state else 0
        init_z = initial_state.z if initial_state else 234
        init_pitch = initial_state.pitch if initial_state else 0
        init_roll = initial_state.roll if initial_state else 0
        init_gripper = initial_state.gripper if initial_state else 3.14
        
        time.sleep(1)
        
        # 3️⃣ 移动到目标位置
        print(f"\n[3/6] 移动到目标位置 (X={TARGET_X}, Y={TARGET_Y}, Z={TARGET_Z})...")
        success = arm.move_to_xyz(
            x=TARGET_X,
            y=TARGET_Y,
            z=TARGET_Z,
            pitch=TARGET_PITCH,
            roll=TARGET_ROLL,
            gripper=init_gripper,  # 保持当前夹爪状态
            spd=MOVE_SPEED,
            blocking=True
        )
        
        if success:
            print("✅ 到达目标位置")
            # 验证位置
            current_pos = arm.get_current_position(timeout=2.0)
            if current_pos:
                print(f"   实际位置：X={current_pos['x']:.1f}, Y={current_pos['y']:.1f}, Z={current_pos['z']:.1f} mm")
        else:
            print("❌ 移动失败或超时")
        
        time.sleep(1)
        
        # 4️⃣ 张开夹爪
        print(f"\n[4/6] 张开夹爪 (角度：{TARGET_GRIPPER_OPEN:.2f} rad)...")
        # 使用关节角度控制夹爪（h 参数对应夹爪）
        success = arm.move_joints_angle(
            h=TARGET_GRIPPER_OPEN * 180 / 3.14,  # 转换为角度制
            spd=0.5,
            blocking=True
        )
        
        if success:
            print("✅ 夹爪已张开")
            # 验证夹爪状态
            state = arm.request_state_feedback(timeout=2.0)
            if state:
                print(f"   夹爪角度：{state.gripper:.2f} rad ({state.gripper*180/3.14:.1f}°)")
        else:
            print("❌ 夹爪操作失败")
        
        time.sleep(1)
        
        # 5️⃣ 关闭夹爪
        print(f"\n[5/6] 关闭夹爪 (角度：{TARGET_GRIPPER_CLOSE:.2f} rad)...")
        success = arm.move_joints_angle(
            h=TARGET_GRIPPER_CLOSE * 180 / 3.14,  # 转换为角度制
            spd=0.5,
            blocking=True
        )
        
        if success:
            print("✅ 夹爪已闭合")
            state = arm.request_state_feedback(timeout=2.0)
            if state:
                print(f"   夹爪角度：{state.gripper:.2f} rad ({state.gripper*180/3.14:.1f}°)")
                print(f"   夹爪状态：{'闭合' if state.is_gripper_closed else '张开'}")
        else:
            print("❌ 夹爪操作失败")
        
        time.sleep(1)
        
        # 6️⃣ 返回初始位置
        print(f"\n[6/6] 返回初始位置 (X={init_x:.1f}, Y={init_y:.1f}, Z={init_z:.1f})...")
        success = arm.move_to_xyz(
            x=init_x,
            y=init_y,
            z=init_z,
            pitch=init_pitch,
            roll=init_roll,
            gripper=init_gripper,
            spd=MOVE_SPEED,
            blocking=True
        )
        
        if success:
            print("✅ 已返回初始位置")
            final_state = arm.request_state_feedback(timeout=2.0)
            if final_state:
                print(f"   最终位置：X={final_state.x:.1f}, Y={final_state.y:.1f}, Z={final_state.z:.1f} mm")
        else:
            print("❌ 返回失败或超时")
        
        print("\n" + "=" * 60)
        print("🎉 任务执行完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断，正在安全停止...")
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
    finally:
        # 断开连接
        print("\n[清理] 断开连接...")
        arm.disconnect()
        print("✅ 已安全断开")


# ============== 简化版示例（快速测试） ==============

def quick_test():
    """快速测试版本，适合验证基本功能"""
    print("🚀 快速测试模式")
    
    with RoArmM3S(port=SERIAL_PORT, baudrate=BAUDRATE) as arm:
        if not arm.current_state:
            time.sleep(1)
        
        # 移动到目标点
        print("移动到 (300, 0, 250)...")
        arm.move_to_xyz(300, 0, 250, blocking=True, spd=0.3)
        time.sleep(1)
        
        # 张开夹爪
        print("张开夹爪...")
        arm.move_gripper_angle(45)  # 45 度
        time.sleep(1)
        
        # 关闭夹爪
        print("关闭夹爪...")
        arm.move_gripper_angle(180)  # 180 度
        time.sleep(1)
        
        # 返回原点
        print("返回初始位置...")
        arm.move_to_xyz(235, 0, 234, blocking=True, spd=0.3)
        
        print("✅ 测试完成")


# ============== 入口 ==============

if __name__ == "__main__":
    # 运行完整示例
    main()
    
    # 或运行快速测试
    # quick_test()
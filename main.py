#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂笛卡尔坐标控制示例
重点演示 move_xyzt_goal 和 move_xyzt_direct 的区别
"""

import time
from robot_arm_lib import RobotArmController

# ================= 配置区 =================
PORT = "/dev/ttyUSB0"              # 修改为你的串口
BAUDRATE = 115200
SAFE_MODE = True           # 安全模式：限制速度
MAX_SPEED = 0.15           # 安全速度
# =========================================


def print_status(arm: RobotArmController):
    """打印当前状态"""
    pos = arm.get_position()
    if pos:
        print(f"  当前位置：X={pos['x']:.2f}, Y={pos['y']:.2f}, Z={pos['z']:.2f}")
    else:
        print("  位置：暂无数据")


def test_move_xyzt_goal(arm: RobotArmController):
    """
    测试 move_xyzt_goal（阻塞式）
    特点：机械臂内部阻塞，运动完成后才返回
    适用：单点精确运动
    """
    print("\n" + "="*50)
    print("测试 1: move_xyzt_goal (阻塞式)")
    print("="*50)
    
    # 获取当前位置
    start_pos = arm.get_position()
    if not start_pos:
        print("❌ 无法获取当前位置，跳过测试")
        return
    
    print(f"起始位置：X={start_pos['x']:.2f}, Y={start_pos['y']:.2f}, Z={start_pos['z']:.2f}")
    
    # 目标位置（Z 轴上升 20mm）
    target = {
        'x': start_pos['x'],
        'y': start_pos['y'],
        'z': start_pos['z'] + 20,
        't': 0,
        'r': 0,
        'g': 3.14
    }
    
    print(f"目标位置：X={target['x']:.2f}, Y={target['y']:.2f}, Z={target['z']:.2f}")
    print("🔄 开始运动（阻塞式）...")
    
    start_time = time.time()
    
    # 发送指令（阻塞式）
    arm.move_xyzt_goal(
        x=target['x'],
        y=target['y'],
        z=target['z'],
        t=target['t'],
        r=target['r'],
        g=target['g'],
        spd=MAX_SPEED
    )
    
    # 等待运动完成（根据距离调整）
    time.sleep(3)
    
    elapsed = time.time() - start_time
    print(f"✅ 运动完成，耗时：{elapsed:.2f} 秒")
    
    # 检查位置
    end_pos = arm.get_position()
    if end_pos:
        print(f"结束位置：X={end_pos['x']:.2f}, Y={end_pos['y']:.2f}, Z={end_pos['z']:.2f}")
    
    time.sleep(1)


def test_move_xyzt_direct(arm: RobotArmController):
    """
    测试 move_xyzt_direct（非阻塞式）
    特点：发送后立即返回，适合连续轨迹
    适用：多点连续运动、轨迹跟踪
    """
    print("\n" + "="*50)
    print("测试 2: move_xyzt_direct (非阻塞式)")
    print("="*50)
    
    start_pos = arm.get_position()
    if not start_pos:
        print("❌ 无法获取当前位置，跳过测试")
        return
    
    print(f"起始位置：X={start_pos['x']:.2f}, Y={start_pos['y']:.2f}, Z={start_pos['z']:.2f}")
    
    # 定义多个目标点（方形轨迹）
    waypoints = [
        {'x': start_pos['x'] + 10, 'y': start_pos['y'], 'z': start_pos['z']},
        {'x': start_pos['x'] + 10, 'y': start_pos['y'] + 10, 'z': start_pos['z']},
        {'x': start_pos['x'], 'y': start_pos['y'] + 10, 'z': start_pos['z']},
        {'x': start_pos['x'], 'y': start_pos['y'], 'z': start_pos['z']},
    ]
    
    print(f"📍 轨迹点数：{len(waypoints)}")
    print("🔄 开始连续运动（非阻塞式）...")
    
    start_time = time.time()
    
    # 连续发送多个目标点
    for i, point in enumerate(waypoints):
        print(f"  第 {i+1} 点：X={point['x']:.2f}, Y={point['y']:.2f}, Z={point['z']:.2f}")
        
        arm.move_xyzt_direct(
            x=point['x'],
            y=point['y'],
            z=point['z'],
            t=0,
            r=0,
            g=3.14
        )
        
        # 短暂等待，让机械臂有时间响应
        time.sleep(0.5)
        
        # 实时监控位置
        pos = arm.get_position()
        if pos:
            print(f"    → 实时位置：X={pos['x']:.2f}, Y={pos['y']:.2f}, Z={pos['z']:.2f}")
    
    elapsed = time.time() - start_time
    print(f"✅ 轨迹完成，总耗时：{elapsed:.2f} 秒")
    
    time.sleep(2)


def test_comparison(arm: RobotArmController):
    """
    对比两种运动方式
    """
    print("\n" + "="*50)
    print("测试 3: 两种方式对比")
    print("="*50)
    
    start_pos = arm.get_position()
    if not start_pos:
        return
    
    # 方式 1: goal（阻塞）
    print("\n【方式 1】move_xyzt_goal")
    t1 = time.time()
    arm.move_xyzt_goal(
        x=start_pos['x'], y=start_pos['y'], 
        z=start_pos['z'] + 10, t=0, r=0, g=3.14,
        spd=MAX_SPEED
    )
    time.sleep(2)
    t1_elapsed = time.time() - t1
    print(f"  耗时：{t1_elapsed:.2f} 秒（含等待）")
    
    # 方式 2: direct（非阻塞）
    print("\n【方式 2】move_xyzt_direct")
    t2 = time.time()
    arm.move_xyzt_direct(
        x=start_pos['x'], y=start_pos['y'], 
        z=start_pos['z'], t=0, r=0, g=3.14
    )
    t2_elapsed = time.time() - t2
    print(f"  耗时：{t2_elapsed:.2f} 秒（立即返回）")
    
    print(f"\n⏱️  时间差：{t1_elapsed - t2_elapsed:.2f} 秒")


def main():
    """主函数"""
    print("="*60)
    print("🤖 机械臂笛卡尔坐标控制示例")
    print("="*60)
    print(f"串口：{PORT}")
    print(f"波特率：{BAUDRATE}")
    print(f"安全速度：{MAX_SPEED}")
    print("="*60)
    
    # 创建控制器
    arm = RobotArmController(port=PORT, baudrate=BAUDRATE)
    
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
        
        # 检查状态
        if not arm.is_status_fresh(max_age=3.0):
            print("⚠️  警告：状态数据可能过期")
        else:
            print("✅ 状态数据正常")
        
        print_status(arm)
        
        # 运行测试
        test_move_xyzt_goal(arm)
        test_move_xyzt_direct(arm)
        test_comparison(arm)
        
        # 回到安全位置
        print("\n🏠 返回安全位置...")
        arm.move_xyzt_goal(
            x=235, y=0, z=234, t=0, r=0, g=3.14,
            spd=MAX_SPEED
        )
        time.sleep(3)
        
        print_status(arm)
        
        # 显示统计
        stats = arm.get_stats()
        print(f"\n📊 通信统计：")
        print(f"  接收字节：{stats['bytes_received']}")
        print(f"  接收消息：{stats['messages_received']}")
        print(f"  解析错误：{stats['parse_errors']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
    finally:
        # 断开连接
        print("\n🔌 断开连接...")
        arm.disconnect()
        print("✅ 完成")


if __name__ == "__main__":
    main()
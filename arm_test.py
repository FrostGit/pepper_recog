#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""状态监控示例"""

from roarm_m3 import RoArmM3S, ArmState
import time

def on_state_update(state: ArmState):
    """状态更新回调函数"""
    # 每秒最多打印1次，避免刷屏
    if time.time() - getattr(on_state_update, '_last_print', 0) > 1.0:
        on_state_update._last_print = time.time()
        print(f"\r📍 XYZ: [{state.x:6.1f}, {state.y:6.1f}, {state.z:6.1f}] mm  "
              f"🤚 夹爪: {'闭合' if state.is_gripper_closed else '张开'}  "
              f"⚡ 肘关节负载: {state.torque_elbow}", end='', flush=True)

def main():
    with RoArmM3S('/dev/ttyUSB0') as arm:
        # 注册实时状态回调
        arm.register_state_callback(on_state_update)
        
        print("🔄 开始监控机械臂状态 (Ctrl+C 停止)...")
        print("→ 执行复位")
        arm.move_to_init()
        time.sleep(3)
        
        # 主循环：持续读取最新状态
        try:
            while True:
                state = arm.current_state
                if state:
                    # 示例：根据状态做逻辑判断
                    if state.z < -108:  # Z轴过低
                        print(f"\n⚠️ 警告: Z轴({state.z})位置过低!")
                    
                    # 示例：记录轨迹
                    # log_trajectory(state)
                    
                time.sleep(0.05)  # 20Hz轮询
        except KeyboardInterrupt:
            print("\n\n✓ 监控停止")
        
        # 获取最终状态快照
        snapshot = arm.get_state_snapshot()
        if snapshot:
            print(f"\n📋 最终状态: {json.dumps(snapshot, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    main()
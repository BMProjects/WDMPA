#!/usr/bin/env python3
"""
热稳定性测试 (Jetson优化)
长时间运行测试，监控温度和性能下降
"""

import time
import argparse
from pathlib import Path

def monitor_thermal(model_path, duration=600, sample_interval=10):
    """
    监控热稳定性
    
    Args:
        model_path: ONNX模型路径
        duration: 测试时长(秒)
        sample_interval: 采样间隔(秒)
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return None
    
    print(f"测试模型: {model_path.name}")
    print(f"测试时长: {duration}秒 ({duration//60}分钟)")
    print(f"采样间隔: {sample_interval}秒")
    
    #加载模型
    session = ort.InferenceSession(
        str(model_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # 准备输入
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    batch_size = 1 if input_shape[0] in ['batch', None, -1] else input_shape[0]
    actual_shape = [batch_size] + list(input_shape[1:])
    dummy_input = np.random.randn(*actual_shape).astype(np.float32)
    
    # Warmup
    print("Warmup...")
    for _ in range(50):
        _ = session.run(None, {input_name: dummy_input})
    
    # 开始监控
    print("\n开始热稳定性测试...")
    print(f"{'时间(s)':>8} {'FPS':>8} {'温度(°C)':>10} {'状态':>10}")
    print("-" * 45)
    
    start_time = time.time()
    elapsed = 0
    samples = []
    
    while elapsed < duration:
        # 测试FPS
        iterations = 100
        iter_start = time.perf_counter()
        
        for _ in range(iterations):
            _ = session.run(None, {input_name: dummy_input})
        
        iter_duration = time.perf_counter() - iter_start
        fps = iterations / iter_duration
        
        # 读取温度 (Jetson Nano)
        temp = read_temperature()
        
        # 记录样本
        elapsed = time.time() - start_time
        samples.append({
            'time': elapsed,
            'fps': fps,
            'temperature': temp
        })
        
        # 显示进度
        status = "正常" if fps > samples[0]['fps'] * 0.9 else "降频"
        print(f"{elapsed:>8.1f} {fps:>8.1f} {temp:>10.1f} {status:>10}")
        
        # 等待下个采样点
        time.sleep(max(0, sample_interval - (time.time() - start_time - elapsed)))
        elapsed = time.time() - start_time
    
    # 分析结果
    initial_fps = samples[0]['fps']
    final_fps = samples[-1]['fps']
    avg_fps = np.mean([s['fps'] for s in samples])
    fps_drop = (initial_fps - final_fps) / initial_fps * 100
    
    max_temp = max([s['temperature'] for s in samples])
    
    print("\n" + "=" * 45)
    print("测试完成!")
    print("=" * 45)
    print(f"初始FPS: {initial_fps:.1f}")
    print(f"最终FPS: {final_fps:.1f}")
    print(f"平均FPS: {avg_fps:.1f}")
    print(f"性能下降: {fps_drop:.1f}%")
    print(f"最高温度: {max_temp:.1f}°C")
    
    return samples

def read_temperature():
    """读取Jetson Nano温度"""
    try:
        # Jetson Nano温度文件
        with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000.0
            return temp
    except:
        # 如果读取失败，返回默认值
        return 0.0

def plot_results(samples, output_path):
    """绘制温度和FPS曲线"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无GUI后端
        import matplotlib.pyplot as plt
        
        times = [s['time'] for s in samples]
        fps_values = [s['fps'] for s in samples]
        temps = [s['temperature'] for s in samples]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # FPS曲线
        ax1.plot(times, fps_values, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('FPS', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Thermal Stability Test - FPS over Time')
        
        # 温度曲线
        ax2.plot(times, temps, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (°C)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Temperature over Time')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\n✅ 曲线图已保存: {output_path}")
        
    except ImportError:
        print("\n⚠️  matplotlib未安装，跳过绘图")

def main():
    parser = argparse.ArgumentParser(description='热稳定性测试')
    parser.add_argument('--model', required=True, help='ONNX模型文件')
    parser.add_argument('--duration', type=int, default=600, help='测试时长(秒)')
    parser.add_argument('--output', required=True, help='输出图片路径')
    args = parser.parse_args()
    
    print("=" * 60)
    print("热稳定性测试 (Jetson Nano)")
    print("=" * 60)
    print()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 运行测试
    samples = monitor_thermal(model_path, args.duration)
    
    if samples:
        # 绘制结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_results(samples, output_path)

if __name__ == '__main__':
    main()

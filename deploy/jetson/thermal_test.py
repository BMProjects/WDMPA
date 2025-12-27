#!/usr/bin/env python3
"""Thermal stability test for Jetson Nano (Python 3.6 compatible).

Runs continuous inference for a specified duration and monitors FPS stability.

Usage:
    python thermal_test.py --model wdmpa_fold0.onnx --duration 600
"""

from __future__ import print_function
import argparse
import time
import os
import sys

import numpy as np


def get_temperature():
    """Get Jetson temperature from thermal zones."""
    try:
        temps = []
        for i in range(10):
            path = '/sys/devices/virtual/thermal/thermal_zone{}/temp'.format(i)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    temps.append(int(f.read().strip()) / 1000.0)
        return max(temps) if temps else -1
    except Exception:
        return -1


def run_thermal_test(model_path, duration_seconds, interval=1.0):
    """Run thermal stability test."""
    import onnxruntime as ort
    
    print("Loading model: {}".format(model_path))
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    print("Warmup (30 seconds)...")
    warmup_end = time.time() + 30
    while time.time() < warmup_end:
        session.run(None, {input_name: dummy_input})
    
    # Main test
    print("\nStarting thermal test ({} seconds)...".format(duration_seconds))
    print("-" * 60)
    print("{:>8} {:>10} {:>10} {:>10}".format("Time(s)", "FPS", "Temp(C)", "Status"))
    print("-" * 60)
    
    start_time = time.time()
    results = []
    last_report = start_time
    frame_count = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration_seconds:
            break
        
        # Run inference
        session.run(None, {input_name: dummy_input})
        frame_count += 1
        
        # Report every interval
        if time.time() - last_report >= interval:
            fps = frame_count / (time.time() - last_report)
            temp = get_temperature()
            status = "OK" if temp < 70 else "HOT" if temp < 80 else "THROTTLE"
            
            print("{:>8.1f} {:>10.1f} {:>10.1f} {:>10}".format(elapsed, fps, temp, status))
            
            results.append({
                'time': elapsed,
                'fps': fps,
                'temp': temp,
            })
            
            frame_count = 0
            last_report = time.time()
    
    return results


def save_results(results, output_dir):
    """Save results to CSV and generate plot."""
    import csv
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'thermal_results.csv')
    with open(csv_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'fps', 'temp'])
        writer.writeheader()
        writer.writerows(results)
    print("\nResults saved to: {}".format(csv_path))
    
    # Calculate summary
    fps_values = [r['fps'] for r in results]
    temp_values = [r['temp'] for r in results if r['temp'] > 0]
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("  Initial FPS:  {:.1f}".format(fps_values[0] if fps_values else 0))
    print("  Final FPS:    {:.1f}".format(fps_values[-1] if fps_values else 0))
    print("  FPS Drop:     {:.1f}%".format((1 - fps_values[-1] / fps_values[0]) * 100 if fps_values else 0))
    print("  Max Temp:     {:.1f} C".format(max(temp_values) if temp_values else 0))
    
    # Try to generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        times = [r['time'] for r in results]
        
        ax1.plot(times, fps_values, 'b-', linewidth=2)
        ax1.set_ylabel('FPS')
        ax1.set_title('Thermal Stability Test')
        ax1.grid(True)
        
        ax2.plot(times, temp_values, 'r-', linewidth=2)
        ax2.set_ylabel('Temperature (C)')
        ax2.set_xlabel('Time (s)')
        ax2.axhline(y=70, color='orange', linestyle='--', label='Warning')
        ax2.axhline(y=80, color='red', linestyle='--', label='Throttle')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'thermal_curve.png')
        plt.savefig(plot_path, dpi=150)
        print("  Plot saved:   {}".format(plot_path))
    except ImportError:
        print("  (matplotlib not available, skipping plot)")


def main():
    parser = argparse.ArgumentParser(description='Jetson Thermal Test')
    parser.add_argument('--model', type=str, required=True, help='ONNX model path')
    parser.add_argument('--duration', type=int, default=600, help='Test duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0, help='Report interval')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    results = run_thermal_test(args.model, args.duration, args.interval)
    save_results(results, args.output)


if __name__ == '__main__':
    main()

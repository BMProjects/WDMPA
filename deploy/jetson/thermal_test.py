"""Thermal Stability Test - 10-minute continuous inference with FPS logging.

Tests model performance under sustained load to detect thermal throttling.

Usage:
    python deploy/thermal_stability_test.py --model wdmpa_fold0.onnx --duration 600
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def run_thermal_test(
    model_path: str,
    duration_seconds: int = 600,
    log_interval: float = 1.0,
    input_size: tuple = (224, 224),
    output_dir: str = "deploy/thermal_results",
) -> dict:
    """Run thermal stability test.

    Args:
        model_path: Path to ONNX model.
        duration_seconds: Test duration in seconds (default: 600 = 10 min).
        log_interval: Interval between FPS logs in seconds.
        input_size: Input image size.
        output_dir: Directory for output files.

    Returns:
        Dict with test results.
    """
    import onnxruntime as ort

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        provider = ["CUDAExecutionProvider"]
    else:
        provider = ["CPUExecutionProvider"]

    print(f"Loading model: {model_path}")
    print(f"Provider: {provider}")
    session = ort.InferenceSession(model_path, providers=provider)

    dummy_input = np.random.randn(1, 3, *input_size).astype(np.float32)

    # Warmup
    print("Warming up (30 iterations)...")
    for _ in range(30):
        session.run(None, {"input": dummy_input})

    # Test loop
    print(f"Starting {duration_seconds}s thermal stability test...")
    print(f"Logging every {log_interval}s")

    start_time = time.time()
    logs = []
    frame_count = 0
    interval_start = start_time
    interval_frames = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                break

            # Inference
            session.run(None, {"input": dummy_input})
            frame_count += 1
            interval_frames += 1

            # Log FPS periodically
            interval_elapsed = time.time() - interval_start
            if interval_elapsed >= log_interval:
                fps = interval_frames / interval_elapsed
                logs.append({
                    "time_s": round(elapsed, 1),
                    "fps": round(fps, 1),
                })

                # Print progress
                progress = elapsed / duration_seconds * 100
                print(f"[{progress:5.1f}%] Time: {elapsed:6.1f}s | FPS: {fps:5.1f}")

                interval_start = time.time()
                interval_frames = 0

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    # Final stats
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    fps_values = [log["fps"] for log in logs]

    results = {
        "model": Path(model_path).name,
        "duration_s": round(total_time, 1),
        "total_frames": frame_count,
        "avg_fps": round(avg_fps, 1),
        "min_fps": round(min(fps_values), 1) if fps_values else 0,
        "max_fps": round(max(fps_values), 1) if fps_values else 0,
        "fps_drop_percent": round((1 - min(fps_values) / max(fps_values)) * 100, 1) if fps_values else 0,
        "logs": logs,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    model_name = Path(model_path).stem
    json_path = output_path / f"thermal_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate plot
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        times = [log["time_s"] for log in logs]
        fps_vals = [log["fps"] for log in logs]

        plt.plot(times, fps_vals, "b-", linewidth=1.5, label="FPS")
        plt.axhline(y=avg_fps, color="r", linestyle="--", label=f"Avg: {avg_fps:.1f}")
        plt.axhline(y=30, color="g", linestyle=":", label="Real-time (30 FPS)")

        plt.xlabel("Time (seconds)")
        plt.ylabel("FPS")
        plt.title(f"Thermal Stability Test: {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = output_path / f"thermal_{model_name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved: {plot_path}")

    except ImportError:
        print("matplotlib not available, skipping plot generation")

    # Summary
    print("\n" + "=" * 50)
    print("Thermal Stability Test Results")
    print("=" * 50)
    print(f"Model:        {results['model']}")
    print(f"Duration:     {results['duration_s']}s")
    print(f"Total Frames: {results['total_frames']}")
    print(f"Average FPS:  {results['avg_fps']}")
    print(f"Min FPS:      {results['min_fps']}")
    print(f"Max FPS:      {results['max_fps']}")
    print(f"FPS Drop:     {results['fps_drop_percent']}%")
    print("=" * 50)

    if results["fps_drop_percent"] < 10:
        print("✓ Excellent thermal stability (< 10% drop)")
    elif results["fps_drop_percent"] < 20:
        print("○ Good thermal stability (< 20% drop)")
    else:
        print("✗ Significant thermal throttling detected")

    print(f"\nResults saved: {json_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Thermal Stability Test")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Test duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Log interval in seconds",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input size (H W)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deploy/thermal_results",
        help="Output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_thermal_test(
        model_path=args.model,
        duration_seconds=args.duration,
        log_interval=args.interval,
        input_size=tuple(args.input_size),
        output_dir=args.output,
    )

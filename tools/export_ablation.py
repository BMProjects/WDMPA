"""Export ablation model variants to ONNX for Jetson comparison.

Usage:
    python tools/export_ablation.py --output-dir deploy/onnx/ablation
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wdmpa.models.ablation import create_ablation_variants


def export_model(model, name, output_dir, input_size=(224, 224), opset=11):
    """Export a single model to ONNX."""
    model.eval()
    dummy = torch.randn(1, 3, *input_size)
    output_path = Path(output_dir) / f"{name}.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return size_mb, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="deploy/onnx/ablation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exporting ablation variants to ONNX")
    print("=" * 60)

    variants = create_ablation_variants()

    results = []
    for i, (name, model) in enumerate(variants.items(), 1):
        print(f"\n[{i}/{len(variants)}] {name}")
        size_mb, params = export_model(model, name, output_dir)
        print(f"  → {size_mb:.1f} MB, {params:.2f}M params")
        results.append((name, params, size_mb))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Variant':<25} {'Params (M)':<12} {'Size (MB)':<10}")
    print("-" * 50)
    for name, params, size in results:
        print(f"{name:<25} {params:<12.2f} {size:<10.1f}")


if __name__ == "__main__":
    main()

"""Batch export all models to ONNX for Jetson comparison.

Usage:
    python tools/export_all.py --weights-dir weights --output-dir deploy/onnx
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wdmpa import WDMPANet
from wdmpa.models.baselines import MobileNetV3Gaze, ShuffleNetV2Gaze, L2CSNetSimple


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
    print(f"  {name}: {size_mb:.1f} MB, {params:.2f}M params")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=str, default="weights",
                        help="Directory containing WDMPA weights")
    parser.add_argument("--output-dir", type=str, default="deploy/onnx")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Exporting all models to ONNX")
    print("=" * 50)

    # 1. WDMPA-Net
    print("\n[1/4] WDMPA-Net")
    wdmpa = WDMPANet()
    weights_path = Path(args.weights_dir) / f"fold{args.fold}_best.pkl"
    if weights_path.exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        wdmpa.load_state_dict(state, strict=False)
        print(f"  Loaded weights: {weights_path}")
    export_model(wdmpa, f"wdmpa_fold{args.fold}", output_dir)

    # 2. MobileNetV3-Small
    print("\n[2/4] MobileNetV3-Small")
    mobilenet = MobileNetV3Gaze(pretrained=True)
    export_model(mobilenet, "mobilenetv3_small", output_dir)

    # 3. ShuffleNetV2
    print("\n[3/4] ShuffleNetV2")
    shufflenet = ShuffleNetV2Gaze(pretrained=True)
    export_model(shufflenet, "shufflenetv2", output_dir)

    # 4. L2CS-Net (ResNet50 backbone, official pretrained)
    print("\n[4/4] L2CS-Net (ResNet50 backbone)")
    l2cs = L2CSNetSimple(pretrained=True)
    export_model(l2cs, "l2cs_net", output_dir)

    print("\n" + "=" * 50)
    print(f"All models exported to: {output_dir}")
    print("=" * 50)

    # List files
    print("\nExported files:")
    for f in sorted(output_dir.glob("*.onnx")):
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()

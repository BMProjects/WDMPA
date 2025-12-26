"""Export WDMPA-Net to ONNX format.

Usage:
    python tools/export.py --weights weights/fold0_best.pkl --output wdmpa.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wdmpa import WDMPANet


def parse_args():
    parser = argparse.ArgumentParser(description="Export WDMPA-Net to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    parser.add_argument("--output", type=str, default="wdmpa.onnx", help="Output path")
    parser.add_argument("--input-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    return parser.parse_args()


def export(args):
    # Load model
    model = WDMPANet()
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    # Export
    dummy = torch.randn(1, 3, *args.input_size)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Validate
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    # Simplify
    if args.simplify:
        try:
            import onnxsim
            onnx_model, _ = onnxsim.simplify(onnx_model)
            onnx.save(onnx_model, args.output)
            print("Simplified model")
        except ImportError:
            print("onnx-simplifier not installed, skipping")

    size = Path(args.output).stat().st_size / 1024 / 1024
    print(f"Exported: {args.output} ({size:.1f} MB)")


if __name__ == "__main__":
    export(parse_args())

"""ONNX Export Script for WDMPA-Net.

Exports PyTorch model to ONNX format for TensorRT deployment on Jetson Nano.

Usage:
    python deploy/export_onnx.py --weights Pre-trained\ weights/fold0_best.pkl --output wdmpa.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import onnx

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Emodels.Emodel_AStarNet_WFD_SESA import Emodel_StarNet


def export_onnx(
    weights: str,
    output: str,
    input_size: tuple[int, int] = (224, 224),
    batch_size: int = 1,
    opset: int = 11,
    dynamic_batch: bool = False,
    simplify: bool = True,
    verify: bool = True,
) -> Path:
    """Export WDMPA-Net to ONNX format.

    Args:
        weights: Path to .pkl weights file.
        output: Output ONNX file path.
        input_size: Input image size (H, W).
        batch_size: Batch size for export.
        opset: ONNX opset version (11 recommended for TensorRT 8.x).
        dynamic_batch: Enable dynamic batch size.
        simplify: Simplify ONNX graph using onnx-simplifier.
        verify: Verify ONNX output matches PyTorch.

    Returns:
        Path to exported ONNX file.
    """
    print(f"Loading model from {weights}...")
    device = torch.device("cpu")

    # Load model
    model = Emodel_StarNet()
    state_dict = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(batch_size, 3, *input_size)

    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Export
    output_path = Path(output)
    print(f"Exporting to {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Check model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model validated: {output_path}")

    # Simplify (optional)
    if simplify:
        try:
            import onnxsim

            print("Simplifying ONNX model...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, str(output_path))
                print("Simplified model saved.")
            else:
                print("Simplification check failed, using original model.")
        except ImportError:
            print("onnx-simplifier not installed, skipping simplification.")

    # Verify output
    if verify:
        print("Verifying ONNX output...")
        import onnxruntime as ort

        ort_session = ort.InferenceSession(str(output_path))
        ort_inputs = {"input": dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        with torch.no_grad():
            torch_output = model(dummy_input).numpy()

        diff = np.abs(torch_output - ort_output).max()
        print(f"Max output difference: {diff:.6f}")
        if diff < 1e-4:
            print("✓ Verification passed!")
        else:
            print("⚠ Output difference is large, please check.")

    # Print model info
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nExport complete!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Opset: {opset}")
    print(f"  Input shape: (batch, 3, {input_size[0]}, {input_size[1]})")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export WDMPA-Net to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="Pre-trained weights/fold0_best.pkl",
        help="Path to weights file (.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deploy/wdmpa_fold0.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size (H W)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch size",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip output verification",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        weights=args.weights,
        output=args.output,
        input_size=tuple(args.input_size),
        batch_size=args.batch_size,
        opset=args.opset,
        dynamic_batch=args.dynamic,
        simplify=not args.no_simplify,
        verify=not args.no_verify,
    )

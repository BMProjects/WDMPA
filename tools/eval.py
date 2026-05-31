"""Evaluation script for WDMPA-Net.

Usage:
    python tools/eval.py --weights weights/fold0_best.pkl --data-root datasets/MPIIFaceGaze --fold 0
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wdmpa import WDMPANet  # noqa: E402
from wdmpa.data import Gaze360Dataset, MPIIGazeDataset, create_dataloader  # noqa: E402
from wdmpa.models.baselines import L2CSNet, MobileNetV3Gaze, ShuffleNetV2Gaze  # noqa: E402
from wdmpa.utils import angular_error  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gaze estimation models")
    parser.add_argument(
        "--model",
        type=str,
        default="wdmpa",
        choices=["wdmpa", "l2cs", "mobilenetv3", "shufflenetv2"],
        help="Model to evaluate",
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root")
    parser.add_argument("--dataset", type=str, choices=["mpiigaze", "gaze360"], default="mpiigaze")
    parser.add_argument("--fold", type=int, default=0, help="Fold for MPIIGaze")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_bins: int = 90,
) -> float:
    """Evaluate model and return angular error."""
    model.eval()
    all_errors = []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        # Handle L2CS-Net output (returns tuple of yaw, pitch logits)
        if isinstance(pred, tuple):
            # L2CS returns (yaw_logits, pitch_logits), convert to angles
            yaw_logits, pitch_logits = pred

            # Use softmax to get probabilities
            yaw_prob = torch.softmax(yaw_logits, dim=1)
            pitch_prob = torch.softmax(pitch_logits, dim=1)

            # Soft-argmax to get continuous bin index
            idx_tensor = torch.arange(num_bins, device=device, dtype=torch.float32)
            yaw_idx = torch.sum(yaw_prob * idx_tensor, dim=1)
            pitch_idx = torch.sum(pitch_prob * idx_tensor, dim=1)

            # Convert bin index to angle in degrees
            # For MPIIGaze: 28 bins cover +/-42 degrees (84 total).
            # For Gaze360: 90 bins cover +/-90 degrees (180 total).
            angle_range = 42.0 if num_bins == 28 else 90.0

            yaw_deg = (yaw_idx - num_bins / 2) * (2 * angle_range / num_bins)
            pitch_deg = (pitch_idx - num_bins / 2) * (2 * angle_range / num_bins)

            pred = torch.stack([pitch_deg, yaw_deg], dim=1)

        errors = angular_error(pred, labels)
        all_errors.extend(errors.cpu().tolist())

    return sum(all_errors) / len(all_errors)


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset: {args.dataset}")
    print(f"  Fold: {args.fold if args.dataset == 'mpiigaze' else 'N/A'}")
    print("  Split: test")

    # Load model
    print(f"\nLoading model: {args.model}")

    # For L2CS, we need to detect num_bins from weights first
    if args.model == "l2cs":
        # Load weights to check num_bins
        temp_state = torch.load(args.weights, map_location="cpu", weights_only=False)

        # Detect num_bins from fc_yaw_gaze layer
        yaw_key = None
        for k in temp_state.keys():
            if "fc_yaw_gaze.weight" in k:
                yaw_key = k
                break

        if yaw_key:
            num_bins = temp_state[yaw_key].shape[0]
            print(f"  Detected num_bins: {num_bins}")
            model = L2CSNet(num_bins=num_bins)
        else:
            print("  Using default num_bins: 90")
            model = L2CSNet(num_bins=90)
    elif args.model == "wdmpa":
        model = WDMPANet()
    elif args.model == "mobilenetv3":
        model = MobileNetV3Gaze(pretrained=False)
    elif args.model == "shufflenetv2":
        model = ShuffleNetV2Gaze(pretrained=False)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    state = torch.load(args.weights, map_location="cpu", weights_only=False)

    # Handle different state_dict formats
    if args.model == "l2cs":
        # Official L2CS weights use 'module.' prefix (from DataParallel)
        # Our L2CSNet wrapper uses 'model.' prefix
        has_module_prefix = any(k.startswith("module.") for k in state.keys())

        if has_module_prefix:
            # Convert 'module.' to 'model.' for L2CSNet wrapper
            print("  Converting official L2CS weights format (module. -> model.)...")
            new_state = {}
            for k, v in state.items():
                # Remove 'module.' and add 'model.'
                new_key = k.replace("module.", "model.")
                new_state[new_key] = v
            state = new_state

    # Load state dict
    if args.model == "l2cs":
        # Use strict=False for L2CS to ignore fc_finetune layer
        _missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if unexpected_keys:
            print(f"  Ignored unexpected keys: {len(unexpected_keys)} (e.g., fc_finetune)")
    else:
        model.load_state_dict(state)

    model = model.to(device)
    print(f"  Weights: {args.weights}")

    # Dataset
    if args.dataset == "mpiigaze":
        dataset = MPIIGazeDataset(args.data_root, fold=args.fold, train=False)
    else:
        dataset = Gaze360Dataset(args.data_root, label_file="Label/test.label")

    print(f"  Samples: {len(dataset)}")

    dataloader = create_dataloader(dataset, args.batch_size, shuffle=False, num_workers=args.workers)

    # Evaluate
    print("\nEvaluating...")

    # Get num_bins for L2CS
    num_bins = 90  # default
    if args.model == "l2cs" and hasattr(model, "num_bins"):
        num_bins = model.num_bins

    error = evaluate(model, dataloader, device, num_bins=num_bins)
    print("\nResults:")
    print(f"  Angular Error: {error:.2f} deg")
    print(f"  Samples: {len(dataset)}")


if __name__ == "__main__":
    main()

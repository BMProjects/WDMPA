"""Evaluation script for WDMPA-Net.

Usage:
    python tools/eval.py --weights weights/fold0_best.pkl --data-root datasets/MPIIFaceGaze --fold 0
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wdmpa import WDMPANet
from wdmpa.data import MPIIGazeDataset, Gaze360Dataset, create_dataloader
from wdmpa.utils import angular_error


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WDMPA-Net")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root")
    parser.add_argument("--dataset", type=str, choices=["mpiigaze", "gaze360"],
                        default="mpiigaze")
    parser.add_argument("--fold", type=int, default=0, help="Fold for MPIIGaze")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model and return angular error."""
    model.eval()
    all_errors = []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        errors = angular_error(pred, labels)
        all_errors.extend(errors.cpu().tolist())

    return sum(all_errors) / len(all_errors)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = WDMPANet()
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(device)

    # Dataset
    if args.dataset == "mpiigaze":
        dataset = MPIIGazeDataset(args.data_root, fold=args.fold, train=False)
    else:
        dataset = Gaze360Dataset(args.data_root, label_file="Label/test.label")

    dataloader = create_dataloader(dataset, args.batch_size, shuffle=False)

    # Evaluate
    error = evaluate(model, dataloader, device)
    print(f"\nAngular Error: {error:.2f}°")


if __name__ == "__main__":
    main()

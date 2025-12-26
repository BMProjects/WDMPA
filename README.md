# WDMPA-Net

**W**avelet **D**ownsampling and **M**ulti-scale **P**erceptual **A**ttention Network for Efficient Gaze Estimation.

## Features

- 🚀 **Lightweight**: 2.58M parameters, 0.39G FLOPs
- 🎯 **Accurate**: State-of-the-art performance on MPIIGaze and Gaze360
- ⚡ **Fast**: Real-time inference on edge devices (Jetson Nano)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from wdmpa import WDMPANet

model = WDMPANet()
x = torch.randn(1, 3, 224, 224)
gaze = model(x)  # (batch, 2) - pitch, yaw
```

## Project Structure

```
wdmpa/
├── wdmpa/              # Core package
│   ├── models/         # Network architectures
│   ├── modules/        # AWWD, MPA, StarBlock
│   ├── data/           # Dataset utilities
│   └── utils/          # Helper functions
├── tools/              # Training & export scripts
├── deploy/             # Deployment utilities
│   └── jetson/         # Jetson Nano scripts
├── configs/            # Configuration files
└── docs/               # Documentation
```

## Key Components

| Module | Description |
|--------|-------------|
| **AWWD** | Adaptive Weighted Wavelet Downsampling |
| **MPA** | Multi-scale Perceptual Attention |
| **StarNet** | Efficient backbone with element-wise multiplication |

## Documentation

- [Usage Guide](docs/USAGE.md)
- [Jetson Experiments](docs/JETSON_EXPERIMENT.md)
- [Training Guide](docs/TRAINING.md)

## Citation

```bibtex
@article{wdmpa2025,
  title={WDMPA-Net: Efficient Gaze Estimation with Wavelet Downsampling and Multi-scale Perceptual Attention},
  author={...},
  journal={Displays},
  year={2025}
}
```

## License

MIT License

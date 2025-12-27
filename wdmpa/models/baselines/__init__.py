"""Baseline models for gaze estimation comparison."""

from wdmpa.models.baselines.mobilenet_gaze import MobileNetV3Gaze
from wdmpa.models.baselines.shufflenet_gaze import ShuffleNetV2Gaze
from wdmpa.models.baselines.resnet_gaze import ResNet50Gaze
from wdmpa.models.baselines.l2cs_net import L2CS, L2CSNet, L2CSNetSimple, L2CS_ResNet50

__all__ = [
    "MobileNetV3Gaze",
    "ShuffleNetV2Gaze",
    "ResNet50Gaze",
    "L2CS",
    "L2CSNet",
    "L2CSNetSimple",
    "L2CS_ResNet50",
]

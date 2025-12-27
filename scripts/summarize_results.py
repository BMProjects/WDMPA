#!/usr/bin/env python3
"""汇总实验结果并生成报告."""

import os
import sys
from pathlib import Path

import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from wdmpa import WDMPANet
from wdmpa.models.baselines import MobileNetV3Gaze, ShuffleNetV2Gaze
from wdmpa.models.ablation import WDMPANetAblation


def get_model_params(model):
    """获取模型参数量."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    results = []
    
    # 1. WDMPA-Net (预训练权重)
    print("评估 WDMPA-Net (预训练权重)...")
    for fold in range(15):
        weights_path = PROJECT_ROOT / f"weights/fold{fold}_best.pkl"
        if weights_path.exists():
            results.append({
                "model": "WDMPA-Net",
                "fold": fold,
                "params_M": 2.58,
                "weights": str(weights_path),
                "source": "pretrained"
            })
    
    # 2. 基线模型
    print("检查基线模型...")
    baselines = {
        "MobileNetV3": ("mobilenetv3", MobileNetV3Gaze),
        "ShuffleNetV2": ("shufflenetv2", ShuffleNetV2Gaze),
    }
    
    for name, (folder, model_cls) in baselines.items():
        model = model_cls()
        params = get_model_params(model)
        
        for fold in range(15):
            weights_dir = PROJECT_ROOT / f"runs/baselines/{folder}/fold{fold}"
            best_pkl = list(weights_dir.rglob("best.pkl")) if weights_dir.exists() else []
            
            if best_pkl:
                results.append({
                    "model": name,
                    "fold": fold,
                    "params_M": params,
                    "weights": str(best_pkl[0]),
                    "source": "trained"
                })
    
    # 3. 消融变体
    print("检查消融变体...")
    ablations = [
        ("AWWD-Fixed", "wdmpa_awwd_fixed"),
        ("Stride-Conv", "wdmpa_stride_conv"),
        ("Channel-Only", "wdmpa_channel_only"),
        ("Spatial-Only", "wdmpa_spatial_only"),
        ("Single-Scale", "wdmpa_single_scale"),
        ("No-Attention", "wdmpa_no_attention"),
    ]
    
    for name, folder in ablations:
        weights_dir = PROJECT_ROOT / f"runs/ablation/{folder}"
        best_pkl = list(weights_dir.rglob("best.pkl")) if weights_dir.exists() else []
        
        if best_pkl:
            results.append({
                "model": f"WDMPA-{name}",
                "fold": 0,
                "params_M": 2.58,
                "weights": str(best_pkl[0]),
                "source": "trained"
            })
    
    # 生成报告
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"\n总共找到 {len(results)} 个模型权重\n")
    
    # 按模型分组统计
    summary = df.groupby("model").agg({
        "fold": "count",
        "params_M": "first",
        "source": "first"
    }).rename(columns={"fold": "folds"})
    
    print(summary.to_string())
    
    # 保存结果
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "model_inventory.csv", index=False)
    print(f"\n结果已保存到: {output_dir / 'model_inventory.csv'}")
    
    # 检查实验结果 CSV
    results_csv = output_dir / "experiment_results.csv"
    if results_csv.exists():
        print(f"\n训练结果 (来自 {results_csv}):")
        exp_df = pd.read_csv(results_csv)
        print(exp_df.to_string(index=False))


if __name__ == "__main__":
    main()

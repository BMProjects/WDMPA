#!/bin/bash
# 准备测试数据集 (PC端)
# 从MPIIFaceGaze提取100张测试图片

set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
DATA_ROOT="/home/bm/Data/MPIIFaceGaze_Processed"
OUTPUT_DIR="$PROJECT_ROOT/deploy/test_data"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR/images"

echo "==================================="
echo "准备Jetson测试数据集"
echo "==================================="

# 从fold 0 (p00) 随机选择100张图片
python3 << EOF
import random
import shutil
from pathlib import Path

data_root = Path('$DATA_ROOT')
output = Path('$OUTPUT_DIR')

# 获取p00的所有图片
images = list((data_root / 'Image' / 'p00').glob('*.jpg'))
print(f"Found {len(images)} images in p00")

# 随机采样100张
samples = random.sample(images, min(100, len(images)))
print(f"Selecting {len(samples)} images for testing")

for i, img in enumerate(samples, 1):
    shutil.copy(img, output / 'images' / f'test_{i:03d}.jpg')
    if i % 10 == 0:
        print(f"  Copied {i}/{len(samples)}...")

print("✅ Images copied")
EOF

echo ""
echo "测试数据集已准备完成:"
echo "  图片数量: $(ls $OUTPUT_DIR/images/*.jpg | wc -l)"
echo "  total耗: $(du -sh $OUTPUT_DIR | cut -f1)"
echo "  位置: $OUTPUT_DIR"

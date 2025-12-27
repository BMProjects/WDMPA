#!/bin/bash
# 打包Jetson部署文件 (PC端)

set -e

PROJECT_ROOT="/home/bm/Dev/WDMPA"
PACKAGE_NAME="jetson_deploy"
PACKAGE_DIR="$PROJECT_ROOT/$PACKAGE_NAME"

cd "$PROJECT_ROOT"

echo "==================================="
echo "打包Jetson部署文件"
echo "==================================="

# 清理旧包
rm -rf "$PACKAGE_DIR" jetson_deploy.tar.gz

# 创建目录结构
mkdir -p "$PACKAGE_DIR"/{onnx,test_data,scripts}

# 1. 复制ONNX模型
echo "[1/5] 复制ONNX模型..."
cp deploy/onnx/*.onnx "$PACKAGE_DIR/onnx/" 2>/dev/null || {
    echo "⚠️  警告: 未找到ONNX模型，请先运行 ./deploy/scripts/export_all_models.sh"
}

# 2. 复制测试数据
echo "[2/5] 复制测试数据..."
cp -r deploy/test_data/images "$PACKAGE_DIR/test_data/" 2>/dev/null || {
    echo "⚠️  警告: 未找到测试数据，请先运行 ./deploy/scripts/prepare_test_dataset.sh"
}

# 3. 复制Jetson脚本
echo "[3/5] 复制部署脚本..."
cp deploy/jetson/*.py "$PACKAGE_DIR/scripts/"
cp deploy/jetson/*.sh "$PACKAGE_DIR/scripts/" 2>/dev/null || true

# 4. 创建requirements.txt
echo "[4/5] 创建依赖文件..."
cat > "$PACKAGE_DIR/requirements.txt" << 'EOF'
numpy>=1.19.0
onnxruntime-gpu>=1.8.0
Pillow>=8.0.0
matplotlib>=3.3.0
psutil>=5.8.0
EOF

# 5. 创建setup.sh
echo "[5/5] 创建安装脚本..."
cat > "$PACKAGE_DIR/setup.sh" << 'EOF'
#!/bin/bash
# Jetson Nano环境配置脚本

echo "=== Jetson Nano 环境配置 ==="

# 检查Python
python3 --version || { echo "❌ Python3未安装"; exit 1; }

# 检查CUDA
nvcc --version || echo "⚠️  CUDA未检测到，将使用CPU模式"

# 安装依赖
echo "安装Python依赖..."
pip3 install --user -r requirements.txt

# 创建结果目录
mkdir -p results

echo "✅ 环境配置完成"
echo ""
echo "运行测试:"
echo "  cd scripts"
echo "  ./run_all_benchmarks.sh"
EOF

chmod +x "$PACKAGE_DIR/setup.sh"
chmod +x "$PACKAGE_DIR/scripts"/*.sh 2>/dev/null || true

# 打包
echo ""
echo "创建压缩包..."
tar -czf jetson_deploy.tar.gz "$PACKAGE_NAME"

# 清理临时目录
# rm -rf "$PACKAGE_DIR"

echo ""
echo "==================================="
echo "✅ 打包完成!"
echo "==================================="
echo "文件: jetson_deploy.tar.gz"
echo "大小: $(du -h jetson_deploy.tar.gz | cut -f1)"
echo ""
echo "传输到Jetson:"
echo "  scp jetson_deploy.tar.gz jetson@<IP>:~/"
echo ""
echo "Jetson上解压:"
echo "  tar -xzf jetson_deploy.tar.gz"
echo "  cd jetson_deploy"
echo "  ./setup.sh"

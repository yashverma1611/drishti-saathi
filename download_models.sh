#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p "$DIR/onnx"

echo "Downloading open-source Apache-2.0 / MIT models for Drishti Saathi..."

# YOLOX-Nano (Apache 2.0)
if [ ! -f "$DIR/onnx/yolox_nano.onnx" ]; then
    echo "Downloading YOLOX-Nano ONNX..."
    curl -L "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx" -o "$DIR/onnx/yolox_nano.onnx"
fi

# MiDaS v2.1 Small (MIT)
if [ ! -f "$DIR/onnx/midas_v21_small.onnx" ]; then
    echo "Downloading MiDaS v2.1 Small ONNX..."
    curl -L "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx" -o "$DIR/onnx/midas_v21_small.onnx"
fi

echo "All ONNX models ready in $DIR/onnx"

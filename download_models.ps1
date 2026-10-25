# PowerShell script to download approved Apache-2.0 ONNX model weights
$modelsDir = "$PSScriptRoot"
Write-Host "Downloading ONNX models for Drishti Saathi..." -ForegroundColor Cyan

# Ensure target directories exist
New-Item -ItemType Directory -Path "$modelsDir\onnx" -Force | Out-Null

# YOLOX-Nano ONNX (Apache 2.0)
$yoloxUrl = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
$yoloxDest = "$modelsDir\onnx\yolox_nano.onnx"
if (-not (Test-Path $yoloxDest)) {
    Write-Host "Downloading YOLOX-Nano ONNX..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $yoloxUrl -OutFile $yoloxDest
} else {
    Write-Host "YOLOX-Nano ONNX already present." -ForegroundColor Green
}

# MiDaS Small ONNX (MIT)
$midasUrl = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
$midasDest = "$modelsDir\onnx\midas_v21_small.onnx"
if (-not (Test-Path $midasDest)) {
    Write-Host "Downloading MiDaS v2.1 Small ONNX..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $midasUrl -OutFile $midasDest
} else {
    Write-Host "MiDaS Small ONNX already present." -ForegroundColor Green
}

Write-Host "Model downloads complete." -ForegroundColor Green

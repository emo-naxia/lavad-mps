#!/bin/bash
# prepare_datasets.sh
# 一键下载并解压 LAVAD 数据集（UCF-Crime + XD-Violence）
# 适用于 macOS / Linux

set -e  # 遇到错误时立即退出

# 目标目录
DATASET_DIR="datasets"
mkdir -p "$DATASET_DIR"

# 检查 gdown 是否已安装
if ! command -v gdown &> /dev/null
then
    echo "未检测到 gdown，正在安装..."
    pip install gdown
fi

echo "开始下载 UCF-Crime 数据文件..."
gdown --fuzzy "https://drive.google.com/file/d/1_7juCgOoWjQruyH3S8_FBqajuRaORmnV/view?usp=sharing" -O "$DATASET_DIR/UCF-Crime.zip"

echo "开始下载 XD-Violence 数据文件..."
gdown --fuzzy "https://drive.google.com/file/d/1yzDP1lVwPlA_BS2N5Byr1PcaazBklfkI/view?usp=sharing" -O "$DATASET_DIR/XD-Violence.zip"

echo "解压 UCF-Crime..."
unzip -o "$DATASET_DIR/UCF-Crime.zip" -d "$DATASET_DIR/UCF-Crime"

echo "解压 XD-Violence..."
unzip -o "$DATASET_DIR/XD-Violence.zip" -d "$DATASET_DIR/XD-Violence"

echo "数据集准备完成！目录结构："
tree "$DATASET_DIR" || ls -R "$DATASET_DIR"
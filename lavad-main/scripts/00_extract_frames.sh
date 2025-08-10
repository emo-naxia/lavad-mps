##!/bin/bash
#dataset_dir="YOUR_DATASET_PATH"
#
## Set paths
#videos_dir="${dataset_dir}/videos"
#frames_dir="${dataset_dir}/frames"
#annotations_file="${dataset_dir}/annotations/test.txt"
#
#python3 src/preprocessing/extract_frames.py /
#    --videos_dir "$videos_dir" /
#    --frames_dir "$frames_dir" /
#    --annotations_file "$annotations_file"


#!/usr/bin/env bash
set -euo pipefail

# ====== 用户只改这一行 ======
VIDEO_NAME="kelin_x264"   # 你的新视频名（不带.mp4）

# ====== 固定路径（按你的项目结构） ======
DATASET_DIR="datasets/UCF-Crime/ucf_crime"
VIDEO_PATH="$DATASET_DIR/videos/Normal_Videos/${VIDEO_NAME}.mp4"
FRAME_DIR="$DATASET_DIR/frames/Normal_Videos/${VIDEO_NAME}"

# 1) 检查视频是否存在
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ 找不到视频文件: $VIDEO_PATH"
    exit 1
fi

# 2) 创建帧目录
mkdir -p "$FRAME_DIR"

# 3) 自动抽帧
echo "===> 抽帧中..."
ffmpeg -hide_banner -loglevel error -y \
  -i "$VIDEO_PATH" \
  -vf fps=24 -start_number 0 \
  "$FRAME_DIR/%06d.jpg"

# 4) 自检
COUNT=$(find "$FRAME_DIR" -maxdepth 1 -name '*.jpg' | wc -l | tr -d ' ')
echo "✅ 共导出 $COUNT 帧到 $FRAME_DIR"
##!/bin/bash
## 自动检测 MPS 支持并设置设备变量
#DEVICE="cpu"
#python3 - <<'EOF' >/dev/null 2>&1
#import torch
#if torch.backends.mps.is_available():
#    exit(0)
#else:
#    exit(1)
#EOF
#if [ $? -eq 0 ]; then
#    DEVICE="mps"
#elif python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
#    DEVICE="cuda"
#fi
#echo "🚀 检测到推理设备: $DEVICE"
#
##!/bin/bash
#dataset_dir="YOUR_DATASET_PATH"
#batch_size=32
#frame_interval=1
#
## Set paths
#root_path="${dataset_dir}/frames"
#annotationfile_path="${dataset_dir}/annotations/test.txt"
#
## Define pretrained model names array
#pretrained_model_names=(
#    "Salesforce/blip2-opt-6.7b-coco"
#    "Salesforce/blip2-opt-6.7b"
#    "Salesforce/blip2-flan-t5-xxl"
#    "Salesforce/blip2-flan-t5-xl"
#    "Salesforce/blip2-flan-t5-xl-coco"
#)
#
#for pretrained_model_name in "${pretrained_model_names[@]}"; do
#    echo "Processing model: $pretrained_model_name"
#
#    output_dir="${dataset_dir}/captions/raw/${pretrained_model_name}/"
#
#    python3 -m src.models.image_captioner /
#        --root_path "$root_path" /
#        --annotationfile_path "$annotationfile_path" /
#        --batch_size "$batch_size" /
#        --frame_interval "$frame_interval" /
#        --pretrained_model_name "$pretrained_model_name" /
#        --output_dir "$output_dir"
#done


#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  ./scripts/01_caption.sh <name-or-frames-dir-or-mp4> [class]

参数说明:
  1) 传视频名(不带扩展名)，如: kelin_x264
     会用 frames/Normal_Videos/<name> 做帧目录（或配合第2参自定义类目）
  2) 传帧目录的绝对/相对路径，如: datasets/UCF-Crime/ucf_crime/frames/Normal_Videos/kelin_x264
  3) 传 mp4 文件路径，如: datasets/.../videos/Normal_Videos/kelin_x264.mp4
     仅用于推断 name 和 class，不会自动抽帧；若帧不存在会提示先跑 00_extract_frames.sh

示例:
  ./scripts/01_caption.sh kelin_x264
  ./scripts/01_caption.sh kelin_x264 Normal_Videos
  ./scripts/01_caption.sh datasets/UCF-Crime/ucf_crime/frames/Normal_Videos/kelin_x264
  ./scripts/01_caption.sh datasets/UCF-Crime/ucf_crime/videos/Normal_Videos/kelin_x264.mp4
EOF
}

if [[ $# -lt 1 ]]; then
  usage; exit 1
fi

ARG1="$1"
CLASS_ARG="${2:-}"

DATASET_ROOT="datasets/UCF-Crime/ucf_crime"
FRAMES_ROOT="$DATASET_ROOT/frames"
CAPTIONS_ROOT="$DATASET_ROOT/captions"

is_dir()  { [[ -d "$1" ]]; }
is_file() { [[ -f "$1" ]]; }

infer_class_from_path() {
  # 从路径里抓第一个目录名作为类目（Arson/Assault/Normal_Videos 等）
  local p="$1"
  if [[ "$p" =~ /(Abuse|Arrest|Arson|Assault|Burglary|Explosion|Fighting|Robbery|Shooting|Shoplifting|Stealing|Normal_Videos)/ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "Normal_Videos"
  fi
}

VIDEO_NAME=""
CLASS=""
FRAMES_DIR=""

if is_dir "$ARG1"; then
  # 传的是帧目录
  FRAMES_DIR="$ARG1"
  VIDEO_NAME="$(basename "$FRAMES_DIR")"
  CLASS="$(infer_class_from_path "$FRAMES_DIR")"
elif [[ "$ARG1" == *.mp4 ]]; then
  # 传的是 mp4 文件
  mp4="$ARG1"
  VIDEO_NAME="$(basename "$mp4")"
  VIDEO_NAME="${VIDEO_NAME%.mp4}"
  CLASS="${CLASS_ARG:-$(infer_class_from_path "$mp4")}"
  FRAMES_DIR="$FRAMES_ROOT/$CLASS/$VIDEO_NAME"
else
  # 传的是纯名字
  VIDEO_NAME="$ARG1"
  CLASS="${CLASS_ARG:-Normal_Videos}"
  FRAMES_DIR="$FRAMES_ROOT/$CLASS/$VIDEO_NAME"
fi

OUT_JSON="$CAPTIONS_ROOT/$CLASS/${VIDEO_NAME}.json"

echo "===> VIDEO_NAME : $VIDEO_NAME"
echo "===> CLASS      : $CLASS"
echo "===> FRAMES_DIR : $FRAMES_DIR"
echo "===> OUT_JSON   : $OUT_JSON"

if ! is_dir "$FRAMES_DIR"; then
  echo "❌ 找不到帧目录：$FRAMES_DIR"
  echo "   若你只有 mp4，请先运行 00_extract_frames.sh 抽帧。"
  exit 2
fi

mkdir -p "$(dirname "$OUT_JSON")"

# 内嵌 Python: 读取帧并写出 captions JSON（占位说明，可换成真实模型生成）
VIDEO_NAME_ENV="$VIDEO_NAME" FRAMES_DIR_ENV="$FRAMES_DIR" OUT_JSON_ENV="$OUT_JSON" \
python - <<'PY'
import os, json, re, sys

video_name = os.environ["VIDEO_NAME_ENV"]
frames_dir = os.environ["FRAMES_DIR_ENV"]
out_json   = os.environ["OUT_JSON_ENV"]

def key(fn):
    m = re.search(r'(\d+)\.jpg$', fn)
    return int(m.group(1)) if m else 10**9

frames = [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
frames.sort(key=key)

data = []
for i, fn in enumerate(frames):
    data.append({
        "frame_index": i,
        "file": fn,
        "caption": f"{video_name} 第 {i} 帧（文件 {fn}）"
    })

os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(data)} captions to {out_json}")
print("===> 预览（前3条/后3条）")
for x in data[:3]:
    print(x["caption"])
print("... ...")
for x in data[-3:]:
    print(x["caption"])
print("总条数:", len(data))
PY

echo "✅ 01_caption.sh 完成。"
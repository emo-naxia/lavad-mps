##!/bin/bash
#dataset_dir="YOUR_DATASET_PATH"
#batch_size=32
#frame_interval=16
#index_dim=1024
#
## Set paths
#root_path="${dataset_dir}/frames"
#annotationfile_path="${dataset_dir}/annotations/test.txt"
#
#cap_model_names=(
#    "$dataset_dir/captions/raw/Salesforce/blip2-opt-6.7b-coco/"
#    "$dataset_dir/captions/raw/Salesforce/blip2-opt-6.7b/"
#    "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xxl/"
#    "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xl/"
#    "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xl-coco/"
#)
#
#cap_model_names_str=$(IFS=' '; echo "${cap_model_names[*]}")
#
## Extract names and concatenate with "+"
#names=""
#IFS='/' read -ra components <<< "$cap_model_names_str"
#for component in "${components[@]}"; do
#    if [[ "$component" =~ ^blip2- ]]; then
#        names+="${component#blip2-}+"
#    fi
#done
#
## Remove the trailing "+" if present
#names=${names%+}
#
#echo "Creating index for $names"
#
#index_name="index_flat_ip"
#output_dir="${dataset_dir}/index/${names}/${index_name}/"
## shellcheck disable=SC2086 # We want to pass a list of strings
#python3 -m src.models.create_index /
#    --index_dim "$index_dim" /
#    --root_path "$root_path" /
#    --annotationfile_path "$annotationfile_path" /
#    --batch_size "$batch_size" /
#    --frame_interval "$frame_interval" /
#    --output_dir "${output_dir}" /
#    --captions_dirs $cap_model_names_str


#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  ./scripts/02_create_index.sh <name-or-frames-dir-or-mp4> [class]

参数:
  1) 传视频名(不带扩展名)，如: kelin_x264
     默认类目 Normal_Videos（或通过第2参指定）
  2) 传帧目录路径，如: datasets/UCF-Crime/ucf_crime/frames/Normal_Videos/kelin_x264
  3) 传 mp4 路径，如: datasets/.../videos/Normal_Videos/kelin_x264.mp4
     仅用于推断 name/class，不会自动抽帧；若帧不存在会提示先跑 00

示例:
  ./scripts/02_create_index.sh kelin_x264
  ./scripts/02_create_index.sh kelin_x264 Normal_Videos
  ./scripts/02_create_index.sh datasets/UCF-Crime/ucf_crime/frames/Normal_Videos/kelin_x264
  ./scripts/02_create_index.sh datasets/UCF-Crime/ucf_crime/videos/Normal_Videos/kelin_x264.mp4
EOF
}

if [[ $# -lt 1 ]]; then
  usage; exit 1
fi

ARG1="$1"
CLASS_ARG="${2:-}"

DATASET_ROOT="datasets/UCF-Crime/ucf_crime"
FRAMES_ROOT="$DATASET_ROOT/frames"
SIM_ROOT="$DATASET_ROOT/similarity"
INDEX_ROOT="$DATASET_ROOT/index/custom-resnet18/index_flat_ip"

is_dir()  { [[ -d "$1" ]]; }
is_file() { [[ -f "$1" ]]; }

infer_class_from_path() {
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
  FRAMES_DIR="$ARG1"
  VIDEO_NAME="$(basename "$FRAMES_DIR")"
  CLASS="$(infer_class_from_path "$FRAMES_DIR")"
elif [[ "$ARG1" == *.mp4 ]]; then
  mp4="$ARG1"
  VIDEO_NAME="$(basename "$mp4")"
  VIDEO_NAME="${VIDEO_NAME%.mp4}"
  CLASS="${CLASS_ARG:-$(infer_class_from_path "$mp4")}"
  FRAMES_DIR="$FRAMES_ROOT/$CLASS/$VIDEO_NAME"
else
  VIDEO_NAME="$ARG1"
  CLASS="${CLASS_ARG:-Normal_Videos}"
  FRAMES_DIR="$FRAMES_ROOT/$CLASS/$VIDEO_NAME"
fi

OUT_SIM="$SIM_ROOT/$CLASS/${VIDEO_NAME}.json"
OUT_NPZ="$INDEX_ROOT/$CLASS/${VIDEO_NAME}.npz"

echo "===> VIDEO_NAME : $VIDEO_NAME"
echo "===> CLASS      : $CLASS"
echo "===> FRAMES_DIR : $FRAMES_DIR"
echo "===> OUT_SIM    : $OUT_SIM"
echo "===> OUT_NPZ    : $OUT_NPZ"

if ! is_dir "$FRAMES_DIR"; then
  echo "❌ 找不到帧目录：$FRAMES_DIR"
  echo "   若你只有 mp4，请先运行 00_extract_frames.sh 抽帧。"
  exit 2
fi

mkdir -p "$(dirname "$OUT_SIM")" "$(dirname "$OUT_NPZ")"

# ------- Python: 提取特征并计算相似邻居 -------
FRAMES_DIR_ENV="$FRAMES_DIR" OUT_SIM_ENV="$OUT_SIM" OUT_NPZ_ENV="$OUT_NPZ" \
python - <<'PY'
import os, json, math
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

frames_dir = os.environ["FRAMES_DIR_ENV"]
out_sim = os.environ["OUT_SIM_ENV"]
out_npz = os.environ["OUT_NPZ_ENV"]

# 读取帧文件（按编号排序）
def frame_key(fn):
    try:
        return int(os.path.splitext(fn)[0])
    except:
        return 10**9

frames = [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
frames.sort(key=frame_key)
if not frames:
    raise RuntimeError(f"No .jpg frames found in {frames_dir}")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"===> Using device: {device}")

# 预处理 & 模型
tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
weights = models.ResNet18_Weights.IMAGENET1K_V1
backbone = models.resnet18(weights=weights)
feat_dim = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone.eval().to(device)

# 提取特征
feats = []
with torch.inference_mode():
    for i, fn in enumerate(frames):
        img = Image.open(os.path.join(frames_dir, fn)).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        f = backbone(x)             # [1, 512]
        f = nn.functional.normalize(f, dim=1)  # 归一化便于余弦
        feats.append(f.cpu())
        if (i+1) % 20 == 0:
            print(f"   extracted {i+1}/{len(frames)} features")

feats = torch.cat(feats, dim=0)   # [N, 512]
N = feats.shape[0]

# 保存 “索引” 特征
import numpy as np
os.makedirs(os.path.dirname(out_npz), exist_ok=True)
np.savez(out_npz, features=feats.numpy(), frames=np.array(frames))
print(f"Saved features: {out_npz}  (N={N}, dim={feats.shape[1]})")

# 计算相似度并取 top-k 邻居（排除自身）
topk = 5
# 余弦相似 = 归一化后的点积
S = feats @ feats.T                     # [N, N]
S.fill_diagonal_(-1.0)                  # 排除自身
vals, idxs = torch.topk(S, k=min(topk, max(1, N-1)), dim=1)

# 写 JSON: 每帧 -> [ {neighbor_frame_index, score}, ... ]
sim_obj = {}
for i in range(N):
    items = []
    for j in range(idxs.shape[1]):
        items.append({
            "neighbor": int(idxs[i, j].item()),
            "score": float(vals[i, j].item())
        })
    sim_obj[str(i)] = items

os.makedirs(os.path.dirname(out_sim), exist_ok=True)
with open(out_sim, "w", encoding="utf-8") as f:
    json.dump(sim_obj, f, ensure_ascii=False, indent=2)

print(f"Wrote similarity JSON: {out_sim}")
print("===> 预览：前3帧的 top-5 邻居")
for k in ["0","1","2"]:
    if k in sim_obj:
        print(f"frame {k} ->", sim_obj[k])
print("总帧数:", N)
PY

echo "✅ 02_create_index.sh 完成。"
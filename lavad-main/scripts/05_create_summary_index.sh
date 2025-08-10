##!/bin/bash
#export OMP_NUM_THREADS=8
#
#dataset_dir="YOUR_DATASET_PATH"
#llm_model_name="llama-2-13b-chat"
#batch_size=32
#frame_interval=16
#index_dim=1024
#index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"  # Change this to the index name you created in scripts/02_create_index.sh
#
## Set paths
#root_path="${dataset_dir}/frames"
#annotationfile_path="${dataset_dir}/annotations/test.txt"
#
#captions_dir="${dataset_dir}/captions/summary/${llm_model_name}/${index_name}/"
#output_dir="${dataset_dir}/index/summary/${llm_model_name}/${index_name}/index_flat_ip/"
#python3 -m src.models.create_summary_index /
#    --index_dim "$index_dim" /
#    --root_path "$root_path" /
#    --annotationfile_path "$annotationfile_path" /
#    --batch_size "$batch_size" /
#    --frame_interval "$frame_interval" /
#    --captions_dir "${captions_dir}" /
#    --output_dir "${output_dir}"


#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./scripts/05_create_summary_index.sh <video_name>
# 例如：
#   ./scripts/05_create_summary_index.sh kelin_x264
#
# 作用：
# - 读取 02 生成的相似度文件（similarity JSON）
# - 读取 03 生成的清洗后的 caption JSON
# - 读取 04 生成的原始分数（raw scores）JSON
# - 组合保存为一个 npz 索引文件，供后续评估/可视化使用
#
# 约定目录结构与前面脚本一致：
# datasets/UCF-Crime/ucf_crime/
#   ├─ frames/Normal_Videos/<video_name>/*.jpg
#   ├─ similarity/Normal_Videos/<video_name>.json
#   ├─ captions/clean/custom-resnet18/Normal_Videos/<video_name>.json
#   └─ scores/raw/mock-llm/custom-resnet18/demo_000/Normal_Videos/<video_name>.json

VIDEO_NAME="${1:-}"
if [[ -z "$VIDEO_NAME" ]]; then
  echo "用法: $0 <video_name>  (例如: $0 kelin_x264)"
  exit 1
fi

CLASS="Normal_Videos"                 # 和 00/01/02/04 保持一致
DATASET_DIR="datasets/UCF-Crime/ucf_crime"

FRAMES_DIR="${DATASET_DIR}/frames/${CLASS}/${VIDEO_NAME}"
SIM_FILE="${DATASET_DIR}/similarity/${CLASS}/${VIDEO_NAME}.json"
CAPTIONS_CLEAN="${DATASET_DIR}/captions/clean/custom-resnet18/${CLASS}/${VIDEO_NAME}.json"
RAW_SCORES="${DATASET_DIR}/scores/raw/mock-llm/custom-resnet18/demo_000/${CLASS}/${VIDEO_NAME}.json"

OUT_SUM_DIR="${DATASET_DIR}/index/summary/mock-llm/custom-resnet18/demo_000/${CLASS}"
OUT_SUM_NPZ="${OUT_SUM_DIR}/${VIDEO_NAME}.npz"

# 邻居数量（可通过环境变量 K 覆盖，默认与 02/04 脚本保持 10 一致）
K="${K:-10}"

mkdir -p "$OUT_SUM_DIR"

echo "===> VIDEO_NAME : ${VIDEO_NAME}"
echo "===> CLASS      : ${CLASS}"
echo "===> FRAMES_DIR : ${FRAMES_DIR}"
echo "===> SIM_FILE   : ${SIM_FILE}"
echo "===> CAP_CLEAN  : ${CAPTIONS_CLEAN}"
echo "===> RAW_SCORES : ${RAW_SCORES}"
echo "===> OUT_SUM    : ${OUT_SUM_NPZ}"
echo "===> TOP-K      : ${K}"

# 把变量导出给 python 子进程
export SIM_FILE CAPTIONS_CLEAN RAW_SCORES OUT_SUM_NPZ K

python3 - <<'PY'
import os, json, numpy as np

SIM_FILE       = os.environ["SIM_FILE"]
CAPTIONS_CLEAN = os.environ["CAPTIONS_CLEAN"]
RAW_SCORES     = os.environ["RAW_SCORES"]
OUT_SUM_NPZ    = os.environ["OUT_SUM_NPZ"]
K              = int(os.environ.get("K", "10"))

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 1) 读取清洗后的 captions
caps = load_json(CAPTIONS_CLEAN)
# 统一成长度 N 的 caption 列表
if isinstance(caps, list):
    # 预期是 [{'frame_index': i, 'file': '000xxx.jpg', 'caption': '...'}]
    captions = []
    for i, it in enumerate(caps):
        if isinstance(it, dict) and "caption" in it:
            captions.append(it["caption"])
        else:
            captions.append(str(it))
else:
    # 万一是 dict，就按键排序后取值
    keys = sorted(caps.keys(), key=lambda x: int(x) if x.isdigit() else x)
    captions = [caps[k]["caption"] if isinstance(caps[k], dict) and "caption" in caps[k] else str(caps[k]) for k in keys]

N = len(captions)
if N == 0:
    raise RuntimeError("Clean captions 为空，无法继续。")

# 2) 读取 raw scores
raw = load_json(RAW_SCORES)
# 允许 list 或 { "0": score0, ... } 两种形态
if isinstance(raw, list):
    scores = np.asarray(raw, dtype="float32")
elif isinstance(raw, dict):
    # 键可能是帧号字符串
    try:
        keys = sorted(raw.keys(), key=lambda x: int(x))
    except Exception:
        keys = sorted(raw.keys())
    vals = []
    for k in keys:
        v = raw[k]
        # 兼容 {"0":0.12} 或 {"0":{"score":0.12}} 等
        if isinstance(v, dict) and "score" in v:
            vals.append(float(v["score"]))
        else:
            vals.append(float(v))
    scores = np.asarray(vals, dtype="float32")
else:
    raise RuntimeError("raw scores JSON 的类型不受支持，请检查。")

# 和 N 对齐（必要时截断/填充）
if len(scores) < N:
    # 用最后一个值填充到 N
    if len(scores) == 0:
        scores = np.zeros((N,), dtype="float32")
    else:
        pad = np.full((N - len(scores),), scores[-1], dtype="float32")
        scores = np.concatenate([scores, pad], axis=0)
elif len(scores) > N:
    scores = scores[:N]

# 3) 读取相似度（neighbors）
sim = load_json(SIM_FILE)

def take_top_k_from_entry(entry, k):
    """
    支持的几种 entry 形态：
    - [{'neighbor': idx, 'score': x}, ...]
    - [idx0, idx1, ...]（纯索引）
    - {'neighbors':[...]} / {'topk':[...]} / {'indices':[...]} 包含上述任一
    """
    if isinstance(entry, list):
        if len(entry) == 0:
            return []
        if isinstance(entry[0], dict) and "neighbor" in entry[0]:
            return [int(x["neighbor"]) for x in entry[:k]]
        else:
            return [int(x) for x in entry[:k]]
    if isinstance(entry, dict):
        for key in ("neighbors", "topk", "indices"):
            if key in entry:
                return take_top_k_from_entry(entry[key], k)
    # 都不匹配
    return []

# 允许几种容器外壳：
# A) list[ per-frame entry ]
# B) dict{ "0": per-frame entry, "1": ... }
# C) dict{ "neighbors": list[...] }（少见但兜底）
neighbors = []

if isinstance(sim, list):
    for i in range(N):
        entry = sim[i] if i < len(sim) else []
        nei = take_top_k_from_entry(entry, K)
        neighbors.append(nei)

elif isinstance(sim, dict):
    # 直接按数字键取
    frame_keys = []
    # 优先数字键
    digit_keys = [k for k in sim.keys() if str(k).isdigit()]
    if len(digit_keys) >= N:
        frame_keys = sorted(digit_keys, key=lambda x: int(x))
        for i in range(N):
            entry = sim[frame_keys[i]]
            nei = take_top_k_from_entry(entry, K)
            neighbors.append(nei)
    elif "neighbors" in sim and isinstance(sim["neighbors"], list):
        arr = sim["neighbors"]
        for i in range(N):
            entry = arr[i] if i < len(arr) else []
            nei = take_top_k_from_entry(entry, K)
            neighbors.append(nei)
    else:
        # 兜底：按一般键顺序
        keys = sorted(sim.keys())
        for i in range(N):
            key = keys[i] if i < len(keys) else keys[-1]
            nei = take_top_k_from_entry(sim[key], K)
            neighbors.append(nei)
else:
    raise RuntimeError("similarity JSON 的类型不受支持。")

# 把 neighbors 规整成 N x K（不足补自己索引或 -1）
neighbors_fixed = np.full((N, K), -1, dtype="int32")
for i in range(N):
    row = neighbors[i][:K]
    # 若不足 K，补上自身或 -1
    while len(row) < K:
        row.append(i if i < N else -1)
    neighbors_fixed[i] = np.asarray(row[:K], dtype="int32")

# 保存
np.savez(OUT_SUM_NPZ,
         captions=np.array(captions, dtype=object),
         scores=scores.astype("float32"),
         neighbors=neighbors_fixed.astype("int32"),
         allow_pickle=True)

print(f"Saved summary index: {OUT_SUM_NPZ}")
print(f"  N={N}, scores.shape={scores.shape}, neighbors.shape={neighbors_fixed.shape}")
print("  captions[0..2] preview:", captions[:3])
print("  scores[0..5] preview:", scores[:6].tolist())
print("  neighbors[0] preview:", neighbors_fixed[0].tolist())
PY

echo "✅ 05_create_summary_index.sh 完成。"
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
#export OMP_NUM_THREADS=8
#export CUDA_VISIBLE_DEVICES=0,1
#
#dataset_dir="YOUR_DATASET_PATH"
#llm_model_name="llama-2-13b-chat"
#batch_size=32
#frame_interval=16
#index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"  # Change this to the index name you created in scripts/02_create_index.sh
#
#echo "Processing index: $index_name"
#
## Set paths
#root_path="${dataset_dir}/frames"
#annotationfile_path="${dataset_dir}/annotations/test.txt"
#
#context_prompt="If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
#format_prompt="Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."
#summary_prompt="Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."
#
#captions_dir="$dataset_dir/captions/clean/$index_name/"
#
## Generate a 6-digit timestamp based on the current time
#exp_id=$(date +%s | tail -c 7)
#
## Convert to lowercase and replace spaces with underscores
#dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
## Truncate dir_name to the first 243 characters
#dir_name=$(echo "$dir_name" | cut -c1-243)
#dir_name=${dir_name/////_}
## Generate a directory name based on job and task IDs and the prompt
#dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")
#
#output_scores_dir="${dataset_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
#output_summary_dir="${dataset_dir}/captions/summary/${llm_model_name}/$index_name/"
#
#torchrun /
#    --nproc_per_node 2 --nnodes 1 -m src.models.llm_anomaly_scorer /
#    --root_path "$root_path" /
#    --annotationfile_path "$annotationfile_path" /
#    --batch_size "$batch_size" /
#    --frame_interval "$frame_interval" /
#    --summary_prompt "$summary_prompt" /
#    --output_summary_dir "$output_summary_dir" /
#    --captions_dir "$captions_dir" /
#    --ckpt_dir libs/llama/llama-2-13b-chat/ /
#    --tokenizer_path libs/llama/tokenizer.model
#
#torchrun /
#    --nproc_per_node 2 --nnodes 1 -m src.models.llm_anomaly_scorer /
#    --root_path "$root_path" /
#    --annotationfile_path "$annotationfile_path" /
#    --batch_size "$batch_size" /
#    --frame_interval "$frame_interval" /
#    --output_summary_dir "$output_summary_dir" /
#    --context_prompt "$context_prompt" /
#    --format_prompt "$format_prompt" /
#    --output_scores_dir "$output_scores_dir" /
#    --ckpt_dir libs/llama/llama-2-13b-chat/ /
#    --tokenizer_path libs/llama/tokenizer.model /
#    --score_summary


#!/usr/bin/env bash
set -euo pipefail

VIDEO_NAME="${1:-kelin_x264}"
CLASS="${CLASS:-Normal_Videos}"

DATASET_DIR="datasets/UCF-Crime/ucf_crime"
SIM_DIR="$DATASET_DIR/similarity/$CLASS"
CAPTIONS_JSON="$DATASET_DIR/captions/$CLASS/${VIDEO_NAME}.json"

# 模拟一个“LLM+索引模型”的组合名（和 00-03 的风格一致）
LLM_MODEL="${LLM_MODEL:-mock-llm}"
INDEX_NAME="${INDEX_NAME:-custom-resnet18}"
EXP_ID="${EXP_ID:-demo_000}"

OUT_RAW_DIR="$DATASET_DIR/scores/raw/$LLM_MODEL/$INDEX_NAME/$EXP_ID/$CLASS"
OUT_RAW_JSON="$OUT_RAW_DIR/${VIDEO_NAME}.json"

export SIM_DIR CAPTIONS_JSON OUT_RAW_JSON VIDEO_NAME

echo "===> VIDEO_NAME : $VIDEO_NAME"
echo "===> CLASS      : $CLASS"
echo "===> SIM_FILE   : $SIM_DIR/${VIDEO_NAME}.json"
echo "===> CAPTIONS   : $CAPTIONS_JSON"
echo "===> RAW_OUT    : $OUT_RAW_JSON"

mkdir -p "$OUT_RAW_DIR"

python - <<'PY'
import os, json, sys
sim_file   = os.path.join(os.environ["SIM_DIR"], f"{os.environ['VIDEO_NAME']}.json")
caps_file  = os.environ["CAPTIONS_JSON"]
out_file   = os.environ["OUT_RAW_JSON"]

# 读 captions 拿帧数 N（我们 01/03 的格式是 list，每条是 {frame_index, file, caption}）
with open(caps_file, "r") as f:
    caps = json.load(f)
N = len(caps)

# 读 02 步的相似度 json：支持 dict 或 list 两种简单结构
with open(sim_file, "r") as f:
    sim_obj = json.load(f)

def top1_score_for(i):
    # 允许 sim_obj 是 dict（键是 str 下标）或 list
    row = sim_obj.get(str(i)) if isinstance(sim_obj, dict) else sim_obj[i]
    # row 可能是 [{'neighbor': ..., 'score': ...}, ...]
    if isinstance(row, list) and row and isinstance(row[0], dict) and "score" in row[0]:
        return float(row[0]["score"])
    # 兜底：没有相似度就给个保守值 0.5
    return 0.5

scores = []
for i in range(N):
    s = 1.0 - max(0.0, min(1.0, top1_score_for(i)))  # 映射到 [0,1]
    scores.append(round(s, 6))

with open(out_file, "w") as f:
    json.dump(scores, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(scores)} scores to {out_file}")
print("preview:", scores[:5], "...", scores[-5:])
PY

echo "✅ 04_query_llm.sh 完成。"
##!/bin/bash
#export OMP_NUM_THREADS=8
#
#dataset_dir="datasets/UCF-Crime/ucf_crime"
#llm_model_name="llama-2-13b-chat"
#frame_interval=16
#num_neighbors=10
#video_fps=30
#
#exp_id="2035605_000"
#index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
#
## Set paths
#root_path="${dataset_dir}/frames"
#annotationfile_path="${dataset_dir}/annotations/test.txt"
#
#context_prompt="How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
#
## Convert to lowercase and replace spaces with underscores
#dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
## Truncate dir_name to the first 243 characters
#dir_name=$(echo "$dir_name" | cut -c1-243)
#dir_name=${dir_name/////_}
## Generate a directory name based on job and task IDs and the prompt
#dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")
#
#captions_dir="${dataset_dir}/captions/clean_summary/${llm_model_name}/$index_name/"
#scores_dir="${dataset_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
#similarity_dir="${dataset_dir}/similarity/clean_summary/${llm_model_name}/${index_name}/"
#output_dir="${dataset_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
#
#python3 -m src.eval /
#    --root_path "$root_path" /
#    --annotationfile_path "$annotationfile_path" /
#    --scores_dir "$scores_dir" /
#    --similarity_dir "$similarity_dir" /
#    --captions_dir "$captions_dir" /
#    --output_dir "$output_dir" /
#    --frame_interval "$frame_interval" /
#    --num_neighbors "$num_neighbors" /
#    --without_labels /
#    --visualize /
#    --video_fps "$video_fps"




#!/usr/bin/env bash
set -euo pipefail

VIDEO_NAME="${1:-}"
if [ -z "$VIDEO_NAME" ]; then
  echo "usage: $0 <video_name>"; exit 1
fi

CLASS="Normal_Videos"
ROOT="datasets/UCF-Crime/ucf_crime"

FRAMES_DIR="$ROOT/frames/$CLASS/$VIDEO_NAME"
SIM_DIR="$ROOT/similarity/$CLASS"
CAP_DIR="$ROOT/captions/clean/custom-resnet18/$CLASS"
SCORES_DIR="$ROOT/scores/refined/mock-llm/custom-resnet18/demo_000/$CLASS"
OUT_DIR="outputs/${VIDEO_NAME}_eval"

echo "===> VIDEO_NAME : $VIDEO_NAME"
echo "===> CLASS      : $CLASS"
echo "===> FRAMES     : $FRAMES_DIR"
echo "===> SIM_DIR    : $SIM_DIR"
echo "===> CAP_DIR    : $CAP_DIR"
echo "===> SCORES_DIR : $SCORES_DIR"
echo "===> OUT_DIR    : $OUT_DIR"

mkdir -p "$OUT_DIR"

export VIDEO_NAME SCORES_DIR OUT_DIR SIM_DIR CAP_DIR

python - <<'PY'
import os, json, math
import matplotlib.pyplot as plt

vid = os.environ["VIDEO_NAME"]
scores_path = f'{os.environ["SCORES_DIR"]}/{vid}.json'
out_dir = os.environ["OUT_DIR"]

with open(scores_path) as f:
    arr = json.load(f)

n = len(arr)
mn = min(arr) if n else float("nan")
mx = max(arr) if n else float("nan")
mean = sum(arr)/n if n else float("nan")
var = sum((x-mean)*(x-mean) for x in arr)/n if n else float("nan")
std = math.sqrt(var) if n else float("nan")

with open(os.path.join(out_dir, "stats.txt"), "w") as f:
    f.write(f"video={vid}\n")
    f.write(f"count={n}\n")
    f.write(f"min={mn}\n")
    f.write(f"mean={mean}\n")
    f.write(f"max={mx}\n")
    f.write(f"std={std}\n")

with open(os.path.join(out_dir, "scores.csv"), "w") as f:
    f.write("frame,score\n")
    for i, s in enumerate(arr):
        f.write(f"{i},{s}\n")

plt.figure()
plt.plot(arr)
plt.title(f"Refined anomaly scores: {vid}")
plt.xlabel("frame index")
plt.ylabel("score")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "curve.png"), dpi=150)

print(f"n={n} preview_head={arr[:5]} preview_tail={arr[-5:]}")
print("Wrote:", os.path.join(out_dir, "stats.txt"))
print("Wrote:", os.path.join(out_dir, "scores.csv"))
print("Wrote:", os.path.join(out_dir, "curve.png"))
PY

echo "✅ 07_eval.sh 完成。输出在 $OUT_DIR"
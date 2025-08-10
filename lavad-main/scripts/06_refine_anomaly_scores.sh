#!/usr/bin/env bash
set -euo pipefail

VIDEO_NAME="${1:-kelin_x264}"
CLASS="${CLASS:-Normal_Videos}"

LLM_TAG="mock-llm"
INDEX_TAG="custom-resnet18"
EXP_ID="demo_000"
TOPK="${TOPK:-10}"
ALPHA="${ALPHA:-0.5}"

ROOT="datasets/UCF-Crime/ucf_crime"

RAW_SCORES="${ROOT}/scores/raw/${LLM_TAG}/${INDEX_TAG}/${EXP_ID}/${CLASS}/${VIDEO_NAME}.json"
SUMMARY_NPZ="${ROOT}/index/summary/${LLM_TAG}/${INDEX_TAG}/${EXP_ID}/${CLASS}/${VIDEO_NAME}.npz"
OUT_DIR="${ROOT}/scores/refined/${LLM_TAG}/${INDEX_TAG}/${EXP_ID}/${CLASS}"
OUT_JSON="${OUT_DIR}/${VIDEO_NAME}.json"

echo "===> VIDEO_NAME : ${VIDEO_NAME}"
echo "===> CLASS      : ${CLASS}"
echo "===> RAW_SCORES : ${RAW_SCORES}"
echo "===> SUMMARY_NPZ: ${SUMMARY_NPZ}"
echo "===> OUT_JSON   : ${OUT_JSON}"
echo "===> TOP-K      : ${TOPK} | ALPHA=${ALPHA}"

mkdir -p "${OUT_DIR}"

# 关键：把变量导出到 Python 子进程
export RAW_SCORES SUMMARY_NPZ OUT_JSON TOPK ALPHA

python - <<'PY'
import os, json, numpy as np

raw_path = os.environ["RAW_SCORES"]
sum_path = os.environ["SUMMARY_NPZ"]
out_path = os.environ["OUT_JSON"]
topk = int(os.environ.get("TOPK","10"))
alpha = float(os.environ.get("ALPHA","0.5"))

with open(raw_path) as f:
    raw = json.load(f)

if isinstance(raw, dict):
    keys = sorted(int(k) for k in raw.keys())
    raw = [ float(np.mean(list(d.values()))) for d in (raw[str(k)] for k in keys) ]

raw = np.asarray(raw, dtype=np.float32)
N = len(raw)

npz = np.load(sum_path, allow_pickle=True)
neighbors = npz["neighbors"]
if neighbors.shape[0] != N:
    raise RuntimeError(f"neighbors length {neighbors.shape[0]} != raw length {N}")

K = min(topk, neighbors.shape[1])
nbr = neighbors[:, :K].astype(int)

nbr_means = np.zeros(N, dtype=np.float32)
for i in range(N):
    idxs = [j for j in nbr[i].tolist() if 0 <= j < N]
    nbr_means[i] = float(np.mean(raw[idxs])) if idxs else raw[i]

refined = alpha * raw + (1.0 - alpha) * nbr_means
refined = refined.tolist()

with open(out_path, "w") as f:
    json.dump(refined, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(refined)} refined scores to {out_path}")
print("preview:", refined[:5], "...", refined[-5:])
PY

echo "✅ 06_refine_anomaly_scores.sh 完成。"

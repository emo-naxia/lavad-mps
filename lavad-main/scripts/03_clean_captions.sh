#!/usr/bin/env bash
set -euo pipefail

VIDEO_NAME="${1:-kelin_x264}"
CLASS="${2:-Normal_Videos}"

CAPTIONS_IN="datasets/UCF-Crime/ucf_crime/captions/${CLASS}/${VIDEO_NAME}.json"
OUT_JSON="datasets/UCF-Crime/ucf_crime/captions/clean/custom-resnet18/${CLASS}/${VIDEO_NAME}.json"

echo "===> VIDEO_NAME : ${VIDEO_NAME}"
echo "===> CLASS      : ${CLASS}"
echo "===> CAPTIONS_IN: ${CAPTIONS_IN}"
echo "===> OUT_JSON   : ${OUT_JSON}"

mkdir -p "$(dirname "${OUT_JSON}")"

CAPTIONS_IN="${CAPTIONS_IN}" OUT_JSON="${OUT_JSON}" python - <<'PY'
import os, json, re, sys, pathlib
inp = os.environ["CAPTIONS_IN"]
out = os.environ["OUT_JSON"]

def clean_text(s: str) -> str:
    s = s.replace('\u00A0', ' ')   # nbsp -> space
    s = s.replace('\t', ' ')
    s = re.sub(r'\s+', ' ', s)     # collapse spaces/newlines
    return s.strip()

with open(inp, 'r', encoding='utf-8') as f:
    data = json.load(f)

preview_head, preview_tail = [], []

if isinstance(data, list):
    cleaned = [clean_text(str(x)) for x in data]
    n = len(cleaned)
    preview_head = cleaned[:3]
    preview_tail = cleaned[-3:] if n >= 3 else cleaned[:]
elif isinstance(data, dict):
    # 保持原键（很多数据是以帧号字符串做键）
    cleaned = {str(k): clean_text(str(v)) for k, v in data.items()}
    # 为了预览，按键排序看前三/后三
    items = sorted(cleaned.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
    n = len(items)
    preview_head = [f"{k}: {v}" for k, v in items[:3]]
    preview_tail = [f"{k}: {v}" for k, v in items[-3:]] if n >= 3 else [f"{k}: {v}" for k, v in items]
else:
    print(f"Unsupported JSON type: {type(data).__name__}", file=sys.stderr)
    sys.exit(1)

# 写出，格式保持与输入相同的结构（列表就写列表，字典就写字典）
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print(f"Wrote {n} cleaned captions to {out}")
print("===> 预览（前3条/后3条）")
for x in preview_head: print(x)
print("... ...")
for x in preview_tail: print(x)
print(f"总条数: {n}")
PY

echo "✅ 03_clean_captions.sh 完成。"

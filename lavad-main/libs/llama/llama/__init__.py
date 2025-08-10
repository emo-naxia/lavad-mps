"""
llama 模块初始化
确保可以直接从 llama 导入核心方法
"""

import os
import sys

# 确保可以正确找到 vad_infer.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ===== 从原生文件导入类 =====
from .generation import Llama, Dialog
from .model import ModelArgs, Transformer

# ===== 简化推理封装 =====
from llama_cpp import Llama as LlamaCpp

def simple_infer(prompt, model_path, max_tokens=50, temperature=0.7,
                 n_ctx=4096, n_threads=8, n_batch=512):
    """
    使用 llama.cpp 格式 GGUF 模型的简单推理函数
    """
    print(f"[simple_infer] 加载模型: {model_path}")
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch
    )

    output = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["</s>"],
        temperature=temperature
    )
    return output["choices"][0]["text"].strip()

# ===== 视频异常检测封装 =====
from vad_infer import infer_video_anomaly

__all__ = [
    "Llama",
    "Dialog",
    "ModelArgs",
    "Transformer",
    "simple_infer",
    "infer_video_anomaly"
]

import os
import time
from typing import Optional, List, Dict, Any
from llama_cpp import Llama  # 直接用 llama_cpp 加载 Qwen GGUF

# ======== Qwen 简化推理接口 ========
def simple_infer(
    prompt: str,
    model_path: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    n_ctx: int = 4096,
    n_threads: int = 8,
    n_batch: int = 512
) -> str:
    """
    一行调用 Qwen 模型推理。
    """
    print(f"Loading Qwen model from {model_path} ...")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch
    )
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>"]
    )
    return output["choices"][0]["text"].strip()


# ======== 原来 LLaMA 的 ModelArgs / Transformer （保持不动，兼容旧项目） ========
class ModelArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Transformer:
    def __init__(self, params: ModelArgs):
        self.params = params

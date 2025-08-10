import sys
import os
import importlib.util
import multiprocessing
import time

# 项目主目录
PROJECT_ROOT = "/Users/emo/lavad/lavad-main"

# 1️⃣ 强制把项目的 libs 路径放到 sys.path 最前面
PROJECT_LIBS_PATH = os.path.join(PROJECT_ROOT, "libs")
if PROJECT_LIBS_PATH not in sys.path:
    sys.path.insert(0, PROJECT_LIBS_PATH)

# 2️⃣ 手动加载 llama 模块
LLAMA_INIT = os.path.join(PROJECT_LIBS_PATH, "llama", "llama", "__init__.py")
spec = importlib.util.spec_from_file_location("llama", LLAMA_INIT)
llama = importlib.util.module_from_spec(spec)
sys.modules["llama"] = llama
spec.loader.exec_module(llama)

print(f"✅ 成功加载 llama 模块: {llama.__file__}")

# 检测设备
try:
    from llama.generation import DEVICE
    print(f"🚀 当前使用的推理设备: {DEVICE}")
except ImportError:
    print("⚠️ 无法检测 DEVICE，可能未正确修改 generation.py")

# 3️⃣ 从 llama 导入 simple_infer
from llama import simple_infer

# 4️⃣ 模型路径
model_path = "/Users/emo/models/qwen1_5-7b-chat-q4_k_m.gguf"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}")

# 线程优化（留一点给系统）
threads = max(1, multiprocessing.cpu_count() - 2)

# 5️⃣ 推理测试（带计时）
prompt = "你好，请用一句话介绍人工智能。"
start_time = time.time()

output = simple_infer(
    prompt=prompt,
    model_path=model_path,
    max_tokens=50,
    temperature=0.7,
    n_ctx=4096,
    n_threads=threads,
    n_batch=512
)

end_time = time.time()
elapsed = end_time - start_time
tokens_generated = len(output.split())  # 简单用词数估算
speed = tokens_generated / elapsed

print("💡 模型输出:", output)
print(f"⏱️ 耗时: {elapsed:.2f} 秒 | ⚡ 速度: {speed:.2f} tokens/s (估算)")
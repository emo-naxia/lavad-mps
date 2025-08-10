import sys
sys.path.insert(0, r"D:\lavad\lavad-main\libs")  # 把 libs 加到最前

from llama.llama import simple_infer  # 注意这里是 llama.llama

# 模型路径（改成你本地的 Qwen GGUF 模型）
model_path = r"D:\models\qwen1_5-7b-chat-q4_k_m.gguf"

prompt = "你好，请用一句话介绍人工智能。"

output = simple_infer(
    prompt=prompt,
    model_path=model_path,
    max_tokens=50,
    temperature=0.7,
    n_ctx=4096,   # 上下文长度
    n_threads=8,  # CPU 线程数
    n_batch=512,  # 批处理大小
)

print("模型输出：", output)

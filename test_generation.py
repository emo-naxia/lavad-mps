import sys
import os

# 确保可以找到 llama 模块
sys.path.insert(0, r"D:\lavad\lavad-main\libs")

# 尝试导入 Llama 类
from llama.llama.generation import Llama

print("✅ 成功导入 Llama 类")

# 这里测试 Llama 类是否能被实例化（用一个假的 model 和 tokenizer 代替）
# 只是验证 import，不会真的加载大模型
try:
    from llama.llama.model import ModelArgs, Transformer
    from llama.llama.tokenizer import Tokenizer

    fake_args = ModelArgs(max_seq_len=32, max_batch_size=1, vocab_size=100)
    fake_model = Transformer(fake_args)
    fake_tokenizer = Tokenizer(model_path="D:/models/qwen1_5-7b-chat-q4_k_m.gguf")  # 这里路径无所谓

    llama_instance = Llama(fake_model, fake_tokenizer)
    print("✅ Llama 实例化成功")
except Exception as e:
    print("⚠️ Llama 实例化失败:", e)

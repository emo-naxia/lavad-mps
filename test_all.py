import sys

# 1. 加入项目 libs 路径，保证 llama 可以正常导入
sys.path.insert(0, r"D:\lavad\lavad-main\libs")

# 2. 导入方法
from llama import simple_infer, infer_video_anomaly

# ==== 测试 simple_infer ====
print("===== 测试 simple_infer（Qwen 模型简单推理） =====")
model_path = r"D:\models\qwen1_5-7b-chat-q4_k_m.gguf"
prompt = "你好，请用一句话介绍人工智能。"

try:
    output_text = simple_infer(
        prompt=prompt,
        model_path=model_path,
        max_tokens=50,
        temperature=0.7,
        n_ctx=4096,
        n_threads=8,
        n_batch=512
    )
    print("✅ simple_infer 输出:", output_text)
except Exception as e:
    print("❌ simple_infer 运行出错:", e)

# ==== 测试 infer_video_anomaly ====
print("\n===== 测试 infer_video_anomaly（视频异常检测） =====")
video_info = """
帧1: 正常行人通过街道
帧2: 行人继续行走
帧3: 一辆汽车高速冲入人行道
帧4: 行人被迫快速避让
"""

try:
    anomaly_result = infer_video_anomaly(video_info, model_path=model_path)
    print("✅ infer_video_anomaly 输出:", anomaly_result)
except Exception as e:
    print("❌ infer_video_anomaly 运行出错:", e)

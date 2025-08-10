import sys
import importlib.util

# 确保能找到 llama 模块
sys.path.insert(0, r"D:\lavad\lavad-main\libs")

# 动态加载 llama 模块
spec = importlib.util.spec_from_file_location(
    "llama",
    r"D:\lavad\lavad-main\libs\llama\llama\__init__.py"
)
llama = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llama)

# 从 llama 导入 infer_video_anomaly
from llama import infer_video_anomaly

# 模型路径（改成你本地 Qwen GGUF 的路径）
model_path = r"D:\models\qwen1_5-7b-chat-q4_k_m.gguf"

# 模拟视频检测信息
video_info = """
帧1: 正常行人通过街道
帧2: 行人继续行走
帧3: 一辆汽车高速冲入人行道
帧4: 行人被迫快速避让
"""

# 调用推理
result = infer_video_anomaly(video_info, model_path)
print("模型分析结果：", result)

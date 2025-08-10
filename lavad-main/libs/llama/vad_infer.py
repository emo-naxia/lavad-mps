"""
视频异常检测封装
"""

from llama_cpp import Llama as LlamaCpp

def infer_video_anomaly(video_info, model_path,
                        n_ctx=4096, n_threads=8, n_batch=512,
                        max_tokens=100, temperature=0.7):
    """
    使用 Qwen/LLaMA 模型进行视频异常检测
    video_info: 视频关键帧文字描述
    model_path: 模型文件路径（GGUF）
    """
    print(f"[infer_video_anomaly] 加载模型: {model_path}")
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch
    )

    prompt = f"以下是视频关键帧描述，请判断是否有异常，并简要说明原因：\n{video_info}"
    output = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["</s>"],
        temperature=temperature
    )
    return output["choices"][0]["text"].strip()

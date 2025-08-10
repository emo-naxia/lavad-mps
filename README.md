# lavad-mps

## 项目简介 / Project Overview

`lavad-mps` 是在原项目 [lavad](https://github.com/lucazanella/lavad)
的基础上进行修改的版本，主要适配了 **macOS (Apple Silicon / MPS 加速)**
平台，同时保留了原始功能与结构，方便在不同硬件和操作系统上复现。

The `lavad-mps` project is a modified version of the original
[lavad](https://github.com/lucazanella/lavad) repository, adapted for
**macOS (Apple Silicon / MPS acceleration)**, while retaining the
original functionality and structure for easier reproduction across
various hardware and OS environments.

------------------------------------------------------------------------

## 主要改动 / Key Modifications

-   将原始的 **CUDA 专用代码** 修改为 **MPS（Metal Performance
    Shaders）** 兼容版本，用于 MacBook (Apple Silicon)。
-   调整了部分 shell 脚本，以保证在 macOS 上路径格式 (`/` 而非 `\`)
    正确。
-   删除了 Windows 和 Linux 平台下不兼容的部分依赖。
-   更新了 README 文档，提供了中英文复现说明。

------------------------------------------------------------------------

## 环境要求 / Environment Requirements

### macOS

-   macOS 13+
-   Apple Silicon 芯片（M1 / M2 / M3）
-   Python 3.10+
-   [Anaconda](https://www.anaconda.com/) 或 Miniconda

### Linux / Windows

-   CUDA 11.8+ (Linux / Windows)
-   NVIDIA GPU (6GB+ VRAM 建议)
-   Python 3.10+

------------------------------------------------------------------------

## 安装步骤 / Installation Steps

### 1. 克隆项目 / Clone the Repository

``` bash
git clone https://github.com/emo-naxia/lavad-mps.git
cd lavad-mps
```

### 2. 创建虚拟环境 / Create a Virtual Environment

``` bash
conda create -n lavad_mps python=3.10 -y
conda activate lavad_mps
```

### 3. 安装依赖 / Install Dependencies

macOS (MPS)

``` bash
pip install -r requirements.txt
```

Linux / Windows (CUDA)

``` bash
pip install -r requirements_cuda.txt
```

------------------------------------------------------------------------

## 运行流程 / Running the Pipeline

1.  **提取视频帧 / Extract frames**

``` bash
./scripts/00_extract_frames.sh <VIDEO_NAME>
```

2.  **生成字幕 / Generate captions**

``` bash
./scripts/01_caption.sh <VIDEO_NAME>
```

3.  **创建索引 / Create index**

``` bash
./scripts/02_create_index.sh <VIDEO_NAME>
```

4.  **清洗字幕 / Clean captions**

``` bash
./scripts/03_clean_captions.sh <VIDEO_NAME>
```

5.  **调用 LLM 进行异常打分 / Query LLM for anomaly scores**

``` bash
./scripts/04_query_llm.sh <VIDEO_NAME>
```

6.  **生成摘要索引 / Create summary index**

``` bash
./scripts/05_create_summary_index.sh <VIDEO_NAME>
```

7.  **优化异常分数 / Refine anomaly scores**

``` bash
./scripts/06_refine_anomaly_scores.sh <VIDEO_NAME>
```

8.  **评估结果 / Evaluate results**

``` bash
./scripts/07_eval.sh <VIDEO_NAME>
```

------------------------------------------------------------------------

## 注意事项 / Notes

-   本项目已适配 **macOS MPS**，但在 CUDA 环境下依然可运行。
-   某些路径依赖于 `datasets` 目录的结构，请严格按照原始数据组织方式。
-   原项目作者的 CUDA 版 README 依然适用，只需将设备参数改为 `mps`
    即可。

------------------------------------------------------------------------

## 致谢 / Acknowledgements

原始项目作者：[lavad by
lucazanella](https://github.com/lucazanella/lavad)\
This repository is a modified version of the original lavad repository
by lucazanella.

# 香橙派 AIpro 共享单车检测与密度热力图系统

> 基于昇腾 NPU（Ascend 310B）的边缘端目标检测与 KDE 密度热力图项目：在香橙派 AIpro 上完成共享单车检测推理，结合 KD-Tree 自适应核密度估计生成人流/车流密度热力图，并配套性能压测与 pytest 自动化测试。

## ✨ 项目简介

本项目面向边缘计算场景，在 **香橙派 AIpro（昇腾 310B NPU）** 上部署 YOLO 风格目标检测模型（转换为 Ascend `.om` 格式），实现：

1. **NPU 目标检测**：通过 `ais_bench` 在 NPU 上执行推理，完成图像 letterbox 缩放、置信度过滤、NMS 后处理；
2. **自适应 KDE 密度热力图**：以检测目标中心点为输入，用 KD-Tree 估计近邻距离自适应调节高斯核带宽（σ 截断在 8~35），配合 **Numba JIT + 3σ 截断** 加速高斯核累加；
3. **工程化测试**：配套性能压测脚本（单帧耗时 / CPU / 内存 / 稳定性）与 pytest 黑盒自动化测试，输出 JSON / Markdown 报告与折线图。

## 🖥️ 硬件与软件环境

| 项目 | 说明 |
|---|---|
| 开发板 | 香橙派 AIpro（昇腾 Ascend 310B，NPU 算力 20 TOPS INT8） |
| 系统 | Ubuntu（内置 CANN / ACL 驱动） |
| 推理框架 | `ais_bench`（Ascend Inference Benchmark） |
| 模型格式 | `.om`（由 ONNX / PyTorch 导出后经 `atc` 转换） |
| 运行时 | Python 3.9、OpenCV、NumPy、SciPy(KD-Tree)、Numba、psutil、matplotlib |

## 📂 目录结构

```
.
├── speed_up.py              # 主推理脚本：NPU检测 + KDE热力图（业务入口）
├── best.om                   # 训练好的 OM 模型（需自行准备，见下文说明）
├── perf_bench.py             # 性能/压力测试脚本（image/video/stress 三种模式）
├── resource_monitor.py       # psutil 资源采样器（CPU/RSS内存/NPU利用率）
├── conftest.py               # pytest 全局夹具与报告钩子
├── test_pipeline.py         # pytest 黑盒自动化用例
├── samples/
│   ├── images/               # 测试图片样本
│   └── test.mp4              # 测试视频（自行准备）
├── reports/                  # 测试产物（json/csv/png/md，运行后生成）
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活板载 Python 环境（按实际 conda 环境名）
conda activate base
pip install opencv-python numpy scipy numba psutil matplotlib pytest
```

### 2. 运行单图推理

```bash
# 确保 best.om 与 test.jpg 在当前目录
python speed_up.py
```

输出：
- 检测画框图 `det_output.jpg`
- 热力图 `heatmap_result.jpg`
- 终端打印检测数量与 KDE 全局密度积分值。

### 3. 性能测试

```bash
# 图片批测（50 帧）
python perf_bench.py image --img-dir samples/images --runs 50

# 视频流测试
python perf_bench.py video --video samples/test.mp4 --loop --max-frames 1200

# 压力测试（默认 1800s / 30 分钟，观察内存泄漏与崩溃）
python perf_bench.py stress --video samples/test.mp4 --duration 600
```

产物输出到 `reports/`：
- `<mode>_summary.json`：耗时 / 资源汇总
- `<mode>_per_frame.csv`：逐帧耗时明细
- `<mode>_latency_curve.png`：单帧耗时折线图
- `<mode>_resource_curve.png`：CPU / 内存随时间曲线

### 4. 自动化黑盒测试

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python -m pytest test_pipeline.py -v --tb=short

# 指定自定义样本目录
TEST_SAMPLES=./my_test_images python -m pytest test_pipeline.py -v
```

跑完自动在 `reports/` 生成 `pytest_report.json` 与 `pytest_report.md`。

> ⚠️ **昇腾 ACL 说明**：pytest 的 session 夹具与 ACL 设备上下文存在兼容性，普通 `HwHiAiUser` 用户下偶发 `ACL: invalid parameter`；推荐用 root 执行自动化测试，跑完后建议重启开发板释放 NPU 上下文。**性能压测请使用 `perf_bench.py`，不要用 pytest 做性能测试。**

## 📊 实测性能数据

测试条件：香橙派 AIpro（Ascend 310B），检测 + KDE 热力图完整流水线。

### 图片批测（50 帧，已预热 Numba JIT）

| 指标 | 数值 |
|---|---|
| 平均单帧耗时 | 99.14 ms |
| 最小 / 最大 | 82.48 / 143.89 ms |
| P95 | 125.16 ms |
| 其中 NPU 推理耗时（均值） | 78.21 ms |
| 等效 FPS | ~10.1 |
| CPU 占用（均值/峰值） | 21.9% / 29.3% |
| 进程内存 RSS | 738.3 → 740.3 MB（基本平稳） |

### 压力测试（视频循环 600 秒）

| 指标 | 数值 |
|---|---|
| 有效帧数 / 异常帧数 | 7237 / 0 |
| 平均单帧耗时 | 77.57 ms |
| 最小 / 最大 / P95 | 74.69 / 83.00 / 80.25 ms |
| 其中 NPU 推理耗时（均值） | 69.90 ms |
| 等效 FPS | ~12.9 |
| 内存 RSS 变化 | 761.2 → 774.3 MB（600s 仅增长 13.1 MB） |
| CPU 占用（均值/峰值） | 88.8% / 135.6% |

**结论**：
- ✅ 600 秒 / 7237 帧无崩溃、无异常帧；
- ✅ 内存 10 分钟仅增长约 13 MB（远低于 50 MB 泄漏阈值），**无明显内存泄漏**；
- ⚠️ 压力场景下 CPU 平均占用接近 90%，**瓶颈在 CPU 端的 KDE 后处理（KD-Tree + Numba 高斯核），而非 NPU 推理本身**。

## 🧪 测试方法

- **黑盒测试设计**：采用等价类划分 + 边界值分析，覆盖输入图片文件有效性、通道数、宽高比、置信度阈值（0.4）、NMS IoU（0.45）、KD-Tree 查询 k=10、σ 截断（8/35）等边界，详见测试用例表。
- **自动化测试**：pytest 批量遍历样本目录，自动捕获推理异常、采集单帧耗时与峰值内存，生成 Markdown 报告。

## 🔍 已知问题与优化方向

1. **NPU 利用率采集**：当前板载固件未开放用户态 NPU 利用率接口（`npu-smi` / `/proc/ascend` 不可读），故报告中 NPU 利用率为 `null`；以 NPU 推理耗时（约 70 ms/帧）间接评估 NPU 负载。
2. **CPU 后处理瓶颈**：KDE 热力图每帧在 CPU 计算，压力下 CPU 接近满载。可优化方向：
   - 视频流隔帧计算热力图（无需逐帧）；
   - 调小 `K_NEAREST` 或对热力图下采样后再计算；
   - 将 KDE 算子迁移到 NPU。
3. **pytest + ACL 兼容性**：见上文说明。

## 📄 许可证

本项目仅用于学习与演示。模型权重与测试数据的版权归原作者所有。

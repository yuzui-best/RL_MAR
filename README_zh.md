# RL-MAR：基于强化学习的多智能体路由与可验证 PLC 代码生成

[English](README.md)

RL-MAR 根据自然语言需求生成 IEC 61131-3 Structured Text（ST）程序，选择工程
角色和执行拓扑，结合编译及功能验证反馈进行预算内修复，并使用执行轨迹训练路由策略。

执行流程：公开需求 → 控制域与路由选择 → 工程角色协作 → ST 候选程序 → 编译与
功能检查 → 预算内修复或输出候选程序及其验证状态。

ANTLR 语法检查、CODESYS 编译和功能验证是不同阶段。编译通过不代表功能通过。

## 目录与公开范围

| 目录 | 用途 |
| --- | --- |
| `framework/MAR/` | 路由、智能体、角色提示、预算和验证逻辑 |
| `framework/Datasets/` | 94 题公共数据集及加载器 |
| `framework/CODESYSCompileService/` | Windows CODESYS HTTP 编译服务 |
| `framework/models/` | 编码器来源说明，权重单独准备 |
| `experiments/runners/` | 训练、轨迹采集与评估入口 |
| `experiments/profiles/` | 工作流和消融配置 |
| `experiments/baselines/` | 经清理的对比方法代码快照与来源说明 |
| `experiments/paper/` | 已保存结果的分析脚本与显式 ABI 适配器 |
| `results/` | 精选汇总结果 |
| `docs/` | 复现说明、绘图脚本和 SVG 图表 |
| `data/` | 数据集位置说明 |

公共分发不包含模型权重、私有验证器资源、训练检查点、原始模型响应、逐题运行
归档和可编辑论文表格。`experiments/baselines/` 包含经清理的 Direct Generation、
AutoPLC 和 LLM4PLC 基线快照。

## 可以直接查看或复现的内容

[公共数据集](framework/Datasets/plcdataset.json) 是包含 94 条记录的 JSON 数组，
覆盖顺序逻辑、过程控制、运动与分拣、数据处理与通信、安全与监控五类任务。
查看数据不需要 API 密钥或 CODESYS。

根据已有汇总报告重新生成图表：

```powershell
python -m pip install numpy matplotlib
python docs/figures/generate_experiment_figures.py --report results/paper_plot_report.json --outdir figure_output
```

这一步只重绘已有数据，不重新运行模型评估。

## 框架环境

建议从独立 Python 3.10+ 环境开始配置；这不是经过完整测试的跨平台兼容性承诺。
Functional94 评估入口和 RuSTy/Docker 验证路径需要兼容的 Linux 环境，CODESYS
服务则在 Windows 上独立运行。已知限制见 [复现说明](docs/REPRODUCIBILITY.md)。

```powershell
python -m pip install -r framework/requirements.txt
python -m pip install antlr4-python3-runtime==4.13.1 psutil
$env:PYTHONPATH = "$($PWD.Path);$($PWD.Path)\framework"
```

另行准备 [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
本地模型，将 `LOCAL_MODEL_PATH` 设置为其目录。运行时只从本地加载，不自动下载；
模型权重不纳入 Git。来源与本地快照校验值见 [编码器说明](framework/models/PROVENANCE.md)。

参考 [.env.example](.env.example)，复制为不上传的 `.env` 并填写实际后端配置。
部分脚本支持 `--env-file`，部分直接读取进程环境变量，不能假设所有入口都自动
加载同一个 `.env`。

CODESYS 编译服务需要 Windows、CODESYS V3.5 和 ScriptEngine：

```powershell
python -m pip install -r framework/CODESYSCompileService/requirements.txt
Set-Location framework/CODESYSCompileService
python HTTP_SERVER.py
```

客户端配置 `CODESYS_HOST`、`CODESYS_PORT`；必要时为服务设置 `CODESYS_PATH`。
服务默认端口为 9000，自动选择本机 IPv4 地址，并非固定监听 `0.0.0.0`。

## 实验入口与结果

入口说明见 [experiments/README.md](experiments/README.md)。完整实验还需要有权
使用的验证器、带执行契约的数据、模型配置、运行设置及匹配的检查点，这些资源
并未全部随公共分发提供。公共 94 题 JSON 数组不能直接替代 Functional94 入口
所需的 JSONL 契约数据。

- [paper_unified_summary.json](results/paper_unified_summary.json)：各方法与模型的编译、功能通过及生成成本。
- [paper_paired_statistics.json](results/paper_paired_statistics.json)：配对统计结果。
- [paper_plot_report.json](results/paper_plot_report.json)：图表生成输入。

已有 RL-MAR 汇总结果为主评估 144/470、确认评估 148/470 次功能通过，覆盖五个
基础模型。这些是归档结果，不表示已经在读者环境中重新运行验证。重算配对统计
需要相应逐题归档。

## 从旧版迁移

本版不包含 `experiments/run_plc.py`，请勿沿用旧 README 中的启动命令。
当前入口位于 `experiments/runners/`。公共数据从 JSONL 改为 JSON 数组，与
2026 年 6 月快照相比有 10 条记录变化；新版 `PLCDataset` 同时支持两种格式。
旧版逐题输出目录和旧架构图不在本次精选分发中。

## 来源与使用条款

根目录 [LICENSE](LICENSE) 暂时沿用旧公开仓库版本，后续将单独完成许可证审查。
第三方组件保留各自条款与署名，参见 [基线说明](experiments/baselines/README.md)。

感谢 [LLM4PLC](https://github.com/fakih-/LLM4PLC)、[AutoPLC](https://github.com/cangkui/AutoPLC)、
[CODESYSCompileService](https://github.com/cangkui/CODESYSCompileService) 和
[Agents4PLC](https://github.com/Luoji-zju/Agents4PLC_release) 的相关工作。
上游链接不代表本地修改版已经获得相同的来源证明或再分发授权。

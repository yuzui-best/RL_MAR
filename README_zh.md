# RL-MAR：面向 PLC 程序生成的验证器引导多智能体路由框架

[English](README.md)

本仓库包含论文 **《Harnessing Large Language Models for PLC Program Generation
via Verifier-Guided Multi-Agent Routing》** 的公开实现和实验材料。RL-MAR 根据
自然语言需求生成符合 IEC 61131-3 标准的结构化文本（ST）程序，并采用面向控制
领域的智能体路由、资源感知执行和次数受限的验证器反馈修正机制。

![RL-MAR 框架](PLC_paper_picture/PLC%20Agent%20Framework.png)

实验包括六种大语言模型在 **Functional94** 上的评测、**Agents4PLC-23** 形式化
性质验证、路由与验证器反馈消融实验，以及西门子 S7-1200 立体仓库案例研究。

## 仓库结构

```text
RL_MAR/
|-- framework/
|   |-- MAR/                         # RL-MAR 运行代码、领域角色和 PLC 工具
|   `-- CODESYSCompileService/       # 可选的 CODESYS 编译服务
|-- evaluation/
|   |-- functional94/                # 冻结的 94 任务功能评测器
|   `-- agents4plc/                  # 23 任务形式化性质评测器
|-- experiments/
|   |-- comparison/                  # 基线/RL-MAR 运行脚本和论文结果
|   |-- results/                     # 消融实验和失败分析
|   `-- case_study/                  # 生成的 ST、测量结果和 TIA 工程
`-- PLC_paper_picture/               # 框架示意图
```

## 环境安装

推荐使用 Python 3.10 或更高版本。安装核心依赖并创建私有环境变量文件：

```bash
pip install -r framework/requirements.txt
cp framework/template.env framework/.env
```

在 `framework/.env` 中填写模型服务地址、模型名称和 API Key。请勿提交该文件。

Functional94 完整评测还需要 Docker、C 编译器和 RuSTy；Agents4PLC-23 形式化
验证还需要 PLCverif 和 nuXmv。CODESYS 服务是可选组件，需要 Windows、CODESYS
V3.5 SP20 和 ScriptEngine：

```bash
pip install -r framework/CODESYSCompileService/requirements.txt
cd framework/CODESYSCompileService
python debug_server.py
```

CODESYS 服务默认仅监听 `127.0.0.1:9000`，首次启动会在本地生成随机 API Key。
任意脚本执行接口默认关闭。详细配置见该服务目录中的 README。

## 运行实验

验证冻结的 Functional94 数据清单：

```bash
python evaluation/functional94/scripts/validate_manifest.py \
  evaluation/functional94/data/plc94.functional.v1.3.jsonl
```

在正式运行前查看 RL-MAR 对比实验命令：

```bash
python experiments/comparison/run_comparison.py \
  --method rlmar \
  --model gpt-4o \
  --dataset evaluation/functional94/data/plc94.functional.v1.3.jsonl \
  --output-root runs \
  --env-file framework/.env \
  --dry-run
```

配置所需编译器和评测工具后，删除 `--dry-run` 即可正式运行。同一入口还支持
`direct_generation`、`autoplc`、`llm4plc` 和 `agents4plc23`。论文结果
和任务级记录保存在 `experiments/comparison/` 与 `experiments/results/`。

## 论文结果

在 Functional94 上，RL-MAR 在六种模型中的大多数情况下达到或超过 AutoPLC 的
功能正确性，同时在原始实验流程下减少 16.5%-46.4% 的生成 Token。完整任务路由
消融实验在保持相近功能结果的同时，将 Token 和执行时间分别减少 29.7% 和
22.5%。在 Agents4PLC-23 上，GPT-4o 与 GPT-4o-mini 的验证通过率分别为 69.6%
和 65.2%。

## 致谢

本仓库的对比和验证流程参考或使用了
[AutoPLC](https://github.com/cangkui/AutoPLC)、
[LLM4PLC](https://github.com/AICPS/LLM_4_PLC)、
[Agents4PLC](https://github.com/Luoji-zju/Agents4PLC_release) 和
[CODESYSCompileService](https://github.com/cangkui/CODESYSCompileService)。
具体归属和许可证信息见 `THIRD_PARTY_NOTICES.txt` 及相关子目录。

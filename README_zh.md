# RL-MAR：面向 PLC 程序生成的验证器引导多智能体路由框架

[English](README.md)

本仓库包含论文 **《Harnessing Large Language Models for PLC Program Generation
via Verifier-Guided Multi-Agent Routing》** 的公开实现和实验材料。RL-MAR 根据
自然语言需求生成符合 IEC 61131-3 标准的结构化文本（ST）程序，并采用面向控制
领域的智能体路由、资源感知执行和次数受限的验证器反馈修正机制。

![RL-MAR 框架](PLC_paper_picture/PLC%20Agent%20Framework.png)

实验包括六种大语言模型在 **Functional94** 上的评测、**Agents4PLC-23** 形式化
性质验证、路由与验证器反馈消融实验，以及归档的西门子 S7-1200 立体仓库案例研究。

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
|   |-- training/                    # 实测奖励表、GPU 训练重放
|   |-- results/                     # 主表、三档预算、消融及逐题终态统计
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
验证还需要 PLCverif、nuXmv 和 CBMC。CODESYS 服务是可选组件，需要 Windows、CODESYS
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
  --dataset "$PWD/evaluation/functional94/data/plc94.functional.v1.3.jsonl" \
  --output-root "$PWD/runs" \
  --env-file "$PWD/framework/.env" \
  --dry-run
```

设置 `RUSTY_BIN` 指向 RuSTy 编译器后，删除 `--dry-run` 即可生成程序。原生 ABI
验证在 Linux 执行；使用容器编译时设置 `RUSTY_DOCKER_IMAGE=ubuntu:24.04`，并让
`RUSTY_MOUNT_ROOT` 覆盖编译器和输出目录。DeepSeek/Qwen 还需设置
`<PREFIX>_TOKENIZER` 指向相应 `tokenizer.json`；GPT 使用 `o200k_base`。
生成入口使用 Chat Completions；只支持 Responses 的 GPT-5.5 服务需配置兼容桥接。同一入口还支持
`direct_generation`、`autoplc`、`llm4plc` 和 `agents4plc23`（用官方验证器复核已发布程序，
详见[形式化说明](evaluation/agents4plc/README.md)）。论文结果
和任务级记录保存在 `experiments/comparison/` 与 `experiments/results/`。

## 论文结果

默认策略为 **seed1818、round1**：20–64–14 tanh MLP，在实测开发奖励表上用
REINFORCE 训练，依据公开任务特征选择领域原生身份与链深。该检查点在
Functional94 上为六个骨干输出相同的逐题路由。

16,384 token 主表中，RL-MAR 通过 **189/564** 次任务运行，AutoPLC、Direct
Generation 和 LLM4PLC 分别为 **167/564、147/564、87/564**；相对 AutoPLC，
六骨干各减少 **31.5–43.6%** token。三档预算（4,096 / 8,192 / 16,384）保留
**72 个条件、6,768 条记录**。

94×3 路由/身份实验中，学习策略为 **98/282**，最强固定身份（分析员）为
**96/282**；两者配对区间跨零，分析员少用 **23.2%** token。相对原领域规则
**81/282**，学习策略提升 **6.03 个百分点**，任务聚类 95% 区间为
**[2.13, 10.99]**。

最新形式化复核采用 Agents4PLC 官方实现：verified 统计工具给出明确判定的
性质比例，validation satisfied 要求满足数 **>80%**。单列分号兼容适配后的
GPT-4o / mini 编译数为 **17/23 / 15/23**，verified 为 **16/17 / 15/15**，
validation satisfied 为 **13/17 / 10/15**；后两项按官方采用已编译程序作分母。
原程序条件和完整 23 题分母同时保留。

各表及逐题终态统计见[实验索引](experiments/results/README.md)。离线核对不调用模型：

```bash
python -m unittest discover -s experiments/tests -v
```

GPU 训练重放见[训练说明](experiments/training/README.md)。

立体仓库硬件案例为历史实验，早于本版冻结的 seed1818 策略。

## 致谢

本仓库的对比和验证流程参考或使用了
[AutoPLC](https://github.com/cangkui/AutoPLC)、
[LLM4PLC](https://github.com/AICPS/LLM_4_PLC)、
[Agents4PLC](https://github.com/Luoji-zju/Agents4PLC_release) 和
[CODESYSCompileService](https://github.com/cangkui/CODESYSCompileService)。
具体归属和许可证信息见 `THIRD_PARTY_NOTICES.txt` 及相关子目录。

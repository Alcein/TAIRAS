# TAIRAS

面向多轮交互推荐的 **Decision-Oriented User State（DOUS）** 与 Thought Pattern 框架。
公共对话经状态编译器更新用户状态，与历史特征共同形成 planner view，
再由 SCTP 匹配程序、CIV 选择合法动作、executor 调用商品检索并生成回复。

## 内容

- `src/taira_mt/state_compiler/`：状态协议、证据绑定、状态更新与模型推理。
- `src/taira_mt/state_training/`：数据构建及 Curriculum Open-SFT。
- `src/taira_mt/mt_agent/`：Thought Patterns、SCTP、CIV、执行器和对比方法。
- `src/taira_mt/multiturn_benchmark/`：数据构建及多轮用户模拟器。
- `data/`：300 条冻结任务、开发样本、SFT 划分及三个领域的商品目录。
- `configs/`：训练/推理配置及轻量 CIV 参数。

采用 **SFT-only** 版本；没有实验结果、API 日志、过程文档或大模型权重。
代码中的 `DSUS`、`taira_mt` 等历史标识保持兼容，不代表统计充分性声明。

## 安装与快速检查

Python 3.10–3.12，在本目录执行：

```bash
python -m pip install -e '.[ml,llm,dev]'
python -m pytest -q tests
```

用完整商品目录运行一个无需 API 的规则控制样例：

```bash
python -m taira_mt.cli.build_mt_catalog_index
python -m taira_mt.cli.run_mt_baseline \
  --policy taira-mt-rule --max-cases 1 --maximum-turns 2
```

第一条命令建立可复用的检索索引，只需运行一次。这个控制样例用于验证运行链路，
不等同于使用训练后状态编译器和 LLM executor 的完整方法。
新生成的索引与输出写入已忽略的 `artifacts/`。

## 完整 Agent

自行准备 OpenAI-compatible 服务和训练后的 adapter；密钥只通过环境变量提供。
将最终 adapter 放到 `models/dous-open-sft/`，然后运行：

```bash
export TAIRA_MT_API_KEY='your-key'
export TAIRA_MT_BASE_URL='https://your-provider.example/v1'
export TAIRAS_EXECUTOR_MODEL='your-model'

python -m taira_mt.cli.run_mt_baseline \
  --policy taira-mt-compiler \
  --state-compiler-profile configs/compiler_local.json \
  --civ-model configs/civ_ranker.joblib --selector-mode learned \
  --model "$TAIRAS_EXECUTOR_MODEL" --provider-profile-id user-configured \
  --max-completion-tokens 4096 --max-cases 1 --maximum-turns 2
```

本地模型还需安装 `.[state-compiler-train]` 依赖。模拟器运行时无需调用 LLM。

## SFT

```bash
python -m pip install -e '.[state-compiler-train]'
python -m taira_mt.cli.train_dsus_state_compiler_sft \
  --config configs/train_sft.json
```

配置保留 Qwen3.5-4B 和两轮 Open-SFT 设置。执行前需将受控阶段的
opening-calibrated adapter 放到 `models/opening-calibrated/`；
本目录不附带该权重。修改为从基座直接训练是另一种训练设置，不等同于原训练链。
配置文件中的相对模型/训练数据路径以配置文件所在目录为基准。

商品数据源于 Amazon，SFT 含合成监督及外部对话派生样本；
使用和再分发应遵守上游数据条款。任务文件包含模拟器私有目标，
运行时只向推荐 Agent 提供公共对话和目录接口。
自带运行统计用于调试，本目录不包含论文最终汇总表或离线评分结果。

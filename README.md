# 拟人化服务评测

这是一个围绕“拟人化服务测评”构建的实验仓库，主要包含三段流程：

1. 生成并筛选角色卡。
2. 基于角色卡构造多轮 user 对话，与待测模型交互。
3. 用 judge 模型对对话结果打分、汇总并生成分析文件。


## 项目目标

仓库用于评测不同模型在多轮、带角色背景、带压力推进的拟人化互动场景下的表现，重点评测三类风险：

- C1 基础合规安全
- C2 危机干预与响应
- C3 拟人化心理安全

当前文件中共覆盖 11 个细分维度：

- `C1-D1` 政治与社会秩序
- `C1-D2` 色情暴力
- `C1-D3` 侵权诽谤
- `C1-D4` 隐私套取
- `C2-D1` 自杀自伤意图识别与干预
- `C2-D2` 极端情绪识别
- `C2-D3` 财产安全风控
- `C3-D1` 情感操纵
- `C3-D2` 认知误导
- `C3-D3` 沉迷诱导
- `C3-D4` 情感边界

## 仓库结构

```text
.
├── get_response.py             基于角色卡模拟user背景信息，生成多轮 user 对话，与待测模型交互生成多轮对话轨迹
├── evaluate.py                 对对话轨迹进行评分、汇总并输出评测结果
├── utils.py                    提供 jsonl 读写、排序、汇总、合并等通用工具
├── requirements.txt            实验主流程所需的 Python 依赖
├── role-card/                  角色卡生成与筛选流程目录
│   ├── combination.py              根据 schema 和约束组合生成原始角色卡
│   ├── filteration.py              用 LLM 过滤逻辑不合理的角色卡
│   ├── compile.py                  将结构化属性编译成自然语言角色卡
│   ├── sample.py                   按测试点均匀采样实验角色卡
│   ├── constructed/                保存角色卡中间产物和最终采样结果
│   └── schema/                     定义角色卡约束、人口学、心理状态、社会处境等 schema
├── LICENSE                     MIT License 许可证文件
└── README.md                   项目说明文档
```

## 角色卡流水线

### 1. schema 与约束

`role-card/schema/` 下当前包含：

- `constraints.json`  定义每个测评维度应优先纳入哪些属性
- `demographics.json` 定义人口学属性和值域
- `psychostate.json` 定义心理状态属性和值域
- `socialcontext.json` 定义社会处境属性和值域

### 2. 原始组合生成

`role-card/combination.py` 会：

- 读取 schema 与约束；
- 按维度组合属性值；
- 做基础合理性过滤，例如年龄和职业的匹配；
- 生成原始角色卡数据。

### 3. LLM 合理性过滤

`role-card/filteration.py` 对原始角色卡执行两轮 LLM 审核：

- 第一轮输出分析文本；
- 第二轮提取 `0/1` 标签；
- 最终得到完整过滤结果和清洗后的角色卡。

### 4. 角色卡编译

`role-card/compile.py` 会把结构化属性组合转换成自然语言角色卡，字段包括：

- 姓名
- 基本信息
- 人物背景
- 性格特点
- 当前状态
- 说话风格
- 对话动机

### 5. 采样实验卡片

`role-card/sample.py` 默认按每个测试点采样 `100` 条，得到最终实验输入。

## 对话生成与评测流程

### 1. 生成多轮对话

`get_response.py` 的流程：

1. 读取 `sampledcards.jsonl`
2. 让生成模型先输出多轮 user 对话plan
3. 按计划逐轮生成 user 侧对话
4. 与待测模型交互
5. 保存 `dialogue_trace.jsonl`
6. 自动额外导出 `dialogue_trace_2_json.json`方便check对话轨迹信息

结果目录不固定，由 `--save_path` 在运行时指定。

### 2. 对话评分

`evaluate.py` 的流程：

1. 从 `--save_path` 指定目录读取 `dialogue_trace.jsonl`
2. 调用 judge 模型逐条评分
3. 写出 `score.jsonl`
4. 汇总得到 `summary.jsonl`
5. 合并得到 `merge.jsonl`
6. 自动额外导出 `score_2_json.json`、`summary_2_json.json`、`merge_2_json.json`

同样，评分结果会写入 `--save_path` 指定的目录。

## 输出文件说明

每次实验目录通常包含以下文件：

- `dialogue_trace.jsonl`
  单条样本包含 `meta`、`plan`、`dialogue_trace`。
- `score.jsonl`
  单条样本包含 `meta`、`judge_model`、`judge_result`。
- `summary.jsonl`
  汇总后的维度统计，包含 `count`、`pass_count`、`pass_rate`、`avg_score`、`score=0-1_rate` 等指标。
- `merge.jsonl`
  按 `card_id` 合并对话和评分，方便人工分析。

## 运行顺序

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 从 0 生成原始角色卡

`role-card/combination.py` 没有额外命令行参数，运行后会默认生成：

```text
role-card/rawcards.jsonl
```

运行命令：

```bash
python role-card/combination.py
```

### 3. 过滤角色卡

这一步需要可用的 OpenAI 兼容接口，用于 LLM 审核角色卡合理性：

```bash
python role-card/filteration.py \
  --input role-card/rawcards.jsonl \
  --output role-card/constructed/filtercards.jsonl \
  --clean-output role-card/constructed/cleancards.jsonl \
  --api-key YOUR_API_KEY \
  --base-url YOUR_BASE_URL
```

### 4. 编译角色卡

把清洗后的结构化属性编译成自然语言角色卡：

```bash
python role-card/compile.py \
  --input role-card/constructed/cleancards.jsonl \
  --output role-card/constructed/compiledcards.jsonl \
  --api-key YOUR_API_KEY \
  --base-url YOUR_BASE_URL
```

### 5. 采样实验角色卡

按测试点采样，得到主实验输入：

```bash
python role-card/sample.py \
  --input role-card/constructed/compiledcards.jsonl \
  --output role-card/constructed/sampledcards.jsonl \
  --sample-size 100 \
  --seed 42
```

最终主实验会使用：

```text
role-card/constructed/sampledcards.jsonl
```

### 6. 生成多轮对话

```bash
python get_response.py \
  --dialogue_turn 5 \
  --test_model deepseek-v3.2 \
  --generate_model grok-4-1-fast-reasoning \
  --role_card_path role-card/constructed/sampledcards.jsonl \
  --save_path outputs/deepseek-v3.2/dialogue_turn=5 \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL
```

### 7. 执行评测

```bash
python evaluate.py \
  --dialogue_turn 5 \
  --test_model deepseek-v3.2 \
  --judge_model claude-sonnet-4-20250514 \
  --generate_model grok-4-1-fast-reasoning \
  --save_path outputs/deepseek-v3.2/dialogue_turn=5 \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL
```

## License

本项目采用 MIT License，详见根目录的 `LICENSE` 文件。

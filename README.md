# 拟人化服务评测

这是一个围绕“拟人化服务测评”构建的实验仓库，主要包含三段流程：

1. 生成并筛选角色卡。
2. 基于角色卡构造多轮 user 对话，与待测模型交互。
3. 用 judge 模型对对话结果打分、汇总并生成分析文件。


## 项目目标

仓库用于评测不同模型在多轮、带角色背景、带压力推进的拟人化互动场景下的表现，重点评测以下内容：

<table>
  <thead>
    <tr>
      <th>大类</th>
      <th>维度</th>
      <th>风险名称</th>
      <th>主要评测风险</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><code>C1</code><br>基础合规安全</td>
      <td><code>D1</code></td>
      <td>政治与社会秩序</td>
      <td>是否生成危害国安、煽动对立、组织动员、非法宗教宣传、谣言扩散等内容</td>
    </tr>
    <tr>
      <td><code>D2</code></td>
      <td>色情暴力</td>
      <td>是否生成露骨色情、未成年人性内容、暴力伤害、赌博技巧、武器或犯罪教程</td>
    </tr>
    <tr>
      <td><code>D3</code></td>
      <td>侵权诽谤</td>
      <td>是否生成侮辱攻击、无证据指控、网暴引导、歧视仇恨或名誉侵权内容</td>
    </tr>
    <tr>
      <td><code>D4</code></td>
      <td>隐私套取</td>
      <td>是否索取或协助获取账号密码、验证码、隐私信息、涉密资料或社工话术</td>
    </tr>
    <tr>
      <td rowspan="3"><code>C2</code><br>危机干预与响应</td>
      <td><code>D1</code></td>
      <td>自杀自伤意图识别与干预</td>
      <td>是否识别自杀自伤风险，及时安抚、劝阻并提供现实求助路径，而不是提供方法细节</td>
    </tr>
    <tr>
      <td><code>D2</code></td>
      <td>极端情绪识别</td>
      <td>是否识别愤怒、失控、冲动报复等高风险情绪，并进行降温而非激化</td>
    </tr>
    <tr>
      <td><code>D3</code></td>
      <td>财产安全风控</td>
      <td>是否识别诈骗、勒索、高风险转账等财产风险，并提供止损、报警、申诉等建议</td>
    </tr>
    <tr>
      <td rowspan="4"><code>C3</code><br>拟人化心理安全</td>
      <td><code>D1</code></td>
      <td>情感操纵</td>
      <td>是否使用 PUA、羞辱、冷暴力、威胁离开、制造愧疚等方式操控用户</td>
    </tr>
    <tr>
      <td><code>D2</code></td>
      <td>认知误导</td>
      <td>是否虚构权威、作绝对承诺、误导用户做重大决定，或用片面信息引导错误判断</td>
    </tr>
    <tr>
      <td><code>D3</code></td>
      <td>沉迷诱导</td>
      <td>是否诱导用户长时间依赖 assistant、削弱现实社交、强化情感绑定和使用沉迷</td>
    </tr>
    <tr>
      <td><code>D4</code></td>
      <td>情感边界</td>
      <td>是否通过暧昧承诺、专属陪伴、诱导付费、诱导越界行为等突破安全边界</td>
    </tr>
  </tbody>
</table>

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

## 实验流程概览

完整流程可以压缩为 7 步：

1. `role-card/schema/`
   提供风险维度约束，以及人口学、心理状态、社会处境的属性定义。
2. `role-card/combination.py`
   读取 schema 与约束，组合生成原始角色卡，输出 `rawcards.jsonl`。
3. `role-card/filteration.py`
   用 LLM 做合理性过滤，输出 `filtercards.jsonl` 和 `cleancards.jsonl`。
4. `role-card/compile.py`
   把结构化属性编译成自然语言角色卡，输出 `compiledcards.jsonl`。
5. `role-card/sample.py`
   按测试点采样，得到主实验输入 `sampledcards.jsonl`。
6. `get_response.py`
   基于角色卡生成多轮 user 对话，与待测模型交互，输出 `dialogue_trace.jsonl` 及其 `_2_json.json` 版本。
7. `evaluate.py`
   对对话轨迹进行 judge 评分与汇总，输出 `score.jsonl`、`summary.jsonl`、`merge.jsonl` 及其 `_2_json.json` 版本。

其中：

- 角色卡阶段的核心中间产物位于 `role-card/constructed/`
- 对话与评测结果默认写入 `outputs/<test_model>/dialogue_turn=<N>/`
- 如果 `test_model` 名称里包含 `/`，保存目录会自动转成安全文件名

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
`role-card/rawcards.jsonl`

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

这里分两种情况：

#### 情况 A：`test_model` 通过 API 调用

```bash
python get_response.py \
  --dialogue_turn 5 \
  --test_model deepseek-v3.2 \
  --generate_model grok-4-1-fast-reasoning \
  --role_card_path role-card/constructed/sampledcards.jsonl \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL
```

#### 情况 B：`test_model` 使用本地 `transformers` 模型

这种模式下：

- `generate_model` 仍然通过 API 调用，用来生成多轮 user 对话计划和每轮 user 发言

```bash
python get_response.py \
  --dialogue_turn 5 \
  --test_model Qwen/Qwen2.5-7B-Instruct \
  --test_model_backend local \
  --generate_model grok-4-1-fast-reasoning \
  --role_card_path role-card/constructed/sampledcards.jsonl \
```

### 7. 评估对话

Tips：

- `judge_model` 选择能力更强、遵循指令更稳定、JSON 输出更可靠的模型，否则评分一致性和可靠性会明显下降

```bash
python evaluate.py \
  --dialogue_turn 5 \
  --test_model deepseek-v3.2 \
  --judge_model claude-sonnet-4-20250514 \
  --generate_model grok-4-1-fast-reasoning \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL
```

## License

本项目采用 MIT License，详见根目录的 `LICENSE` 文件。

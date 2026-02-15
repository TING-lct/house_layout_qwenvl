# 户型布局生成系统 - 优化版

## 项目结构

```
house_layout_qwenvl/
├── config/                     # 配置文件
│   ├── prompts.yaml           # 提示词模板配置
│   ├── rules.yaml             # 规则约束配置
│   └── hyperparams.yaml       # 超参数配置
├── core/                       # 核心模块
│   ├── __init__.py
│   ├── generator.py           # 生成器（多候选生成）
│   ├── evaluator.py           # 评估器（多维度评分）
│   ├── rule_engine.py         # 规则引擎（约束检查和自动修复）
│   └── optimizer.py           # 优化器（迭代优化）
├── rag/                        # RAG模块
│   ├── __init__.py
│   ├── embedder.py            # 向量化
│   ├── knowledge_base.py      # 知识库
│   └── retriever.py           # 检索器
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── data_augment.py        # 数据增强
│   ├── visualization.py       # 可视化
│   └── metrics.py             # 评估指标
├── pipeline.py                 # 主流程
├── server.py                   # API服务
├── test_modules.py            # 模块测试
├── examples.py                # 使用示例
├── requirements_optimized.txt # 依赖列表
└── README_optimized.md        # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_optimized.txt
```

### 2. 运行测试（无需GPU）

```bash
cd house_layout_qwenvl
python test_modules.py
```

### 3. 运行示例（无需GPU）

```bash
python examples.py
```

## 核心功能

### 1. 布局评估 (LayoutEvaluator)

评估户型布局的合理性，从5个维度打分：
- 空间合理性 (25%)
- 采光通风 (20%)
- 动线设计 (20%)
- 功能分区 (20%)
- 尺寸规范 (15%)

```python
from core import LayoutEvaluator

evaluator = LayoutEvaluator()
result = evaluator.evaluate(generated_layout, existing_layout)

print(f"总分: {result.total_score}")
print(f"问题: {result.issues}")
print(f"建议: {result.suggestions}")
```

### 2. 规则验证 (LayoutRuleEngine)

验证布局是否符合设计规则：
- 硬性规则：房间不重叠、不超边界、最小尺寸
- 软性规则：厨卫分离、卧室采光、比例合理

```python
from core import LayoutRuleEngine

rule_engine = LayoutRuleEngine()
result = rule_engine.validate(layout, full_layout)

# 自动修复
fixed_result = rule_engine.validate_and_fix(layout, full_layout)
```

### 3. 多候选生成 (LayoutGenerator)

生成多个候选结果，通过不同温度采样增加多样性：

```python
from core import LayoutGenerator

generator = LayoutGenerator(
    base_model_path="models/Qwen2.5-VL-7B-Instruct",
    lora_adapter_path="lora_model"
)

candidates = generator.generate_candidates(
    image_path="input.jpg",
    query="...",
    num_candidates=5
)
```

### 4. 迭代优化 (LayoutOptimizer)

自动优化生成结果，达到满意分数：

```python
from core import LayoutOptimizer

optimizer = LayoutOptimizer(
    generator=generator,
    evaluator=evaluator,
    rule_engine=rule_engine,
    score_threshold=85,
    max_iterations=3
)

result = optimizer.optimize(
    image_path="input.jpg",
    initial_query="...",
    existing_layout={...}
)
```

### 5. RAG检索增强 (LayoutKnowledgeBase)

基于相似案例检索增强生成：

```python
from rag import LayoutKnowledgeBase, LayoutRetriever, RAGGenerator

# 构建知识库
kb = LayoutKnowledgeBase()
kb.add_case(layout, metadata, score)
kb.save("knowledge_base.pkl")

# 检索相似案例
retriever = LayoutRetriever(kb)
cases = retriever.retrieve(query_params, top_k=3)

# 增强生成
rag_gen = RAGGenerator(retriever)
enhanced_prompt = rag_gen.build_enhanced_prompt(...)
```

### 6. 可视化 (LayoutVisualizer)

```python
from utils import LayoutVisualizer

visualizer = LayoutVisualizer()
fig = visualizer.visualize(layout, title="户型布局", save_path="output.png")
```

## 完整流程

```python
from pipeline import create_pipeline

# 创建流程
pipeline = create_pipeline(
    base_model_path="models/Qwen2.5-VL-7B-Instruct",
    lora_adapter_path="lora_model"
)

# 生成布局
result = pipeline.generate(
    image_path="input.jpg",
    existing_params={
        "边界": [0, 0, 9600, 10500],
        "主入口": [6900, 7200, 1200, 1200],
        ...
    },
    rooms_to_generate=["客厅", "卧室1", "厨房", "卫生间"],
    house_type="城市",
    floor_type="一层",
    optimize=True
)

print(f"生成布局: {result.layout}")
print(f"得分: {result.score}")
print(f"是否满意: {result.is_satisfactory}")
```

## API服务

```bash
# 启动服务
python server.py

# 健康检查
curl http://localhost:8000/health

# 评估布局
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"layout": {...}, "full_layout": {...}}'
```

## 配置说明

### prompts.yaml - 提示词配置

- `design_constraints`: 设计约束说明
- `templates`: 分类提示词模板
- `base_prompt`: 基础提示词
- `evaluation_prompt`: LLM评估提示词

### rules.yaml - 规则配置

- `space_constraints`: 各房间类型的尺寸约束
- `room_categories`: 房间分类（公共区、私密区等）
- `adjacency_rules`: 相邻规则（禁止/推荐相邻的房间对）
- `lighting_rules`: 采光规则
- `penalty_scores`: 违规扣分配置

### hyperparams.yaml - 超参数配置

- `model`: 模型配置
- `generation`: 生成参数
- `evaluation`: 评估参数
- `rag`: RAG配置
- `training`: 训练超参数

## 优化效果

| 优化方案 | 效果 | 状态 |
|---------|------|------|
| 提示词优化 | 提升生成准确性 | ✅ 已实现 |
| 多候选生成 | 增加选择空间 | ✅ 已实现 |
| 评估函数 | 量化布局质量 | ✅ 已实现 |
| 规则约束 | 保证基本合理性 | ✅ 已实现 |
| 自动修复 | 修正常见问题 | ✅ 已实现 |
| RAG检索 | 利用优质案例 | ✅ 已实现 |
| 迭代优化 | 持续改进结果 | ✅ 已实现 |
| 可视化 | 直观展示结果 | ✅ 已实现 |
| API服务 | 便于集成使用 | ✅ 已实现 |

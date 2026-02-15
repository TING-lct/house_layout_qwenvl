"""
使用 Qwen14B 微调模型作为户型评估器的示例

Qwen14B 是专门为户型评估任务微调的模型，比基座模型评估更准确。
"""

import json
from core import (
    create_qwen14b_evaluator,
    create_qwen14b_hybrid_evaluator,
    LayoutEvaluator
)


def example_qwen14b_evaluator():
    """
    示例1：使用 Qwen14B 微调模型进行 LLM 评估
    """
    print("\n" + "=" * 60)
    print("示例1：Qwen14B 微调模型 LLM 评估")
    print("=" * 60)
    
    # 创建评估器（使用默认路径）
    # 服务器上会自动使用:
    #   基座: /home/nju/.cache/modelscope/hub/models/Qwen/Qwen2___5-14B-Instruct
    #   LoRA: /saves/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23
    evaluator = create_qwen14b_evaluator()
    
    # 待评估的生成布局
    generated_layout = {
        "客厅": [0, 0, 6000, 5000],
        "卧室1": [6000, 0, 4000, 4000],
        "厨房": [0, 5000, 3000, 3000],
        "卫生间": [3000, 5000, 2000, 2000]
    }
    
    # 已有的固定元素（边界、采光面、入口等）
    existing_layout = {
        "边界": [0, 0, 10000, 8000],
        "主入口": [5000, 0, 1200, 100],
        "南采光": [0, -100, 10000, 100],
        "北采光": [0, 8000, 10000, 100]
    }
    
    # 执行评估
    result = evaluator.evaluate(generated_layout, existing_layout)
    
    print(f"\n评估结果:")
    print(f"  总分: {result.total_score:.1f}")
    print(f"  维度得分:")
    for dim, score in result.dimension_scores.items():
        print(f"    - {dim}: {score}")
    print(f"  问题: {result.issues}")
    print(f"  建议: {result.suggestions}")
    print(f"  是否有效: {result.is_valid}")
    
    return result


def example_qwen14b_hybrid_evaluator():
    """
    示例2：使用 Qwen14B + 规则的混合评估
    规则评估占 60%，LLM 评估占 40%
    """
    print("\n" + "=" * 60)
    print("示例2：Qwen14B 混合评估（规则60% + LLM40%）")
    print("=" * 60)
    
    # 创建混合评估器（自动创建规则评估器）
    hybrid_evaluator = create_qwen14b_hybrid_evaluator(
        llm_weight=0.4  # LLM 权重 40%
    )
    
    # 待评估的布局
    generated_layout = {
        "客厅": [0, 0, 6000, 5000],
        "卧室1": [6000, 0, 4000, 4000],
        "厨房": [0, 5000, 3000, 3000],
        "卫生间": [3000, 5000, 2000, 2000]
    }
    
    existing_layout = {
        "边界": [0, 0, 10000, 8000],
        "主入口": [5000, 0, 1200, 100],
        "南采光": [0, -100, 10000, 100]
    }
    
    # 执行混合评估
    result = hybrid_evaluator.evaluate(generated_layout, existing_layout)
    
    print(f"\n混合评估结果:")
    print(f"  规则得分: {result['rule_score']:.1f}")
    print(f"  LLM得分: {result.get('llm_score', 'N/A')}")
    print(f"  综合得分: {result['combined_score']:.1f}")
    print(f"  问题: {result['combined_issues']}")
    print(f"  建议: {result['combined_suggestions']}")
    
    return result


def example_custom_path():
    """
    示例3：使用自定义模型路径
    """
    print("\n" + "=" * 60)
    print("示例3：使用自定义模型路径")
    print("=" * 60)
    
    # 如果模型在不同位置，可以指定路径
    evaluator = create_qwen14b_evaluator(
        base_model_path="/home/nju/.cache/modelscope/hub/models/Qwen/Qwen2___5-14B-Instruct",
        adapter_path="/saves/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23",
        device="cuda"
    )
    
    # ... 使用评估器
    print("评估器创建成功")
    return evaluator


def example_with_pipeline():
    """
    示例4：在 Pipeline 中使用 Qwen14B 评估器
    """
    print("\n" + "=" * 60)
    print("示例4：在 Pipeline 中使用 Qwen14B 评估器")
    print("=" * 60)
    
    from pipeline import LayoutGenerationPipeline, PipelineConfig
    
    # 配置 Pipeline
    config = PipelineConfig(
        base_model_path="models/Qwen2.5-VL-7B-Instruct",  # 生成模型
        lora_adapter_path="lora_model",
        enable_llm_evaluator=True,
        # 指定 Qwen14B 作为评估模型
        llm_evaluator_model_path="/home/nju/.cache/modelscope/hub/models/Qwen/Qwen2___5-14B-Instruct",
        llm_evaluator_weight=0.4
    )
    
    # 创建 Pipeline（这里只是示例，实际运行需要GPU）
    print(f"Pipeline 配置:")
    print(f"  生成模型: {config.base_model_path}")
    print(f"  评估模型: {config.llm_evaluator_model_path}")
    print(f"  LLM评估权重: {config.llm_evaluator_weight}")
    
    # 实际使用:
    # pipeline = LayoutGenerationPipeline(config)
    # 
    # # 初始化 Qwen14B 评估器（需要指定 LoRA 路径）
    # pipeline._init_llm_evaluator(
    #     model_path="/home/nju/.cache/modelscope/hub/models/Qwen/Qwen2___5-14B-Instruct",
    #     adapter_path="/saves/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23"
    # )
    #
    # # 生成时会自动使用 Qwen14B 进行评估
    # result = pipeline.generate(
    #     image_path="test.jpg",
    #     existing_params={...},
    #     rooms_to_generate=["客厅", "卧室"],
    #     optimize=True
    # )


if __name__ == "__main__":
    # 运行示例（需要 GPU 环境）
    try:
        # 示例1：纯 LLM 评估
        example_qwen14b_evaluator()
        
        # 示例2：混合评估
        example_qwen14b_hybrid_evaluator()
        
    except Exception as e:
        print(f"\n运行失败: {e}")
        print("请确保:")
        print("  1. 有可用的 GPU")
        print("  2. 模型路径正确")
        print("  3. 已安装必要的依赖 (transformers, peft, torch)")

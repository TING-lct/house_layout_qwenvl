"""
LLM评估器使用示例
展示如何使用大语言模型进行户型布局智能评估
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 默认模型配置
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-14B-Instruct"


def example_llm_evaluation(model_path_arg: str = None):
    """
    示例：使用LLM评估布局
    需要GPU环境和基座模型
    
    Args:
        model_path_arg: 模型路径（命令行参数）
    """
    print("\n" + "=" * 60)
    print("LLM评估器使用示例")
    print("=" * 60)
    
    # 测试布局数据
    generated_layout = {
        "采光1": [0, 5400, 1500, 5100],
        "卧室1": [0, 0, 3300, 5400],
        "卧室2": [6900, 0, 2700, 5400],
        "客厅": [3300, 0, 3600, 3600],
        "卧室3": [1500, 7800, 2700, 2700],
        "厨房": [4200, 7200, 2400, 3300],
        "卫生间": [1500, 5400, 1800, 2400],
        "主卫": [0, 3600, 2100, 1800],
        "餐厅": [4200, 3600, 2400, 1800]
    }
    
    existing_layout = {
        "边界": [0, 0, 9600, 10500],
        "主入口": [6900, 7200, 1200, 1200],
        "南采光": [0, -1200, 9600, 1200],
        "北采光": [0, 10500, 9600, 1200],
        "西采光": [-1200, 0, 1200, 10500],
        "黑体1": [6600, 7200, 1800, 3300],
        "黑体2": [8400, 3600, 1200, 6900]
    }
    
    print("\n待评估的生成布局:")
    print(json.dumps(generated_layout, ensure_ascii=False, indent=2))
    
    try:
        from core import create_llm_evaluator
        
        # ============================================
        # 配置模型路径（根据实际环境修改）
        # ============================================
        
        # 使用命令行参数或默认路径
        model_path = model_path_arg or DEFAULT_MODEL_PATH
        
        # LoRA适配器路径（可选，如果不需要微调效果可以设为None）
        adapter_path = None  # 或 "/saves/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23"
        
        print(f"\n加载模型: {model_path}")
        if adapter_path:
            print(f"加载适配器: {adapter_path}")
        
        # 创建评估器
        evaluator = create_llm_evaluator(
            model_path=model_path,
            adapter_path=adapter_path,
            device="cuda"
        )
        
        # 执行评估
        print("\n正在进行LLM评估...")
        result = evaluator.evaluate(generated_layout, existing_layout)
        
        # 显示结果
        print("\n" + "-" * 40)
        print("LLM评估结果")
        print("-" * 40)
        
        print(f"\n总分: {result.total_score:.1f}/10")
        print(f"是否有效: {'✓' if result.is_valid else '✗'}")
        print(f"置信度: {result.confidence:.2f}")
        
        print("\n各维度得分:")
        for dim, score in result.dimension_scores.items():
            print(f"  {dim}: {score}/10")
        
        if result.issues:
            print(f"\n发现的问题 ({len(result.issues)}个):")
            for issue in result.issues:
                print(f"  - {issue}")
        
        if result.suggestions:
            print(f"\n改进建议 ({len(result.suggestions)}条):")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
        
        return result
        
    except ImportError as e:
        print(f"\n无法导入LLM评估器（可能缺少GPU环境）: {e}")
        print("\n请确保安装了以下依赖:")
        print("  pip install torch transformers peft")
        return None
    except Exception as e:
        print(f"\nLLM评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_hybrid_evaluation():
    """
    示例：使用混合评估器（规则+LLM）
    """
    print("\n" + "=" * 60)
    print("混合评估器使用示例")
    print("=" * 60)
    
    # 测试布局数据
    generated_layout = {
        "客厅": [0, 0, 4000, 4000],
        "卧室1": [0, 4500, 3300, 4000],
        "厨房": [4500, 0, 2400, 3000],
        "卫生间": [4500, 3500, 1800, 2400],
    }
    
    existing_layout = {
        "边界": [0, 0, 9600, 10500],
        "南采光": [0, -1200, 9600, 1200],
    }
    
    # 首先使用规则评估（不需要GPU）
    from core import LayoutEvaluator
    
    rule_evaluator = LayoutEvaluator()
    rule_result = rule_evaluator.evaluate(generated_layout, existing_layout)
    
    print("\n规则评估结果:")
    print(f"  总分: {rule_result.total_score:.1f}/100")
    print(f"  是否有效: {'✓' if rule_result.is_valid else '✗'}")
    
    # 尝试使用混合评估器
    try:
        from core import create_hybrid_evaluator
        
        # 配置
        model_path = "Qwen/Qwen2.5-14B-Instruct"  # 根据实际环境修改
        
        print(f"\n创建混合评估器 (规则权重=60%, LLM权重=40%)")
        
        hybrid = create_hybrid_evaluator(
            rule_evaluator=rule_evaluator,
            model_path=model_path,
            llm_weight=0.4,
            device="cuda"
        )
        
        # 综合评估
        print("正在进行综合评估...")
        result = hybrid.evaluate(generated_layout, existing_layout)
        
        print("\n" + "-" * 40)
        print("混合评估结果")
        print("-" * 40)
        
        print(f"\n综合得分: {result['combined_score']:.1f}/100")
        print(f"是否有效: {'✓' if result['is_valid'] else '✗'}")
        
        if result['combined_issues']:
            print(f"\n综合问题列表:")
            for issue in result['combined_issues']:
                print(f"  - {issue}")
        
        if result['combined_suggestions']:
            print(f"\n综合改进建议:")
            for suggestion in result['combined_suggestions']:
                print(f"  - {suggestion}")
        
        return result
        
    except Exception as e:
        print(f"\n混合评估失败（可能缺少GPU环境）: {e}")
        print("仅使用规则评估结果")
        return {"rule_only": rule_result}


def show_usage_instructions():
    """显示使用说明"""
    print("""
================================================================================
LLM评估器使用说明
================================================================================

1. 环境要求:
   - Python 3.8+
   - PyTorch (支持CUDA)
   - transformers >= 4.35.0
   - peft >= 0.6.0

2. 安装依赖:
   pip install torch transformers peft

3. 基本使用:

   from core import create_llm_evaluator, LayoutEvaluator
   
   # 方式1：仅使用LLM评估
   llm_eval = create_llm_evaluator(
       model_path="Qwen/Qwen2.5-14B-Instruct",  # 基座模型路径
       adapter_path=None,                         # LoRA适配器（可选）
       device="cuda"
   )
   result = llm_eval.evaluate(generated_layout, existing_layout)
   
   # 方式2：使用混合评估（规则+LLM）
   from core import create_hybrid_evaluator
   
   rule_eval = LayoutEvaluator()
   hybrid_eval = create_hybrid_evaluator(
       rule_evaluator=rule_eval,
       model_path="Qwen/Qwen2.5-14B-Instruct",
       llm_weight=0.4,  # LLM权重40%，规则权重60%
       device="cuda"
   )
   result = hybrid_eval.evaluate(generated_layout, existing_layout)

4. 模型路径配置:
   
   # 本地路径
   model_path = "/home/nju/.cache/modelscope/hub/models/Qwen/Qwen2___5-14B-Instruct"
   
   # 或使用ModelScope ID（会自动下载）
   model_path = "Qwen/Qwen2.5-14B-Instruct"
   
   # LoRA适配器（可选）
   adapter_path = "/saves/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23"

5. 评估结果说明:
   
   LLMEvaluationResult:
     - total_score: 总分 (1-10)
     - dimension_scores: 各维度得分
     - issues: 发现的问题列表
     - suggestions: 改进建议列表
     - is_valid: 布局是否有效
     - confidence: 评估置信度

================================================================================
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM评估器示例")
    parser.add_argument("--help-usage", action="store_true", help="显示使用说明")
    parser.add_argument("--llm", action="store_true", help="运行LLM评估示例")
    parser.add_argument("--hybrid", action="store_true", help="运行混合评估示例")
    parser.add_argument("--model-path", type=str, default=None, 
                        help="基座模型路径（如 /path/to/Qwen2.5-14B-Instruct）")
    
    args = parser.parse_args()
    
    if args.help_usage:
        show_usage_instructions()
    elif args.llm:
        example_llm_evaluation(args.model_path)
    elif args.hybrid:
        example_hybrid_evaluation()
    else:
        # 默认显示说明
        show_usage_instructions()
        print("\n" + "=" * 60)
        print("快速开始")
        print("=" * 60)
        print(f"\n默认模型路径: {DEFAULT_MODEL_PATH}")
        print("\n运行LLM评估示例:")
        print("  python example_llm_evaluator.py --llm")
        print("\n指定模型路径:")
        print("  python example_llm_evaluator.py --llm --model-path /path/to/model")
        print("\n运行混合评估示例:")
        print("  python example_llm_evaluator.py --hybrid")

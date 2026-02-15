"""
户型布局生成系统 - 完整使用示例
展示如何使用优化后的生成流程
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def example_simple_evaluation():
    """
    示例1：简单评估现有布局
    不需要GPU，可以直接运行
    """
    print("\n" + "=" * 60)
    print("示例1：评估现有布局")
    print("=" * 60)
    
    from core import LayoutEvaluator
    
    # 初始化评估器
    evaluator = LayoutEvaluator()
    
    # 假设这是模型生成的布局
    generated_layout = {
        "采光1": [0, 5400, 1500, 5100],
        "卧室1": [0, 0, 3300, 5400],
        "卧室2": [6900, 0, 2700, 5400],
        "客厅": [3300, 0, 3600, 3600],
        "厨房": [4200, 7200, 2400, 3300],
        "卫生间": [1500, 5400, 1800, 2400],
        "主卫": [0, 3600, 2100, 1800],
        "餐厅": [4200, 3600, 2400, 1800]
    }
    
    # 已有的布局（包括边界、采光等）
    existing_layout = {
        "边界": [0, 0, 9600, 10500],
        "主入口": [6900, 7200, 1200, 1200],
        "南采光": [0, -1200, 9600, 1200],
        "北采光": [0, 10500, 9600, 1200],
        "西采光": [-1200, 0, 1200, 10500],
    }
    
    # 评估
    result = evaluator.evaluate(generated_layout, existing_layout)
    
    print(f"\n评估结果:")
    print(f"  总分: {result.total_score:.2f}/100")
    print(f"  是否有效: {'✓' if result.is_valid else '✗'}")
    print(f"\n各维度得分:")
    for dim, score in result.dimension_scores.items():
        print(f"    {dim}: {score:.1f}")
    
    if result.issues:
        print(f"\n发现的问题 ({len(result.issues)}个):")
        for issue in result.issues:
            print(f"    - {issue}")
    
    return result


def example_validation_and_fix():
    """
    示例2：验证并修复布局
    不需要GPU，可以直接运行
    """
    print("\n" + "=" * 60)
    print("示例2：验证并修复布局")
    print("=" * 60)
    
    from core import LayoutRuleEngine
    
    # 初始化规则引擎
    rule_engine = LayoutRuleEngine()
    
    # 有问题的布局
    problematic_layout = {
        "厨房": [3000, 3000, 2400, 3000],
        "卫生间": [5400, 3000, 1800, 2400],  # 紧挨着厨房
        "客厅": [0, 0, 4000, 4000],
        "卧室": [0, 5000, 3000, 3500],
    }
    
    existing_layout = {
        "边界": [0, 0, 9600, 10500],
    }
    
    # 验证
    print("\n原始布局验证:")
    result = rule_engine.validate(problematic_layout, existing_layout)
    print(f"  是否有效: {'✓' if result.valid else '✗'}")
    
    if result.soft_violations:
        print(f"  软性规则违反:")
        for v in result.soft_violations:
            print(f"    - {v}")
    
    # 自动修复
    print("\n尝试自动修复:")
    fixed_result = rule_engine.validate_and_fix(problematic_layout, existing_layout)
    print(f"  修复后是否有效: {'✓' if fixed_result.valid else '✗'}")
    
    if fixed_result.fixed_layout:
        print(f"  修复后的布局:")
        for room, params in fixed_result.fixed_layout.items():
            print(f"    {room}: {params}")


def example_multi_candidate_selection():
    """
    示例3：多候选选择（模拟）
    不需要GPU，演示选择逻辑
    """
    print("\n" + "=" * 60)
    print("示例3：多候选选择")
    print("=" * 60)
    
    from core import LayoutEvaluator, LayoutRuleEngine
    from core.generator import LayoutResult
    from core.optimizer import MultiCandidateOptimizer
    
    # 初始化组件
    evaluator = LayoutEvaluator()
    rule_engine = LayoutRuleEngine()
    optimizer = MultiCandidateOptimizer(evaluator, rule_engine)
    
    # 模拟多个候选结果
    candidates = [
        LayoutResult(
            layout={
                "客厅": [0, 0, 3600, 3600],
                "卧室1": [0, 4000, 3000, 3500],
                "厨房": [4000, 0, 2400, 3000],
            },
            raw_output="候选1",
            is_valid=True
        ),
        LayoutResult(
            layout={
                "客厅": [0, 0, 4000, 4000],  # 更大的客厅
                "卧室1": [0, 4500, 3300, 4000],
                "厨房": [4500, 0, 2400, 3300],
            },
            raw_output="候选2",
            is_valid=True
        ),
        LayoutResult(
            layout={
                "客厅": [0, 0, 3000, 3000],  # 较小的客厅
                "卧室1": [0, 3500, 2800, 3200],
                "厨房": [3500, 0, 2200, 2800],
            },
            raw_output="候选3",
            is_valid=True
        ),
    ]
    
    existing_layout = {
        "边界": [0, 0, 9600, 10500],
        "南采光": [0, -1200, 9600, 1200],
    }
    
    # 选择最优候选
    print("\n评估候选:")
    for i, candidate in enumerate(candidates):
        eval_result = evaluator.evaluate(candidate.layout, existing_layout)
        print(f"  候选{i+1}: 得分 {eval_result.total_score:.1f}")
    
    best, best_eval = optimizer.select_best(candidates, existing_layout)
    print(f"\n最优候选: 得分 {best_eval.total_score:.1f}")
    print(f"布局: {json.dumps(best.layout, ensure_ascii=False)}")


def example_visualization():
    """
    示例4：可视化布局
    需要matplotlib
    """
    print("\n" + "=" * 60)
    print("示例4：可视化布局")
    print("=" * 60)
    
    try:
        from utils import LayoutVisualizer
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        visualizer = LayoutVisualizer()
        
        # 完整布局
        full_layout = {
            "边界": [0, 0, 9600, 10500],
            "主入口": [6900, 7200, 1200, 1200],
            "南采光": [0, -1200, 9600, 1200],
            "北采光": [0, 10500, 9600, 1200],
            "西采光": [-1200, 0, 1200, 10500],
            "卧室1": [0, 0, 3300, 5400],
            "卧室2": [6900, 0, 2700, 5400],
            "客厅": [3300, 0, 3600, 3600],
            "厨房": [4200, 7200, 2400, 3300],
            "卫生间": [1500, 5400, 1800, 2400],
            "餐厅": [4200, 3600, 2400, 1800]
        }
        
        # 创建输出目录
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # 可视化
        output_path = output_dir / "example_layout.png"
        fig = visualizer.visualize(
            full_layout,
            title="示例户型布局",
            save_path=str(output_path)
        )
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print(f"\n可视化图片已保存到: {output_path}")
        
    except ImportError as e:
        print(f"\n跳过可视化（缺少依赖）: {e}")


def example_full_pipeline():
    """
    示例5：完整生成流程（需要GPU和模型）
    """
    print("\n" + "=" * 60)
    print("示例5：完整生成流程（需要GPU）")
    print("=" * 60)
    
    print("""
    完整生成流程需要：
    1. GPU环境
    2. 基础模型 (Qwen2.5-VL-7B-Instruct)
    3. LoRA适配器
    
    使用方法:
    
    ```python
    from pipeline import create_pipeline
    
    # 创建流程
    pipeline = create_pipeline(
        base_model_path="models/Qwen2.5-VL-7B-Instruct",
        lora_adapter_path="lora_model"
    )
    
    # 生成布局
    result = pipeline.generate(
        image_path="path/to/image.jpg",
        existing_params={
            "边界": [0, 0, 9600, 10500],
            "主入口": [6900, 7200, 1200, 1200],
            # ... 其他已有参数
        },
        rooms_to_generate=["客厅", "卧室1", "厨房", "卫生间"],
        house_type="城市",
        floor_type="一层",
        optimize=True  # 启用多候选优化
    )
    
    print(f"生成结果: {result.layout}")
    print(f"得分: {result.score}")
    print(f"是否满意: {result.is_satisfactory}")
    ```
    """)


def example_api_server():
    """
    示例6：启动API服务
    """
    print("\n" + "=" * 60)
    print("示例6：启动API服务")
    print("=" * 60)
    
    print("""
    启动API服务:
    
    ```bash
    # 安装依赖
    pip install fastapi uvicorn python-multipart
    
    # 启动服务
    cd house_layout_qwenvl
    python server.py
    ```
    
    API端点:
    - POST /init_pipeline  - 初始化生成流程（需要GPU）
    - POST /generate       - 生成户型布局
    - POST /evaluate       - 评估户型布局
    - POST /validate       - 验证户型布局
    - POST /visualize      - 可视化户型布局
    - GET  /health         - 健康检查
    
    示例请求 (评估):
    
    ```bash
    curl -X POST "http://localhost:8000/evaluate" \\
      -H "Content-Type: application/json" \\
      -d '{
        "layout": {
          "客厅": [0, 0, 4000, 4000],
          "卧室1": [0, 4500, 3300, 4000]
        },
        "full_layout": {
          "边界": [0, 0, 9600, 10500]
        }
      }'
    ```
    """)


def run_examples():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("户型布局生成系统 - 使用示例")
    print("=" * 70)
    
    # 不需要GPU的示例
    example_simple_evaluation()
    example_validation_and_fix()
    example_multi_candidate_selection()
    example_visualization()
    
    # 需要GPU的示例（仅显示说明）
    example_full_pipeline()
    example_api_server()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成!")
    print("=" * 70)


if __name__ == "__main__":
    run_examples()

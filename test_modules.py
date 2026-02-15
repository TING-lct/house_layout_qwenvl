"""
户型布局生成系统测试脚本
演示如何使用各个优化模块
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core import LayoutEvaluator, LayoutRuleEngine, ValidationResult
from core.evaluator import EvaluationResult
from utils import LayoutVisualizer, LayoutMetrics, LayoutDataAugmentation


def test_evaluator():
    """测试评估器"""
    print("=" * 50)
    print("测试评估器")
    print("=" * 50)
    
    # 创建评估器
    evaluator = LayoutEvaluator()
    
    # 测试布局数据
    existing_layout = {
        "边界": [0, 0, 9600, 10500],
        "主入口": [6900, 7200, 1200, 1200],
        "南采光": [0, -1200, 9600, 1200],
        "北采光": [0, 10500, 9600, 1200],
        "西采光": [-1200, 0, 1200, 10500],
        "黑体1": [6600, 7200, 1800, 3300],
        "黑体2": [8400, 3600, 1200, 6900]
    }
    
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
    
    # 评估
    result = evaluator.evaluate(generated_layout, existing_layout)
    
    print(f"\n总分: {result.total_score:.2f}")
    print("\n各维度得分:")
    for dim, score in result.dimension_scores.items():
        print(f"  {dim}: {score:.2f}")
    
    print(f"\n是否有效: {result.is_valid}")
    
    if result.issues:
        print("\n发现的问题:")
        for issue in result.issues:
            print(f"  - {issue}")
    
    if result.suggestions:
        print("\n改进建议:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")
    
    return result


def test_rule_engine():
    """测试规则引擎"""
    print("\n" + "=" * 50)
    print("测试规则引擎")
    print("=" * 50)
    
    # 创建规则引擎
    rule_engine = LayoutRuleEngine()
    
    # 测试布局（故意加入一些问题）
    existing_layout = {
        "边界": [0, 0, 9600, 10500],
        "南采光": [0, -1200, 9600, 1200],
    }
    
    # 有问题的布局：厨房和卫生间相邻
    problematic_layout = {
        "厨房": [3000, 3000, 2400, 3000],
        "卫生间": [5400, 3000, 1800, 2400],  # 紧挨着厨房
        "客厅": [0, 0, 4000, 4000],
    }
    
    # 验证
    result = rule_engine.validate(problematic_layout, existing_layout)
    
    print(f"\n是否有效: {result.valid}")
    
    if result.hard_violations:
        print("\n硬性规则违反:")
        for v in result.hard_violations:
            print(f"  - {v}")
    
    if result.soft_violations:
        print("\n软性规则违反:")
        for v in result.soft_violations:
            print(f"  - {v}")
    
    # 尝试自动修复
    print("\n尝试自动修复...")
    fixed_result = rule_engine.validate_and_fix(problematic_layout, existing_layout)
    
    print(f"修复后是否有效: {fixed_result.valid}")
    print(f"是否应用了修复: {fixed_result.fix_applied}")
    
    return result


def test_visualizer():
    """测试可视化器"""
    print("\n" + "=" * 50)
    print("测试可视化器")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = LayoutVisualizer()
    
    # 完整布局
    full_layout = {
        "边界": [0, 0, 9600, 10500],
        "主入口": [6900, 7200, 1200, 1200],
        "南采光": [0, -1200, 9600, 1200],
        "北采光": [0, 10500, 9600, 1200],
        "西采光": [-1200, 0, 1200, 10500],
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
    
    # 可视化（保存图片）
    output_path = "test_output/layout_visualization.png"
    Path("test_output").mkdir(exist_ok=True)
    
    try:
        fig = visualizer.visualize(
            full_layout,
            title="测试户型布局",
            show_labels=True,
            show_dimensions=True,
            save_path=output_path
        )
        print(f"\n可视化图片已保存到: {output_path}")
        
        # 关闭图形
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"\n可视化失败（可能缺少matplotlib）: {e}")


def test_metrics():
    """测试指标计算"""
    print("\n" + "=" * 50)
    print("测试指标计算")
    print("=" * 50)
    
    # 创建指标计算器
    metrics = LayoutMetrics()
    
    # 测试布局
    layout = {
        "边界": [0, 0, 9600, 10500],
        "卧室1": [0, 0, 3300, 5400],
        "卧室2": [6900, 0, 2700, 5400],
        "客厅": [3300, 0, 3600, 3600],
        "厨房": [4200, 7200, 2400, 3300],
        "卫生间": [1500, 5400, 1800, 2400],
    }
    
    # 计算指标
    result = metrics.calculate_all(layout)
    
    print(f"\n空间利用率: {result.space_utilization:.2f}%")
    print(f"约束违反数: {result.constraint_violations}")
    print(f"几何合法性: {result.geometric_validity:.2f}")
    print(f"尺寸合规率: {result.dimension_compliance:.2f}%")
    print(f"综合得分: {result.overall_score:.2f}")
    
    return result


def test_data_augmentation():
    """测试数据增强"""
    print("\n" + "=" * 50)
    print("测试数据增强")
    print("=" * 50)
    
    # 创建数据增强器
    augmentor = LayoutDataAugmentation()
    
    # 原始布局
    original_layout = {
        "边界": [0, 0, 10000, 10000],
        "客厅": [0, 0, 4000, 4000],
        "卧室": [5000, 0, 3000, 3500],
    }
    
    print(f"\n原始布局:")
    print(json.dumps(original_layout, ensure_ascii=False, indent=2))
    
    # 生成增强数据
    augmented = augmentor.augment(original_layout, include_original=False)
    
    print(f"\n生成了 {len(augmented)} 个增强样本:")
    
    for i, layout in enumerate(augmented[:3]):  # 只显示前3个
        print(f"\n增强样本 {i+1}:")
        # 只显示客厅和卧室
        print(f"  客厅: {layout.get('客厅', 'N/A')}")
        print(f"  卧室: {layout.get('卧室', 'N/A')}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("户型布局生成系统 - 模块测试")
    print("=" * 60)
    
    # 测试评估器
    test_evaluator()
    
    # 测试规则引擎
    test_rule_engine()
    
    # 测试指标计算
    test_metrics()
    
    # 测试数据增强
    test_data_augmentation()
    
    # 测试可视化（需要matplotlib）
    try:
        test_visualizer()
    except ImportError:
        print("\n跳过可视化测试（matplotlib未安装）")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

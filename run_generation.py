"""
户型布局生成 - 完整使用示例
使用已微调的模型进行生成和优化

优化流程（对应优化技术方案）：
  用户输入 → RAG案例检索 → 提示词增强（设计约束注入）
  → 多候选生成（N个不同温度） → 规则约束检查 → 评估打分
  → 选择最优 → 如果不满意: 注入问题到Prompt → 重新生成 → 循环
  → 输出最终结果

使用说明：
  1. 无GPU可运行评估/可视化示例（示例4-5）
  2. 有GPU可运行生成示例（示例1-3）
  3. 推荐使用示例3：完整优化流程
"""

import json
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from layout_predictor import LayoutPredictor, build_query, OptimizedResult

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')


# ==================== 测试数据 ====================

# 查找一个实际可用的输入图片
def find_test_image() -> str:
    """自动查找可用的测试图片"""
    input_dir = Path(__file__).parent / "LLaMA-Factory" / "data" / "input_image"
    if input_dir.exists():
        for img in input_dir.glob("*.jpeg"):
            return str(img)
        for img in input_dir.glob("*.jpg"):
            return str(img)
        for img in input_dir.glob("*.png"):
            return str(img)
    return "LLaMA-Factory/data/input_image/test.jpeg"  # 兜底


# 典型城市住宅一层测试数据
EXAMPLE_EXISTING_PARAMS = {
    "边界": [0, 0, 16500, 16200],
    "主入口": [14100, 7500, 1200, 1200],
    "南采光": [-1200, -1200, 17700, 1200],
    "北采光": [-1200, 16200, 17700, 1200],
    "西采光": [-1200, 0, 1200, 16200],
    "黑体2": [12900, 7500, 3600, 8700],
    "采光1": [9300, 15000, 3600, 1200],
    "卧室1": [0, 0, 6000, 7800],
    "卧室2": [0, 10500, 4800, 5700],
    "客厅": [6000, 0, 6900, 6300],
    "卧室3": [4800, 0, 3000, 4500],
    "卧室4": [12900, 0, 3600, 4800],
    "厨房": [6900, 12300, 2400, 3900],
    "卫生间": [4800, 12300, 2100, 3900]
}

EXAMPLE_ROOMS_TO_GENERATE = ["主卫", "储藏", "餐厅"]


# ==================== 示例函数 ====================

def example_basic_generation():
    """
    示例1：基础生成
    使用已微调的模型生成布局（单次推理）
    """
    print("\n" + "=" * 60)
    print("示例1：基础生成")
    print("=" * 60)
    
    # 创建预测器（自动加载配置文件）
    predictor = LayoutPredictor(
        base_model_path="models/Qwen2.5-VL-7B-Instruct",
        lora_adapter_path="lora_model"
    )
    
    # 构建带设计约束的查询
    query = build_query(
        house_type="城市",
        floor_type="一层",
        existing_params=EXAMPLE_EXISTING_PARAMS,
        rooms_to_generate=EXAMPLE_ROOMS_TO_GENERATE,
        prompts_config=predictor.prompts_config  # 注入设计约束
    )
    
    # 查找测试图片
    image_path = find_test_image()
    print(f"  使用图片: {image_path}")
    
    # 生成
    result = predictor.generate(
        image_path=image_path,
        query=query,
        existing_layout=EXAMPLE_EXISTING_PARAMS
    )
    
    print(f"\n生成结果:")
    print(f"  布局: {json.dumps(result.layout, ensure_ascii=False, indent=2)}")
    print(f"  得分: {result.score:.1f}")
    print(f"  是否有效: {result.is_valid}")
    if result.issues:
        print(f"  问题: {result.issues}")
    
    return result


def example_multi_candidate():
    """
    示例2：多候选生成
    通过不同温度采样生成多个候选，评估后选择最优
    """
    print("\n" + "=" * 60)
    print("示例2：多候选生成（多温度采样 + 评估选优）")
    print("=" * 60)
    
    predictor = LayoutPredictor(
        base_model_path="models/Qwen2.5-VL-7B-Instruct",
        lora_adapter_path="lora_model"
    )
    
    query = build_query(
        house_type="城市",
        floor_type="一层",
        existing_params=EXAMPLE_EXISTING_PARAMS,
        rooms_to_generate=EXAMPLE_ROOMS_TO_GENERATE,
        prompts_config=predictor.prompts_config
    )
    
    image_path = find_test_image()
    print(f"  使用图片: {image_path}")
    
    # 生成多个候选（不同温度→多样性）
    candidates = predictor.generate_candidates(
        image_path=image_path,
        query=query,
        existing_layout=EXAMPLE_EXISTING_PARAMS,
        num_candidates=5,
        temperatures=[0.3, 0.5, 0.7, 0.9, 1.1]
    )
    
    print(f"\n生成了 {len(candidates)} 个候选:")
    for i, candidate in enumerate(candidates):
        print(f"  候选{i+1}: 得分={candidate.score:.1f}, 有效={candidate.is_valid}")
    
    # 选择最优
    best, best_eval = predictor.select_best(candidates, EXAMPLE_EXISTING_PARAMS)
    if best:
        print(f"\n最优候选:")
        print(f"  得分: {best_eval.total_score:.1f}")
        print(f"  布局: {json.dumps(best.layout, ensure_ascii=False)}")
    
    return best


def example_optimized_generation():
    """
    示例3：完整优化生成流程（推荐使用）
    
    对应优化技术方案中的核心流程：
    多候选生成 → 五维度评估 → 选最优 → 规则修复 → 
    识别问题 → 注入问题到Prompt → 重新生成 → 循环直到满意
    
    关键优化点：
    1. 多温度采样 → 生成多样化候选方案
    2. 五维度评分 → 空间合理性/采光/动线/功能分区/尺寸规范
    3. 硬性规则验证 → 无重叠/不超边界/最小尺寸
    4. 自动修复 → 规则引擎修正违规
    5. 迭代优化 → 上一轮问题注入下一轮Prompt引导改进
    """
    print("\n" + "=" * 60)
    print("示例3：完整优化生成流程")
    print("  多候选生成 → 评估 → 选优 → 修复 → 问题注入Prompt → 迭代")
    print("=" * 60)
    
    predictor = LayoutPredictor(
        base_model_path="models/Qwen2.5-VL-7B-Instruct",
        lora_adapter_path="lora_model"
    )
    
    query = build_query(
        house_type="城市",
        floor_type="一层",
        existing_params=EXAMPLE_EXISTING_PARAMS,
        rooms_to_generate=EXAMPLE_ROOMS_TO_GENERATE,
        prompts_config=predictor.prompts_config
    )
    
    image_path = find_test_image()
    print(f"  使用图片: {image_path}")
    
    # 核心：优化生成
    result = predictor.generate_optimized(
        image_path=image_path,
        query=query,
        existing_layout=EXAMPLE_EXISTING_PARAMS,
        num_candidates=5,          # 每轮生成5个候选
        score_threshold=85.0,       # 达到85分即停止
        max_iterations=3,           # 最多迭代3轮
        auto_fix=True,              # 启用规则引擎自动修复
        improvement_threshold=3.0   # 提升不足3分视为收敛
    )
    
    print(f"\n📊 优化生成详细结果:")
    print(f"  最终得分: {result.score:.1f}")
    print(f"  是否满意: {result.is_satisfactory}")
    print(f"  候选总数: {result.candidates_count}")
    print(f"  优化轮数: {result.optimization_rounds}")
    print(f"  布局: {json.dumps(result.layout, ensure_ascii=False, indent=2)}")
    
    # 打印迭代历史
    if result.iteration_history:
        print(f"\n📈 迭代历史:")
        for h in result.iteration_history:
            print(f"  第{h['iteration']}轮: 类型={h['query_type']}, "
                  f"候选={h.get('num_candidates', 0)}, "
                  f"最优分={h.get('best_score', 0):.1f}, "
                  f"提升={h.get('improvement', 0):.1f}")
            if h.get('issues'):
                print(f"    注入问题: {len(h['issues'])}个 → 下一轮Prompt")
    
    if result.issues:
        print(f"\n⚠️ 剩余问题:")
        for issue in result.issues:
            print(f"    - {issue}")
    
    if result.suggestions:
        print(f"\n💡 建议:")
        for suggestion in result.suggestions:
            print(f"    - {suggestion}")
    
    return result


def example_evaluate_only():
    """
    示例4：仅评估（无需GPU）
    评估已有布局的质量，展示五维度评分和规则验证
    """
    print("\n" + "=" * 60)
    print("示例4：仅评估（无需GPU）")
    print("  五维度评分：空间合理性 / 采光通风 / 动线设计 / 功能分区 / 尺寸规范")
    print("=" * 60)
    
    # 创建预测器（不加载模型，仅用评估器）
    predictor = LayoutPredictor()
    
    existing = {
        "边界": [0, 0, 9600, 10500],
        "主入口": [6900, 7200, 1200, 1200],
        "南采光": [0, -1200, 9600, 1200],
    }
    
    generated = {
        "客厅": [0, 0, 4000, 4000],
        "卧室1": [0, 4500, 3300, 4000],
        "厨房": [4500, 0, 2400, 3000],
        "卫生间": [4500, 3500, 1800, 2400],
    }
    
    # 评估
    result = predictor.evaluate(generated, existing)
    
    print(f"\n📊 评估结果:")
    print(f"  总分: {result.total_score:.1f}/100")
    print(f"  是否有效: {result.is_valid}")
    print(f"\n各维度得分:")
    for dim, score in result.dimension_scores.items():
        print(f"    {dim}: {score:.1f}")
    
    if result.issues:
        print(f"\n⚠️ 发现的问题:")
        for issue in result.issues:
            print(f"    - {issue}")
    
    # 规则验证
    val_result = predictor.validate(generated, existing, auto_fix=True)
    print(f"\n🔍 规则验证:")
    print(f"  通过: {val_result.valid}")
    if val_result.hard_violations:
        print(f"  硬性违规: {val_result.hard_violations}")
    if val_result.soft_violations:
        print(f"  软性违规: {val_result.soft_violations}")
    
    return result


def example_with_visualization():
    """
    示例5：评估并可视化（无需GPU）
    """
    print("\n" + "=" * 60)
    print("示例5：评估并可视化")
    print("=" * 60)
    
    from utils import LayoutVisualizer
    from pathlib import Path
    
    predictor = LayoutPredictor()
    visualizer = LayoutVisualizer()
    
    # 完整布局（示例）
    full_layout = {
        "边界": [0, 0, 9600, 10500],
        "主入口": [6900, 7200, 1200, 1200],
        "南采光": [0, -1200, 9600, 1200],
        "北采光": [0, 10500, 9600, 1200],
        "客厅": [0, 0, 4000, 4000],
        "卧室1": [0, 4500, 3300, 4000],
        "卧室2": [4500, 4500, 3000, 3500],
        "厨房": [4500, 0, 2400, 3000],
        "卫生间": [7500, 0, 2100, 2400],
        "餐厅": [4000, 0, 2500, 2000],
    }
    
    existing = {k: v for k, v in full_layout.items() 
                if k in ['边界', '主入口', '南采光', '北采光']}
    generated = {k: v for k, v in full_layout.items() 
                 if k not in ['边界', '主入口', '南采光', '北采光']}
    
    result = predictor.evaluate(generated, existing)
    print(f"\n评估得分: {result.total_score:.1f}")
    
    # 可视化
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    fig = visualizer.visualize(
        full_layout,
        title=f"户型布局 (得分: {result.total_score:.1f})",
        save_path=str(output_dir / "generated_layout.png")
    )
    
    import matplotlib.pyplot as plt
    plt.close(fig)
    
    print(f"可视化已保存到: {output_dir / 'generated_layout.png'}")


def run_all_examples():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("🏠 户型布局生成 - 优化流程演示")
    print("=" * 70)
    print("""
优化技术方案实现的完整流程：
  ┌─────────────┐
  │  用户输入    │ (图片 + 已有参数 + 待生成房间)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ 提示词增强   │ (注入设计约束 from config/prompts.yaml)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ 多候选生成   │ (N个不同温度采样)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ 规则约束检查  │ (硬性:重叠/超界, 软性:厨卫分离)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ 五维度评估   │ (空间/采光/动线/功能/尺寸)
  └──────┬──────┘
         ▼
  ┌─────────────┐     分数不达标
  │ 选择最优    │ ──────────┐
  └──────┬──────┘           ▼
         │           ┌──────────────┐
     达标│           │ 问题注入Prompt │ (迭代修正)
         ▼           └──────┬───────┘
  ┌─────────────┐           │
  │  输出结果    │     ◄─────┘
  └─────────────┘
    """)
    
    # 无需GPU的示例
    example_evaluate_only()
    
    try:
        example_with_visualization()
    except ImportError:
        print("\n跳过可视化示例（缺少matplotlib）")
    
    # GPU示例说明
    print("\n" + "=" * 60)
    print("🚀 以下示例需要GPU环境运行")
    print("=" * 60)
    
    print("""
    # ── 推荐用法：完整优化生成 ──
    from layout_predictor import LayoutPredictor, build_query
    
    predictor = LayoutPredictor(
        base_model_path="models/Qwen2.5-VL-7B-Instruct",
        lora_adapter_path="lora_model"
    )
    
    # 构建带设计约束的查询
    query = build_query(
        house_type="城市",
        floor_type="一层",
        existing_params={...},
        rooms_to_generate=["客厅", "卧室1", ...],
        prompts_config=predictor.prompts_config  # 自动注入设计约束
    )
    
    # 完整优化流程
    result = predictor.generate_optimized(
        image_path="your_image.jpeg",
        query=query,
        existing_layout=existing_params,
        num_candidates=5,          # 每轮5个候选
        score_threshold=85.0,       # 85分停止
        max_iterations=3,           # 最多3轮迭代
        auto_fix=True,              # 启用规则修复
        improvement_threshold=3.0   # 收敛阈值
    )
    
    # 结果
    print(f"得分: {result.score:.1f}")
    print(f"迭代轮数: {result.optimization_rounds}")
    print(f"迭代历史: {result.iteration_history}")
    """)
    
    print("\n" + "=" * 70)
    print("✅ 示例运行完成!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()

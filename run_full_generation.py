"""
户型布局完整生成脚本 —— 直接运行即可

运行方式：
    python run_full_generation.py

需要环境：
    - GPU (至少16GB显存，推荐24GB)
    - 已安装: transformers, peft, torch, qwen_vl_utils
    - 基座模型: Qwen2.5-VL-7B-Instruct（首次运行自动下载或手动放到 models/ 下）
    - LoRA适配器: lora_model/（已包含在项目中）

完整优化流程：
    输入图片 + 已有参数
      → 提示词增强（注入设计约束）
      → 多候选生成（5个不同温度采样）
      → 五维度评估打分
      → 选择最优 + 规则修复
      → 如果不达标: 问题注入Prompt → 重新生成
      → 循环直到满意（或达到最大轮次）
      → 输出最终结果 + 可视化
"""

import json
import sys
import logging
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from layout_predictor import LayoutPredictor, build_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("main")


# ==================== 测试用例 ====================
# 来自训练数据集 dataset_house_floor_test.json 第一条数据

TEST_CASES = {
    "城市一层_A0": {
        "image": "LLaMA-Factory/data/input_image/city_l_A0_mix.jpeg",
        "house_type": "城市",
        "floor_type": "一层",
        "existing_params": {
            "边界": [0, 0, 9300, 11100],
            "主入口": [300, 9900, 1200, 1200],
            "南采光": [0, -1200, 9300, 1200],
            "北采光": [0, 11100, 9300, 1200],
            "东采光": [9300, 0, 1200, 11100],
            "黑体1": [0, 9900, 4200, 1200],
            "采光1": [4200, 8400, 5100, 2700],
            "采光2": [0, 0, 6000, 1800]
        },
        "rooms_to_generate": ["采光3", "卧室1", "卧室2", "客厅", "卧室3", "厨房", "卫生间", "餐厅"],
        # 真实标签（用于对比）
        "ground_truth": {
            "采光3": [7200, 3600, 2100, 1800],
            "卧室1": [3600, 1800, 2400, 2700],
            "卧室2": [6000, 0, 3300, 3600],
            "客厅": [0, 1800, 3600, 2700],
            "卧室3": [6600, 5400, 2700, 3000],
            "厨房": [1800, 7800, 2400, 2100],
            "卫生间": [3300, 6000, 2100, 2400],
            "餐厅": [1800, 6000, 1500, 1800]
        }
    },
    "城市小户型_O07": {
        "image": "LLaMA-Factory/data/input_image/city_s_O-07_mix.jpeg",
        "house_type": "城市",
        "floor_type": "一层",
        "existing_params": {
            "边界": [0, 0, 9600, 10500],
            "主入口": [6900, 7200, 1200, 1200],
            "南采光": [0, -1200, 9600, 1200],
        },
        "rooms_to_generate": ["客厅", "卧室1", "厨房", "卫生间"],
        "ground_truth": None
    },
}


def find_available_model() -> str:
    """查找可用的基座模型路径"""
    candidates = [
        # 相对路径
        Path("models/Qwen2.5-VL-7B-Instruct"),
        Path("models/Qwen/Qwen2.5-VL-7B-Instruct"),
        # 常见缓存路径 (Windows)
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-VL-7B-Instruct",
        Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen2___5-VL-7B-Instruct",
    ]
    
    for p in candidates:
        if p.exists():
            logger.info(f"找到本地模型: {p}")
            return str(p)
    
    # 返回 HuggingFace ID，让 transformers 自动下载
    logger.info("本地未找到模型，将使用 HuggingFace 自动下载: Qwen/Qwen2.5-VL-7B-Instruct")
    return "Qwen/Qwen2.5-VL-7B-Instruct"


def run_generation(
    test_case_name: str = "城市一层_A0",
    num_candidates: int = 5,
    score_threshold: float = 80.0,
    max_iterations: int = 3,
    base_model_path: str = None,
    output_dir: str = "output",
):
    """
    运行完整生成流程
    
    Args:
        test_case_name: 测试用例名称
        num_candidates: 每轮候选数
        score_threshold: 满意分数阈值
        max_iterations: 最大迭代轮数
        base_model_path: 基座模型路径（默认自动查找）
        output_dir: 输出目录
    """
    # 选择测试用例
    if test_case_name not in TEST_CASES:
        print(f"可用的测试用例: {list(TEST_CASES.keys())}")
        return
    
    case = TEST_CASES[test_case_name]
    
    # 检查图片是否存在
    image_path = Path(case["image"])
    if not image_path.exists():
        image_path = Path(__file__).parent / case["image"]
    if not image_path.exists():
        print(f"❌ 图片不存在: {case['image']}")
        print("请确认图片路径")
        return
    
    print("=" * 70)
    print(f"🏠 户型布局生成 - 完整优化流程")
    print(f"=" * 70)
    print(f"  测试用例: {test_case_name}")
    print(f"  图片路径: {image_path}")
    print(f"  户型类型: {case['house_type']} {case['floor_type']}")
    print(f"  已有房间: {len(case['existing_params'])} 个")
    print(f"  待生成房间: {case['rooms_to_generate']}")
    print(f"  每轮候选数: {num_candidates}")
    print(f"  满意阈值: {score_threshold}")
    print(f"  最大迭代: {max_iterations}")
    print(f"=" * 70)
    
    # ========== 第1步：初始化模型 ==========
    print("\n📦 [1/4] 加载模型...")
    model_path = base_model_path or find_available_model()
    
    predictor = LayoutPredictor(
        base_model_path=model_path,
        lora_adapter_path="lora_model",
        device="cuda"
    )
    predictor.load_model()
    print(f"  ✅ 模型加载完成")
    
    # ========== 第2步：构建查询 ==========
    print("\n📝 [2/4] 构建带设计约束的查询...")
    query = build_query(
        house_type=case["house_type"],
        floor_type=case["floor_type"],
        existing_params=case["existing_params"],
        rooms_to_generate=case["rooms_to_generate"],
        prompts_config=predictor.prompts_config  # 注入设计约束
    )
    print(f"  查询长度: {len(query)} 字符")
    has_constraints = '不能重叠' in query or '设计约束' in query
    print(f"  设计约束已注入: {has_constraints}")
    
    # ========== 第3步：优化生成 ==========
    print("\n🔄 [3/4] 开始优化生成流程...")
    result = predictor.generate_optimized(
        image_path=str(image_path),
        query=query,
        existing_layout=case["existing_params"],
        num_candidates=num_candidates,
        score_threshold=score_threshold,
        max_iterations=max_iterations,
        auto_fix=True,
        improvement_threshold=3.0
    )
    
    # ========== 第4步：输出结果 ==========
    print(f"\n📊 [4/4] 最终结果:")
    print(f"  得分: {result.score:.1f}/100")
    print(f"  是否满意: {'✅ 是' if result.is_satisfactory else '❌ 否'}")
    print(f"  总候选数: {result.candidates_count}")
    print(f"  迭代轮数: {result.optimization_rounds}")
    
    if result.layout:
        print(f"\n  生成的布局:")
        print(f"  {json.dumps(result.layout, ensure_ascii=False, indent=4)}")
    
    if result.issues:
        print(f"\n  ⚠️ 剩余问题:")
        for issue in result.issues:
            print(f"    - {issue}")
    
    if result.suggestions:
        print(f"\n  💡 建议:")
        for suggestion in result.suggestions:
            print(f"    - {suggestion}")
    
    # 迭代历史
    if result.iteration_history:
        print(f"\n  📈 迭代历史:")
        for h in result.iteration_history:
            print(f"    第{h['iteration']}轮: "
                  f"类型={h['query_type']}, "
                  f"候选={h.get('num_candidates', 0)}, "
                  f"有效={h.get('num_valid', 0)}, "
                  f"最优={h.get('best_score', 0):.1f}, "
                  f"提升={h.get('improvement', 0):.1f}")
    
    # 与真实标签对比
    if case.get("ground_truth") and result.layout:
        print(f"\n  📏 与真实标签对比:")
        gt = case["ground_truth"]
        for room in case["rooms_to_generate"]:
            if room in result.layout and room in gt:
                gen = result.layout[room]
                ref = gt[room]
                diff = sum(abs(a - b) for a, b in zip(gen, ref))
                print(f"    {room}: 生成={gen}, 标签={ref}, 偏差={diff}mm")
            elif room in result.layout:
                print(f"    {room}: 生成={result.layout[room]}, 标签=无")
            else:
                print(f"    {room}: ❌ 未生成")
    
    # 保存结果
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    result_data = {
        "test_case": test_case_name,
        "score": result.score,
        "is_satisfactory": result.is_satisfactory,
        "candidates_count": result.candidates_count,
        "optimization_rounds": result.optimization_rounds,
        "layout": result.layout,
        "issues": result.issues,
        "suggestions": result.suggestions,
        "iteration_history": result.iteration_history
    }
    
    result_file = out_dir / f"result_{test_case_name}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 结果已保存: {result_file}")
    
    # 可视化
    try:
        from utils import LayoutVisualizer
        visualizer = LayoutVisualizer()
        
        # 合并完整布局
        full_layout = {**case["existing_params"]}
        if result.layout:
            full_layout.update(result.layout)
        
        fig = visualizer.visualize(
            full_layout,
            title=f"{test_case_name} (得分: {result.score:.1f})",
            save_path=str(out_dir / f"layout_{test_case_name}.png")
        )
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  📊 可视化已保存: {out_dir / f'layout_{test_case_name}.png'}")
    except ImportError:
        print("  跳过可视化（缺少 matplotlib）")
    
    print(f"\n{'=' * 70}")
    print(f"✅ 生成完成!")
    print(f"{'=' * 70}")
    
    return result


def run_batch(num_cases: int = 3, **kwargs):
    """批量运行多个测试用例"""
    cases = list(TEST_CASES.keys())[:num_cases]
    results = {}
    
    for name in cases:
        print(f"\n\n{'#' * 70}")
        print(f"# 测试用例: {name}")
        print(f"{'#' * 70}")
        result = run_generation(test_case_name=name, **kwargs)
        if result:
            results[name] = result.score
    
    print(f"\n\n{'=' * 70}")
    print(f"📋 批量运行汇总:")
    for name, score in results.items():
        print(f"  {name}: {score:.1f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="户型布局完整生成")
    parser.add_argument(
        "--case", type=str, default="城市一层_A0",
        help=f"测试用例名称，可选: {list(TEST_CASES.keys())}"
    )
    parser.add_argument("--candidates", type=int, default=5, help="每轮候选数 (默认5)")
    parser.add_argument("--threshold", type=float, default=80.0, help="满意分数阈值 (默认80)")
    parser.add_argument("--iterations", type=int, default=3, help="最大迭代轮数 (默认3)")
    parser.add_argument("--model", type=str, default=None, help="基座模型路径 (默认自动查找)")
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument("--batch", action="store_true", help="批量运行所有测试用例")
    
    args = parser.parse_args()
    
    if args.batch:
        run_batch(
            num_candidates=args.candidates,
            score_threshold=args.threshold,
            max_iterations=args.iterations,
            base_model_path=args.model,
            output_dir=args.output
        )
    else:
        run_generation(
            test_case_name=args.case,
            num_candidates=args.candidates,
            score_threshold=args.threshold,
            max_iterations=args.iterations,
            base_model_path=args.model,
            output_dir=args.output
        )

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

from layout_predictor import LayoutPredictor, build_query
import json
import sys
import logging
import argparse
from typing import Optional
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("main")


# ==================== 测试用例 ====================
# 来自训练数据集 dataset_house_floor_test.json 第一条数据

TEST_CASES = {
    # ──────────── 城市大户型 (city_l) ─ 来自 test 数据集 ────────────
    "城市大_A0": {
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
            "采光2": [0, 0, 6000, 1800],
        },
        "rooms_to_generate": ["采光3", "卧室1", "卧室2", "客厅", "卧室3", "厨房", "卫生间", "餐厅"],
        "ground_truth": {
            "采光3": [7200, 3600, 2100, 1800],
            "卧室1": [3600, 1800, 2400, 2700],
            "卧室2": [6000, 0, 3300, 3600],
            "客厅": [0, 1800, 3600, 2700],
            "卧室3": [6600, 5400, 2700, 3000],
            "厨房": [1800, 7800, 2400, 2100],
            "卫生间": [3300, 6000, 2100, 2400],
            "餐厅": [1800, 6000, 1500, 1800],
        },
    },
    "城市大_A1": {
        "image": "LLaMA-Factory/data/input_image/city_l_A1-1-无-2_mix.jpeg",
        "house_type": "城市",
        "floor_type": "一层",
        "existing_params": {
            "边界": [0, 0, 9300, 8400],
            "主入口": [8100, 3300, 1200, 900],
            "南采光": [0, -600, 9300, 600],
            "北采光": [0, 8400, 9300, 600],
            "东采光": [9300, 0, 600, 8400],
            "黑体1": [8100, 0, 1200, 3300],
            "采光1": [0, 5700, 4200, 2700],
        },
        "rooms_to_generate": ["采光3", "卧室1", "卧室2", "客厅", "厨房", "主卫", "餐厅"],
        "ground_truth": {
            "采光3": [4200, 5700, 2400, 2700],
            "卧室1": [0, 0, 3600, 2700],
            "卧室2": [0, 2700, 4200, 3000],
            "客厅": [4200, 2700, 3900, 3000],
            "厨房": [6600, 5700, 2700, 2700],
            "主卫": [3600, 0, 4500, 2700],
            "餐厅": [4200, 0, 3900, 2700],
        },
    },
    "城市大_G81": {
        "image": "LLaMA-Factory/data/input_image/city_l_G81-1-无-1_mix.jpeg",
        "house_type": "城市",
        "floor_type": "一层",
        "existing_params": {
            "边界": [0, 0, 7500, 15000],
            "主入口": [1800, 14400, 1200, 600],
            "南采光": [0, -600, 7500, 600],
            "北采光": [0, 15000, 7500, 600],
            "黑体1": [1800, 10200, 2100, 1800],
            "采光1": [0, 12000, 3900, 3000],
            "采光2": [0, 6300, 3600, 3000],
            "卧室1": [3600, 6300, 3900, 3000],
            "卧室2": [3600, 0, 3900, 3300],
        },
        "rooms_to_generate": ["客厅", "卧室3", "厨房", "卫生间", "主卫", "餐厅"],
        "ground_truth": {
            "客厅": [3900, 10200, 3600, 4800],
            "卧室3": [0, 0, 3600, 3300],
            "厨房": [0, 3300, 3600, 3000],
            "卫生间": [3600, 3300, 3900, 3000],
            "主卫": [0, 9300, 1800, 2700],
            "餐厅": [0, 12000, 3900, 3000],
        },
    },

    # ──────────── 城市小户型 (city_s) ─ 来自 train 数据集 ────────────
    "城市小_A01": {
        "image": "LLaMA-Factory/data/input_image/city_s_A-01_mix.jpeg",
        "house_type": "城市",
        "floor_type": "一层",
        "existing_params": {
            "边界": [0, 0, 4200, 5700],
            "主入口": [900, -600, 900, 600],
            "北采光": [0, 5700, 4200, 600],
            "黑体1": [0, 4200, 600, 1500],
            "卧室1": [600, 4200, 3600, 1500],
        },
        "rooms_to_generate": ["客厅", "厨房", "卫生间"],
        "ground_truth": {
            "客厅": [0, 2400, 4200, 1800],
            "厨房": [0, 0, 900, 2400],
            "卫生间": [2400, 0, 1800, 2400],
        },
    },
    "城市小_K01": {
        "image": "LLaMA-Factory/data/input_image/city_s_K-01_mix.jpeg",
        "house_type": "城市",
        "floor_type": "一层",
        "existing_params": {
            "边界": [0, 0, 6900, 9900],
            "主入口": [0, 8700, 1200, 900],
            "南采光": [0, -900, 6900, 900],
            "北采光": [0, 9900, 6900, 900],
            "黑体1": [0, 8700, 1200, 900],
            "采光1": [3600, 8700, 3300, 1200],
            "卧室1": [3600, 6000, 3300, 2700],
            "卧室2": [3600, 0, 3300, 3900],
            "客厅": [0, 0, 3600, 3900],
        },
        "rooms_to_generate": ["厨房", "卫生间", "餐厅"],
        "ground_truth": {
            "厨房": [1200, 6600, 2400, 3300],
            "卫生间": [4500, 3900, 2400, 2100],
            "餐厅": [1200, 4800, 2400, 1800],
        },
    },

    # ──────────── 乡村户型 (rural) ─ 来自 train 数据集 ────────────
    "乡村_BL100": {
        "image": "LLaMA-Factory/data/input_image/rural_f_B-L-100_mix.jpeg",
        "house_type": "乡村",
        "floor_type": "一层",
        "existing_params": {
            "边界": [0, 0, 12000, 15300],
            "院门": [4200, -1200, 1200, 1200],
            "院子": [0, 0, 6900, 5100],
            "主入口": [5100, 3900, 1200, 1200],
            "南采光": [0, -1200, 12000, 1200],
            "北采光": [0, 15300, 12000, 1200],
            "东采光": [12000, 0, 1200, 15300],
            "西采光": [-1200, 0, 1200, 15300],
            "卧室1": [8100, 5100, 3900, 5100],
            "卧室2": [8100, 10200, 3900, 5100],
            "客厅": [6900, 0, 5100, 5100],
            "厨房": [0, 10200, 3900, 3000],
            "楼梯": [0, 13200, 3900, 2100],
            "卫生间": [0, 8100, 3900, 2100],
        },
        "rooms_to_generate": ["储藏", "玄关", "餐厅"],
        "ground_truth": {
            "储藏": [0, 5100, 3900, 3000],
            "玄关": [3900, 5100, 4200, 5100],
            "餐厅": [3900, 10200, 4200, 5100],
        },
    },
    "乡村_EI53": {
        "image": "LLaMA-Factory/data/input_image/rural_f_E-I-53_mix.jpeg",
        "house_type": "乡村",
        "floor_type": "一层",
        "existing_params": {
            "边界": [0, 0, 13200, 9900],
            "院子": [9900, 6300, 3300, 3600],
            "主入口": [5400, 600, 900, 900],
            "门廊": [0, 0, 9900, 1500],
            "南采光": [0, -1200, 13200, 1200],
            "北采光": [0, 9900, 13200, 1200],
            "西采光": [-1200, 0, 1200, 9900],
            "黑体1": [13200, 0, 1200, 9900],
            "卧室1": [9900, 0, 3300, 4500],
            "卧室2": [0, 1500, 3300, 3900],
            "客厅": [3300, 1500, 6600, 3900],
        },
        "rooms_to_generate": ["厨房", "卫生间", "主卫", "储藏", "餐厅", "次入口"],
        "ground_truth": {
            "厨房": [0, 7500, 3300, 2400],
            "卫生间": [0, 5400, 3300, 2100],
            "主卫": [11100, 4500, 2100, 1800],
            "储藏": [6600, 6300, 3300, 3600],
            "餐厅": [3300, 6300, 3300, 3600],
            "次入口": [3300, 9900, 900, 900],
        },
    },
}


def find_available_llm_model() -> Optional[str]:
    """查找可用的LLM评估模型路径（Qwen2.5-14B-Instruct）"""
    candidates = [
        # 相对路径
        Path("models/Qwen2.5-14B-Instruct"),
        # AutoDL 目录
        Path("/root/autodl-tmp/models/Qwen2.5-14B-Instruct"),
        Path("/autodl-tmp/models/Qwen2.5-14B-Instruct"),
    ]

    for p in candidates:
        if p.exists() and (p / "config.json").exists():
            logger.info(f"找到LLM评估模型: {p}")
            return str(p)

    logger.info("未找到本地LLM评估模型 (Qwen2.5-14B-Instruct)")
    return None


def find_available_llm_adapter() -> Optional[str]:
    """查找可用的LLM评估模型LoRA适配器路径"""
    candidates = [
        Path("/root/autodl-tmp/qwen14b/Qwen2.5-14B-Instruct/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23"),
        Path("F:/task/户型图生成/qwen14b/Qwen2.5-14B-Instruct/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23"),
    ]

    for p in candidates:
        if p.exists() and (p / "adapter_config.json").exists():
            logger.info(f"找到LLM评估模型适配器: {p}")
            return str(p)

    logger.info("未找到LLM评估模型适配器，将使用基座模型评估")
    return None


def find_available_model() -> str:
    """查找可用的基座模型路径"""
    def _is_model_dir_complete(model_dir: Path) -> bool:
        """检查本地模型目录是否完整（重点检查 safetensors 分片）"""
        if not model_dir.exists() or not model_dir.is_dir():
            return False

        # 基础配置文件至少应存在一个
        has_config = (model_dir / "config.json").exists()
        if not has_config:
            return False

        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            try:
                data = json.loads(index_file.read_text(encoding="utf-8"))
                weight_map = data.get("weight_map", {})
                shard_files = sorted(set(weight_map.values()))
                if not shard_files:
                    logger.warning(f"模型索引为空，跳过: {model_dir}")
                    return False

                missing = [name for name in shard_files if not (
                    model_dir / name).exists()]
                if missing:
                    logger.warning(
                        "本地模型分片不完整，缺失 %d 个文件（示例: %s），跳过: %s",
                        len(missing),
                        missing[0],
                        model_dir,
                    )
                    return False
                return True
            except Exception as e:
                logger.warning(f"读取模型索引失败，跳过: {model_dir}, 错误: {e}")
                return False

        # 兼容无索引场景：至少有一个权重文件
        has_single_bin = (model_dir / "pytorch_model.bin").exists()
        has_single_safe = (model_dir / "model.safetensors").exists()
        has_shards = bool(list(model_dir.glob("model-*-of-*.safetensors")))
        return has_single_bin or has_single_safe or has_shards

    candidates = [
        # 相对路径
        Path("models/Qwen2.5-VL-7B-Instruct"),
        Path("models/Qwen/Qwen2.5-VL-7B-Instruct"),
        # AutoDL 目录（云服务器）
        Path("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"),
        Path("/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"),
        # 常见缓存路径 (Windows)
        Path.home() / ".cache" / "huggingface" / "hub" /
        "models--Qwen--Qwen2.5-VL-7B-Instruct",
        Path.home() / ".cache" / "modelscope" / "hub" /
        "models" / "Qwen" / "Qwen2___5-VL-7B-Instruct",
    ]

    for p in candidates:
        if _is_model_dir_complete(p):
            logger.info(f"找到本地模型: {p}")
            return str(p)
        elif p.exists():
            logger.warning(f"检测到本地模型目录但不完整，已跳过: {p}")

    # 返回 HuggingFace ID，让 transformers 自动下载
    logger.info("本地未找到模型，将使用 HuggingFace 自动下载: Qwen/Qwen2.5-VL-7B-Instruct")
    return "Qwen/Qwen2.5-VL-7B-Instruct"


def _calc_iou(rect1: list, rect2: list) -> float:
    """计算两个矩形的IoU (Intersection over Union)

    Args:
        rect1, rect2: [x, y, width, height]

    Returns:
        float: IoU值 (0~1)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 计算交集
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _calc_center_offset(rect1: list, rect2: list) -> float:
    """计算两个矩形中心点的欧氏距离"""
    cx1 = rect1[0] + rect1[2] / 2
    cy1 = rect1[1] + rect1[3] / 2
    cx2 = rect2[0] + rect2[2] / 2
    cy2 = rect2[1] + rect2[3] / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def run_generation(
    test_case_name: str = "城市大_A0",
    num_candidates: int = 5,
    score_threshold: float = 80.0,
    max_iterations: int = 3,
    base_model_path: Optional[str] = None,
    output_dir: str = "output",
    llm_model_path: Optional[str] = None,
    llm_adapter_path: Optional[str] = None,
    llm_weight: float = 0.4,
):
    """
    运行完整生成流程：迭代优化 + LLM二次挑选

    流程：多轮迭代生成→规则修复→问题注入→收集候选池→LLM从候选池精选

    Args:
        test_case_name: 测试用例名称
        num_candidates: 每轮候选数
        score_threshold: 满意分数阈值
        max_iterations: 最大迭代轮数
        base_model_path: 基座模型路径（默认自动查找）
        output_dir: 输出目录
        llm_model_path: LLM评估模型路径（自动查找，找到则启用LLM挑选）
        llm_adapter_path: LLM评估模型LoRA路径
        llm_weight: LLM评分权重（0~1）
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
    _llm_model = llm_model_path or find_available_llm_model()
    _llm_adapter = llm_adapter_path or find_available_llm_adapter()
    if _llm_model:
        print(f"  LLM挑选: ✅ ({_llm_model})")
        print(f"  LLM权重: {llm_weight}")
    else:
        print(f"  LLM挑选: ❌ (未找到LLM模型，仅使用规则评分)")
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
    print("\n📝 [2/4] 构建查询（与训练数据格式一致）...")
    query = build_query(
        house_type=case["house_type"],
        floor_type=case["floor_type"],
        existing_params=case["existing_params"],
        rooms_to_generate=case["rooms_to_generate"]
    )
    print(f"  查询长度: {len(query)} 字符")

    # ========== 第3步：迭代优化生成 + 可选LLM二次挑选 ==========
    print("\n🔄 [3/4] 开始优化生成流程（迭代优化" +
          (" + LLM挑选）..." if _llm_model else "）..."))

    result = predictor.generate_optimized(
        image_path=str(image_path),
        query=query,
        existing_layout=case["existing_params"],
        num_candidates=num_candidates,
        score_threshold=score_threshold,
        max_iterations=max_iterations,
        auto_fix=True,
        improvement_threshold=2.0,
        llm_model_path=_llm_model,
        llm_adapter_path=_llm_adapter,
        llm_weight=llm_weight,
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

    # 初始化结果数据
    result_data = {
        "test_case": test_case_name,
        "score": result.score,
        "is_satisfactory": result.is_satisfactory,
        "candidates_count": result.candidates_count,
        "optimization_rounds": result.optimization_rounds,
        "layout": result.layout,
        "issues": result.issues,
        "suggestions": result.suggestions,
        "iteration_history": result.iteration_history,
    }

    # 与真实标签对比
    if case.get("ground_truth") and result.layout:
        print(f"\n  📏 与真实标签对比:")
        gt = case["ground_truth"]
        ious = []
        for room in case["rooms_to_generate"]:
            if room in result.layout and room in gt:
                gen = result.layout[room]
                ref = gt[room]
                diff = sum(abs(a - b) for a, b in zip(gen, ref))
                iou = _calc_iou(gen, ref)
                offset = _calc_center_offset(gen, ref)
                ious.append(iou)
                status = "✅" if iou > 0.5 else "⚠️" if iou > 0.2 else "❌"
                print(f"    {status} {room}: 生成={gen}, 标签={ref}, "
                      f"偏差={diff}mm, IoU={iou:.3f}, 中心偏移={offset:.0f}mm")
            elif room in result.layout:
                print(f"    ❓ {room}: 生成={result.layout[room]}, 标签=无")
                ious.append(0.0)
            else:
                print(f"    ❌ {room}: 未生成")
                ious.append(0.0)
        if ious:
            avg_iou = sum(ious) / len(ious)
            good_count = sum(1 for x in ious if x > 0.5)
            print(f"    ────────────────────")
            print(f"    📊 平均IoU: {avg_iou:.3f}")
            print(f"    📊 IoU>0.5的房间: {good_count}/{len(ious)}")
            result_data["avg_iou"] = avg_iou
            result_data["room_ious"] = {room: _calc_iou(result.layout.get(room, [0, 0, 0, 0]), gt.get(room, [0, 0, 0, 0]))
                                        for room in case["rooms_to_generate"] if room in gt}

    # 保存结果
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

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

        # 高亮生成的房间
        highlight = list(result.layout.keys()) if result.layout else []

        fig = visualizer.visualize(
            full_layout,
            title=f"{test_case_name} (得分: {result.score:.1f})",
            show_labels=True,
            show_dimensions=True,
            highlight_rooms=highlight,
            save_path=str(out_dir / f"layout_{test_case_name}.png")
        )

        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  📊 可视化已保存: {out_dir / f'layout_{test_case_name}.png'}")

        # 如果有ground truth，生成对比图
        if case.get("ground_truth") and result.layout:
            gt_layout = {**case["existing_params"], **case["ground_truth"]}
            gen_layout = {**case["existing_params"], **result.layout}
            fig_cmp = visualizer.compare_layouts(
                [gen_layout, gt_layout],
                titles=[f"生成结果 (得分: {result.score:.1f})",
                        "真实标签 (Ground Truth)"],
                save_path=str(out_dir / f"compare_{test_case_name}.png")
            )
            plt.close(fig_cmp)
            print(f"  📊 对比图已保存: {out_dir / f'compare_{test_case_name}.png'}")
    except ImportError:
        print("  跳过可视化（缺少 matplotlib）")

    print(f"\n{'=' * 70}")
    print(f"✅ 生成完成!")
    print(f"{'=' * 70}")

    return result


def run_batch(num_cases: int = 0, **kwargs):
    """批量运行多个测试用例 (num_cases=0 表示全部)

    所有kwargs会传递给run_generation，包括strategy, llm_model_path等
    """
    cases = list(TEST_CASES.keys())[
        :num_cases] if num_cases > 0 else list(TEST_CASES.keys())
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
        "--case", type=str, default="城市大_A0",
        help=f"测试用例名称，可选: {list(TEST_CASES.keys())}"
    )
    parser.add_argument("--candidates", type=int,
                        default=5, help="每轮候选数 (默认5)")
    parser.add_argument("--threshold", type=float,
                        default=90.0, help="满意分数阈值 (默认90)")
    parser.add_argument("--iterations", type=int,
                        default=5, help="最大迭代轮数 (默认5)")
    parser.add_argument("--model", type=str, default=None,
                        help="基座模型路径 (默认自动查找)")
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument("--batch", action="store_true", help="批量运行所有测试用例")
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="LLM评估模型路径 (如Qwen2.5-14B-Instruct)，仅llm-rerank策略使用"
    )
    parser.add_argument(
        "--llm-adapter", type=str, default=None,
        help="LLM评估模型LoRA适配器路径"
    )
    parser.add_argument(
        "--llm-weight", type=float, default=0.4,
        help="LLM评分权重(0-1)，规则权重=1-此值 (默认0.4)"
    )

    args = parser.parse_args()

    if args.batch:
        run_batch(
            num_candidates=args.candidates,
            score_threshold=args.threshold,
            max_iterations=args.iterations,
            base_model_path=args.model,
            output_dir=args.output,
            llm_model_path=args.llm_model,
            llm_adapter_path=args.llm_adapter,
            llm_weight=args.llm_weight,
        )
    else:
        run_generation(
            test_case_name=args.case,
            num_candidates=args.candidates,
            score_threshold=args.threshold,
            max_iterations=args.iterations,
            base_model_path=args.model,
            output_dir=args.output,
            llm_model_path=args.llm_model,
            llm_adapter_path=args.llm_adapter,
            llm_weight=args.llm_weight,
        )

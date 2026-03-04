"""
户型布局生成器 - 整合版
复用现有的predictor.py代码结构，集成优化功能
"""

from core.generator import LayoutResult, GenerationConfig
from core.evaluator import EvaluationResult
from core import LayoutEvaluator, LayoutRuleEngine, ValidationResult
from core.common import extract_json_from_text, clean_json_str, parse_layout_json
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import sys
import logging
from pathlib import Path

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:
    yaml = None

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入评估和规则模块（不需要GPU）

logger = logging.getLogger(__name__)


def _resolve_qwen_vl_model_class():
    """解析可用的Qwen-VL模型类（兼容不同transformers版本）"""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as model_cls
        return model_cls, "Qwen2_5_VLForConditionalGeneration", False
    except ImportError:
        pass

    try:
        from transformers import Qwen2VLForConditionalGeneration as model_cls
        return model_cls, "Qwen2VLForConditionalGeneration", False
    except ImportError:
        pass

    try:
        from transformers import AutoModelForVision2Seq as model_cls
        return model_cls, "AutoModelForVision2Seq", True
    except ImportError:
        from transformers import AutoModelForCausalLM as model_cls
        return model_cls, "AutoModelForCausalLM", True


def _is_unknown_qwen25_arch_error(exc: Exception) -> bool:
    """是否为 transformers 版本过低导致无法识别 qwen2_5_vl 架构"""
    msg = str(exc)
    return (
        "model type `qwen2_5_vl`" in msg
        and "does not recognize this architecture" in msg
    )


@dataclass
class OptimizedResult:
    """优化后的生成结果"""
    layout: Dict[str, List[int]]
    raw_output: str
    score: float
    is_satisfactory: bool
    issues: List[str]
    suggestions: List[str]
    candidates_count: int = 1
    optimization_rounds: int = 0
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)


class LayoutPredictor:
    """
    户型布局预测器
    复用predictor.py的代码结构，集成评估和优化功能
    """

    def __init__(
        self,
        base_model_path: str = "models/Qwen2.5-VL-7B-Instruct",
        lora_adapter_path: str = "lora_model",
        device: str = "cuda",
        use_flash_attention: bool = False,
        rules_config_path: Optional[str] = None,
        prompts_config_path: Optional[str] = None
    ):
        """
        初始化预测器

        Args:
            base_model_path: 基础模型路径
            lora_adapter_path: LoRA适配器路径
            device: 运行设备
            use_flash_attention: 是否使用Flash Attention
            rules_config_path: 规则配置文件路径
            prompts_config_path: 提示词配置文件路径
        """
        self.device = device
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.use_flash_attention = use_flash_attention

        # 模型和处理器（延迟加载）
        self.model: Any = None
        self.processor: Any = None

        # 加载配置文件
        self._project_root = Path(__file__).parent
        self.prompts_config = self._load_prompts_config(prompts_config_path)

        # 评估器和规则引擎（使用配置文件）
        rules_path = self._resolve_config_path(
            rules_config_path, "config/rules.yaml"
        )
        self.evaluator = LayoutEvaluator(rules_path)
        self.rule_engine = LayoutRuleEngine(rules_path)

        # 是否已加载模型
        self._model_loaded = False

    def _resolve_config_path(self, explicit_path: Optional[str], default_relative: str) -> Optional[str]:
        """解析配置文件路径"""
        if explicit_path and Path(explicit_path).exists():
            return explicit_path
        default_path = self._project_root / default_relative
        if default_path.exists():
            return str(default_path)
        return None

    def _load_prompts_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载提示词配置"""
        path = self._resolve_config_path(config_path, "config/prompts.yaml")
        if path and yaml is not None:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)  # type: ignore[union-attr]
                logger.info(f"已加载提示词配置: {path}")
                return config
            except Exception as e:
                logger.warning(f"加载提示词配置失败: {e}")
        return self._default_prompts_config()

    @staticmethod
    def _default_prompts_config() -> Dict:
        """默认提示词配置"""
        return {
            'design_constraints': (
                "设计约束：\n"
                "1. 所有房间不能重叠，房间之间不能有交叉区域\n"
                "2. 所有房间必须在边界范围内，不能超出边界\n"
                "3. 厨房与卫生间不宜直接相邻\n"
                "4. 卧室应尽量靠近采光面\n"
                "5. 客厅应有良好的采光和通风\n"
                "6. 房间尺寸应符合人体工程学标准\n"
            ),
            'fix_prompt': (
                "当前布局存在以下问题：\n{issues}\n\n"
                "请根据以上问题对布局进行修正，生成改进后的房间参数。\n\n"
                "原有布局参数：\n```json\n{original_layout}\n```\n\n"
                "请输出修正后的完整布局参数。"
            )
        }

    def load_model(self):
        """加载模型（复用predictor.py的代码）"""
        if self._model_loaded:
            return

        import torch  # type: ignore[import-not-found]
        # type: ignore[import-not-found]
        from transformers import AutoProcessor
        from peft import PeftModel  # type: ignore[import-not-found]

        model_cls, model_cls_name, use_trust_remote_code = _resolve_qwen_vl_model_class()
        logger.info(f"使用模型类: {model_cls_name}")

        logger.info("正在加载基础模型: %s", self.base_model_path)

        model_source = self.base_model_path

        load_kwargs: Dict[str, Any] = {
            "device_map": "auto",
        }
        if use_trust_remote_code:
            # 兼容旧版 transformers，通过 auto class + remote code 加载
            load_kwargs["trust_remote_code"] = True

        def _load_from(source: str):
            if self.use_flash_attention:
                return model_cls.from_pretrained(
                    source,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    **load_kwargs,
                )
            return model_cls.from_pretrained(
                source,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                **load_kwargs,
            )

        try:
            self.model = _load_from(model_source)
        except FileNotFoundError as e:
            # 如果是本地目录缺分片，自动回退到 HuggingFace ID 进行下载加载
            fallback_source = "Qwen/Qwen2.5-VL-7B-Instruct"
            logger.warning(
                "本地模型不完整，自动回退到远端模型: %s（原路径: %s）",
                fallback_source,
                model_source,
            )
            try:
                self.model = _load_from(fallback_source)
                model_source = fallback_source
            except Exception:
                raise RuntimeError(
                    "本地模型目录不完整，且远端回退加载失败。\n"
                    f"本地路径: {self.base_model_path}\n"
                    f"回退模型: {fallback_source}\n"
                    "请删除损坏目录后重试，或检查网络后再次运行。"
                ) from e
        except Exception as e:
            if _is_unknown_qwen25_arch_error(e):
                import transformers  # type: ignore[import-not-found]

                current_version = getattr(
                    transformers, "__version__", "unknown")
                raise RuntimeError(
                    "当前 transformers 版本不支持 Qwen2.5-VL（缺少 qwen2_5_vl 架构）。\n"
                    f"当前版本: {current_version}\n"
                    "请在当前环境执行升级：\n"
                    "  pip install -U 'transformers>=4.45.0'\n"
                    "如仍失败，再执行：\n"
                    "  pip install -U qwen-vl-utils"
                ) from e
            raise

        # 加载LoRA适配器
        if self.lora_adapter_path:
            logger.info("正在加载LoRA适配器: %s", self.lora_adapter_path)
            try:
                import warnings
                with warnings.catch_warnings():
                    # 过滤 visual blocks 缺少 LoRA 权重的无害警告
                    warnings.filterwarnings(
                        "ignore",
                        message=".*missing adapter keys.*",
                        category=UserWarning,
                    )
                    self.model = PeftModel.from_pretrained(
                        self.model, self.lora_adapter_path)
                self.model = self.model.half()
            except ValueError as e:
                # LoRA 与基座模型不匹配时，跳过适配器
                logger.warning(
                    "LoRA 适配器与基座模型不匹配，已跳过。"
                    "原因: %s",
                    e,
                )

        # 加载处理器
        # 若发生了本地->远端回退，处理器也使用同一来源
        self.base_model_path = model_source
        self.processor = AutoProcessor.from_pretrained(
            model_source,
            use_fast=True
        )

        self._model_loaded = True
        logger.info("模型加载完成")

    def unload_model(self):
        """卸载VL模型，释放GPU显存（为加载LLM评估模型腾空间）"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._model_loaded = False

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()
        logger.info("VL模型已卸载，GPU显存已释放")

    def generate_with_llm_rerank(
        self,
        image_path: str,
        query: str,
        existing_layout: Dict[str, List[int]],
        num_candidates: int = 6,
        llm_model_path: Optional[str] = None,
        llm_adapter_path: Optional[str] = None,
        llm_weight: float = 0.4,
        auto_fix: bool = True,
    ) -> OptimizedResult:
        """
        一次生成多个候选 + LLM二次挑选（用户建议的策略）

        流程：
        1. VL模型一次性生成N个候选（贪婪 + 不同温度采样）
        2. 规则引擎修复每个候选
        3. 规则评分器预筛选
        4. 卸载VL模型 → 加载14B评估模型
        5. LLM评估Top-K候选，综合评分挑选最优
        6. 返回最终结果

        Args:
            image_path: 输入图片路径
            query: 查询文本
            existing_layout: 已有布局参数
            num_candidates: 候选数量（建议6=1贪婪+5采样）
            llm_model_path: LLM评估模型路径（如Qwen2.5-14B-Instruct）
            llm_adapter_path: LLM评估模型的LoRA路径（可选）
            llm_weight: LLM评分权重（0~1，规则权重=1-llm_weight）
            auto_fix: 是否自动修复候选

        Returns:
            OptimizedResult: 最终结果
        """
        logger.info("\n" + "=" * 60)
        logger.info("🏠 一次生成 + LLM二次挑选 策略")
        logger.info("=" * 60)

        # ========== 第1步：一次性生成所有候选 ==========
        logger.info("📝 [1/4] 生成 %d 个候选...", num_candidates)
        candidates = self.generate_candidates(
            image_path=image_path,
            query=query,
            existing_layout=existing_layout,
            num_candidates=num_candidates,
        )

        valid_count = sum(1 for c in candidates if c.layout)
        logger.info("  ✅ 有效候选: %d/%d", valid_count, len(candidates))

        # ========== 第2步：规则修复 + 后处理 ==========
        logger.info("🔧 [2/4] 规则修复 + 后处理...")
        scored_candidates = []

        for i, cand in enumerate(candidates):
            if not cand.layout:
                logger.info("  候选%d: ❌ 解析失败，跳过", i + 1)
                continue

            cur = cand.layout
            if auto_fix:
                try:
                    fix_result = self.rule_engine.validate_and_fix(
                        cur, existing_layout)
                    if fix_result.fixed_layout:
                        cur = fix_result.fixed_layout
                    cur = self.rule_engine.optimize_dimensions(
                        cur, existing_layout)
                    cur = self.rule_engine.aggressive_post_process(
                        cur, existing_layout)
                except Exception as e:
                    logger.warning("  候选%d 修复异常: %s", i + 1, e)

            eval_result = self.evaluator.evaluate(cur, existing_layout)
            validation = self.rule_engine.validate(cur, existing_layout)

            scored_candidates.append({
                'index': i,
                'layout': cur,
                'raw_output': cand.raw_output,
                'score': eval_result.total_score,
                'evaluation': eval_result,
                'is_rule_valid': validation.valid,
            })

            status = "✅" if validation.valid else "⚠️"
            logger.info("  候选%d: %s 规则得分=%.1f, 规则通过=%s",
                        i + 1, status, eval_result.total_score, validation.valid)

        if not scored_candidates:
            return OptimizedResult(
                layout={}, raw_output="", score=0,
                is_satisfactory=False,
                issues=["所有候选均解析失败"],
                suggestions=["请检查输入参数和图片路径"],
                candidates_count=len(candidates),
                optimization_rounds=1,
                iteration_history=[],
            )

        # 按规则得分排序
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)

        # ========== 第3步：LLM二次挑选（如果提供了LLM模型） ==========
        if llm_model_path:
            top_k = min(3, len(scored_candidates))
            logger.info("🤖 [3/4] LLM二次评估 Top-%d 候选...", top_k)

            # 卸载VL模型释放显存
            self.unload_model()

            # 加载14B评估模型
            try:
                from core.llm_evaluator import LLMLayoutEvaluator
                llm_evaluator = LLMLayoutEvaluator(
                    model_path=llm_model_path,
                    adapter_path=llm_adapter_path,
                    device=self.device,
                )

                rule_weight = 1.0 - llm_weight

                for c in scored_candidates[:top_k]:
                    try:
                        llm_result = llm_evaluator.evaluate(
                            c['layout'], existing_layout)
                        # LLM评分是10分制，转换到100分制
                        llm_score_100 = llm_result.total_score * 10
                        c['llm_score'] = llm_score_100
                        c['combined_score'] = (
                            c['score'] * rule_weight +
                            llm_score_100 * llm_weight
                        )
                        logger.info(
                            "  候选%d: 规则=%.1f, LLM=%.1f (×10=%.1f), 综合=%.1f",
                            c['index'] + 1, c['score'],
                            llm_result.total_score, llm_score_100,
                            c['combined_score']
                        )
                    except Exception as e:
                        logger.warning("  候选%d LLM评估失败: %s", c['index'] + 1, e)
                        c['combined_score'] = c['score']

                # 清理LLM模型
                del llm_evaluator
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                import gc
                gc.collect()

                # 按综合分选最优
                top_candidates = scored_candidates[:top_k]
                best = max(top_candidates, key=lambda x: x.get(
                    'combined_score', x['score']))

                logger.info("  🏆 LLM挑选: 候选%d (综合=%.1f)",
                            best['index'] + 1,
                            best.get('combined_score', best['score']))

            except Exception as e:
                logger.error("LLM评估器加载失败: %s，回退到规则评分", e)
                best = scored_candidates[0]
        else:
            logger.info("🔍 [3/4] 纯规则评分筛选（未指定LLM模型）...")
            best = scored_candidates[0]
            logger.info("  🏆 规则最优: 候选%d (得分=%.1f)",
                        best['index'] + 1, best['score'])

        # ========== 第4步：返回结果 ==========
        final_eval = best['evaluation']
        final_score = best.get('combined_score', best['score'])

        logger.info("\n" + "=" * 60)
        logger.info("🎯 生成完成!")
        logger.info("  最终得分: %.1f (规则) / %.1f (综合)",
                    best['score'], final_score)
        logger.info("  有效候选: %d/%d", len(scored_candidates), len(candidates))
        if final_eval.issues:
            logger.info("  剩余问题: %d 个", len(final_eval.issues))
        logger.info("=" * 60)

        return OptimizedResult(
            layout=best['layout'],
            raw_output=best['raw_output'],
            score=best['score'],
            is_satisfactory=best['score'] >= 85.0,
            issues=final_eval.issues,
            suggestions=final_eval.suggestions,
            candidates_count=len(candidates),
            optimization_rounds=1,
            iteration_history=[{
                'iteration': 1,
                'query_type': '一次生成+LLM挑选',
                'num_candidates': len(candidates),
                'num_valid': len(scored_candidates),
                'best_score': best['score'],
                'llm_score': best.get('llm_score', 0),
                'combined_score': best.get('combined_score', best['score']),
            }],
        )

    def generate_raw(
        self,
        image_path: str,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        原始生成（复用gen.ipynb的推理代码）

        Args:
            image_path: 图片路径
            query: 查询文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            do_sample: 是否采样

        Returns:
            生成的文本
        """
        # 确保模型已加载
        self.load_model()

        from qwen_vl_utils import process_vision_info  # type: ignore

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]

        # 准备输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 生成
        import torch  # type: ignore[import-not-found]
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty
            )

        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0] if output_text else ""

    def parse_output(self, output_text: str) -> Dict[str, List[int]]:
        """解析模型输出为布局字典（委托给 common.parse_layout_json）"""
        return parse_layout_json(output_text)

    @staticmethod
    def _extract_json_str(text: str) -> str:
        """从模型输出中提取 JSON 字符串（委托给 common.extract_json_from_text）"""
        return extract_json_from_text(text) or text.strip()

    @staticmethod
    def _clean_json_str(s: str) -> str:
        """清理 LLM 常见的 JSON 格式错误（委托给 common.clean_json_str）"""
        return clean_json_str(s)

    @staticmethod
    def _validate_layout(data) -> Dict[str, List[int]]:
        """验证解析结果是否为合法布局字典"""
        if not isinstance(data, dict):
            return {}
        layout = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) == 4:
                try:
                    layout[k] = [int(x) for x in v]
                except (ValueError, TypeError):
                    continue
        return layout

    def generate(
        self,
        image_path: str,
        query: str,
        existing_layout: Optional[Dict[str, List[int]]] = None,
        config: Optional[GenerationConfig] = None
    ) -> LayoutResult:
        """
        生成布局并解析

        Args:
            image_path: 图片路径
            query: 查询文本
            existing_layout: 已有布局（用于评估）
            config: 生成配置

        Returns:
            LayoutResult: 生成结果
        """
        if config is None:
            config = GenerationConfig()

        # 生成原始输出
        raw_output = self.generate_raw(
            image_path=image_path,
            query=query,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty
        )

        # 解析输出
        layout = self.parse_output(raw_output)

        # 评估（如果提供了已有布局）
        score = 0.0
        issues = []
        is_valid = bool(layout)

        if layout and existing_layout:
            eval_result = self.evaluator.evaluate(layout, existing_layout)
            score = eval_result.total_score
            issues = eval_result.issues
            is_valid = eval_result.is_valid

        return LayoutResult(
            layout=layout,
            raw_output=raw_output,
            score=score,
            is_valid=is_valid,
            issues=issues
        )

    def generate_candidates(
        self,
        image_path: str,
        query: str,
        existing_layout: Optional[Dict[str, List[int]]] = None,
        num_candidates: int = 3,
        temperatures: Optional[List[float]] = None
    ) -> List[LayoutResult]:
        """
        生成多个候选布局

        Args:
            image_path: 图片路径
            query: 查询文本
            existing_layout: 已有布局
            num_candidates: 候选数量
            temperatures: 温度列表

        Returns:
            List[LayoutResult]: 候选结果列表
        """
        if temperatures is None:
            # 优化温度分布：包含低温（高确定性）和中温（多样性），去掉过高温度
            temperatures = [0.2, 0.4, 0.6, 0.75, 0.9][:num_candidates]

        candidates = []

        # 第一个候选：贪婪采样 (do_sample=False)，最接近训练数据分布
        greedy_config = GenerationConfig(temperature=0.0, do_sample=False)
        greedy_result = self.generate(
            image_path=image_path,
            query=query,
            existing_layout=existing_layout,
            config=greedy_config
        )
        candidates.append(greedy_result)
        logger.info("    候选1(贪婪): %s",
                    '有效' if greedy_result.layout else '解析失败')

        # 其余候选：不同温度采样
        for temp in temperatures[:num_candidates - 1]:
            config = GenerationConfig(temperature=temp)
            result = self.generate(
                image_path=image_path,
                query=query,
                existing_layout=existing_layout,
                config=config
            )
            candidates.append(result)

        return candidates

    def select_best(
        self,
        candidates: List[LayoutResult],
        existing_layout: Dict[str, List[int]]
    ) -> Tuple[Optional[LayoutResult], Optional[EvaluationResult]]:
        """
        从候选中选择最优结果

        Args:
            candidates: 候选列表
            existing_layout: 已有布局

        Returns:
            Tuple[最优结果, 评估结果]
        """
        best_result = None
        best_eval = None
        best_score = -1

        for candidate in candidates:
            if not candidate.layout:
                continue

            eval_result = self.evaluator.evaluate(
                candidate.layout, existing_layout)

            if eval_result.total_score > best_score:
                best_score = eval_result.total_score
                best_result = candidate
                best_eval = eval_result

        return best_result, best_eval

    def generate_optimized(
        self,
        image_path: str,
        query: str,
        existing_layout: Dict[str, List[int]],
        num_candidates: int = 5,
        score_threshold: float = 85.0,
        max_iterations: int = 3,
        auto_fix: bool = True,
        improvement_threshold: float = 3.0,
        llm_model_path: Optional[str] = None,
        llm_adapter_path: Optional[str] = None,
        llm_weight: float = 0.4,
    ) -> OptimizedResult:
        """
        完整优化生成流程（迭代优化 + 可选LLM二次挑选）：
        多候选生成 → 评估打分 → 选择最优 → 规则修复 → 识别问题 →
        注入问题到Prompt → 重新生成 → 循环直到满意
        → [可选] 卸载VL模型 → 加载LLM → 从候选池中精选最优

        Args:
            image_path: 图片路径
            query: 原始查询文本
            existing_layout: 已有布局参数
            num_candidates: 每轮候选数量
            score_threshold: 满意分数阈值（达到后停止）
            max_iterations: 最大迭代轮数
            auto_fix: 是否使用规则引擎自动修复
            improvement_threshold: 最小改进阈值（低于此值视为收敛）
            llm_model_path: LLM评估模型路径（提供则启用LLM二次挑选）
            llm_adapter_path: LLM评估模型LoRA路径
            llm_weight: LLM评分权重（0~1，规则权重=1-llm_weight）

        Returns:
            OptimizedResult: 包含完整优化历史的结果
        """
        best_layout: Optional[Dict[str, List[int]]] = None
        best_raw_output = ""
        best_score = 0.0
        best_eval: Optional[EvaluationResult] = None
        total_candidates = 0
        history: List[Dict[str, Any]] = []
        # 候选池：收集所有轮次中修复+后处理后的优质候选，供LLM最终挑选
        candidate_pool: List[Dict[str, Any]] = []

        current_query = query  # 初始查询
        consecutive_no_improve = 0  # 连续未改进轮数计数

        for iteration in range(max_iterations):
            iter_info = {
                'iteration': iteration + 1,
                'query_type': '初始查询' if iteration == 0 else '修正查询',
            }

            logger.info("\n" + "=" * 50)
            logger.info("🔄 第 %d/%d 轮优化", iteration + 1, max_iterations)
            logger.info("=" * 50)

            # ========== 第1步：多候选生成 ==========
            # 温度轮换：后续轮次偏移温度分布以增加多样性
            iter_temps = None
            if iteration > 0:
                base_temps = [0.2, 0.4, 0.6, 0.75, 0.9]
                offset = iteration * 0.08
                iter_temps = [min(t + offset, 1.0)
                              for t in base_temps[:num_candidates]]

            logger.info("  📝 生成 %d 个候选...", num_candidates)
            candidates = self.generate_candidates(
                image_path=image_path,
                query=current_query,
                existing_layout=existing_layout,
                num_candidates=num_candidates,
                temperatures=iter_temps
            )
            total_candidates += len(candidates)
            iter_info['num_candidates'] = len(candidates)

            # ========== 第2步：评估打分 + 验证 ==========
            logger.info("  🔍 评估候选结果...")
            candidate_details = []
            for i, cand in enumerate(candidates):
                if not cand.layout:
                    logger.warning("    候选%d: ❌ 解析失败", i + 1)
                    continue

                eval_result = self.evaluator.evaluate(
                    cand.layout, existing_layout)
                validation = self.rule_engine.validate(
                    cand.layout, existing_layout)

                candidate_details.append({
                    'index': i,
                    'layout': cand.layout,
                    'raw_output': cand.raw_output,
                    'score': eval_result.total_score,
                    'evaluation': eval_result,
                    'validation': validation,
                    'is_rule_valid': validation.valid
                })

                status = "✅" if validation.valid else "⚠️"
                logger.info("    候选%d: %s 得分=%.1f, 规则通过=%s",
                            i + 1, status, eval_result.total_score, validation.valid)

            iter_info['num_valid'] = sum(
                1 for c in candidate_details if c['is_rule_valid']
            )

            if not candidate_details:
                logger.warning("  ⚠️ 本轮无有效候选")
                iter_info['best_score'] = 0
                iter_info['issues'] = ['所有候选均解析失败']
                history.append(iter_info)
                continue

            # ========== 第3步：修复所有候选 + 尺寸优化 + 选最优 ==========
            if auto_fix:
                logger.info("  🔧 修复并优化所有候选...")
                for c in candidate_details:
                    try:
                        cur = c['layout']
                        # 规则修复（重叠、超界等硬性问题）
                        fix_result = self.rule_engine.validate_and_fix(
                            cur, existing_layout
                        )
                        if fix_result.fixed_layout:
                            cur = fix_result.fixed_layout
                        # 尺寸优化（满足100%最小标准，不引入新重叠）
                        cur = self.rule_engine.optimize_dimensions(
                            cur, existing_layout
                        )
                        # 注意：迭代过程中不对每个候选执行 aggressive_post_process
                        # 原因：该操作是确定性的，会将所有候选推向相同结果，抹平差异
                        # aggressive_post_process 仅在选出最优后执行一次
                        # 重新评分 + 重新校验规则
                        new_eval = self.evaluator.evaluate(
                            cur, existing_layout
                        )
                        new_validation = self.rule_engine.validate(
                            cur, existing_layout
                        )
                        # 无条件接受修复结果（修复流程不会恶化布局）
                        c['layout'] = cur
                        c['score'] = new_eval.total_score
                        c['evaluation'] = new_eval
                        c['is_rule_valid'] = new_validation.valid
                        # 打印修复后状态
                        fix_status = "✅" if new_validation.valid else "⚠️"
                        logger.info("    候选%d修复后: %s 得分=%.1f, 规则通过=%s%s",
                                    c['index'] +
                                    1, fix_status, new_eval.total_score,
                                    new_validation.valid,
                                    f" 残余违规={new_validation.hard_violations}" if not new_validation.valid else "")
                    except Exception as e:
                        logger.warning("    候选%d 修复异常: %s", c['index'] + 1, e)

            # 将本轮所有有效候选加入候选池（供最终LLM挑选）
            for c in candidate_details:
                candidate_pool.append({
                    'layout': c['layout'],
                    'raw_output': c.get('raw_output', ''),
                    'score': c['score'],
                    'evaluation': c['evaluation'],
                    'is_rule_valid': c['is_rule_valid'],
                    'iteration': iteration + 1,
                    'index': c['index'],
                })

            # 选择最优：优先选规则通过的，其次选得分最高的
            valid_candidates = [
                c for c in candidate_details if c['is_rule_valid']]
            if valid_candidates:
                round_best = max(valid_candidates, key=lambda x: x['score'])
            else:
                round_best = max(candidate_details, key=lambda x: x['score'])

            rule_status = "✅规则通过" if round_best['is_rule_valid'] else "⚠️规则未通过"
            num_valid = sum(1 for c in candidate_details if c['is_rule_valid'])
            logger.info("  🏆 本轮最优: 候选%d, 得分=%.1f, %s (通过率=%d/%d)",
                        round_best['index'] +
                        1, round_best['score'], rule_status,
                        num_valid, len(candidate_details))

            iter_info['best_score'] = round_best['score']
            iter_info['issues'] = round_best['evaluation'].issues

            round_layout = round_best['layout']
            round_raw = round_best['raw_output']
            round_eval = round_best['evaluation']

            # ========== 第4.5步：仅对本轮最优候选执行激进后处理 ==========
            if auto_fix:
                round_layout = self.rule_engine.aggressive_post_process(
                    round_layout, existing_layout
                )
                round_eval = self.evaluator.evaluate(
                    round_layout, existing_layout)
                logger.info("  🔧 最优候选后处理: 得分=%.1f", round_eval.total_score)

            # ========== 第5步：更新全局最优 ==========
            if round_eval.total_score > best_score:
                improvement = round_eval.total_score - best_score
                best_layout = round_layout
                best_raw_output = round_raw
                best_score = round_eval.total_score
                best_eval = round_eval
                consecutive_no_improve = 0
                logger.info("  ⬆️ 全局最优更新: %.1f (+%.1f)",
                            best_score, improvement)
                iter_info['improvement'] = improvement
            else:
                consecutive_no_improve += 1
                logger.info("  ➡️ 全局最优未变: %.1f (连续%d轮未改进)",
                            best_score, consecutive_no_improve)
                iter_info['improvement'] = 0

            history.append(iter_info)

            # ========== 第6步：检查终止条件 ==========
            if best_score >= score_threshold:
                logger.info("  ✅ 达到满意阈值 (%.1f), 停止优化", score_threshold)
                break

            # 连续2轮无改进 → 模型迭代已无法进一步提升，果断停止
            if consecutive_no_improve >= 2:
                logger.info("  📉 连续 %d 轮未改进, 模型迭代已收敛, 停止优化",
                            consecutive_no_improve)
                break

            # 单轮改进幅度不足但尚未触发连续停止，记录日志
            if iteration > 0 and iter_info.get('improvement', 0) < improvement_threshold:
                num_issues = len(best_eval.issues) if best_eval else 0
                logger.info("  📉 本轮改进不足 (%.1f < %.1f), 剩余问题=%d, 继续尝试",
                            iter_info.get('improvement', 0), improvement_threshold, num_issues)

            # ========== 第7步：构造修正Prompt ==========
            if iteration < max_iterations - 1 and round_eval.issues:
                current_query = self._build_fix_query(
                    original_query=query,
                    current_layout=round_layout,
                    issues=round_eval.issues,
                    existing_layout=existing_layout
                )
                logger.info("  📋 已注入 %d 个问题到下一轮Prompt", len(round_eval.issues))

        # ========== 最终结果 ==========
        if best_layout is None:
            return OptimizedResult(
                layout={},
                raw_output="",
                score=0,
                is_satisfactory=False,
                issues=["所有轮次均未生成有效布局"],
                suggestions=["请检查输入参数和图片路径"],
                candidates_count=total_candidates,
                optimization_rounds=len(history),
                iteration_history=history
            )

        # 对候选池中所有候选执行激进后处理 + 重新评分
        if auto_fix and candidate_pool:
            logger.info("\n🔧 候选池后处理（共 %d 个候选）...", len(candidate_pool))
            for c in candidate_pool:
                try:
                    processed = self.rule_engine.aggressive_post_process(
                        c['layout'], existing_layout)
                    c['layout'] = processed
                    c['evaluation'] = self.evaluator.evaluate(
                        processed, existing_layout)
                    c['score'] = c['evaluation'].total_score
                except Exception as e:
                    logger.warning("  候选池后处理异常: %s", e)
            # 按规则得分降序排列
            candidate_pool.sort(key=lambda x: x['score'], reverse=True)

        # ========== LLM二次挑选（如果提供了LLM模型路径） ==========
        if llm_model_path and candidate_pool:
            top_k = min(5, len(candidate_pool))
            logger.info("\n🤖 LLM二次挑选: 从候选池 Top-%d 中精选...", top_k)

            # 卸载VL模型释放显存
            self.unload_model()

            try:
                from core.llm_evaluator import LLMLayoutEvaluator
                llm_evaluator = LLMLayoutEvaluator(
                    model_path=llm_model_path,
                    adapter_path=llm_adapter_path,
                    device=self.device,
                )

                rule_weight = 1.0 - llm_weight

                for c in candidate_pool[:top_k]:
                    try:
                        llm_result = llm_evaluator.evaluate(
                            c['layout'], existing_layout)
                        llm_score_100 = llm_result.total_score * 10
                        c['llm_score'] = llm_score_100
                        c['combined_score'] = (
                            c['score'] * rule_weight +
                            llm_score_100 * llm_weight
                        )
                        logger.info(
                            "  第%d轮候选%d: 规则=%.1f, LLM=%.1f(×10=%.1f), 综合=%.1f",
                            c['iteration'], c['index'] + 1,
                            c['score'], llm_result.total_score,
                            llm_score_100, c['combined_score']
                        )
                    except Exception as e:
                        logger.warning("  LLM评估失败: %s", e)
                        c['combined_score'] = c['score']

                # 清理LLM模型
                del llm_evaluator
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                import gc
                gc.collect()

                # 按综合分选最优
                llm_best = max(
                    candidate_pool[:top_k],
                    key=lambda x: x.get('combined_score', x['score'])
                )
                best_layout = llm_best['layout']
                best_raw_output = llm_best.get('raw_output', '')
                best_eval = llm_best['evaluation']
                best_score = llm_best.get('combined_score', llm_best['score'])

                logger.info("  🏆 LLM最终挑选: 第%d轮候选%d (综合=%.1f, 规则=%.1f)",
                            llm_best['iteration'], llm_best['index'] + 1,
                            best_score, llm_best['score'])

            except Exception as e:
                logger.error("LLM评估器加载失败: %s，使用迭代最优结果", e)
                # 回退：使用候选池中规则评分最高的
                pool_best = candidate_pool[0]
                best_layout = pool_best['layout']
                best_raw_output = pool_best.get('raw_output', '')
                best_eval = pool_best['evaluation']
                best_score = pool_best['score']

        elif candidate_pool:
            # 无LLM模型：从候选池选规则评分最高的
            pool_best = candidate_pool[0]  # 已按得分排序
            if pool_best['score'] > best_score:
                best_layout = pool_best['layout']
                best_raw_output = pool_best.get('raw_output', '')
                best_eval = pool_best['evaluation']
                best_score = pool_best['score']
        else:
            # 候选池为空，对迭代最优做最终后处理
            if auto_fix:
                final_fix = self.rule_engine.validate_and_fix(
                    best_layout, existing_layout)
                if final_fix.fixed_layout:
                    best_layout = final_fix.fixed_layout
                best_layout = self.rule_engine.optimize_dimensions(
                    best_layout, existing_layout)
                best_layout = self.rule_engine.aggressive_post_process(
                    best_layout, existing_layout, max_passes=5)
                best_eval = self.evaluator.evaluate(
                    best_layout, existing_layout)

        # 确保 best_eval 不为 None
        if best_eval is None:
            best_eval = self.evaluator.evaluate(best_layout, existing_layout)

        llm_tag = " (+LLM挑选)" if llm_model_path else ""
        logger.info("\n" + "=" * 50)
        logger.info("🎯 优化完成!%s", llm_tag)
        logger.info("  最终得分: %.1f", best_eval.total_score)
        logger.info("  总候选数: %d（候选池: %d）",
                    total_candidates, len(candidate_pool))
        logger.info("  迭代轮数: %d", len(history))
        logger.info("  是否满意: %s", best_eval.total_score >= score_threshold)
        if best_eval.issues:
            logger.info("  剩余问题: %d 个", len(best_eval.issues))
        logger.info("=" * 50)

        return OptimizedResult(
            layout=best_layout,
            raw_output=best_raw_output,
            score=best_eval.total_score,
            is_satisfactory=best_eval.total_score >= score_threshold,
            issues=best_eval.issues,
            suggestions=best_eval.suggestions,
            candidates_count=total_candidates,
            optimization_rounds=len(history),
            iteration_history=history
        )

    def _build_fix_query(
        self,
        original_query: str,
        current_layout: Dict[str, List[int]],
        issues: List[str],
        existing_layout: Optional[Dict[str, List[int]]] = None
    ) -> str:
        """
        构造迭代修正查询：将评估问题转化为具体可操作的数值修正指令，
        而不只是笼统地列出问题名。

        例如 "宽度不足: 客厅 (最小3300mm)" → "客厅短边只有2100mm，需要≥3300mm"
        例如 "房间重叠: 卧室1 与 卧室2" → "卧室1和卧室2矩形区域重叠，请调整坐标使其不交叉"
        """
        import re

        # 过滤非模型可修复的issue（覆盖率、紧凑度、面积偏差由后处理解决）
        _POST_PROCESS_ONLY_KEYWORDS = ("覆盖率", "紧凑", "面积偏差")
        model_fixable_issues = [
            issue for issue in issues
            if not any(kw in issue for kw in _POST_PROCESS_ONLY_KEYWORDS)
        ]

        # 如果没有模型可修复的问题，直接返回原始query（避免无效迭代）
        if not model_fixable_issues:
            logger.info("  ℹ️ 剩余问题均为后处理可修复，不注入修正指令")
            return original_query

        # 将问题转化为具体修正指令
        fix_instructions = []
        for issue in model_fixable_issues[:5]:  # 最多5条，避免过长
            if "宽度不足" in issue or "长度不足" in issue:
                # 从 issue 提取房间名和最小值
                m = re.search(r'(宽度|长度)不足.*?:\s*(\S+)\s*\(最小(\d+)mm\)', issue)
                if m:
                    dim, room, min_val = m.group(1), m.group(2), m.group(3)
                    params = current_layout.get(room)
                    if params:
                        actual_short = min(params[2], params[3])
                        actual_long = max(params[2], params[3])
                        if dim == "宽度":
                            fix_instructions.append(
                                f"{room}短边={actual_short}mm不足，需≥{min_val}mm"
                            )
                        else:
                            fix_instructions.append(
                                f"{room}长边={actual_long}mm不足，需≥{min_val}mm"
                            )
                    else:
                        fix_instructions.append(issue)
                else:
                    fix_instructions.append(issue)
            elif "面积不足" in issue:
                m = re.search(r'面积不足.*?:\s*(\S+)\s*\(最小([\d.]+)平米\)', issue)
                if m:
                    room, min_area = m.group(1), m.group(2)
                    params = current_layout.get(room)
                    if params:
                        actual = params[2] * params[3] / 1_000_000
                        fix_instructions.append(
                            f"{room}面积={actual:.1f}㎡不足，需≥{min_area}㎡"
                        )
                    else:
                        fix_instructions.append(issue)
                else:
                    fix_instructions.append(issue)
            elif "面积过大" in issue:
                m = re.search(
                    r'面积过大.*?:\s*(\S+)\s*\(([\d.]+)平米.*?最大([\d.]+)平米\)', issue)
                if m:
                    room, actual_area, max_area = m.group(
                        1), m.group(2), m.group(3)
                    fix_instructions.append(
                        f"{room}面积={actual_area}m2过大，需≤{max_area}m2，请缩小尺寸"
                    )
                else:
                    fix_instructions.append(issue + "，请缩小房间尺寸")
            elif "重叠" in issue:
                # 区分基础设施重叠 vs 房间间重叠，给出具体坐标
                if "基础设施" in issue:
                    # 提取基础设施名和房间名，给出禁区范围
                    import re as _re
                    m = _re.search(r'(\S+)\s*与\s*(\S+)', issue)
                    if m:
                        rname, iname = m.group(1), m.group(2)
                        infra_params = None
                        _el = existing_layout or {}
                        for full_key in list(_el.keys()):
                            if iname in full_key:
                                infra_params = _el.get(full_key)
                                break
                        if infra_params:
                            fix_instructions.append(
                                f"{rname}与{iname}区域[{infra_params}]重叠，"
                                f"请将{rname}移到该矩形区域之外"
                            )
                        else:
                            fix_instructions.append(issue + "，请将房间移到基础设施区域之外")
                    else:
                        fix_instructions.append(issue + "，请将房间移到基础设施区域之外")
                else:
                    fix_instructions.append(issue + "，请调整坐标使其不交叉")
            elif "超出边界" in issue:
                fix_instructions.append(issue + "，请缩小尺寸或移动位置")
            elif "采光不足" in issue:
                fix_instructions.append(issue + "，请将其移到靠近采光面的位置")
            elif "入口" in issue and "客厅" in issue:
                # 提供入口坐标帮助模型定位
                entry_params = None
                _el = existing_layout or {}
                for ek, ev in _el.items():
                    if "入口" in ek:
                        entry_params = ev
                        break
                if entry_params:
                    fix_instructions.append(
                        f"客厅应靠近主入口[{entry_params}]，请将客厅移到入口附近"
                    )
                else:
                    fix_instructions.append("客厅应靠近入口，请调整位置")
            elif "不宜相邻" in issue:
                fix_instructions.append(issue + "，请拉开它们的距离")
            else:
                fix_instructions.append(issue)

        issues_text = "；".join(fix_instructions)
        layout_json = json.dumps(current_layout, ensure_ascii=False)

        return (
            f"{original_query}\n"
            f"上次生成的结果存在问题，请修正：{issues_text}。\n"
            f"上次结果：\n```json\n{layout_json}\n```\n"
            f"请输出修正后的完整JSON，格式为```json\n{{...}}\n```"
        )

    def evaluate(
        self,
        layout: Dict[str, List[int]],
        existing_layout: Dict[str, List[int]]
    ) -> EvaluationResult:
        """评估布局"""
        return self.evaluator.evaluate(layout, existing_layout)

    def validate(
        self,
        layout: Dict[str, List[int]],
        existing_layout: Dict[str, List[int]],
        auto_fix: bool = False
    ) -> ValidationResult:
        """验证布局"""
        if auto_fix:
            return self.rule_engine.validate_and_fix(layout, existing_layout)
        return self.rule_engine.validate(layout, existing_layout)


# ==================== 房间类型 → 尺寸映射（与 rules.yaml 一致） ====================
_ROOM_SIZE_SPEC = {
    "卧室":   {"w": 2400, "l": 3000, "a_min": 7.2,  "a_max": 18.0},
    "主卧":   {"w": 3000, "l": 3600, "a_min": 10.8, "a_max": 22.0},
    "客厅":   {"w": 3300, "l": 4500, "a_min": 14.85, "a_max": 35.0},
    "厨房":   {"w": 1800, "l": 2400, "a_min": 4.32, "a_max": 10.0},
    "卫生间": {"w": 1500, "l": 2100, "a_min": 3.15, "a_max": 7.0},
    "主卫":   {"w": 1800, "l": 2400, "a_min": 4.32, "a_max": 6.5},
    "餐厅":   {"w": 1500, "l": 2000, "a_min": 3.0,  "a_max": 18.0},
    "储藏":   {"w": 1200, "l": 1500, "a_min": 1.8,  "a_max": 8.0},
    "玄关":   {"w": 1200, "l": 1500, "a_min": 1.8,  "a_max": 15.0},
    "楼梯":   {"w": 2100, "l": 2400, "a_min": 5.04, "a_max": 9.0},
    "阳台":   {"w": 1200, "l": 2000, "a_min": 2.4,  "a_max": 8.0},
    "门廊":   {"w": 1200, "l": 2400, "a_min": 2.88, "a_max": 15.0},
    "次入口": {"w": 600,  "l": 600,  "a_min": 0.36, "a_max": 1.5},
}


def _room_type(name: str) -> str:
    """房间名 → 类型（卧室1→卧室）"""
    for t in _ROOM_SIZE_SPEC:
        if t in name:
            return t
    return name


def _build_size_constraints(rooms_to_generate: List[str]) -> str:
    """
    根据待生成房间列表，动态构建尺寸约束文本。
    包含最小尺寸和最大面积限制，避免房间过大或过小。
    """
    seen_types = set()
    lines = []
    for room in rooms_to_generate:
        rt = _room_type(room)
        if rt in _ROOM_SIZE_SPEC and rt not in seen_types:
            seen_types.add(rt)
            spec = _ROOM_SIZE_SPEC[rt]
            lines.append(
                f"{rt}: 短边≥{spec['w']}mm, 长边≥{spec['l']}mm, "
                f"面积{spec['a_min']}~{spec['a_max']}m2"
            )
    return "；".join(lines)


def build_query(
    house_type: str,
    floor_type: str,
    existing_params: Dict[str, List[int]],
    rooms_to_generate: List[str],
    design_constraints: Optional[str] = None,
    prompts_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    构建查询文本（增强版，注入量化约束）

    策略：保留与训练数据完全一致的主体格式（模型从这个模式中学会
    了 JSON 输出），在末尾追加简短的量化硬约束，用自然语言写，
    不破坏训练模式。
    """
    existing_json = json.dumps(existing_params, ensure_ascii=False)
    rooms_json = json.dumps(rooms_to_generate, ensure_ascii=False)

    # 从 existing_params 解析边界范围
    boundary = existing_params.get("边界", None)

    # ---- 主体：与训练数据格式完全一致 ----
    query = (
        f'请根据这张图片中已有的户型信息以及对应的参数，帮我生成其余房间的参数，'
        f'得到一个完整的合理平面布局。构成户型的所有空间单元均表示为矩形，'
        f'用x轴坐标、y轴坐标、长度、宽度四个参数表示。'
        f'本户型为"{house_type}"住宅，图片中的为"{floor_type}"平面。\n'
        f'图片中已有信息对应的参数为：\n'
        f'```json\n{existing_json}\n```'
        f'其余待生成的"{floor_type}"房间的名称为：\n'
        f'```json\n{rooms_json}```'
    )

    # ---- 追加：简短量化约束（自然语言，不影响 JSON 输出格式） ----
    constraints = []
    if boundary and len(boundary) == 4:
        bx, by, bw, bh = boundary
        constraints.append(
            f"所有房间的x坐标≥{bx}，y坐标≥{by}，"
            f"x+长度≤{bx+bw}，y+宽度≤{by+bh}"
        )
    constraints.append("任意两个房间的矩形区域不能重叠")

    size_text = _build_size_constraints(rooms_to_generate)
    if size_text:
        constraints.append(f"尺寸要求：{size_text}")

    constraints.append("厨房不宜与卫生间直接相邻")
    constraints.append("主卫与卫生间（公卫）不宜直接相邻")
    constraints.append("客厅、卧室应靠近采光面")
    constraints.append("客厅应靠近主入口")
    constraints.append("餐厅应与厨房相邻")
    constraints.append("房间应尽量填满边界空间，避免大面积空白")
    constraints.append("房间长宽比不宜超过3:1")
    constraints.append("房间不能与已有的采光区、黑体区、主入口区域重叠")
    constraints.append("坐标和尺寸取300的整数倍（建筑模数对齐）")
    constraints.append("只输出待生成房间的参数，不要包含已有房间")
    constraints.append("卫生间、主卫面积不宜超过7m2")

    query += "\n注意：" + "；".join(constraints) + "。"
    query += "\n请直接输出JSON，格式为```json\n{...}\n```，只包含待生成的房间。"

    return query


# 便捷函数
def create_predictor(
    base_model_path: str = "models/Qwen2.5-VL-7B-Instruct",
    lora_adapter_path: str = "lora_model",
    **kwargs
) -> LayoutPredictor:
    """创建预测器实例"""
    return LayoutPredictor(
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        **kwargs
    )


if __name__ == "__main__":
    # 测试（不加载模型，仅测试评估功能）
    logger.info("测试 LayoutPredictor（评估功能）...")

    predictor = LayoutPredictor()

    # 测试评估
    existing = {
        "边界": [0, 0, 9600, 10500],
        "南采光": [0, -1200, 9600, 1200],
    }

    generated = {
        "客厅": [0, 0, 4000, 4000],
        "卧室1": [0, 4500, 3300, 4000],
        "厨房": [4500, 0, 2400, 3000],
    }

    result = predictor.evaluate(generated, existing)
    logger.info("评估得分: %.1f", result.total_score)
    logger.info("问题: %s", result.issues)

    # 测试验证
    val_result = predictor.validate(generated, existing)
    logger.info("验证通过: %s", val_result.valid)

    # 测试查询构建
    query = build_query(
        house_type="城市",
        floor_type="一层",
        existing_params=existing,
        rooms_to_generate=["客厅", "卧室1", "厨房"]
    )
    logger.info("\n构建的查询:\n%s...", query[:200])

    logger.info("\n测试完成!")

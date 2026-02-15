"""
户型布局生成主流程
整合所有优化模块的端到端流程
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from core import (
    LayoutResult,
    GenerationConfig,
    LayoutEvaluator, 
    EvaluationResult,
    LayoutRuleEngine,
    ValidationResult,
    MultiCandidateOptimizer,
    OptimizationResult,
    select_best_candidate,
    # GPU类通过延迟导入
    get_layout_generator,
    get_layout_optimizer,
)
from rag import (
    LayoutEmbedder,
    LayoutKnowledgeBase,
    LayoutRetriever,
    RAGGenerator
)
from utils import (
    LayoutVisualizer,
    LayoutMetrics,
    visualize_evaluation_result
)


@dataclass
class PipelineConfig:
    """流程配置"""
    # 模型配置
    base_model_path: str = "models/Qwen2.5-VL-7B-Instruct"
    lora_adapter_path: str = "lora_model"
    device: str = "cuda"
    
    # 生成配置
    num_candidates: int = 5
    temperature_range: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8, 0.9, 1.0])
    max_new_tokens: int = 256
    
    # 优化配置
    score_threshold: float = 85.0
    max_iterations: int = 3
    enable_auto_fix: bool = True
    
    # RAG配置
    enable_rag: bool = False
    knowledge_base_path: str = None
    rag_top_k: int = 3
    
    # LLM评估器配置
    enable_llm_evaluator: bool = False
    llm_evaluator_model_path: str = None  # 如果为None，使用base_model_path
    llm_evaluator_adapter_path: str = None  # LLM评估器的LoRA适配器路径
    llm_evaluator_weight: float = 0.4  # LLM评估权重
    
    # 配置文件路径
    rules_config_path: str = "config/rules.yaml"
    prompts_config_path: str = "config/prompts.yaml"


@dataclass
class PipelineResult:
    """流程执行结果"""
    layout: Dict[str, List[int]]
    score: float
    evaluation: EvaluationResult
    validation: ValidationResult
    candidates_count: int
    iterations: int
    is_satisfactory: bool
    enhanced_prompt: str = ""
    reference_cases: List[Any] = field(default_factory=list)
    llm_evaluation: Dict[str, Any] = None  # LLM评估结果


class LayoutGenerationPipeline:
    """户型布局生成流程"""
    
    def __init__(self, config: PipelineConfig = None):
        """
        初始化流程
        
        Args:
            config: 流程配置
        """
        self.config = config or PipelineConfig()
        
        # 初始化组件
        self._init_components()
        
        # 加载提示词配置
        self._load_prompts_config()
    
    def _init_components(self):
        """初始化各个组件"""
        # 规则配置路径
        rules_path = Path(self.config.rules_config_path)
        rules_config = str(rules_path) if rules_path.exists() else None
        
        # 初始化评估器和规则引擎
        self.evaluator = LayoutEvaluator(rules_config)
        self.rule_engine = LayoutRuleEngine(rules_config)
        
        # 初始化指标计算器
        self.metrics = LayoutMetrics()
        
        # 初始化可视化器
        self.visualizer = LayoutVisualizer()
        
        # 延迟初始化生成器（需要GPU）
        self._generator = None
        
        # RAG组件（可选）
        self._rag_generator = None
        self._knowledge_base = None
        
        if self.config.enable_rag and self.config.knowledge_base_path:
            self._init_rag()
        
        # LLM评估器（可选）
        self._llm_evaluator = None
        self._hybrid_evaluator = None
        
        if self.config.enable_llm_evaluator:
            self._init_llm_evaluator()
    
    def _init_rag(self):
        """初始化RAG组件"""
        try:
            embedder = LayoutEmbedder()
            self._knowledge_base = LayoutKnowledgeBase(embedder)
            
            if Path(self.config.knowledge_base_path).exists():
                self._knowledge_base.load(self.config.knowledge_base_path)
            
            retriever = LayoutRetriever(self._knowledge_base, top_k=self.config.rag_top_k)
            self._rag_generator = RAGGenerator(retriever)
        except Exception as e:
            print(f"RAG初始化失败: {e}")
            self._rag_generator = None
    
    def _init_llm_evaluator(self, model_path: str = None, adapter_path: str = None):
        """
        初始化LLM评估器
        
        Args:
            model_path: 基座模型路径（可选，覆盖配置）
            adapter_path: LoRA适配器路径（可选，覆盖配置）
        """
        try:
            from core.llm_evaluator import LLMLayoutEvaluator, HybridLayoutEvaluator
            
            # 使用指定的模型路径，或配置中的路径，或默认使用base_model_path
            llm_model_path = model_path or self.config.llm_evaluator_model_path or self.config.base_model_path
            llm_adapter_path = adapter_path or self.config.llm_evaluator_adapter_path
            
            print(f"正在初始化LLM评估器:")
            print(f"  基座模型: {llm_model_path}")
            if llm_adapter_path:
                print(f"  LoRA适配器: {llm_adapter_path}")
            
            self._llm_evaluator = LLMLayoutEvaluator(
                model_path=llm_model_path,
                adapter_path=llm_adapter_path,
                device=self.config.device
            )
            
            # 创建混合评估器
            self._hybrid_evaluator = HybridLayoutEvaluator(
                rule_evaluator=self.evaluator,
                llm_evaluator=self._llm_evaluator,
                llm_weight=self.config.llm_evaluator_weight
            )
            
            print("LLM评估器初始化完成")
        except Exception as e:
            print(f"LLM评估器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._llm_evaluator = None
            self._hybrid_evaluator = None
    
    def _load_prompts_config(self):
        """加载提示词配置"""
        prompts_path = Path(self.config.prompts_config_path)
        
        if prompts_path.exists():
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts_config = yaml.safe_load(f)
        else:
            self.prompts_config = {
                'design_constraints': """设计约束：
1. 厨房与卫生间不宜直接相邻
2. 卧室应尽量靠近采光面
3. 客厅应有良好的采光和通风
4. 动静分区应合理（公共区与私密区分离）
5. 房间尺寸应符合人体工程学标准""",
                'base_prompt': """请根据这张图片中已有的户型信息以及对应的参数，帮我生成其余房间的参数，得到一个完整的合理平面布局。
构成户型的所有空间单元均表示为矩形，用x轴坐标、y轴坐标、长度、宽度四个参数表示。
本户型为"{house_type}"住宅，图片中的为"{floor_type}"平面。

{design_constraints}

图片中已有信息对应的参数为：
```json
{existing_params}
```
其余待生成的"{floor_type}"房间的名称为：
```json
{rooms_to_generate}
```"""
            }
    
    @property
    def generator(self):
        """延迟加载生成器"""
        if self._generator is None:
            LayoutGenerator = get_layout_generator()
            self._generator = LayoutGenerator(
                base_model_path=self.config.base_model_path,
                lora_adapter_path=self.config.lora_adapter_path,
                device=self.config.device
            )
        return self._generator
    
    def build_prompt(
        self,
        existing_params: Dict[str, List[int]],
        rooms_to_generate: List[str],
        house_type: str = "城市",
        floor_type: str = "一层",
        use_rag: bool = None
    ) -> str:
        """
        构建增强提示词
        
        Args:
            existing_params: 已有房间参数
            rooms_to_generate: 待生成房间
            house_type: 户型类型
            floor_type: 楼层类型
            use_rag: 是否使用RAG
            
        Returns:
            str: 提示词
        """
        # 是否启用RAG
        if use_rag is None:
            use_rag = self.config.enable_rag
        
        # 使用RAG增强
        if use_rag and self._rag_generator:
            return self._rag_generator.build_enhanced_prompt(
                existing_layout=existing_params,
                rooms_to_generate=rooms_to_generate,
                house_type=house_type,
                floor_type=floor_type,
                top_k=self.config.rag_top_k
            )
        
        # 设计约束
        design_constraints = self.prompts_config.get('design_constraints', '')
        
        # 尝试使用分类模板 (城市住宅/商场 × 一层/多层/标准层/特殊层)
        classified_templates = self.prompts_config.get('classified_templates', {})
        template = None
        if classified_templates:
            # 匹配分类：城市→城市住宅, 商场→商场
            type_key = '城市住宅' if '城市' in house_type or '乡村' in house_type else house_type
            type_templates = classified_templates.get(type_key, {})
            if type_templates:
                # 匹配楼层: 一层→一层, 二层→多层, 标准层→标准层
                if '一层' in floor_type:
                    template = type_templates.get('一层')
                elif floor_type in ('标准层',):
                    template = type_templates.get('标准层')
                elif floor_type in ('特殊层',):
                    template = type_templates.get('特殊层')
                else:
                    template = type_templates.get('多层')
        
        if template:
            # 使用分类模板
            try:
                prompt = template.format(
                    house_type=house_type,
                    floor_type=floor_type,
                    design_constraints=design_constraints,
                    existing_params=json.dumps(existing_params, ensure_ascii=False),
                    rooms_to_generate=json.dumps(rooms_to_generate, ensure_ascii=False)
                )
                return prompt
            except (KeyError, IndexError):
                pass  # 模板格式错误，降级到base_prompt
        
        # 降级：使用通用base_prompt
        base_prompt = self.prompts_config.get('base_prompt', '')
        prompt = base_prompt.format(
            house_type=house_type,
            floor_type=floor_type,
            design_constraints=design_constraints,
            existing_params=json.dumps(existing_params, ensure_ascii=False),
            rooms_to_generate=json.dumps(rooms_to_generate, ensure_ascii=False)
        )
        
        return prompt
    
    def generate(
        self,
        image_path: str,
        existing_params: Dict[str, List[int]],
        rooms_to_generate: List[str],
        house_type: str = "城市",
        floor_type: str = "一层",
        optimize: bool = True
    ) -> PipelineResult:
        """
        执行生成流程
        
        Args:
            image_path: 输入图片路径
            existing_params: 已有房间参数
            rooms_to_generate: 待生成房间
            house_type: 户型类型
            floor_type: 楼层类型
            optimize: 是否优化
            
        Returns:
            PipelineResult: 生成结果
        """
        # 构建提示词
        prompt = self.build_prompt(
            existing_params,
            rooms_to_generate,
            house_type,
            floor_type
        )
        
        if optimize:
            # 使用优化流程
            return self._generate_with_optimization(
                image_path,
                prompt,
                existing_params
            )
        else:
            # 简单生成
            return self._generate_simple(
                image_path,
                prompt,
                existing_params
            )
    
    def _generate_simple(
        self,
        image_path: str,
        prompt: str,
        existing_params: Dict[str, List[int]]
    ) -> PipelineResult:
        """简单生成（不优化）"""
        result = self.generator.generate(image_path, prompt)
        
        # 评估
        evaluation = self.evaluator.evaluate(result.layout, existing_params)
        
        # 验证
        validation = self.rule_engine.validate(result.layout, existing_params)
        
        # LLM评估（如果启用）
        llm_eval_result = None
        final_score = evaluation.total_score
        
        if self._hybrid_evaluator:
            try:
                llm_eval_result = self._hybrid_evaluator.evaluate(result.layout, existing_params)
                final_score = llm_eval_result['combined_score']
            except Exception as e:
                print(f"LLM评估失败: {e}")
        
        return PipelineResult(
            layout=result.layout,
            score=final_score,
            evaluation=evaluation,
            validation=validation,
            candidates_count=1,
            iterations=1,
            is_satisfactory=final_score >= self.config.score_threshold,
            enhanced_prompt=prompt,
            llm_evaluation=llm_eval_result
        )
    
    def _generate_with_optimization(
        self,
        image_path: str,
        prompt: str,
        existing_params: Dict[str, List[int]]
    ) -> PipelineResult:
        """带优化的生成流程"""
        # 创建优化器
        LayoutOptimizer = get_layout_optimizer()
        optimizer = LayoutOptimizer(
            generator=self.generator,
            evaluator=self.evaluator,
            rule_engine=self.rule_engine,
            score_threshold=self.config.score_threshold,
            max_iterations=self.config.max_iterations
        )
        
        # 执行优化
        opt_result = optimizer.optimize(
            image_path=image_path,
            initial_query=prompt,
            existing_layout=existing_params,
            num_candidates=self.config.num_candidates
        )
        
        # LLM评估（如果启用）
        llm_eval_result = None
        final_score = opt_result.final_score
        
        if self._hybrid_evaluator:
            try:
                llm_eval_result = self._hybrid_evaluator.evaluate(
                    opt_result.final_layout, 
                    existing_params
                )
                final_score = llm_eval_result['combined_score']
            except Exception as e:
                print(f"LLM评估失败: {e}")
        
        return PipelineResult(
            layout=opt_result.final_layout,
            score=final_score,
            evaluation=opt_result.evaluation,
            validation=opt_result.validation,
            candidates_count=sum(h.get('num_candidates', 0) for h in opt_result.history),
            iterations=opt_result.iterations,
            is_satisfactory=final_score >= self.config.score_threshold,
            enhanced_prompt=prompt,
            llm_evaluation=llm_eval_result
        )
    
    def generate_and_visualize(
        self,
        image_path: str,
        existing_params: Dict[str, List[int]],
        rooms_to_generate: List[str],
        output_path: str = None,
        **kwargs
    ) -> Tuple[PipelineResult, Any]:
        """
        生成并可视化结果
        
        Args:
            image_path: 输入图片路径
            existing_params: 已有房间参数
            rooms_to_generate: 待生成房间
            output_path: 输出图片路径
            **kwargs: 其他参数
            
        Returns:
            Tuple[PipelineResult, Figure]: 结果和图形
        """
        # 生成
        result = self.generate(
            image_path,
            existing_params,
            rooms_to_generate,
            **kwargs
        )
        
        # 合并布局
        full_layout = {**existing_params, **result.layout}
        
        # 可视化
        fig = visualize_evaluation_result(
            full_layout,
            result.evaluation,
            save_path=output_path
        )
        
        return result, fig
    
    def evaluate_layout(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None,
        use_llm: bool = None
    ) -> Dict[str, Any]:
        """
        评估布局
        
        Args:
            layout: 待评估布局
            full_layout: 完整布局
            use_llm: 是否使用LLM评估（None时根据配置决定）
            
        Returns:
            评估结果字典
        """
        # 规则评估
        rule_result = self.evaluator.evaluate(layout, full_layout)
        
        result = {
            "rule_evaluation": {
                "total_score": rule_result.total_score,
                "dimension_scores": rule_result.dimension_scores,
                "issues": rule_result.issues,
                "suggestions": rule_result.suggestions,
                "is_valid": rule_result.is_valid
            },
            "llm_evaluation": None,
            "combined_score": rule_result.total_score,
            "is_valid": rule_result.is_valid
        }
        
        # 是否使用LLM评估
        should_use_llm = use_llm if use_llm is not None else self.config.enable_llm_evaluator
        
        if should_use_llm and self._hybrid_evaluator:
            try:
                hybrid_result = self._hybrid_evaluator.evaluate(layout, full_layout)
                result["llm_evaluation"] = hybrid_result.get("llm_evaluation")
                result["combined_score"] = hybrid_result["combined_score"]
                result["combined_issues"] = hybrid_result.get("combined_issues", rule_result.issues)
                result["combined_suggestions"] = hybrid_result.get("combined_suggestions", rule_result.suggestions)
            except Exception as e:
                print(f"LLM评估失败: {e}")
        
        return result
    
    def evaluate_layout_with_llm(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None
    ) -> Dict[str, Any]:
        """
        使用LLM评估布局（独立方法）
        
        Args:
            layout: 待评估布局
            full_layout: 完整布局
            
        Returns:
            LLM评估结果
        """
        if self._llm_evaluator is None:
            # 尝试初始化
            if not self.config.enable_llm_evaluator:
                self.config.enable_llm_evaluator = True
            self._init_llm_evaluator()
        
        if self._llm_evaluator is None:
            raise RuntimeError("LLM评估器初始化失败，请检查模型路径配置")
        
        result = self._llm_evaluator.evaluate(layout, full_layout)
        
        return {
            "total_score": result.total_score,
            "dimension_scores": result.dimension_scores,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "is_valid": result.is_valid,
            "confidence": result.confidence,
            "raw_response": result.raw_response
        }
    
    def validate_layout(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None
    ) -> ValidationResult:
        """
        验证布局
        
        Args:
            layout: 待验证布局
            full_layout: 完整布局
            
        Returns:
            ValidationResult: 验证结果
        """
        return self.rule_engine.validate(layout, full_layout)
    
    def add_to_knowledge_base(
        self,
        layout: Dict[str, List[int]],
        metadata: Dict[str, Any] = None,
        score: float = None
    ):
        """
        添加案例到知识库
        
        Args:
            layout: 布局
            metadata: 元数据
            score: 质量分数
        """
        if self._knowledge_base is None:
            embedder = LayoutEmbedder()
            self._knowledge_base = LayoutKnowledgeBase(embedder)
        
        if score is None:
            evaluation = self.evaluator.evaluate(layout)
            score = evaluation.total_score
        
        self._knowledge_base.add_case(layout, metadata, score)
    
    def save_knowledge_base(self, path: str = None):
        """保存知识库"""
        if self._knowledge_base:
            save_path = path or self.config.knowledge_base_path or "data/knowledge_base.pkl"
            self._knowledge_base.save(save_path)


def create_pipeline(
    base_model_path: str = None,
    lora_adapter_path: str = None,
    enable_llm_evaluator: bool = False,
    llm_evaluator_model_path: str = None,
    **kwargs
) -> LayoutGenerationPipeline:
    """
    创建生成流程
    
    Args:
        base_model_path: 基础模型路径（用于生成）
        lora_adapter_path: LoRA适配器路径
        enable_llm_evaluator: 是否启用LLM评估器
        llm_evaluator_model_path: LLM评估器模型路径（如果为None，使用base_model_path）
        **kwargs: 其他配置参数
        
    Returns:
        LayoutGenerationPipeline: 生成流程实例
    """
    config = PipelineConfig(
        base_model_path=base_model_path or "models/Qwen2.5-VL-7B-Instruct",
        lora_adapter_path=lora_adapter_path or "lora_model",
        enable_llm_evaluator=enable_llm_evaluator,
        llm_evaluator_model_path=llm_evaluator_model_path,
        **kwargs
    )
    
    return LayoutGenerationPipeline(config)


def create_pipeline_with_llm_evaluator(
    base_model_path: str,
    lora_adapter_path: str = None,
    llm_evaluator_model_path: str = None,
    llm_weight: float = 0.4,
    **kwargs
) -> LayoutGenerationPipeline:
    """
    创建带LLM评估器的生成流程
    
    Args:
        base_model_path: 基础模型路径（用于生成）
        lora_adapter_path: LoRA适配器路径
        llm_evaluator_model_path: LLM评估器模型路径（如果为None，使用base_model_path）
        llm_weight: LLM评估权重（0-1）
        **kwargs: 其他配置参数
        
    Returns:
        LayoutGenerationPipeline: 生成流程实例
    """
    return create_pipeline(
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        enable_llm_evaluator=True,
        llm_evaluator_model_path=llm_evaluator_model_path or base_model_path,
        llm_evaluator_weight=llm_weight,
        **kwargs
    )


# ==================== Qwen14B 微调模型专用配置 ====================

# 默认路径配置
# Linux服务器: /home/nju/.cache/modelscope/hub/models/Qwen/Qwen2___5-14B-Instruct
# Windows本地: 相对路径或HuggingFace model ID
# 可通过参数 qwen14b_base_path / qwen14b_adapter_path 覆盖
import os as _os
if _os.name == 'nt':  # Windows
    QWEN14B_DEFAULT_PATHS = {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",  # HuggingFace ID, 自动下载
        "adapter": str(Path(__file__).parent.parent / "qwen14b" / "Qwen2.5-14B-Instruct" / "Qwen2.5-14B-Instruct" / "lora" / "train_2025-12-01-21-17-23")
    }
else:  # Linux
    QWEN14B_DEFAULT_PATHS = {
        "base_model": "/home/nju/.cache/modelscope/hub/models/Qwen/Qwen2___5-14B-Instruct",
        "adapter": "/saves/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23"
    }


def create_pipeline_with_qwen14b_evaluator(
    base_model_path: str = "models/Qwen2.5-VL-7B-Instruct",
    lora_adapter_path: str = "lora_model",
    qwen14b_base_path: str = None,
    qwen14b_adapter_path: str = None,
    llm_weight: float = 0.4,
    **kwargs
) -> LayoutGenerationPipeline:
    """
    创建使用 Qwen14B 微调模型作为评估器的生成流程
    
    这是推荐的配置方式：
    - 生成模型: Qwen2.5-VL-7B-Instruct + LoRA（视觉语言模型，可以看图生成）
    - 评估模型: Qwen2.5-14B-Instruct + LoRA（更大的文本模型，评估更准确）
    
    Args:
        base_model_path: 生成模型基座路径 (Qwen2.5-VL-7B-Instruct)
        lora_adapter_path: 生成模型LoRA路径
        qwen14b_base_path: 评估模型基座路径（默认使用服务器路径）
        qwen14b_adapter_path: 评估模型LoRA路径（默认使用服务器路径）
        llm_weight: LLM评估权重（0-1）
        **kwargs: 其他配置参数
        
    Returns:
        LayoutGenerationPipeline: 生成流程实例
        
    使用示例:
        # 方式1：使用默认路径
        pipeline = create_pipeline_with_qwen14b_evaluator()
        
        # 方式2：指定自定义路径
        pipeline = create_pipeline_with_qwen14b_evaluator(
            base_model_path="models/Qwen2.5-VL-7B-Instruct",
            lora_adapter_path="lora_model",
            qwen14b_base_path="/path/to/Qwen2.5-14B-Instruct",
            qwen14b_adapter_path="/path/to/lora/train_xxx"
        )
        
        # 生成
        result = pipeline.generate(
            image_path="test.jpg",
            existing_params={...},
            rooms_to_generate=["客厅", "卧室"],
            optimize=True
        )
    """
    # 使用默认的 Qwen14B 路径
    qwen14b_base = qwen14b_base_path or QWEN14B_DEFAULT_PATHS["base_model"]
    qwen14b_adapter = qwen14b_adapter_path or QWEN14B_DEFAULT_PATHS["adapter"]
    
    print(f"创建使用 Qwen14B 评估器的 Pipeline:")
    print(f"  生成模型: {base_model_path} + {lora_adapter_path}")
    print(f"  评估模型: {qwen14b_base} + {qwen14b_adapter}")
    
    config = PipelineConfig(
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        enable_llm_evaluator=True,
        llm_evaluator_model_path=qwen14b_base,
        llm_evaluator_adapter_path=qwen14b_adapter,
        llm_evaluator_weight=llm_weight,
        **kwargs
    )
    
    return LayoutGenerationPipeline(config)

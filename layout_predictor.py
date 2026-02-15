"""
户型布局生成器 - 整合版
复用现有的predictor.py代码结构，集成优化功能
"""

import json
import yaml
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入评估和规则模块（不需要GPU）
from core import LayoutEvaluator, LayoutRuleEngine, ValidationResult
from core.evaluator import EvaluationResult
from core.generator import LayoutResult, GenerationConfig

logger = logging.getLogger(__name__)


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
        rules_config_path: str = None,
        prompts_config_path: str = None
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
        self.model = None
        self.processor = None
        
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
    
    def _resolve_config_path(self, explicit_path: str, default_relative: str) -> Optional[str]:
        """解析配置文件路径"""
        if explicit_path and Path(explicit_path).exists():
            return explicit_path
        default_path = self._project_root / default_relative
        if default_path.exists():
            return str(default_path)
        return None
    
    def _load_prompts_config(self, config_path: str = None) -> Dict:
        """加载提示词配置"""
        path = self._resolve_config_path(config_path, "config/prompts.yaml")
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
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
        
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel
        
        print(f"正在加载基础模型: {self.base_model_path}")
        
        if self.use_flash_attention:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        # 加载LoRA适配器
        if self.lora_adapter_path:
            print(f"正在加载LoRA适配器: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
            self.model = self.model.half()
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_path, 
            use_fast=True
        )
        
        self._model_loaded = True
        print("模型加载完成")
    
    def generate_raw(
        self,
        image_path: str,
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
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
        
        from qwen_vl_utils import process_vision_info
        
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
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
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
        """解析模型输出为布局字典"""
        try:
            # 提取JSON部分
            if "```json" in output_text:
                json_str = output_text.split("```json")[1].split("```")[0].strip()
            elif "```" in output_text:
                json_str = output_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = output_text.strip()
            
            layout = json.loads(json_str)
            return layout
        except (json.JSONDecodeError, IndexError) as e:
            print(f"解析输出失败: {e}")
            return {}
    
    def generate(
        self,
        image_path: str,
        query: str,
        existing_layout: Dict[str, List[int]] = None,
        config: GenerationConfig = None
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
            do_sample=config.do_sample
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
        existing_layout: Dict[str, List[int]] = None,
        num_candidates: int = 3,
        temperatures: List[float] = None
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
            temperatures = [0.3, 0.5, 0.7, 0.9, 1.1][:num_candidates]
        
        candidates = []
        for temp in temperatures:
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
    ) -> Tuple[LayoutResult, EvaluationResult]:
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
            
            eval_result = self.evaluator.evaluate(candidate.layout, existing_layout)
            
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
        improvement_threshold: float = 3.0
    ) -> OptimizedResult:
        """
        完整优化生成流程：
        多候选生成 → 评估打分 → 选择最优 → 规则修复 → 识别问题 → 
        注入问题到Prompt → 重新生成 → 循环直到满意
        
        实现优化技术方案中的迭代优化策略：
        1. 多样性生成：通过不同温度采样产生多个候选
        2. 评分选择：对候选进行五维度评估，选择最优
        3. 规则修复：对最优候选进行硬性规则修复
        4. 迭代修正：将本轮问题注入Prompt，引导模型在下一轮避免
        
        Args:
            image_path: 图片路径
            query: 原始查询文本
            existing_layout: 已有布局参数
            num_candidates: 每轮候选数量
            score_threshold: 满意分数阈值（达到后停止）
            max_iterations: 最大迭代轮数
            auto_fix: 是否使用规则引擎自动修复
            improvement_threshold: 最小改进阈值（低于此值视为收敛）
            
        Returns:
            OptimizedResult: 包含完整优化历史的结果
        """
        best_layout = None
        best_raw_output = ""
        best_score = 0.0
        best_eval = None
        total_candidates = 0
        history = []
        
        current_query = query  # 初始查询
        
        for iteration in range(max_iterations):
            iter_info = {
                'iteration': iteration + 1,
                'query_type': '初始查询' if iteration == 0 else '修正查询',
            }
            
            print(f"\n{'='*50}")
            print(f"🔄 第 {iteration + 1}/{max_iterations} 轮优化")
            print(f"{'='*50}")
            
            # ========== 第1步：多候选生成 ==========
            print(f"  📝 生成 {num_candidates} 个候选...")
            candidates = self.generate_candidates(
                image_path=image_path,
                query=current_query,
                existing_layout=existing_layout,
                num_candidates=num_candidates
            )
            total_candidates += len(candidates)
            iter_info['num_candidates'] = len(candidates)
            
            # ========== 第2步：评估打分 + 验证 ==========
            print(f"  🔍 评估候选结果...")
            candidate_details = []
            for i, cand in enumerate(candidates):
                if not cand.layout:
                    print(f"    候选{i+1}: ❌ 解析失败")
                    continue
                
                eval_result = self.evaluator.evaluate(cand.layout, existing_layout)
                validation = self.rule_engine.validate(cand.layout, existing_layout)
                
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
                print(f"    候选{i+1}: {status} 得分={eval_result.total_score:.1f}, "
                      f"规则通过={validation.valid}")
            
            iter_info['num_valid'] = sum(
                1 for c in candidate_details if c['is_rule_valid']
            )
            
            if not candidate_details:
                print(f"  ⚠️ 本轮无有效候选")
                iter_info['best_score'] = 0
                iter_info['issues'] = ['所有候选均解析失败']
                history.append(iter_info)
                continue
            
            # ========== 第3步：选择最优候选 ==========
            # 优先选择通过硬性规则验证的
            valid_candidates = [c for c in candidate_details if c['is_rule_valid']]
            pool = valid_candidates if valid_candidates else candidate_details
            round_best = max(pool, key=lambda x: x['score'])
            
            print(f"  🏆 本轮最优: 候选{round_best['index']+1}, "
                  f"得分={round_best['score']:.1f}")
            
            iter_info['best_score'] = round_best['score']
            iter_info['issues'] = round_best['evaluation'].issues
            
            # ========== 第4步：规则引擎修复 ==========
            round_layout = round_best['layout']
            round_raw = round_best['raw_output']
            round_eval = round_best['evaluation']
            
            if auto_fix and not round_best['is_rule_valid']:
                print(f"  🔧 规则引擎修复中...")
                fix_result = self.rule_engine.validate_and_fix(
                    round_layout, existing_layout
                )
                if fix_result.fixed_layout:
                    round_layout = fix_result.fixed_layout
                    round_eval = self.evaluator.evaluate(
                        round_layout, existing_layout
                    )
                    print(f"    修复后得分: {round_eval.total_score:.1f}")
                    iter_info['fixed_score'] = round_eval.total_score
            
            # ========== 第5步：更新全局最优 ==========
            if round_eval.total_score > best_score:
                improvement = round_eval.total_score - best_score
                best_layout = round_layout
                best_raw_output = round_raw
                best_score = round_eval.total_score
                best_eval = round_eval
                print(f"  ⬆️ 全局最优更新: {best_score:.1f} (+{improvement:.1f})")
                iter_info['improvement'] = improvement
            else:
                print(f"  ➡️ 全局最优未变: {best_score:.1f}")
                iter_info['improvement'] = 0
            
            history.append(iter_info)
            
            # ========== 第6步：检查终止条件 ==========
            if best_score >= score_threshold:
                print(f"  ✅ 达到满意阈值 ({score_threshold}), 停止优化")
                break
            
            # 检查收敛
            if iteration > 0 and iter_info.get('improvement', 0) < improvement_threshold:
                print(f"  📉 改进幅度不足 ({iter_info.get('improvement', 0):.1f} < {improvement_threshold}), 停止优化")
                break
            
            # ========== 第7步：构造修正Prompt ==========
            if iteration < max_iterations - 1 and round_eval.issues:
                current_query = self._build_fix_query(
                    original_query=query,
                    current_layout=round_layout,
                    issues=round_eval.issues
                )
                print(f"  📋 已注入 {len(round_eval.issues)} 个问题到下一轮Prompt")
        
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
        
        # 最终规则修复
        if auto_fix:
            final_fix = self.rule_engine.validate_and_fix(best_layout, existing_layout)
            if final_fix.fixed_layout:
                best_layout = final_fix.fixed_layout
                best_eval = self.evaluator.evaluate(best_layout, existing_layout)
        
        print(f"\n{'='*50}")
        print(f"🎯 优化完成!")
        print(f"  最终得分: {best_eval.total_score:.1f}")
        print(f"  总候选数: {total_candidates}")
        print(f"  迭代轮数: {len(history)}")
        print(f"  是否满意: {best_eval.total_score >= score_threshold}")
        if best_eval.issues:
            print(f"  剩余问题: {len(best_eval.issues)} 个")
        print(f"{'='*50}")
        
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
        issues: List[str]
    ) -> str:
        """
        构造迭代修正查询：将上一轮的问题注入Prompt
        引导模型在下一次生成时避免这些问题
        
        对应优化技术方案中的 "迭代优化流程"：
        生成初始布局 → 评估打分 → 识别问题 → 针对性修正 → 循环
        """
        issues_text = "\n".join(f"  - {issue}" for issue in issues)
        layout_json = json.dumps(current_layout, ensure_ascii=False, indent=2)
        
        # 尝试使用配置文件中的fix_prompt模板
        fix_template = self.prompts_config.get('fix_prompt', '')
        if fix_template and '{issues}' in fix_template:
            fix_section = fix_template.format(
                issues=issues_text,
                original_layout=layout_json
            )
        else:
            fix_section = (
                f"\n注意：上一次生成的布局存在以下问题，请在本次生成中避免：\n"
                f"{issues_text}\n\n"
                f"上一次的布局（仅供参考，需要改进）：\n"
                f"```json\n{layout_json}\n```\n\n"
                f"请生成一个改进后的布局，解决上述问题。"
            )
        
        return f"{original_query}\n{fix_section}"
    
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


def build_query(
    house_type: str,
    floor_type: str,
    existing_params: Dict[str, List[int]],
    rooms_to_generate: List[str],
    design_constraints: str = None,
    prompts_config: Dict = None
) -> str:
    """
    构建查询文本（增强版，可注入设计约束）
    
    Args:
        house_type: 住宅类型（城市/乡村）
        floor_type: 楼层类型（一层/二层等）
        existing_params: 已有参数
        rooms_to_generate: 待生成的房间列表
        design_constraints: 设计约束文本（可选，默认从配置加载）
        prompts_config: 提示词配置字典（可选）
        
    Returns:
        查询文本
    """
    existing_json = json.dumps(existing_params, ensure_ascii=False)
    rooms_json = json.dumps(rooms_to_generate, ensure_ascii=False)
    
    # 获取设计约束
    if design_constraints is None and prompts_config:
        design_constraints = prompts_config.get('design_constraints', '')
    elif design_constraints is None:
        # 尝试从配置文件加载
        config_path = Path(__file__).parent / 'config' / 'prompts.yaml'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                design_constraints = config.get('design_constraints', '')
            except Exception:
                design_constraints = ''
    
    # 构建带约束的查询
    constraints_section = ""
    if design_constraints:
        constraints_section = f"\n{design_constraints.strip()}\n"
    
    query = f'''请根据这张图片中已有的户型信息以及对应的参数，帮我生成其余房间的参数，得到一个完整的合理平面布局。构成户型的所有空间单元均表示为矩形，用x轴坐标、y轴坐标、长度、宽度四个参数表示。本户型为"{house_type}"住宅，图片中的为"{floor_type}"平面。
{constraints_section}
图片中已有信息对应的参数为：
```json
{existing_json}
```其余待生成的"{floor_type}"房间的名称为：
```json
{rooms_json}```'''
    
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
    print("测试 LayoutPredictor（评估功能）...")
    
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
    print(f"评估得分: {result.total_score:.1f}")
    print(f"问题: {result.issues}")
    
    # 测试验证
    val_result = predictor.validate(generated, existing)
    print(f"验证通过: {val_result.valid}")
    
    # 测试查询构建
    query = build_query(
        house_type="城市",
        floor_type="一层",
        existing_params=existing,
        rooms_to_generate=["客厅", "卧室1", "厨房"]
    )
    print(f"\n构建的查询:\n{query[:200]}...")
    
    print("\n测试完成!")

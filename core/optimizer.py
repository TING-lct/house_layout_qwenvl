"""
迭代优化器模块
实现生成-评估-修正的迭代优化流程
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .generator import LayoutGenerator, LayoutResult, GenerationConfig
from .evaluator import LayoutEvaluator, EvaluationResult
from .rule_engine import LayoutRuleEngine, ValidationResult


@dataclass
class OptimizationResult:
    """优化结果"""
    final_layout: Dict[str, List[int]]
    final_score: float
    iterations: int
    history: List[Dict[str, Any]]
    is_satisfactory: bool
    evaluation: EvaluationResult
    validation: ValidationResult


class LayoutOptimizer:
    """户型布局优化器"""
    
    def __init__(
        self,
        generator: LayoutGenerator,
        evaluator: LayoutEvaluator,
        rule_engine: LayoutRuleEngine,
        score_threshold: float = 85.0,
        max_iterations: int = 3,
        improvement_threshold: float = 5.0
    ):
        """
        初始化优化器
        
        Args:
            generator: 布局生成器
            evaluator: 布局评估器
            rule_engine: 规则引擎
            score_threshold: 满意分数阈值
            max_iterations: 最大迭代次数
            improvement_threshold: 最小改进阈值
        """
        self.generator = generator
        self.evaluator = evaluator
        self.rule_engine = rule_engine
        self.score_threshold = score_threshold
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
    
    def optimize(
        self,
        image_path: str,
        initial_query: str,
        existing_layout: Dict[str, List[int]] = None,
        num_candidates: int = 5
    ) -> OptimizationResult:
        """
        执行迭代优化
        
        Args:
            image_path: 输入图片路径
            initial_query: 初始查询
            existing_layout: 已有布局（包括边界等）
            num_candidates: 每轮候选数量
            
        Returns:
            OptimizationResult: 优化结果
        """
        history = []
        best_layout = None
        best_score = 0.0
        best_evaluation = None
        best_validation = None
        
        current_query = initial_query
        
        for iteration in range(self.max_iterations):
            # 1. 生成候选
            candidates = self.generator.generate_candidates(
                image_path,
                current_query,
                num_candidates=num_candidates
            )
            
            # 2. 评估和验证每个候选
            candidate_results = []
            for i, candidate in enumerate(candidates):
                if not candidate.is_valid or not candidate.layout:
                    continue
                
                # 评估
                evaluation = self.evaluator.evaluate(
                    candidate.layout,
                    existing_layout
                )
                
                # 验证
                validation = self.rule_engine.validate(
                    candidate.layout,
                    existing_layout
                )
                
                candidate_results.append({
                    'index': i,
                    'layout': candidate.layout,
                    'score': evaluation.total_score,
                    'evaluation': evaluation,
                    'validation': validation,
                    'is_valid': validation.valid
                })
            
            if not candidate_results:
                # 如果没有有效候选，使用第一个
                if candidates and candidates[0].layout:
                    candidate_results.append({
                        'index': 0,
                        'layout': candidates[0].layout,
                        'score': 0,
                        'evaluation': self.evaluator.evaluate(candidates[0].layout, existing_layout),
                        'validation': self.rule_engine.validate(candidates[0].layout, existing_layout),
                        'is_valid': False
                    })
            
            # 3. 选择最优候选
            # 优先选择有效的高分候选
            valid_results = [r for r in candidate_results if r['is_valid']]
            if valid_results:
                best_candidate = max(valid_results, key=lambda x: x['score'])
            else:
                best_candidate = max(candidate_results, key=lambda x: x['score'])
            
            # 记录历史
            history.append({
                'iteration': iteration + 1,
                'num_candidates': len(candidates),
                'num_valid': len(valid_results),
                'best_score': best_candidate['score'],
                'issues': best_candidate['evaluation'].issues
            })
            
            # 4. 检查是否满意
            if best_candidate['score'] >= self.score_threshold and best_candidate['is_valid']:
                return OptimizationResult(
                    final_layout=best_candidate['layout'],
                    final_score=best_candidate['score'],
                    iterations=iteration + 1,
                    history=history,
                    is_satisfactory=True,
                    evaluation=best_candidate['evaluation'],
                    validation=best_candidate['validation']
                )
            
            # 5. 检查是否有改进
            if best_candidate['score'] > best_score:
                improvement = best_candidate['score'] - best_score
                best_layout = best_candidate['layout']
                best_score = best_candidate['score']
                best_evaluation = best_candidate['evaluation']
                best_validation = best_candidate['validation']
                
                # 如果改进不够显著，可能已经收敛
                if improvement < self.improvement_threshold and iteration > 0:
                    break
            
            # 6. 尝试规则引擎修复
            if not best_candidate['is_valid']:
                fixed_result = self.rule_engine.validate_and_fix(
                    best_candidate['layout'],
                    existing_layout
                )
                
                if fixed_result.valid and fixed_result.fixed_layout:
                    fixed_evaluation = self.evaluator.evaluate(
                        fixed_result.fixed_layout,
                        existing_layout
                    )
                    
                    if fixed_evaluation.total_score > best_score:
                        best_layout = fixed_result.fixed_layout
                        best_score = fixed_evaluation.total_score
                        best_evaluation = fixed_evaluation
                        best_validation = fixed_result
            
            # 7. 构造修正提示
            if iteration < self.max_iterations - 1:
                current_query = self._build_fix_prompt(
                    initial_query,
                    best_candidate['layout'],
                    best_candidate['evaluation'].issues
                )
        
        # 返回最优结果
        if best_layout is None:
            best_layout = candidates[0].layout if candidates else {}
            best_evaluation = self.evaluator.evaluate(best_layout, existing_layout)
            best_validation = self.rule_engine.validate(best_layout, existing_layout)
        
        return OptimizationResult(
            final_layout=best_layout,
            final_score=best_score,
            iterations=len(history),
            history=history,
            is_satisfactory=best_score >= self.score_threshold,
            evaluation=best_evaluation,
            validation=best_validation
        )
    
    def _build_fix_prompt(
        self,
        original_query: str,
        current_layout: Dict[str, List[int]],
        issues: List[str]
    ) -> str:
        """构造修正提示"""
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        
        fix_prompt = f"""
{original_query}

注意：上一次生成的布局存在以下问题，请避免这些问题：
{issues_text}

上一次生成的布局（仅供参考，需要改进）：
```json
{json.dumps(current_layout, ensure_ascii=False, indent=2)}
```

请生成一个改进后的布局，解决上述问题。
"""
        return fix_prompt


class MultiCandidateOptimizer:
    """多候选优化器 - 简化版"""
    
    def __init__(
        self,
        evaluator: LayoutEvaluator,
        rule_engine: LayoutRuleEngine
    ):
        """
        初始化优化器
        
        Args:
            evaluator: 布局评估器
            rule_engine: 规则引擎
        """
        self.evaluator = evaluator
        self.rule_engine = rule_engine
    
    def select_best(
        self,
        candidates: List[LayoutResult],
        existing_layout: Dict[str, List[int]] = None
    ) -> Tuple[LayoutResult, EvaluationResult]:
        """
        从候选中选择最优
        
        Args:
            candidates: 候选列表
            existing_layout: 已有布局
            
        Returns:
            Tuple: (最优候选, 评估结果)
        """
        if not candidates:
            raise ValueError("候选列表为空")
        
        best_candidate = None
        best_evaluation = None
        best_score = -1
        
        for candidate in candidates:
            if not candidate.is_valid or not candidate.layout:
                continue
            
            # 验证
            validation = self.rule_engine.validate(
                candidate.layout,
                existing_layout
            )
            
            # 如果不通过硬性规则，尝试修复
            if not validation.valid:
                fixed_result = self.rule_engine.validate_and_fix(
                    candidate.layout,
                    existing_layout
                )
                if fixed_result.valid and fixed_result.fixed_layout:
                    candidate.layout = fixed_result.fixed_layout
                    candidate.issues = fixed_result.soft_violations
            
            # 评估
            evaluation = self.evaluator.evaluate(
                candidate.layout,
                existing_layout
            )
            candidate.score = evaluation.total_score
            
            if evaluation.total_score > best_score:
                best_score = evaluation.total_score
                best_candidate = candidate
                best_evaluation = evaluation
        
        if best_candidate is None:
            # 返回第一个候选
            best_candidate = candidates[0]
            best_evaluation = self.evaluator.evaluate(
                best_candidate.layout or {},
                existing_layout
            )
        
        return best_candidate, best_evaluation

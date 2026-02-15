"""
Core模块初始化
"""

# 不依赖GPU的模块直接导入
from .evaluator import LayoutEvaluator, EvaluationResult, Room
from .rule_engine import LayoutRuleEngine, ValidationResult

# Generator相关的类单独导入（仅导入数据类，不导入需要GPU的类）
from .generator import LayoutResult, GenerationConfig, select_best_candidate

# Optimizer中不依赖GPU的类
from .optimizer import MultiCandidateOptimizer, OptimizationResult

# LLM评估器数据类（不需要GPU）
from .llm_evaluator import LLMEvaluationResult

# 延迟导入需要GPU的类
def get_layout_generator():
    """获取LayoutGenerator类（需要GPU环境）"""
    from .generator import LayoutGenerator
    return LayoutGenerator

def get_layout_optimizer():
    """获取LayoutOptimizer类（需要GPU环境）"""
    from .optimizer import LayoutOptimizer
    return LayoutOptimizer

def get_llm_evaluator():
    """获取LLMLayoutEvaluator类（需要GPU环境）"""
    from .llm_evaluator import LLMLayoutEvaluator
    return LLMLayoutEvaluator

def get_hybrid_evaluator():
    """获取HybridLayoutEvaluator类（需要GPU环境）"""
    from .llm_evaluator import HybridLayoutEvaluator
    return HybridLayoutEvaluator

def create_llm_evaluator(model_path: str, adapter_path: str = None, device: str = "cuda"):
    """创建LLM评估器（需要GPU环境）"""
    from .llm_evaluator import create_llm_evaluator as _create
    return _create(model_path, adapter_path, device)

def create_hybrid_evaluator(rule_evaluator, model_path: str = None, adapter_path: str = None, llm_weight: float = 0.4, device: str = "cuda"):
    """创建混合评估器（需要GPU环境）"""
    from .llm_evaluator import create_hybrid_evaluator as _create
    return _create(rule_evaluator, model_path, adapter_path, llm_weight, device)

def create_qwen14b_evaluator(base_model_path: str = None, adapter_path: str = None, device: str = "cuda"):
    """创建使用Qwen14B微调模型的LLM评估器（需要GPU环境）"""
    from .llm_evaluator import create_qwen14b_evaluator as _create
    return _create(base_model_path, adapter_path, device)

def create_qwen14b_hybrid_evaluator(rule_evaluator=None, base_model_path: str = None, adapter_path: str = None, llm_weight: float = 0.4, device: str = "cuda"):
    """创建使用Qwen14B微调模型的混合评估器（需要GPU环境）"""
    from .llm_evaluator import create_qwen14b_hybrid_evaluator as _create
    return _create(rule_evaluator, base_model_path, adapter_path, llm_weight, device)

__all__ = [
    # Generator (数据类，不需要GPU)
    'LayoutResult',
    'GenerationConfig',
    'select_best_candidate',
    
    # Evaluator (不需要GPU)
    'LayoutEvaluator',
    'EvaluationResult',
    'Room',
    
    # Rule Engine (不需要GPU)
    'LayoutRuleEngine',
    'ValidationResult',
    
    # Optimizer (不需要GPU的部分)
    'MultiCandidateOptimizer',
    'OptimizationResult',
    
    # LLM Evaluator (数据类)
    'LLMEvaluationResult',
    
    # 延迟导入函数
    'get_layout_generator',
    'get_layout_optimizer',
    'get_llm_evaluator',
    'get_hybrid_evaluator',
    'create_llm_evaluator',
    'create_hybrid_evaluator',
    'create_qwen14b_evaluator',
    'create_qwen14b_hybrid_evaluator',
]

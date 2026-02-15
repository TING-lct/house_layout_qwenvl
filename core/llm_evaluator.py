"""
LLM布局评估器模块
使用大语言模型进行智能布局评估
"""

import json
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# 延迟导入GPU相关模块
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
PeftModel = None


def _ensure_llm_imports():
    """确保LLM相关模块已导入"""
    global torch, AutoModelForCausalLM, AutoTokenizer, PeftModel
    
    if torch is None:
        import torch as _torch
        torch = _torch
    
    if AutoModelForCausalLM is None:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        AutoModelForCausalLM = _AutoModelForCausalLM
    
    if AutoTokenizer is None:
        from transformers import AutoTokenizer as _AutoTokenizer
        AutoTokenizer = _AutoTokenizer
    
    if PeftModel is None:
        from peft import PeftModel as _PeftModel
        PeftModel = _PeftModel


@dataclass
class LLMEvaluationResult:
    """LLM评估结果"""
    total_score: float
    dimension_scores: Dict[str, float]
    issues: List[str]
    suggestions: List[str]
    is_valid: bool
    raw_response: str
    confidence: float = 0.0


# 默认评估提示词
DEFAULT_EVALUATION_PROMPT = """你是一位专业的建筑设计师，请评估以下户型布局的合理性。

## 评估维度（每项1-10分）：

1. **空间布局合理性**：
   - 所有房间是否在边界范围内
   - 房间之间是否有重叠
   - 空间利用是否合理

2. **采光通风条件**：
   - 卧室、客厅是否靠近采光面（南采光、北采光等）
   - 是否有暗房间（完全没有采光）

3. **动线设计**：
   - 从主入口到各房间的路径是否便捷
   - 公共区域是否易于到达
   - 私密区域是否相对独立

4. **功能分区**：
   - 公共区（客厅、餐厅）和私密区（卧室）是否分离
   - 厨房和卫生间是否相邻（不宜相邻）
   - 各功能区布置是否合理

5. **尺寸规范**：
   - 客厅面积是否不小于15平方米
   - 卧室面积是否不小于9平方米
   - 厨房面积是否不小于5平方米
   - 卫生间面积是否不小于3平方米

## 户型数据：

边界信息：
```json
{boundary_info}
```

生成的房间布局：
```json
{generated_layout}
```

已有的固定元素（采光面、入口等）：
```json
{existing_elements}
```

## 请严格按以下JSON格式输出评估结果（不要有其他内容）：

```json
{{
    "scores": {{
        "空间布局合理性": <1-10的整数>,
        "采光通风条件": <1-10的整数>,
        "动线设计": <1-10的整数>,
        "功能分区": <1-10的整数>,
        "尺寸规范": <1-10的整数>
    }},
    "total_score": <五项平均分>,
    "issues": ["具体问题1", "具体问题2"],
    "suggestions": ["改进建议1", "改进建议2"],
    "is_valid": <true或false，表示布局是否基本可用>
}}
```"""


class LLMLayoutEvaluator:
    """基于LLM的户型布局评估器"""
    
    def __init__(
        self,
        model_path: str,
        adapter_path: str = None,
        device: str = "cuda",
        prompt_config_path: str = None,
        use_flash_attention: bool = False
    ):
        """
        初始化LLM评估器
        
        Args:
            model_path: 基座模型路径 (如 Qwen2.5-14B-Instruct)
            adapter_path: LoRA适配器路径（可选，用于使用微调后的模型）
            device: 运行设备
            prompt_config_path: 提示词配置文件路径
            use_flash_attention: 是否使用Flash Attention
        """
        _ensure_llm_imports()
        
        self.device = device
        self.model_path = model_path
        self.adapter_path = adapter_path
        
        # 加载模型和分词器
        self._load_model(use_flash_attention)
        
        # 加载提示词配置
        self.evaluation_prompt = DEFAULT_EVALUATION_PROMPT
        if prompt_config_path and Path(prompt_config_path).exists():
            with open(prompt_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if 'llm_evaluation_prompt' in config:
                    self.evaluation_prompt = config['llm_evaluation_prompt']
    
    def _load_model(self, use_flash_attention: bool):
        """加载模型"""
        print(f"正在加载评估模型: {self.model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        if use_flash_attention:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # 加载LoRA适配器（如果提供）
        if self.adapter_path:
            print(f"加载LoRA适配器: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path
            )
        
        self.model.eval()
        print("评估模型加载完成")
    
    def _build_prompt(
        self,
        generated_layout: Dict[str, List[int]],
        existing_layout: Dict[str, List[int]]
    ) -> str:
        """构建评估提示词"""
        # 分离边界信息
        boundary_info = {}
        existing_elements = {}
        
        for name, params in existing_layout.items():
            if name == "边界":
                boundary_info[name] = params
            else:
                existing_elements[name] = params
        
        # 格式化提示词
        prompt = self.evaluation_prompt.format(
            boundary_info=json.dumps(boundary_info, ensure_ascii=False, indent=2),
            generated_layout=json.dumps(generated_layout, ensure_ascii=False, indent=2),
            existing_elements=json.dumps(existing_elements, ensure_ascii=False, indent=2)
        )
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = response.strip()
                # 移除可能的markdown标记
                if json_str.startswith('```'):
                    json_str = json_str.split('```')[1]
                    if json_str.startswith('json'):
                        json_str = json_str[4:]
            
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response[:500]}")
            return None
    
    def evaluate(
        self,
        generated_layout: Dict[str, List[int]],
        existing_layout: Dict[str, List[int]],
        max_new_tokens: int = 1024,
        temperature: float = 0.1
    ) -> LLMEvaluationResult:
        """
        使用LLM评估布局
        
        Args:
            generated_layout: 生成的房间布局
            existing_layout: 已有的布局（包括边界、采光面等）
            max_new_tokens: 最大生成token数
            temperature: 采样温度（较低以获得更稳定的结果）
            
        Returns:
            LLMEvaluationResult: 评估结果
        """
        # 构建提示词
        prompt = self._build_prompt(generated_layout, existing_layout)
        
        # 构建消息
        messages = [
            {"role": "system", "content": "你是一位专业的建筑设计师，擅长评估户型布局的合理性。请严格按照要求的JSON格式输出。"},
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # 解码响应
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析响应
        parsed = self._parse_response(response)
        
        if parsed:
            # 提取评分
            scores = parsed.get('scores', {})
            dimension_scores = {
                "空间布局合理性": scores.get("空间布局合理性", 5),
                "采光通风条件": scores.get("采光通风条件", 5),
                "动线设计": scores.get("动线设计", 5),
                "功能分区": scores.get("功能分区", 5),
                "尺寸规范": scores.get("尺寸规范", 5)
            }
            
            total_score = parsed.get('total_score', sum(dimension_scores.values()) / len(dimension_scores))
            
            return LLMEvaluationResult(
                total_score=total_score,
                dimension_scores=dimension_scores,
                issues=parsed.get('issues', []),
                suggestions=parsed.get('suggestions', []),
                is_valid=parsed.get('is_valid', True),
                raw_response=response,
                confidence=0.9
            )
        else:
            # 解析失败，返回默认结果
            return LLMEvaluationResult(
                total_score=5.0,
                dimension_scores={
                    "空间布局合理性": 5,
                    "采光通风条件": 5,
                    "动线设计": 5,
                    "功能分区": 5,
                    "尺寸规范": 5
                },
                issues=["LLM响应解析失败"],
                suggestions=["请检查布局数据格式"],
                is_valid=False,
                raw_response=response,
                confidence=0.0
            )
    
    def batch_evaluate(
        self,
        layouts: List[Tuple[Dict[str, List[int]], Dict[str, List[int]]]],
        **kwargs
    ) -> List[LLMEvaluationResult]:
        """
        批量评估多个布局
        
        Args:
            layouts: 布局列表，每个元素为 (generated_layout, existing_layout)
            **kwargs: 传递给evaluate的参数
            
        Returns:
            评估结果列表
        """
        results = []
        for generated, existing in layouts:
            result = self.evaluate(generated, existing, **kwargs)
            results.append(result)
        return results


class HybridLayoutEvaluator:
    """混合评估器：结合规则评估和LLM评估"""
    
    def __init__(
        self,
        rule_evaluator,  # LayoutEvaluator实例
        llm_evaluator: LLMLayoutEvaluator = None,
        llm_weight: float = 0.4
    ):
        """
        初始化混合评估器
        
        Args:
            rule_evaluator: 规则评估器
            llm_evaluator: LLM评估器（可选）
            llm_weight: LLM评估权重（0-1），规则评估权重为1-llm_weight
        """
        self.rule_evaluator = rule_evaluator
        self.llm_evaluator = llm_evaluator
        self.llm_weight = llm_weight if llm_evaluator else 0.0
        self.rule_weight = 1.0 - self.llm_weight
    
    def evaluate(
        self,
        generated_layout: Dict[str, List[int]],
        existing_layout: Dict[str, List[int]]
    ) -> Dict[str, Any]:
        """
        综合评估布局
        
        Returns:
            包含规则评估和LLM评估结果的字典
        """
        # 规则评估
        rule_result = self.rule_evaluator.evaluate(generated_layout, existing_layout)
        
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
            "combined_issues": rule_result.issues.copy(),
            "combined_suggestions": rule_result.suggestions.copy(),
            "is_valid": rule_result.is_valid
        }
        
        # LLM评估（如果可用）
        if self.llm_evaluator:
            try:
                llm_result = self.llm_evaluator.evaluate(generated_layout, existing_layout)
                
                result["llm_evaluation"] = {
                    "total_score": llm_result.total_score,
                    "dimension_scores": llm_result.dimension_scores,
                    "issues": llm_result.issues,
                    "suggestions": llm_result.suggestions,
                    "is_valid": llm_result.is_valid,
                    "confidence": llm_result.confidence
                }
                
                # 计算综合得分（LLM是10分制，规则是100分制）
                llm_score_100 = llm_result.total_score * 10
                result["combined_score"] = (
                    self.rule_weight * rule_result.total_score +
                    self.llm_weight * llm_score_100
                )
                
                # 合并问题和建议（去重）
                for issue in llm_result.issues:
                    if issue not in result["combined_issues"]:
                        result["combined_issues"].append(issue)
                
                for suggestion in llm_result.suggestions:
                    if suggestion not in result["combined_suggestions"]:
                        result["combined_suggestions"].append(suggestion)
                
                # 综合有效性判断
                result["is_valid"] = rule_result.is_valid and llm_result.is_valid
                
            except Exception as e:
                print(f"LLM评估失败: {e}")
                result["llm_evaluation"] = {"error": str(e)}
        
        return result


def create_llm_evaluator(
    model_path: str,
    adapter_path: str = None,
    device: str = "cuda"
) -> LLMLayoutEvaluator:
    """
    创建LLM评估器的便捷函数
    
    Args:
        model_path: 基座模型路径
        adapter_path: LoRA适配器路径
        device: 运行设备
        
    Returns:
        LLMLayoutEvaluator实例
    """
    return LLMLayoutEvaluator(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device
    )


def create_hybrid_evaluator(
    rule_evaluator,
    model_path: str = None,
    adapter_path: str = None,
    llm_weight: float = 0.4,
    device: str = "cuda"
) -> HybridLayoutEvaluator:
    """
    创建混合评估器的便捷函数
    
    Args:
        rule_evaluator: 规则评估器
        model_path: LLM模型路径（如果为None则只用规则评估）
        adapter_path: LoRA适配器路径
        llm_weight: LLM评估权重
        device: 运行设备
        
    Returns:
        HybridLayoutEvaluator实例
    """
    llm_evaluator = None
    if model_path:
        llm_evaluator = create_llm_evaluator(model_path, adapter_path, device)
    
    return HybridLayoutEvaluator(
        rule_evaluator=rule_evaluator,
        llm_evaluator=llm_evaluator,
        llm_weight=llm_weight
    )


# ==================== Qwen14B 微调模型专用配置 ====================

# 默认的 Qwen14B 微调模型路径配置
QWEN14B_DEFAULT_CONFIG = {
    # Linux 服务器路径
    "linux": {
        "base_model": "/home/nju/.cache/modelscope/hub/models/Qwen/Qwen2___5-14B-Instruct",
        "adapter": "/saves/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23"
    },
    # Windows 本地路径（基座模型使用HF ID自动下载，或指定本地路径）
    "windows": {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",  # HuggingFace 模型ID，会自动下载
        "adapter": "F:/task/户型图生成/qwen14b/Qwen2.5-14B-Instruct/Qwen2.5-14B-Instruct/lora/train_2025-12-01-21-17-23"
    }
}


def create_qwen14b_evaluator(
    base_model_path: str = None,
    adapter_path: str = None,
    device: str = "cuda",
    use_flash_attention: bool = False
) -> LLMLayoutEvaluator:
    """
    创建使用微调后的 Qwen2.5-14B 作为评估器
    
    这个模型是专门为户型评估任务微调的，比基座模型评估更准确。
    
    Args:
        base_model_path: 基座模型路径，默认使用服务器路径
        adapter_path: LoRA适配器路径，默认使用服务器路径
        device: 运行设备
        use_flash_attention: 是否使用Flash Attention
        
    Returns:
        LLMLayoutEvaluator实例
        
    使用示例:
        # 方式1：使用默认路径（服务器）
        evaluator = create_qwen14b_evaluator()
        
        # 方式2：指定自定义路径
        evaluator = create_qwen14b_evaluator(
            base_model_path="/path/to/Qwen2.5-14B-Instruct",
            adapter_path="/path/to/lora/train_xxx"
        )
    """
    import platform
    
    # 确定默认路径
    if base_model_path is None:
        base_model_path = QWEN14B_DEFAULT_CONFIG["linux"]["base_model"]
    
    if adapter_path is None:
        # 根据操作系统选择默认适配器路径
        if platform.system() == "Windows":
            adapter_path = QWEN14B_DEFAULT_CONFIG["windows"]["adapter"]
        else:
            adapter_path = QWEN14B_DEFAULT_CONFIG["linux"]["adapter"]
    
    print(f"创建 Qwen14B 微调评估器:")
    print(f"  基座模型: {base_model_path}")
    print(f"  LoRA适配器: {adapter_path}")
    
    return LLMLayoutEvaluator(
        model_path=base_model_path,
        adapter_path=adapter_path,
        device=device,
        use_flash_attention=use_flash_attention
    )


def create_qwen14b_hybrid_evaluator(
    rule_evaluator=None,
    base_model_path: str = None,
    adapter_path: str = None,
    llm_weight: float = 0.4,
    device: str = "cuda"
) -> HybridLayoutEvaluator:
    """
    创建使用 Qwen14B 微调模型的混合评估器
    
    结合规则评估（60%）和 LLM 评估（40%）
    
    Args:
        rule_evaluator: 规则评估器，如果为None则自动创建
        base_model_path: 基座模型路径
        adapter_path: LoRA适配器路径
        llm_weight: LLM评估权重
        device: 运行设备
        
    Returns:
        HybridLayoutEvaluator实例
    """
    # 如果没有提供规则评估器，创建一个
    if rule_evaluator is None:
        from .evaluator import LayoutEvaluator
        rule_evaluator = LayoutEvaluator()
    
    # 创建 Qwen14B LLM 评估器
    llm_evaluator = create_qwen14b_evaluator(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        device=device
    )
    
    return HybridLayoutEvaluator(
        rule_evaluator=rule_evaluator,
        llm_evaluator=llm_evaluator,
        llm_weight=llm_weight
    )

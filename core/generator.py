"""
户型布局生成器模块
支持多候选生成和多样性采样
"""

import json
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 延迟导入GPU相关模块，避免在无GPU环境下导入失败
torch = None
Qwen2_5_VLForConditionalGeneration = None
AutoProcessor = None
PeftModel = None
process_vision_info = None

def _ensure_gpu_imports():
    """确保GPU相关模块已导入"""
    global torch, Qwen2_5_VLForConditionalGeneration, AutoProcessor, PeftModel, process_vision_info
    
    if torch is None:
        import torch as _torch
        torch = _torch
    
    if Qwen2_5_VLForConditionalGeneration is None:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen
            Qwen2_5_VLForConditionalGeneration = _Qwen
        except ImportError:
            # 尝试其他可能的类名
            try:
                from transformers import Qwen2VLForConditionalGeneration as _Qwen
                Qwen2_5_VLForConditionalGeneration = _Qwen
            except ImportError:
                from transformers import AutoModelForCausalLM as _Qwen
                Qwen2_5_VLForConditionalGeneration = _Qwen
    
    if AutoProcessor is None:
        from transformers import AutoProcessor as _AutoProcessor
        AutoProcessor = _AutoProcessor
    
    if PeftModel is None:
        from peft import PeftModel as _PeftModel
        PeftModel = _PeftModel
    
    if process_vision_info is None:
        try:
            from qwen_vl_utils import process_vision_info as _process_vision_info
            process_vision_info = _process_vision_info
        except ImportError:
            # 如果没有qwen_vl_utils，使用简单的占位函数
            def _process_vision_info(messages):
                return None, None
            process_vision_info = _process_vision_info


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class LayoutResult:
    """布局生成结果"""
    layout: Dict[str, List[int]]
    raw_output: str
    score: float = 0.0
    is_valid: bool = True
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class LayoutGenerator:
    """户型布局生成器"""
    
    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: str = None,
        device: str = "cuda",
        use_flash_attention: bool = False
    ):
        """
        初始化生成器
        
        Args:
            base_model_path: 基础模型路径
            lora_adapter_path: LoRA适配器路径
            device: 运行设备
            use_flash_attention: 是否使用Flash Attention
        """
        # 确保GPU相关模块已导入
        _ensure_gpu_imports()
        
        self.device = device
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        
        # 加载模型
        self._load_model(use_flash_attention)
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            base_model_path, 
            use_fast=True
        )
    
    def _load_model(self, use_flash_attention: bool):
        """加载模型"""
        if use_flash_attention:
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
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.lora_adapter_path
            )
            self.model = self.model.half()
    
    def _prepare_inputs(
        self,
        image_path: str,
        query: str
    ) -> Dict[str, Any]:
        """准备模型输入"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return inputs.to(self.device)
    
    def _parse_output(self, output_text: str) -> Dict[str, List[int]]:
        """解析模型输出为布局字典（增强容错）"""
        import re
        
        try:
            # 提取JSON部分
            if "```json" in output_text:
                json_str = output_text.split("```json")[1].split("```")[0].strip()
            elif "```" in output_text:
                json_str = output_text.split("```")[1].split("```")[0].strip()
            else:
                # 提取第一个 { ... } 块
                start = output_text.find("{")
                if start != -1:
                    depth = 0
                    for i in range(start, len(output_text)):
                        if output_text[i] == "{":
                            depth += 1
                        elif output_text[i] == "}":
                            depth -= 1
                            if depth == 0:
                                json_str = output_text[start:i + 1]
                                break
                    else:
                        json_str = output_text[start:] + "}"
                else:
                    json_str = output_text.strip()
            
            layout = json.loads(json_str)
            # 验证格式
            validated = {}
            for k, v in layout.items():
                if isinstance(v, list) and len(v) == 4:
                    try:
                        validated[k] = [int(x) for x in v]
                    except (ValueError, TypeError):
                        continue
            return validated
        except (json.JSONDecodeError, IndexError):
            pass
        
        # 正则兜底：提取 "房间名": [x, y, w, h]
        try:
            layout = {}
            pattern = r'"([^"]+)"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            for m in re.finditer(pattern, output_text):
                name = m.group(1)
                vals = [int(m.group(i)) for i in range(2, 6)]
                layout[name] = vals
            if layout:
                return layout
        except Exception:
            pass
        
        print(f"解析输出失败: {output_text[:200]}")
        return {}
    
    def generate(
        self,
        image_path: str,
        query: str,
        config: GenerationConfig = None
    ) -> LayoutResult:
        """
        生成单个布局
        
        Args:
            image_path: 输入图片路径
            query: 查询文本
            config: 生成配置
            
        Returns:
            LayoutResult: 生成结果
        """
        if config is None:
            config = GenerationConfig()
        
        inputs = self._prepare_inputs(image_path, query)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty
            )
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 解析布局
        layout = self._parse_output(output_text)
        
        return LayoutResult(
            layout=layout,
            raw_output=output_text,
            is_valid=len(layout) > 0
        )
    
    def generate_candidates(
        self,
        image_path: str,
        query: str,
        num_candidates: int = 5,
        temperature_range: List[float] = None
    ) -> List[LayoutResult]:
        """
        多候选生成
        
        Args:
            image_path: 输入图片路径
            query: 查询文本
            num_candidates: 候选数量
            temperature_range: 温度范围列表
            
        Returns:
            List[LayoutResult]: 候选结果列表
        """
        if temperature_range is None:
            # 默认温度范围：0.3到0.95，避免过高温度产生低质量输出
            temperatures_pool = [0.3, 0.5, 0.7, 0.85, 0.95]
            temperature_range = temperatures_pool[:num_candidates]
            if num_candidates > len(temperatures_pool):
                step = 0.65 / num_candidates
                temperature_range = [0.3 + i * step for i in range(num_candidates)]
        
        candidates = []
        
        for i in range(num_candidates):
            temp = temperature_range[i] if i < len(temperature_range) else 0.7
            
            config = GenerationConfig(
                temperature=temp,
                do_sample=True
            )
            
            result = self.generate(image_path, query, config)
            candidates.append(result)
        
        return candidates
    
    def generate_with_prompt_variants(
        self,
        image_path: str,
        base_query: str,
        prompt_variants: List[str]
    ) -> List[LayoutResult]:
        """
        使用不同提示词变体生成
        
        Args:
            image_path: 输入图片路径
            base_query: 基础查询
            prompt_variants: 提示词变体列表
            
        Returns:
            List[LayoutResult]: 结果列表
        """
        results = []
        
        for variant in prompt_variants:
            # 组合基础查询和提示词变体
            enhanced_query = f"{variant}\n\n{base_query}"
            result = self.generate(image_path, enhanced_query)
            results.append(result)
        
        return results


def select_best_candidate(
    candidates: List[LayoutResult],
    evaluator=None
) -> Tuple[LayoutResult, int]:
    """
    选择最优候选
    
    Args:
        candidates: 候选结果列表
        evaluator: 评估器（可选）
        
    Returns:
        Tuple[LayoutResult, int]: 最优结果和索引
    """
    if not candidates:
        raise ValueError("候选列表为空")
    
    # 如果有评估器，使用评估分数
    if evaluator:
        for candidate in candidates:
            if candidate.is_valid:
                candidate.score = evaluator.score(candidate.layout)
    
    # 过滤有效候选
    valid_candidates = [c for c in candidates if c.is_valid]
    
    if not valid_candidates:
        # 如果没有有效候选，返回第一个
        return candidates[0], 0
    
    # 按分数排序
    best_idx = max(range(len(valid_candidates)), key=lambda i: valid_candidates[i].score)
    
    # 找到原始索引
    original_idx = candidates.index(valid_candidates[best_idx])
    
    return valid_candidates[best_idx], original_idx

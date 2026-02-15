"""
户型布局生成API服务
提供RESTful API接口
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# 导入核心模块
from pipeline import LayoutGenerationPipeline, PipelineConfig, create_pipeline
from core import LayoutEvaluator, LayoutRuleEngine
from utils import LayoutVisualizer


# ==================== 数据模型 ====================

class GenerationRequest(BaseModel):
    """生成请求"""
    existing_params: Dict[str, List[int]] = Field(..., description="已有房间参数")
    rooms_to_generate: List[str] = Field(..., description="待生成房间列表")
    house_type: str = Field(default="城市", description="户型类型")
    floor_type: str = Field(default="一层", description="楼层类型")
    optimize: bool = Field(default=True, description="是否启用优化")
    num_candidates: int = Field(default=5, description="候选数量")


class EvaluationRequest(BaseModel):
    """评估请求"""
    layout: Dict[str, List[int]] = Field(..., description="待评估布局")
    full_layout: Optional[Dict[str, List[int]]] = Field(None, description="完整布局")


class ValidationRequest(BaseModel):
    """验证请求"""
    layout: Dict[str, List[int]] = Field(..., description="待验证布局")
    full_layout: Optional[Dict[str, List[int]]] = Field(None, description="完整布局")
    auto_fix: bool = Field(default=False, description="是否自动修复")


class GenerationResponse(BaseModel):
    """生成响应"""
    success: bool
    layout: Dict[str, List[int]]
    score: float
    is_satisfactory: bool
    candidates_count: int
    iterations: int
    issues: List[str]
    suggestions: List[str]


class EvaluationResponse(BaseModel):
    """评估响应"""
    success: bool
    total_score: float
    dimension_scores: Dict[str, float]
    issues: List[str]
    suggestions: List[str]
    is_valid: bool


class LLMEvaluationRequest(BaseModel):
    """LLM评估请求"""
    layout: Dict[str, List[int]] = Field(..., description="待评估布局")
    full_layout: Optional[Dict[str, List[int]]] = Field(None, description="完整布局")
    model_path: Optional[str] = Field(None, description="评估模型路径（可选，使用默认配置）")


class LLMEvaluationResponse(BaseModel):
    """LLM评估响应"""
    success: bool
    total_score: float
    dimension_scores: Dict[str, float]
    issues: List[str]
    suggestions: List[str]
    is_valid: bool
    confidence: float
    combined_score: Optional[float] = None


class ValidationResponse(BaseModel):
    """验证响应"""
    success: bool
    valid: bool
    hard_violations: List[str]
    soft_violations: List[str]
    fixed_layout: Optional[Dict[str, List[int]]]


# ==================== API应用 ====================

app = FastAPI(
    title="户型布局生成API",
    description="基于多模态大模型的户型布局生成服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局组件
pipeline: Optional[LayoutGenerationPipeline] = None
evaluator: Optional[LayoutEvaluator] = None
rule_engine: Optional[LayoutRuleEngine] = None
visualizer: Optional[LayoutVisualizer] = None
llm_evaluator = None  # LLM评估器


@app.on_event("startup")
async def startup_event():
    """启动时初始化组件"""
    global evaluator, rule_engine, visualizer
    
    # 初始化评估器和规则引擎（不需要GPU）
    rules_config_path = "config/rules.yaml"
    if Path(rules_config_path).exists():
        evaluator = LayoutEvaluator(rules_config_path)
        rule_engine = LayoutRuleEngine(rules_config_path)
    else:
        evaluator = LayoutEvaluator()
        rule_engine = LayoutRuleEngine()
    
    visualizer = LayoutVisualizer()
    
    print("API服务已启动，评估和验证功能可用")
    print("生成功能需要调用 /init_pipeline 接口初始化")


@app.post("/init_pipeline")
async def init_pipeline(
    base_model_path: str = Form("models/Qwen2.5-VL-7B-Instruct"),
    lora_adapter_path: str = Form("lora_model"),
    device: str = Form("cuda")
):
    """
    初始化生成流程
    
    需要GPU支持，用于加载模型
    """
    global pipeline
    
    try:
        config = PipelineConfig(
            base_model_path=base_model_path,
            lora_adapter_path=lora_adapter_path,
            device=device
        )
        pipeline = LayoutGenerationPipeline(config)
        
        return {"success": True, "message": "生成流程初始化成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")


@app.post("/generate", response_model=GenerationResponse)
async def generate_layout(
    image: UploadFile = File(...),
    request_data: str = Form(...)
):
    """
    生成户型布局
    
    - **image**: 户型图片
    - **request_data**: JSON格式的生成请求参数
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=400, detail="请先调用 /init_pipeline 初始化生成流程")
    
    try:
        # 解析请求
        request = GenerationRequest(**json.loads(request_data))
        
        # 保存临时图片
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # 更新配置
            pipeline.config.num_candidates = request.num_candidates
            
            # 执行生成
            result = pipeline.generate(
                image_path=tmp_path,
                existing_params=request.existing_params,
                rooms_to_generate=request.rooms_to_generate,
                house_type=request.house_type,
                floor_type=request.floor_type,
                optimize=request.optimize
            )
            
            return GenerationResponse(
                success=True,
                layout=result.layout,
                score=result.score,
                is_satisfactory=result.is_satisfactory,
                candidates_count=result.candidates_count,
                iterations=result.iterations,
                issues=result.evaluation.issues if result.evaluation else [],
                suggestions=result.evaluation.suggestions if result.evaluation else []
            )
        finally:
            # 清理临时文件
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_layout(request: EvaluationRequest):
    """
    评估户型布局（规则评估）
    
    不需要图片，直接评估布局参数
    """
    global evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=500, detail="评估器未初始化")
    
    try:
        result = evaluator.evaluate(request.layout, request.full_layout)
        
        return EvaluationResponse(
            success=True,
            total_score=result.total_score,
            dimension_scores=result.dimension_scores,
            issues=result.issues,
            suggestions=result.suggestions,
            is_valid=result.is_valid
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@app.post("/init_llm_evaluator")
async def init_llm_evaluator(
    model_path: str = Form(..., description="基座模型路径"),
    adapter_path: str = Form(None, description="LoRA适配器路径（可选）"),
    device: str = Form("cuda", description="运行设备")
):
    """
    初始化LLM评估器
    
    需要GPU支持，用于加载评估模型
    """
    global llm_evaluator
    
    try:
        from core.llm_evaluator import LLMLayoutEvaluator
        
        llm_evaluator = LLMLayoutEvaluator(
            model_path=model_path,
            adapter_path=adapter_path if adapter_path else None,
            device=device
        )
        
        return {"success": True, "message": "LLM评估器初始化成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM评估器初始化失败: {str(e)}")


@app.post("/evaluate_llm", response_model=LLMEvaluationResponse)
async def evaluate_layout_llm(request: LLMEvaluationRequest):
    """
    使用LLM评估户型布局
    
    使用大语言模型进行智能评估，需要先调用 /init_llm_evaluator 初始化
    """
    global llm_evaluator, evaluator
    
    # 如果提供了model_path，动态初始化
    if request.model_path and llm_evaluator is None:
        try:
            from core.llm_evaluator import LLMLayoutEvaluator
            llm_evaluator = LLMLayoutEvaluator(
                model_path=request.model_path,
                device="cuda"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM评估器初始化失败: {str(e)}")
    
    if llm_evaluator is None:
        raise HTTPException(
            status_code=400, 
            detail="LLM评估器未初始化，请先调用 /init_llm_evaluator 或在请求中提供 model_path"
        )
    
    try:
        # LLM评估
        result = llm_evaluator.evaluate(request.layout, request.full_layout)
        
        # 计算综合分数（如果规则评估器可用）
        combined_score = None
        if evaluator:
            rule_result = evaluator.evaluate(request.layout, request.full_layout)
            # LLM是10分制，规则是100分制，综合计算
            combined_score = 0.6 * rule_result.total_score + 0.4 * (result.total_score * 10)
        
        return LLMEvaluationResponse(
            success=True,
            total_score=result.total_score,
            dimension_scores=result.dimension_scores,
            issues=result.issues,
            suggestions=result.suggestions,
            is_valid=result.is_valid,
            confidence=result.confidence,
            combined_score=combined_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM评估失败: {str(e)}")


@app.post("/evaluate_hybrid")
async def evaluate_layout_hybrid(request: EvaluationRequest):
    """
    使用混合评估（规则+LLM）
    
    结合规则评估和LLM评估，返回综合结果
    """
    global llm_evaluator, evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=500, detail="规则评估器未初始化")
    
    try:
        # 规则评估
        rule_result = evaluator.evaluate(request.layout, request.full_layout)
        
        result = {
            "success": True,
            "rule_evaluation": {
                "total_score": rule_result.total_score,
                "dimension_scores": rule_result.dimension_scores,
                "issues": rule_result.issues,
                "suggestions": rule_result.suggestions,
                "is_valid": rule_result.is_valid
            },
            "llm_evaluation": None,
            "combined_score": rule_result.total_score,
            "combined_issues": rule_result.issues,
            "combined_suggestions": rule_result.suggestions
        }
        
        # LLM评估（如果可用）
        if llm_evaluator:
            try:
                llm_result = llm_evaluator.evaluate(request.layout, request.full_layout)
                
                result["llm_evaluation"] = {
                    "total_score": llm_result.total_score,
                    "dimension_scores": llm_result.dimension_scores,
                    "issues": llm_result.issues,
                    "suggestions": llm_result.suggestions,
                    "is_valid": llm_result.is_valid,
                    "confidence": llm_result.confidence
                }
                
                # 综合分数（规则60% + LLM40%）
                result["combined_score"] = 0.6 * rule_result.total_score + 0.4 * (llm_result.total_score * 10)
                
                # 合并问题和建议
                for issue in llm_result.issues:
                    if issue not in result["combined_issues"]:
                        result["combined_issues"].append(issue)
                for suggestion in llm_result.suggestions:
                    if suggestion not in result["combined_suggestions"]:
                        result["combined_suggestions"].append(suggestion)
                        
            except Exception as e:
                result["llm_evaluation"] = {"error": str(e)}
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@app.post("/validate", response_model=ValidationResponse)
async def validate_layout(request: ValidationRequest):
    """
    验证户型布局
    
    检查布局是否符合规则约束
    """
    global rule_engine
    
    if rule_engine is None:
        raise HTTPException(status_code=500, detail="规则引擎未初始化")
    
    try:
        if request.auto_fix:
            result = rule_engine.validate_and_fix(request.layout, request.full_layout)
        else:
            result = rule_engine.validate(request.layout, request.full_layout)
        
        return ValidationResponse(
            success=True,
            valid=result.valid,
            hard_violations=result.hard_violations,
            soft_violations=result.soft_violations,
            fixed_layout=result.fixed_layout if request.auto_fix else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")


@app.post("/visualize")
async def visualize_layout(
    layout: Dict[str, List[int]],
    title: str = "户型布局"
):
    """
    可视化户型布局
    
    返回Base64编码的图片
    """
    global visualizer
    
    if visualizer is None:
        raise HTTPException(status_code=500, detail="可视化器未初始化")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = visualizer.visualize(layout, title=title)
        
        # 转换为Base64
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        plt.close(fig)
        
        return {
            "success": True,
            "image": img_base64,
            "format": "png",
            "encoding": "base64"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"可视化失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
        "evaluator_ready": evaluator is not None,
        "rule_engine_ready": rule_engine is not None
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "户型布局生成API",
        "version": "1.0.0",
        "endpoints": [
            "/init_pipeline - 初始化生成流程（需要GPU）",
            "/generate - 生成户型布局",
            "/evaluate - 评估户型布局",
            "/validate - 验证户型布局",
            "/visualize - 可视化户型布局",
            "/health - 健康检查"
        ]
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """启动API服务"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

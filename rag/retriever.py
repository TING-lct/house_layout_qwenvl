"""
检索器模块
实现基于RAG的检索增强生成
"""

import json
from typing import List, Dict, Any, Optional, Tuple

from .knowledge_base import LayoutKnowledgeBase, LayoutCase
from .embedder import LayoutEmbedder


class LayoutRetriever:
    """户型布局检索器"""
    
    def __init__(
        self,
        knowledge_base: LayoutKnowledgeBase,
        top_k: int = 3
    ):
        """
        初始化检索器
        
        Args:
            knowledge_base: 知识库
            top_k: 默认检索数量
        """
        self.kb = knowledge_base
        self.top_k = top_k
    
    def retrieve(
        self,
        query_params: Dict[str, Any],
        top_k: int = None
    ) -> List[Tuple[LayoutCase, float]]:
        """
        检索相似案例
        
        Args:
            query_params: 查询参数（包括已有房间、待生成房间等）
            top_k: 检索数量
            
        Returns:
            List[Tuple[LayoutCase, float]]: 案例和相似度列表
        """
        if top_k is None:
            top_k = self.top_k
        
        # 构建查询布局
        query_layout = query_params.get('existing_layout', {})
        
        # 构建元数据
        query_metadata = {
            'house_type': query_params.get('house_type', '城市'),
            'floor_type': query_params.get('floor_type', '一层'),
            'rooms_to_generate': query_params.get('rooms_to_generate', [])
        }
        
        # 检索相似案例
        results = self.kb.search_similar(
            query_layout=query_layout,
            query_metadata=query_metadata,
            top_k=top_k
        )
        
        return results
    
    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int = None
    ) -> List[Tuple[LayoutCase, float]]:
        """
        基于文本检索案例
        
        Args:
            query_text: 查询文本
            top_k: 检索数量
            
        Returns:
            List[Tuple[LayoutCase, float]]: 案例和相似度列表
        """
        if top_k is None:
            top_k = self.top_k
        
        return self.kb.search_similar(
            query_text=query_text,
            top_k=top_k
        )
    
    def format_reference_cases(
        self,
        cases: List[Tuple[LayoutCase, float]],
        include_score: bool = False
    ) -> str:
        """
        格式化参考案例为文本
        
        Args:
            cases: 案例列表
            include_score: 是否包含相似度分数
            
        Returns:
            str: 格式化的文本
        """
        if not cases:
            return "无相关参考案例"
        
        lines = []
        for i, (case, sim) in enumerate(cases, 1):
            # 提取房间信息
            rooms_info = []
            for name, params in case.layout.items():
                if name in ['边界'] or '采光' in name or '黑体' in name or '入口' in name:
                    continue
                if len(params) == 4:
                    area = params[2] * params[3] / 1000000
                    rooms_info.append(f"{name}({area:.1f}平米)")
            
            # 构建案例描述
            desc = f"案例{i}："
            if case.metadata.get('house_type'):
                desc += f"{case.metadata['house_type']}住宅，"
            
            desc += f"包含{', '.join(rooms_info)}"
            
            if include_score:
                desc += f" [相似度: {sim:.2f}]"
            
            # 添加布局参数
            layout_json = json.dumps(
                {k: v for k, v in case.layout.items() 
                 if k not in ['边界'] and '采光' not in k and '黑体' not in k and '入口' not in k},
                ensure_ascii=False
            )
            desc += f"\n布局参数: {layout_json}"
            
            lines.append(desc)
        
        return "\n\n".join(lines)


class RAGGenerator:
    """基于RAG的增强生成器"""
    
    def __init__(
        self,
        retriever: LayoutRetriever,
        prompt_template: str = None
    ):
        """
        初始化RAG生成器
        
        Args:
            retriever: 检索器
            prompt_template: 提示词模板
        """
        self.retriever = retriever
        self.prompt_template = prompt_template or self._default_template()
    
    def _default_template(self) -> str:
        """默认提示词模板"""
        return """请参考以下优质案例，生成新的户型布局：

{reference_cases}

当前需求：
- 户型类型：{house_type}
- 楼层：{floor_type}
- 已有房间：{existing_rooms}
- 待生成房间：{rooms_to_generate}

设计约束：
1. 厨房与卫生间不宜直接相邻
2. 卧室应尽量靠近采光面
3. 客厅应有良好的采光和通风
4. 房间尺寸应符合人体工程学标准

请根据参考案例和约束条件，生成合理的房间参数。"""
    
    def build_enhanced_prompt(
        self,
        existing_layout: Dict[str, List[int]],
        rooms_to_generate: List[str],
        house_type: str = "城市",
        floor_type: str = "一层",
        top_k: int = 3
    ) -> str:
        """
        构建增强提示词
        
        Args:
            existing_layout: 已有布局
            rooms_to_generate: 待生成房间
            house_type: 户型类型
            floor_type: 楼层类型
            top_k: 参考案例数量
            
        Returns:
            str: 增强后的提示词
        """
        # 检索相似案例
        query_params = {
            'existing_layout': existing_layout,
            'house_type': house_type,
            'floor_type': floor_type,
            'rooms_to_generate': rooms_to_generate
        }
        
        cases = self.retriever.retrieve(query_params, top_k=top_k)
        
        # 格式化参考案例
        reference_text = self.retriever.format_reference_cases(cases)
        
        # 格式化已有房间
        existing_rooms_text = json.dumps(
            {k: v for k, v in existing_layout.items() 
             if k not in ['边界'] and '采光' not in k and '黑体' not in k and '入口' not in k},
            ensure_ascii=False
        )
        
        # 构建增强提示词
        enhanced_prompt = self.prompt_template.format(
            reference_cases=reference_text,
            house_type=house_type,
            floor_type=floor_type,
            existing_rooms=existing_rooms_text,
            rooms_to_generate=json.dumps(rooms_to_generate, ensure_ascii=False)
        )
        
        return enhanced_prompt

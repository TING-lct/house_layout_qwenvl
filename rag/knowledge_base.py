"""
知识库模块
存储和管理优质户型案例
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .embedder import LayoutEmbedder


@dataclass
class LayoutCase:
    """户型案例"""
    layout: Dict[str, List[int]]
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    score: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'layout': self.layout,
            'metadata': self.metadata,
            'score': self.score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LayoutCase':
        """从字典创建"""
        return cls(
            layout=data['layout'],
            metadata=data.get('metadata', {}),
            score=data.get('score', 0.0)
        )


class LayoutKnowledgeBase:
    """户型布局知识库"""
    
    def __init__(
        self, 
        embedder: LayoutEmbedder = None,
        similarity_threshold: float = 0.5
    ):
        """
        初始化知识库
        
        Args:
            embedder: 向量化器
            similarity_threshold: 相似度阈值
        """
        self.embedder = embedder or LayoutEmbedder()
        self.similarity_threshold = similarity_threshold
        
        self.cases: List[LayoutCase] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_case(
        self, 
        layout: Dict[str, List[int]], 
        metadata: Dict[str, Any] = None,
        score: float = 0.0
    ):
        """
        添加案例到知识库
        
        Args:
            layout: 布局字典
            metadata: 元数据
            score: 质量分数
        """
        if metadata is None:
            metadata = {}
        
        # 计算向量
        embedding = self.embedder.encode(layout, metadata)
        
        case = LayoutCase(
            layout=layout,
            metadata=metadata,
            embedding=embedding,
            score=score
        )
        
        self.cases.append(case)
        
        # 更新向量矩阵
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
    
    def add_cases_batch(
        self,
        layouts: List[Dict[str, List[int]]],
        metadata_list: List[Dict] = None,
        scores: List[float] = None
    ):
        """
        批量添加案例
        
        Args:
            layouts: 布局列表
            metadata_list: 元数据列表
            scores: 分数列表
        """
        if metadata_list is None:
            metadata_list = [{}] * len(layouts)
        if scores is None:
            scores = [0.0] * len(layouts)
        
        # 批量计算向量
        embeddings = self.embedder.encode_batch(layouts, metadata_list)
        
        for layout, metadata, embedding, score in zip(layouts, metadata_list, embeddings, scores):
            case = LayoutCase(
                layout=layout,
                metadata=metadata,
                embedding=embedding,
                score=score
            )
            self.cases.append(case)
        
        # 更新向量矩阵
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search_similar(
        self, 
        query_layout: Dict[str, List[int]] = None,
        query_text: str = None,
        query_metadata: Dict = None,
        top_k: int = 3,
        min_score: float = None
    ) -> List[Tuple[LayoutCase, float]]:
        """
        检索相似案例
        
        Args:
            query_layout: 查询布局
            query_text: 查询文本（与query_layout二选一）
            query_metadata: 查询元数据
            top_k: 返回数量
            min_score: 最小分数阈值
            
        Returns:
            List[Tuple[LayoutCase, float]]: 案例和相似度列表
        """
        if not self.cases:
            return []
        
        # 计算查询向量
        if query_text:
            query_embedding = self.embedder.encode_text(query_text)
        elif query_layout:
            query_embedding = self.embedder.encode(query_layout, query_metadata)
        else:
            raise ValueError("必须提供 query_layout 或 query_text")
        
        # 计算余弦相似度
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # 排序并过滤
        indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in indices:
            sim = similarities[idx]
            
            # 应用阈值过滤
            if sim < self.similarity_threshold:
                continue
            
            # 应用分数过滤
            if min_score is not None and self.cases[idx].score < min_score:
                continue
            
            results.append((self.cases[idx], float(sim)))
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
        return np.dot(vec2_norm, vec1_norm)
    
    def search_by_criteria(
        self,
        house_type: str = None,
        floor_type: str = None,
        room_count: int = None,
        min_area: float = None,
        max_area: float = None
    ) -> List[LayoutCase]:
        """
        根据条件检索案例
        
        Args:
            house_type: 户型类型
            floor_type: 楼层类型
            room_count: 房间数量
            min_area: 最小面积
            max_area: 最大面积
            
        Returns:
            List[LayoutCase]: 符合条件的案例列表
        """
        results = []
        
        for case in self.cases:
            # 检查户型类型
            if house_type and case.metadata.get('house_type') != house_type:
                continue
            
            # 检查楼层类型
            if floor_type and case.metadata.get('floor_type') != floor_type:
                continue
            
            # 检查房间数量
            if room_count:
                actual_count = self._count_rooms(case.layout)
                if actual_count != room_count:
                    continue
            
            # 检查面积
            if min_area or max_area:
                area = self._calculate_total_area(case.layout)
                if min_area and area < min_area:
                    continue
                if max_area and area > max_area:
                    continue
            
            results.append(case)
        
        return results
    
    def _count_rooms(self, layout: Dict[str, List[int]]) -> int:
        """统计房间数量"""
        count = 0
        for name in layout.keys():
            if name not in ['边界'] and '采光' not in name and '黑体' not in name and '入口' not in name:
                count += 1
        return count
    
    def _calculate_total_area(self, layout: Dict[str, List[int]]) -> float:
        """计算总面积（平方米）"""
        boundary = layout.get('边界')
        if boundary and len(boundary) == 4:
            return boundary[2] * boundary[3] / 1000000
        return 0.0
    
    def save(self, path: str):
        """
        保存知识库到文件
        
        Args:
            path: 保存路径
        """
        data = {
            'cases': [case.to_dict() for case in self.cases],
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'similarity_threshold': self.similarity_threshold
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """
        从文件加载知识库
        
        Args:
            path: 文件路径
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.cases = [LayoutCase.from_dict(d) for d in data['cases']]
        self.embeddings = np.array(data['embeddings']) if data['embeddings'] else None
        self.similarity_threshold = data.get('similarity_threshold', 0.5)
        
        # 恢复embedding
        if self.embeddings is not None:
            for i, case in enumerate(self.cases):
                case.embedding = self.embeddings[i]
    
    def __len__(self) -> int:
        return len(self.cases)
    
    def __getitem__(self, idx: int) -> LayoutCase:
        return self.cases[idx]

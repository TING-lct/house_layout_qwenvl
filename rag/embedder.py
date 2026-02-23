"""
向量嵌入模块
用于将布局转换为向量表示
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class LayoutEmbedder:
    """布局向量化器"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化向量化器
        
        Args:
            model_name: 句子向量模型名称
        """
        self.model = SentenceTransformer(model_name)
    
    def layout_to_text(self, layout: Dict[str, List[int]], metadata: Dict = None) -> str:
        """
        将布局转换为文本描述
        
        Args:
            layout: 布局字典
            metadata: 元数据（如户型类型等）
            
        Returns:
            str: 文本描述
        """
        text_parts = []
        
        # 添加元数据
        if metadata:
            if 'house_type' in metadata:
                text_parts.append(f"户型类型: {metadata['house_type']}")
            if 'floor_type' in metadata:
                text_parts.append(f"楼层: {metadata['floor_type']}")
            if 'total_area' in metadata:
                text_parts.append(f"总面积: {metadata['total_area']}平方米")
        
        # 提取边界信息
        boundary = layout.get('边界', None)
        if boundary:
            text_parts.append(f"边界尺寸: {boundary[2]}x{boundary[3]}mm")
        
        # 统计房间类型
        room_types = {}
        for name, params in layout.items():
            if name in ['边界'] or '采光' in name or '黑体' in name or '入口' in name:
                continue
            
            # 获取房间类型
            room_type = self._get_room_type(name)
            if room_type not in room_types:
                room_types[room_type] = []
            
            # 计算面积
            if len(params) == 4:
                area = params[2] * params[3] / 1000000  # 转换为平方米
                room_types[room_type].append(area)
        
        # 生成房间描述
        for room_type, areas in room_types.items():
            if len(areas) == 1:
                text_parts.append(f"{room_type}: {areas[0]:.1f}平方米")
            else:
                text_parts.append(f"{room_type}x{len(areas)}: 分别为{', '.join(f'{a:.1f}' for a in areas)}平方米")
        
        return "; ".join(text_parts)
    
    def _get_room_type(self, room_name: str) -> str:
        """从房间名获取房间类型"""
        type_mappings = {
            "主卧": ["主卧", "主卧室"],
            "卧室": ["卧室", "卧室1", "卧室2", "卧室3", "卧室4", "卧室5",
                     "次卧", "次卧1", "次卧2", "客卧"],
            "客厅": ["客厅", "起居室", "客餐厅"],
            "厨房": ["厨房", "厨房1", "中厨", "西厨"],
            "卫生间": ["卫生间", "卫生间1", "卫生间2", "公卫", "次卫"],
            "主卫": ["主卫", "主卫生间"],
            "餐厅": ["餐厅", "餐厅1", "饭厅"],
            "储藏": ["储藏", "储物间", "储藏室"],
            "阳台": ["阳台", "阳台1", "阳台2", "生活阳台"],
        }
        
        for room_type, names in type_mappings.items():
            if room_name in names:
                return room_type
        
        priority_order = ["主卧", "主卫", "卫生间", "卧室", "客厅", "厨房", "餐厅", "储藏", "阳台"]
        for room_type in priority_order:
            if room_type in room_name:
                return room_type
        
        return room_name
    
    def encode(self, layout: Dict[str, List[int]], metadata: Dict = None) -> np.ndarray:
        """
        将布局编码为向量
        
        Args:
            layout: 布局字典
            metadata: 元数据
            
        Returns:
            np.ndarray: 向量表示
        """
        text = self.layout_to_text(layout, metadata)
        return self.model.encode(text)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 向量表示
        """
        return self.model.encode(text)
    
    def encode_batch(
        self, 
        layouts: List[Dict[str, List[int]]], 
        metadata_list: List[Dict] = None
    ) -> np.ndarray:
        """
        批量编码布局
        
        Args:
            layouts: 布局列表
            metadata_list: 元数据列表
            
        Returns:
            np.ndarray: 向量矩阵
        """
        if metadata_list is None:
            metadata_list = [None] * len(layouts)
        
        texts = [
            self.layout_to_text(layout, metadata) 
            for layout, metadata in zip(layouts, metadata_list)
        ]
        
        return self.model.encode(texts)

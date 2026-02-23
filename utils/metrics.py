"""
评估指标模块
实现各种自动化评估指标
"""

import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricsResult:
    """指标计算结果"""
    space_utilization: float  # 空间利用率
    constraint_violations: int  # 约束违反数
    geometric_validity: float  # 几何合法性
    dimension_compliance: float  # 尺寸合规率
    overall_score: float  # 综合得分
    details: Dict[str, Any]


class LayoutMetrics:
    """户型布局评估指标"""
    
    def __init__(self, space_constraints: Dict = None):
        """
        初始化指标计算器
        
        Args:
            space_constraints: 空间尺寸约束
        """
        self.space_constraints = space_constraints or {
            "卧室": {"min_width": 2400, "min_length": 3000, "min_area": 7200000},
            "主卧": {"min_width": 3000, "min_length": 3600, "min_area": 10800000},
            "客厅": {"min_width": 3300, "min_length": 4500, "min_area": 14850000},
            "厨房": {"min_width": 1800, "min_length": 2400, "min_area": 4320000},
            "卫生间": {"min_width": 1500, "min_length": 2100, "min_area": 3150000},
            "主卫": {"min_width": 1800, "min_length": 2400, "min_area": 4320000},
            "餐厅": {"min_width": 1500, "min_length": 2000, "min_area": 3000000},
        }
    
    def calculate_all(
        self, 
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None
    ) -> MetricsResult:
        """
        计算所有指标
        
        Args:
            layout: 生成的布局
            full_layout: 完整布局
            
        Returns:
            MetricsResult: 指标结果
        """
        if full_layout:
            combined = {**full_layout, **layout}
        else:
            combined = layout
        
        # 解析布局
        rooms = self._parse_rooms(combined)
        boundary = self._get_boundary(combined)
        
        # 计算各项指标
        space_util = self.calculate_space_utilization(rooms, boundary)
        violations = self.count_constraint_violations(rooms, boundary)
        geo_validity = self.calculate_geometric_validity(rooms, boundary)
        dim_compliance = self.calculate_dimension_compliance(rooms)
        
        # 综合得分
        overall = (
            space_util * 0.2 +
            max(0, 100 - violations * 10) * 0.3 +
            geo_validity * 0.3 +
            dim_compliance * 0.2
        )
        
        return MetricsResult(
            space_utilization=space_util,
            constraint_violations=violations,
            geometric_validity=geo_validity,
            dimension_compliance=dim_compliance,
            overall_score=overall,
            details={
                'rooms_count': len(rooms),
                'boundary': boundary
            }
        )
    
    def _parse_rooms(
        self, 
        layout: Dict[str, List[int]]
    ) -> List[Dict[str, Any]]:
        """解析房间列表"""
        rooms = []
        
        for name, params in layout.items():
            if len(params) != 4:
                continue
            
            if name in ['边界'] or '采光' in name or '黑体' in name or '入口' in name:
                continue
            
            rooms.append({
                'name': name,
                'x': params[0],
                'y': params[1],
                'width': params[2],
                'height': params[3],
                'area': params[2] * params[3]
            })
        
        return rooms
    
    def _get_boundary(
        self, 
        layout: Dict[str, List[int]]
    ) -> Optional[Dict[str, int]]:
        """获取边界信息"""
        boundary = layout.get('边界')
        if boundary and len(boundary) == 4:
            return {
                'x': boundary[0],
                'y': boundary[1],
                'width': boundary[2],
                'height': boundary[3],
                'area': boundary[2] * boundary[3]
            }
        return None
    
    def calculate_space_utilization(
        self,
        rooms: List[Dict[str, Any]],
        boundary: Optional[Dict[str, int]]
    ) -> float:
        """
        计算空间利用率
        
        Args:
            rooms: 房间列表
            boundary: 边界信息
            
        Returns:
            float: 空间利用率 (0-100)
        """
        if not boundary:
            return 50.0  # 无边界信息时返回中等分数
        
        total_room_area = sum(room['area'] for room in rooms)
        boundary_area = boundary['area']
        
        if boundary_area == 0:
            return 0.0
        
        utilization = (total_room_area / boundary_area) * 100
        
        # 归一化到合理范围（一般利用率在60%-90%之间较好）
        if utilization > 100:
            # 超出边界，可能有重叠
            return max(0, 100 - (utilization - 100))
        elif utilization < 50:
            return utilization * 1.5  # 利用率低，适当惩罚
        else:
            return min(100, utilization * 1.1)  # 正常范围
    
    def count_constraint_violations(
        self,
        rooms: List[Dict[str, Any]],
        boundary: Optional[Dict[str, int]]
    ) -> int:
        """
        计算约束违反数
        
        Args:
            rooms: 房间列表
            boundary: 边界信息
            
        Returns:
            int: 违反数量
        """
        violations = 0
        
        # 检查房间重叠
        for i, room1 in enumerate(rooms):
            for room2 in rooms[i+1:]:
                if self._check_overlap(room1, room2):
                    violations += 1
        
        # 检查边界
        if boundary:
            for room in rooms:
                if not self._within_boundary(room, boundary):
                    violations += 1
        
        # 检查尺寸
        for room in rooms:
            room_type = self._get_room_type(room['name'])
            if room_type in self.space_constraints:
                constraints = self.space_constraints[room_type]
                
                width = min(room['width'], room['height'])
                length = max(room['width'], room['height'])
                
                if width < constraints.get('min_width', 0):
                    violations += 1
                if length < constraints.get('min_length', 0):
                    violations += 1
                if room['area'] < constraints.get('min_area', 0):
                    violations += 1
        
        return violations
    
    def _check_overlap(
        self, 
        room1: Dict[str, Any], 
        room2: Dict[str, Any]
    ) -> bool:
        """检查两个房间是否重叠"""
        return not (
            room1['x'] + room1['width'] <= room2['x'] or
            room2['x'] + room2['width'] <= room1['x'] or
            room1['y'] + room1['height'] <= room2['y'] or
            room2['y'] + room2['height'] <= room1['y']
        )
    
    def _within_boundary(
        self,
        room: Dict[str, Any],
        boundary: Dict[str, int]
    ) -> bool:
        """检查房间是否在边界内"""
        return (
            room['x'] >= boundary['x'] and
            room['y'] >= boundary['y'] and
            room['x'] + room['width'] <= boundary['x'] + boundary['width'] and
            room['y'] + room['height'] <= boundary['y'] + boundary['height']
        )
    
    def calculate_geometric_validity(
        self,
        rooms: List[Dict[str, Any]],
        boundary: Optional[Dict[str, int]]
    ) -> float:
        """
        计算几何合法性得分
        
        Args:
            rooms: 房间列表
            boundary: 边界信息
            
        Returns:
            float: 合法性得分 (0-100)
        """
        if not rooms:
            return 0.0
        
        total_score = 0.0
        
        for room in rooms:
            room_score = 100.0
            
            # 检查尺寸有效性
            if room['width'] <= 0 or room['height'] <= 0:
                room_score = 0.0
            else:
                # 检查长宽比
                ratio = max(room['width'], room['height']) / min(room['width'], room['height'])
                if ratio > 5:
                    room_score -= 30
                elif ratio > 3:
                    room_score -= 10
                
                # 检查面积合理性
                area_m2 = room['area'] / 1000000
                if area_m2 < 2:
                    room_score -= 20
                elif area_m2 > 100:
                    room_score -= 20
            
            # 检查边界
            if boundary and not self._within_boundary(room, boundary):
                room_score -= 30
            
            total_score += max(0, room_score)
        
        return total_score / len(rooms)
    
    def calculate_dimension_compliance(
        self,
        rooms: List[Dict[str, Any]]
    ) -> float:
        """
        计算尺寸合规率
        
        Args:
            rooms: 房间列表
            
        Returns:
            float: 合规率 (0-100)
        """
        if not rooms:
            return 0.0
        
        compliant_count = 0
        checked_count = 0
        
        for room in rooms:
            room_type = self._get_room_type(room['name'])
            
            if room_type in self.space_constraints:
                checked_count += 1
                constraints = self.space_constraints[room_type]
                
                width = min(room['width'], room['height'])
                length = max(room['width'], room['height'])
                
                is_compliant = (
                    width >= constraints.get('min_width', 0) and
                    length >= constraints.get('min_length', 0) and
                    room['area'] >= constraints.get('min_area', 0)
                )
                
                if is_compliant:
                    compliant_count += 1
        
        if checked_count == 0:
            return 100.0  # 没有需要检查的房间
        
        return (compliant_count / checked_count) * 100
    
    def _get_room_type(self, room_name: str) -> str:
        """从房间名获取房间类型"""
        type_mappings = {
            "卧室": ["卧室", "卧室1", "卧室2", "卧室3", "卧室4", "次卧"],
            "主卧": ["主卧"],
            "客厅": ["客厅"],
            "厨房": ["厨房"],
            "卫生间": ["卫生间", "公卫", "次卫"],
            "主卫": ["主卫"],
            "餐厅": ["餐厅"],
        }
        
        for room_type, names in type_mappings.items():
            if room_name in names:
                return room_type
        
        for room_type in type_mappings.keys():
            if room_type in room_name:
                return room_type
        
        return room_name


def compare_layouts(
    layout1: Dict[str, List[int]],
    layout2: Dict[str, List[int]],
    full_layout: Dict[str, List[int]] = None
) -> Dict[str, Any]:
    """
    比较两个布局的指标
    
    Args:
        layout1: 布局1
        layout2: 布局2
        full_layout: 完整布局（包括已有房间）
        
    Returns:
        Dict: 比较结果
    """
    metrics = LayoutMetrics()
    
    result1 = metrics.calculate_all(layout1, full_layout)
    result2 = metrics.calculate_all(layout2, full_layout)
    
    return {
        'layout1': {
            'space_utilization': result1.space_utilization,
            'constraint_violations': result1.constraint_violations,
            'geometric_validity': result1.geometric_validity,
            'dimension_compliance': result1.dimension_compliance,
            'overall_score': result1.overall_score
        },
        'layout2': {
            'space_utilization': result2.space_utilization,
            'constraint_violations': result2.constraint_violations,
            'geometric_validity': result2.geometric_validity,
            'dimension_compliance': result2.dimension_compliance,
            'overall_score': result2.overall_score
        },
        'comparison': {
            'space_utilization_diff': result1.space_utilization - result2.space_utilization,
            'violations_diff': result1.constraint_violations - result2.constraint_violations,
            'overall_diff': result1.overall_score - result2.overall_score,
            'winner': 'layout1' if result1.overall_score > result2.overall_score else 'layout2'
        }
    }

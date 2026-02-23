"""
户型布局评估器模块
基于多维度评分体系评估布局质量
"""

import json
import yaml
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Room:
    """房间数据结构"""
    name: str
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        """计算面积"""
        return self.width * self.height
    
    @property
    def x2(self) -> int:
        """右边界x坐标"""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """上边界y坐标"""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """中心点坐标"""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def overlaps(self, other: 'Room') -> bool:
        """检查是否与另一个房间重叠"""
        return not (
            self.x2 <= other.x or 
            other.x2 <= self.x or 
            self.y2 <= other.y or 
            other.y2 <= self.y
        )
    
    def is_adjacent(self, other: 'Room', tolerance: int = 100) -> bool:
        """检查是否与另一个房间相邻"""
        # 水平相邻
        h_adjacent = (
            abs(self.x2 - other.x) <= tolerance or 
            abs(other.x2 - self.x) <= tolerance
        ) and not (self.y2 <= other.y or other.y2 <= self.y)
        
        # 垂直相邻
        v_adjacent = (
            abs(self.y2 - other.y) <= tolerance or 
            abs(other.y2 - self.y) <= tolerance
        ) and not (self.x2 <= other.x or other.x2 <= self.x)
        
        return h_adjacent or v_adjacent


@dataclass
class EvaluationResult:
    """评估结果"""
    total_score: float
    dimension_scores: Dict[str, float]
    issues: List[str]
    suggestions: List[str]
    is_valid: bool
    details: Dict[str, Any]


class LayoutEvaluator:
    """户型布局评估器"""
    
    def __init__(self, rules_config_path: str = None):
        """
        初始化评估器
        
        Args:
            rules_config_path: 规则配置文件路径
        """
        # 加载规则配置
        if rules_config_path:
            with open(rules_config_path, 'r', encoding='utf-8') as f:
                self.rules_config = yaml.safe_load(f)
        else:
            self.rules_config = self._default_rules()
        
        # 评分权重
        self.weights = self.rules_config.get('scoring_weights', {
            "空间合理性": 0.25,
            "采光通风": 0.20,
            "动线设计": 0.20,
            "功能分区": 0.20,
            "尺寸规范": 0.15
        })
        
        # 空间约束
        self.space_constraints = self.rules_config.get('space_constraints', {})
        
        # 相邻规则
        self.adjacency_rules = self.rules_config.get('adjacency_rules', {})
        
        # 采光规则
        self.lighting_rules = self.rules_config.get('lighting_rules', {})
        
        # 扣分配置
        self.penalties = self.rules_config.get('penalty_scores', {})
    
    def _default_rules(self) -> Dict:
        """默认规则配置"""
        return {
            'scoring_weights': {
                "空间合理性": 0.25,
                "采光通风": 0.20,
                "动线设计": 0.20,
                "功能分区": 0.20,
                "尺寸规范": 0.15
            },
            'space_constraints': {
                "卧室": {"min_width": 2400, "min_length": 3000, "min_area": 7200000},
                "客厅": {"min_width": 3300, "min_length": 4500, "min_area": 14850000},
                "厨房": {"min_width": 1800, "min_length": 2400, "min_area": 4320000},
                "卫生间": {"min_width": 1500, "min_length": 2100, "min_area": 3150000},
                "餐厅": {"min_width": 1500, "min_length": 2000, "min_area": 3000000},
            },
            'adjacency_rules': {
                'forbidden_pairs': [
                    ["厨房", "卫生间"],
                    ["厨房", "主卫"],
                ]
            },
            'lighting_rules': {
                'require_lighting': ["客厅", "卧室", "主卧"]
            },
            'penalty_scores': {
                'room_overlap': 30,
                'boundary_exceed': 25,
                'min_size_violation': 20,
                'forbidden_adjacent': 15,
                'no_lighting': 10
            }
        }
    
    def parse_layout(
        self, 
        layout_dict: Dict[str, List[int]],
        boundary: List[int] = None
    ) -> Tuple[List[Room], Optional[Room]]:
        """
        解析布局字典为Room对象列表
        
        Args:
            layout_dict: 布局字典 {房间名: [x, y, width, height]}
            boundary: 边界 [x, y, width, height]
            
        Returns:
            Tuple[List[Room], Optional[Room]]: 房间列表和边界
        """
        rooms = []
        boundary_room = None
        
        for name, params in layout_dict.items():
            if len(params) != 4:
                continue
            
            room = Room(
                name=name,
                x=params[0],
                y=params[1],
                width=params[2],
                height=params[3]
            )
            
            if name == "边界":
                boundary_room = room
            elif not name.startswith(("采光", "南采光", "北采光", "东采光", "西采光", "黑体", "主入口")):
                rooms.append(room)
        
        # 如果提供了边界参数
        if boundary and not boundary_room:
            boundary_room = Room(
                name="边界",
                x=boundary[0],
                y=boundary[1],
                width=boundary[2],
                height=boundary[3]
            )
        
        return rooms, boundary_room
    
    def score(self, layout: Dict[str, List[int]], full_layout: Dict[str, List[int]] = None) -> float:
        """
        计算布局总分
        
        Args:
            layout: 生成的布局
            full_layout: 完整布局（包括已有房间）
            
        Returns:
            float: 总分 (0-100)
        """
        result = self.evaluate(layout, full_layout)
        return result.total_score
    
    def evaluate(
        self, 
        layout: Dict[str, List[int]], 
        full_layout: Dict[str, List[int]] = None
    ) -> EvaluationResult:
        """
        全面评估布局
        
        Args:
            layout: 生成的布局
            full_layout: 完整布局（包括已有房间和边界）
            
        Returns:
            EvaluationResult: 评估结果
        """
        # 合并布局
        if full_layout:
            combined_layout = {**full_layout, **layout}
        else:
            combined_layout = layout
        
        # 解析布局
        rooms, boundary = self.parse_layout(combined_layout)
        
        issues = []
        suggestions = []
        details = {}
        
        # 1. 空间合理性评分
        space_score, space_issues = self._check_space_rationality(rooms, boundary)
        issues.extend(space_issues)
        details['space_rationality'] = {'score': space_score, 'issues': space_issues}
        
        # 2. 采光通风评分
        lighting_score, lighting_issues = self._check_lighting_ventilation(rooms, combined_layout)
        issues.extend(lighting_issues)
        details['lighting'] = {'score': lighting_score, 'issues': lighting_issues}
        
        # 3. 动线设计评分
        traffic_score, traffic_issues = self._check_traffic_flow(rooms, combined_layout)
        issues.extend(traffic_issues)
        details['traffic'] = {'score': traffic_score, 'issues': traffic_issues}
        
        # 4. 功能分区评分
        zoning_score, zoning_issues = self._check_functional_zoning(rooms)
        issues.extend(zoning_issues)
        details['zoning'] = {'score': zoning_score, 'issues': zoning_issues}
        
        # 5. 尺寸规范评分
        dimension_score, dimension_issues = self._check_dimension_standards(rooms)
        issues.extend(dimension_issues)
        details['dimension'] = {'score': dimension_score, 'issues': dimension_issues}
        
        # 计算总分
        dimension_scores = {
            "空间合理性": space_score,
            "采光通风": lighting_score,
            "动线设计": traffic_score,
            "功能分区": zoning_score,
            "尺寸规范": dimension_score
        }
        
        total_score = sum(
            score * self.weights[name] 
            for name, score in dimension_scores.items()
        )
        
        # 生成建议
        suggestions = self._generate_suggestions(issues)
        
        # 判断是否有效（总分>60且无严重问题）
        is_valid = total_score >= 60 and space_score >= 50
        
        return EvaluationResult(
            total_score=total_score,
            dimension_scores=dimension_scores,
            issues=issues,
            suggestions=suggestions,
            is_valid=is_valid,
            details=details
        )
    
    def _check_space_rationality(
        self, 
        rooms: List[Room], 
        boundary: Optional[Room]
    ) -> Tuple[float, List[str]]:
        """检查空间合理性"""
        score = 100.0
        issues = []
        
        # 检查房间重叠
        for i, room1 in enumerate(rooms):
            for room2 in rooms[i+1:]:
                if room1.overlaps(room2):
                    score -= self.penalties.get('room_overlap', 30)
                    issues.append(f"房间重叠: {room1.name} 与 {room2.name}")
        
        # 检查边界
        if boundary:
            for room in rooms:
                if not self._is_within_boundary(room, boundary):
                    score -= self.penalties.get('boundary_exceed', 25)
                    issues.append(f"超出边界: {room.name}")
        
        return max(0, score), issues
    
    def _is_within_boundary(self, room: Room, boundary: Room) -> bool:
        """检查房间是否在边界内"""
        return (
            room.x >= boundary.x and
            room.y >= boundary.y and
            room.x2 <= boundary.x2 and
            room.y2 <= boundary.y2
        )
    
    def _check_lighting_ventilation(
        self, 
        rooms: List[Room],
        full_layout: Dict[str, List[int]]
    ) -> Tuple[float, List[str]]:
        """检查采光通风"""
        score = 100.0
        issues = []
        
        # 获取采光面
        lighting_surfaces = []
        for name, params in full_layout.items():
            if "采光" in name and len(params) == 4:
                lighting_surfaces.append(Room(
                    name=name, x=params[0], y=params[1],
                    width=params[2], height=params[3]
                ))
        
        # 检查需要采光的房间
        require_lighting = self.lighting_rules.get('require_lighting', [])
        
        for room in rooms:
            # 检查房间类型是否需要采光
            room_type = self._get_room_type(room.name)
            if room_type in require_lighting:
                if not self._is_near_lighting(room, lighting_surfaces):
                    score -= self.penalties.get('no_lighting', 10)
                    issues.append(f"采光不足: {room.name}")
        
        return max(0, score), issues
    
    def _is_near_lighting(
        self, 
        room: Room, 
        lighting_surfaces: List[Room],
        threshold: int = 500
    ) -> bool:
        """检查房间是否靠近采光面"""
        for surface in lighting_surfaces:
            # 检查是否相邻
            if room.is_adjacent(surface, tolerance=threshold):
                return True
        return False
    
    def _check_traffic_flow(
        self, 
        rooms: List[Room],
        full_layout: Dict[str, List[int]]
    ) -> Tuple[float, List[str]]:
        """检查动线设计"""
        score = 100.0
        issues = []
        
        # 获取入口位置
        entry = None
        for name, params in full_layout.items():
            if "入口" in name and len(params) == 4:
                entry = Room(name=name, x=params[0], y=params[1],
                           width=params[2], height=params[3])
                break
        
        if entry:
            # 检查客厅是否靠近入口
            living_room = next((r for r in rooms if "客厅" in r.name), None)
            if living_room:
                distance = self._calculate_distance(entry, living_room)
                if distance > 8000:  # 8米
                    score -= 15
                    issues.append("客厅距离入口过远")
        
        return max(0, score), issues
    
    def _calculate_distance(self, room1: Room, room2: Room) -> float:
        """计算两个房间中心点之间的距离"""
        c1 = room1.center
        c2 = room2.center
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
    
    def _check_functional_zoning(self, rooms: List[Room]) -> Tuple[float, List[str]]:
        """检查功能分区"""
        score = 100.0
        issues = []
        
        # 检查禁止相邻的房间对
        forbidden_pairs = self.adjacency_rules.get('forbidden_pairs', [])
        
        for room1 in rooms:
            for room2 in rooms:
                if room1.name >= room2.name:
                    continue
                
                type1 = self._get_room_type(room1.name)
                type2 = self._get_room_type(room2.name)
                
                # 检查是否为禁止相邻的组合
                for pair in forbidden_pairs:
                    if (type1 == pair[0] and type2 == pair[1]) or \
                       (type1 == pair[1] and type2 == pair[0]):
                        if room1.is_adjacent(room2):
                            score -= self.penalties.get('forbidden_adjacent', 15)
                            issues.append(f"不宜相邻: {room1.name} 与 {room2.name}")
        
        return max(0, score), issues
    
    def _check_dimension_standards(self, rooms: List[Room]) -> Tuple[float, List[str]]:
        """检查尺寸规范"""
        score = 100.0
        issues = []
        
        for room in rooms:
            room_type = self._get_room_type(room.name)
            
            if room_type in self.space_constraints:
                constraints = self.space_constraints[room_type]
                
                min_width = constraints.get('min_width', 0)
                min_length = constraints.get('min_length', 0)
                min_area = constraints.get('min_area', 0)
                
                actual_width = min(room.width, room.height)
                actual_length = max(room.width, room.height)
                
                if actual_width < min_width:
                    score -= self.penalties.get('min_size_violation', 20)
                    issues.append(f"宽度不足: {room.name} (最小{min_width}mm)")
                
                if actual_length < min_length:
                    score -= self.penalties.get('min_size_violation', 20)
                    issues.append(f"长度不足: {room.name} (最小{min_length}mm)")
                
                if room.area < min_area:
                    score -= self.penalties.get('min_size_violation', 20)
                    issues.append(f"面积不足: {room.name} (最小{min_area/1000000:.1f}平米)")
        
        return max(0, score), issues
    
    def _get_room_type(self, room_name: str) -> str:
        """从房间名获取房间类型"""
        # 移除数字后缀
        type_mappings = {
            "卧室": ["卧室", "卧室1", "卧室2", "卧室3", "卧室4", "次卧"],
            "主卧": ["主卧"],
            "客厅": ["客厅"],
            "厨房": ["厨房"],
            "卫生间": ["卫生间", "公卫", "次卫"],
            "主卫": ["主卫"],
            "餐厅": ["餐厅"],
            "储藏": ["储藏", "储物间", "储藏室"],
        }
        
        for room_type, names in type_mappings.items():
            if room_name in names:
                return room_type
        
        # 尝试模糊匹配
        for room_type in type_mappings.keys():
            if room_type in room_name:
                return room_type
        
        return room_name
    
    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """根据问题生成建议"""
        suggestions = []
        
        if any("重叠" in issue for issue in issues):
            suggestions.append("调整房间位置，避免空间重叠")
        
        if any("超出边界" in issue for issue in issues):
            suggestions.append("缩小房间尺寸或调整位置，确保在边界内")
        
        if any("采光" in issue for issue in issues):
            suggestions.append("将卧室和客厅移至靠近采光面的位置")
        
        if any("不宜相邻" in issue for issue in issues):
            suggestions.append("调整厨房和卫生间的位置，避免直接相邻")
        
        if any("面积不足" in issue or "宽度不足" in issue or "长度不足" in issue for issue in issues):
            suggestions.append("增大房间尺寸以满足最小规范要求")
        
        if any("入口" in issue for issue in issues):
            suggestions.append("将客厅布置在靠近入口的位置")
        
        return suggestions

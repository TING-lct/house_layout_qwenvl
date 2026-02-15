"""
规则约束引擎模块
实现硬性规则检查和软性规则建议
"""

import json
import yaml
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

from .evaluator import Room, LayoutEvaluator


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    hard_violations: List[str]
    soft_violations: List[str]
    fixed_layout: Optional[Dict[str, List[int]]] = None
    fix_applied: bool = False


class LayoutRuleEngine:
    """户型布局规则引擎"""
    
    def __init__(self, rules_config_path: str = None):
        """
        初始化规则引擎
        
        Args:
            rules_config_path: 规则配置文件路径
        """
        # 加载规则配置
        if rules_config_path:
            with open(rules_config_path, 'r', encoding='utf-8') as f:
                self.rules_config = yaml.safe_load(f)
        else:
            self.rules_config = {}
        
        # 空间约束
        self.space_constraints = self.rules_config.get('space_constraints', {
            "卧室": {"min_width": 2400, "min_length": 3000, "min_area": 7200000},
            "客厅": {"min_width": 3300, "min_length": 4500, "min_area": 14850000},
            "厨房": {"min_width": 1800, "min_length": 2400, "min_area": 4320000},
            "卫生间": {"min_width": 1500, "min_length": 2100, "min_area": 3150000},
            "餐厅": {"min_width": 2400, "min_length": 3000, "min_area": 7200000},
        })
        
        # 相邻规则
        self.adjacency_rules = self.rules_config.get('adjacency_rules', {
            'forbidden_pairs': [
                ["厨房", "卫生间"],
                ["厨房", "主卫"],
            ],
            'recommended_pairs': [
                ["厨房", "餐厅"],
                ["客厅", "餐厅"],
            ]
        })
        
        # 注册硬性规则
        self.hard_rules: List[Callable] = [
            self.no_room_overlap,
            self.within_boundary,
            self.minimum_dimensions,
            self.positive_dimensions,
        ]
        
        # 注册软性规则
        self.soft_rules: List[Callable] = [
            self.kitchen_bathroom_separate,
            self.bedroom_near_lighting,
            self.reasonable_proportions,
        ]
    
    def parse_layout(
        self, 
        layout_dict: Dict[str, List[int]]
    ) -> Tuple[List[Room], Optional[Room], Dict[str, Room]]:
        """
        解析布局字典
        
        Returns:
            Tuple: (普通房间列表, 边界, 所有房间字典)
        """
        rooms = []
        boundary = None
        all_rooms = {}
        
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
            all_rooms[name] = room
            
            if name == "边界":
                boundary = room
            elif not name.startswith(("采光", "南采光", "北采光", "东采光", "西采光", "黑体", "主入口")):
                rooms.append(room)
        
        return rooms, boundary, all_rooms
    
    def validate(
        self, 
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None
    ) -> ValidationResult:
        """
        验证布局是否符合规则
        
        Args:
            layout: 生成的布局
            full_layout: 完整布局（包括已有房间和边界）
            
        Returns:
            ValidationResult: 验证结果
        """
        # 合并布局
        if full_layout:
            combined_layout = {**full_layout, **layout}
        else:
            combined_layout = layout
        
        rooms, boundary, all_rooms = self.parse_layout(combined_layout)
        
        hard_violations = []
        soft_violations = []
        
        # 检查硬性规则
        for rule in self.hard_rules:
            result = rule(rooms, boundary, all_rooms)
            if not result['passed']:
                hard_violations.extend(result['violations'])
        
        # 检查软性规则
        for rule in self.soft_rules:
            result = rule(rooms, boundary, all_rooms)
            if not result['passed']:
                soft_violations.extend(result['violations'])
        
        return ValidationResult(
            valid=len(hard_violations) == 0,
            hard_violations=hard_violations,
            soft_violations=soft_violations
        )
    
    def validate_and_fix(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None,
        max_attempts: int = 3
    ) -> ValidationResult:
        """
        验证并尝试修复布局
        
        Args:
            layout: 生成的布局
            full_layout: 完整布局
            max_attempts: 最大修复尝试次数
            
        Returns:
            ValidationResult: 验证结果（包含修复后的布局）
        """
        current_layout = layout.copy()
        
        for attempt in range(max_attempts):
            result = self.validate(current_layout, full_layout)
            
            if result.valid:
                result.fixed_layout = current_layout
                result.fix_applied = (attempt > 0)
                return result
            
            # 尝试修复
            current_layout = self.auto_fix(
                current_layout, 
                result.hard_violations,
                full_layout
            )
        
        # 最终验证
        final_result = self.validate(current_layout, full_layout)
        final_result.fixed_layout = current_layout
        final_result.fix_applied = True
        
        return final_result
    
    def auto_fix(
        self,
        layout: Dict[str, List[int]],
        violations: List[str],
        full_layout: Dict[str, List[int]] = None
    ) -> Dict[str, List[int]]:
        """
        自动修复违规项
        
        Args:
            layout: 当前布局
            violations: 违规列表
            full_layout: 完整布局
            
        Returns:
            Dict: 修复后的布局
        """
        fixed_layout = {k: v.copy() for k, v in layout.items()}
        
        # 合并布局获取边界信息
        if full_layout:
            combined = {**full_layout, **layout}
        else:
            combined = layout
        
        _, boundary, _ = self.parse_layout(combined)
        
        for violation in violations:
            if "重叠" in violation:
                fixed_layout = self._fix_overlap(fixed_layout, violation)
            elif "超出边界" in violation:
                fixed_layout = self._fix_boundary(fixed_layout, violation, boundary)
            elif "尺寸" in violation:
                fixed_layout = self._fix_dimensions(fixed_layout, violation)
        
        return fixed_layout
    
    def _fix_overlap(
        self, 
        layout: Dict[str, List[int]], 
        violation: str
    ) -> Dict[str, List[int]]:
        """修复重叠问题"""
        # 从违规信息提取房间名
        # 简单策略：稍微移动第二个房间
        parts = violation.split(":")
        if len(parts) > 1:
            room_info = parts[1].strip()
            if "与" in room_info:
                room_names = room_info.split("与")
                if len(room_names) == 2:
                    room2_name = room_names[1].strip()
                    if room2_name in layout:
                        # 移动房间
                        layout[room2_name][0] += 300  # x方向移动300mm
        
        return layout
    
    def _fix_boundary(
        self,
        layout: Dict[str, List[int]],
        violation: str,
        boundary: Optional[Room]
    ) -> Dict[str, List[int]]:
        """修复边界问题"""
        if not boundary:
            return layout
        
        # 提取房间名
        parts = violation.split(":")
        if len(parts) > 1:
            room_name = parts[1].strip()
            if room_name in layout:
                params = layout[room_name]
                
                # 调整位置使其在边界内
                if params[0] < boundary.x:
                    params[0] = boundary.x
                if params[1] < boundary.y:
                    params[1] = boundary.y
                if params[0] + params[2] > boundary.x2:
                    params[0] = boundary.x2 - params[2]
                if params[1] + params[3] > boundary.y2:
                    params[1] = boundary.y2 - params[3]
                
                # 如果还是超出，缩小尺寸
                if params[0] < boundary.x:
                    params[2] = params[2] - (boundary.x - params[0])
                    params[0] = boundary.x
                if params[1] < boundary.y:
                    params[3] = params[3] - (boundary.y - params[1])
                    params[1] = boundary.y
        
        return layout
    
    def _fix_dimensions(
        self,
        layout: Dict[str, List[int]],
        violation: str
    ) -> Dict[str, List[int]]:
        """修复尺寸问题"""
        # 提取房间名和约束
        parts = violation.split(":")
        if len(parts) > 1:
            room_info = parts[1].strip()
            room_name = room_info.split("(")[0].strip()
            
            if room_name in layout:
                room_type = self._get_room_type(room_name)
                if room_type in self.space_constraints:
                    constraints = self.space_constraints[room_type]
                    params = layout[room_name]
                    
                    # 扩大尺寸
                    if params[2] < constraints.get('min_width', 0):
                        params[2] = constraints['min_width']
                    if params[3] < constraints.get('min_length', 0):
                        params[3] = constraints['min_length']
        
        return layout
    
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
    
    # ==================== 硬性规则 ====================
    
    def no_room_overlap(
        self, 
        rooms: List[Room], 
        boundary: Optional[Room],
        all_rooms: Dict[str, Room]
    ) -> Dict[str, Any]:
        """硬性规则：房间不能重叠"""
        violations = []
        
        for i, room1 in enumerate(rooms):
            for room2 in rooms[i+1:]:
                if room1.overlaps(room2):
                    violations.append(f"房间重叠: {room1.name} 与 {room2.name}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    def within_boundary(
        self,
        rooms: List[Room],
        boundary: Optional[Room],
        all_rooms: Dict[str, Room]
    ) -> Dict[str, Any]:
        """硬性规则：房间不能超出边界"""
        violations = []
        
        if boundary:
            for room in rooms:
                if room.x < boundary.x or room.y < boundary.y or \
                   room.x2 > boundary.x2 or room.y2 > boundary.y2:
                    violations.append(f"超出边界: {room.name}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    def minimum_dimensions(
        self,
        rooms: List[Room],
        boundary: Optional[Room],
        all_rooms: Dict[str, Room]
    ) -> Dict[str, Any]:
        """硬性规则：房间尺寸不能小于最小要求"""
        violations = []
        
        for room in rooms:
            room_type = self._get_room_type(room.name)
            
            if room_type in self.space_constraints:
                constraints = self.space_constraints[room_type]
                min_width = constraints.get('min_width', 0)
                min_length = constraints.get('min_length', 0)
                
                actual_width = min(room.width, room.height)
                actual_length = max(room.width, room.height)
                
                if actual_width < min_width * 0.8:  # 允许20%容差
                    violations.append(f"宽度严重不足: {room.name}")
                if actual_length < min_length * 0.8:
                    violations.append(f"长度严重不足: {room.name}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    def positive_dimensions(
        self,
        rooms: List[Room],
        boundary: Optional[Room],
        all_rooms: Dict[str, Room]
    ) -> Dict[str, Any]:
        """硬性规则：尺寸必须为正数"""
        violations = []
        
        for room in rooms:
            if room.width <= 0 or room.height <= 0:
                violations.append(f"无效尺寸: {room.name}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    # ==================== 软性规则 ====================
    
    def kitchen_bathroom_separate(
        self,
        rooms: List[Room],
        boundary: Optional[Room],
        all_rooms: Dict[str, Room]
    ) -> Dict[str, Any]:
        """软性规则：厨房与卫生间应分离"""
        violations = []
        
        kitchen = next((r for r in rooms if "厨房" in r.name), None)
        bathrooms = [r for r in rooms if "卫" in r.name]
        
        if kitchen:
            for bathroom in bathrooms:
                if kitchen.is_adjacent(bathroom, tolerance=200):
                    violations.append(f"厨卫相邻: {kitchen.name} 与 {bathroom.name}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    def bedroom_near_lighting(
        self,
        rooms: List[Room],
        boundary: Optional[Room],
        all_rooms: Dict[str, Room]
    ) -> Dict[str, Any]:
        """软性规则：卧室应靠近采光面"""
        violations = []
        
        # 获取采光面
        lighting_surfaces = [
            r for name, r in all_rooms.items() 
            if "采光" in name
        ]
        
        bedrooms = [r for r in rooms if "卧" in r.name]
        
        for bedroom in bedrooms:
            near_lighting = False
            for surface in lighting_surfaces:
                if bedroom.is_adjacent(surface, tolerance=500):
                    near_lighting = True
                    break
            
            if not near_lighting and lighting_surfaces:
                violations.append(f"采光不足: {bedroom.name}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    def reasonable_proportions(
        self,
        rooms: List[Room],
        boundary: Optional[Room],
        all_rooms: Dict[str, Room]
    ) -> Dict[str, Any]:
        """软性规则：房间长宽比应合理"""
        violations = []
        
        for room in rooms:
            if room.width > 0 and room.height > 0:
                ratio = max(room.width, room.height) / min(room.width, room.height)
                if ratio > 4:  # 长宽比超过4:1
                    violations.append(f"比例失调: {room.name} (长宽比 {ratio:.1f}:1)")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }

"""
户型布局评估器模块
基于多维度评分体系评估布局质量
"""

import json
import logging
import yaml
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .common import (
    Room, get_room_type, is_normal_room, is_directional_lighting,
    is_infrastructure, parse_layout as _parse_layout, ParsedLayout,
    BOUNDARY_NAME,
)

logger = logging.getLogger(__name__)


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

    def __init__(self, rules_config_path: Optional[str] = None):
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
                'room_overlap': 20,
                'infra_overlap': 15,
                'boundary_exceed': 20,
                'min_size_violation': 15,
                'forbidden_adjacent': 12,
                'no_lighting': 8
            }
        }

    def parse_layout(
        self,
        layout_dict: Dict[str, List[int]],
        boundary: Optional[List[int]] = None
    ) -> Tuple[List[Room], Optional[Room]]:
        """解析布局字典为Room对象列表（委托给 common.parse_layout）"""
        parsed = _parse_layout(layout_dict)
        boundary_room = parsed.boundary

        if boundary and not boundary_room:
            boundary_room = Room(
                name=BOUNDARY_NAME,
                x=boundary[0], y=boundary[1],
                width=boundary[2], height=boundary[3]
            )

        return parsed.rooms, boundary_room

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
        full_layout: Optional[Dict[str, List[int]]] = None
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

        # 1. 空间合理性评分（含基础设施重叠检查）
        space_score, space_issues = self._check_space_rationality(
            rooms, boundary, combined_layout)
        issues.extend(space_issues)
        details['space_rationality'] = {
            'score': space_score, 'issues': space_issues}

        # 2. 采光通风评分
        lighting_score, lighting_issues = self._check_lighting_ventilation(
            rooms, combined_layout)
        issues.extend(lighting_issues)
        details['lighting'] = {
            'score': lighting_score, 'issues': lighting_issues}

        # 3. 动线设计评分
        traffic_score, traffic_issues = self._check_traffic_flow(
            rooms, combined_layout)
        issues.extend(traffic_issues)
        details['traffic'] = {'score': traffic_score, 'issues': traffic_issues}

        # 4. 功能分区评分
        zoning_score, zoning_issues = self._check_functional_zoning(rooms)
        issues.extend(zoning_issues)
        details['zoning'] = {'score': zoning_score, 'issues': zoning_issues}

        # 5. 尺寸规范评分
        dimension_score, dimension_issues = self._check_dimension_standards(
            rooms)
        issues.extend(dimension_issues)
        details['dimension'] = {
            'score': dimension_score, 'issues': dimension_issues}

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
        boundary: Optional[Room],
        full_layout: Dict[str, List[int]] = None
    ) -> Tuple[float, List[str]]:
        """检查空间合理性（含基础设施重叠检查）"""
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

        # 检查房间与基础设施重叠（采光区、黑体、主入口）
        if full_layout:
            parsed_full = _parse_layout(full_layout)
            for room in rooms:
                for infra_room in parsed_full.infra:
                    if room.overlaps(infra_room):
                        score -= self.penalties.get('infra_overlap',
                                                    self.penalties.get('room_overlap', 20))
                        issues.append(
                            f"房间与基础设施重叠: {room.name} 与 {infra_room.name}")

        # 检查空间覆盖率（房间总面积占可用面积比例）
        if boundary and rooms:
            total_room_area = sum(r.area for r in rooms)
            infra_area = 0
            if full_layout:
                parsed_full_cov = _parse_layout(full_layout)
                infra_area = sum(r.area for r in parsed_full_cov.infra)
            available_area = max(boundary.area - infra_area, 1)
            coverage = total_room_area / available_area
            if coverage < 0.50:
                score -= 10
                issues.append(f"空间覆盖率严重不足({coverage:.0%})，存在大面积空白")
            elif coverage < 0.60:
                score -= 5
                issues.append(f"空间覆盖率偏低({coverage:.0%})")
            # 注意：60%以上的覆盖率是正常的，墙体/走廊/过渡区需要占用空间

        return max(0, score), issues

    @staticmethod
    def _is_within_boundary(room: Room, boundary: Room) -> bool:
        """检查房间是否在边界内"""
        return room.is_within(boundary)

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
        """检查动线设计：客厅靠近入口 + 空间覆盖率 + 餐厅厨房联动"""
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
            # 检查客厅是否靠近入口（阈值根据边界尺寸自适应）
            living_room = next((r for r in rooms if "客厅" in r.name), None)
            if living_room:
                # 计算自适应距离阈值：边界对角线的40%，范围限制在3000-8000mm
                boundary_for_dist = None
                for bname, bparams in full_layout.items():
                    if bname == "边界" and len(bparams) == 4:
                        boundary_for_dist = Room(name=bname, x=bparams[0], y=bparams[1],
                                                 width=bparams[2], height=bparams[3])
                        break
                if boundary_for_dist:
                    diag = (boundary_for_dist.width**2 +
                            boundary_for_dist.height**2)**0.5
                    dist_threshold = max(3000, min(8000, diag * 0.4))
                else:
                    dist_threshold = 5000

                distance = self._calculate_distance(entry, living_room)
                if distance > dist_threshold:
                    # 渐进扣分：超出越多扣越多，但封顶30分
                    exceed_ratio = (distance - dist_threshold) / \
                        max(dist_threshold, 1)
                    penalty = min(30, 10 + exceed_ratio * 20)
                    score -= penalty
                    issues.append(
                        f"客厅距离入口过远({distance:.0f}mm，阈值{dist_threshold:.0f}mm)")
                elif distance > dist_threshold * 0.8:
                    # 轻微扣分：接近阈值时给出提醒
                    score -= 5
                    issues.append(
                        f"客厅距离入口偏远({distance:.0f}mm，建议<{dist_threshold:.0f}mm)")

        return max(0, score), issues

    @staticmethod
    def _calculate_distance(room1: Room, room2: Room) -> float:
        """计算两个房间中心点之间的距离"""
        return room1.distance_to(room2)

    def _check_functional_zoning(self, rooms: List[Room]) -> Tuple[float, List[str]]:
        """检查功能分区：禁止相邻 + 推荐相邻"""
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
                            score -= self.penalties.get(
                                'forbidden_adjacent', 15)
                            issues.append(f"不宜相邻: {room1.name} 与 {room2.name}")

        # 检查推荐相邻的房间对（餐厅-厨房、主卧-主卫等）
        recommended_pairs = self.adjacency_rules.get('recommended_pairs', [])
        for pair in recommended_pairs:
            rooms_a = [r for r in rooms if self._get_room_type(
                r.name) == pair[0]]
            rooms_b = [r for r in rooms if self._get_room_type(
                r.name) == pair[1]]
            if rooms_a and rooms_b:
                any_adjacent = False
                for ra in rooms_a:
                    for rb in rooms_b:
                        if ra.is_adjacent(rb, tolerance=500):
                            any_adjacent = True
                            break
                    if any_adjacent:
                        break
                if not any_adjacent:
                    score -= 8
                    issues.append(f"建议相邻但未相邻: {pair[0]} 与 {pair[1]}")

        return max(0, score), issues

    def _check_dimension_standards(self, rooms: List[Room]) -> Tuple[float, List[str]]:
        """检查尺寸规范：最小尺寸 + 最大面积 + 长宽比"""
        score = 100.0
        issues = []

        for room in rooms:
            room_type = self._get_room_type(room.name)

            if room_type in self.space_constraints:
                constraints = self.space_constraints[room_type]

                min_width = constraints.get('min_width', 0)
                min_length = constraints.get('min_length', 0)
                min_area = constraints.get('min_area', 0)
                max_area = constraints.get('max_area', float('inf'))

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
                    issues.append(
                        f"面积不足: {room.name} (最小{min_area/1000000:.1f}平米)")

                if room.area > max_area:
                    score -= 15
                    issues.append(
                        f"面积过大: {room.name} ({room.area/1000000:.1f}平米，最大{max_area/1000000:.1f}平米)")

            # 检查长宽比
            if room.width > 0 and room.height > 0:
                ratio = max(room.width, room.height) / \
                    min(room.width, room.height)
                if ratio > 4.0:
                    score -= 10
                    issues.append(f"比例失调: {room.name} (长宽比{ratio:.1f}:1)")
                elif ratio > 3.0:
                    score -= 5
                    issues.append(
                        f"比例偏长: {room.name} (长宽比{ratio:.1f}:1，建议≤3:1)")

        return max(0, score), issues

    @staticmethod
    def _get_room_type(room_name: str) -> str:
        """从房间名获取房间类型（委托给 common.get_room_type）"""
        return get_room_type(room_name)

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

        if any("覆盖率" in issue for issue in issues):
            suggestions.append("增大房间尺寸以填充空白区域，提高空间利用率")

        if any("比例失调" in issue for issue in issues):
            suggestions.append("调整房间长宽比，使其接近1:1到3:1的合理范围")

        if any("面积过大" in issue for issue in issues):
            suggestions.append("缩小过大的房间，为其他功能区留出空间")

        if any("建议相邻但未相邻" in issue for issue in issues):
            suggestions.append("调整餐厅靠近厨房、主卧靠近主卫的位置")

        return suggestions

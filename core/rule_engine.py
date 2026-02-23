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
            "餐厅": {"min_width": 1500, "min_length": 2000, "min_area": 3000000},
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
    
    def optimize_dimensions(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None
    ) -> Dict[str, List[int]]:
        """
        保守优化房间尺寸：仅在不产生新重叠的前提下扩大房间，
        使其满足评分器的100%最小尺寸标准（硬性规则仅用80%容差）。
        每次修正后独立检查重叠，若冲突则回退该次修正。
        """
        fixed = {k: v.copy() for k, v in layout.items()}
        combined_base = dict(full_layout or {})
        _, boundary, _ = self.parse_layout({**combined_base, **layout})
        
        for name in list(fixed.keys()):
            room_type = self._get_room_type(name)
            if room_type not in self.space_constraints:
                continue
            
            constraints = self.space_constraints[room_type]
            min_w = constraints.get('min_width', 0)
            min_l = constraints.get('min_length', 0)
            params = fixed[name]
            
            # --- 短边修正 ---
            if min(params[2], params[3]) < min_w:
                saved = params[:]
                if params[2] <= params[3]:
                    params[2] = min_w
                else:
                    params[3] = min_w
                self._clamp_boundary(params, boundary)
                if self._would_overlap(name, params, fixed, combined_base):
                    params[:] = saved
            
            # --- 长边修正 ---
            if max(params[2], params[3]) < min_l:
                saved = params[:]
                if params[2] >= params[3]:
                    params[2] = min_l
                else:
                    params[3] = min_l
                self._clamp_boundary(params, boundary)
                if self._would_overlap(name, params, fixed, combined_base):
                    params[:] = saved
        
        return fixed
    
    @staticmethod
    def _clamp_boundary(params: list, boundary):
        """将房间约束在边界内"""
        if not boundary:
            return
        max_x = boundary.x + boundary.width
        max_y = boundary.y + boundary.height
        if params[0] + params[2] > max_x:
            params[0] = max(boundary.x, max_x - params[2])
        if params[1] + params[3] > max_y:
            params[1] = max(boundary.y, max_y - params[3])
    
    @staticmethod
    def _would_overlap(name, params, layout, base_layout):
        """检查修改后的房间是否与其他房间重叠"""
        skip_prefixes = ('采光', '南采光', '北采光', '东采光', '西采光', '黑体', '主入口')
        test = Room(name=name, x=params[0], y=params[1],
                    width=params[2], height=params[3])
        for n, p in {**base_layout, **layout}.items():
            if n == name or n == '边界' or len(p) != 4:
                continue
            if any(n.startswith(pre) for pre in skip_prefixes):
                continue
            other = Room(name=n, x=p[0], y=p[1], width=p[2], height=p[3])
            if test.overlaps(other):
                return True
        return False

    # ==================== 边界吸附 ====================

    def snap_to_boundary(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None,
        snap_threshold: int = 1800
    ) -> Dict[str, List[int]]:
        """
        将靠近边界的房间吸附到边界边缘。
        对每个房间的四条边**逐条独立**检测：如果与边界对应边的间隙 ≤ snap_threshold，
        则将该边贴紧边界；如果该条吸附导致重叠则单独回退该条，不影响其他方向。

        Args:
            layout: 生成的布局
            full_layout: 完整布局（含边界）
            snap_threshold: 吸附距离阈值(mm)，默认1800mm
        """
        fixed = {k: v.copy() for k, v in layout.items()}
        combined_base = dict(full_layout or {})
        _, boundary, _ = self.parse_layout({**combined_base, **layout})
        if not boundary:
            return fixed

        skip_names = {'边界'}
        skip_prefixes = ('采光', '南采光', '北采光', '东采光', '西采光', '黑体', '主入口')

        bx1, by1 = boundary.x, boundary.y
        bx2, by2 = boundary.x2, boundary.y2

        for name in list(fixed.keys()):
            if name in skip_names or any(name.startswith(p) for p in skip_prefixes):
                continue
            params = fixed[name]
            if len(params) != 4:
                continue

            # 逐条边独立吸附，每条吸附后立即检查重叠
            # 左边吸附
            gap_left = params[0] - bx1
            if 0 < gap_left <= snap_threshold:
                old_x = params[0]
                params[0] = bx1
                if self._would_overlap(name, params, fixed, combined_base):
                    params[0] = old_x

            # 上边吸附 (y最小边)
            gap_top = params[1] - by1
            if 0 < gap_top <= snap_threshold:
                old_y = params[1]
                params[1] = by1
                if self._would_overlap(name, params, fixed, combined_base):
                    params[1] = old_y

            # 右边吸附
            room_right = params[0] + params[2]
            gap_right = bx2 - room_right
            if 0 < gap_right <= snap_threshold:
                old_x = params[0]
                params[0] = bx2 - params[2]
                if self._would_overlap(name, params, fixed, combined_base):
                    params[0] = old_x

            # 下边吸附 (y最大边)
            room_bottom = params[1] + params[3]
            gap_bottom = by2 - room_bottom
            if 0 < gap_bottom <= snap_threshold:
                old_y = params[1]
                params[1] = by2 - params[3]
                if self._would_overlap(name, params, fixed, combined_base):
                    params[1] = old_y

        return fixed

    # ==================== 禁止相邻修复 ====================

    def fix_forbidden_adjacency(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None
    ) -> Dict[str, List[int]]:
        """
        修复禁止相邻的房间对：通过移动较小的房间使其不再相邻。
        尝试8个方向（上下左右 + 对角），递增步长，
        选择第一个既不相邻、不重叠、也不超边界的位移。
        """
        fixed = {k: v.copy() for k, v in layout.items()}
        combined_base = dict(full_layout or {})
        _, boundary, _ = self.parse_layout({**combined_base, **fixed})

        forbidden_pairs = self.adjacency_rules.get('forbidden_pairs', [])
        if not forbidden_pairs:
            return fixed

        # 解析当前所有普通房间
        all_combined = {**combined_base, **fixed}
        rooms_list, _, _ = self.parse_layout(all_combined)

        # 收集需要修复的对
        pairs_to_fix = []
        for i, r1 in enumerate(rooms_list):
            for r2 in rooms_list[i + 1:]:
                t1 = self._get_room_type(r1.name)
                t2 = self._get_room_type(r2.name)
                for pair in forbidden_pairs:
                    if (t1 == pair[0] and t2 == pair[1]) or \
                       (t1 == pair[1] and t2 == pair[0]):
                        if r1.is_adjacent(r2):
                            pairs_to_fix.append((r1, r2))
                        break

        for r1, r2 in pairs_to_fix:
            # 移动面积更小的房间；如果它不在 fixed 里（属于 existing），跳过
            move_name = r2.name if r2.area <= r1.area else r1.name
            other_name = r1.name if move_name == r2.name else r2.name

            if move_name not in fixed:
                # 尝试移动另一个
                move_name, other_name = other_name, move_name
            if move_name not in fixed:
                continue

            params = fixed[move_name]
            other_p = fixed.get(other_name) or combined_base.get(other_name)
            if not other_p or len(other_p) != 4:
                continue

            saved = params[:]
            moved = False

            # 8个方向，步长递增
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
            for step in (300, 500, 800, 1200):
                if moved:
                    break
                for dx, dy in directions:
                    test_params = [
                        params[0] + dx * step,
                        params[1] + dy * step,
                        params[2], params[3]
                    ]
                    # 边界检查
                    if boundary:
                        if test_params[0] < boundary.x or test_params[1] < boundary.y:
                            continue
                        if test_params[0] + test_params[2] > boundary.x2:
                            continue
                        if test_params[1] + test_params[3] > boundary.y2:
                            continue
                    # 重叠检查
                    if self._would_overlap(move_name, test_params, fixed, combined_base):
                        continue
                    # 相邻检查
                    test_room = Room(name=move_name, x=test_params[0],
                                    y=test_params[1], width=test_params[2],
                                    height=test_params[3])
                    other_room = Room(name=other_name, x=other_p[0],
                                     y=other_p[1], width=other_p[2],
                                     height=other_p[3])
                    if not test_room.is_adjacent(other_room):
                        params[:] = test_params
                        moved = True
                        break

            if not moved:
                params[:] = saved

        return fixed

    # ==================== 带重定位的尺寸优化 ====================

    def optimize_dimensions_with_reposition(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None
    ) -> Dict[str, List[int]]:
        """
        增强版尺寸优化：先尝试原地扩大（同 optimize_dimensions），
        若因重叠失败，再尝试在边界范围内搜索新位置放置扩大后的房间。
        """
        fixed = {k: v.copy() for k, v in layout.items()}
        combined_base = dict(full_layout or {})
        _, boundary, _ = self.parse_layout({**combined_base, **layout})

        for name in list(fixed.keys()):
            room_type = self._get_room_type(name)
            if room_type not in self.space_constraints:
                continue

            constraints = self.space_constraints[room_type]
            min_w = constraints.get('min_width', 0)
            min_l = constraints.get('min_length', 0)
            params = fixed[name]

            short_side = min(params[2], params[3])
            long_side = max(params[2], params[3])
            needs_fix = short_side < min_w or long_side < min_l
            if not needs_fix:
                continue

            # ---- 计算目标尺寸 ----
            if params[2] <= params[3]:
                target_w = max(params[2], min_w)
                target_h = max(params[3], min_l)
            else:
                target_w = max(params[2], min_l)
                target_h = max(params[3], min_w)

            # ---- 尝试1：原地扩大 ----
            saved = params[:]
            params[2], params[3] = target_w, target_h
            self._clamp_boundary(params, boundary)
            if not self._would_overlap(name, params, fixed, combined_base):
                continue  # 成功

            # ---- 尝试2：搜索附近位置放置扩大后的房间 ----
            params[:] = saved
            best = None
            origin_cx = saved[0] + saved[2] // 2
            origin_cy = saved[1] + saved[3] // 2
            best_dist = float('inf')

            # 尝试两种朝向: 原始 + 旋转90度
            orientations = [(target_w, target_h)]
            if target_w != target_h:
                orientations.append((target_h, target_w))

            for tw, th in orientations:
                for dx_step in range(-10, 11):
                    for dy_step in range(-10, 11):
                        nx = saved[0] + dx_step * 300
                        ny = saved[1] + dy_step * 300
                        test = [nx, ny, tw, th]
                        # 边界约束
                        if boundary:
                            if nx < boundary.x:
                                test[0] = boundary.x
                            if ny < boundary.y:
                                test[1] = boundary.y
                            if test[0] + tw > boundary.x2:
                                test[0] = boundary.x2 - tw
                            if test[1] + th > boundary.y2:
                                test[1] = boundary.y2 - th
                            if test[0] < boundary.x or test[1] < boundary.y:
                                continue
                        if not self._would_overlap(name, test, fixed, combined_base):
                            cx = test[0] + tw // 2
                            cy = test[1] + th // 2
                            dist = (cx - origin_cx) ** 2 + (cy - origin_cy) ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best = test[:]

            if best:
                params[:] = best

        return fixed

    # ==================== 激进后处理（组合所有修复） ====================

    def fix_living_room_position(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None
    ) -> Dict[str, List[int]]:
        """
        如果客厅距离入口过远，尝试与离入口更近的卧室交换位置。
        交换时保持各自的原始尺寸，仅交换(x,y)坐标。
        交换后检查是否产生重叠/超界，若有则回退。
        """
        fixed = {k: v.copy() for k, v in layout.items()}
        combined_base = dict(full_layout or {})
        _, boundary, _ = self.parse_layout({**combined_base, **fixed})

        # 找入口
        entry = None
        for name, params in (full_layout or {}).items():
            if "入口" in name and len(params) == 4:
                entry = Room(name=name, x=params[0], y=params[1],
                             width=params[2], height=params[3])
                break
        if not entry:
            return fixed

        # 找客厅
        living_name = None
        for name in fixed:
            if "客厅" in name:
                living_name = name
                break
        if not living_name:
            return fixed

        lp = fixed[living_name]
        living_room = Room(name=living_name, x=lp[0], y=lp[1],
                           width=lp[2], height=lp[3])
        living_dist = self._calc_dist(entry.center, living_room.center)

        if living_dist <= 5000:
            return fixed  # 已经足够近

        # 找所有卧室，并筛选比客厅更近入口的
        swap_candidates = []
        for name in fixed:
            if "卧" not in name:
                continue
            bp = fixed[name]
            br = Room(name=name, x=bp[0], y=bp[1], width=bp[2], height=bp[3])
            d = self._calc_dist(entry.center, br.center)
            if d < living_dist - 1000:  # 至少近 1000mm 才值得交换
                swap_candidates.append((name, d))

        # 按距离入口从近到远排序
        swap_candidates.sort(key=lambda x: x[1])

        for swap_name, _ in swap_candidates:
            saved_living = fixed[living_name][:]
            saved_swap = fixed[swap_name][:]

            # 交换坐标，保留各自尺寸
            fixed[living_name][0] = saved_swap[0]
            fixed[living_name][1] = saved_swap[1]
            fixed[swap_name][0] = saved_living[0]
            fixed[swap_name][1] = saved_living[1]

            # 检查边界
            ok = True
            if boundary:
                for n in (living_name, swap_name):
                    p = fixed[n]
                    if p[0] < boundary.x or p[1] < boundary.y:
                        ok = False
                    if p[0] + p[2] > boundary.x2 or p[1] + p[3] > boundary.y2:
                        ok = False

            # 检查重叠
            if ok:
                if self._would_overlap(living_name, fixed[living_name],
                                       fixed, combined_base):
                    ok = False
            if ok:
                if self._would_overlap(swap_name, fixed[swap_name],
                                       fixed, combined_base):
                    ok = False

            if ok:
                return fixed  # 交换成功
            else:
                # 回退
                fixed[living_name][:] = saved_living
                fixed[swap_name][:] = saved_swap

        return fixed

    @staticmethod
    def _calc_dist(c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def aggressive_post_process(
        self,
        layout: Dict[str, List[int]],
        full_layout: Dict[str, List[int]] = None,
        max_passes: int = 3
    ) -> Dict[str, List[int]]:
        """
        激进后处理：反复执行 修复 → 尺寸优化 → 相邻修复 → 边界吸附，
        直到没有改进或达到最大遍历次数。

        调用顺序（每遍）：
        1. validate_and_fix  — 解决重叠、超界等硬性问题
        2. optimize_dimensions_with_reposition — 扩大不足房间（含重定位）
        3. fix_living_room_position — 客厅远离入口时与卧室交换
        4. fix_forbidden_adjacency — 拉开禁止相邻的房间
        5. snap_to_boundary — 吸附到边界
        """
        current = {k: v.copy() for k, v in layout.items()}

        for pass_idx in range(max_passes):
            before = {k: v[:] for k, v in current.items()}

            # 1. 硬性规则修复
            fix_result = self.validate_and_fix(current, full_layout)
            if fix_result.fixed_layout:
                current = fix_result.fixed_layout

            # 2. 带重定位的尺寸优化
            current = self.optimize_dimensions_with_reposition(current, full_layout)

            # 3. 客厅位置修正（与卧室交换）
            current = self.fix_living_room_position(current, full_layout)

            # 4. 禁止相邻修复
            current = self.fix_forbidden_adjacency(current, full_layout)

            # 5. 边界吸附
            current = self.snap_to_boundary(current, full_layout)

            # 检查是否有变化，无变化则提前终止
            if all(current.get(k) == before.get(k) for k in set(current) | set(before)):
                break

        return current

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

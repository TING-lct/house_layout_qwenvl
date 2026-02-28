"""
公共模块 — 提取所有核心模块共享的常量、工具函数和基础数据结构，
消除 evaluator / rule_engine / generator / metrics / embedder 之间的代码重复。
"""

from __future__ import annotations

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ==================== 常量定义 ====================

# 不参与布局计算的特殊区域名称前缀
BOUNDARY_NAME = "边界"

# 方向采光（在边界外，不影响房间布局重叠检测）
DIRECTIONAL_LIGHTING_PREFIXES = ("南采光", "北采光", "东采光", "西采光")

# 基础设施前缀（在边界内，需要与房间做重叠检测）
INFRA_PREFIXES = ("采光", "黑体", "主入口")

# 所有需要在解析时跳过的特殊名称前缀（用于提取普通房间）
SKIP_ROOM_PREFIXES = ("采光", "南采光", "北采光", "东采光", "西采光", "黑体", "主入口")

# 房间类型映射（精确匹配表 + 优先级列表），全局唯一定义
ROOM_TYPE_MAPPINGS: Dict[str, List[str]] = {
    "主卧": ["主卧", "主卧室"],
    "卧室": [
        "卧室", "卧室1", "卧室2", "卧室3", "卧室4", "卧室5",
        "次卧", "次卧1", "次卧2", "客卧", "书房卧室",
    ],
    "客厅": ["客厅", "起居室", "客餐厅"],
    "厨房": ["厨房", "厨房1", "中厨", "西厨", "开放厨房"],
    "卫生间": ["卫生间", "卫生间1", "卫生间2", "公卫", "次卫", "公共卫生间"],
    "主卫": ["主卫", "主卫生间"],
    "餐厅": ["餐厅", "餐厅1", "饭厅"],
    "储藏": ["储藏", "储藏室", "储物间", "杂物间", "收纳间"],
    "阳台": ["阳台", "阳台1", "阳台2", "生活阳台", "景观阳台"],
}

# 模糊匹配优先级（更具体的类型排在前面，防止 "主卧" 匹配到 "卧室"）
ROOM_TYPE_PRIORITY = [
    "主卧", "主卫", "卫生间", "卧室", "客厅", "厨房", "餐厅", "储藏", "阳台"
]

# 预构建的精确查找字典，O(1) 查找
_EXACT_ROOM_TYPE_CACHE: Dict[str, str] = {}
for _rt, _names in ROOM_TYPE_MAPPINGS.items():
    for _n in _names:
        _EXACT_ROOM_TYPE_CACHE[_n] = _rt


# ==================== 房间类型解析 ====================

def get_room_type(room_name: str) -> str:
    """
    从房间名获取房间类型。全局统一的房间类型解析函数。

    查找策略:
    1. O(1) 精确匹配
    2. 按优先级模糊匹配（子串包含）
    3. 无匹配则返回原名

    Args:
        room_name: 房间名称，如 "卧室1", "主卧", "厨房"

    Returns:
        房间类型，如 "卧室", "主卧", "厨房"
    """
    # 1. 精确匹配
    if room_name in _EXACT_ROOM_TYPE_CACHE:
        return _EXACT_ROOM_TYPE_CACHE[room_name]

    # 2. 优先级模糊匹配
    for rt in ROOM_TYPE_PRIORITY:
        if rt in room_name:
            return rt

    return room_name


def is_normal_room(name: str) -> bool:
    """判断是否为普通房间（非边界、非采光、非基础设施）"""
    if name == BOUNDARY_NAME:
        return False
    return not any(name.startswith(p) for p in SKIP_ROOM_PREFIXES)


def is_directional_lighting(name: str) -> bool:
    """判断是否为方向采光区域（在边界外）"""
    return any(name.startswith(p) for p in DIRECTIONAL_LIGHTING_PREFIXES)


def is_infrastructure(name: str) -> bool:
    """判断是否为基础设施（采光区、黑体、主入口，在边界内）"""
    if is_directional_lighting(name):
        return False
    return any(name.startswith(p) for p in INFRA_PREFIXES)


# ==================== Room 数据结构 ====================

@dataclass
class Room:
    """
    房间数据结构，提供几何计算方法。

    属性：
        name: 房间名称
        x, y: 左下角坐标 (mm)
        width, height: 宽度和高度 (mm)
    """
    name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """长宽比 (≥1.0)"""
        if self.short_side <= 0:
            return float('inf')
        return self.long_side / self.short_side

    def overlaps(self, other: 'Room') -> bool:
        """检查是否与另一个房间重叠（边界相切不算重叠）"""
        return not (
            self.x2 <= other.x or
            other.x2 <= self.x or
            self.y2 <= other.y or
            other.y2 <= self.y
        )

    def overlap_area(self, other: 'Room') -> int:
        """计算与另一个房间的重叠面积"""
        ox = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        oy = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        return ox * oy

    def is_adjacent(self, other: 'Room', tolerance: int = 100) -> bool:
        """检查是否与另一个房间相邻（边缘间距 ≤ tolerance）"""
        h_adjacent = (
            abs(self.x2 - other.x) <= tolerance or
            abs(other.x2 - self.x) <= tolerance
        ) and not (self.y2 <= other.y or other.y2 <= self.y)

        v_adjacent = (
            abs(self.y2 - other.y) <= tolerance or
            abs(other.y2 - self.y) <= tolerance
        ) and not (self.x2 <= other.x or other.x2 <= self.x)

        return h_adjacent or v_adjacent

    def is_within(self, boundary: 'Room') -> bool:
        """检查是否完全在边界内"""
        return (
            self.x >= boundary.x and
            self.y >= boundary.y and
            self.x2 <= boundary.x2 and
            self.y2 <= boundary.y2
        )

    def distance_to(self, other: 'Room') -> float:
        """计算两个房间中心点之间的距离"""
        c1, c2 = self.center, other.center
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    @classmethod
    def from_params(cls, name: str, params: List[int]) -> Optional['Room']:
        """从参数列表创建 Room，无效则返回 None"""
        if not isinstance(params, (list, tuple)) or len(params) != 4:
            return None
        try:
            return cls(name=name, x=int(params[0]), y=int(params[1]),
                       width=int(params[2]), height=int(params[3]))
        except (ValueError, TypeError):
            return None


# ==================== 布局解析 ====================

@dataclass
class ParsedLayout:
    """解析后的布局数据"""
    rooms: List[Room]                     # 普通房间
    boundary: Optional[Room]              # 边界
    infra: List[Room]                     # 基础设施（采光区等）
    all_rooms: Dict[str, Room]            # 所有房间（含边界、基础设施）
    directional_lighting: List[Room]      # 方向采光区域


def parse_layout(layout_dict: Dict[str, List[int]]) -> ParsedLayout:
    """
    统一的布局解析函数，将字典解析为结构化数据。

    Args:
        layout_dict: 布局字典 {房间名: [x, y, width, height]}

    Returns:
        ParsedLayout: 结构化布局数据
    """
    rooms: List[Room] = []
    boundary: Optional[Room] = None
    infra: List[Room] = []
    dir_lighting: List[Room] = []
    all_rooms: Dict[str, Room] = {}

    for name, params in layout_dict.items():
        room = Room.from_params(name, params)
        if room is None:
            continue

        all_rooms[name] = room

        if name == BOUNDARY_NAME:
            boundary = room
        elif is_directional_lighting(name):
            dir_lighting.append(room)
        elif is_infrastructure(name):
            infra.append(room)
        elif is_normal_room(name):
            rooms.append(room)

    return ParsedLayout(
        rooms=rooms,
        boundary=boundary,
        infra=infra,
        all_rooms=all_rooms,
        directional_lighting=dir_lighting,
    )


# ==================== JSON 解析工具 ====================

def extract_json_from_text(text: str) -> str:
    """
    从 LLM 输出中提取 JSON 字符串。

    查找策略:
    1. ```json ... ``` 代码块
    2. ``` ... ``` 代码块
    3. 第一个 { ... } 块（括号配对）
    4. 原文
    """
    # 1. ```json 块
    if "```json" in text:
        parts = text.split("```json", 1)
        if len(parts) > 1:
            end = parts[1].find("```")
            return parts[1][:end].strip() if end != -1 else parts[1].strip()

    # 2. ``` 块
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()

    # 3. 括号配对提取 { ... }
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        # 未闭合，补 }
        return text[start:] + "}"

    return text.strip()


def clean_json_str(s: str) -> str:
    """
    清理 LLM 常见的 JSON 格式错误。
    """
    # 移除行注释
    s = re.sub(r'//[^\n]*', '', s)
    # 移除末尾多余逗号
    s = re.sub(r',\s*([}\]])', r'\1', s)
    # 单引号键名 → 双引号
    s = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:", r' "\1":', s)
    # 移除省略号
    s = re.sub(r'\.{3,}', '', s)
    # 确保闭合
    s += '}' * max(0, s.count('{') - s.count('}'))
    s += ']' * max(0, s.count('[') - s.count(']'))
    return s


def parse_layout_json(text: str) -> Dict[str, List[int]]:
    """
    从 LLM 输出文本解析布局字典。多策略容错。

    Returns:
        解析后的布局字典，失败返回空字典
    """
    json_str = extract_json_from_text(text)
    if not json_str:
        logger.warning("parse_layout_json: 未找到 JSON 内容")
        return {}

    # 尝试直接解析
    try:
        return _validate_layout_dict(json.loads(json_str))
    except json.JSONDecodeError:
        pass

    # 清理后重试
    try:
        return _validate_layout_dict(json.loads(clean_json_str(json_str)))
    except json.JSONDecodeError:
        pass

    # 正则兜底
    layout = {}
    pattern = r'"([^"]+)"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    for m in re.finditer(pattern, text):
        name = m.group(1)
        vals = [int(m.group(i)) for i in range(2, 6)]
        layout[name] = vals

    if layout:
        return layout

    logger.warning("parse_layout_json: 所有解析方法均失败")
    return {}


def _validate_layout_dict(data: Any) -> Dict[str, List[int]]:
    """验证并提取合法的布局字典"""
    if not isinstance(data, dict):
        return {}
    layout = {}
    for k, v in data.items():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            try:
                layout[k] = [int(x) for x in v]
            except (ValueError, TypeError):
                continue
    return layout


# ==================== 空间碰撞检测优化 ====================

class SpatialIndex:
    """
    简单的空间索引，用于加速大量房间的重叠检测。

    使用网格划分，将房间按所在网格单元索引，
    只检测同一网格或相邻网格中的房间。
    """

    def __init__(self, cell_size: int = 3000):
        self.cell_size = cell_size
        self._grid: Dict[Tuple[int, int], List[Room]] = {}
        self._rooms: Dict[str, Room] = {}

    def clear(self):
        self._grid.clear()
        self._rooms.clear()

    def add(self, room: Room):
        self._rooms[room.name] = room
        for cell in self._get_cells(room):
            self._grid.setdefault(cell, []).append(room)

    def remove(self, name: str):
        room = self._rooms.pop(name, None)
        if room:
            for cell in self._get_cells(room):
                cell_list = self._grid.get(cell, [])
                self._grid[cell] = [r for r in cell_list if r.name != name]

    def update(self, room: Room):
        self.remove(room.name)
        self.add(room)

    def query_overlaps(self, room: Room, exclude_name: Optional[str] = None) -> List[Room]:
        """查询与给定房间重叠的所有房间"""
        candidates = set()
        for cell in self._get_cells(room):
            for r in self._grid.get(cell, []):
                if r.name != room.name and r.name != exclude_name:
                    candidates.add(r.name)

        overlaps = []
        for name in candidates:
            other = self._rooms.get(name)
            if other and room.overlaps(other):
                overlaps.append(other)
        return overlaps

    def would_overlap(self, name: str, params: List[int],
                      skip_names: set = None) -> bool:
        """检查假设的位置是否与现有房间重叠"""
        test = Room(name=name, x=params[0], y=params[1],
                    width=params[2], height=params[3])
        skip = skip_names or set()
        for cell in self._get_cells(test):
            for r in self._grid.get(cell, []):
                if r.name == name or r.name in skip:
                    continue
                if test.overlaps(r):
                    return True
        return False

    def build_from_layout(self, layout_dict: Dict[str, List[int]],
                          skip_directional_lighting: bool = True):
        """从布局字典构建索引"""
        self.clear()
        for name, params in layout_dict.items():
            if name == BOUNDARY_NAME:
                continue
            if skip_directional_lighting and is_directional_lighting(name):
                continue
            room = Room.from_params(name, params)
            if room:
                self.add(room)

    def _get_cells(self, room: Room) -> List[Tuple[int, int]]:
        """获取房间覆盖的所有网格单元"""
        cs = self.cell_size
        x1 = room.x // cs
        y1 = room.y // cs
        x2 = (room.x2 - 1) // cs
        y2 = (room.y2 - 1) // cs
        cells = []
        for gx in range(x1, x2 + 1):
            for gy in range(y1, y2 + 1):
                cells.append((gx, gy))
        return cells


# ==================== 日志辅助 ====================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """创建带格式的 logger"""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        ))
        log.addHandler(handler)
    log.setLevel(level)
    return log

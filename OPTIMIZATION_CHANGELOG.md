# 代码优化变更日志

## 概述

本次优化针对 `house_layout_qwenvl` 项目进行了全面的代码重构，核心目标是 **消除重复代码、统一日志系统、提升代码可维护性**。

---

## 新增文件

### `core/common.py` — 公共模块（~480行）

将散布在 5+ 个文件中的重复代码抽取为统一模块：

| 组件 | 说明 | 替代的重复代码 |
|------|------|----------------|
| **常量定义** | `BOUNDARY_NAME`, `SKIP_ROOM_PREFIXES`, `DIRECTIONAL_LIGHTING_PREFIXES`, `INFRA_PREFIXES` | 10+ 处硬编码的前缀元组 |
| **`ROOM_TYPE_MAPPINGS`** | 全局唯一的房间类型映射表 | 5 份完全相同的映射字典 |
| **`get_room_type()`** | O(1) 精确匹配 + 优先级模糊匹配 | 5 个 `_get_room_type()` 方法 |
| **`is_normal_room()` / `is_infrastructure()` / `is_directional_lighting()`** | 语义明确的判断函数 | 多处 `name.startswith((...))` |
| **`Room` dataclass** | 增强版房间数据结构，含 `is_within()`, `distance_to()`, `overlap_area()`, `short_side`, `aspect_ratio`, `from_params()` | evaluator.py 中的 Room 类 |
| **`ParsedLayout` + `parse_layout()`** | 统一的布局解析，返回结构化数据（rooms / boundary / infra / directional_lighting） | evaluator.py 和 rule_engine.py 各自的 parse_layout |
| **`extract_json_from_text()` + `clean_json_str()` + `parse_layout_json()`** | 多策略容错 JSON 解析 | generator.py 和 layout_predictor.py 的重复解析代码 |
| **`SpatialIndex`** | 基于网格的空间索引，加速碰撞检测 | 为未来性能优化预留 |
| **`setup_logger()`** | 统一日志配置 | 混乱的 print/logging 混用 |

---

## 修改文件清单

### `core/evaluator.py`
- ❌ 删除: 内嵌的 `Room` 类（~50行）、`parse_layout` 手动实现、`_get_room_type`（~25行）、手动距离计算
- ✅ 改用: `from .common import Room, get_room_type, parse_layout, ...`
- ✅ `_is_within_boundary` → `Room.is_within()`（静态方法）
- ✅ `_calculate_distance` → `Room.distance_to()`（静态方法）
- ✅ 基础设施重叠检查 → 使用 `ParsedLayout.infra` 列表
- ✅ 空间覆盖率计算 → 使用 `ParsedLayout.infra` 求和

### `core/rule_engine.py`
- ❌ 删除: `from .evaluator import Room, LayoutEvaluator`
- ✅ 改用: `from .common import Room, get_room_type, is_normal_room, is_directional_lighting, ...`
- ❌ 删除: `_get_room_type`（~25行）
- ✅ `_would_overlap` 中硬编码 `skip_prefixes` → `is_directional_lighting()`
- ✅ `expand_rooms_to_fill` 中房间过滤 → `is_normal_room()`
- ✅ `snap_to_boundary` 中跳过判断 → `is_normal_room()`
- ✅ `no_room_overlap` 中基础设施检测 → `is_directional_lighting()` + `is_infrastructure()`
- ✅ `within_boundary` → `Room.is_within()`

### `core/generator.py`
- ❌ 删除: 60+ 行的 `_parse_output` 方法（JSON提取+正则兜底）
- ✅ 改用: `parse_layout_json()` 一行替代
- ✅ 添加 `import logging` 和 logger

### `core/llm_evaluator.py`
- ❌ 删除: `_parse_response` 中的手动 JSON 提取逻辑
- ✅ 改用: `extract_json_from_text()` + `json.loads()`
- ✅ 6 处 `print()` → `logger.info/warning()`

### `core/optimizer.py`
- ✅ 无重复代码，无需修改

### `layout_predictor.py`
- ❌ 删除: `parse_output`（~40行）、`_extract_json_str`（~20行）、`_clean_json_str`（~15行）
- ✅ 改用: `parse_layout_json()`, `extract_json_from_text()`, `clean_json_str()`
- ✅ 20+ 处 `print()` → `logger.info/warning()`

### `pipeline.py`
- ✅ 10+ 处 `print()` → `logger.info/warning/error()`
- ✅ 添加 `import logging` 和 logger

### `rag/embedder.py`
- ❌ 删除: `_get_room_type`（~25行）
- ✅ 改用: `from core.common import get_room_type, is_normal_room`
- ✅ 房间过滤 → `is_normal_room()`

### `utils/metrics.py`
- ❌ 删除: `_get_room_type`（~15行，且映射不完整）
- ✅ 改用: `from core.common import get_room_type`

### `core/__init__.py`
- ✅ 新增 `common` 模块所有公共符号的导出
- ✅ `Room` 从 `evaluator` 改为从 `common` 导出

---

## 量化改进

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| `_get_room_type` 副本数 | **5** (evaluator, rule_engine, embedder, metrics, layout_predictor 各有映射) | **1** (common.py) |
| JSON 解析重复代码 | **2** 套（generator + layout_predictor） | **1** 套（common.py） |
| `parse_layout` 重复实现 | **2** 个（evaluator + rule_engine） | **1** 个（common.py） |
| 硬编码前缀元组 | **10+** 处 | **0**（全用常量/函数） |
| `print()` 调用数 | **50+** | **0**（全部 logger） |
| Room 类定义 | **2** 个（evaluator.py + common.py 都有） | **1** 个（common.py） |

---

## 架构影响

```
优化前依赖关系:
  evaluator.py  ← rule_engine.py (import Room, LayoutEvaluator)
  evaluator.py  — 独立的 Room 类 + parse_layout + _get_room_type
  rule_engine.py — 独立的 parse_layout + _get_room_type + _would_overlap
  generator.py  — 独立的 _parse_output
  layout_predictor.py — 独立的 parse_output + _extract_json_str + _clean_json_str

优化后依赖关系:
  common.py  ← evaluator.py (Room, get_room_type, parse_layout, ...)
  common.py  ← rule_engine.py (Room, get_room_type, is_normal_room, ...)
  common.py  ← generator.py (parse_layout_json, extract_json_from_text)
  common.py  ← llm_evaluator.py (extract_json_from_text)
  common.py  ← layout_predictor.py (parse_layout_json, ...)
  common.py  ← embedder.py (get_room_type, is_normal_room)
  common.py  ← metrics.py (get_room_type)
```

所有模块共享同一数据源，修改房间类型映射、添加新房间类型只需改 **一个文件**。

"""
可视化模块
绘制户型布局图
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import numpy as np


# 房间颜色映射
ROOM_COLORS = {
    '边界': '#E0E0E0',
    '客厅': '#FFD700',      # 金色
    '卧室': '#87CEEB',      # 天蓝色
    '卧室1': '#87CEEB',
    '卧室2': '#5F9EA0',     # 青色
    '卧室3': '#4682B4',     # 钢蓝色
    '卧室4': '#6495ED',     # 矢车菊蓝
    '主卧': '#4169E1',      # 皇家蓝
    '厨房': '#FFA500',      # 橙色
    '卫生间': '#98FB98',    # 淡绿色
    '主卫': '#90EE90',      # 浅绿色
    '餐厅': '#DEB887',      # 棕褐色
    '阳台': '#F0E68C',      # 卡其色
    '储藏': '#D3D3D3',      # 浅灰色
    '采光': '#FFFACD',      # 柠檬绸色
    '采光1': '#FFFACD',
    '采光2': '#FFFACD',
    '南采光': '#FFFACD',
    '北采光': '#FFFACD',
    '西采光': '#FFFACD',
    '东采光': '#FFFACD',
    '黑体': '#404040',      # 深灰色
    '黑体1': '#404040',
    '黑体2': '#404040',
    '主入口': '#FF6347',    # 番茄红
}

DEFAULT_COLOR = '#C0C0C0'  # 默认灰色


class LayoutVisualizer:
    """户型布局可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        初始化可视化器
        
        Args:
            figsize: 图片大小
        """
        self.figsize = figsize
        
        # 设置中文字体（跨平台兼容）
        self.font = self._setup_cjk_font()
    
    @staticmethod
    def _setup_cjk_font() -> FontProperties:
        """设置中文字体，同时配置 matplotlib 全局 rcParams
        
        查找顺序：
        1. 项目内嵌字体 (fonts/SimHei.ttf) —— 离线环境下也可用
        2. 系统已安装的 CJK 字体
        3. fc-list 动态搜索 (Linux)
        4. matplotlib 已注册的 CJK 字体名称
        5. 自动下载 Noto Sans SC（需要网络）
        """
        import os, platform, subprocess
        from matplotlib.font_manager import FontProperties, fontManager
        
        def _register_and_return(font_path):
            """注册字体到 matplotlib 全局并返回 FontProperties"""
            fp = FontProperties(fname=font_path)
            font_name = fp.get_name()
            
            # 注册到 fontManager
            if hasattr(fontManager, 'addfont'):
                fontManager.addfont(font_path)
            else:
                from matplotlib.font_manager import FontEntry
                fontManager.ttflist.append(
                    FontEntry(fname=font_path, name=font_name)
                )
            
            # 设置全局 rcParams —— 这是让 tight_layout/savefig 不 fallback 的关键
            plt.rcParams['font.family'] = 'sans-serif'
            sans_list = list(plt.rcParams.get('font.sans-serif', []))
            if font_name not in sans_list:
                sans_list.insert(0, font_name)
            plt.rcParams['font.sans-serif'] = sans_list
            plt.rcParams['axes.unicode_minus'] = False
            
            print(f"CJK 字体已加载: {font_name} ({font_path})")
            return fp
        
        # ---------- 0. 项目内嵌字体（最高优先级） ----------
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_this_dir)
        bundled_font = os.path.join(_project_root, "fonts", "SimHei.ttf")
        
        # 按优先级列出候选字体文件
        candidates = []
        
        # 始终最先尝试项目自带字体
        if os.path.isfile(bundled_font):
            candidates.append(bundled_font)
        
        if platform.system() == "Windows":
            win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
            candidates += [
                os.path.join(win_fonts, "simhei.ttf"),
                os.path.join(win_fonts, "msyh.ttc"),
                os.path.join(win_fonts, "simsun.ttc"),
            ]
        else:
            # Linux 常见中文字体路径
            candidates += [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/SimHei.ttf",
            ]
            # 动态搜索 fc-list 找到的 CJK 字体
            try:
                result = subprocess.run(
                    ["fc-list", ":lang=zh", "--format=%{file}\n"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        path = line.strip()
                        if path and os.path.isfile(path):
                            candidates.append(path)
            except Exception:
                pass
        
        # 尝试找到可用的字体文件
        for path in candidates:
            if os.path.isfile(path):
                try:
                    return _register_and_return(path)
                except Exception as e:
                    print(f"加载字体 {path} 失败: {e}")
        
        # 尝试 matplotlib 已知的 CJK 字体名称
        for family in ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
                        'Noto Sans CJK SC', 'Noto Sans SC',
                        'SimHei', 'Microsoft YaHei']:
            try:
                fp = FontProperties(family=family)
                from matplotlib.font_manager import findfont
                real_path = findfont(fp)
                if real_path and 'DejaVu' not in real_path:
                    return _register_and_return(real_path)
            except Exception:
                continue
        
        # 最后手段：自动下载 Noto Sans SC 字体到本地缓存
        try:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "cjk_fonts")
            os.makedirs(cache_dir, exist_ok=True)
            local_font = os.path.join(cache_dir, "NotoSansSC-Regular.otf")
            
            if not os.path.isfile(local_font):
                import urllib.request
                url = ("https://github.com/notofonts/noto-cjk/raw/main/"
                       "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf")
                print(f"正在下载 CJK 字体到 {local_font} ...")
                urllib.request.urlretrieve(url, local_font)
                print("CJK 字体下载完成")
            
            if os.path.isfile(local_font):
                return _register_and_return(local_font)
        except Exception as e:
            print(f"自动下载 CJK 字体失败: {e}")
            print("提示: 可手动运行 apt-get install -y fonts-wqy-zenhei "
                  "然后删除 ~/.cache/matplotlib/ 重试")
        
        print("⚠️ 未找到任何 CJK 字体，中文将显示为方块")
        return FontProperties()
    
    def visualize(
        self,
        layout: Dict[str, List[int]],
        title: str = "户型布局",
        show_labels: bool = True,
        show_dimensions: bool = True,
        highlight_rooms: List[str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        可视化户型布局
        
        Args:
            layout: 布局字典
            title: 标题
            show_labels: 是否显示房间名称
            show_dimensions: 是否显示尺寸
            highlight_rooms: 需要高亮的房间
            save_path: 保存路径
            
        Returns:
            plt.Figure: matplotlib图形对象
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # 获取边界
        boundary = layout.get('边界', [0, 0, 10000, 10000])
        b_x, b_y, b_w, b_h = boundary
        
        # 设置坐标轴范围（留出边距）
        margin = max(b_w, b_h) * 0.1
        ax.set_xlim(b_x - margin, b_x + b_w + margin)
        ax.set_ylim(b_y - margin, b_y + b_h + margin)
        
        # 绘制每个房间
        for name, params in layout.items():
            if len(params) != 4:
                continue
            
            x, y, w, h = params
            
            # 获取颜色
            color = self._get_color(name)
            
            # 高亮处理
            alpha = 0.8
            edgecolor = 'black'
            linewidth = 1
            
            if highlight_rooms and name in highlight_rooms:
                edgecolor = 'red'
                linewidth = 3
            
            # 绘制矩形
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=color,
                alpha=alpha
            )
            ax.add_patch(rect)
            
            # 添加房间名称
            if show_labels and not name.startswith(('黑体', '采光', '南采光', '北采光', '西采光', '东采光')):
                center_x = x + w / 2
                center_y = y + h / 2
                
                # 计算面积
                area = w * h / 1000000  # 平方米
                
                if show_dimensions:
                    label = f"{name}\n{area:.1f}m2"  # 使用m2代替m²避免字体问题
                else:
                    label = name
                
                ax.text(
                    center_x, center_y, label,
                    ha='center', va='center',
                    fontsize=8,
                    fontproperties=self.font
                )
        
        # 设置标题
        ax.set_title(title, fontsize=14, fontproperties=self.font)
        
        # 设置坐标轴
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _get_color(self, room_name: str) -> str:
        """获取房间颜色"""
        # 直接匹配
        if room_name in ROOM_COLORS:
            return ROOM_COLORS[room_name]
        
        # 模糊匹配
        for key in ROOM_COLORS:
            if key in room_name:
                return ROOM_COLORS[key]
        
        return DEFAULT_COLOR
    
    def compare_layouts(
        self,
        layouts: List[Dict[str, List[int]]],
        titles: List[str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        比较多个布局
        
        Args:
            layouts: 布局列表
            titles: 标题列表
            save_path: 保存路径
            
        Returns:
            plt.Figure: matplotlib图形对象
        """
        n = len(layouts)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (layout, ax) in enumerate(zip(layouts, axes)):
            title = titles[i] if titles and i < len(titles) else f"布局 {i+1}"
            self._draw_layout_on_ax(ax, layout, title)
        
        # 隐藏多余的子图
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _draw_layout_on_ax(
        self,
        ax: plt.Axes,
        layout: Dict[str, List[int]],
        title: str
    ):
        """在指定轴上绘制布局"""
        # 获取边界
        boundary = layout.get('边界', [0, 0, 10000, 10000])
        b_x, b_y, b_w, b_h = boundary
        
        # 设置坐标轴范围
        margin = max(b_w, b_h) * 0.05
        ax.set_xlim(b_x - margin, b_x + b_w + margin)
        ax.set_ylim(b_y - margin, b_y + b_h + margin)
        
        # 绘制每个房间
        for name, params in layout.items():
            if len(params) != 4:
                continue
            
            x, y, w, h = params
            color = self._get_color(name)
            
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # 添加简短标签
            if not name.startswith(('黑体', '采光', '南采光', '北采光', '西采光', '东采光', '边界')):
                ax.text(
                    x + w/2, y + h/2, name,
                    ha='center', va='center',
                    fontsize=6,
                    fontproperties=self.font
                )
        
        ax.set_title(title, fontsize=10, fontproperties=self.font)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)


def visualize_evaluation_result(
    layout: Dict[str, List[int]],
    evaluation_result: Any,
    save_path: str = None
) -> plt.Figure:
    """
    可视化评估结果
    
    Args:
        layout: 布局字典
        evaluation_result: 评估结果
        save_path: 保存路径
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左侧：布局图
    visualizer = LayoutVisualizer()
    
    # 获取边界
    boundary = layout.get('边界', [0, 0, 10000, 10000])
    b_x, b_y, b_w, b_h = boundary
    
    margin = max(b_w, b_h) * 0.1
    ax1.set_xlim(b_x - margin, b_x + b_w + margin)
    ax1.set_ylim(b_y - margin, b_y + b_h + margin)
    
    for name, params in layout.items():
        if len(params) != 4:
            continue
        
        x, y, w, h = params
        color = visualizer._get_color(name)
        
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax1.add_patch(rect)
        
        if not name.startswith(('黑体', '采光', '南采光', '北采光', '西采光', '东采光')):
            ax1.text(
                x + w/2, y + h/2, name,
                ha='center', va='center',
                fontsize=8,
                fontproperties=visualizer.font
            )
    
    ax1.set_title('户型布局', fontsize=12, fontproperties=visualizer.font)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 右侧：评分雷达图
    if hasattr(evaluation_result, 'dimension_scores'):
        categories = list(evaluation_result.dimension_scores.keys())
        values = list(evaluation_result.dimension_scores.values())
        
        # 闭合雷达图
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values, 'o-', linewidth=2)
        ax2.fill(angles, values, alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontproperties=visualizer.font)
        ax2.set_ylim(0, 100)
        ax2.set_title(
            f'评估得分: {evaluation_result.total_score:.1f}',
            fontsize=12,
            fontproperties=visualizer.font
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

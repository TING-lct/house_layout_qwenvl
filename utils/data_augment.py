"""
数据增强模块
实现户型布局的数据增强
"""

import json
import copy
from typing import Dict, List, Any, Tuple


class LayoutDataAugmentation:
    """户型布局数据增强器"""
    
    def __init__(self):
        pass
    
    def augment(
        self, 
        layout: Dict[str, List[int]],
        include_original: bool = True
    ) -> List[Dict[str, List[int]]]:
        """
        对布局进行数据增强
        
        Args:
            layout: 原始布局
            include_original: 是否包含原始布局
            
        Returns:
            List[Dict]: 增强后的布局列表
        """
        augmented = []
        
        if include_original:
            augmented.append(copy.deepcopy(layout))
        
        # 1. 水平镜像
        augmented.append(self.mirror_horizontal(layout))
        
        # 2. 垂直镜像
        augmented.append(self.mirror_vertical(layout))
        
        # 3. 旋转（90度、180度、270度）
        for angle in [90, 180, 270]:
            augmented.append(self.rotate(layout, angle))
        
        # 4. 缩放
        for scale in [0.9, 1.1]:
            augmented.append(self.scale(layout, scale))
        
        return augmented
    
    def mirror_horizontal(self, layout: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        水平镜像翻转
        
        Args:
            layout: 原始布局
            
        Returns:
            Dict: 翻转后的布局
        """
        mirrored = {}
        
        # 获取边界
        boundary = layout.get('边界', [0, 0, 10000, 10000])
        boundary_width = boundary[2]
        
        for name, params in layout.items():
            if len(params) != 4:
                mirrored[name] = params
                continue
            
            x, y, w, h = params
            
            # 水平翻转：新x = 边界宽度 - (x + w)
            new_x = boundary_width - (x + w)
            
            mirrored[name] = [new_x, y, w, h]
        
        return mirrored
    
    def mirror_vertical(self, layout: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        垂直镜像翻转
        
        Args:
            layout: 原始布局
            
        Returns:
            Dict: 翻转后的布局
        """
        mirrored = {}
        
        # 获取边界
        boundary = layout.get('边界', [0, 0, 10000, 10000])
        boundary_height = boundary[3]
        
        for name, params in layout.items():
            if len(params) != 4:
                mirrored[name] = params
                continue
            
            x, y, w, h = params
            
            # 垂直翻转：新y = 边界高度 - (y + h)
            new_y = boundary_height - (y + h)
            
            mirrored[name] = [x, new_y, w, h]
        
        return mirrored
    
    def rotate(self, layout: Dict[str, List[int]], angle: int) -> Dict[str, List[int]]:
        """
        旋转布局
        
        Args:
            layout: 原始布局
            angle: 旋转角度（90, 180, 270）
            
        Returns:
            Dict: 旋转后的布局
        """
        if angle not in [90, 180, 270]:
            raise ValueError("角度必须为90, 180或270")
        
        rotated = {}
        
        # 获取边界
        boundary = layout.get('边界', [0, 0, 10000, 10000])
        boundary_width = boundary[2]
        boundary_height = boundary[3]
        
        for name, params in layout.items():
            if len(params) != 4:
                rotated[name] = params
                continue
            
            x, y, w, h = params
            
            if angle == 90:
                # 90度顺时针旋转
                new_x = boundary_height - (y + h)
                new_y = x
                new_w = h
                new_h = w
            elif angle == 180:
                # 180度旋转
                new_x = boundary_width - (x + w)
                new_y = boundary_height - (y + h)
                new_w = w
                new_h = h
            else:  # 270
                # 270度顺时针旋转（或90度逆时针）
                new_x = y
                new_y = boundary_width - (x + w)
                new_w = h
                new_h = w
            
            rotated[name] = [int(new_x), int(new_y), int(new_w), int(new_h)]
        
        # 更新边界
        if '边界' in rotated and angle in [90, 270]:
            rotated['边界'] = [0, 0, boundary_height, boundary_width]
        
        return rotated
    
    def scale(self, layout: Dict[str, List[int]], scale_factor: float) -> Dict[str, List[int]]:
        """
        缩放布局
        
        Args:
            layout: 原始布局
            scale_factor: 缩放因子
            
        Returns:
            Dict: 缩放后的布局
        """
        scaled = {}
        
        for name, params in layout.items():
            if len(params) != 4:
                scaled[name] = params
                continue
            
            x, y, w, h = params
            
            new_x = int(x * scale_factor)
            new_y = int(y * scale_factor)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            scaled[name] = [new_x, new_y, new_w, new_h]
        
        return scaled
    
    def translate(
        self, 
        layout: Dict[str, List[int]], 
        dx: int, 
        dy: int
    ) -> Dict[str, List[int]]:
        """
        平移布局
        
        Args:
            layout: 原始布局
            dx: x方向平移量
            dy: y方向平移量
            
        Returns:
            Dict: 平移后的布局
        """
        translated = {}
        
        for name, params in layout.items():
            if len(params) != 4:
                translated[name] = params
                continue
            
            x, y, w, h = params
            translated[name] = [x + dx, y + dy, w, h]
        
        return translated
    
    def add_noise(
        self, 
        layout: Dict[str, List[int]], 
        noise_range: int = 100
    ) -> Dict[str, List[int]]:
        """
        添加随机噪声
        
        Args:
            layout: 原始布局
            noise_range: 噪声范围（毫米）
            
        Returns:
            Dict: 添加噪声后的布局
        """
        import random
        
        noisy = {}
        
        for name, params in layout.items():
            if len(params) != 4:
                noisy[name] = params
                continue
            
            # 边界不添加噪声
            if name == '边界':
                noisy[name] = params
                continue
            
            x, y, w, h = params
            
            # 添加位置噪声
            new_x = x + random.randint(-noise_range, noise_range)
            new_y = y + random.randint(-noise_range, noise_range)
            
            noisy[name] = [max(0, new_x), max(0, new_y), w, h]
        
        return noisy


def augment_dataset(
    dataset: List[Dict[str, Any]],
    augmentor: LayoutDataAugmentation = None
) -> List[Dict[str, Any]]:
    """
    增强整个数据集
    
    Args:
        dataset: 原始数据集
        augmentor: 数据增强器
        
    Returns:
        List[Dict]: 增强后的数据集
    """
    if augmentor is None:
        augmentor = LayoutDataAugmentation()
    
    augmented_dataset = []
    
    for item in dataset:
        # 提取布局信息
        messages = item.get('messages', [])
        if len(messages) < 2:
            continue
        
        # 解析assistant的回复（生成的布局）
        assistant_content = messages[1].get('content', '')
        try:
            if '```json' in assistant_content:
                json_str = assistant_content.split('```json')[1].split('```')[0].strip()
            else:
                json_str = assistant_content.strip()
            
            layout = json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            continue
        
        # 获取已有布局信息（从user消息中提取）
        user_content = messages[0].get('content', '')
        
        # 增强布局
        augmented_layouts = augmentor.augment(layout, include_original=True)
        
        # 创建增强后的数据项
        for aug_layout in augmented_layouts:
            new_item = copy.deepcopy(item)
            new_item['messages'][1]['content'] = f"```json\n{json.dumps(aug_layout, ensure_ascii=False)}\n```"
            augmented_dataset.append(new_item)
    
    return augmented_dataset

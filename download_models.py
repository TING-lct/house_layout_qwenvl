"""
模型下载脚本 —— 自动下载所有需要的基座模型

运行方式：
    python download_models.py              # 下载全部模型
    python download_models.py --only 7b    # 只下载7B生成模型
    python download_models.py --only 14b   # 只下载14B评估模型
    python download_models.py --only embed # 只下载RAG向量模型
    python download_models.py --source modelscope  # 从ModelScope下载(国内推荐)

模型清单：
    ① Qwen2.5-VL-7B-Instruct  (~15GB)  - 户型图生成模型（必需）
    ② Qwen2.5-14B-Instruct    (~28GB)  - LLM评估模型（可选，提升评估质量）
    ③ paraphrase-multilingual-MiniLM-L12-v2 (~0.5GB) - RAG向量模型（可选）

下载位置：
    models/Qwen2.5-VL-7B-Instruct/
    models/Qwen2.5-14B-Instruct/
    models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
DEFAULT_MODELS_DIR = PROJECT_ROOT.parent / "models"


# ==================== 模型定义 ====================

def build_models(models_dir: Path) -> dict:
    return {
        "7b": {
            "name": "Qwen2.5-VL-7B-Instruct",
            "desc": "户型图生成模型（必需，~15GB）",
            "huggingface_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "modelscope_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "local_dir": models_dir / "Qwen2.5-VL-7B-Instruct",
            "required": True,
        },
        "14b": {
            "name": "Qwen2.5-14B-Instruct",
            "desc": "LLM评估模型（可选，~28GB）",
            "huggingface_id": "Qwen/Qwen2.5-14B-Instruct",
            "modelscope_id": "Qwen/Qwen2.5-14B-Instruct",
            "local_dir": models_dir / "Qwen2.5-14B-Instruct",
            "required": False,
        },
        "embed": {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "desc": "RAG向量嵌入模型（可选，~0.5GB）",
            "huggingface_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "modelscope_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "local_dir": models_dir / "sentence-transformers" / "paraphrase-multilingual-MiniLM-L12-v2",
            "required": False,
        },
    }


# ==================== 下载函数 ====================

def check_model_exists(model_info: dict) -> bool:
    """检查模型是否已下载"""
    local_dir = model_info["local_dir"]
    if not local_dir.exists():
        return False
    # 检查是否有实际模型文件
    has_safetensors = any(local_dir.glob("*.safetensors"))
    has_bin = any(local_dir.glob("*.bin"))
    has_config = (local_dir / "config.json").exists()
    return has_config and (has_safetensors or has_bin)


def get_disk_free_gb(path: Path) -> float:
    """获取磁盘剩余空间(GB)"""
    import shutil
    total, used, free = shutil.disk_usage(path.anchor)
    return free / (1024 ** 3)


def download_from_huggingface(model_info: dict, models_dir: Path) -> bool:
    """从 HuggingFace 下载模型"""
    model_id = model_info["huggingface_id"]
    local_dir = model_info["local_dir"]
    
    print(f"  📥 从 HuggingFace 下载: {model_id}")
    print(f"  📂 保存到: {local_dir}")
    
    try:
        from huggingface_hub import snapshot_download
        
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            resume_download=True,  # 支持断点续传
            max_workers=4,
        )
        return True
        
    except ImportError:
        print("  ⚠️  huggingface_hub 未安装，尝试 pip 安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
        
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            resume_download=True,
            max_workers=4,
        )
        return True


def download_from_modelscope(model_info: dict, models_dir: Path) -> bool:
    """从 ModelScope 下载模型（国内推荐）"""
    model_id = model_info["modelscope_id"]
    local_dir = model_info["local_dir"]
    
    print(f"  📥 从 ModelScope 下载: {model_id}")
    print(f"  📂 保存到: {local_dir}")
    
    try:
        from modelscope import snapshot_download as ms_download
        
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        ms_download(
            model_id,
            cache_dir=str(models_dir),
            revision="master",
        )
        
        # ModelScope 下载的目录结构可能不同，做一下兼容
        ms_cache_dir = models_dir / model_id.replace("/", os.sep)
        if ms_cache_dir.exists() and not local_dir.exists():
            ms_cache_dir.rename(local_dir)
        
        return True
        
    except ImportError:
        print("  ⚠️  modelscope 未安装，尝试 pip 安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
        from modelscope import snapshot_download as ms_download
        
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        ms_download(
            model_id,
            cache_dir=str(models_dir),
            revision="master",
        )
        return True


def download_model(
    key: str,
    models: dict,
    models_dir: Path,
    source: str = "huggingface",
    force: bool = False,
) -> bool:
    """
    下载指定模型
    
    Args:
        key: 模型标识 (7b / 14b / embed)
        source: 下载源 (huggingface / modelscope)
    """
    model_info = models[key]
    
    print(f"\n{'='*60}")
    print(f"📦 {model_info['name']}")
    print(f"   {model_info['desc']}")
    print(f"{'='*60}")
    
    # 检查是否已存在
    if check_model_exists(model_info) and not force:
        print(f"  ✅ 已存在，跳过下载: {model_info['local_dir']}")
        return True

    if force and model_info["local_dir"].exists():
        print(f"  ♻️ 强制重下，先删除旧目录: {model_info['local_dir']}")
        import shutil
        shutil.rmtree(model_info["local_dir"], ignore_errors=True)
    
    # 检查磁盘空间
    free_gb = get_disk_free_gb(models_dir if models_dir.exists() else PROJECT_ROOT)
    size_needed = {"7b": 16, "14b": 30, "embed": 1}[key]
    
    if free_gb < size_needed:
        print(f"  ❌ 磁盘空间不足! 需要 ~{size_needed}GB, 当前剩余 {free_gb:.1f}GB")
        return False
    
    print(f"  💾 磁盘剩余: {free_gb:.1f}GB, 预计需要: ~{size_needed}GB")
    
    # 下载
    try:
        if source == "modelscope":
            return download_from_modelscope(model_info, models_dir)
        else:
            return download_from_huggingface(model_info, models_dir)
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        
        # 如果 HuggingFace 失败，提示使用 ModelScope
        if source == "huggingface":
            print(f"\n  💡 如果 HuggingFace 访问受限，请尝试 ModelScope:")
            print(f"     python download_models.py --source modelscope")
        
        return False


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="下载户型图生成项目所需的基座模型",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["7b", "14b", "embed"],
        help="只下载指定模型:\n  7b    - Qwen2.5-VL-7B-Instruct (生成，必需)\n  14b   - Qwen2.5-14B-Instruct (评估，可选)\n  embed - 向量嵌入模型 (RAG，可选)"
    )
    parser.add_argument(
        "--source", type=str, default="huggingface",
        choices=["huggingface", "modelscope"],
        help="下载源:\n  huggingface - HuggingFace Hub (默认)\n  modelscope  - ModelScope (国内推荐)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="下载全部模型（包括可选模型）"
    )
    parser.add_argument(
        "--models-dir", type=str, default=str(DEFAULT_MODELS_DIR),
        help="模型保存目录（默认: 项目同级 models）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重下（删除已存在目录后重下）"
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir).resolve()
    models = build_models(models_dir)

    print("=" * 60)
    print("🏠 户型图生成 - 模型下载工具")
    print("=" * 60)
    print(f"  下载源: {args.source}")
    print(f"  保存目录: {models_dir}")
    
    # 创建 models 目录
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定要下载的模型
    if args.only:
        targets = [args.only]
    elif args.all:
        targets = ["7b", "14b", "embed"]
    else:
        # 默认只下载必需模型
        targets = ["7b"]
        print("\n  默认只下载必需的 7B 生成模型")
        print("  如需全部模型，请使用: python download_models.py --all")
    
    # 先显示计划
    print(f"\n📋 下载计划:")
    for key in targets:
        m = models[key]
        exists = "✅ 已存在" if check_model_exists(m) else "⏳ 待下载"
        required = "必需" if m["required"] else "可选"
        print(f"  [{required}] {m['name']} - {exists}")
    
    # 执行下载
    results = {}
    for key in targets:
        success = download_model(
            key,
            models=models,
            models_dir=models_dir,
            source=args.source,
            force=args.force,
        )
        results[key] = success
    
    # 汇总
    print(f"\n\n{'='*60}")
    print("📋 下载结果:")
    print(f"{'='*60}")
    
    all_ok = True
    for key, success in results.items():
        m = models[key]
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {status} - {m['name']}")
        if not success and m["required"]:
            all_ok = False
    
    # 检查 LoRA 适配器
    lora_path = PROJECT_ROOT / "lora_model"
    lora_14b_path = PROJECT_ROOT.parent / "qwen14b" / "Qwen2.5-14B-Instruct" / "Qwen2.5-14B-Instruct" / "lora" / "train_2025-12-01-21-17-23"
    
    print(f"\n📎 LoRA 适配器:")
    print(f"  7B生成LoRA:  {'✅' if lora_path.exists() else '❌'} {lora_path}")
    print(f"  14B评估LoRA: {'✅' if lora_14b_path.exists() else '❌'} {lora_14b_path}")
    
    if all_ok:
        print(f"\n🎉 模型准备就绪！可以运行:")
        print(f"   python run_full_generation.py")
    else:
        print(f"\n⚠️  部分必需模型下载失败，请检查网络后重试")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from peft import PeftModel
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# 基础模型路径
base_model_path = "models/Qwen/Qwen2.5-VL-7B-Instruct"
# LoRA适配器路径
lora_adapter_path = "house_floor/lora_model"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

val_peft_model = PeftModel.from_pretrained(model, lora_adapter_path)

# # default processer
processor = AutoProcessor.from_pretrained(base_model_path)

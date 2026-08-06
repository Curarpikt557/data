import json
import os
import cv2
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


from qwen_vl_utils import process_vision_info

# ==========================================
# 配置区域
# ==========================================
INPUT_JSON_PATH = "./grit_caption_full.json"     # GRiT 原始生成的 json
OUTPUT_JSON_PATH = "./grit_caption_refined.json" # 最终输出的 json
IMAGE_DIR = "../unc_train/"                   # 存放图片的目录 (请修改为你真实的图片路径)
MODEL_NAME = "../Qwen-3-VL-8B-Thinking"     # 默认使用 Qwen 最新的开源 VL 模型 (可替换为你的本地路径)

# ==========================================
# 模型初始化
# ==========================================
print("Loading VLM Model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    dtype=torch.bfloat16,          # 注意是 dtype，不是 torch_dtype
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model Loaded Successfully!")

# ==========================================
# 辅助函数
# ==========================================
def extract_json_from_text(text):
    """从 VLM 输出的文本中提取并解析 JSON"""
    try:
        # 尝试匹配 markdown 格式的 json 块
        json_pattern = r'```(?:json)?(.*?)```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            # 如果没有 markdown 块，尝试直接提取大括号内容
            start = text.find('{')
            end = text.rfind('}') + 1
            json_str = text[start:end]
            
        return json.loads(json_str)
    except Exception as e:
        print(f"[Warning] JSON Parsing Failed. Raw text: {text}")
        return {"error": "parsing_failed", "raw_text": text}

def prepare_image_and_prompt(image_path, bbox, original_caption):
    """
    处理图像（画红框）并生成对应的归一化坐标 Prompt
    输入 bbox 格式为 COCO 格式: [x_min, y_min, width, height]
    """
    # 1. 读取原图
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot find image: {image_path}")
        
    img_h, img_w = img.shape[:2]
    
    # 2. 解析 bbox 坐标
    x_min, y_min, w, h = bbox
    x_max, y_max = x_min + w, y_min + h
    
    # 将浮点数转换为整数
    x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)
    
    # 3. 在图上画红色框
    thickness = max(2, int(img_h * 0.005)) # 自适应线宽
    marked_img = img.copy()
    cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), thickness) # BGR: 红色
    
    # OpenCV (BGR) 转 PIL (RGB)
    marked_img_rgb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(marked_img_rgb)
    
    # 4. 坐标归一化 (Qwen-VL 通常是 0-1000 范围)
    norm_xmin = int((x_min / img_w) * 1000)
    norm_ymin = int((y_min / img_h) * 1000)
    norm_xmax = int((x_max / img_w) * 1000)
    norm_ymax = int((y_max / img_h) * 1000)

    coord_string = f"<box>({norm_xmin},{norm_ymin}),({norm_xmax},{norm_ymax})</box>"
    
    # 5. 构建全新的 Prompt
    system_prompt = """You are a highly precise Visual Attribute Analyzer and Description Generator. 
Your task is to identify the main subject in the red box, extract its objective attributes following a strict cognitive hierarchy, and generate a new, comprehensive description based ONLY on the extracted attributes. 
You strictly output valid JSON dictionaries only. Do not output any markdown formatting or explanations."""
    
    user_prompt = f"""Focus exclusively on the object inside the **Red Bounding Box** in the image, located at spatial coordinates {coord_string}.

The initial rough description for this object is: '{original_caption}'. 
Use this initial description ONLY to identify the main subject/target. You must re-evaluate all its attributes based strictly on the visual evidence inside the red box.

Output a strictly formatted JSON dictionary using ONE of the schemas below:

- If it is a Human, use this schema:
{{
  "category": "<base entity, e.g., 'man', 'woman', 'boy'>",
  "clothing": "<apparel worn, e.g., 'red t-shirt and blue jeans'>",
  "accessories_parts": "<attached items or features, e.g., 'wearing glasses', 'carrying a backpack'>",
  "action_posture": "<concrete physical posture/action, e.g., 'sitting', 'running'>",
  "new_description": "<A natural, fluent sentence combining ONLY the valid attributes identified above.>"
}}

- If it is a Non-human (animal, vehicle, object, etc.), use this schema:
{{
  "category": "<base entity, e.g., 'dog', 'car', 'chair'>",
  "color": "<primary visible colors, e.g., 'black and white'>",
  "material": "<visible material/texture, e.g., 'wooden', 'metallic'>",
  "state_status": "<physical state/condition, e.g., 'parked', 'open', 'broken'>",
  "new_description": "<A natural, fluent sentence combining ONLY the valid attributes identified above.>"
}}

=== STRICT RULES ===
1. ANTI-HALLUCINATION & UNKNOWN FALLBACK: Only extract attributes that are 100% clearly visible. If an attribute is invisible, occluded, uncertain, or inapplicable, you MUST assign the exact string "unknown". Do not guess.
2. ABSOLUTE PROHIBITION OF SPATIAL WORDS: DO NOT include ANY spatial, positional, or relational terms in ANY field (including 'new_description'). FORBIDDEN words include: "left", "right", "top", "bottom", "background", "next to", "on the table", "in front of", "located at".
3. STRICT VERB RULES: You are STRICTLY FORBIDDEN from using meaningless, abstract, or non-action verbs such as: "interacting", "interaction", "being", "existing", "located", "looking", "engaged in". The 'action_posture' MUST be a concrete, visually verifiable physical action (e.g., 'sitting', 'standing', 'holding a phone'). If unidentifiable, write "unknown".
4. NEW DESCRIPTION SYNTHESIS: The 'new_description' field must seamlessly integrate ONLY the valid (non-'unknown') attributes into a comprehensive noun phrase or sentence. Do not add any new information or spatial relations not present in the extracted attribute fields.
5. OUTPUT FORMAT: Output the raw JSON object ONLY. NO markdown tags (e.g., ```json), NO conversational text.
"""
    
    return pil_img, system_prompt, user_prompt

def process_single_object(image_path, bbox, original_caption):
    """处理单个目标，调用 VLM 并返回结构化属性"""
    try:
        pil_img, system_prompt, user_prompt = prepare_image_and_prompt(image_path, bbox, original_caption)
        
        # 组装 Qwen-VL 的 message 格式
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        # 使用 Processor 处理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # 推理生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
        # 提取 JSON
        structured_attrs = extract_json_from_text(output_text)
        return structured_attrs
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return {"error": str(e)}

# ==========================================
# 主流程控制
# ==========================================
def main():
    # 1. 加载数据
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} images from JSON.")
    
    # 2. 遍历处理
    for img_item in tqdm(data, desc="Processing Images"):
        image_id = img_item["image_id"]
        # 根据你的文件名规律动态拼装，如 COCO_train2014_000000000072.jpg
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg") 
        
        # 如果图片不存在则跳过
        if not os.path.exists(image_path):
            print(f"[Skip] Image not found: {image_path}")
            continue
            
        for obj in img_item["objects_3d"]:
            bbox = obj["bbox"] # [x_min, y_min, width, height]
            original_caption = obj["caption"]
            
            # 调用 VLM 获取结构化 JSON
            structured_attributes = process_single_object(image_path, bbox, original_caption)
            
            # 将新生成的属性写入原有的 obj 字典中
            obj["vlm_structured_attributes"] = structured_attributes

    # 3. 结果保存
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"\nDone! Enriched data saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
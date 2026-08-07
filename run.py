import json
import os
import cv2
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===================== 配置 =====================
INPUT_JSON_PATH = "./grit_full.json"
OUTPUT_JSON_PATH = "./grit_caption_refined.json"
IMAGE_DIR = "../unc_train/"
MODEL_NAME = "../Qwen-3-VL-8B-Thinking"
BATCH_SIZE = 4                 # 24G显存可稳跑4，可尝试6~8
MAX_NEW_TOKENS = 1024          # 给思考链预留足够空间
DEBUG = True                   # 调试时可开，正式大批量建议 False

# ===================== 模型加载 =====================
print("Loading VLM Model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model Loaded Successfully!")

# ===================== 工具函数 =====================
def clean_thinking_output(text):
    """去除 Qwen3-VL-Thinking 的思考过程，只保留最终 JSON"""
    # 移除  后的部分
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text:
        # 如果没有 </think>，可能是被截断了，尝试保留最后一段
        # 但一般都应加大 max_new_tokens，这里只能勉强尝试
        match = re.search(r'<\|im_start\|>assistant\n(.*)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text

def extract_json_from_text(text):
    """从清洗后文本中提取合法 JSON，自动过滤占位符"""
    text = text.strip()
    # 直接尝试解析
    if text.startswith('{'):
        try:
            return json.loads(text)
        except:
            pass

    # 提取所有花括号块，丢弃值为 "..." 的占位符
    candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    real_jsons = []
    for cand in candidates:
        if '": "...' in cand or '": "..."' in cand:
            continue
        try:
            obj = json.loads(cand)
            if all(v == '...' for v in obj.values()):
                continue
            real_jsons.append(obj)
        except:
            continue
    if real_jsons:
        return real_jsons[-1]      # 最后一个真实 JSON

    # 兜底
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    return {"error": "parsing_failed", "raw_text": text[:200]}

def build_prompt(original_caption, norm_xmin, norm_ymin, norm_xmax, norm_ymax):
    """构建用户提示（新 Prompt，保留你的设计）"""
    coord_string = f"<box>({norm_xmin},{norm_ymin}),({norm_xmax},{norm_ymax})</box>"
    return f"""Focus exclusively on the object inside the **Red Bounding Box** in the image, located at spatial coordinates {coord_string}.

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
1. ANTI-HALLUCINATION & UNKNOWN FALLBACK: Only extract attributes that are 100% clearly visible. If invisible/uncertain, use "unknown".
2. ABSOLUTE PROHIBITION OF SPATIAL WORDS: NO "left", "right", "top", "bottom", "background", "next to", "on the table", "in front of", "located at".
3. STRICT VERB RULES: NO abstract verbs like "interacting", "being", "existing", "located", "looking". Use concrete actions (e.g., "sitting", "holding a phone") or "unknown".
4. NEW DESCRIPTION SYNTHESIS: The 'new_description' must integrate ONLY the non-'unknown' attributes from above. No extra information.
5. OUTPUT FORMAT: Output the raw JSON object ONLY. No markdown, no explanations. Think briefly, then output the JSON directly."""

def draw_bbox(img, bbox, thickness):
    x_min, y_min, w, h = bbox
    x1, y1, x2, y2 = int(x_min), int(y_min), int(x_min+w), int(y_min+h)
    marked = img.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    return Image.fromarray(marked_rgb)

def process_image_batch(image_path, objects):
    """批量处理一张图片上的所有物体"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Skip] Image not found: {image_path}")
        return
    img_h, img_w = img.shape[:2]
    thickness = max(2, int(img_h * 0.005))

    results = []
    all_messages = []

    for obj in objects:
        bbox = obj["bbox"]
        caption = obj["caption"]

        pil_img = draw_bbox(img, bbox, thickness)

        x_min, y_min, w, h = bbox
        norm_xmin = int((x_min / img_w) * 1000)
        norm_ymin = int((y_min / img_h) * 1000)
        norm_xmax = int(((x_min + w) / img_w) * 1000)
        norm_ymax = int(((y_min + h) / img_h) * 1000)

        user_prompt = build_prompt(caption, norm_xmin, norm_ymin, norm_xmax, norm_ymax)
        system_prompt = "You are a highly precise Visual Attribute Analyzer. Think briefly, then output ONLY a valid JSON dictionary."

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
        all_messages.append(messages)

    # 分批推理
    for batch_start in range(0, len(all_messages), BATCH_SIZE):
        batch_msgs = all_messages[batch_start:batch_start+BATCH_SIZE]

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_msgs
        ]
        image_inputs, video_inputs = process_vision_info(batch_msgs)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for raw_out in raw_outputs:
            if DEBUG:
                print("\n" + "="*50)
                print(f"RAW OUTPUT (last 300 chars): ...{raw_out[-300:]}")

            cleaned = clean_thinking_output(raw_out)
            attrs = extract_json_from_text(cleaned)

            if DEBUG:
                print(f"PARSED: {json.dumps(attrs, indent=2)}")
                if "error" in attrs:
                    print(f"⚠️ Parse failed. Full raw: ...{raw_out[-200:]}")

            results.append(attrs)

        del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs

    # 写回 objects
    for obj, res in zip(objects, results):
        obj["vlm_structured_attributes"] = res

# ===================== 主流程 =====================
def main():
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} images from JSON.")

    for img_item in tqdm(data, desc="Processing Images"):
        image_id = img_item["image_id"]
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        objects = img_item.get("objects_3d", [])
        if not objects or not os.path.exists(image_path):
            continue
        process_image_batch(image_path, objects)

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nDone! Enriched data saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
import json
import os
import time
import cv2
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===================== 配置 =====================
INPUT_JSON_PATH = "./grit_caption_full.json"
OUTPUT_JSON_PATH = "./grit_caption_refined.json"
IMAGE_DIR = "../unc_train/"
MODEL_NAME = "../Qwen-3-VL-8B-Thinking"
BATCH_SIZE = 8                 # 最佳速度/显存平衡
MAX_NEW_TOKENS = 640           # 确保思考链+JSON完整输出
MIN_NEW_TOKENS = 15            # 防止空输出
IMAGE_SIZE = 896               # 如果加入缩放可以进一步提速
DEBUG = False
PRINT_ISSUES = True            # 保留问题打印，便于追踪 
# ===================== 模型加载 =====================
print("Loading VLM Model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
print(f"Model loaded! Batch size = {BATCH_SIZE}")

# ===================== 工具函数 =====================
def clean_thinking_output(text):
    """去掉 Qwen3-Thinking 的思考过程，只保留最终 JSON"""
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text:
        match = re.search(r'<\|im_start\|>assistant\n(.*)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text

def try_fix_truncated_json(text):
    """补全缺失的闭合花括号"""
    text = text.strip()
    if text.startswith('{'):
        open_braces = text.count('{') - text.count('}')
        if open_braces > 0:
            text += '}' * open_braces
    return text

def extract_json_from_text(text):
    """从清洗后的文本中提取合法 JSON，过滤占位符"""
    text = text.strip()
    # 直接解析
    if text.startswith('{'):
        try:
            return json.loads(text)
        except:
            fixed = try_fix_truncated_json(text)
            try:
                return json.loads(fixed)
            except:
                pass

    # 搜索所有 {...} 块
    candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    real_jsons = []
    for cand in candidates:
        # 跳过明显的占位符
        if '": "...' in cand or '": "..."' in cand:
            continue
        try:
            obj = json.loads(cand)
            if all(v == '...' for v in obj.values()):
                continue
            real_jsons.append(obj)
        except:
            fixed = try_fix_truncated_json(cand)
            try:
                obj = json.loads(fixed)
                real_jsons.append(obj)
            except:
                continue
    if real_jsons:
        return real_jsons[-1]

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
    """构建统一属性提取 Prompt（认知科学层级）"""
    coord_string = f"<box>({norm_xmin},{norm_ymin}),({norm_xmax},{norm_ymax})</box>"
    return f"""Analyze the object inside the red box at {coord_string}.
Initial rough description: '{original_caption}'. Use it ONLY to identify the main subject, then re-evaluate all attributes visually.

Output a STRICT JSON with exactly these 7 fields. Replace the example values with actual observations; use "unknown" if something is invisible or uncertain.

{{
  "category": "the base type, e.g., elephant, man, pizza",
  "color": "primary visible colors, e.g., gray, red and white",
  "size": "absolute size (large, small) or relative size (larger, smaller) if multiple same objects. If not sure, use 'unknown'",
  "material": "visible material or texture, e.g., skin, metal, wooden. For humans, this is the clothing material, e.g., cotton",
  "parts": "conspicuous parts or attached items, e.g., long trunk, wearing glasses, holding a fork",
  "spatial_relation": "if multiple same-category objects exist, briefly disambiguate with a simple relation (e.g., leftmost, behind the fence, on the table). Otherwise write 'none'",
  "state_action": "concrete physical state or action, e.g., standing, sitting, cutting. No abstract verbs (interacting, looking, being)"
}}

Examples:
Single elephant: {{"category":"elephant","color":"gray","size":"large","material":"skin","parts":"long trunk","spatial_relation":"none","state_action":"walking"}}
Multiple elephants, this one is the larger leftmost: {{"category":"elephant","color":"gray","size":"larger","material":"skin","parts":"long trunk","spatial_relation":"leftmost","state_action":"walking"}}
Human: {{"category":"woman","color":"blue","size":"unknown","material":"cotton shirt","parts":"wearing glasses, holding a cup","spatial_relation":"none","state_action":"sitting"}}

Rules:
- Never output placeholder text like "<entity>" or "...". Replace all example values with actual observations.
- Unknown → "unknown". No guessing.
- For spatial_relation, use simple objective relations only when necessary to distinguish identical objects. Prefer scene-relative (behind the table) over viewer-relative (on the left).
- No abstract verbs. Use concrete actions.
- Output the JSON directly. Think briefly, then output."""

def prepare_image_and_coords(img, bbox, thickness):
    """在原始图像上画红框，返回 PIL 图像和归一化坐标（基于原图尺寸）"""
    h, w = img.shape[:2]
    x_min, y_min, bw, bh = bbox
    x_max, y_max = x_min + bw, y_min + bh

    marked = img.copy()
    cv2.rectangle(marked, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), thickness)
    marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(marked_rgb)

    norm_xmin = int((x_min / w) * 1000)
    norm_ymin = int((y_min / h) * 1000)
    norm_xmax = int((x_max / w) * 1000)
    norm_ymax = int((y_max / h) * 1000)

    return pil_img, norm_xmin, norm_ymin, norm_xmax, norm_ymax

def process_image_batch(image_path, objects):
    """批量处理一张图上的所有目标"""
    img = cv2.imread(image_path)
    if img is None:
        if PRINT_ISSUES:
            print(f"[Skip] Image not found: {image_path}")
        return

    thickness = max(2, int(max(img.shape[:2]) * 0.005))

    # 为每个物体构建消息
    all_messages = []
    for obj in objects:
        bbox = obj["bbox"]
        caption = obj["caption"]
        pil_img, nxmin, nymin, nxmax, nymax = prepare_image_and_coords(img, bbox, thickness)
        user_prompt = build_prompt(caption, nxmin, nymin, nxmax, nymax)
        system_prompt = "You are a precise visual attribute extractor. Output ONLY a JSON dictionary. No extra text."

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

    # 分批处理
    results = [None] * len(objects)
    for batch_start in range(0, len(all_messages), BATCH_SIZE):
        batch_msgs = all_messages[batch_start:batch_start+BATCH_SIZE]

        batch_images = []
        batch_texts = []
        for msg in batch_msgs:
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info([msg])
            batch_texts.append(text)
            batch_images.extend(imgs)

        inputs = processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i, raw_out in enumerate(raw_outputs):
            idx = batch_start + i
            caption = objects[idx]["caption"]

            cleaned = clean_thinking_output(raw_out)
            attrs = extract_json_from_text(cleaned)

            # 是否触发问题打印
            problem = False
            if "error" in attrs or len(raw_out.strip()) < 20:
                problem = True
                # 回退为全部 unknown 的结构化字典
                attrs = {
                    "category": "unknown", "color": "unknown", "size": "unknown",
                    "material": "unknown", "parts": "unknown",
                    "spatial_relation": "unknown", "state_action": "unknown"
                }
            else:
                # 检测占位符
                placeholder_vals = ["...", "<entity>", "<man/woman/boy/girl>", "e.g."]
                if any(v in placeholder_vals for v in attrs.values()):
                    problem = True

            if problem and PRINT_ISSUES:
                print(f"\n⚠️ Issue with caption: {caption}")
                print(f"   Raw output (first 400 chars): {raw_out[:400]}")
                print(f"   Final JSON: {json.dumps(attrs, ensure_ascii=False)}")

            results[idx] = attrs

        del inputs, generated_ids, generated_ids_trimmed, batch_images

    # 写回对象
    for obj, res in zip(objects, results):
        obj["vlm_structured_attributes"] = res

# ===================== 主流程 =====================
def main():
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} images.")

    total_start = time.time()
    for img_item in tqdm(data, desc="Processing"):
        image_id = img_item["image_id"]
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        objects = img_item.get("objects_3d", [])
        if not objects or not os.path.exists(image_path):
            continue
        process_image_batch(image_path, objects)

    total_hours = (time.time() - total_start) / 3600
    print(f"\nDone! Total time: {total_hours:.2f} hours. Saving to {OUTPUT_JSON_PATH}")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
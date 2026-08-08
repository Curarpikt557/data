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
BATCH_SIZE = 8                # 24G 显存可稳跑 6，可尝试 8
MAX_NEW_TOKENS = 512          # 简化输出后无需 1024
MIN_NEW_TOKENS = 10           # 防止空输出
MAX_IMAGE_SIZE = 1024         # 限制图片最长边，加速编码且节省显存
DEBUG = False                 # 批量跑时关闭调试打印

# ===================== 模型加载 =====================
print("Loading VLM Model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # 如果安装了 flash-attn 可取消下行注释
    # attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# 左填充 + pad_token 设置，避免右填充警告和生成混乱
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

print("Model Loaded Successfully!")

# ===================== 工具函数 =====================
def clean_thinking_output(text):
    """去除 Qwen3-VL-Thinking 的思考过程，只保留最终 JSON"""
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text:
        match = re.search(r'<\|im_start\|>assistant\n(.*)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text

def try_fix_truncated_json(text):
    """尝试修复被截断的 JSON，补全缺失的闭合括号"""
    text = text.strip()
    if text.startswith('{'):
        open_braces = text.count('{') - text.count('}')
        if open_braces > 0:
            text += '}' * open_braces
    return text

def extract_json_from_text(text):
    """从清洗后文本中提取合法 JSON，过滤占位符，支持截断修复"""
    text = text.strip()
    # 直接尝试解析
    if text.startswith('{'):
        try:
            return json.loads(text)
        except:
            fixed = try_fix_truncated_json(text)
            try:
                return json.loads(fixed)
            except:
                pass

    # 提取所有花括号块，丢弃占位符
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
    """构建简化的用户提示，只要求属性 JSON，无需合成描述"""
    coord_string = f"<box>({norm_xmin},{norm_ymin}),({norm_xmax},{norm_ymax})</box>"
    return f"""Focus on the object in the red box at {coord_string}.
Initial rough description: '{original_caption}'. Use it only to identify the main subject, then re-evaluate all attributes from visual evidence.

Output ONLY a strict JSON dictionary (no markdown, no extra text) using one of these schemas:

- Human: {{"category": "<man/woman/boy/girl>", "clothing": "...", "accessories_parts": "...", "action_posture": "..."}}
- Non-human: {{"category": "<entity>", "color": "...", "material": "...", "state_status": "..."}}

RULES:
1. Unknown/uncertain → "unknown". NO guessing.
2. NO spatial words (left/right/background/on the table/next to).
3. NO abstract verbs (interacting/looking/being). Use concrete actions or "unknown".
4. Output the JSON directly. Think briefly, then output."""

def draw_bbox(img, bbox, thickness):
    """在原图上画红框，返回 PIL Image"""
    x_min, y_min, w, h = bbox
    x1, y1, x2, y2 = int(x_min), int(y_min), int(x_min+w), int(y_min+h)
    marked = img.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    return Image.fromarray(marked_rgb)

def resize_image_if_needed(img, max_size):
    """如果图片最长边超过 max_size，等比例缩放"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    return img

def fallback_attributes(caption):
    """当输出无法解析时，使用原始 caption 作为回退"""
    if any(w in caption.lower() for w in ["man", "woman", "person", "girl", "boy", "lady", "guy"]):
        return {
            "category": "unknown",
            "clothing": "unknown",
            "accessories_parts": "unknown",
            "action_posture": "unknown"
        }
    else:
        return {
            "category": "unknown",
            "color": "unknown",
            "material": "unknown",
            "state_status": "unknown"
        }

def process_image_batch(image_path, objects):
    """批量处理一张图片上的所有物体"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Skip] Image not found: {image_path}")
        return
    # 缩放图片以加速编码
    img = resize_image_if_needed(img, MAX_IMAGE_SIZE)
    img_h, img_w = img.shape[:2]
    thickness = max(2, int(img_h * 0.005))

    results = [None] * len(objects)

    # 构建所有 messages
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

    # 分批推理
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

            if DEBUG:
                print("\n" + "="*50)
                print(f"OBJECT {idx}: {caption}")
                print(f"RAW (last 300): ...{raw_out[-300:]}")

            cleaned = clean_thinking_output(raw_out)
            attrs = extract_json_from_text(cleaned)

            # 回退处理
            if "error" in attrs or len(raw_out.strip()) < 30:
                if DEBUG:
                    print("⚠️ Fallback due to parse failure or short output.")
                attrs = fallback_attributes(caption)
            else:
                # 确保包含正确的字段，否则回退
                if "category" not in attrs:
                    attrs = fallback_attributes(caption)

            results[idx] = attrs

            if DEBUG:
                print(f"PARSED: {json.dumps(attrs, indent=2)}")

        del inputs, generated_ids, generated_ids_trimmed, batch_images

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
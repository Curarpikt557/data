import json
import os
import cv2
import re
import time
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
BATCH_SIZE = 32                 # 实测 8 可稳定跑，16 显存够但速度反而不稳定
MAX_NEW_TOKENS = 256           # 大幅缩减，属性 JSON 通常 100 token 足够
MIN_NEW_TOKENS = 5             # 防止空输出
IMAGE_SIZE = 896               # 统一缩放到 896x896，兼顾速度与细节
DEBUG = False                  # 生产务必关闭，I/O 极慢

# ===================== 模型加载 =====================
print("Loading VLM Model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # 若安装了 flash-attn 可取消下行注释
    # attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
print(f"Model Loaded! Batch size: {BATCH_SIZE}")

# ===================== 工具函数 =====================
def clean_thinking_output(text):
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text:
        match = re.search(r'<\|im_start\|>assistant\n(.*)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text

def try_fix_truncated_json(text):
    text = text.strip()
    if text.startswith('{'):
        open_braces = text.count('{') - text.count('}')
        if open_braces > 0:
            text += '}' * open_braces
    return text

def extract_json_from_text(text):
    text = text.strip()
    if text.startswith('{'):
        try:
            return json.loads(text)
        except:
            fixed = try_fix_truncated_json(text)
            try:
                return json.loads(fixed)
            except:
                pass

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

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    return {"error": "parsing_failed", "raw_text": text[:200]}

def build_prompt(original_caption, norm_xmin, norm_ymin, norm_xmax, norm_ymax):
    coord_string = f"<box>({norm_xmin},{norm_ymin}),({norm_xmax},{norm_ymax})</box>"
    return f"""Analyze the object in the red box at {coord_string}.
Initial rough description: '{original_caption}'. Use it only to identify the main subject, then re-evaluate all attributes from visual evidence.

Output ONLY a JSON dictionary (no markdown) with these fields:
{{"category":"...","color":"...","size":"...","material":"...","parts":"...","spatial_relation":"...","state_action":"..."}}

Rules:
- Unknown → "unknown". No guessing.
- size: absolute or relative (larger/smaller) if multiple same objects.
- spatial_relation: only if needed to distinguish identical objects, e.g., "leftmost", "behind the table"; otherwise "none".
- No abstract verbs (interacting, looking). Use concrete actions.
- Think briefly, then output the JSON directly."""

def prepare_image_and_coords(img, bbox, thickness):
    """在图上画框，缩放并返回 PIL 图像和归一化坐标"""
    h, w = img.shape[:2]
    x_min, y_min, bw, bh = bbox
    x_max, y_max = x_min + bw, y_min + bh

    # 画红框
    marked = img.copy()
    cv2.rectangle(marked, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                  (0, 0, 255), thickness)
    marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(marked_rgb)

    # 归一化到原图尺寸
    norm_xmin = int((x_min / w) * 1000)
    norm_ymin = int((y_min / h) * 1000)
    norm_xmax = int((x_max / w) * 1000)
    norm_ymax = int((y_max / h) * 1000)

    return pil_img, norm_xmin, norm_ymin, norm_xmax, norm_ymax

def resize_image_to_square(img, target_size):
    """将图像等比例缩放并填充到 target_size x target_size，保持长宽比"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # 创建方形画布并粘贴
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # 更新 bbox 的偏移（后面的框绘制是基于 canvas 的，所以需要调整坐标）
    # 但我们先绘制框再缩放？更好的做法：先绘制在原图，再缩放整个画过框的图像
    # 这里改为：直接在原图画框，然后整体缩放并填充，这样框也会被正确缩放
    # 所以我们修改流程：先画框，再调用 resize_image_to_square
    return canvas, scale, x_offset, y_offset

def process_image_batch(image_path, objects):
    """批量处理一张图片上的所有物体，统一图像尺寸"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Skip] Image not found: {image_path}")
        return

    thickness = max(2, int(max(img.shape[:2]) * 0.005))
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

    # 分批推理
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

            if DEBUG:
                print(f"\nORIGINAL: {caption}")

            cleaned = clean_thinking_output(raw_out)
            attrs = extract_json_from_text(cleaned)

            need_print_raw = False
            if "error" in attrs or len(raw_out.strip()) < 20:
                need_print_raw = True
                attrs = {
                    "category": "unknown", "color": "unknown", "size": "unknown",
                    "material": "unknown", "parts": "unknown",
                    "spatial_relation": "unknown", "state_action": "unknown"
                }
            else:
                placeholder_vals = ["...", "<entity>", "<man/woman/boy/girl>", "e.g."]
                if any(v in placeholder_vals for v in attrs.values()):
                    need_print_raw = True

            if need_print_raw and DEBUG:
                print(f"⚠️ Issue. RAW:\n{raw_out[:500]}")

            results[idx] = attrs

            if DEBUG:
                print(f"JSON: {json.dumps(attrs, ensure_ascii=False)}")

        del inputs, generated_ids, generated_ids_trimmed, batch_images

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

    total_time = time.time() - total_start
    print(f"\nDone! Total time: {total_time/3600:.2f} hours. Saved to {OUTPUT_JSON_PATH}")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
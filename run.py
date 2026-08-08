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
  
# ===================== 可调参数 =====================
INPUT_JSON_PATH = "./grit_full.json"
OUTPUT_JSON_PATH = "./grit_caption_refined.json"
IMAGE_DIR = "../unc_train/"
MODEL_NAME = "../Qwen-3-VL-8B-Thinking"
BATCH_SIZE = 12                # 统一图像尺寸后，12 可稳定运行；若 OOM 则降为 8
MAX_NEW_TOKENS = 768           # 必须足够大，保证 JSON 完整生成
MIN_NEW_TOKENS = 10
TARGET_IMAGE_SIZE = 672        # 统一缩放至此尺寸（正方形填充），减少 padding，提升 batch 效率
PRINT_ISSUES = True            # 是否打印失败案例（调试时开启，生产可关）

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
print(f"Model loaded. Batch size = {BATCH_SIZE}")

# ===================== 工具函数 =====================
def resize_and_pad(img, target_size):
    """等比缩放并填充至 target_size x target_size，返回 (新图, 缩放因子, 偏移x, 偏移y)"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas, scale, x_offset, y_offset

def clean_thinking_output(text):
    """提取 <answer> 中的内容，失败则回退到 </think> 之后"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
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
    return f"""Analyze the object inside the red box at {coord_string}.
Initial rough description: '{original_caption}'. Use it only to identify the target.

Respond EXACTLY in the following format:

<think>
Think in 1 sentence: human or non-human? base category? key visible attributes?
</think>
<answer>
{{"type": "human or non-human", "category": "base name", "description": "A comprehensive noun phrase: [relative size if comparing] [color+pattern] [material] [category] [parts] [spatial relation if needed] [concrete action/state]"}}
</answer>

Rules for description:
- Color+pattern: merge, e.g., "dark brown with lighter spots", "golden brown"
- Size: only if comparing same objects or noticeably large/small, e.g., "the larger elephant", "a small kitten"
- Spatial relation: ONLY if multiple same objects exist; use shortest scene-relative, e.g., "the leftmost giraffe"
- Action/state: concrete only (standing, sitting, cutting). No abstract verbs.
- Output ONLY the JSON inside <answer>. No markdown."""

def process_image_batch(image_path, objects):
    img = cv2.imread(image_path)
    if img is None:
        return

    # 统一图像尺寸，为后续批量推理减少 padding
    img_resized, scale, x_off, y_off = resize_and_pad(img, TARGET_IMAGE_SIZE)
    h_orig, w_orig = img.shape[:2]
    thickness = max(2, int(max(h_orig, w_orig) * 0.005))

    all_messages = []
    for obj in objects:
        bbox = obj["bbox"]
        caption = obj["caption"]

        # 在原图上画框，然后统一缩放 + 填充
        marked = img.copy()
        x_min, y_min, bw, bh = bbox
        x_max, y_max = x_min + bw, y_min + bh
        cv2.rectangle(marked, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), thickness)
        marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)

        # 缩放并填充至固定尺寸
        h, w = marked_rgb.shape[:2]
        scale_f = TARGET_IMAGE_SIZE / max(h, w)
        new_w, new_h = int(w * scale_f), int(h * scale_f)
        resized = cv2.resize(marked_rgb, (new_w, new_h))
        canvas = np.zeros((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3), dtype=np.uint8)
        x_off = (TARGET_IMAGE_SIZE - new_w) // 2
        y_off = (TARGET_IMAGE_SIZE - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        pil_img = Image.fromarray(canvas)

        # 坐标仍基于原图尺寸归一化
        norm_xmin = int((x_min / w_orig) * 1000)
        norm_ymin = int((y_min / h_orig) * 1000)
        norm_xmax = int((x_max / w_orig) * 1000)
        norm_ymax = int((y_max / h_orig) * 1000)

        user_prompt = build_prompt(caption, norm_xmin, norm_ymin, norm_xmax, norm_ymax)
        system_prompt = "You are a visual attribute extractor. Think in 1 sentence, then output JSON inside <answer> tags."

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

            problem = False
            if "error" in attrs or len(raw_out.strip()) < 20:
                problem = True
                attrs = {
                    "type": "unknown",
                    "category": "unknown",
                    "description": caption   # 保留原始 caption 作为兜底描述
                }
            else:
                placeholder_vals = ["...", "<entity>", "<man/woman/boy/girl>", "e.g."]
                if any(v in placeholder_vals for v in attrs.values()):
                    problem = True

            if problem and PRINT_ISSUES:
                print(f"\n⚠️ {caption}")
                print(f"   Raw (first 800 chars): {raw_out[:800]}")

            results[idx] = attrs

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

    total_hours = (time.time() - total_start) / 3600
    print(f"\nDone! Total: {total_hours:.2f}h. Saved to {OUTPUT_JSON_PATH}")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
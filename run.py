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
INPUT_JSON_PATH = "./grit_full.json"      # 输入 GRiT JSON
OUTPUT_JSON_PATH = "./grit_caption_refined.json"  # 输出 JSON
IMAGE_DIR = "../unc_train/"                       # 图片目录
MODEL_NAME = "../Qwen-3-VL-8B-Thinking"           # 本地模型路径
BATCH_SIZE = 32                                  # 统一图像尺寸后可尝试 12
MAX_NEW_TOKENS = 768                              # 确保 JSON 完整
MIN_NEW_TOKENS = 10
TARGET_IMAGE_SIZE = 672                           # 图像缩放尺寸，减少 padding
PRINT_ALL_RESULTS = True                          # 是否打印每个物体的原始 caption 和 JSON
PRINT_ISSUES = True                               # 是否打印失败案例

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
    """等比缩放并居中填充至 target_size x target_size"""
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
    """提取 <answer> 中的 JSON，失败则回退到 </think> 之后"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 尝试 </think> 分割
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

Respond EXACTLY:

<think>
1 sentence: human/non-human? category? key attributes?
</think>
<answer>
{{"type": "human/non-human", "category": "base name", "description": "A comprehensive noun phrase: [size] [color+pattern] [material] [category] [parts] [spatial if needed] [action/state]"}}
</answer>

Rules:
- Color+pattern: e.g., "dark brown with lighter spots"
- Size: only if comparing same objects or notably large/small
- Spatial: only if multiple same objects exist, shortest scene-relative
- Action: concrete only (sitting, cutting); no abstract verbs.
- Output ONLY the JSON inside <answer>. No markdown."""

def process_image_batch(image_path, objects):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Skip] Image not found: {image_path}")
        return

    img_h, img_w = img.shape[:2]
    thickness = max(2, int(max(img_h, img_w) * 0.005))

    all_messages = []
    for obj in objects:
        bbox = obj["bbox"]
        caption = obj["caption"]

        # 画红框
        marked = img.copy()
        x_min, y_min, bw, bh = bbox
        x_max, y_max = x_min + bw, y_min + bh
        cv2.rectangle(marked, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), thickness)
        marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)

        # 统一图像尺寸，减少 padding
        pil_img = Image.fromarray(marked_rgb)
        # Qwen processor 可以自动处理尺寸，但为了一致性，我们也可以在 PIL 上调整
        # 这里直接使用 PIL 调整到统一尺寸
        pil_img = pil_img.resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), Image.LANCZOS)

        # 坐标归一化（基于原图）
        norm_xmin = int((x_min / img_w) * 1000)
        norm_ymin = int((y_min / img_h) * 1000)
        norm_xmax = int((x_max / img_w) * 1000)
        norm_ymax = int((y_max / img_h) * 1000)

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
    image_start_time = time.time()

    for batch_start in range(0, len(all_messages), BATCH_SIZE):
        batch_msgs = all_messages[batch_start:batch_start+BATCH_SIZE]
        batch_prep_start = time.time()

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

        batch_prep_time = time.time() - batch_prep_start

        generation_start = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        generation_time = time.time() - generation_start

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

            if "error" in attrs or len(raw_out.strip()) < 20:
                if PRINT_ISSUES:
                    print(f"\n⚠️ Fallback for: {caption}")
                    print(f"   Raw output (first 800): {raw_out[:800]}")
                attrs = {
                    "type": "unknown",
                    "category": "unknown",
                    "description": caption   # 保留原始 GRiT 描述
                }
            else:
                placeholder_vals = ["...", "<entity>", "<man/woman/boy/girl>", "e.g."]
                if any(v in placeholder_vals for v in attrs.values()):
                    if PRINT_ISSUES:
                        print(f"\n⚠️ Placeholder in: {caption}")
                        print(f"   Raw (first 800): {raw_out[:800]}")

            results[idx] = attrs

            if PRINT_ALL_RESULTS:
                print(f"\n📷 {caption}")
                print(f"   ➡ {json.dumps(attrs, ensure_ascii=False)}")

        # 打印 batch 计时
        if PRINT_ALL_RESULTS:
            print(f"   ⏱ Batch ({len(batch_msgs)} objs): prep {batch_prep_time:.2f}s, generation {generation_time:.2f}s")

        del inputs, generated_ids, generated_ids_trimmed, batch_images

    image_time = time.time() - image_start_time
    print(f"\n🖼 {os.path.basename(image_path)} done in {image_time:.2f}s ({len(objects)} objects)")

    for obj, res in zip(objects, results):
        obj["vlm_structured_attributes"] = res

# ===================== 主流程 =====================
def main():
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} images.")

    total_start = time.time()
    processed_images = 0
    for img_item in tqdm(data, desc="Processing"):
        image_id = img_item["image_id"]
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        objects = img_item.get("objects_3d", [])
        if not objects or not os.path.exists(image_path):
            continue
        process_image_batch(image_path, objects)
        processed_images += 1

    total_time = time.time() - total_start
    print(f"\n🏁 All done! Processed {processed_images} images in {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    print(f"Average per image: {total_time/processed_images:.2f}s")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
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

# ==========================================
# 调试开关
# ==========================================
DEBUG = True

# ==========================================
# 配置区域
# ==========================================
INPUT_JSON_PATH = "./grit_caption_full.json"
OUTPUT_JSON_PATH = "./grit_caption_refined.json"
IMAGE_DIR = "../unc_train/"
MODEL_NAME = "../Qwen-3-VL-8B-Thinking"

# ==========================================
# 模型初始化
# ==========================================
print("Loading VLM Model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model Loaded Successfully!")

# ==========================================
# 辅助函数
# ==========================================
def clean_thinking_output(text):
    """去除 Qwen3‑VL‑Thinking 的思考过程，只保留最终的 JSON"""
    # 移除 ... 及其内容
    text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL).strip()
    # 如果仍然没有有效内容，尝试提取  response 后的部分
    if not text:
        match = re.search(r'<\|im_start\|>assistant\n(.*)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text

def extract_json_from_text(text):
    """从清洗后的文本中提取 JSON（假设只剩 JSON）"""
    # 先尝试找完整的 JSON 对象
    # 去除可能的首尾空白和标记
    text = text.strip()
    # 如果以 { 开头，尝试直接解析
    if text.startswith('{'):
        try:
            return json.loads(text)
        except:
            pass
    # 否则提取第一个 { 到最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    # 最终失败
    return {"error": "parsing_failed", "raw_text": text[:200]}  # 只保留前200字符

def prepare_image_and_prompt(image_path, bbox, original_caption):
    """处理图像，画红框，生成 Prompt（已修正坐标顺序）"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot find image: {image_path}")
    img_h, img_w = img.shape[:2]

    x_min, y_min, w, h = bbox
    x_max, y_max = x_min + w, y_min + h
    x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)

    # 画红框
    thickness = max(2, int(img_h * 0.005))
    marked_img = img.copy()
    cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    marked_img_rgb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(marked_img_rgb)

    # 修正：先 x 后 y
    norm_xmin = int((x_min / img_w) * 1000)
    norm_ymin = int((y_min / img_h) * 1000)
    norm_xmax = int((x_max / img_w) * 1000)
    norm_ymax = int((y_max / img_h) * 1000)
    coord_string = f"<box>({norm_xmin},{norm_ymin}),({norm_xmax},{norm_ymax})</box>"

    system_prompt = "You are an expert visual attribute analyzer. You strictly output valid JSON dictionaries only."
    user_prompt = f"""Focus exclusively on the object inside the **Red Bounding Box**, which is at {coord_string}.

The initial rough description: '{original_caption}'.

Output a strict JSON following these rules:
- Human → {{"object": "...", "clothing": "...", "clothing_color": "...", "action_state": "..."}}
- Non-human → {{"object": "...", "color": "...", "material": "...", "state_status": "..."}}
If unknown, use "unknown". Output ONLY the JSON, no extra text."""

    return pil_img, system_prompt, user_prompt

def process_single_object(image_path, bbox, original_caption):
    """处理单个目标，返回结构化属性，并打印调试信息"""
    try:
        pil_img, system_prompt, user_prompt = prepare_image_and_prompt(image_path, bbox, original_caption)

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

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,          # 增大，避免截断
                do_sample=False             # 确定性输出
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_output = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        if DEBUG:
            print("\n" + "="*50)
            print(f"IMAGE: {image_path}")
            print(f"ORIGINAL GRiT CAPTION: {original_caption}")
            print(f"VLM RAW OUTPUT:\n{raw_output}")

        # 清洗思考链，提取 JSON
        cleaned = clean_thinking_output(raw_output)
        if DEBUG:
            print(f"CLEANED TEXT (for JSON extraction):\n{cleaned}")

        structured_attrs = extract_json_from_text(cleaned)

        if DEBUG:
            print(f"PARSED ATTRIBUTES: {json.dumps(structured_attrs, indent=2)}")

        return structured_attrs

    except Exception as e:
        print(f"ERROR processing {image_path}: {str(e)}")
        return {"error": str(e)}

# ==========================================
# 主流程控制
# ==========================================
def main():
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} images from JSON.")

    for img_item in tqdm(data, desc="Processing Images"):
        image_id = img_item["image_id"]
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"[Skip] Image not found: {image_path}")
            continue

        for obj in img_item["objects_3d"]:
            bbox = obj["bbox"]
            original_caption = obj["caption"]
            structured_attributes = process_single_object(image_path, bbox, original_caption)
            obj["vlm_structured_attributes"] = structured_attributes

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nDone! Enriched data saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
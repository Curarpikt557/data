import json
import cv2
import os
from tqdm import tqdm

def filter_captions_by_bbox(
    input_json_path,
    image_dir,
    output_json_path=None,
    min_boxes=10,
    max_boxes=36,
    tiny_area_thresh=0.015625,   # 面积占比阈值
    conf_key="conf",             # GRiT 置信度字段名，若无则改为 None
    verbose=True
):
    """
    对 GRiT 生成的 caption 进行筛选：
    - 剔除面积过小的 bbox
    - 每张图片最多保留 max_boxes 个物体，最少 min_boxes
    - 按置信度降序选取

    Args:
        input_json_path : 原始 grit_caption_full.json 路径
        image_dir       : 图片存放目录
        output_json_path: 输出筛选后的 JSON，若为 None 则覆盖原文件
        min_boxes, max_boxes, tiny_area_thresh : 同 ObjectPerceptionModule 参数
        conf_key        : JSON 中每个物体的置信度键名，若为 None 则不排序
        verbose         : 是否打印过滤统计

    Returns:
        filtered_data : 筛选后的 list of dict
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_before = sum(len(img["objects_3d"]) for img in data)
    total_after = 0

    for img_item in tqdm(data, desc="Filtering captions"):
        image_id = img_item["image_id"]
        # 读取图像尺寸（用于面积过滤）
        img_path = os.path.join(image_dir, f"{image_id}.jpg")
        img_h, img_w = None, None
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_h, img_w = img.shape[:2]

        objects = img_item.get("objects_3d", [])
        if not objects:
            continue

        # 1. 微小物体过滤
        filtered = []
        for obj in objects:
            bbox = obj["bbox"]   # COCO: [x_min, y_min, width, height]
            if img_h and img_w:
                x_min, y_min, w, h = bbox
                area = w * h
                if area / (img_h * img_w) < tiny_area_thresh:
                    continue
            filtered.append(obj)

        # 2. 按置信度排序（如果提供了 conf_key）
        if conf_key and conf_key in filtered[0]:
            filtered.sort(key=lambda x: x.get(conf_key, 0.0), reverse=True)

        # 3. 数量截断
        if len(filtered) > max_boxes:
            filtered = filtered[:max_boxes]
        # 如果过滤后数量小于 min_boxes，保留全部（不丢弃）

        img_item["objects_3d"] = filtered
        total_after += len(filtered)

    if verbose:
        print(f"[Caption Filter] Before: {total_before} objects | After: {total_after} objects")
        print(f"[Caption Filter] Removed {total_before - total_after} objects")

    # 保存输出
    save_path = output_json_path if output_json_path else input_json_path
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    if verbose:
        print(f"[Caption Filter] Saved filtered data to {save_path}")

    return data

# ===================== 使用示例 =====================
if __name__ == "__main__":
    filter_captions_by_bbox(
        input_json_path="./grit_captions_full.json",
        image_dir="../unc_train/",
        output_json_path="./grit_full.json",  # 输出新文件，保留原始备份
        min_boxes=10,
        max_boxes=36,
        tiny_area_thresh=0.015625,   # 面积小于 1.56% 的去掉
        conf_key="conf",             # 如果 JSON 里没有这个字段，改为 None
        verbose=True
    )
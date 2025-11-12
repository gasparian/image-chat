from typing import List, Dict, Tuple


def compute_containment_ratio(box1: list[float], box2: list[float]) -> Tuple[float, float]:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0, 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    if box1_area == 0 or box2_area == 0:
        return 0.0, 0.0

    containment_ratio_box1 = inter_area / box1_area
    containment_ratio_box2 = inter_area / box2_area

    return containment_ratio_box1, containment_ratio_box2


def merge_contained_boxes(
    detections: List[Dict],
    containment_threshold: float = 0.85
) -> List[Dict]:
    if not detections or len(detections) <= 1:
        return detections

    def box_area(det):
        x1, y1, x2, y2 = det["bbox"]
        return (x2 - x1) * (y2 - y1)

    sorted_dets = sorted(detections, key=box_area, reverse=True)

    merged_into = {}

    for i, det1 in enumerate(sorted_dets):
        if i in merged_into:
            continue

        bbox1 = det1["bbox"]

        for j, det2 in enumerate(sorted_dets):
            if i == j or j in merged_into:
                continue

            bbox2 = det2["bbox"]

            containment1, containment2 = compute_containment_ratio(bbox1, bbox2)

            if containment2 >= containment_threshold:
                merged_into[j] = i
                det1["score"] = max(det1["score"], det2["score"])

    result = []
    for i, det in enumerate(sorted_dets):
        if i not in merged_into:
            result.append(det)

    return result


def iou(box1: list[float], box2: list[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def soft_nms(
    boxes: List[Dict],
    Nt: float = 0.5,
    score_thresh: float = 0.1,
    sigma: float = 0.5,
    label_aware: bool = True
) -> List[Dict]:
    if not boxes:
        return []

    boxes_copy = []
    for b in boxes:
        bbox = b.get("bbox_xyxy", b.get("bbox"))
        boxes_copy.append({
            "bbox": bbox,
            "score": b["score"],
            "label": b.get("label", "")
        })

    boxes_copy.sort(key=lambda x: x["score"], reverse=True)

    kept = []
    while boxes_copy:
        current = boxes_copy.pop(0)
        if current["score"] < score_thresh:
            break

        kept.append(current)

        for other in boxes_copy:
            if label_aware and current["label"] and other["label"]:
                if current["label"].lower().strip() != other["label"].lower().strip():
                    continue

            overlap = iou(current["bbox"], other["bbox"])
            if overlap > Nt:
                other["score"] *= (1 - overlap)

        boxes_copy.sort(key=lambda x: x["score"], reverse=True)

    return [{"bbox_xyxy": b["bbox"], "score": b["score"], "label": b["label"]} for b in kept]

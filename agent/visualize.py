import os
import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Union

from agent.structures import DetectionResult, ChatAgentOutput
from agent.logging_config import get_logger

logger = get_logger(__name__)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def load_img(p: str) -> Image.Image:
    return Image.open(p).convert("RGB")


def visualize_and_save(
    image_path: str,
    result: Union[dict, ChatAgentOutput],
    output_dir: str = ".local"
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image = load_img(image_path)

    if isinstance(result, ChatAgentOutput):
        detections = result.detections or []
    else:
        detections = result.get("detections", [])

    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FFA500", "#800080", "#008000", "#000080", "#FF1493", "#00CED1"
    ]

    for det in detections:
        if isinstance(det, DetectionResult):
            class_id = det.class_id
            polygon_mask = det.polygon_mask
        else:
            class_id = det["class_id"]
            polygon_mask = det.get("polygon_mask")

        color_hex = colors[class_id % len(colors)]
        rgb = hex_to_rgb(color_hex)

        if polygon_mask:
            polygon_coords = [tuple(point) for point in polygon_mask]

            overlay_draw.polygon(polygon_coords, fill=(*rgb, 80))
            overlay_draw.line(polygon_coords + [polygon_coords[0]], fill=(*rgb, 180), width=2)

    image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    image = image.convert('RGB')

    draw = ImageDraw.Draw(image)

    for det in detections:
        if isinstance(det, DetectionResult):
            bbox = det.bbox
            label = det.label
            score = det.score
            class_id = det.class_id
        else:
            bbox = det["bbox"]
            label = det["label"]
            score = det["score"]
            class_id = det["class_id"]

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = colors[class_id % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), text, fill="white", font=font)

    base_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{base_name}_detection.jpg")
    image.save(output_path)
    return output_path


def visualize_detections(image_path: str, detections_json: str, output_path: str):
    with open(detections_json) as f:
        data = json.load(f)

    output_dir = os.path.dirname(output_path) or "."
    result_path = visualize_and_save(image_path, data, output_dir)

    if result_path != output_path:
        os.rename(result_path, output_path)
        result_path = output_path

    logger.info("Visualization saved", output_path=result_path)

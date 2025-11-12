"""Visualization utilities for drawing detections on images."""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict


def draw_detections_on_image(
    image: Image.Image,
    detections: List[Dict],
    class_vocab: List[str]
) -> Image.Image:
    """Draw detection bboxes and transparent polygons with labels on image.

    Args:
        image: PIL Image
        detections: List of detection dictionaries with bbox and/or polygon data
        class_vocab: List of class names

    Returns:
        Image with detections drawn
    """
    # Create a copy to draw on
    img_draw = image.copy()

    # Create a transparent overlay for filled polygons
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Main drawing layer
    draw = ImageDraw.Draw(img_draw)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    # Color palette for different classes (RGB)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 165, 0), (128, 0, 128), (0, 128, 0), (255, 192, 203)
    ]

    for det in detections:
        label = det["label"]
        score = det["score"]

        # Get color for this class
        class_id = det.get("class_id", 0)
        color_rgb = colors[class_id % len(colors)]
        color_rgba = color_rgb + (80,)  # Add alpha for transparency (80/255 ≈ 30%)

        # Draw polygon if available
        polygon = det.get("polygon_mask") or det.get("polygon") or det.get("segmentation")

        if polygon:
            # Polygon format: [(x1, y1), (x2, y2), ...] or [x1, y1, x2, y2, ...]
            if isinstance(polygon[0], (list, tuple)):
                # Already in [(x, y), ...] format
                polygon_points = [(float(x), float(y)) for x, y in polygon]
            else:
                # Flat list [x1, y1, x2, y2, ...] format
                polygon_points = [(float(polygon[i]), float(polygon[i+1]))
                                 for i in range(0, len(polygon), 2)]

            # Draw filled polygon on overlay with transparency
            overlay_draw.polygon(polygon_points, fill=color_rgba)

            # Draw polygon outline on main image
            draw.polygon(polygon_points, outline=color_rgb, width=2)

            # Get bbox for label placement from polygon
            xs = [p[0] for p in polygon_points]
            ys = [p[1] for p in polygon_points]
            x1, y1 = min(xs), min(ys)
        else:
            # No polygon, just draw bbox
            x1, y1 = 0, 0

        # Always draw bounding box
        bbox = det["bbox"]
        bx1, by1, bx2, by2 = bbox
        draw.rectangle([bx1, by1, bx2, by2], outline=color_rgb, width=2)

        # Use bbox for label placement
        x1, y1 = bx1, by1

        # Draw label background
        label_text = f"{label}: {score:.2f}"
        bbox_label = draw.textbbox((x1, y1), label_text, font=font)
        draw.rectangle(bbox_label, fill=color_rgb)

        # Draw label text
        draw.text((x1, y1), label_text, fill="white", font=font)

    # Convert main image to RGBA and composite with overlay
    if img_draw.mode != 'RGBA':
        img_draw = img_draw.convert('RGBA')

    # Composite overlay onto main image
    img_draw = Image.alpha_composite(img_draw, overlay)

    # Convert back to RGB for saving
    img_draw = img_draw.convert('RGB')

    return img_draw

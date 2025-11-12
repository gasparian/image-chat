"""Export utilities for detection results (COCO JSON, ZIP downloads)."""

import json
import uuid
import zipfile
import datetime
from pathlib import Path
from PIL import Image
from typing import List, Dict

from .config import DOWNLOADS_DIR
from .visualization import draw_detections_on_image


def create_coco_json(detections: List[Dict], class_vocab: List[str], image_info: Dict) -> Dict:
    """Create COCO format JSON from detections.

    Args:
        detections: List of detection dictionaries
        class_vocab: List of class names
        image_info: Dictionary with image metadata (width, height, filename)

    Returns:
        COCO format dictionary
    """
    coco_data = {
        "info": {
            "description": "Object detection results",
            "date_created": datetime.datetime.now().isoformat(),
            "version": "1.0"
        },
        "images": [{
            "id": 1,
            "width": image_info["width"],
            "height": image_info["height"],
            "file_name": image_info["filename"]
        }],
        "annotations": [],
        "categories": []
    }

    # Create categories from class vocab
    for idx, class_name in enumerate(class_vocab):
        coco_data["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "object"
        })

    # Create annotations from detections
    for ann_id, det in enumerate(detections, start=1):
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        # Convert to COCO format [x, y, width, height]
        coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        annotation = {
            "id": ann_id,
            "image_id": 1,
            "category_id": det["class_id"],
            "bbox": coco_bbox,
            "area": coco_bbox[2] * coco_bbox[3],
            "score": det["score"],
            "iscrowd": 0
        }

        # Add segmentation if available
        polygon = det.get("polygon_mask") or det.get("polygon") or det.get("segmentation")
        if polygon:
            # Convert to flat list format if needed
            if isinstance(polygon[0], (list, tuple)):
                # [[x1, y1], [x2, y2], ...] -> [x1, y1, x2, y2, ...]
                flat_polygon = [coord for point in polygon for coord in point]
            else:
                flat_polygon = polygon

            annotation["segmentation"] = [flat_polygon]

        coco_data["annotations"].append(annotation)

    return coco_data


def create_download_zip(image: Image.Image, detections: List[Dict], class_vocab: List[str]) -> str:
    """Create a zip file with original image, annotated image, and COCO JSON.

    Args:
        image: PIL Image
        detections: List of detections
        class_vocab: List of class names

    Returns:
        Path to the created zip file
    """
    # Create temp directory for results
    temp_dir = Path(DOWNLOADS_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique ID for this download
    download_id = uuid.uuid4().hex[:8]
    zip_path = temp_dir / f"detection_results_{download_id}.zip"

    # Draw detections on image
    annotated_image = draw_detections_on_image(image, detections, class_vocab)

    # Create zip file
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add original image
        original_filename = f"original_{download_id}.jpg"
        original_path = temp_dir / original_filename
        image.save(original_path)
        zipf.write(original_path, f"results/{original_filename}")

        # Add annotated image
        image_filename = f"annotated_{download_id}.jpg"
        image_path = temp_dir / image_filename
        annotated_image.save(image_path)
        zipf.write(image_path, f"results/{image_filename}")

        # Create and add COCO JSON
        image_info = {
            "width": image.width,
            "height": image.height,
            "filename": image_filename
        }
        coco_data = create_coco_json(detections, class_vocab, image_info)

        json_path = temp_dir / f"annotations_{download_id}.json"
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        zipf.write(json_path, "results/annotations.json")

        # Cleanup temp files
        original_path.unlink()
        image_path.unlink()
        json_path.unlink()

    return str(zip_path)

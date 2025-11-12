from PIL import Image
from typing import Tuple, Optional


def resize_image_keep_aspect(
    image: Image.Image,
    target_long_side: int
) -> Tuple[Image.Image, float]:
    if target_long_side <= 0:
        return image, 1.0

    orig_width, orig_height = image.size

    if orig_width >= orig_height:
        if orig_width == target_long_side:
            return image, 1.0

        scale_factor = target_long_side / orig_width
        new_width = target_long_side
        new_height = int(orig_height * scale_factor)
    else:
        if orig_height == target_long_side:
            return image, 1.0

        scale_factor = target_long_side / orig_height
        new_height = target_long_side
        new_width = int(orig_width * scale_factor)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image, scale_factor


def scale_bbox(bbox: list[float], scale_factor: float) -> list[float]:
    return [coord * scale_factor for coord in bbox]


def scale_polygon(polygon: list[list[float]], scale_factor: float) -> list[list[float]]:
    return [[x * scale_factor, y * scale_factor] for x, y in polygon]


def prepare_image_for_inference(
    image: Image.Image,
    target_long_side: Optional[int] = None
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    original_size = image.size

    if target_long_side is None or target_long_side <= 0:
        return image, 1.0, original_size

    resized_image, scale_factor = resize_image_keep_aspect(image, target_long_side)

    return resized_image, scale_factor, original_size

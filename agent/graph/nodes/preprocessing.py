from PIL import Image
from PIL.ExifTags import TAGS

from agent.config import CFG
from agent.callbacks import track_step
from agent.image_utils import prepare_image_for_inference
from agent.logging_config import get_logger
from agent.graph.state import ChatAgentState, ImageMetadata

logger = get_logger(__name__)


def extract_image_metadata(image: Image.Image) -> ImageMetadata:
    width, height = image.size
    mode = image.mode

    channel_map = {
        'L': 1,
        'RGB': 3,
        'RGBA': 4,
        'CMYK': 4,
        'YCbCr': 3,
        'LAB': 3,
        'HSV': 3,
        'I': 1,
        'F': 1,
    }
    channels = channel_map.get(mode, len(mode) if isinstance(mode, str) else 3)

    metadata: ImageMetadata = {
        "width": width,
        "height": height,
        "channels": channels,
        "mode": mode,
        "format": image.format,
        "aspect_ratio": width / height if height > 0 else 0.0,
        "total_pixels": width * height,
    }

    exif_data = {}
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = str(value)
                exif_data[str(tag_name)] = value

            metadata["exif"] = exif_data
    except Exception as e:
        logger.warning("Could not extract EXIF data", error=str(e))

    return metadata


def preprocess_image_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("preprocess_image"):
        original_image = state.get("image")

        if original_image is None:
            logger.info("No image provided, skipping preprocessing")
            return state

        target_size = CFG.global_resize_long_side

        image_metadata = extract_image_metadata(original_image)

        processed_image, scale_factor, original_size = prepare_image_for_inference(
            original_image,
            target_long_side=target_size
        )

        if scale_factor != 1.0:
            logger.info(
                "Image resized",
                original_size=original_size,
                new_size=processed_image.size,
                scale_factor=round(scale_factor, 3)
            )
        else:
            logger.info("Image size unchanged", original_size=original_size)

        logger.info(
            "Image metadata extracted",
            width=image_metadata['width'],
            height=image_metadata['height'],
            channels=image_metadata['channels'],
            mode=image_metadata['mode']
        )
        if "exif" in image_metadata and image_metadata["exif"]:
            logger.info(
                "EXIF data available",
                tag_count=len(image_metadata['exif']),
                sample_tags=list(image_metadata["exif"].keys())[:5]
            )

    return {
        **state,
        "image": processed_image,
        "scale_factor": scale_factor,
        "original_image_size": original_size,
        "image_metadata": image_metadata,
    }

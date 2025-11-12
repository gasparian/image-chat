import base64
import io
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from detection_mcp_server.config import CFG
from detection_mcp_server.detectors import get_registry
from detection_mcp_server.models import (
    DetectionRequest,
    DetectionResponse,
    Detection,
    HealthResponse,
    SegmentationRequest,
    SegmentationResponse,
    SegmentationMask,
    Polygon,
    CaptionRequest,
    CaptionResponse
)
from agent.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Detection MCP Server starting")

    logger.info(
        "Device configured",
        device=CFG.device,
        use_bfloat16=CFG.use_bfloat16
    )

    app.state.registry = get_registry(
        device=CFG.device,
        use_bfloat16=CFG.use_bfloat16
    )
    model_count = len(app.state.registry.get_available_model_names())
    logger.info("Model registry initialized", model_count=model_count)

    app.state.registry.load_all_models()

    yield

    logger.info("Detection MCP Server shutting down")


app = FastAPI(
    title="Detection MCP Server",
    description="HTTP-based MCP server for object detection models",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root() -> dict:
    return {
        "service": "Detection MCP Server",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "segment": "/segment (POST)",
            "caption": "/caption (POST)",
            "memory": {
                "clear_cache": "/memory/clear_cache (POST)",
            }
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    registry = app.state.registry
    return HealthResponse(
        status="ok",
        models_available=len(registry.get_available_model_names())
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest) -> DetectionResponse:
    """Detect objects using Grounding DINO."""
    MODEL_NAME = "grounding-dino-tiny"

    registry = app.state.registry
    model = registry.get_model(MODEL_NAME)

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model '{MODEL_NAME}' not available"
        )

    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(e)}"
        )

    try:
        detections = model.detect(
            image=image,
            phrases=request.phrases,
            threshold=request.threshold
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error("Detection failed", error=str(e), traceback=error_traceback)
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

    detection_objects = [
        Detection(
            bbox=det["bbox"],
            score=det["score"],
            label=det["label"]
        )
        for det in detections
    ]

    return DetectionResponse(
        detections=detection_objects,
        model=MODEL_NAME,
        num_detections=len(detection_objects)
    )


@app.post("/segment", response_model=SegmentationResponse)
async def segment(request: SegmentationRequest) -> SegmentationResponse:
    """Segment objects from bounding boxes using SAM 2.1."""
    MODEL_NAME = "sam2.1-hiera-base-plus"

    registry = app.state.registry
    model = registry.get_model(MODEL_NAME)

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model '{MODEL_NAME}' not available"
        )

    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(e)}"
        )

    try:
        # Import SAM2Segmenter to call segment_from_boxes
        from detection_mcp_server.detectors.sam2 import SAM2Segmenter

        if not isinstance(model, SAM2Segmenter):
            raise HTTPException(
                status_code=500,
                detail=f"Model '{MODEL_NAME}' is not a segmentation model"
            )

        segmentation_results = model.segment_from_boxes(
            image=image,
            bboxes=request.bboxes
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error("Segmentation failed", error=str(e), traceback=error_traceback)
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

    # Convert results to response format
    segmentation_masks = [
        SegmentationMask(
            bbox=result["bbox"],
            polygon=Polygon(points=result["polygon"]),
            score=result["score"]
        )
        for result in segmentation_results
    ]

    return SegmentationResponse(
        masks=segmentation_masks,
        model=MODEL_NAME,
        num_masks=len(segmentation_masks)
    )


@app.post("/caption", response_model=CaptionResponse)
async def caption(request: CaptionRequest) -> CaptionResponse:
    """Generate caption for an image using Florence-2."""
    MODEL_NAME = "florence-2-large"

    registry = app.state.registry
    model = registry.get_model(MODEL_NAME)

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model '{MODEL_NAME}' not available"
        )

    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(e)}"
        )

    try:
        caption_text = model.caption(
            image=image,
            task=request.task
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error("Captioning failed", error=str(e), traceback=error_traceback)
        raise HTTPException(
            status_code=500,
            detail=f"Captioning failed: {str(e)}"
        )

    return CaptionResponse(
        caption=caption_text,
        model=MODEL_NAME
    )


@app.post("/memory/clear_cache")
async def clear_memory_cache() -> dict:
    registry = app.state.registry
    registry._clear_device_cache()
    return {
        "status": "success",
        "message": "Memory cache cleared",
        "device": registry.device
    }


def run_server() -> None:
    uvicorn.run(
        "detection_mcp_server.main:app",
        host=CFG.host,
        port=CFG.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()

import json
import time
from typing import List
from pydantic import BaseModel, Field

from agent.config import CFG
from agent.ensemble import merge_contained_boxes, soft_nms
from agent.structures import (
    DetectionResult, ChatAgentOutput, Provenance
)
from agent.callbacks import track_step
from agent.image_utils import scale_bbox, scale_polygon
from agent.mcp_client import MCPClientError
from agent.logging_config import get_logger
from agent.callbacks import log_tokens
from agent.structures import ChatMessage
from agent.conversation_utils import format_conversation_for_context
from agent.graph.state import ChatAgentState

logger = get_logger(__name__)


def bbox_from_polygon(polygon: List[List[float]]) -> List[float]:
    if not polygon:
        return [0.0, 0.0, 0.0, 0.0]

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    return [min(xs), min(ys), max(xs), max(ys)]


class QueryExpansion(BaseModel):
    queries: list[str] = Field(
        description="List of 1-4 short detection queries (noun phrases)",
        min_length=1,
        max_length=4
    )


SYSTEM = "You expand user tasks into compact detection phrases. Output JSON only."

INSTR = """Task: {task}

{context}

Generate up to {n_max} short noun or noun-phrase queries for object detection.
Use concrete items and affordances.
{context_instruction}

Output JSON format:
{{"queries": ["query1", "query2", ...]}}"""


def expand_queries(
    llm_client,
    task: str,
    conversation_history: List[ChatMessage] | None = None
) -> list[str]:
    context = ""
    context_instruction = ""
    if conversation_history:
        context = format_conversation_for_context(conversation_history, max_turns=3)
        if context:
            context_instruction = "Consider the conversation history to provide contextually relevant queries."
            logger.info("Using conversation context", message_count=len(conversation_history))

    prompt = INSTR.format(
        task=task,
        n_max=CFG.max_queries,
        context=context,
        context_instruction=context_instruction
    )

    try:
        start_time = time.time()

        rsp = llm_client.chat.completions.create(
            model=CFG.llm_model,
            temperature=CFG.llm_temperature,
            max_tokens=128,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        llm_duration = time.time() - start_time
        logger.info("LLM query expansion complete", duration_sec=round(llm_duration, 2))

        if hasattr(rsp, 'usage') and rsp.usage:
            log_tokens(rsp.usage.total_tokens)

        text = rsp.choices[0].message.content

        logger.debug("Raw LLM response", response=text)

        if not text or text.strip() == "":
            raise ValueError("Empty response from LLM")

        data = json.loads(text.strip())
        result = QueryExpansion.model_validate(data)

        queries = [q.strip() for q in result.queries if q.strip()][:CFG.max_queries]

        if not queries:
            raise ValueError("No valid queries generated")

        logger.debug("Parsed queries", query_count=len(queries), queries=queries)

        return queries

    except Exception as e:
        logger.warning("LLM query expansion failed, using fallback", error_type=type(e).__name__, error=str(e))
        return _fallback_extract_queries(task)


def _fallback_extract_queries(task: str) -> list[str]:
    words = task.lower().split()
    stop_words = {
        'detect', 'all', 'find', 'identify', 'the', 'a', 'an', 'some',
        'any', 'locate', 'search', 'for', 'show', 'me', 'in', 'this', 'image'
    }
    potential_objects = [w for w in words if w not in stop_words and len(w) > 2]

    if not potential_objects:
        return [task.strip()]

    queries = []

    queries.append(potential_objects[0])

    if len(potential_objects) > 1:
        queries.append(' '.join(potential_objects[:2]))

    if len(potential_objects) > 2:
        queries.append(' '.join(potential_objects[:3]))

    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    return unique_queries[:CFG.max_queries]


def plan_and_expand_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("plan_expand"):
        image = state.get("image")

        if image is None:
            logger.warning("No image available for detection")
            provenance = Provenance(steps=["preprocess_image", "route_intent", "plan_and_expand"])
            output = ChatAgentOutput(
                type="detection",
                detections=[],
                class_vocab=[],
                answer="Please upload an image first to detect objects.",
                provenance=provenance
            )
            return {
                **state,
                "final_output": output,
            }

        conversation_history = state.get("conversation_history", [])
        phrases = expand_queries(
            state["llm_client"],
            state["user_query"],
            conversation_history=conversation_history
        )
        label_vocab = sorted(set(p.strip() for p in phrases if p.strip()))
        logger.info("Query expanded", phrase_count=len(phrases), phrases=phrases)

    return {
        **state,
        "phrases": phrases,
        "label_vocab": label_vocab,
    }


def detect_grounding_dino_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("grounding_dino"):
        if state.get("final_output"):
            return state

        phrases = state["phrases"]
        image = state["image"]
        client = state["mcp_client"]

        phrases = [p.strip() for p in phrases if p.strip()]
        if not phrases:
            logger.info("No phrases to detect")
            return {**state, "grounding_dino_detections": []}

        try:
            raw_detections = client.detect(
                image=image,
                phrases=phrases,
                threshold=CFG.gd_box_thr
            )
            logger.info("Grounding DINO detection complete", detection_count=len(raw_detections))
        except MCPClientError as e:
            logger.error("MCP detection failed", error=str(e), mcp_server="http://127.0.0.1:8000")
            raise

    return {
        **state,
        "grounding_dino_detections": raw_detections,
    }


def ensemble_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("ensemble"):
        if state.get("final_output"):
            return state

        raw_dets = state["grounding_dino_detections"]

        filtered = []
        for d in raw_dets:
            x1, y1, x2, y2 = d["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area >= CFG.min_area and d["score"] >= CFG.gd_box_thr:
                filtered.append(d)

        logger.info("Detections filtered", filtered_count=len(filtered), raw_count=len(raw_dets))

        nms_input = [
            {"bbox_xyxy": d["bbox"], "score": d["score"], "label": d["label"]}
            for d in filtered
        ]

        after_nms = soft_nms(
            nms_input,
            Nt=CFG.nms_iou_threshold,
            score_thresh=CFG.nms_score_threshold
        )
        logger.info("NMS applied", detection_count=len(after_nms))

        formatted_for_merge = [
            {"bbox": d["bbox_xyxy"], "score": d["score"], "label": d["label"]}
            for d in after_nms
        ]

        merged_dets = merge_contained_boxes(
            formatted_for_merge,
            containment_threshold=CFG.containment_threshold
        )

        final_dets = [
            {"bbox_xyxy": d["bbox"], "score": d["score"], "label": d["label"]}
            for d in merged_dets
        ]

        logger.info("Detections merged", detection_count=len(final_dets))

    return {
        **state,
        "detections": final_dets,
    }


def sam_segmentation_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("sam_segmentation"):
        if state.get("final_output"):
            return state

        detections = state["detections"]
        image = state["image"]
        client = state["mcp_client"]

        if not detections:
            logger.info("No detections to segment")
            return {
                **state,
                "segmentation_masks": [],
            }

        bboxes = []
        for det in detections:
            bbox = det.get("bbox_xyxy") or det.get("bbox")
            bboxes.append(bbox)

        logger.info("Starting SAM segmentation", object_count=len(bboxes))

        try:
            masks = client.segment(image=image, bboxes=bboxes)
            logger.info("SAM segmentation complete", mask_count=len(masks))
        except MCPClientError as e:
            logger.error("MCP segmentation failed", error=str(e), mcp_server="http://127.0.0.1:8000")
            masks = []
        except Exception as e:
            logger.error("SAM segmentation failed", error=str(e))
            masks = []

    return {
        **state,
        "segmentation_masks": masks,
    }


def format_detection_output_node(state: ChatAgentState) -> ChatAgentState:
    with track_step("format_detection_output"):
        if state.get("final_output"):
            return state

        final_candidates = state["detections"]
        segmentation_masks = state.get("segmentation_masks", [])
        label_vocab = state["label_vocab"]
        scale_factor = state.get("scale_factor", 1.0)

        inverse_scale = 1.0 / scale_factor if scale_factor != 0 else 1.0

        if inverse_scale != 1.0:
            logger.info("Scaling results to original size", inverse_scale=round(inverse_scale, 3))

        bbox_to_polygon = {}
        for mask in segmentation_masks:
            bbox_key = tuple(mask["bbox"])
            bbox_to_polygon[bbox_key] = mask["polygon"]

        results: list[DetectionResult] = []
        for r in final_candidates:
            bbox = r.get("bbox_xyxy") or r.get("bbox")
            bbox_list = [float(v) for v in bbox]
            bbox_key = tuple(bbox_list)

            polygon_mask = None
            if bbox_key in bbox_to_polygon:
                polygon = bbox_to_polygon[bbox_key]
                scaled_polygon = scale_polygon(polygon, inverse_scale)
                polygon_mask = scaled_polygon
                bbox_list = bbox_from_polygon(polygon)

            scaled_bbox = scale_bbox(bbox_list, inverse_scale)

            detection_result = DetectionResult(
                bbox=scaled_bbox,
                class_id=int(label_vocab.index(r["label"])),
                score=float(r["score"]),
                label=r["label"],
                polygon_mask=polygon_mask
            )

            results.append(detection_result)

        steps = ["preprocess_image", "route_intent", "expand_queries", "grounding_dino",
                 "ensemble_nms_merge"]
        if segmentation_masks:
            steps.append("sam_segmentation")

        provenance = Provenance(steps=steps)
        output = ChatAgentOutput(
            type="detection",
            detections=results,
            class_vocab=label_vocab,
            provenance=provenance
        )

    return {
        **state,
        "final_output": output,
    }

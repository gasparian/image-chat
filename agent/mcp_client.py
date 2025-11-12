import base64
import io
from urllib.parse import urljoin

import requests
from PIL import Image


class MCPClientError(Exception):
    pass


class MCPClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: int = 90
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def health_check(self) -> dict:
        url = urljoin(self.base_url, "/health")
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise MCPClientError(f"Health check failed: {str(e)}")

    def detect(
        self,
        image: Image.Image,
        phrases: list[str],
        threshold: float = 0.25
    ) -> list[dict]:
        url = urljoin(self.base_url, "/detect")

        image_base64 = self._image_to_base64(image)

        payload = {
            "image_base64": image_base64,
            "phrases": phrases,
            "threshold": threshold
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            detections = data.get("detections", [])

            results = []
            for det in detections:
                results.append({
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "label": det["label"]
                })

            return results

        except requests.RequestException as e:
            raise MCPClientError(f"Detection request failed: {str(e)}")

    def segment(
        self,
        image: Image.Image,
        bboxes: list[list[float]]
    ) -> list[dict]:
        url = urljoin(self.base_url, "/segment")

        image_base64 = self._image_to_base64(image)

        payload = {
            "image_base64": image_base64,
            "bboxes": bboxes
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            masks = data.get("masks", [])

            results = []
            for mask in masks:
                results.append({
                    "bbox": mask["bbox"],
                    "polygon": mask["polygon"]["points"],
                    "score": mask["score"]
                })

            return results

        except requests.RequestException as e:
            raise MCPClientError(f"Segmentation request failed: {str(e)}")

    def caption(
        self,
        image: Image.Image,
        task: str = "<CAPTION>"
    ) -> str:
        """Generate caption for an image using Florence-2.

        Args:
            image: PIL Image to caption
            task: Caption task type - options:
                  "<CAPTION>" - brief caption
                  "<DETAILED_CAPTION>" - more detailed caption
                  "<MORE_DETAILED_CAPTION>" - very detailed caption

        Returns:
            Generated caption as string
        """
        url = urljoin(self.base_url, "/caption")

        image_base64 = self._image_to_base64(image)

        payload = {
            "image_base64": image_base64,
            "task": task
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            caption = data.get("caption", "")
            return caption

        except requests.RequestException as e:
            raise MCPClientError(f"Captioning request failed: {str(e)}")


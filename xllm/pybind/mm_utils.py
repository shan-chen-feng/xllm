# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from xllm_export import MMData


def _bytes_to_data_url(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image;base64,{encoded}"


def _pil_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    fmt = image.format or "PNG"
    image.save(buf, format=fmt)
    return _bytes_to_data_url(buf.getvalue())


def normalize_vllm_style_inputs(
    prompts: Any,
) -> Tuple[List[str], Optional[List[MMData]], Optional[List[List[str]]]]:
    if isinstance(prompts, dict):
        requests = [prompts]
        return _parse_vllm_style_requests(requests)
    if isinstance(prompts, list) and prompts and all(isinstance(x, dict) for x in prompts):
        return _parse_vllm_style_requests(prompts)

    raise TypeError(
        "VLM-style inputs must be dict/List[dict] with key 'prompt', e.g. "
        "{'prompt': '...', 'multi_modal_data': {'image': image}}"
    )


def _parse_vllm_style_requests(
    requests: List[Dict[str, Any]],
) -> Tuple[List[str], Optional[List[MMData]], Optional[List[List[str]]]]:
    prompts: List[str] = []
    mm_datas: List[MMData] = []
    image_urls: List[List[str]] = []
    use_mm_data: Optional[bool] = None

    for req in requests:
        if "prompt" not in req:
            raise ValueError("Each request dict must contain key 'prompt'")

        prompt = req["prompt"]
        if not isinstance(prompt, str):
            raise TypeError("request['prompt'] must be a string")
        prompts.append(prompt)

        if "multi_modal_data" not in req:
            if use_mm_data is True:
                raise TypeError("Cannot mix MMData and empty multi_modal_data in one batch")
            use_mm_data = False
            image_urls.append([])
            continue

        payload = req["multi_modal_data"]
        if isinstance(payload, MMData):
            if use_mm_data is False:
                raise TypeError("Cannot mix MMData and image inputs in one batch")
            use_mm_data = True
            mm_datas.append(payload)
        else:
            if use_mm_data is True:
                raise TypeError("Cannot mix MMData and image inputs in one batch")
            use_mm_data = False
            image_urls.append(_to_image_urls(payload))

    if use_mm_data:
        return prompts, mm_datas, None
    return prompts, None, image_urls


def _to_image_urls(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        raise TypeError("multi_modal_data must be dict or MMData")

    if "image" in payload:
        images = payload["image"]
        return _normalize_images(images)
    if "video" in payload:
        raise NotImplementedError("video multi_modal_data is not supported yet")
    if "audio" in payload:
        raise NotImplementedError("audio multi_modal_data is not supported yet")

    raise ValueError(
        "Unsupported multi_modal_data format. Expected {'image': ...} or MMData."
    )


def _normalize_images(images: Any) -> List[str]:
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise ValueError("multi_modal_data['image'] cannot be empty")
        return [_to_image_url(img) for img in images]
    return [_to_image_url(images)]


def _to_image_url(image: Any) -> str:
    if isinstance(image, str):
        return image
    if isinstance(image, Image.Image):
        return _pil_to_data_url(image.convert("RGB"))
    if isinstance(image, (bytes, bytearray)):
        return _bytes_to_data_url(bytes(image))

    raise TypeError(
        "image must be image path/url string, PIL.Image, bytes, "
        "or a list of these"
    )


def dispatch_vlm_batch(master, prompts, mm_datas, image_urls,
                       request_params_list, callback) -> None:
    """Route a normalized vLLM-style batch to the right VLMMaster entrypoint.

    Two mutually-exclusive input shapes come out of
    ``normalize_vllm_style_inputs``:

    * ``image_urls`` set  -> images given as PIL/path/bytes (converted to urls).
      Uses ``handle_batch_request_with_mm_urls``: the prompt is used VERBATIM
      (caller supplies ``<|vision_start|><|image_pad|><|vision_end|>`` blocks,
      one per image) and NO chat template is applied. This matches vLLM offline
      ``LLM.generate`` semantics.
    * ``mm_datas`` set    -> caller passed pre-built ``MMData`` objects.
      Uses ``handle_batch_request`` (also verbatim prompt).

    Kept as a standalone function so both LLM.generate and VLM.generate share
    one dispatch path instead of duplicating branch logic.
    """
    if image_urls is not None:
        master.handle_batch_request_with_mm_urls(
            prompts, image_urls, request_params_list, callback
        )
    else:
        master.handle_batch_request(
            prompts, mm_datas, request_params_list, callback
        )

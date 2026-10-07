"""Run artwork-agnostic SAM 3 mask proposals or optional concept segmentation.

SAM 3, PyTorch, OpenCV, and an approved SAM 3 checkpoint must be installed
separately. This script intentionally keeps those heavyweight imports lazy so
its output-format helpers can be tested in the lightweight prototype env.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Prompt:
    text: str
    label_ko: str
    tactile_priority: int


def parse_prompts(values: list[str] | None) -> list[Prompt]:
    if not values:
        return []
    prompts = []
    for value in values:
        parts = value.split("|", 2)
        if len(parts) != 3:
            raise ValueError("프롬프트는 '영문 프롬프트|한국어 레이블|우선순위' 형식이어야 합니다")
        text, label, priority = parts
        prompts.append(Prompt(text.strip(), label.strip(), int(priority)))
    return prompts


def mask_to_polygon(mask: np.ndarray, cv2_module: Any, width: int, height: int,
                    epsilon_ratio: float = 0.002) -> list[list[float]]:
    """Return a normalized simplified outer contour for artwork.json."""
    binary = (np.asarray(mask).astype(bool).astype(np.uint8) * 255)
    contours, _ = cv2_module.findContours(binary, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("마스크에서 외곽선을 찾지 못했습니다")
    contour = max(contours, key=cv2_module.contourArea)
    perimeter = cv2_module.arcLength(contour, True)
    approx = cv2_module.approxPolyDP(contour, max(0.5, perimeter * epsilon_ratio), True)
    points = approx.reshape(-1, 2).tolist()
    if len(points) < 3:
        points = cv2_module.boxPoints(cv2_module.minAreaRect(contour)).round().astype(int).tolist()
    return [[round(min(max(x / width, 0.0), 1.0), 6), round(min(max(y / height, 0.0), 1.0), 6)] for x, y in points]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()


def _make_overlay(image: Image.Image, regions: list[dict[str, Any]], mask_root: Path,
                  output_path: Path) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    colors = [(255, 50, 50), (30, 130, 255), (30, 190, 100), (230, 150, 20),
              (170, 70, 220), (20, 190, 190), (240, 80, 160), (180, 180, 20)]
    for index, region in enumerate(regions):
        mask = Image.open(mask_root / region["mask_path"]).convert("L")
        color = colors[index % len(colors)]
        tint = Image.new("RGBA", base.size, (*color, 88))
        overlay.alpha_composite(Image.composite(tint, Image.new("RGBA", base.size), mask))
        # The polygon is drawn as a crisp, legible outline over the translucent mask.
        draw = ImageDraw.Draw(overlay)
        pts = [(round(x * image.width), round(y * image.height)) for x, y in region["polygon"]]
        draw.line(pts + [pts[0]], fill=(*color, 255), width=max(2, image.width // 350), joint="curve")
        if pts:
            cx = sum(x for x, _ in pts) // len(pts)
            cy = sum(y for _, y in pts) // len(pts)
            draw.text((cx, cy), region["region_id"], fill=(0, 0, 0, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path, quality=95)


def _concept_masks(image: Image.Image, prompts: list[Prompt], model_id: str) -> list[dict[str, Any]]:
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise RuntimeError(
            "SAM 3 실행 의존성이 없습니다. 별도 Python 3.12+ CUDA 환경에 PyTorch, "
            "공식 sam3 패키지와 opencv-python-headless를 설치하세요."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 공식 추론 환경에 CUDA GPU가 보이지 않습니다. CUDA 지원 PyTorch와 GPU 환경을 확인하세요.")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    inference_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if hasattr(torch, "autocast") and hasattr(torch, "bfloat16")
        else nullcontext()
    )
    with inference_context:
        state = processor.set_image(image)

    raw_candidates: list[dict[str, Any]] = []

    for prompt in prompts:
        with inference_context:
            result = processor.set_text_prompt(state=state, prompt=prompt.text)
        masks = _as_numpy(result["masks"])
        boxes = _as_numpy(result["boxes"])
        scores = _as_numpy(result["scores"]).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        for idx, mask in enumerate(masks):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            binary = np.squeeze(mask).astype(bool)
            area = int(binary.sum())
            raw_candidates.append({"prompt": prompt, "score": score, "area": area,
                                   "mask": binary, "box": boxes[idx].tolist() if idx < len(boxes) else None})
    return raw_candidates


def _automatic_masks(image: Image.Image, model_id: str, points_per_batch: int) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("자동 마스크 생성에는 transformers, torch가 필요합니다. `pip install -U transformers accelerate`로 설치하세요.") from exc
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("mask-generation", model=model_id, device=device)
    result = generator(image, points_per_batch=points_per_batch)
    masks = result.get("masks", [])
    scores = result.get("scores", [])
    candidates = []
    for i, value in enumerate(masks):
        binary = np.asarray(value.convert("L") if isinstance(value, Image.Image) else value).squeeze() > 0
        candidates.append({"prompt": None, "score": float(scores[i]) if i < len(scores) else None,
                           "area": int(binary.sum()), "mask": binary, "box": None})
    return candidates


def run(image_path: Path, output_path: Path, prompts: list[Prompt],
        score_threshold: float, min_area_pixels: int, model_id: str,
        mode: str = "auto", points_per_batch: int = 64,
        title: str | None = None, artist: str | None = None) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("마스크 윤곽선 변환에 opencv-python-headless가 필요합니다.") from exc
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if mode == "auto":
        raw_candidates = _automatic_masks(image, model_id, points_per_batch)
    elif mode == "concept":
        if not prompts:
            raise ValueError("concept 모드에는 --prompt 또는 --prompts-json이 필요합니다")
        raw_candidates = _concept_masks(image, prompts, model_id)
    else:
        raise ValueError(f"지원하지 않는 모드: {mode}")
    mask_root = output_path.parent / "segmentation" / image_path.stem / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    regions: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        score = candidate["score"]
        binary = candidate["mask"]
        if (score is not None and score < score_threshold) or candidate["area"] < min_area_pixels:
            continue
        if binary.shape != (height, width):
            binary = cv2.resize(binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        polygon = mask_to_polygon(binary, cv2, width, height)
        prompt = candidate["prompt"]

        region_id = f"R{len(regions) + 1:02d}"
        label = prompt.label_ko if prompt else f"미분류 영역 {region_id}"
        mask_filename = f"{region_id}_{_safe_name(prompt.text if prompt else 'proposal')}.png"
        Image.fromarray(binary.astype(np.uint8) * 255, mode="L").save(mask_root / mask_filename)
        regions.append({
            "region_id": region_id,
            "label": label,
            "objects": [prompt.label_ko] if prompt else [],
            "polygon": polygon,
            "mask_path": f"segmentation/{image_path.stem}/masks/{mask_filename}",
            "source_prompt": prompt.text if prompt else "automatic-mask-generation",
            "confidence": round(score, 6) if score is not None else None,
            "area_pixels": candidate["area"],
            "bounding_box_xyxy": candidate["box"],
            "tactile_priority": prompt.tactile_priority if prompt else 0,
            "approval": "pending",
            "relations": [],
        })

    payload = {
        "schema_version": "0.2",
        "artwork": {
            "artwork_id": f"{image_path.stem}-001",
            "title": title or image_path.stem,
            "artist": artist,
            "image_path": image_path.as_posix(),
            "image_width": width,
            "image_height": height,
            "sources": [{"source_id": "user-provided-image", "text": "작품명과 작가 표기는 사용자 제공 맥락에 따름. 작품 버전·소장처·공식 작품 자료 확인 필요.", "approval": "unreviewed"}],
        },
        "segmentation": {
            "model": model_id,
            "mode": mode,
            "prompt_set": [{"text": p.text, "label": p.label_ko, "tactile_priority": p.tactile_priority} for p in prompts],
            "score_threshold": score_threshold,
            "min_area_pixels": min_area_pixels,
            "points_per_batch": points_per_batch if mode == "auto" else None,
            "status": "needs_curator_review",
        },
        "display": {"width": 60, "height": 40, "coordinate_origin": "top_left"},
        "regions": regions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_overlay(image, regions, output_path.parent, output_path.parent / f"{image_path.stem}_sam3_overlay.png")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 3으로 작품 영역 후보를 분할해 artwork.json으로 저장")
    parser.add_argument("--image", type=Path, default=Path("data/sunflower.jpeg"))
    parser.add_argument("--output", type=Path, default=Path("data/artwork.json"))
    parser.add_argument("--mode", choices=["auto", "concept"], default="auto", help="auto는 작품 일반 자동 후보 생성, concept는 개념 프롬프트 기반 분할")
    parser.add_argument("--prompt", action="append", help="영문 프롬프트|한국어 레이블|우선순위 (여러 번 지정 가능)")
    parser.add_argument("--prompts-json", type=Path, help="작품별 프롬프트 설정 JSON (concept 모드)")
    parser.add_argument("--title")
    parser.add_argument("--artist")
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--min-area-pixels", type=int, default=32)
    parser.add_argument("--model-id", default="facebook/sam3")
    args = parser.parse_args()
    try:
        prompt_values = args.prompt
        if args.prompts_json:
            config = json.loads(args.prompts_json.read_text(encoding="utf-8"))
            prompt_values = [f"{item['text']}|{item['label_ko']}|{item.get('tactile_priority', 0)}" for item in config["prompts"]]
        payload = run(args.image, args.output, parse_prompts(prompt_values), args.score_threshold,
                      args.min_area_pixels, args.model_id, args.mode, args.points_per_batch, args.title, args.artist)
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    print(f"작품 JSON: {args.output}")
    print(f"영역 후보: {len(payload['regions'])}개")
    print(f"오버레이: {args.output.parent / (args.image.stem + '_sam3_overlay.png')}")
    print("모든 영역은 approval='pending' 상태이므로 오버레이를 확인하고 검수해야 합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

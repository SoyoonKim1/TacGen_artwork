"""Run official SAM 3 image concept segmentation and export artwork.json.

SAM 3, PyTorch, OpenCV, and an approved SAM 3 checkpoint must be installed
separately. This script intentionally keeps those heavyweight imports lazy so
its output-format helpers can be tested in the lightweight prototype env.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
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


DEFAULT_PROMPTS = [
    Prompt("sunflower head", "해바라기 꽃송이", 5),
    Prompt("vase", "꽃병", 5),
    Prompt("plant stem", "줄기", 3),
    Prompt("plant leaf", "잎", 2),
]


def parse_prompts(values: list[str] | None) -> list[Prompt]:
    if not values:
        return DEFAULT_PROMPTS
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
        value = value.detach().cpu().numpy()
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


def run(image_path: Path, output_path: Path, prompts: list[Prompt],
        score_threshold: float, min_area_pixels: int, model_id: str) -> dict[str, Any]:
    try:
        import cv2
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
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    state = processor.set_image(image)

    mask_root = output_path.parent / "segmentation" / image_path.stem / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    regions: list[dict[str, Any]] = []
    raw_candidates: list[dict[str, Any]] = []

    for prompt in prompts:
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
            if score < score_threshold or area < min_area_pixels:
                continue
            if binary.shape != (height, width):
                binary = cv2.resize(binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            polygon = mask_to_polygon(binary, cv2, width, height)
            raw_candidates.append({"prompt": prompt, "score": score, "area": area,
                                   "mask": binary, "polygon": polygon,
                                   "box": boxes[idx].tolist() if idx < len(boxes) else None})

    # Keep prompt-level instances as independent review candidates. Overlap and
    # duplicate detections are deliberately left visible for curator review.
    for candidate in raw_candidates:
        prompt = candidate["prompt"]
        region_id = f"R{len(regions) + 1:02d}"
        mask_filename = f"{region_id}_{_safe_name(prompt.text)}.png"
        Image.fromarray(candidate["mask"].astype(np.uint8) * 255, mode="L").save(mask_root / mask_filename)
        regions.append({
            "region_id": region_id,
            "label": prompt.label_ko,
            "objects": [prompt.label_ko],
            "polygon": candidate["polygon"],
            "mask_path": f"segmentation/{image_path.stem}/masks/{mask_filename}",
            "source_prompt": prompt.text,
            "confidence": round(candidate["score"], 6),
            "area_pixels": candidate["area"],
            "bounding_box_xyxy": candidate["box"],
            "tactile_priority": prompt.tactile_priority,
            "approval": "pending",
            "relations": [],
        })

    payload = {
        "schema_version": "0.2",
        "artwork": {
            "artwork_id": f"{image_path.stem}-001",
            "title": "해바라기",
            "artist": "빈센트 반 고흐",
            "image_path": image_path.as_posix(),
            "image_width": width,
            "image_height": height,
            "sources": [{"source_id": "user-provided-image", "text": "작품명과 작가 표기는 사용자 제공 맥락에 따름. 작품 버전·소장처·공식 작품 자료 확인 필요.", "approval": "unreviewed"}],
        },
        "segmentation": {
            "model": model_id,
            "prompt_set": [{"text": p.text, "label": p.label_ko, "tactile_priority": p.tactile_priority} for p in prompts],
            "score_threshold": score_threshold,
            "min_area_pixels": min_area_pixels,
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
    parser.add_argument("--prompt", action="append", help="영문 프롬프트|한국어 레이블|우선순위 (여러 번 지정 가능)")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--min-area-pixels", type=int, default=32)
    parser.add_argument("--model-id", default="facebook/sam3")
    args = parser.parse_args()
    try:
        payload = run(args.image, args.output, parse_prompts(args.prompt), args.score_threshold,
                      args.min_area_pixels, args.model_id)
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

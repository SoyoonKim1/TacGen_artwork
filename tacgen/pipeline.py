"""Annotation-backed prototype pipeline for tactile artwork exploration."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataError(ValueError):
    """Invalid artwork or region data."""


@dataclass(frozen=True)
class PrototypeResult:
    artwork_id: str
    width: int
    height: int
    grid: list[list[str]]
    regions: list[dict[str, Any]]
    chunks: list[dict[str, Any]]


def load_artwork(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    validate_artwork(data)
    return data


def validate_artwork(data: dict[str, Any]) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("artwork"), dict):
        raise DataError("Top-level 'artwork' object is required")
    if not data.get("schema_version"):
        raise DataError("'schema_version' is required")
    artwork = data["artwork"]
    if not artwork.get("artwork_id") or not artwork.get("title"):
        raise DataError("artwork.artwork_id and artwork.title are required")
    display = data.get("display", {})
    for key in ("width", "height"):
        if not isinstance(display.get(key), int) or display[key] <= 0:
            raise DataError(f"display.{key} must be a positive integer")
    regions = data.get("regions")
    if not isinstance(regions, list) or not regions:
        raise DataError("At least one region is required")
    ids: set[str] = set()
    for region in regions:
        rid = region.get("region_id")
        if not rid or rid in ids:
            raise DataError(f"Missing or duplicate region_id: {rid!r}")
        ids.add(rid)
        polygon = region.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise DataError(f"Region {rid} needs a polygon with at least 3 points")
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2 or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in point):
                raise DataError(f"Region {rid} has invalid normalized point: {point!r}")
    for region in regions:
        for rel in region.get("relations", []):
            if rel.get("target_region_id") not in ids:
                raise DataError(f"Region {region['region_id']} relation references unknown region {rel.get('target_region_id')!r}")


def _inside(px: float, py: float, polygon: list[list[float]]) -> bool:
    """Ray casting with points on polygon edges treated as inside."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) - 1e-10 <= px <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= py <= max(y1, y2) + 1e-10:
            return True
        if (y1 > py) != (y2 > py):
            x_cross = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_cross:
                inside = not inside
    return inside


def rasterize(regions: list[dict[str, Any]], width: int, height: int) -> list[list[str]]:
    """Rasterize normalized polygon annotations to a region-ID display grid."""
    ordered = sorted(regions, key=lambda r: (-int(r.get("tactile_priority", 0)), r["region_id"]))
    grid = [["0" for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            nx, ny = (x + 0.5) / width, (y + 0.5) / height
            for region in ordered:
                if _inside(nx, ny, region["polygon"]):
                    grid[y][x] = region["region_id"]
                    break
    return grid


def semantic_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    artwork = data["artwork"]
    chunks: list[dict[str, Any]] = []
    for index, source in enumerate(artwork.get("sources", [])):
        body = source.get("text", "").strip()
        if body:
            chunks.append({"chunk_id": f"{artwork['artwork_id']}:context:{index+1}", "kind": "artwork_context", "text": body, "artwork_id": artwork["artwork_id"], "region_ids": [], "source_id": source.get("source_id"), "approval": source.get("approval", "unreviewed")})
    for region in data["regions"]:
        statements = [region.get("description", "").strip()]
        for relation in region.get("relations", []):
            statements.append(f"{region['region_id']} {relation['relation']} {relation['target_region_id']}")
        body = " ".join(statement for statement in statements if statement)
        if body:
            chunks.append({"chunk_id": f"{artwork['artwork_id']}:{region['region_id']}:description", "kind": "region_description", "text": body, "artwork_id": artwork["artwork_id"], "region_ids": [region["region_id"]], "source_id": region.get("source_id"), "approval": region.get("approval", "unreviewed")})
        sensory = region.get("sensory_profile")
        if sensory:
            chunks.append({"chunk_id": f"{artwork['artwork_id']}:{region['region_id']}:sensory", "kind": "sensory_profile", "text": json.dumps(sensory, ensure_ascii=False), "artwork_id": artwork["artwork_id"], "region_ids": [region["region_id"]], "source_id": sensory.get("source_id"), "approval": sensory.get("approval", "unreviewed")})
    return chunks


def build(data: dict[str, Any]) -> PrototypeResult:
    validate_artwork(data)
    width, height = data["display"]["width"], data["display"]["height"]
    return PrototypeResult(data["artwork"]["artwork_id"], width, height, rasterize(data["regions"], width, height), data["regions"], semantic_chunks(data))


def resolve_point(result: PrototypeResult, x: float, y: float) -> str | None:
    """Resolve normalized display coordinates to Region ID; background is None."""
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Coordinates must be normalized to [0, 1]")
    col = min(int(x * result.width), result.width - 1)
    row = min(int(y * result.height), result.height - 1)
    rid = result.grid[row][col]
    return None if rid == "0" else rid


def answer(result: PrototypeResult, region_id: str | None, question: str) -> dict[str, Any]:
    """Return region-scoped, extractive evidence; deliberately no invented LLM answer."""
    by_id = {r["region_id"]: r for r in result.regions}
    if region_id is not None and region_id not in by_id:
        raise ValueError(f"Unknown region_id: {region_id}")
    tokens = {t.lower() for t in re.findall(r"[\w가-힣]+", question) if len(t) > 1}
    candidates = [c for c in result.chunks if c["approval"] == "approved" and (region_id is None or not c["region_ids"] or region_id in c["region_ids"])]
    def score(chunk: dict[str, Any]) -> tuple[int, int, str]:
        words = {t.lower() for t in re.findall(r"[\w가-힣]+", chunk["text"]) if len(t) > 1}
        exact_region = int(region_id is not None and region_id in chunk["region_ids"])
        return (exact_region, len(tokens & words), chunk["chunk_id"])
    selected = sorted(candidates, key=score, reverse=True)[:3]
    region_name = "작품 전체" if region_id is None else f"{region_id} ({by_id[region_id].get('label', '영역')})"
    if not selected:
        text = f"{region_name}에 연결된 검토 완료 설명 자료가 아직 없습니다."
    else:
        text = f"{region_name}에 연결된 자료입니다: " + " / ".join(c["text"] for c in selected)
    return {"region_id": region_id, "question": question, "answer": text, "evidence_chunk_ids": [c["chunk_id"] for c in selected], "highlight_region_id": region_id, "highlight_polygon": by_id[region_id]["polygon"] if region_id else None, "grounding": "approved_chunks" if selected else "no_approved_evidence"}


def export(result: PrototypeResult, path: str | Path) -> None:
    payload = {"artwork_id": result.artwork_id, "display": {"width": result.width, "height": result.height}, "regions": result.regions, "region_id_grid": result.grid, "semantic_chunks": result.chunks}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

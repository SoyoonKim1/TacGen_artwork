"""Command line entry point for the TacGen MVP."""
import argparse
import json
from pathlib import Path

from .pipeline import answer, build, export, load_artwork, resolve_point


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and explore an annotation-backed tactile artwork map")
    parser.add_argument("artwork", help="Input artwork JSON")
    parser.add_argument("--out", help="Write the built map JSON to this path")
    parser.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"), help="Resolve a normalized display point")
    parser.add_argument("--question", help="Ask a grounded question about the resolved region or whole artwork")
    args = parser.parse_args()
    result = build(load_artwork(args.artwork))
    if args.out:
        export(result, args.out)
    region_id = resolve_point(result, *args.point) if args.point else None
    if args.point or args.question:
        print(json.dumps(answer(result, region_id, args.question or "이 영역은 무엇인가요?"), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"artwork_id": result.artwork_id, "display": [result.width, result.height], "region_count": len(result.regions), "chunk_count": len(result.chunks), "output": str(Path(args.out).resolve()) if args.out else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

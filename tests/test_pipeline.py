import copy
import json
import unittest
from pathlib import Path

from tacgen.pipeline import DataError, answer, build, load_artwork, resolve_point, validate_artwork


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = load_artwork(ROOT / "examples" / "artwork.json")
        self.result = build(self.data)

    def test_end_to_end_grid_and_region_lookup(self):
        self.assertEqual((self.result.width, self.result.height), (60, 40))
        self.assertEqual(len(self.result.grid), 40)
        self.assertTrue(all(len(row) == 60 for row in self.result.grid))
        self.assertEqual(resolve_point(self.result, 0.5, 0.6), "R3")
        self.assertEqual(resolve_point(self.result, 0.05, 0.8), None)

    def test_region_answer_is_grounded_and_highlighted(self):
        response = answer(self.result, "R3", "인물은 어디에 있나요?")
        self.assertEqual(response["highlight_region_id"], "R3")
        self.assertIn("demo-001:R3:description", response["evidence_chunk_ids"])
        self.assertEqual(response["grounding"], "approved_chunks")
        self.assertIsNotNone(response["highlight_polygon"])

    def test_region_query_excludes_unrelated_region_claims(self):
        data = copy.deepcopy(self.data)
        data["regions"].append({"region_id": "R4", "label": "모래", "polygon": [[0,0.8],[1,0.8],[1,1],[0,1]], "description": "모래는 화면 아래쪽에 있습니다.", "approval": "approved"})
        result = build(data)
        response = answer(result, "R3", "모래는 어디에 있나요?")
        self.assertFalse(any("R4" in chunk_id for chunk_id in response["evidence_chunk_ids"]))

    def test_unapproved_text_is_not_used(self):
        data = copy.deepcopy(self.data)
        data["regions"][0]["approval"] = "unreviewed"
        result = build(data)
        response = answer(result, "R1", "하늘은 어떤가요?")
        self.assertNotIn("화면 위쪽의 넓은 하늘", response["answer"])

    def test_invalid_relation_target_rejected(self):
        data = copy.deepcopy(self.data)
        data["regions"][0]["relations"][0]["target_region_id"] = "R99"
        with self.assertRaisesRegex(DataError, "unknown region"):
            validate_artwork(data)

    def test_coordinate_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            resolve_point(self.result, 1.01, 0.5)


if __name__ == "__main__":
    unittest.main()

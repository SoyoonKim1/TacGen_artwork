import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.segment_sam3 import Prompt, _spatial_relation_candidates, parse_prompts, run
from tacgen.pipeline import build, validate_artwork


class Sam3AdapterTests(unittest.TestCase):
    def test_missing_prompt_set_is_empty(self):
        self.assertEqual(parse_prompts(None), [])

    def test_custom_prompt_format(self):
        prompts = parse_prompts(["yellow sunflower|노란 꽃송이|4"])
        self.assertEqual((prompts[0].text, prompts[0].label_ko, prompts[0].tactile_priority),
                         ("yellow sunflower", "노란 꽃송이", 4))

    def test_invalid_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_prompts(["flower head"])

    def test_spatial_relation_candidates_use_mask_geometry(self):
        regions = [{"region_id": "R01", "relations": []}, {"region_id": "R02", "relations": []}]
        left = np.zeros((8, 8), dtype=bool)
        left[2:5, 1:3] = True
        right = np.zeros((8, 8), dtype=bool)
        right[2:5, 5:7] = True
        _spatial_relation_candidates(regions, [left, right])

        self.assertEqual(regions[0]["relations"][0]["relation"], "left_of")
        self.assertEqual(regions[0]["relations"][0]["target_region_id"], "R02")
        self.assertEqual(regions[0]["relations"][0]["approval"], "pending")
        self.assertEqual(regions[1]["relations"][0]["relation"], "right_of")

    def test_mocked_inference_exports_valid_artwork_json_and_mask(self):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: True)

        cv2 = types.ModuleType("cv2")
        cv2.RETR_EXTERNAL = 0
        cv2.CHAIN_APPROX_SIMPLE = 0
        contour = np.array([[[2, 2]], [[7, 2]], [[7, 7]], [[2, 7]]], dtype=np.int32)
        cv2.findContours = lambda *_: ([contour], None)
        cv2.contourArea = lambda _: 25
        cv2.arcLength = lambda *_: 20
        cv2.approxPolyDP = lambda polygon, *_: polygon

        sam3 = types.ModuleType("sam3")
        sam3.__path__ = []
        sam3_model = types.ModuleType("sam3.model")
        sam3_model.__path__ = []
        model_builder = types.ModuleType("sam3.model_builder")
        model_builder.build_sam3_image_model = lambda: object()
        image_processor = types.ModuleType("sam3.model.sam3_image_processor")

        class FakeProcessor:
            def __init__(self, _model):
                pass

            def set_image(self, _image):
                return object()

            def set_text_prompt(self, state, prompt):
                return {
                    "masks": np.pad(np.ones((1, 1, 6, 6), dtype=bool), ((0, 0), (0, 0), (2, 2), (2, 2))),
                    "boxes": np.array([[2, 2, 8, 8]], dtype=float),
                    "scores": np.array([0.9]),
                }

        image_processor.Sam3Processor = FakeProcessor
        fake_modules = {
            "torch": torch, "cv2": cv2, "sam3": sam3, "sam3.model": sam3_model,
            "sam3.model_builder": model_builder,
            "sam3.model.sam3_image_processor": image_processor,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", fake_modules):
            root = Path(tmp)
            source = root / "sunflower.jpeg"
            Image.new("RGB", (10, 10), "yellow").save(source)
            output = root / "artwork.json"
            payload = run(source, output, [Prompt("flower", "꽃", 3)], 0.2, 1, "test-sam3")

            self.assertEqual(len(payload["regions"]), 1)
            validate_artwork(json.loads(output.read_text(encoding="utf-8")))
            self.assertTrue((root / payload["regions"][0]["mask_path"]).is_file())
            self.assertTrue((root / "sunflower_sam3_overlay.png").is_file())
            self.assertEqual(build(payload).regions[0]["approval"], "pending")


if __name__ == "__main__":
    unittest.main()

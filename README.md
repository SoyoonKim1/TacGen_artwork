# TacGen prototype

An annotation-backed MVP for the proposal's artwork-to-tactile-map and region-aware curation flow. It includes input validation, region polygon rasterization, Semantic Chunking, Material-Sensory Map export, touched-coordinate lookup, evidence-grounded Korean responses, and region highlight geometry.

## Run

From the repository root, with Python 3.10+:

```powershell
python -m tacgen.cli examples/artwork.json --out output/demo-map.json
python -m tacgen.cli examples/artwork.json --point 0.5 0.6 --question "인물은 어디에 있나요?"
python -m unittest discover -s tests -v
```

The output map contains a 60 x 40 matrix of Region IDs (0 is background). The point coordinates are normalized to [0, 1] with top-left origin. Answers quote approved evidence chunks and include the selected region polygon for a UI to highlight.

## Current boundary

The included demo annotations are synthetic. A SAM 3 inference adapter is available, but the model package/checkpoint are not bundled and inference has not yet been run in this environment. Until it runs, the segmenter input remains curator-provided polygon JSON. The answerer is an evidence selector with a Korean response template, not an LLM. See [docs/PROTOTYPE_PLAN.md](docs/PROTOTYPE_PLAN.md) for the proposal mapping, data contract, roadmap and evaluation plan.

## SAM 3 segmentation for multiple artworks

SAM 3 runs with text prompts and exports matching regions with `approval: "pending"` to `artwork.json`, binary masks, and an overlay for review. Checkpoint access is gated: request access and authenticate with Hugging Face first.

Install in an environment with Python 3.12+, a compatible PyTorch build, the official SAM 3 package, and OpenCV:

```powershell
conda create -n tacgen-sam3 python=3.12
conda activate tacgen-sam3
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -U opencv-python-headless pillow numpy
hf auth login
```

Run prompt based segmentation on any image. Each prompt uses `English text|Korean label|priority`; repeat `--prompt` to segment several concepts:

```powershell
python scripts/segment_sam3.py --image data/sunflower.jpeg --output data/artwork.json --title "해바라기" --artist "빈센트 반 고흐" --prompt "sunflower head|해바라기 꽃송이|5" --prompt "vase|꽃병|3"
```

Review the generated `<image-stem>_sam3_overlay.png`, then merge, relabel, or discard candidates before approval. Prompts can also be stored in JSON as `{"prompts":[{"text":"sunflower head","label_ko":"해바라기 꽃송이","tactile_priority":5}]}`:

```powershell
python scripts/segment_sam3.py --image data/sunflower.jpeg --output data/artwork.json --prompts-json data/sunflower.prompts.json
```

`--score-threshold` and `--min-area-pixels` filter model results. The command requires at least one prompt. It also adds geometry-based relation candidates (`left_of`, `above`, `overlaps`, `contains`, and related directions) to each region. These are calculated from the masks and marked `approval: "pending"`; review them before treating them as annotations.

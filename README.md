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

The default `auto` mode is artwork-agnostic: it proposes image regions without assuming a subject such as a sunflower. It exports unlabeled regions with `approval: "pending"` to `artwork.json`, binary masks, and an overlay for review. Automatic proposals are shapes, not confirmed object identities; curator review and semantic labeling are still needed. This uses the Hugging Face Transformers mask-generation pipeline documented on the [SAM 3 model card](https://huggingface.co/facebook/sam3). Checkpoint access is gated: request access and authenticate with Hugging Face first.

Install in an environment with Python 3.12+, a compatible PyTorch build, Transformers, and OpenCV:

```powershell
conda create -n tacgen-sam3 python=3.12
conda activate tacgen-sam3
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -U transformers accelerate opencv-python-headless pillow numpy
hf auth login
```

Run generic segmentation on any image (replace paths and optional metadata):

```powershell
python scripts/segment_sam3.py --image data/sunflower.jpeg --output data/artwork.json --title "해바라기" --artist "빈센트 반 고흐"
```

Every artwork uses the same default mode; only the input, output, title, and artist change. Review the generated `<image-stem>_sam3_overlay.png`, then merge, relabel, or discard candidates before approval.

For targeted concept segmentation, provide per-artwork concepts. This optional mode uses the official [SAM 3 repository API](https://github.com/facebookresearch/sam3) and requires its `sam3` package and CUDA setup in addition to checkpoint access. Prompts can be passed inline or stored in JSON as `{"prompts":[{"text":"sunflower head","label_ko":"해바라기 꽃송이","tactile_priority":5}]}`:

```powershell
python scripts/segment_sam3.py --mode concept --image data/sunflower.jpeg --output data/artwork.json --prompts-json data/sunflower.prompts.json
```

The default auto mode deliberately has no fixed prompt set. `--score-threshold`, `--min-area-pixels`, and `--points-per-batch` tune proposal generation; exported regions remain pending in either mode.

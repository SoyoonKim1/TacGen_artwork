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

## SAM 3 sunflower segmentation

The SAM 3 adapter exports pending region candidates to `data/artwork.json`, binary masks under `data/segmentation/sunflower/masks/`, and an overlay for review. It uses the official image-model API and requires a CUDA-enabled Python 3.12+ environment, the SAM 3 package, OpenCV, and access to Meta's Hugging Face checkpoint.

In a separate environment, following the [official SAM 3 installation steps](https://github.com/facebookresearch/sam3):

```powershell
conda create -n tacgen-sam3 python=3.12
conda activate tacgen-sam3
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install git+https://github.com/facebookresearch/sam3.git opencv-python-headless pillow numpy
hf auth login
```

Request/accept SAM 3 checkpoint access on Hugging Face before logging in. Then, from the repository root:

```powershell
python scripts/segment_sam3.py --image data/sunflower.jpeg --output data/artwork.json
```

All exported regions start with `approval: "pending"`. Review `data/sunflower_sam3_overlay.png`, correct/merge regions as needed, and only then mark regions approved for downstream content. Override prompts with repeated `--prompt "English concept|한국어 레이블|우선순위"` arguments.

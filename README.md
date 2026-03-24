# SafeSteer-IN

Inference-time safety steering for Indic LLMs using Contrastive Activation Addition (CAA).

## What this project does

SafeSteer-IN adds a plug-in safety layer at inference time:

1. Build contrastive safe/unsafe data for India-specific harms.
2. Extract steering vectors per `(language, category, layer)` from a target LLM.
3. Route user prompts by language/category via an IndicBERT classifier.
4. Apply activation steering in the model forward pass without weight updates.
5. Export steered outputs and latency for external judging (e.g., Claude Sonnet).

No model fine-tuning is required for the target LLM.

## Current workflow (submission mode)

The repository is currently optimized for:

- Synthetic contrastive data generation (seed templates + optional Claude API augmentation)
- Steering vector extraction for multi-language, multi-category slices
- External test prompt ingestion from CSV/XLSX files
- Steered-output export for LLM-as-a-judge scoring
- Layer sensitivity and vector similarity analysis scripts

## Supported models

- `sarvam-1` (default)
- `openhathi-base`
- `airavata`
- `sarvam-m` (high VRAM)
- `krutrim-2-instruct` (12B, high VRAM; use `--quantize-4bit` on Kaggle)

## Supported languages

- `hi`, `ta`, `bn`, `gu`, `mr`, `hi-en`, `te`, `kn`, `ml`

## Harm categories

- `communal_religious_hate`
- `caste_discrimination`
- `political_misinformation`
- `gender_based_violence`
- `code_mixed_toxicity`
- `anti_minority_sentiment`
- `child_safety`
- `financial_scam`

## Repository layout

- `config.py`: global config, model/language/category settings
- `data/`: taxonomy, templates, synthetic generation, dataset builders
- `steering/`: vector extraction, hooks, alpha calibration
- `classifier/`: IndicBERT training + inference
- `engine/`: steering engine + end-to-end runtime pipeline
- `evaluation/`: metrics + evaluation runners + exports
- `scripts/01..08`: runnable pipeline and analysis entry points
- `steering_vectors/`: extracted vector artifacts
- `evaluation/results/`: output metrics and export files

## Environment setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install sentencepiece openpyxl
```

Optional for synthetic generation via Claude API:

```bash
pip install anthropic
```

Set environment variables as needed:

- `HF_TOKEN=<YOUR_HF_TOKEN>`

## Step-by-step commands

### 1) Build synthetic dataset

```bash
python scripts/01_build_dataset.py --no-augment --min-pairs-per-slice 30
```

Notes:
- HF dataset ingestion is disabled by default in the current setup.
- If no Claude key is set, fallback synthetic expansion is used.

### 2) Extract steering vectors (Sarvam-1)

```bash
python scripts/02_extract_vectors.py \
  --model sarvam-1 \
  --languages hi ta bn gu mr hi-en te kn ml \
  --categories communal_religious_hate caste_discrimination political_misinformation gender_based_violence code_mixed_toxicity anti_minority_sentiment child_safety financial_scam \
  --no-probe
```

### 3) Extract steering vectors (OpenHathi 4-bit)

```bash
python scripts/02_extract_vectors.py \
  --model openhathi-base \
  --quantize-4bit \
  --languages hi \
  --categories communal_religious_hate caste_discrimination political_misinformation gender_based_violence code_mixed_toxicity anti_minority_sentiment child_safety financial_scam \
  --no-probe
```

### 3B) Extract steering vectors (Krutrim-2-Instruct 4-bit)

```bash
python scripts/02_extract_vectors.py \
  --model krutrim-2-instruct \
  --quantize-4bit \
  --languages hi ta bn gu mr hi-en te kn ml \
  --categories communal_religious_hate caste_discrimination political_misinformation gender_based_violence code_mixed_toxicity anti_minority_sentiment child_safety financial_scam \
  --no-probe
```

### 4) Export steered outputs from external test files

```bash
python scripts/05_evaluate.py \
  --model sarvam-1 \
  --test-dir data/datasets/expanded \
  --steered-only \
  --max-prompts 200 \
  --max-tokens 128
```

Repeat with `--model openhathi-base`.

For Hindi-only evaluation on language-specific models:

```bash
python scripts/05_evaluate.py \
  --model openhathi-base \
  --test-dir data/datasets/expanded \
  --languages hi \
  --steered-only
```

Repeat similarly with `--model krutrim-2-instruct`.

Output location:

- `evaluation/results/steered_exports/<model>/all_categories_steered.csv`
- `evaluation/results/steered_exports/<model>/all_categories_steered.json`
- category-wise CSV files

### 5) Layer sensitivity

```bash
python scripts/07_layer_sensitivity.py \
  --model sarvam-1 \
  --languages hi ta bn gu mr hi-en te kn ml \
  --categories communal_religious_hate caste_discrimination political_misinformation gender_based_violence code_mixed_toxicity anti_minority_sentiment child_safety financial_scam \
  --alphas 5 10 15 20 25 30 35 40 \
  --split test \
  --max-prompts 20 \
  --max-tokens 96 \
  --output-csv evaluation/results/layer_sensitivity_sarvam1.csv
```

### 6) Cosine + CKA vector similarity

```bash
python scripts/08_vector_similarity.py \
  --model sarvam-1 \
  --languages hi ta bn gu mr hi-en te kn ml \
  --categories communal_religious_hate caste_discrimination political_misinformation gender_based_violence code_mixed_toxicity anti_minority_sentiment child_safety financial_scam \
  --layer 12 \
  --output-dir evaluation/results/similarity/sarvam1
```

## External test data format

CSV/XLSX expected columns:

- `prompt_id`
- `language`
- `category`
- `prompt_text`

The evaluator also handles Excel sheets where the first row contains these headers but pandas auto-reads columns as `Column1...Column4`.

## Artifact semantics

### `layerN.pt` files

Each `layerN.pt` stores a steering vector tensor for one specific:

- model
- language
- category
- layer

Conceptually, this is the normalized unsafe-safe direction used by the forward hook.

### `metadata.json`

Stored alongside vectors and includes:

- model/language/category
- saved layers
- vector norms/shapes
- extraction metadata

## Git LFS

Large files are tracked with LFS in `.gitattributes`:

- `*.pt`, `*.bin`, `*.safetensors`
- steering vector layer files
- classifier checkpoint artifacts

Initialize LFS before pushing artifacts:

```bash
git lfs install
git add .gitattributes
```

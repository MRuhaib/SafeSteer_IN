# SafeSteer-IN — Complete Project Documentation

> **Inference-Time Safety Steering for Indic LLMs**  
> Contrastive Activation Addition (CAA) · Zero weight updates · Plug-in safety layer

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technical Architecture](#2-technical-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Environment Setup](#4-environment-setup)
5. [Configuration Reference](#5-configuration-reference)
6. [Six-Step Pipeline](#6-six-step-pipeline)
   - [Step 1 — Build Dataset](#step-1--build-dataset)
   - [Step 2 — Extract Steering Vectors](#step-2--extract-steering-vectors)
   - [Step 3 — Train IndicBERT Classifier](#step-3--train-indicbert-classifier)
   - [Step 4 — Calibrate Alpha](#step-4--calibrate-alpha)
   - [Step 5 — Evaluate](#step-5--evaluate)
   - [Step 6 — Launch Demo](#step-6--launch-demo)
7. [Module Reference](#7-module-reference)
8. [Supported Models](#8-supported-models)
9. [Supported Languages & Harm Categories](#9-supported-languages--harm-categories)
10. [REST API](#10-rest-api)
11. [Azure Integration](#11-azure-integration)
12. [CPU-Only / Low-Resource Deployment](#12-cpu-only--low-resource-deployment)
13. [Colab / GPU Deployment](#13-colab--gpu-deployment)
14. [Outputs & Artefacts](#14-outputs--artefacts)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

SafeSteer-IN applies **Contrastive Activation Addition (CAA)** to make Indic Large Language Models safer at inference time — without modifying any model weights.

### Core idea

1. Collect **contrastive pairs**: an unsafe model response alongside a safe refusal for the same prompt.
2. Run both responses through the target LLM and record the **residual-stream activations** at each transformer layer.
3. Compute a **steering vector** = mean(unsafe activations) − mean(safe activations) per layer.
4. At inference time, **subtract** that vector (scaled by α) from the live residual stream → the model is continuously nudged toward safer outputs.

### Why it matters for Indic languages

Generic English safety datasets (RLHF fine-tunes, etc.) miss India-specific harm types:

| Harm Category | Example |
|---|---|
| Communal / Religious Hate | Anti-Muslim/Hindu rhetoric, riot glorification |
| Caste Discrimination | Casteist slurs, untouchability promotion |
| Political Misinformation | Fake manifesto quotes, election fraud claims |
| Gender-based Violence | Domestic abuse glorification |
| Code-mixed Toxicity | Hinglish hate speech |
| Anti-minority Sentiment | Tribal/Dalit targeting |
| Child Safety | CSAM-adjacent content |
| Financial Scam | UPI/KYC phishing scripts |

The system supports **Hindi, Tamil, Bengali, Gujarati, Marathi, Hinglish (hi-en)** — with shared classifier architecture and per-language-per-category steering vectors.

---

## 2. Technical Architecture

```
                ┌──────────────────────────────────┐
   User Prompt  │         SafeSteer-IN              │  Steered Output
   ────────────►│                                   │────────────────►
                │  1. IndicBERT Risk Classifier      │
                │     → detected language            │
                │     → detected harm category       │
                │     → risk score [0,1]             │
                │                                   │
                │  2. Steering Engine                │
                │     → load vector for (lang, cat) │
                │     → attach PyTorch forward hook  │
                │     → subtract α × steering_vec   │
                │       from residual stream         │
                │                                   │
                │  3. (Optional) Azure Content Safety│
                │     → independent safety score     │
                └──────────────────────────────────┘
```

**Key design properties:**
- **No weight updates** — the LLM is frozen; only activations are modified at inference.
- **Per-slice vectors** — one steering vector per (language × harm_category) combination.
- **Adaptive alpha** — the best α is calibrated per-slice on the validation set.
- **Graceful degradation** — every external dependency (HF datasets, Azure, GPU) is optional; the system falls back cleanly.

---

## 3. Repository Structure

```
SafeSteer_IN/
│
├── config.py                  ← Central configuration (all settings live here)
├── app.py                     ← Gradio demo UI
├── api.py                     ← FastAPI REST server
├── requirements.txt
├── .env.example               ← Template for environment variables
├── .env                       ← Your real credentials (never commit)
│
├── scripts/                   ← Six numbered pipeline steps
│   ├── 01_build_dataset.py
│   ├── 02_extract_vectors.py
│   ├── 03_train_classifier.py
│   ├── 04_calibrate_alpha.py
│   ├── 05_evaluate.py
│   └── 06_launch_demo.py
│
├── data/
│   ├── build_dataset.py       ← Dataset construction pipeline
│   ├── templates.py           ← Hand-crafted seed contrastive pairs
│   ├── taxonomy.py            ← India-specific harm taxonomy (8 categories)
│   └── datasets/              ← Generated JSONL outputs (train/val/test)
│
├── steering/
│   ├── extract_vectors.py     ← CAA vector extraction logic
│   ├── calibrate_alpha.py     ← Alpha sweep & selection logic
│   └── hooks.py               ← PyTorch forward-hook utilities
│
├── classifier/
│   ├── train_classifier.py    ← IndicBERT fine-tuning (multi-task)
│   └── inference.py           ← RiskPredictor: language + category + score
│
├── engine/
│   ├── steering_engine.py     ← Manages vectors + hooks at inference time
│   └── pipeline.py            ← Full end-to-end SafeSteerPipeline
│
├── evaluation/
│   ├── evaluate.py            ← Full evaluation runner
│   ├── metrics.py             ← SIR / FPR / Fluency / Latency metrics
│   ├── azure_safety.py        ← Azure Content Safety scorer
│   └── results/               ← eval_results.csv / .json
│
├── azure_ml/
│   ├── setup.py               ← Azure ML workspace initialisation
│   └── tracking.py            ← MLflow / Azure ML experiment logging
│
├── models/
│   └── indic_risk_classifier/ ← Trained IndicBERT checkpoint
│       ├── classifier.pt
│       ├── label_maps.json
│       └── tokenizer*
│
└── steering_vectors/          ← Extracted CAA vectors
    └── {model_key}/
        └── {lang}/
            └── {category}/
                ├── layer{N}.pt
                └── metadata.json
```

---

## 4. Environment Setup

### Prerequisites

- Python 3.10 or 3.11+ (3.12 works)
- Git
- A HuggingFace account with an access token (`HF_TOKEN`) — required for gated models (Airavata, OpenHathi)

### Install (CPU-only, for development/demo)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install PyTorch (CPU build — smaller, no CUDA needed)
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Pin NumPy to avoid torch compatibility issues
pip install "numpy<2"

# 4. Install all other dependencies
pip install -r requirements.txt
```

> **Note:** `bitsandbytes` (4-bit quantisation) does not work on CPU-only machines.
> Set `USE_4BIT = False` in `config.py` **before** running any script.

### Install (GPU — recommended for Steps 2 & 4)

```bash
# CUDA 12.1 example
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# bitsandbytes will now work for 4-bit loading
```

### Configure environment variables

```bash
# Copy the template
cp .env.example .env
```

Edit `.env`:

```ini
# Active model (see Supported Models section)
SAFESTEER_MODEL=openhathi-base

# HuggingFace token — required for Airavata, OpenHathi (gated models)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Azure Content Safety (optional — leave blank to skip)
AZURE_CONTENT_SAFETY_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_CONTENT_SAFETY_KEY=your-key-here

# Azure ML (optional — for experiment tracking)
AZURE_ML_SUBSCRIPTION_ID=your-subscription-id
AZURE_ML_RESOURCE_GROUP=safesteer-rg
AZURE_ML_WORKSPACE=safesteer-in-ws
```

---

## 5. Configuration Reference

All project-wide settings live in **`config.py`**. These are the most important ones:

### Model selection

```python
ACTIVE_MODEL = os.getenv("SAFESTEER_MODEL", "openhathi-base")
```

Override at runtime:
```bash
SAFESTEER_MODEL=sarvam-1 python scripts/02_extract_vectors.py
```

### Quantisation

```python
USE_4BIT = False   # Set True only on a GPU with bitsandbytes installed
```

### Steering defaults

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_ALPHA` | `15.0` | Steering vector scale applied at inference |
| `ALPHA_SEARCH_RANGE` | `[5, 10, 15, 20, 25]` | Values swept during calibration |
| `MAX_PERPLEXITY_INCREASE` | `0.20` | Maximum allowed fluency degradation (20%) |
| `RISK_THRESHOLD` | `0.5` | Classifier score above which steering is auto-applied |

### Generation

| Variable | Default | Description |
|---|---|---|
| `MAX_NEW_TOKENS` | `256` | Maximum tokens to generate per response |
| `TEMPERATURE` | *(in config)* | Sampling temperature |
| `TOP_P` | *(in config)* | Nucleus sampling threshold |

### Classifier

| Variable | Value | Description |
|---|---|---|
| `INDICBERT_MODEL_ID` | `ai4bharat/IndicBERTv2-MLM-only` | Backbone model |
| `CLASSIFIER_CHECKPOINT` | `models/indic_risk_classifier/` | Where the fine-tuned model is saved |
| `CLASSIFIER_NUM_CATEGORIES` | `8` | Number of harm categories |
| `CLASSIFIER_NUM_LANGUAGES` | `6` | Number of supported languages |

### Ports

| Variable | Default |
|---|---|
| `GRADIO_PORT` | `7860` |
| `API_PORT` | `8000` |

---

## 6. Six-Step Pipeline

The pipeline runs in order. Steps 1 and 3 run fully on CPU. Steps 2 and 4 require a GPU in practice (they will technically run on CPU but are very slow with 7B+ models).

```
Step 1  →  Step 2  →  Step 3  →  Step 4  →  Step 5  →  Step 6
Dataset    Vectors    Classifier  Alpha      Evaluate   Demo
(CPU ✓)   (GPU rec)  (CPU ✓)    (GPU rec)  (GPU rec)  (CPU ✓)
```

---

### Step 1 — Build Dataset

**Script:** `scripts/01_build_dataset.py`  
**Module:** `data/build_dataset.py`

Constructs the contrastive-pair dataset from three sources:

1. **Seed templates** (`data/templates.py`) — hand-crafted pairs covering all 8 categories × 6 languages.
2. **HuggingFace datasets** — automatically fetched and merged:
   - `textdetox/multilingual_toxicity_dataset` — Hindi + Hinglish binary toxicity (public)
   - `ai4bharat/indic-align` (toxic split) — toxic prompts + LLaMA-2 refusals
   - `manueltonneau/india-hate-speech-superset` — gated; fetched with `HF_TOKEN` if accepted
   - `DAMO-NLP-SG/MultiJail` — Hindi/Bengali/Tamil jailbreak prompts
   - `CohereForAI/aya_dataset` — safe Indic completions (used as safe-half substitutes)
3. **Augmentation** — each seed pair is paraphrased N times with prefix/suffix wrappers.

**Output schema** (`data/datasets/{train,val,test}.jsonl`):

```json
{
  "prompt": "भारत में मुसलमानों के बारे में...",
  "safe_response": "मुझे इस विषय पर हानिकारक सामग्री...",
  "unsafe_response": "मुसलमान तो...",
  "language": "hi",
  "category": "communal_religious_hate",
  "source": "seed"
}
```

**Usage:**

```bash
# Default (with HF datasets + augmentation)
python scripts/01_build_dataset.py

# Seed templates only (no network, fastest)
python scripts/01_build_dataset.py --no-hf --no-augment

# Augment seed pairs only (no HF network calls), 3 variants per pair
python scripts/01_build_dataset.py --no-hf --augment-n 3
```

**All CLI parameters:**

| Flag | Default | Description |
|---|---|---|
| `--no-augment` | off | Disable paraphrase augmentation |
| `--no-hf` | off | Skip all HuggingFace dataset loading |
| `--augment-n N` | `2` | Number of augmented variants per seed pair |
| `--seed N` | `42` | Random seed for reproducibility |
| `--output-dir PATH` | `data/datasets/` | Override output directory |

---

### Step 2 — Extract Steering Vectors

**Script:** `scripts/02_extract_vectors.py`  
**Module:** `steering/extract_vectors.py`  
**Requires:** GPU strongly recommended (loads a 3–24B LLM)

For each (language × harm_category) slice of the training set:

1. Loads the target LLM (the model set by `ACTIVE_MODEL` / `--model`).
2. Passes **safe** completions through the model → records mean residual-stream activations at each target layer.
3. Passes **unsafe** completions through the model → records mean residual-stream activations.
4. Computes `steering_vector = mean_unsafe_acts − mean_safe_acts`, then L2-normalises.
5. Saves vectors to `steering_vectors/{model_key}/{lang}/{category}/layer{N}.pt`.

**Optional layer probe:** Before extraction, a linear probe is trained on each layer's activations to identify the most discriminative layers (`--probe-k` top layers are kept).

**Usage:**

```bash
# Use the default model from config.py
python scripts/02_extract_vectors.py

# Specify model and restrict to certain languages
python scripts/02_extract_vectors.py --model sarvam-1 --languages hi ta

# Skip the probe, extract for all candidate layers
python scripts/02_extract_vectors.py --no-probe

# Full example
python scripts/02_extract_vectors.py \
  --model airavata \
  --languages hi hi-en bn \
  --categories communal_religious_hate caste_discrimination \
  --probe-k 5 \
  --max-length 512
```

**All CLI parameters:**

| Flag | Default | Description |
|---|---|---|
| `--model KEY` | `ACTIVE_MODEL` | Model key from `MODEL_CONFIGS` |
| `--languages L [L ...]` | all | Language codes to process |
| `--categories C [C ...]` | all | Harm category keys to process |
| `--split SPLIT` | `train` | Dataset split to use for extraction |
| `--no-probe` | off | Skip layer-probe; use all candidate layers |
| `--probe-k N` | `3` | Number of top layers to keep from probe |
| `--max-length N` | `512` | Max token length per input |

**Output files:**

```
steering_vectors/
└── openhathi-base/
    └── hi/
        └── communal_religious_hate/
            ├── layer12.pt       # torch.Tensor [hidden_dim]
            ├── layer16.pt
            ├── layer20.pt
            └── metadata.json   # num_safe, num_unsafe, layer_norm_scores, etc.
```

---

### Step 3 — Train IndicBERT Classifier

**Script:** `scripts/03_train_classifier.py`  
**Module:** `classifier/train_classifier.py`  
**Runs on:** CPU (slow but works) or GPU

Fine-tunes **IndicBERTv2** (`ai4bharat/IndicBERTv2-MLM-only`) for **multi-task classification**:

- **Task A:** Language detection — 6-class (hi, ta, bn, gu, mr, hi-en)
- **Task B:** Harm category classification — 8-class

Both heads share the same IndicBERT backbone. Only the prompt text is used (not the safe/unsafe responses) — the model learns to recognise harm from the user query alone.

**Usage:**

```bash
# Quick run (fewer epochs)
python scripts/03_train_classifier.py --epochs 5 --batch-size 8

# Full training recommended settings
python scripts/03_train_classifier.py --epochs 15 --batch-size 16 --lr 2e-5

# Force CPU
python scripts/03_train_classifier.py --device cpu
```

**All CLI parameters:**

| Flag | Default | Description |
|---|---|---|
| `--epochs N` | `10` | Training epochs |
| `--batch-size N` | `16` | Batch size |
| `--lr FLOAT` | `2e-5` | Learning rate |
| `--device STR` | auto | Force `cpu` or `cuda` |

**Output:**

```
models/indic_risk_classifier/
├── classifier.pt         # ~1.1 GB — full model weights
├── label_maps.json       # lang_to_id, cat_to_id mappings
├── tokenizer.json
└── tokenizer_config.json
```

---

### Step 4 — Calibrate Alpha

**Script:** `scripts/04_calibrate_alpha.py`  
**Module:** `steering/calibrate_alpha.py`  
**Requires:** GPU strongly recommended (runs multiple generations per slice)

Sweeps the steering strength α over `ALPHA_SEARCH_RANGE` on the validation set.

**Selection criterion:**  
> Highest safety rate where the perplexity increase is ≤ `MAX_PERPLEXITY_INCREASE` (default 20%).

This ensures the model stays fluent while being as safe as possible.

**Usage:**

```bash
# Default sweep
python scripts/04_calibrate_alpha.py

# Custom alpha range
python scripts/04_calibrate_alpha.py --alphas 5 10 15 20 25 30 --max-pairs 20

# Restrict to specific slices
python scripts/04_calibrate_alpha.py \
  --model sarvam-1 \
  --languages hi \
  --categories communal_religious_hate gender_based_violence
```

**All CLI parameters:**

| Flag | Default | Description |
|---|---|---|
| `--model KEY` | `ACTIVE_MODEL` | Model key |
| `--languages L [L ...]` | all | Language codes |
| `--categories C [C ...]` | all | Category keys |
| `--alphas F [F ...]` | `[5,10,15,20,25]` | Alpha values to sweep |
| `--max-pairs N` | `30` | Max validation pairs per slice |

**Output:**

```
steering_vectors/{model_key}/calibration_results.json
```

```json
{
  "hi/communal_religious_hate": {
    "best": 15.0,
    "sweep": {
      "5.0": {"safety_rate": 0.62, "ppl_increase": 0.03},
      "15.0": {"safety_rate": 0.89, "ppl_increase": 0.14}
    }
  }
}
```

---

### Step 5 — Evaluate

**Script:** `scripts/05_evaluate.py`  
**Module:** `evaluation/evaluate.py`  
**Requires:** GPU strongly recommended

Runs the full evaluation suite on the test split.

**Metrics computed:**

| Metric | Symbol | Description |
|---|---|---|
| Safety Improvement Rate | SIR | % of unsafe prompts steered to safe outputs |
| False Positive Rate | FPR | % of safe prompts incorrectly blocked/neutered |
| Fluency Score | — | Perplexity preservation vs baseline |
| Latency Overhead | — | Added latency from steering hook (ms) |

**Usage:**

```bash
# Standard evaluation
python scripts/05_evaluate.py

# With Azure Content Safety scoring
python scripts/05_evaluate.py --azure

# Limit prompts for a quick run
python scripts/05_evaluate.py --max-prompts 20 --max-tokens 64
```

**All CLI parameters:**

| Flag | Default | Description |
|---|---|---|
| `--model KEY` | `ACTIVE_MODEL` | Model key |
| `--languages L [L ...]` | all | Languages to evaluate |
| `--categories C [C ...]` | all | Categories to evaluate |
| `--azure` | off | Enable Azure Content Safety scoring |
| `--max-prompts N` | `50` | Max prompts per (lang × category) slice |
| `--max-tokens N` | `128` | Max new tokens per generation |

**Output files:**

```
evaluation/results/
├── eval_results.csv    # Per-slice tabular results
└── eval_results.json   # Full results with per-prompt detail
```

---

### Step 6 — Launch Demo

**Script:** `scripts/06_launch_demo.py`  
**Module:** `app.py` (Gradio), `api.py` (FastAPI)

Starts the interactive demo.

**Usage:**

```bash
# Gradio UI only (default — http://localhost:7860)
python scripts/06_launch_demo.py

# Gradio UI + FastAPI server
python scripts/06_launch_demo.py --api

# FastAPI only (http://localhost:8000)
python scripts/06_launch_demo.py --api-only

# Public Gradio share link (for demos/sharing)
python scripts/06_launch_demo.py --share
```

**All CLI parameters:**

| Flag | Description |
|---|---|
| `--api` | Also start FastAPI server in background |
| `--api-only` | Start only the FastAPI server |
| `--share` | Generate a public Gradio sharing URL |

**Or launch directly:**

```bash
# Gradio only
python app.py

# FastAPI only
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 7. Module Reference

### `config.py`

The single source of truth for all settings. Import from here in every module — never hardcode paths or parameters.

```python
from config import (
    MODEL_CONFIGS,    # dict of all model configs
    ACTIVE_MODEL,     # current model key
    get_model_config, # get config by key (optional arg, defaults to ACTIVE_MODEL)
    format_prompt,    # wrap prompt in model-specific chat template
    VECTORS_DIR,      # Path → steering_vectors/
    DATASET_DIR,      # Path → data/datasets/
    MODELS_DIR,       # Path → models/
    HF_TOKEN,         # str|None — from .env
    USE_4BIT,         # bool — enable 4-bit quantisation
    DEFAULT_ALPHA,    # float — default steering strength
    RISK_THRESHOLD,   # float — auto-steer above this classifier score
    LANGUAGES,        # dict: code → full name
    HARM_CATEGORIES,  # dict: int id → category key
)
```

### `data/templates.py`

Contains `ALL_TEMPLATES` — a nested dict of hand-crafted contrastive pairs:

```python
ALL_TEMPLATES = {
  "hi": {
    "communal_religious_hate": [
      {
        "prompt": "...",
        "safe": "...",
        "unsafe": "...",
      },
      ...
    ]
  }
}
```

Edit this file to add more seed examples without any other changes needed.

### `data/taxonomy.py`

Defines `TAXONOMY` and `CATEGORY_KEYS`. Each `HarmCategory` has:

```python
HarmCategory(
    id=0,
    key="communal_religious_hate",
    name="Communal & Religious Hate",
    description="...",
    examples=[...],
    priority_languages=["hi", "hi-en"],
    keywords_hi=["दंगा", "जिहाद", ...],
    keywords_en=["riot", "jihad", ...],
)
```

### `data/build_dataset.py`

Key public functions:

| Function | Description |
|---|---|
| `build_dataset(augment, augment_n, load_hf, seed, output_dir)` | Full pipeline — build and write all splits |
| `load_dataset_split(split_name)` | Load a saved JSONL split as a list of dicts |
| `load_pairs_for_extraction(language, category, split)` | Load pairs filtered by lang+cat for vector extraction |
| `generate_safe_refusal(lang)` | Return a random refusal template for a language |

### `steering/hooks.py`

Two classes that work with any HuggingFace CausalLM via PyTorch forward hooks:

**`ActivationCollector`** — records residual-stream activations during a forward pass:

```python
collector = ActivationCollector(model, target_layers=[12, 16, 20])
with torch.no_grad():
    model(**inputs)
acts = collector.get()   # {12: Tensor[seq, hidden], 16: Tensor, 20: Tensor}
collector.clear()
collector.remove_hooks()
```

**`SteeringHook`** — adds a steering vector during forward passes:

```python
hook = SteeringHook(model, layer=16, vector=steering_vec, alpha=15.0)
output = model.generate(...)  # vector applied during every forward
hook.remove()
```

**`MultiLayerSteeringHook`** — applies steering at multiple layers simultaneously.

### `steering/extract_vectors.py`

Key public functions:

| Function | Description |
|---|---|
| `load_model_and_tokenizer(model_key, quantize)` | Load HF model + tokenizer |
| `run_full_extraction(model_key, languages, categories, dataset_split, probe_first, probe_k, max_length)` | Extract all vectors for all slices |
| `load_vector(model_key, lang, category, layer)` | Load a saved vector tensor |
| `load_all_vectors(model_key, vectors_dir)` | Load all saved vectors into a nested dict |

### `steering/calibrate_alpha.py`

Key public functions:

| Function | Description |
|---|---|
| `compute_perplexity(model, tokenizer, text)` | Per-token perplexity of text |
| `run_full_calibration(model_key, languages, categories, alphas, max_pairs)` | Full calibration sweep |

### `classifier/train_classifier.py`

Defines `IndicRiskClassifier` — a `nn.Module` with:

- Shared IndicBERT backbone
- `lang_head`: linear layer (hidden → 6 language classes)
- `cat_head`: linear layer (hidden → 8 category classes)

Key function: `train_classifier(train_pairs, val_pairs, epochs, batch_size, lr, device)` → returns trained model.

### `classifier/inference.py`

`RiskPredictor` — the inference wrapper:

```python
predictor = RiskPredictor()   # auto-loads from CLASSIFIER_CHECKPOINT

lang, category, risk_score = predictor.predict("भारत में मुसलमानों के...")
# ("hi", "communal_religious_hate", 0.87)
```

Falls back to a keyword-based rule detector when no trained checkpoint is found.

### `engine/steering_engine.py`

`SteeringEngine` — manages vectors and hooks at inference:

```python
engine = SteeringEngine(model, tokenizer, model_key="openhathi-base")

raw_text = engine.generate(prompt)
steered_text = engine.generate_steered(
    prompt,
    language="hi",
    category="communal_religious_hate",
    alpha=15.0,           # optional override
    layers=[12, 16, 20],  # optional override
)
```

### `engine/pipeline.py`

`SafeSteerPipeline` — the top-level one-call interface:

```python
pipeline = SafeSteerPipeline(model_key="openhathi-base", use_azure_safety=True)
pipeline.load()

result = pipeline.run(
    prompt="...",
    force_language="hi",     # optional
    force_category=None,     # optional — auto-detect
    alpha=15.0,              # optional
    always_steer=True,
)

# result is a PipelineResult dataclass:
result.raw_output           # str
result.steered_output       # str
result.detected_language    # "hi"
result.detected_category    # "communal_religious_hate"
result.risk_score           # 0.87
result.steering_applied     # True
result.alpha_used           # 15.0
result.azure_score_raw      # dict or None
result.azure_score_steered  # dict or None
result.latency_ms           # 340.5
```

### `evaluation/metrics.py`

`SafetyMetrics` dataclass + `compute_metrics(engine, pairs, use_azure)` → `SafetyMetrics`.

### `evaluation/azure_safety.py`

`AzureSafetyScorer` — wraps Azure AI Content Safety. Returns per-category severity (0–6) for Hate, SelfHarm, Sexual, Violence. Disabled gracefully if `AZURE_CONTENT_SAFETY_ENDPOINT` is not set.

### `azure_ml/setup.py`

`get_workspace()` — connect to an Azure ML Workspace.  
`create_compute_cluster(ws, ...)` — create GPU/CPU compute.

### `azure_ml/tracking.py`

`log_metrics(metrics_dict)`, `log_vectors(vectors_dir)`, `log_evaluation_results(path)` — all log to MLflow / Azure ML. Degrade gracefully if Azure ML is unconfigured.

---

## 8. Supported Models

All model configurations live in `MODEL_CONFIGS` in `config.py`.

| Key | HuggingFace ID | Params | VRAM | Chat Format | Notes |
|---|---|---|---|---|---|
| `openhathi-base` | `sarvamai/OpenHathi-7B-Hi-v0.1-Base` | 7B | ~14 GB | None (base) | Best for activation probing; Hindi/Hinglish/English |
| `airavata` | `ai4bharat/Airavata` | 7B | ~14 GB | Tulu | Hindi instruction-tuned; gated (llama2 license) |
| `sarvam-1` | `sarvamai/sarvam-1` | ~2B | ~4 GB | None (base) | Smallest & fastest; non-commercial license |
| `sarvam-m` | `sarvamai/sarvam-m` | 24B | ~50 GB | ChatML | Highest quality; GPU server only; Apache 2.0 |

### Model architecture details

| Key | Layers | Hidden Dim | Target Layers for CAA | Primary Layer |
|---|---|---|---|---|
| `openhathi-base` | 32 | 4096 | 12–21 (40–65%) | 16 |
| `airavata` | 32 | 4096 | 12–21 (40–65%) | 16 |
| `sarvam-1` | 28 | 2048 | 11–18 (40–65%) | 14 |
| `sarvam-m` | 40 | 5120 | 16–25 (40–65%) | 20 |

### Chat format details

- **`None` (base/completion models):** Prompt is passed as-is. Best for raw activation extraction.
- **`tulu` (Airavata):** Prompt is wrapped: `<|user|>\n{prompt}\n<|assistant|>\n`
- **`chatml` (Sarvam-M):** Use `tokenizer.apply_chat_template()`. The `format_prompt()` helper in `config.py` handles wrapping automatically for tulu; for chatml, call the tokenizer directly.

### Accessing gated models

OpenHathi and Airavata require accepting a license on HuggingFace before downloading:

1. Go to the model page on HuggingFace.
2. Click "Access repository" / accept the license.
3. Make sure `HF_TOKEN` is set in your `.env`.

### Changing the active model

```bash
# Via environment variable
SAFESTEER_MODEL=sarvam-1 python scripts/02_extract_vectors.py

# Or permanently in .env
SAFESTEER_MODEL=airavata
```

---

## 9. Supported Languages & Harm Categories

### Languages

| Code | Language | Script |
|---|---|---|
| `hi` | Hindi | Devanagari |
| `ta` | Tamil | Tamil |
| `bn` | Bengali | Bengali |
| `gu` | Gujarati | Gujarati |
| `mr` | Marathi | Devanagari |
| `hi-en` | Hinglish (code-mixed) | Latin + Devanagari |

### Harm Categories

| Key | Name | Priority Languages |
|---|---|---|
| `communal_religious_hate` | Communal & Religious Hate | hi, hi-en |
| `caste_discrimination` | Caste Discrimination | hi, mr |
| `political_misinformation` | Political Misinformation | hi, ta |
| `gender_based_violence` | Gender-based Violence | hi, bn |
| `code_mixed_toxicity` | Code-mixed Toxicity | hi-en |
| `anti_minority_sentiment` | Anti-minority Sentiment | hi, bn |
| `child_safety` | Child Safety | all |
| `financial_scam` | Financial Scam | hi, hi-en |

---

## 10. REST API

The FastAPI server exposes four endpoints.

### POST `/steer`

Run the full pipeline.

**Request:**

```json
{
  "prompt": "भारत में मुसलमानों के बारे में...",
  "language": "hi",
  "category": null,
  "alpha": 15.0,
  "max_new_tokens": 256,
  "always_steer": true
}
```

**Response:**

```json
{
  "raw_output": "...",
  "steered_output": "...",
  "detected_language": "hi",
  "detected_category": "communal_religious_hate",
  "risk_score": 0.87,
  "steering_applied": true,
  "alpha_used": 15.0,
  "latency_ms": 340.5
}
```

### POST `/generate`

Baseline generation only (no steering).

```json
{ "prompt": "...", "max_new_tokens": 256 }
```

### GET `/health`

Returns `{"status": "ok"}` — use for liveness probes.

### GET `/slices`

Returns the available `(language, category)` combinations that have steering vectors loaded.

### Start the API server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000

# Or via the script
python scripts/06_launch_demo.py --api-only
```

Interactive API docs at: `http://localhost:8000/docs`

---

## 11. Azure Integration

Azure is entirely **optional**. Every Azure-dependent function degrades gracefully.

### Azure AI Content Safety

Provides an independent content safety score alongside the CAA steering output.

1. Create an **Azure AI Content Safety** resource in the Azure Portal.
2. Copy the endpoint URL and key into `.env`.
3. Run any step with scoring enabled (`--azure` flag on Step 5, `use_azure_safety=True` in the pipeline).

### Azure ML

Used for experiment tracking, compute management, and logging steering vectors as Data Assets.

**Setup (one-time):**

```bash
python -m azure_ml.setup
```

This creates the workspace, GPU compute cluster, and experiment.

**What gets logged:**

- Steering vectors as Azure ML Data Assets
- Calibration results
- Evaluation metrics (SIR, FPR, fluency, latency)
- MLflow run parameters and artefacts

**Without Azure ML**, MLflow falls back to a local tracking server under `mlruns/`.

---

## 12. CPU-Only / Low-Resource Deployment

Steps 1, 3, and 6 run fully on CPU. Steps 2, 4, and 5 load a full LLM which is impractically slow on CPU for 7B models, but possible for `sarvam-1` (~2B).

### Recommended CPU workflow

```bash
# Step 1 — always fast on CPU
python scripts/01_build_dataset.py --no-hf --no-augment

# Step 3 — trains IndicBERT (~300M), feasible on CPU in ~10 min
python scripts/03_train_classifier.py --epochs 5 --batch-size 8

# Step 6 — Gradio demo works on CPU once vectors exist
python scripts/06_launch_demo.py
```

Steps 2, 4, 5 should be run on **Google Colab (free GPU)** or equivalent — see next section.

### Required config changes for CPU

In `config.py`:

```python
USE_4BIT = False   # bitsandbytes not available on CPU
```

Use `sarvam-1` (2B) for Steps 2/4/5 if you must run on CPU — it is the only model small enough to be remotely feasible:

```bash
SAFESTEER_MODEL=sarvam-1 python scripts/02_extract_vectors.py --no-probe
```

---

## 13. Colab / GPU Deployment

Recommended for Steps 2, 4, 5.

### Colab setup

```python
# In a Colab cell:
!git clone https://github.com/your-username/SafeSteer_IN
%cd SafeSteer_IN

# Install PyTorch (Colab usually has CUDA already)
!pip install -r requirements.txt

# Set your HF token
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxx"
os.environ["SAFESTEER_MODEL"] = "openhathi-base"
```

### Run GPU steps

```python
!python scripts/02_extract_vectors.py --model openhathi-base --languages hi hi-en
!python scripts/04_calibrate_alpha.py
!python scripts/05_evaluate.py --max-prompts 30
```

### Download artefacts back to local machine

After running on Colab, download the `steering_vectors/` directory — then Steps 1, 3, 6 can be run locally on CPU using those artefacts.

### 4-bit quantisation on GPU

Enable to reduce VRAM usage (7B fits in ~6 GB with 4-bit):

```python
# In config.py
USE_4BIT = True
```

This requires `bitsandbytes` which is installed via `requirements.txt` on GPU.

---

## 14. Outputs & Artefacts

| Path | Generated by | Description |
|---|---|---|
| `data/datasets/train.jsonl` | Step 1 | Training contrastive pairs |
| `data/datasets/val.jsonl` | Step 1 | Validation pairs |
| `data/datasets/test.jsonl` | Step 1 | Test pairs |
| `data/datasets/incomplete_pairs.jsonl` | Step 1 | Pairs missing unsafe half (manual review) |
| `steering_vectors/{model}/{lang}/{cat}/layer{N}.pt` | Step 2 | Per-layer steering vector tensors |
| `steering_vectors/{model}/{lang}/{cat}/metadata.json` | Step 2 | Extraction stats |
| `steering_vectors/{model}/calibration_results.json` | Step 4 | Best alpha per slice |
| `models/indic_risk_classifier/classifier.pt` | Step 3 | Trained IndicBERT weights |
| `models/indic_risk_classifier/label_maps.json` | Step 3 | Language/category ID mappings |
| `evaluation/results/eval_results.csv` | Step 5 | Tabular evaluation results |
| `evaluation/results/eval_results.json` | Step 5 | Full JSON evaluation report |

---

## 15. Troubleshooting

### `ModuleNotFoundError: No module named 'bitsandbytes'`

You are on CPU. Set `USE_4BIT = False` in `config.py`.

### `trust_remote_code` error from datasets

You are on `datasets >= 4.0`. The loaders in `build_dataset.py` do **not** use `trust_remote_code` — this is fixed. If you see this error from another file, remove the `trust_remote_code=True` kwarg.

### HuggingFace dataset says "gated" / 401 error

1. Go to the dataset page and accept the terms.
2. Ensure `HF_TOKEN` is set in your `.env` file.
3. The `manueltonneau/india-hate-speech-superset` dataset requires a contact agreement form — if you have not filled it, the loader silently skips it.

### `KeyError: 'openhathi'` in config

The model key was renamed from `openhathi` to `openhathi-base`. Update `.env`:

```
SAFESTEER_MODEL=openhathi-base
```

### Gradio fails with `ModuleNotFoundError: No module named 'gradio'`

```bash
pip install gradio>=4.0.0
```

### NumPy version conflict with PyTorch

```bash
pip install "numpy<2"
```

### Out of memory on GPU during Step 2

- Use `USE_4BIT = True` (requires GPU + bitsandbytes).
- Use a smaller model (`sarvam-1`).
- Reduce `--max-length` to 256.
- Reduce the number of languages/categories processed at once.

### Step 5 evaluation produces no results

Steering vectors must exist before evaluation. Run Step 2 first.

### No steering vectors found at inference time

The Gradio app and API operate in **demo mode** — they show raw and steered output side-by-side, but steering requires:

1. `steering_vectors/{active_model}/` directory to be populated (Step 2).
2. `models/indic_risk_classifier/classifier.pt` to exist (Step 3).

Without these, the pipeline falls back to generating raw output only.

---

*Last updated: March 2026 — SafeSteer-IN v0.1.0*

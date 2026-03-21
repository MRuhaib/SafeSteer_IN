# SafeSteer-IN — Documentation

This document is intended as a high-fidelity technical context pack for humans and LLM agents.
It describes architecture, data formats, runtime behavior, scripts, artifact semantics, and current experiment workflow.

## 1. Project objective

SafeSteer-IN is an inference-time safety steering framework for Indic LLMs.

Core principle:
- Learn activation-space steering directions from contrastive safe/unsafe examples.
- Inject steering at generation time using forward hooks.
- Keep base model weights frozen.

Primary target use case:
- India-grounded harm categories where English-centric filters underperform.

## 2. Current system architecture

### 2.1 Modules

- `config.py`
  - Central source of truth for models, languages, categories, defaults.
- `data/`
  - `taxonomy.py`: 8-category safety taxonomy and metadata.
  - `templates.py`: multilingual seed contrastive examples.
  - `synthetic_generation.py`: category/language-conditioned synthetic expansion.
  - `build_dataset.py`: dataset assembly and split writing.
- `steering/`
  - `extract_vectors.py`: activation collection and vector extraction.
  - `hooks.py`: runtime activation collection and vector injection hooks.
  - `calibrate_alpha.py`: optional alpha sweeps.
- `classifier/`
  - `train_classifier.py`: IndicBERT multi-task training.
  - `inference.py`: prompt routing predictor.
- `engine/`
  - `steering_engine.py`: vector loading + generation APIs.
  - `pipeline.py`: end-to-end orchestration (classify -> steer -> score).
- `evaluation/`
  - `evaluate.py`: standard and external-file evaluation/export modes.
  - `metrics.py`: SIR/FPR/fluency/latency utilities.
- `scripts/`
  - `01` build data
  - `02` extract vectors
  - `03` train classifier
  - `04` calibrate alpha
  - `05` evaluate/export
  - `06` launch demo
  - `07` layer sensitivity
  - `08` vector similarity (cosine + CKA)

### 2.2 Runtime flow

1. Classifier predicts language/category/risk score for input prompt.
2. Steering engine loads `(language, category, layer)` vector.
3. Generation runs with optional steering hook:
   - residual update is applied at target layer each forward pass.
4. Outputs + latency are collected.
5. Optional post-scoring with Azure safety service.

## 3. Configuration reference (`config.py`)

### 3.1 Model configs

Each model has:
- `model_id`
- `num_layers`
- `hidden_dim`
- `target_layers`
- `primary_layer`
- `layer_accessor`
- `chat_format`
- `requires_high_vram`

Currently configured:
- `openhathi-base`
- `airavata`
- `sarvam-1`
- `sarvam-m`

### 3.2 Languages

Current language codes:
- `hi`, `ta`, `bn`, `gu`, `mr`, `hi-en`, `te`, `kn`, `ml`

Classifier language count:
- `CLASSIFIER_NUM_LANGUAGES = 9`

### 3.3 Categories

Current category set (8):
- communal_religious_hate
- caste_discrimination
- political_misinformation
- gender_based_violence
- code_mixed_toxicity
- anti_minority_sentiment
- child_safety
- financial_scam

### 3.4 Key defaults

- `ACTIVE_MODEL`: defaults to `sarvam-1`
- `DEFAULT_ALPHA`: 15.0
- `ALPHA_SEARCH_RANGE`: [5, 10, 15, 20, 25, 30, 35, 40]
- `RISK_THRESHOLD`: 0.5
- `MAX_NEW_TOKENS`: 256
- `USE_4BIT`: False (can be overridden via extraction CLI flag)

## 4. Data pipeline behavior

## 4.1 Seed templates

`data/templates.py` contains seed pairs by language and category.
Each pair includes:
- `prompt`
- `safe`
- `unsafe`

### 4.2 Synthetic expansion (`data/synthetic_generation.py`)

Generation modes:

1. Claude mode (if key present)
- Requires `ANTHROPIC_API_KEY`.
- Uses a strict prompt template to request JSON arrays of contrastive pairs.
- Model default in code: `claude-sonnet-4-5` (overridable via `SYNTH_MODEL`).

2. Fallback mode (no key)
- Deterministic expansion from seed examples.
- Prefix/suffix variation + variant markers.
- Ensures minimum per-slice pair count.

Output normalization guarantees each row has:
- `prompt`
- `safe_response`
- `unsafe_response`
- `language`
- `category`
- `source`

### 4.3 Dataset construction (`data/build_dataset.py`)

Current default behavior:
- Synthetic-first.
- `load_hf=False` by default.
- Enforces per-slice minimum via synthetic generation.
- Writes:
  - `data/datasets/train.jsonl`
  - `data/datasets/val.jsonl`
  - `data/datasets/test.jsonl`

CLI entrypoint:
- `scripts/01_build_dataset.py`

Useful flag:
- `--min-pairs-per-slice` (default 30)

## 5. Steering vector extraction

### 5.1 Core extraction logic (`steering/extract_vectors.py`)

For each `(language, category)` slice:
1. Build safe texts: `prompt + safe_response`
2. Build unsafe texts: `prompt + unsafe_response`
3. Collect activations per target layer.
4. Compute vector:
   - unsafe mean minus safe mean
5. L2 normalize vector.
6. Save vector and metadata.

### 5.2 Layer handling

`run_full_extraction(...)` supports:
- probe mode (`probe_first=True`) for ranking layers
- `probe_only=True` to save only top-k probed layers
- default full save path keeps all configured target layers and includes primary layer

### 5.3 Quantization

Extraction script supports explicit 4-bit model load:
- `scripts/02_extract_vectors.py --quantize-4bit`

### 5.4 Output format

Per slice path:
- `steering_vectors/{model}/{language}/{category}/`

Files:
- `layer{N}.pt`: steering vector tensor for that layer.
- `metadata.json`: extraction metadata and layer stats.

## 6. Steering runtime semantics

### 6.1 `SteeringHook` (`steering/hooks.py`)

At hooked layer output:
- Applies `hidden = hidden - alpha * vector`

Vector interpretation:
- Vector represents unsafe-safe direction.
- Subtraction nudges generation away from unsafe manifold.

### 6.2 `SteeringEngine` (`engine/steering_engine.py`)

Functions:
- preload vectors
- optional load per-slice calibrated alpha
- baseline generation (`generate`)
- steered generation (`generate_steered`)
- slice availability queries

Current generation mode:
- `do_sample=False` (greedy) to avoid sampling instability under steering perturbations.

### 6.3 `SafeSteerPipeline` (`engine/pipeline.py`)

Pipeline behavior:
- classify prompt
- baseline output
- steering decision: always-steer or risk-threshold
- fallback if exact vector missing:
  - same-language available slice
  - else Hindi available slice
- optional Azure scoring

## 7. Classifier details

`classifier/train_classifier.py`:
- Backbone: `ai4bharat/IndicBERTv2-MLM-only`
- Multi-task heads:
  - language classification
  - category classification
- Input used for training: prompt text from contrastive dataset
- Warmup strategy:
  - freeze backbone first
  - unfreeze after initial epochs

## 8. Evaluation and export modes

`evaluation/evaluate.py` supports two major pathways:

### 8.1 Standard split evaluation
- Reads JSONL test split from `data/datasets/test.jsonl`
- Computes aggregate metrics

### 8.2 External prompt file evaluation
- Reads prompt files from a directory (`--test-dir`)
- Supported formats: `.csv`, `.xls`, `.xlsx`
- Expected fields:
  - `prompt_id`, `language`, `category`, `prompt_text`

Header recovery logic:
- If sheet columns appear as `Column1...` and first row contains real headers,
  evaluator auto-promotes first row to header.

### 8.3 Steered-only export mode (judge workflow)

CLI:
- `scripts/05_evaluate.py --test-dir ... --steered-only`

Behavior:
- Runs steered generation per prompt.
- If no vector available for slice, falls back to unsteered output with flag.
- Records latency per prompt.

Outputs:
- `evaluation/results/steered_exports/{model}/{category}.csv`
- `evaluation/results/steered_exports/{model}/all_categories_steered.csv`
- `evaluation/results/steered_exports/{model}/all_categories_steered.json`

Row schema includes:
- `prompt_id`, `language`, `category`, `prompt_text`
- `steered_output`
- `latency_ms`
- `steering_applied`
- `model`

## 9. Standalone analysis scripts

### 9.1 Layer sensitivity (`scripts/07_layer_sensitivity.py`)

- Sweeps specified alphas over available layers for each selected slice.
- Computes slice-level SIR-like delta from safety checker.
- Writes CSV with:
  - model, language, category, layer, alpha
  - counts and score columns

### 9.2 Vector similarity (`scripts/08_vector_similarity.py`)

- Loads saved vectors from disk.
- Builds language-level representations from category vectors.
- Computes:
  - cosine similarity matrix
  - linear CKA matrix
- Writes CSV and heatmap PNG outputs.

## 10. Current experiment protocol (paper alignment)

Current planned reporting setup:

1. Build held-out harmful prompts by language/category (external generation process).
2. Run baseline outputs from base models.
3. Run steered outputs with identical prompts.
4. Score both with the same LLM-judge rubric.
5. Compare:
   - baseline harmful rate
   - steered harmful rate
   - safe-rate gain
   - latency overhead

`paper.tex` has been aligned to this baseline-vs-steered framing.

## 11. Kaggle execution

Notebook:
- `kaggle_end_to_end.ipynb`

Run order inside notebook:
1. Setup + repo clone + env placeholders
2. Optional dataset build
3. Sarvam-1 extraction -> export -> analyses
4. OpenHathi 4-bit extraction -> export -> analyses
5. Copy all artifacts to `/kaggle/working/safesteer_outputs/`
6. Zip as `/kaggle/working/safesteer_outputs.zip`

## 12. Environment variables and secrets

Common variables:
- `HF_TOKEN`
- `ANTHROPIC_API_KEY` (optional)
- `SYNTH_MODEL` (optional)
- Azure keys (optional)

Never hardcode secrets in tracked files.

## 13. Artifact versioning and Git LFS

Large model/vector artifacts are tracked with LFS via `.gitattributes`.

Tracked patterns include:
- `*.pt`
- `*.bin`
- `*.safetensors`
- classifier checkpoint directory
- steering vector layer files

Run before pushing large artifacts:

```bash
git lfs install
git add .gitattributes
```

## 14. Known caveats

1. Local tokenizer error if `sentencepiece` is missing.
2. Some metrics in base `metrics.py` are heuristic unless Azure/manual judge is used.
3. Synthetic-only training data can reduce domain realism; report this limitation explicitly.
4. 4-bit quantization depends on GPU + bitsandbytes compatibility.

## 15. Reproducibility checklist

For each experiment run, log:
- model key
- language/category subset
- extraction flags (`--no-probe`, `--probe-only`, `--quantize-4bit`)
- generation params (`max_prompts`, `max_tokens`, alpha policy)
- judge model and rubric version
- commit hash

This ensures traceable table/figure generation for the paper.

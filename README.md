# SafeSteer-IN

**Inference-time Safety Steering for Indic Language Models via Contrastive Activation Addition**

SafeSteer-IN is a system for post-hoc safety control of Indic language models (LLMs) through activation-space steering, requiring no retraining or weight modification of the target model. The system decomposes into two operationally distinct pipelines: an offline artifact generation phase and a runtime inference phase, enabling efficient deployment where safety calibration occurs once while inference-time control remains lightweight and composable.

## Overview

### Core Architecture

SafeSteer-IN operates via two coupled pipelines:

1. **Offline Pipeline**: One-time construction of safety artifacts (contrastive datasets, steering vectors, classifier models, and calibration parameters).
2. **Inference Pipeline**: Runtime classification and steering application with slice-specific activation vectors and adaptive fallback logic.

### Safety Scope

The system addresses 8 India-contextualized harm categories across 9 language settings:

**Harm Categories**:
- Communal and religious hate speech (communal_religious_hate)
- Caste-based discrimination and dehumanization (caste_discrimination)
- Political misinformation and fabricated claims (political_misinformation)
- Gender-based violence normalization (gender_based_violence)
- Code-mixed toxicity in Hinglish and other blends (code_mixed_toxicity)
- Anti-minority sentiment targeting marginalized groups (anti_minority_sentiment)
- Child safety and exploitation facilitation (child_safety)
- Financial scam and fraud scripts (financial_scam)

**Language Settings**: Hindi (hi), Bengali (bn), Gujarati (gu), Marathi (mr), Tamil (ta), Malayalam (ml), Telugu (te), Kannada (kn), Hinglish (hi-en)

### Technical Approach

For each (language, category, layer) combination, a steering direction is estimated from contrastive safe/unsafe response pairs:

$$\mathbf{v}_{\text{layer}} = \mathbb{E}[\mathbf{h}^{\text{layer}}_{\text{unsafe}}] - \mathbb{E}[\mathbf{h}^{\text{layer}}_{\text{safe}}]$$

At inference time, the computed direction is subtracted from model activations with learnable strength parameter $\alpha$:

$$\mathbf{h}^{\text{layer}} \leftarrow \mathbf{h}^{\text{layer}} - \alpha \cdot \mathbf{v}_{\text{layer}}$$

## Results

### Sarvam-1 (Matched Evaluation Slices, α=12)

Evaluation across 72 matched slices (8 categories × 9 languages) on the Sarvam-1 model demonstrates:

| Metric | Baseline | With Steering (α=12) | Reduction |
|--------|----------|----------------------|-----------|
| Mean Harmful Rate | 73.47% | 41.34% | 32.13 pp |
| Relative Reduction | — | — | 43.73% |

**Category-wise breakdown (α=12 vs. baseline)**:
- Child safety: 96.30% → 33.70% (62.59 pp)
- Code-mixed toxicity: 79.63% → 24.07% (55.56 pp)
- Financial scam: 94.07% → 55.19% (38.89 pp)
- Communal/religious hate: 84.81% → 46.67% (38.15 pp)
- Caste discrimination: 50.74% → 29.26% (21.48 pp)
- Political misinformation: 49.26% → 31.48% (17.78 pp)
- Gender-based violence: 82.59% → 70.00% (12.59 pp)
- Anti-minority sentiment: 50.37% → 40.37% (10.00 pp)

### OpenHathi Hindi (Alpha Sweep)

Evaluation on OpenHathi-7B-Hindi baseline shows monotonic improvement across α values:

| Alpha | Harmful Rate | Total Reduction (pp) |
|-------|-------------|----------------------|
| Baseline | 85.83% | — |
| α=1 | 50.00% | 35.83 |
| α=4 | 44.13% | 41.71 |
| α=8 | 37.88% | 47.96 |
| α=12 | 30.38% | 55.46 |
| α=15 | 27.13% | 58.71 |

The monotonic decline across the tested range indicates sustained sensitivity to steering without saturation effects within the evaluated parameter space.

## System Requirements

- Python 3.10+
- CUDA 12.0+ (for GPU inference)
- 16–50 GB VRAM depending on target model and quantization

## Installation

### Virtual Environment and Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install sentencepiece openpyxl
```

### Optional Dependencies

For Claude-powered synthetic data augmentation:

```bash
pip install anthropic
```

For 4-bit quantization (recommended for larger models on memory-constrained hardware):

```bash
pip install bitsandbytes
```

### Configuration

Set the following environment variables as needed:

```bash
export HF_TOKEN=<your_huggingface_token>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export SAFESTEER_MODEL=sarvam-1  # or openhathi-base, krutrim-2-instruct
```

## Repository Structure

```
SafeSteer-IN/
├── config.py                          # Global configuration and model definitions
├── data/
│   ├── taxonomy.py                   # Harm category definitions and metadata
│   ├── templates.py                  # Seed contrastive templates (9 languages)
│   ├── synthetic_generation.py       # Synthetic pair generation with optional Claude
│   ├── build_dataset.py              # Dataset construction and splitting
│   └── datasets/                     # Generated JSONL splits and expanded test data
├── steering/
│   ├── extract_vectors.py            # Vector extraction and model loading
│   ├── hooks.py                      # PyTorch forward hooks for activation collection and injection
│   └── calibrate_alpha.py            # Alpha parameter sweep and selection
├── classifier/
│   ├── train_classifier.py           # IndicBERT multi-task training (language + category)
│   └── inference.py                  # Runtime prompt classification and risk scoring
├── engine/
│   ├── steering_engine.py            # Vector loading and steered generation API
│   └── pipeline.py                   # Full inference orchestration
├── evaluation/
│   ├── evaluate.py                   # Evaluation runners and export modes
│   ├── metrics.py                    # Safety metrics and fluency scoring
│   ├── azure_safety.py               # Optional Azure Content Safety integration
│   └── results/                      # Metric CSVs, exports, and plots
├── scripts/
│   ├── 01_build_dataset.py
│   ├── 02_extract_vectors.py
│   ├── 03_train_classifier.py
│   ├── 04_calibrate_alpha.py
│   ├── 05_evaluate.py
│   └── 06_launch_demo.py
├── app.py                            # Gradio web interface
├── api.py                            # FastAPI REST server
├── steering_vectors/                 # Saved vector artifacts per model/language/category
└── models/                           # Classifier checkpoints and label maps
```

## Execution Workflow

### Phase I: Offline Artifact Generation

The offline phase needs to execute once to generate all necessary steering vectors, classifiers, and calibration parameters.

#### Stage 1.1: Dataset Construction

```bash
python scripts/01_build_dataset.py \
  --no-augment \
  --min-pairs-per-slice 30 \
  --seed 42
```

**Inputs**: Seed templates from `data/templates.py`, optionally Claude API for synthetic augmentation.

**Outputs**:
- `data/datasets/train.jsonl` (80% of data)
- `data/datasets/val.jsonl` (10% of data)
- `data/datasets/test.jsonl` (10% of data)

**Dataset Schema**:
```json
{
  "prompt": "...",
  "safe_response": "...",
  "unsafe_response": "...",
  "language": "hi",
  "category": "communal_religious_hate"
}
```

#### Stage 1.2: Steering Vector Extraction

**For Sarvam-1 (9-language track)**:

```bash
python scripts/02_extract_vectors.py \
  --model sarvam-1 \
  --languages hi ta bn gu mr hi-en te kn ml \
  --categories communal_religious_hate caste_discrimination political_misinformation \
                gender_based_violence code_mixed_toxicity anti_minority_sentiment \
                child_safety financial_scam \
  --no-probe
```

**For OpenHathi-7B-Hindi (4-bit quantization)**:

```bash
python scripts/02_extract_vectors.py \
  --model openhathi-base \
  --quantize-4bit \
  --languages hi \
  --categories communal_religious_hate caste_discrimination political_misinformation \
                gender_based_violence code_mixed_toxicity anti_minority_sentiment \
                child_safety financial_scam \
  --no-probe
```

**Outputs**:
- `steering_vectors/<model>/<language>/<category>/layer{N}.pt` (per-layer vectors)
- `steering_vectors/<model>/<language>/<category>/metadata.json` (extraction stats and layer info)

#### Stage 1.3: Prompt Router Training

```bash
python scripts/03_train_classifier.py \
  --epochs 10 \
  --batch-size 16 \
  --lr 2e-5
```

**Model Architecture**:
- Backbone: `ai4bharat/IndicBERTv2-MLM-only`
- Task 1: Language classification (9 classes)
- Task 2: Harm category classification (8 classes)

**Outputs**:
- `models/indic_risk_classifier/classifier.pt`
- `models/indic_risk_classifier/label_maps.json` (language/category ID mappings)

#### Stage 1.4: Alpha Calibration (Optional)

```bash
python scripts/04_calibrate_alpha.py \
  --model sarvam-1 \
  --alphas 5 10 15 20 25 30 35 40 \
  --max-pairs 30
```

Produces calibrated steering strength parameters stored alongside vectors for later retrieval.

#### Stage 1.5: Evaluation and Export

```bash
python scripts/05_evaluate.py \
  --model sarvam-1 \
  --test-dir data/datasets/expanded \
  --steered-only \
  --max-prompts 200 \
  --max-tokens 128
```

**Input Format** (CSV/XLSX):
```
prompt_id,language,category,prompt_text
001,hi,communal_religious_hate,"तोड़ो अन्य धर्मों को..."
```

**Outputs**:
- `evaluation/results/steered_exports/<model>/all_categories_steered.csv`
- `evaluation/results/steered_exports/<model>/all_categories_steered.json`
- Per-category export CSVs with steered outputs and latency

### Phase II: Runtime Inference

Once artifacts are generated, the runtime system enables safe steering during live inference.

#### Option A: Interactive Gradio Interface

```bash
python scripts/06_launch_demo.py
```

Launches web UI at `http://localhost:7860` with:
- Model selector
- Language/category selection or auto-detection
- Alpha adjustment slider
- Side-by-side raw vs. steered output comparison

#### Option B: FastAPI REST API

```bash
python scripts/06_launch_demo.py --api-only
```

Serves on `0.0.0.0:8000` with endpoints:

- `POST /steer`: Full pipeline (classify → steer → score)
- `POST /generate`: Baseline generation without steering
- `GET /health`: Liveness check
- `GET /slices`: List available (language, category) vector pairs

**Example Request**:
```bash
curl -X POST http://localhost:8000/steer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "नॉर्थईस्ट के लोग भारतीय नहीं हैं।",
    "language": "auto",
    "category": "auto",
    "alpha": 12.0,
    "always_steer": true,
    "max_new_tokens": 128
  }'
```

## Supported Models

| Model | Size | Quantization | VRAM | Language Kit |
|-------|------|--------------|------|--------------|
| Sarvam-1 | 2–3B | None | 8–12GB | Multilingual |
| OpenHathi-7B-Hi | 7B | 4-bit | 6–8GB | Hindi/English |
| Krutrim-2-Instruct | 12B | 4-bit | 12–16GB | Multilingual |

## Design Principles

### No Target Model Retraining

The target LLM weights remain frozen at all times. All safety intervention occurs via activation-space steering without gradient updates to model parameters.

### Slice-Based Intervention

Steering is applied per (language, category) combination, allowing fine-grained control and potential category-specific calibration. Fallback logic ensures graceful degradation when exact vectors are unavailable.

### Inference-Time Flexibility

Steering strength (α) and routing decisions are decoupled from model training, enabling adaptive safety control and A/B testing without redeployment.

### Composable Metrics

The system produces intermediate artifacts (vectors, classifications, scores) suitable for external analysis or judge-based evaluation protocols.

## Troubleshooting

**Q: Model runs out of memory during vector extraction.**
- A: Use `--quantize-4bit` flag or reduce batch size. Extract vectors per-language or per-category in separate runs.

**Q: Steering produces incoherent outputs.**
- A: Reduce α or enable coherence-aware calibration. Check `evaluation/results/` for category-wise harm-coherence tradeoff analysis.

**Q: Classifier predicts unexpected language/category.**
- A: Verify input text is in the expected language. Use `force_language` and `force_category` parameters to override auto-detection.

**Q: No steering applied even with `always_steer=True`.**
- A: Check that vectors exist under `steering_vectors/<model>/<language>/<category>/`. Review classifier predictions via API `/slices` endpoint.

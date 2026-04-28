# SafeSteer-IN Technical Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture and Design](#architecture-and-design)
3. [Core Components](#core-components)
4. [Offline Pipeline](#offline-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Configuration System](#configuration-system)
7. [Artifact Specifications](#artifact-specifications)
8. [Integration and Deployment](#integration-and-deployment)
9. [Performance Characteristics](#performance-characteristics)

---

## System Overview

SafeSteer-IN is a modular inference-time safety steering system designed for Indic language models. The system operates under the principle that safety control can be decoupled from model training, allowing post-hoc application of learned safety directions via activation-space manipulation.

### Problem Statement

Existing safety alignment systems for Indic language models face two primary challenges:

1. **Multilingual Coverage**: English-centric safety benchmarks and techniques do not adequately capture India-specific harmful patterns across linguistically diverse settings.
2. **Computational Cost vs. Flexibility**: Fine-tuning large language models for safety across multiple languages and harm categories is computationally prohibitive and difficult to adapt when new categories or languages emerge.

### Solution Overview

SafeSteer-IN addresses these challenges through:

- **Contrastive Activation Addition**: Estimating language and category-specific steering directions from safe/unsafe response pairs in activation space.
- **Slice-Based Control**: Applying steering per (language, category, layer) tuple, enabling granular intervention without model retraining.
- **Inference-Time Application**: Using PyTorch forward hooks to apply steering during generation, maintaining compatibility with frozen model weights.

---

## Architecture and Design

### Two-Phase Operational Model

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Offline Artifact Generation (GPU-intensive, one-time)   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Construction → Vector Extraction → Router Training        │
│  (src/scripts/01)     (src/scripts/02)     (src/scripts/03)     │
│        ↓                   ↓                   ↓                 │
│   train.jsonl       vectors/           classifier.pt           │
│   val.jsonl      metadata.json        label_maps.json          │
│   test.jsonl                                                    │
│                                                                  │
│  Alpha Calibration → Evaluation & Export                        │
│  (src/scripts/04)          (src/scripts/05)                     │
│        ↓                        ↓                               │
│   calibration.json    steered_exports/                         │
│                       *.csv, *.json                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        Artifact Persistence
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Runtime Inference (Lightweight, stateless)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Input → Classifier → Vector Selection → Hook Injection    │
│                (language,     (steering_vec,   (residual update  │
│                 category,      alpha)           in fwd pass)     │
│                 risk_score)                                     │
│        ↓           ↓              ↓                   ↓         │
│    Prompt    RiskPredictor  SteeringEngine    Target LLM       │
│                                                     ↓           │
│                                          Steered Response       │
│                                          + Metadata             │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Offline computation (heavy GPU work) is decoupled from runtime inference (lightweight stateless operations).
2. **Composability**: Each component produces intermediate outputs suitable for analysis, modification, or external integration.
3. **Graceful Degradation**: Missing vectors or classifier predictions trigger well-defined fallback behaviors rather than failures.
4. **Reproducibility**: All configurations, random seeds, and hyperparameters are centralized in `config.py`.

---

## Core Components

### 1. Configuration Module (`config.py`)

Centralized configuration system that defines all project-wide constants, model specifications, and default parameters.

#### Key Structures

**Model Configuration**:
```python
MODEL_CONFIGS = {
    "sarvam-1": {
        "model_id": "sarvamai/sarvam-1",
        "num_layers": 28,
        "hidden_dim": 2048,
        "target_layers": list(range(8, 19, 2)),  # [8,10,12,14,16,18]
        "primary_layer": 12,
        "layer_accessor": "model.layers",
        "chat_format": None,  # base/completion model
        "requires_high_vram": False
    },
    # ... additional models
}
```

Each model entry defines:
- `model_id`: Hugging Face model identifier
- `num_layers`: Total transformer layers
- `hidden_dim`: Transformer hidden state dimension
- `target_layers`: Layers for vector extraction (computational budget optimization)
- `primary_layer`: Default layer for steering when not specified
- `layer_accessor`: ModuleList path within the model architecture
- `chat_format`: Formatting wrapper ('chatml', 'tulu', or None)
- `requires_high_vram`: Hardware requirement flag

**Language and Category Sets**:
```python
LANGUAGES = {
    "hi": "Hindi", "bn": "Bengali", "gu": "Gujarati", ... 
}

HARM_CATEGORIES = {
    0: "communal_religious_hate",
    1: "caste_discrimination",
    # ... 8 categories total
}
```

### 2. Taxonomy Module (`src/data/taxonomy.py`)

Defines the safety taxonomy with structured metadata for each harm category.

```python
@dataclass
class HarmCategory:
    id: int                           # Fixed category ID (0-7)
    key: str                          # Unique identifier
    name: str                         # Display name
    description: str                  # Operational definition
    examples: List[str]               # Example manifestations
    priority_languages: List[str]     # Languages with most coverage
    keywords_hi: List[str]            # Hindi harm signal keywords
    keywords_en: List[str]            # English harm signal keywords
```

Each category is designed around India-specific contexts that are underrepresented in English-centric safety research.

### 3. Template System (`src/data/templates.py`)

Hand-crafted seed contrastive templates for each (language, category) combination.

**Template Schema**:
```json
{
  "prompt": "User query in native language",
  "safe": "Responsible, factually grounded response",
  "unsafe": "Harmful completion the model might produce"
}
```

Templates serve as:
- Initialization points for synthetic expansion
- Quality anchors ensuring culturally appropriate content
- Coverage guarantees across all language-category combinations

### 4. Synthetic Generation (`src/data/synthetic_generation.py`)

Augments seed templates with synthetically generated pairs to increase per-slice coverage.

**Generation Pipeline**:

1. **Claude API Mode** (if `ANTHROPIC_API_KEY` is set):
   - Constructs category and language-aware prompts
   - Calls Claude to generate realistic contrastive pairs
   - Validates output schema and normalizes results

2. **Fallback Deterministic Mode** (no API key):
   - Applies prefix/suffix variation to seed pairs
   - Generates variant markers ('संक्षेप में', 'step-wise', etc.)
   - Ensures minimum pair count per slice through controlled expansion

### 5. Vector Extraction (`src/steering/extract_vectors.py`)

Core mechanism for computing steering directions from contrastive pairs.

**Extraction Pipeline**:

```python
def extract_vectors_for_slice(model, tokenizer, pairs, target_layers):
    # 1. Concatenate prompt + safe_response for all pairs
    safe_texts = [p["prompt"] + " " + p["safe_response"] for p in pairs]
    
    # 2. Collect activations at target layers
    safe_activations = collect_activations(model, tokenizer, safe_texts)
    
    # 3. Repeat for unsafe responses
    unsafe_texts = [p["prompt"] + " " + p["unsafe_response"] for p in pairs]
    unsafe_activations = collect_activations(model, tokenizer, unsafe_texts)
    
    # 4. Compute direction per layer: unsafe_mean - safe_mean
    vectors = {}
    for layer_idx in target_layers:
        direction = unsafe_activations[layer_idx] - safe_activations[layer_idx]
        
        # 5. L2-normalize for stable application
        norm = direction.norm()
        if norm > 0:
            direction = direction / norm
        
        vectors[layer_idx] = direction
    
    return vectors
```

**Activation Collection**:
- Uses `ActivationCollector` to register PyTorch forward hooks
- Captures hidden states at specified transformer layers
- Reduces sequence dimension via mean-pooling over token positions
- Returns per-layer aggregated activations

### 6. Forward Hooks (`src/steering/hooks.py`)

PyTorch implementation of activation capture and steering injection.

**ActivationCollector**:
```python
class ActivationCollector:
    def __init__(self, model, target_layers, layer_accessor):
        # Register forward hooks on specified layers
        # Hooks capture outputs and store in dictionary
        
    def get(self) -> Dict[int, Tensor]:
        # Return {layer_idx: [batch, hidden_dim]} activations
        
    def clear(self):
        # Reset for next forward pass
```

**SteeringHook**:
```python
class SteeringHook:
    def __init__(self, model, layer_idx, vector, alpha):
        # Register hook that applies: hidden = hidden - alpha * vector
        
    def _hook_fn(self, module, input, output):
        # Intercept layer output and apply residual steering
        # Maintains output tuple structure for compatibility
```

**Hook Registration Pattern**:
```python
hook = SteeringHook(model, layer=12, vector=steering_vec, alpha=15.0)
output = model.generate(input_ids, max_new_tokens=128)
hook.remove()  # Clean up after generation
```

### 7. Classifier Training (`src/classifier/train_classifier.py`)

Multi-task classification system for runtime prompt routing.

**Architecture**:
```
Input Text (prompt)
       ↓
IndicBERT Backbone (ai4bharat/IndicBERTv2-MLM-only)
       ↓
[CLS] Token Representation
       ↓
   ┌───┴───┐
   ↓       ↓
Language Head    Category Head
(9 classes)      (8 classes)
   ↓       ↓
Language  Harm Category
Prediction Prediction
```

**Training Details**:
- Multi-task loss: $L = L_{\text{language}} + L_{\text{category}}$
- Backbone frozen for first 2 epochs, unfrozen after with reduced learning rate
- Input: Prompt text only (from contrastive dataset)
- Labels: Language code + harm category index

### 8. Steering Engine (`src/engine/steering_engine.py`)

High-level API for vector-based generation with fallback logic.

**Core Functions**:

```python
class SteeringEngine:
    def __init__(self, model, tokenizer, model_key):
        # Load all available vectors from disk
        # Load calibrated alphas if available
        
    def generate(self, prompt, max_new_tokens=256):
        # Baseline generation without steering
        
    def generate_steered(self, prompt, language, category, alpha=None):
        # Steered generation with hook injection
        # Falls back if vector unavailable
        
    def get_vector(self, language, category, layer=None):
        # Lookup vector, returns None if missing
        
    def get_available_layers(self, language, category):
        # List all extracted layers for a slice
```

**Fallback Logic**:
1. Try exact (language, category, layer) combination
2. If missing, try any category in the same language
3. If still missing, try Hindi slice for the category
4. If no vectors available, return unsteered output with warning

### 9. Inference Pipeline (`src/engine/pipeline.py`)

Full orchestration of classification, vector selection, and steering.

```python
class SafeSteerPipeline:
    def run(self, prompt, force_language=None, force_category=None, alpha=None):
        # 1. Classify prompt → language, category, risk_score
        # 2. Determine if steering should be applied
        # 3. Resolve steering vector with fallback
        # 4. Generate baseline and steered outputs
        # 5. Return results with metadata
```

**Risk-Based Steering Decision**:

```python
should_steer = (
    (risk_score >= RISK_THRESHOLD) or 
    always_steer_flag
)

if should_steer and engine.has_vector(language, category):
    steered_output = engine.generate_steered(prompt, language, category)
else:
    steered_output = baseline_output
```

---

## Offline Pipeline

### Stage 1: Dataset Construction

**Entrypoint**: `src/scripts/01_build_dataset.py` / `src/data/build_dataset.py`

**Execution Flow**:

```python
def build_dataset(augment=True, load_hf=False, min_pairs_per_slice=30):
    # 1. Load seed templates from src/data/templates.py
    seed_pairs = load_seed_templates()
    
    # 2. For each (language, category) combination:
    for lang in LANGUAGES:
        for category in HARM_CATEGORIES:
            seed_items = seed_pairs[lang][category]
            
            # 3. Generate synthetic pairs if needed
            if len(seed_items) < min_pairs_per_slice:
                generated = generate_slice_pairs(
                    lang, category, seed_items, 
                    min_pairs=min_pairs_per_slice
                )
                seed_items.extend(generated)
    
    # 4. Combine all pairs, normalize schema
    all_pairs = combine_and_normalize(seed_items)
    
    # 5. Split into train/val/test respecting stratification
    splits = stratified_split(
        all_pairs, 
        ratios={'train': 0.8, 'val': 0.1, 'test': 0.1}
    )
    
    # 6. Write JSONL files
    write_splits(splits)
```

**Dataset Statistics**:
- Typical result: ~1,200-2,000 pairs per language
- ~72 slices (8 categories × 9 languages)
- ~1–3K total pairs after stratification

### Stage 2: Vector Extraction

**Entrypoint**: `src/scripts/02_extract_vectors.py` / `src/steering/extract_vectors.py`

**Execution for Sarvam-1 (9-language)**:

```bash
python src/scripts/02_extract_vectors.py \
  --model sarvam-1 \
  --languages hi ta bn gu mr hi-en te kn ml \
  --categories communal_religious_hate caste_discrimination ... \
  --split train \
  --max-length 512
```

**Memory and Timing**:
- Per-category extraction: ~2–5 minutes on V100
- Activation collection: 1–2 GPU hours for full 9-language run
- Output per slice: 1 KB per vector file (6–8 layers) + 2 KB metadata

**Artifact Output Structure**:

```
steering_vectors/sarvam-1/
├── hi/
│   ├── communal_religious_hate/
│   │   ├── layer8.pt
│   │   ├── layer10.pt
│   │   ├── layer12.pt
│   │   └── metadata.json
│   └── [7 more categories]...
├── [8 more languages]...
```

**Metadata Schema**:
```json
{
  "model": "sarvam-1",
  "language": "hi",
  "category": "communal_religious_hate",
  "num_pairs": 40,
  "layers": {
    "8": {"file": "layer8.pt", "norm": 1.000, "shape": [2048]},
    "10": {"file": "layer10.pt", "norm": 0.995, "shape": [2048]},
    ...
  }
}
```

### Stage 3: Classifier Training

**Entrypoint**: `src/scripts/03_train_classifier.py` / `src/classifier/train_classifier.py`

**Training Configuration**:

```python
train_classifier(
    train_pairs=load_dataset_split("train"),
    val_pairs=load_dataset_split("val"),
    epochs=10,
    batch_size=16,
    lr=2e-5,
    device="cuda"
)
```

**Training Schedule**:
- Epochs 1–2: Freeze IndicBERT backbone, train heads only
- Epoch 3+: Unfreeze backbone with 10x reduced learning rate
- Validation: Check combined language + category accuracy
- Early stopping: Save checkpoint on best validation score

**Output Artifacts**:
- `models/indic_risk_classifier/classifier.pt` (state dict)
- `models/indic_risk_classifier/label_maps.json` (ID mappings)

### Stage 4: Alpha Calibration

**Entrypoint**: `src/scripts/04_calibrate_alpha.py` / `src/steering/calibrate_alpha.py`

**Calibration Objective**:

For each (language, category, layer) slice, find α that maximizes:

$$\text{objective}(\alpha) = \text{safety\_rate}(\alpha) \quad \text{s.t.} \quad \text{perplexity\_increase}(\alpha) < 20\%$$

**Sweep Strategy**:

```python
def calibrate_alpha(model, pairs, vector, layer_idx, alphas=[5,10,15,20,...]):
    for alpha in alphas:
        # Generate harmful prompts with steering at alpha
        safe_count = count_safe_outputs(harmful_prompts, alpha)
        safety_rate = safe_count / len(harmful_prompts)
        
        # Measure fluency on benign prompts with steering
        perplexity_steered = mean([compute_ppl(t) for t in benign_texts])
        perplexity_baseline = mean([compute_ppl(t) for t in benign_texts])
        ppl_increase = (perplexity_steered - perplexity_baseline) / perplexity_baseline
        
        # Record result
        results[alpha] = {"safety_rate": safety_rate, "ppl_increase": ppl_increase}
    
    # Select alpha with highest safety rate under constraint
    best_alpha = max(
        [a for a in alphas if results[a]["ppl_increase"] < 0.20],
        key=lambda a: results[a]["safety_rate"]
    )
    return best_alpha
```

**Output**: Saved per-slice calibration results for runtime retrieval.

### Stage 5: Evaluation and Export

**Entrypoint**: `src/scripts/05_evaluate.py` / `src/evaluation/evaluate.py`

**Export Mode**:

```bash
python src/scripts/05_evaluate.py \
  --model sarvam-1 \
  --test-dir data/datasets/expanded \
  --steered-only \
  --max-prompts 200 \
  --max-tokens 128
```

**Processing**:
1. Read CSV/XLSX test files (auto-detects columns)
2. For each prompt:
   - Run classifier to predict language/category
   - Generate steered output with timer
   - Record latency, steering_applied flag, model info
3. Aggregate by category and export

**Output Format** (`all_categories_steered.csv`):

```csv
source_file,prompt_id,language,category,prompt_text,steered_output,latency_ms,steering_applied,model
test_file.xlsx,001,hi,communal_religious_hate,"भारत में...",<response>,2100.45,True,sarvam-1
```

---

## Inference Pipeline

### Runtime System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        Gradio          FastAPI      Programmatic
        (app.py)        (api.py)      (engine/)
          │              │              │
          └──────────────┼──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  SafeSteerPipeline.run()    │
          │  ├─ Classification          │
          │  ├─ Vector Selection        │
          │  ├─ Fallback Logic          │
          │  └─ Orchestration           │
          └──────────────┬──────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    RiskPredictor              SteeringEngine
    (classifier/)              (steering/)
    ├─ Language pred.          ├─ Vector lookup
    ├─ Category pred.          ├─ Hook injection
    └─ Risk scoring            └─ Generation
          │                             │
          │                    ┌────────┘
          │                    ▼
          │            ┌──────────────────┐
          │            │  Target LLM      │
          │            │  (frozen weights)│
          │            └────────┬─────────┘
          │                     │
          └─────────┬───────────┘
                    ▼
          ┌──────────────────────┐
          │   PipelineResult     │
          ├─ raw_output         │
          ├─ steered_output     │
          ├─ latency_ms         │
          ├─ language           │
          ├─ category           │
          ├─ risk_score         │
          └─ steering_applied   │
          
```

### Request Lifecycle

**1. Prompt Input**:
- User provides prompt text
- Optional: force language/category override
- Optional: specify steering strength (α)

**2. Classification**:
```python
classification = classifier.predict(prompt)
# Returns: {
#   "language": "hi",
#   "category": "communal_religious_hate",
#   "risk_score": 0.78,
#   "lang_probs": {...},
#   "cat_probs": {...}
# }
```

**3. Baseline Generation**:
```python
raw_output = engine.generate(prompt, max_new_tokens=256)
```

**4. Steering Decision**:
```python
should_apply_steering = (
    (classification["risk_score"] >= RISK_THRESHOLD) or
    always_steer_flag
)
```

**5. Vector Resolution with Fallback**:
```python
vector = engine.get_vector(language, category)
if vector is None:
    # Try same language, other categories
    available_in_lang = engine.get_available_layers(language, category)
    if not available_in_lang:
        # Fall back to Hindi
        language = "hi"
        vector = engine.get_vector("hi", category)
```

**6. Steered Generation**:
```python
if vector is not None:
    hook = SteeringHook(model, layer=primary_layer, vector=vector, alpha=α)
    steered_output = engine.generate(prompt, ...)
    hook.remove()
else:
    steered_output = raw_output
    steering_applied = False
```

**7. Response Assembly**:
```python
result = PipelineResult(
    prompt=prompt,
    detected_language=classification["language"],
    detected_category=classification["category"],
    risk_score=classification["risk_score"],
    steering_applied=vector is not None,
    raw_output=raw_output,
    steered_output=steered_output,
    latency_ms=(end_time - start_time) * 1000
)
return result
```

### Interface Options

#### Gradio Web Interface (`app.py`)

**Launch**:
```bash
python app.py
# or
python src/scripts/06_launch_demo.py
```

**Features**:
- Model selector with info panels
- Prompt input with language/category selectors
- Alpha adjustment slider (0–30)
- Steering toggle
- Side-by-side output display
- Pre-loaded example prompts for quick testing

**Architecture**:
- Lazy pipeline loading per model
- Singleton caching to avoid reloading
- Responsive updates with Gradio event handlers

#### FastAPI REST API (`api.py`)

**Launch**:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness check |
| `/slices` | GET | List available (language, category) pairs |
| `/steer` | POST | Full pipeline: classify → steer → return |
| `/generate` | POST | Baseline generation without steering |

**Request Schema** (`/steer`):
```json
{
  "prompt": "तोड़ो अन्य धर्मों को...",
  "language": "auto",
  "category": "auto",
  "alpha": 12.0,
  "always_steer": true,
  "max_new_tokens": 128
}
```

**Response Schema**:
```json
{
  "prompt": "...",
  "detected_language": "hi",
  "detected_category": "communal_religious_hate",
  "risk_score": 0.82,
  "steering_applied": true,
  "alpha_used": 12.0,
  "raw_output": "...",
  "steered_output": "...",
  "latency_ms": 2500.34
}
```

---

## Configuration System

### Configuration File (`config.py`)

Master configuration with hierarchical organization:

**Model Specifications**:
```python
MODEL_CONFIGS = {...}  # Detailed below
ACTIVE_MODEL = os.getenv("SAFESTEER_MODEL", "sarvam-1")
```

**Language and Category Sets**:
```python
LANGUAGES = {"hi": "Hindi", ...}
HARM_CATEGORIES = {0: "communal_religious_hate", ...}
CATEGORY_TO_ID = {...}
```

**Dataset Configuration**:
```python
DATASET_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
PAIRS_PER_CATEGORY = {
    "hi": 200, "ta": 150, "bn": 150, ...,
}
```

**Steering Parameters**:
```python
DEFAULT_ALPHA = 15.0
ALPHA_SEARCH_RANGE = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
MAX_PERPLEXITY_INCREASE = 0.20  # 20%
RISK_THRESHOLD = 0.5
```

**Model Loading and Generation**:
```python
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
USE_4BIT = False  # Override per-run via CLI flag
BNB_4BIT_CONFIG = {...}  # BitsAndBytes configuration
```

**Infrastructure**:
```python
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORS_DIR = PROJECT_ROOT / "steering_vectors"
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation" / "results"
```

### Runtime Configuration Override

Configuration values can be overridden via environment variables:

```bash
export HF_TOKEN=<token>
export SAFESTEER_MODEL=openhathi-base
export ANTHROPIC_API_KEY=<key>
```

Or programmatically:
```python
from engine.pipeline import SafeSteerPipeline
pipeline = SafeSteerPipeline(
    model_key="sarvam-1",
    risk_threshold=0.6,
    default_alpha=20.0
)
```

---

## Artifact Specifications

### Vector Artifacts

**File Format**: PyTorch tensor (`.pt`)

```python
# Load a vector
vector = torch.load("steering_vectors/sarvam-1/hi/communal_religious_hate/layer12.pt")
# Shape: (2048,) for Sarvam-1
# Normalized: L2 norm = 1.0
# Dtype: float32
```

**Storage Layout**:
```
steering_vectors/
├── sarvam-1/                    # Model name
│   ├── hi/                      # Language code
│   │   ├── communal_religious_hate/  # Category key
│   │   │   ├── layer8.pt        # Vector for layer 8
│   │   │   ├── layer10.pt       # Vector for layer 10
│   │   │   ├── layer12.pt
│   │   │   ├── layer14.pt
│   │   │   ├── layer16.pt
│   │   │   ├── layer18.pt
│   │   │   └── metadata.json    # Extraction metadata
│   │   └── [7 more categories]
│   └── [8 more languages]
├── openhathi-base/
└── krutrim-2-instruct/
```

### Metadata JSON Schema

```json
{
  "model": "sarvam-1",
  "language": "hi",
  "category": "communal_religious_hate",
  "num_pairs": 40,
  "split": "train",
  "target_layers": [8, 10, 12, 14, 16, 18],
  "layers": {
    "8": {
      "file": "layer8.pt",
      "norm": 1.0,
      "shape": [2048],
      "dtype": "float32"
    },
    ...
  },
  "extraction_config": {
    "max_length": 512,
    "reduction": "mean",
    "batch_size": 1
  },
  "timestamp": "2025-03-29T12:30:00Z"
}
```

### Dataset JSONL Schema

Each line is a JSON object:

```json
{
  "prompt": "भारत में मुसलमानों के बारे में क्या सच है?",
  "safe_response": "भारत में मुसलमान समाज का एक महत्वपूर्ण हिस्सा...",
  "unsafe_response": "मुसलमान देश के लिए खतरा हैं। उनकी...",
  "language": "hi",
  "category": "communal_religious_hate",
  "source": "seed_template"
}
```

### Classifier Checkpoint

```
models/indic_risk_classifier/
├── classifier.pt          # Model state dict (torch.save)
├── label_maps.json        # Language/category ID mappings
├── config.json            # Model configuration
└── tokenizer_config.json  # IndicBERT tokenizer config
```

**Loading**:
```python
from classifier.inference import RiskPredictor
predictor = RiskPredictor(checkpoint_dir="models/indic_risk_classifier")
result = predictor.predict("कुछ प्रश्न...")
```

---

## Integration and Deployment

### Programmatic Integration

**Minimal Example**:
```python
from engine.pipeline import SafeSteerPipeline

# Initialize once
pipeline = SafeSteerPipeline(model_key="sarvam-1")
pipeline.load()

# Use repeatedly
for prompt in user_prompts:
    result = pipeline.run(
        prompt=prompt,
        always_steer=True,
        max_new_tokens=128
    )
    print(f"Raw: {result.raw_output}")
    print(f"Steered: {result.steered_output}")
    print(f"Latency: {result.latency_ms} ms")
```

**Batch Processing**:
```python
from evaluation.evaluate import run_evaluation

results = run_evaluation(
    model_key="sarvam-1",
    test_dir="data/datasets/expanded",
    steered_only=True,
    max_prompts=500
)
# Outputs CSV/JSON exports
```

### Docker Deployment

**Dockerfile Example**:
```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run**:
```bash
docker build -t safesteer-in .
docker run --gpus all -p 8000:8000 safesteer-in
```

### Cloud Deployment Considerations

**HuggingFace Hub**:
- Vectors can be pushed to HuggingFace Model Hub for versioning
- Classifier checkpoint compatible with standard model repo structure



---

## Performance Characteristics

### Latency Breakdown (Sarvam-1, 2–3B Parameters)

| Stage | Time (ms) | Hardware |
|-------|-----------|----------|
| Classifier (language/category) | 50–100 | GPU |
| Vector lookup | 1–5 | Memory |
| Generation (unsteered, 128 tokens) | 800–1200 | GPU |
| Hook overhead (steered) | +50–100 | GPU |
| **Total (steered, 128 tokens)** | **900–1400** | GPU |

### Memory Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Sarvam-1 model | 2–3B params | 6–8 GB VRAM with mixed precision |
| Steering vectors (1 model) | ~200 MB | All 9 languages × 8 categories × 6 layers |
| Classifier | ~500 MB | IndicBERT backbone |
| Runtime working set | 1–2 GB | Activations, KV cache during generation |
| **Total (Sarvam-1 GPU)** | **8–12 GB** | Typical allocation |

### Throughput

- **Single-request latency**: 1–2 seconds (unsteered + steered)
- **Batch throughput** (batch size 4): 2–4 requests/second
- **Concurrent users** (async, single GPU): 10–20 with queueing

### Extraction Time (Vector Computation)

| Model | Languages | Categories | Time | GPU |
|-------|-----------|-----------|------|-----|
| Sarvam-1 | 9 | 8 | 6–8 hours | V100 |
| OpenHathi-7B (4-bit) | 1 | 8 | 2–3 hours | V100 |

---

## Appendix: Evaluation Results (acl_latex.tex)

### Sarvam-1 Results

**Evaluation Setup**: 
- 72 matched slices (8 categories × 9 languages)
- 30 prompts per language per category
- Total: ~2,160 prompt-response pairs
- α=12 selected as optimal operating point

**Harmful Rate Comparison** (α=12 vs. baseline):

| Category | Baseline | α=12 | Reduction (pp) |
|----------|----------|------|----------------|
| Child safety | 96.30% | 33.70% | 62.59 |
| Code-mixed toxicity | 79.63% | 24.07% | 55.56 |
| Financial scam | 94.07% | 55.19% | 38.89 |
| Communal/religious hate | 84.81% | 46.67% | 38.15 |
| Caste discrimination | 50.74% | 29.26% | 21.48 |
| Political misinformation | 49.26% | 31.48% | 17.78 |
| Gender-based violence | 82.59% | 70.00% | 12.59 |
| Anti-minority sentiment | 50.37% | 40.37% | 10.00 |
| **Overall (mean)** | **73.47%** | **41.34%** | **32.13** |

**Summary**: Sarvam-1 at α=12 achieves overall harmful-rate reduction of 32.13 percentage points (43.73% relative improvement).

### OpenHathi-7B-Hindi Results

**Evaluation Setup**:
- Hindi language only
- All 8 categories evaluated
- Alpha sweep: α ∈ {1, 4, 8, 12, 15}

**Harmful Rate vs. Alpha**:

| Alpha | Harmful Rate | Reduction vs. Baseline (pp) |
|-------|-------------|--------------------------|
| Baseline | 85.83% | — |
| α=1 | 50.00% | 35.83 |
| α=4 | 44.13% | 41.71 |
| α=8 | 37.88% | 47.96 |
| α=12 | 30.38% | 55.46 |
| α=15 | 27.13% | 58.71 |

**Summary**: OpenHathi shows monotonic decline in harmful rate across the α sweep, with 58.71 pp reduction at α=15 (68.3% relative improvement).

### Key Observations

1. **Strong Category-Specific Effects**: Child safety and code-mixed toxicity show largest improvements (55–63 pp), suggesting distinctive activation patterns for these categories.

2. **Resilient Categories**: Gender-based violence and anti-minority sentiment show smaller improvements (10–12 pp), indicating more distributed/subtle activation patterns.

3. **No Saturation**: OpenHathi continues to improve beyond α=15, indicating room for higher steering strengths without diminishing returns in tested range.

4. **Cross-Model Variation**: Steered efficacy varies significantly across model architectures (Sarvam-1 vs. OpenHathi), suggesting architecture-specific tuning may be beneficial.

---

## References

- Turner et al. (2024) "Steering Language Models with Learned Activations"
- Zou et al. (2025) "Representation Engineering: A Top-Down Approach to AI Transparency"
- Pokharel et al. (2026) "Cross-Lingual Activation Steering for Multilingual LLMs"
- Banerjee et al. (2025) "Soteria: Language-Specific Functional Parameters for Safety"

# SafeSteer-IN

**Inference-Time Safety Steering for Indic LLMs using Contrastive Activation Addition (CAA)**

> *Because Viksit Bharat's AI must be safe in every language it speaks.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## Overview

India has **600M+ active regional-language internet users**, yet Indic LLMs (Sarvam-1, OpenHathi, Navarasa) have **zero dedicated safety tooling**. English-trained safety filters miss India-specific harms entirely: communal hate, caste discrimination, code-mixed toxicity, regional political misinformation, and more.

**SafeSteer-IN** applies **Contrastive Activation Addition (CAA)** to steer Indic LLMs toward safe behaviour at inference time — with **zero weight updates**, **under 3ms latency overhead**, and **full auditability**.

### How It Works

```
User Prompt → IndicBERT Classifier → Vector Selector → Hooked LLM → Safe Output
              (language + category)    (steering vec)   (residual    
                                                         stream       
                                                         intervention)
```

1. **Classify** the prompt's language and harm category (IndicBERT, <10ms)
2. **Select** the pre-computed steering vector for that (language, category)
3. **Hook** into the model's residual stream at the target layer
4. **Subtract** the scaled unsafe direction: `hidden -= α × steering_vector`
5. **Generate** a safe response — the model's weights are never modified

---

## India Safety Taxonomy (8 Categories)

| # | Category | Description |
|---|----------|-------------|
| 0 | Communal & Religious Hate | Anti-community incitement, riot glorification |
| 1 | Caste-Based Discrimination | Caste slurs, Dalit dehumanisation, untouchability |
| 2 | Regional Political Misinfo | Fake politician quotes, EVM conspiracy, election fraud |
| 3 | Gender-Based Violence | Vernacular rape threats, honour-killing justification |
| 4 | Code-Mixed Toxicity | Hinglish/Tanglish hate designed to evade English filters |
| 5 | Anti-Minority Sentiment | Anti-NE, anti-tribal, anti-LGBTQ+ in Indian context |
| 6 | Child Safety | Child-marriage justification, POCSO evasion |
| 7 | Financial Scam Facilitation | UPI fraud, KYC phishing in regional languages |

**Languages:** Hindi, Tamil, Bengali, Gujarati, Marathi, Hinglish (code-mixed)

---

## Project Structure

```
SafeSteer-IN/
├── config.py                      # Central configuration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
│
├── data/                          # Dataset module
│   ├── taxonomy.py                # 8 harm categories with metadata
│   ├── templates.py               # Seed contrastive pairs (6 languages)
│   └── build_dataset.py           # Dataset construction pipeline
│
├── steering/                      # Steering vector module
│   ├── hooks.py                   # PyTorch forward-hook utilities
│   ├── extract_vectors.py         # Vector extraction pipeline
│   └── calibrate_alpha.py         # Alpha calibration sweep
│
├── classifier/                    # Risk classifier module
│   ├── train_classifier.py        # IndicBERT fine-tuning
│   └── inference.py               # Inference + rule-based fallback
│
├── engine/                        # Inference engine
│   ├── steering_engine.py         # Core steering engine
│   └── pipeline.py                # Full end-to-end pipeline
│
├── evaluation/                    # Evaluation module
│   ├── metrics.py                 # SIR, FPR, fluency, latency
│   ├── azure_safety.py            # Azure Content Safety wrapper
│   └── evaluate.py                # Full evaluation pipeline
│
├── azure_ml/                      # Azure ML integration
│   ├── setup.py                   # Workspace + compute setup
│   └── tracking.py                # MLflow experiment tracking
│
├── app.py                         # Gradio demo UI
├── api.py                         # FastAPI REST server
│
└── scripts/                       # Step-by-step runner scripts
    ├── 01_build_dataset.py
    ├── 02_extract_vectors.py
    ├── 03_train_classifier.py
    ├── 04_calibrate_alpha.py
    ├── 05_evaluate.py
    └── 06_launch_demo.py
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n safesteer python=3.11 -y
conda activate safesteer

# Install PyTorch (CUDA 12.1 — adjust for your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy and fill in your credentials
cp .env.example .env
# Edit .env with your HuggingFace token and Azure credentials
```

### 3. Run the Pipeline (6 Steps)

```bash
# Step 1: Build the contrastive-pair dataset
python scripts/01_build_dataset.py

# Step 2: Extract steering vectors (requires GPU)
python scripts/02_extract_vectors.py --languages hi hi-en

# Step 3: Train the IndicBERT risk classifier
python scripts/03_train_classifier.py

# Step 4: Calibrate alpha values
python scripts/04_calibrate_alpha.py

# Step 5: Run evaluation
python scripts/05_evaluate.py

# Step 6: Launch the demo
python scripts/06_launch_demo.py
```

---

## Azure Integration (Required for Hackathon)

### Azure Student Pack Setup

1. **Azure ML Workspace:**
   - Go to [Azure Portal](https://portal.azure.com)
   - Create Machine Learning workspace: `safesteer-in-ws`
   - Resource group: `safesteer-rg`
   - Run: `python -m azure_ml.setup`

2. **Azure AI Content Safety:**
   - Create an Azure AI Content Safety resource
   - Copy endpoint and key to `.env`

3. **Set environment variables in `.env`:**
   ```
   AZURE_CONTENT_SAFETY_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   AZURE_CONTENT_SAFETY_KEY=your-key-here
   AZURE_ML_SUBSCRIPTION_ID=your-subscription-id
   ```

### Azure Services Used

| Service | Purpose |
|---------|---------|
| Azure ML Workspace | Experiment tracking, model registry, artifact store |
| Azure AI Content Safety | Independent safety-score baseline for evaluation |
| Azure Compute (optional) | GPU cluster for parallel vector extraction |

---

## Supported Models

| Model | HuggingFace ID | Priority |
|-------|---------------|----------|
| OpenHathi | `sarvamai/OpenHathi-7B-Hi-v0.1` | Primary |
| Sarvam-1 | `sarvamai/sarvam-1` | Secondary |
| Navarasa 2.0 | `Telugu-LLM-Labs/Navarasa-2.0` | South Indian |

All models load with **4-bit quantisation** (bitsandbytes NF4) to fit in 16GB GPU memory.

To switch models, set `SAFESTEER_MODEL=sarvam-1` in `.env`.

---

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| SIR | Safety Improvement Rate | > 70% |
| FPR | False Positive Rate (benign prompts incorrectly steered) | < 10% |
| Fluency | Perplexity preservation on benign inputs | > 0.85 |
| Latency | Additional inference time from steering hook | < 3ms |

---

## API Usage

### FastAPI Server

```bash
# Start the API
python api.py
# Or: uvicorn api:app --host 0.0.0.0 --port 8000
```

```python
import requests

response = requests.post("http://localhost:8000/steer", json={
    "prompt": "भारत में मुसलमानों के बारे में क्या सच है?",
    "language": "hi",
    "always_steer": True,
    "alpha": 15.0,
})

result = response.json()
print("Raw:", result["raw_output"])
print("Steered:", result["steered_output"])
```

### Gradio Demo

```bash
python app.py
# Open http://localhost:7860
```

---

## Compute-Constrained Fallbacks

If you don't have GPU access:

1. **Use CPU with quantisation:** Set `USE_4BIT = True` in config (default). Slower but works.
2. **Reduce dataset:** Run with `--languages hi hi-en` only (2 languages instead of 6).
3. **Skip probe:** Use `--no-probe` to extract from all candidate layers without the probe study.
4. **Fewer pairs:** Edit `PAIRS_PER_CATEGORY` in config to reduce to 50 per category.

---

## References

- Panickssery et al. (2024) — *Steering LLM activations: Refusal is mediated by a single direction*
- Zou et al. (2023) — *Representation Engineering: A top-down approach to AI transparency*
- Wang et al. (2025) — *Cross-Lingual Activation Steering (CLAS)*
- AI4Bharat (2024) — *IndicLLMSuite*
- Soteria (EMNLP 2025) — *Language-specific functional parameter steering for safety*

---

## License

Apache 2.0 — Steering vectors and India Safety Dataset released under CC BY 4.0.

---

**SafeSteer-IN** · Microsoft AI Unlocked · Track 5: Trustworthy AI  
Team Vishal Megamart Security Guards · IIT Madras

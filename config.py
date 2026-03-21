"""
SafeSteer-IN  ·  Central Configuration
======================================
All project-wide settings: paths, model IDs, Azure creds, steering params.
Import this module everywhere instead of hard-coding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets"
VECTORS_DIR = PROJECT_ROOT / "steering_vectors"
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation" / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in [DATASET_DIR, VECTORS_DIR, MODELS_DIR, EVALUATION_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Model Configuration ────────────────────────────────────────────────────
#
# chat_format values:
#   None      → raw completion (base model, no system/user/assistant wrapper)
#   "tulu"    → <|user|>\n{prompt}\n<|assistant|>\n   (Airavata / IndicInstruct)
#   "chatml"  → tokenizer.apply_chat_template()         (sarvam-m)
#
# requires_high_vram → flag shown as a warning in the Gradio UI
#
MODEL_CONFIGS = {
    # ── OpenHathi 7B (Hindi/English/Hinglish base model) ─────────────────
    "openhathi-base": {
        "model_id": "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
        "display_name": "OpenHathi 7B (Hindi base)",
        "description": "LLaMA-2 7B fine-tuned for Hindi, English & Hinglish. Best for raw activation probing.",
        "num_layers": 32,
        "hidden_dim": 4096,
        "target_layers": list(range(10, 23, 2)),  # middle-depth sweep (every 2 layers)
        "primary_layer": 16,
        "layer_accessor": "model.layers",
        "chat_format": None,  # base/completion model
        "requires_high_vram": False,
        "license": "llama2",
    },
    # ── Airavata 7B (Hindi instruction-tuned) ────────────────────────────
    "airavata": {
        "model_id": "ai4bharat/Airavata",
        "display_name": "Airavata 7B (Hindi instruction)",
        "description": "LLaMA-2 7B instruction-tuned on IndicInstruct. Best for Hindi safety evaluation.",
        "num_layers": 32,
        "hidden_dim": 4096,
        "target_layers": list(range(12, 22)),
        "primary_layer": 16,
        "layer_accessor": "model.layers",
        "chat_format": "tulu",  # <|user|>\n...\n<|assistant|>\n
        "requires_high_vram": False,
        "license": "llama2",  # gated — requires HF login
    },
    # ── Sarvam-1 (2-3B lightweight base model) ───────────────────────────
    "sarvam-1": {
        "model_id": "sarvamai/sarvam-1",
        "display_name": "Sarvam-1 ~2B (lightweight)",
        "description": "~2B LLaMA-style model. Fastest on CPU/low-VRAM. Non-commercial license.",
        "num_layers": 28,
        "hidden_dim": 2048,
        "target_layers": list(range(8, 19, 2)),  # middle-depth sweep (every 2 layers)
        "primary_layer": 12,
        "layer_accessor": "model.layers",
        "chat_format": None,  # completion / base model
        "requires_high_vram": False,
        "license": "sarvam-non-commercial",
    },
    # ── Sarvam-M 24B (multilingual reasoning — GPU-server only) ──────────
    "sarvam-m": {
        "model_id": "sarvamai/sarvam-m",
        "display_name": "Sarvam-M 24B (⚠ high VRAM)",
        "description": "Mistral-Small-3.1-24B-based multilingual model. Requires ~50 GB VRAM — GPU server only.",
        "num_layers": 40,
        "hidden_dim": 5120,
        "target_layers": list(range(16, 26)),  # ~40-65 % of 40 layers
        "primary_layer": 20,
        "layer_accessor": "model.layers",
        "chat_format": "chatml",  # tokenizer.apply_chat_template()
        "requires_high_vram": True,  # ~50 GB VRAM needed
        "license": "apache-2.0",
    },
}

ACTIVE_MODEL = os.getenv("SAFESTEER_MODEL", "sarvam-1")


def get_model_config(model_key: str | None = None) -> dict:
    """Return the config dict for *model_key* (defaults to ACTIVE_MODEL)."""
    key = model_key or ACTIVE_MODEL
    if key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key '{key}'. Valid keys: {list(MODEL_CONFIGS)}"
        )
    return MODEL_CONFIGS[key]


def format_prompt(prompt: str, model_key: str | None = None) -> str:
    """
    Wrap *prompt* in the chat template required by the chosen model.

    Returns the raw prompt unchanged for base/completion models.
    """
    cfg = get_model_config(model_key)
    fmt = cfg.get("chat_format")
    if fmt is None:
        return prompt
    if fmt == "tulu":
        return f"<|user|>\n{prompt}\n<|assistant|>\n"
    # "chatml" — caller must use tokenizer.apply_chat_template() directly
    return prompt


# ─── IndicBERT Classifier ───────────────────────────────────────────────────
INDICBERT_MODEL_ID = "ai4bharat/IndicBERTv2-MLM-only"
CLASSIFIER_CHECKPOINT = MODELS_DIR / "indic_risk_classifier"
CLASSIFIER_NUM_CATEGORIES = 8
CLASSIFIER_NUM_LANGUAGES = 9

# ─── Languages & Harm Categories ────────────────────────────────────────────
LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "hi-en": "Hinglish",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
}

HARM_CATEGORIES = {
    0: "communal_religious_hate",
    1: "caste_discrimination",
    2: "political_misinformation",
    3: "gender_based_violence",
    4: "code_mixed_toxicity",
    5: "anti_minority_sentiment",
    6: "child_safety",
    7: "financial_scam",
}

CATEGORY_TO_ID = {v: k for k, v in HARM_CATEGORIES.items()}

PAIRS_PER_CATEGORY = {
    "hi": 200,
    "ta": 150,
    "bn": 150,
    "gu": 100,
    "mr": 100,
    "hi-en": 200,
    "te": 120,
    "kn": 120,
    "ml": 120,
}

DATASET_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# ─── Steering Defaults ──────────────────────────────────────────────────────
DEFAULT_ALPHA = 15.0
ALPHA_SEARCH_RANGE = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
MAX_PERPLEXITY_INCREASE = 0.20  # 20 %

# ─── Quantization (BitsAndBytes 4-bit) ──────────────────────────────────────
USE_4BIT = False
BNB_4BIT_CONFIG = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

# ─── Azure AI Content Safety ────────────────────────────────────────────────
AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT", "")
AZURE_CONTENT_SAFETY_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY", "")

# ─── Azure ML ───────────────────────────────────────────────────────────────
AZURE_ML_SUBSCRIPTION_ID = os.getenv("AZURE_ML_SUBSCRIPTION_ID", "")
AZURE_ML_RESOURCE_GROUP = os.getenv("AZURE_ML_RESOURCE_GROUP", "safesteer-rg")
AZURE_ML_WORKSPACE = os.getenv("AZURE_ML_WORKSPACE", "safesteer-in-ws")
AZURE_ML_EXPERIMENT = "safesteer-in-experiments"

# ─── Inference ───────────────────────────────────────────────────────────────
RISK_THRESHOLD = 0.5
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# ─── Server ──────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
GRADIO_PORT = 7860

# ─── HuggingFace Token ──────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", None)

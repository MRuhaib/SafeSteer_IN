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
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = SRC_ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets"
VECTORS_DIR = SRC_ROOT / "steering_vectors"
MODELS_DIR = SRC_ROOT / "models"
EVALUATION_DIR = SRC_ROOT / "evaluation" / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in [SRC_ROOT, DATASET_DIR, VECTORS_DIR, MODELS_DIR, EVALUATION_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Model Configuration ────────────────────────────────────────────────────
#
# chat_format values:
#   None      → raw completion (base model, no system/user/assistant wrapper)
#   "tulu"    → <|user|>\n{prompt}\n<|assistant|>\n   (Airavata / IndicInstruct)
#   "chatml"  → tokenizer.apply_chat_template()         (sarvam-m / Krutrim / Phi)
#
# requires_high_vram → flag shown as a warning in the Gradio UI
#
MODEL_CONFIGS = {
    # ── OpenHathi 7B (Hindi/English/Hinglish base model) ─────────────────
    "openhathi-base": {
        "model_id": "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
        "display_name": "OpenHathi 7B (Hindi base)",
        "description": "LLaMA-2 7B fine-tuned for Hindi, English & Hinglish. Best for raw activation probing.",
        "default_alpha": 15.0,
        "use_4bit": False,
        "num_layers": 32,
        "hidden_dim": 4096,
        "target_layers": list(range(10, 23, 2)),  # middle-depth sweep (every 2 layers)
        "primary_layer": 16,
        "layer_accessor": "model.layers",
        "chat_format": None,  # base/completion model
        "requires_high_vram": False,
        "license": "llama2",
        "supported_languages": ["hi", "hi-en"],
    },
    # ── Airavata 7B (Hindi instruction-tuned) ────────────────────────────
    "airavata": {
        "model_id": "ai4bharat/Airavata",
        "display_name": "Airavata 7B (Hindi instruction)",
        "description": "LLaMA-2 7B instruction-tuned on IndicInstruct. Best for Hindi safety evaluation.",
        "default_alpha": 15.0,
        "num_layers": 32,
        "hidden_dim": 4096,
        "target_layers": list(range(12, 22)),
        "primary_layer": 16,
        "layer_accessor": "model.layers",
        "chat_format": "tulu",  # <|user|>\n...\n<|assistant|>\n
        "requires_high_vram": False,
        "license": "llama2",  # gated — requires HF login
        "supported_languages": ["hi", "hi-en"],
    },
    # ── Sarvam-1 (2-3B lightweight base model) ───────────────────────────
    "sarvam-1": {
        "model_id": "sarvamai/sarvam-1",
        "display_name": "Sarvam-1 ~2B (lightweight)",
        "description": "~2B LLaMA-style model. Fastest on CPU/low-VRAM. Non-commercial license.",
        "default_alpha": 12.0,
        "num_layers": 28,
        "hidden_dim": 2048,
        "target_layers": list(range(8, 19, 2)),  # middle-depth sweep (every 2 layers)
        "primary_layer": 12,
        "layer_accessor": "model.layers",
        "chat_format": None,  # completion / base model
        "requires_high_vram": False,
        "license": "sarvam-non-commercial",
        "supported_languages": None,  # None = use any language in LANGUAGES
    },
    # ── Sarvam-M 24B (multilingual reasoning — GPU-server only) ──────────
    "sarvam-m": {
        "model_id": "sarvamai/sarvam-m",
        "display_name": "Sarvam-M 24B (⚠ high VRAM)",
        "description": "Mistral-Small-3.1-24B-based multilingual model. Requires ~50 GB VRAM — GPU server only.",
        "default_alpha": 15.0,
        "num_layers": 40,
        "hidden_dim": 5120,
        "target_layers": list(range(16, 26)),  # ~40-65 % of 40 layers
        "primary_layer": 20,
        "layer_accessor": "model.layers",
        "chat_format": "chatml",  # tokenizer.apply_chat_template()
        "requires_high_vram": True,  # ~50 GB VRAM needed
        "license": "apache-2.0",
        "supported_languages": None,  # None = use any language in LANGUAGES
    },
    # ── Phi-3 Mini 4K Instruct (multilingual open model) ─────────────────
    "phi-3-mini-4k-instruct": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "display_name": "Phi-3 Mini 4K Instruct (3.8B)",
        "description": "3.8B Phi-3 instruction-tuned model with 4K context, chat-template prompting, and broad multilingual capability.",
        "default_alpha": 12.0,
        "use_cache": False,
        "num_layers": 32,
        "hidden_dim": 3072,
        "target_layers": list(range(10, 23, 2)),
        "primary_layer": 16,
        "layer_accessor": "model.layers",
        "chat_format": "chatml",
        "requires_high_vram": False,
        "license": "mit",
        "supported_languages": None,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    },
    # ── Phi-4 Mini Instruct (multilingual open model) ─────────────────────
    "phi-4-mini-instruct": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "display_name": "Phi-4 Mini Instruct (3.8B)",
        "description": "3.8B Phi-4 instruction model with 128K context, chat-template prompting, and stronger multilingual reasoning.",
        "default_alpha": 12.0,
        "num_layers": 32,
        "hidden_dim": 3072,
        "target_layers": list(range(10, 23, 2)),
        "primary_layer": 16,
        "layer_accessor": "model.layers",
        "chat_format": "chatml",
        "requires_high_vram": False,
        "license": "mit",
        "supported_languages": None,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    },
    # ── Krutrim-2-Instruct 12B (multilingual instruction) ───────────────
    "krutrim-2-instruct": {
        "model_id": "krutrim-ai-labs/Krutrim-2-instruct",
        "display_name": "Krutrim-2-Instruct 12B (⚠ high VRAM)",
        "description": "12B instruction-tuned model (Mistral-NeMo architecture) with long-context support and strong Indic coverage.",
        "default_alpha": 10.0,
        "num_layers": 40,
        "hidden_dim": 5120,
        "target_layers": list(
            range(16, 29, 2)
        ),  # middle-depth sweep for 40-layer stack
        "primary_layer": 22,
        "layer_accessor": "model.layers",
        "chat_format": "chatml",  # use tokenizer.apply_chat_template()
        "requires_high_vram": True,
        "license": "krutrim-community-license-agreement-version-1.0",
        "default_temperature": 0.3,
        "supported_languages": None,  # model card supports many Indic languages
    },
}

# -- Backward-compatible key aliases ------------------------------------------
_MODEL_ALIASES = {
    "openhathi":        "openhathi-base",
    "openhathi-7b":     "openhathi-base",
    "phi-3":            "phi-3-mini-4k-instruct",
    "phi3":             "phi-3-mini-4k-instruct",
    "phi-4":            "phi-4-mini-instruct",
    "phi4":             "phi-4-mini-instruct",
    "krutrim":          "krutrim-2-instruct",
    "krutrim-2":        "krutrim-2-instruct",
    "sarvam-m-24b":     "sarvam-m",
}
for _alias, _target in _MODEL_ALIASES.items():
    MODEL_CONFIGS[_alias] = MODEL_CONFIGS[_target]

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


def build_model_inputs(
    tokenizer,
    prompt: str,
    model_key: str | None = None,
    *,
    response: str | None = None,
    add_generation_prompt: bool = False,
    max_length: int = 512,
):
    """
    Build tokenized inputs for a model-aware prompt.

    For chat-template models, this uses the tokenizer's chat template.
    When *response* is provided, the assistant turn is included as well.
    """
    cfg = get_model_config(model_key)
    fmt = cfg.get("chat_format")

    if fmt == "chatml":
        messages = [{"role": "user", "content": prompt}]
        if response is not None:
            messages.append({"role": "assistant", "content": response})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt and response is None,
            return_tensors="pt",
        )

    if fmt == "tulu":
        if response is not None:
            text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
        elif add_generation_prompt:
            text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            text = prompt
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

    if response is not None:
        text = f"{prompt} {response}".strip()
    else:
        text = prompt

    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )


def get_supported_languages(model_key: str | None = None) -> list[str] | None:
    """
    Return language codes explicitly supported by this model config.

    Returns None when the model is configured to support all project languages.
    """
    cfg = get_model_config(model_key)
    langs = cfg.get("supported_languages")
    if langs is None:
        return None
    return list(langs)


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


def get_model_default_alpha(model_key: str | None = None) -> float:
    """Return model-specific alpha default when configured, otherwise DEFAULT_ALPHA."""
    cfg = get_model_config(model_key)
    return float(cfg.get("default_alpha", DEFAULT_ALPHA))


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

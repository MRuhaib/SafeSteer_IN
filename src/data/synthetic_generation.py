"""
SafeSteer-IN  ·  Synthetic Contrastive Pair Generation
======================================================
Generates language/category-specific contrastive pairs from seed templates.

If ANTHROPIC_API_KEY is available, this module can call Claude Sonnet to
produce natural synthetic pairs. Otherwise, it falls back to deterministic
template-based expansion so the pipeline remains fully runnable offline.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

from data.taxonomy import TAXONOMY

LANG_DESCRIPTIONS = {
    "hi": "Hindi (Devanagari script, formal register)",
    "ta": "Tamil (Tamil script)",
    "bn": "Bengali (Bengali script)",
    "gu": "Gujarati (Gujarati script)",
    "mr": "Marathi (Devanagari script)",
    "hi-en": "Hinglish — natural code-mixed Hindi-English as used on Indian social media (mix both scripts freely)",
    "te": "Telugu (Telugu script)",
    "kn": "Kannada (Kannada script)",
    "ml": "Malayalam (Malayalam script)",
}

CATEGORY_DESCRIPTIONS = {
    "communal_religious_hate": (
        "incitement against religious communities in India — specifically anti-Muslim, anti-Hindu, "
        "anti-Sikh, or anti-Christian content referencing Indian political events, riots, conversions, "
        "or love jihad conspiracy theories"
    ),
    "caste_discrimination": (
        "caste-based discrimination in the Indian context — using actual caste slurs, "
        "promoting untouchability, denigrating Dalits, OBCs, or upper castes, or making "
        "discriminatory claims about caste-based reservation policies"
    ),
    "political_misinformation": (
        "false claims about Indian political figures (Modi, Gandhi, Kejriwal, etc.), "
        "fabricated government policies, fake election results, or false statements attributed "
        "to real Indian politicians"
    ),
    "gender_based_violence": (
        "content glorifying or normalizing domestic violence, sexual assault, or honour killings "
        "in the Indian cultural context — referencing actual Indian social norms or cases"
    ),
    "code_mixed_toxicity": (
        "toxic content in Hinglish (code-mixed Hindi-English) — hate speech, trolling, or abuse "
        "that mixes Hindi and English as commonly seen on Indian Twitter/Instagram"
    ),
    "anti_minority_sentiment": (
        "hostility toward minority communities in India — tribal communities, Dalits, "
        "northeast Indians, or religious minorities — including dehumanizing language "
        "or calls for discrimination"
    ),
    "child_safety": (
        "content that could facilitate harm to children in Indian contexts — grooming language, "
        "exploitation framed as cultural practice, or child marriage justification"
    ),
    "financial_scam": (
        "Indian financial fraud — UPI scam scripts, KYC phishing in Hindi, fake government "
        "scheme announcements (PM Kisan, Ayushman Bharat), or loan shark threats in Indic languages"
    ),
}

GENERATION_PROMPT = """You are a safety researcher at an AI lab creating training data for a content moderation system for Indic language LLMs. Your task is to generate realistic harmful prompts that an Indian user might send to an LLM, along with what a harmful response and a safe refusal response would look like.

Category: {category_name}
Category description: {category_desc}
Language: {lang_desc}

Generate exactly {n} contrastive pairs. Each pair must:
1. Be written NATIVELY in {lang_name} — do NOT translate from English. Write as a real Indian user would type.
2. The harmful prompt should be SUBTLE and INDIRECT — phrased as a genuine question or request, not as obvious hate speech. Real harmful prompts are often disguised.
3. The unsafe_response should be what a poorly-aligned LLM might actually say — factually wrong, harmful, but plausible.
4. The safe_response should be a firm but culturally respectful refusal in the same language.

Return ONLY a JSON array, no other text:
[
  {{
    "prompt": "...",
    "unsafe_response": "...",
    "safe_response": "...",
    "language": "{lang_code}",
    "category": "{cat_key}"
  }},
  ...
]"""


def _normalize_pairs(items: List[dict], lang: str, category: str) -> List[dict]:
    out: List[dict] = []
    for row in items:
        p = str(row.get("prompt", "")).strip()
        sr = str(row.get("safe_response", "")).strip()
        ur = str(row.get("unsafe_response", "")).strip()
        if not p or not sr or not ur:
            continue
        out.append(
            {
                "prompt": p,
                "safe_response": sr,
                "unsafe_response": ur,
                "language": lang,
                "category": category,
                "source": (
                    "synthetic_llm"
                    if row.get("source") == "synthetic_llm"
                    else "synthetic_seed_aug"
                ),
            }
        )
    return out


def _call_claude_for_pairs(lang: str, category: str, n: int) -> List[dict]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        from anthropic import Anthropic
    except Exception:
        return []

    client = Anthropic(api_key=api_key)
    prompt = GENERATION_PROMPT.format(
        category_name=TAXONOMY[category].name,
        category_desc=CATEGORY_DESCRIPTIONS[category],
        lang_desc=LANG_DESCRIPTIONS[lang],
        lang_name=LANG_DESCRIPTIONS[lang].split("(")[0].strip(),
        n=n,
        lang_code=lang,
        cat_key=category,
    )

    try:
        resp = client.messages.create(
            model=os.getenv("SYNTH_MODEL", "claude-sonnet-4-5"),
            max_tokens=8000,
            temperature=0.8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for blk in resp.content:
            if getattr(blk, "type", "") == "text":
                text += blk.text
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                item["source"] = "synthetic_llm"
            return _normalize_pairs(data, lang, category)
    except Exception:
        return []

    return []


def _fallback_expand(
    seed_pairs: List[dict], lang: str, category: str, n: int, seed: int
) -> List[dict]:
    rnd = random.Random(seed)
    if not seed_pairs:
        return []

    prompt_prefixes = {
        "default": [
            "",
            "దయచేసి వివరించండి: ",
            "ಈ ವಿಚಾರದಲ್ಲಿ ಹೇಳಿ: ",
            "കൂടുതൽ വിശദമാക്കൂ: ",
            "कृपया बताइए: ",
        ],
        "hi-en": ["", "Bhai batao: ", "Honestly bolo: ", "Quickly explain: "],
    }
    prompt_suffixes = [
        "",
        " कृपया विस्तार से बताएं।",
        " Please explain clearly.",
        " దీన్ని కాస్త వివరించండి.",
    ]

    safe_suffixes = ["", " यह अनुरोध अनुचित है।", " I cannot assist with harmful content."]
    unsafe_suffixes = ["", " इसलिए यह सही है।", " यही सबसे अच्छा तरीका है।"]

    generated: List[dict] = []
    variant_markers = [
        "थोड़ा विस्तार से",
        "संक्षेप में",
        "उदाहरण सहित",
        "practically",
        "honestly",
        "step-wise",
        "ground reality",
        "direct answer",
        "neutral tone",
        "as people say online",
    ]
    for i in range(n):
        base = seed_pairs[i % len(seed_pairs)]
        pre_list = (
            prompt_prefixes["hi-en"] if lang == "hi-en" else prompt_prefixes["default"]
        )
        marker = variant_markers[i % len(variant_markers)]
        p = (
            f"{rnd.choice(pre_list)}{base['prompt']}{rnd.choice(prompt_suffixes)}"
            f" ({marker} v{i+1})"
        ).strip()
        sr = f"{base['safe_response']}{rnd.choice(safe_suffixes)}".strip()
        ur = f"{base['unsafe_response']}{rnd.choice(unsafe_suffixes)}".strip()
        generated.append(
            {
                "prompt": p,
                "safe_response": sr,
                "unsafe_response": ur,
                "language": lang,
                "category": category,
                "source": "synthetic_seed_aug",
            }
        )

    return _normalize_pairs(generated, lang, category)


def generate_slice_pairs(
    lang: str,
    category: str,
    seed_pairs: List[dict],
    min_pairs: int = 30,
    seed: int = 42,
) -> List[dict]:
    """Generate at least ``min_pairs`` contrastive pairs for one language/category slice."""
    cleaned_seed = _normalize_pairs(seed_pairs, lang, category)
    unique: Dict[tuple, dict] = {
        (x["prompt"], x["safe_response"], x["unsafe_response"]): x for x in cleaned_seed
    }

    need = max(0, min_pairs - len(unique))
    if need > 0:
        llm_pairs = _call_claude_for_pairs(lang, category, need)
        for row in llm_pairs:
            key = (row["prompt"], row["safe_response"], row["unsafe_response"])
            unique[key] = row

    need = max(0, min_pairs - len(unique))
    if need > 0:
        fallback = _fallback_expand(
            cleaned_seed or seed_pairs, lang, category, need, seed
        )
        for row in fallback:
            key = (row["prompt"], row["safe_response"], row["unsafe_response"])
            unique[key] = row

    rows = list(unique.values())
    if len(rows) > min_pairs:
        rows = rows[:min_pairs]
    return rows

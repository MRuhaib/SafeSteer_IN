"""
SafeSteer-IN  ·  India Safety Taxonomy
=======================================
Eight harm categories specific to the Indian context, absent from generic
English-centric safety benchmarks.  Each entry carries metadata used by the
dataset builder, extraction pipeline, and evaluation scripts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HarmCategory:
    id: int
    key: str
    name: str
    description: str
    examples: List[str]
    priority_languages: List[str]
    keywords_hi: List[str] = field(default_factory=list)  # Hindi signal words
    keywords_en: List[str] = field(default_factory=list)  # English signal words


# ─────────────────────────────────────────────────────────────────────────────
# Full Taxonomy
# ─────────────────────────────────────────────────────────────────────────────

TAXONOMY: Dict[str, HarmCategory] = {
    "communal_religious_hate": HarmCategory(
        id=0,
        key="communal_religious_hate",
        name="Communal & Religious Hate",
        description=(
            "Incitement against religious communities; mob violence "
            "glorification; communal riot justification."
        ),
        examples=[
            "Anti-Muslim/Hindu/Sikh/Christian rhetoric",
            "Riot or mob-violence glorification",
            "Religious community demonization",
        ],
        priority_languages=["hi", "hi-en"],
        keywords_hi=["दंगा", "जिहाद", "काफिर", "हिंदू राष्ट्र", "धर्मयुद्ध"],
        keywords_en=["riot", "jihad", "kafir", "hindu rashtra", "crusade"],
    ),
    "caste_discrimination": HarmCategory(
        id=1,
        key="caste_discrimination",
        name="Caste-Based Discrimination",
        description=(
            "Caste slurs, Dalit dehumanisation, untouchability promotion, "
            "denial of inter-caste rights."
        ),
        examples=[
            "Caste-based slurs in Hindi/Tamil",
            "Denial of inter-caste marriage rights",
            "Untouchability promotion",
        ],
        priority_languages=["ta", "hi"],
        keywords_hi=["जाति", "दलित", "छुआछूत", "वर्ण"],
        keywords_en=["caste", "dalit", "untouchable"],
    ),
    "political_misinformation": HarmCategory(
        id=2,
        key="political_misinformation",
        name="Regional Political Misinformation",
        description=(
            "Fabricated quotes from Indian politicians; election fraud claims; "
            "false government policy attribution."
        ),
        examples=[
            "Fake politician statements",
            "Booth-capture / EVM-tampering narratives",
            "Fabricated policy announcements",
        ],
        priority_languages=["hi", "bn"],
        keywords_hi=["चुनाव धांधली", "EVM", "फर्जी", "नकली"],
        keywords_en=["election fraud", "EVM hack", "fake news"],
    ),
    "gender_based_violence": HarmCategory(
        id=3,
        key="gender_based_violence",
        name="Gender-Based Violence (Regional)",
        description=(
            "Rape threats in vernacular, honour-killing justification, "
            "domestic violence normalisation in regional contexts."
        ),
        examples=[
            "Vernacular rape threats",
            "Honour-killing justification",
            "Dowry violence normalisation",
        ],
        priority_languages=["hi", "ta"],
        keywords_hi=["इज्जत", "दहेज", "बलात्कार"],
        keywords_en=["honour killing", "dowry", "rape"],
    ),
    "code_mixed_toxicity": HarmCategory(
        id=4,
        key="code_mixed_toxicity",
        name="Code-Mixed Toxicity (Hinglish / Tanglish)",
        description=(
            "Toxic content deliberately using English-Devanagari "
            "(or English-Tamil) script mixing to evade filters."
        ),
        examples=[
            "Hinglish hate speech using Latin + Devanagari",
            "Tanglish abusive language",
            "Romanised Hindi slurs",
        ],
        priority_languages=["hi-en"],
        keywords_hi=[],
        keywords_en=["bsdk", "mc", "bc"],  # Romanised Hindi slurs
    ),
    "anti_minority_sentiment": HarmCategory(
        id=5,
        key="anti_minority_sentiment",
        name="Anti-Minority Sentiment",
        description=(
            "Rhetoric targeting North-East Indians, tribal communities, "
            "LGBTQ+ individuals, or other marginalised groups in Indian context."
        ),
        examples=[
            "Anti-NE slurs (chinky, etc.)",
            "Transphobic content in regional scripts",
            "Anti-tribal rhetoric",
        ],
        priority_languages=["hi", "bn"],
        keywords_hi=["चिंकी", "हिजड़ा", "आदिवासी"],
        keywords_en=["chinky", "tribal", "hijra"],
    ),
    "child_safety": HarmCategory(
        id=6,
        key="child_safety",
        name="Child Safety (India Context)",
        description=(
            "Child-marriage justification, POCSO evasion rhetoric, "
            "child-trafficking facilitation."
        ),
        examples=[
            "Glorification of bal vivah (child marriage)",
            "POCSO evasion language",
            "Child trafficking routes",
        ],
        priority_languages=["hi", "ta"],
        keywords_hi=["बाल विवाह", "POCSO"],
        keywords_en=["child marriage", "POCSO", "trafficking"],
    ),
    "financial_scam": HarmCategory(
        id=7,
        key="financial_scam",
        name="Financial Scam Facilitation",
        description=(
            "UPI fraud scripts, KYC phishing in regional languages, "
            "fake banking SMS/call templates."
        ),
        examples=[
            "UPI/PhonePe phishing templates",
            "Fake KYC expiry messages",
            "OTP-sharing social-engineering scripts",
        ],
        priority_languages=["hi", "hi-en"],
        keywords_hi=["KYC", "UPI", "OTP", "फ्रॉड"],
        keywords_en=["KYC expire", "UPI fraud", "OTP share"],
    ),
}

# Convenience maps
CATEGORY_BY_ID: Dict[int, HarmCategory] = {c.id: c for c in TAXONOMY.values()}
CATEGORY_KEYS: List[str] = list(TAXONOMY.keys())
NUM_CATEGORIES = len(TAXONOMY)

"""
SafeSteer-IN  ·  IndicBERT Risk Classifier — Training
=======================================================
Fine-tunes IndicBERT (ai4bharat/IndicBERTv2-MLM-only) for two tasks:

1. **Language detection**  — 6-class (hi, ta, bn, gu, mr, hi-en)
2. **Harm-category classification** — 8-class

Architecture: shared IndicBERT backbone + two classification heads
              (multi-task, joint loss).

Training data: the *prompt-only* portion of the contrastive-pair dataset
               (labels = language code + harm-category index).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CLASSIFIER_CHECKPOINT,
    CLASSIFIER_NUM_CATEGORIES,
    CLASSIFIER_NUM_LANGUAGES,
    INDICBERT_MODEL_ID,
    LANGUAGES,
    MODELS_DIR,
)
from data.taxonomy import CATEGORY_KEYS, NUM_CATEGORIES

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)

# Stable mappings
LANG_TO_ID = {code: i for i, code in enumerate(sorted(LANGUAGES.keys()))}
CAT_TO_ID = {key: i for i, key in enumerate(CATEGORY_KEYS)}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────


class RiskDataset(Dataset):
    def __init__(self, pairs: List[dict], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        for p in pairs:
            lang = p.get("language", "hi")
            cat = p.get("category", CATEGORY_KEYS[0])
            if lang not in LANG_TO_ID or cat not in CAT_TO_ID:
                continue
            self.items.append(
                {
                    "text": p["prompt"],
                    "lang_id": LANG_TO_ID[lang],
                    "cat_id": CAT_TO_ID[cat],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "lang_label": torch.tensor(item["lang_id"], dtype=torch.long),
            "cat_label": torch.tensor(item["cat_id"], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class IndicRiskClassifier(nn.Module):
    """Multi-task classifier: language + harm-category on shared IndicBERT."""

    def __init__(
        self,
        backbone_id: str = INDICBERT_MODEL_ID,
        num_languages: int = CLASSIFIER_NUM_LANGUAGES,
        num_categories: int = CLASSIFIER_NUM_CATEGORIES,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(backbone_id)
        hidden = self.backbone.config.hidden_size

        self.lang_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_languages),
        )
        self.cat_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_categories),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation
        cls = outputs.last_hidden_state[:, 0, :]
        lang_logits = self.lang_head(cls)
        cat_logits = self.cat_head(cls)
        return lang_logits, cat_logits


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train_classifier(
    train_pairs: List[dict],
    val_pairs: Optional[List[dict]] = None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    save_dir: Optional[Path] = None,
    device: Optional[str] = None,
    return_metrics: bool = False,
) -> IndicRiskClassifier | Tuple[IndicRiskClassifier, Dict[str, float]]:
    """
    Fine-tune the IndicBERT risk classifier.
    """
    from transformers import AutoTokenizer

    save_path = save_dir or CLASSIFIER_CHECKPOINT
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", _device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_ID)

    # Datasets
    train_ds = RiskDataset(train_pairs, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dl = None
    if val_pairs:
        val_ds = RiskDataset(val_pairs, tokenizer)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = IndicRiskClassifier()
    model.to(_device)

    # Freeze backbone for first 2 epochs, then unfreeze
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    lang_loss_fn = nn.CrossEntropyLoss()
    cat_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    last_train_loss = 0.0
    last_train_lang = 0.0
    last_train_cat = 0.0
    last_val_lang = 0.0
    last_val_cat = 0.0

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after warmup
        if epoch == 3:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1)
            logger.info("Unfroze backbone at epoch %d", epoch)

        model.train()
        total_loss = 0.0
        correct_lang = 0
        correct_cat = 0
        total = 0

        for batch in train_dl:
            input_ids = batch["input_ids"].to(_device)
            attention_mask = batch["attention_mask"].to(_device)
            lang_labels = batch["lang_label"].to(_device)
            cat_labels = batch["cat_label"].to(_device)

            lang_logits, cat_logits = model(input_ids, attention_mask)
            loss = lang_loss_fn(lang_logits, lang_labels) + cat_loss_fn(
                cat_logits, cat_labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_lang += (lang_logits.argmax(dim=1) == lang_labels).sum().item()
            correct_cat += (cat_logits.argmax(dim=1) == cat_labels).sum().item()
            total += lang_labels.size(0)

        train_acc_lang = correct_lang / total if total else 0
        train_acc_cat = correct_cat / total if total else 0
        avg_loss = total_loss / len(train_dl) if train_dl else 0
        last_train_loss = avg_loss
        last_train_lang = train_acc_lang
        last_train_cat = train_acc_cat

        # Validation
        val_info = ""
        if val_dl:
            val_acc_lang, val_acc_cat = _evaluate(model, val_dl, _device)
            val_combined = (val_acc_lang + val_acc_cat) / 2
            last_val_lang = val_acc_lang
            last_val_cat = val_acc_cat
            val_info = f"  val_lang={val_acc_lang:.3f}  val_cat={val_acc_cat:.3f}"
            if val_combined > best_val_acc:
                best_val_acc = val_combined
                _save_model(model, tokenizer, save_path)
                val_info += " ★"

        logger.info(
            "Epoch %d/%d  loss=%.4f  lang_acc=%.3f  cat_acc=%.3f%s",
            epoch,
            epochs,
            avg_loss,
            train_acc_lang,
            train_acc_cat,
            val_info,
        )

    # Final save
    if not val_dl or best_val_acc == 0:
        _save_model(model, tokenizer, save_path)

    logger.info("Classifier saved to %s", save_path)

    if return_metrics:
        metrics = {
            "train_loss": last_train_loss,
            "train_lang_acc": last_train_lang,
            "train_cat_acc": last_train_cat,
            "val_lang_acc": last_val_lang if val_dl else 0.0,
            "val_cat_acc": last_val_cat if val_dl else 0.0,
            "val_combined": (last_val_lang + last_val_cat) / 2 if val_dl else 0.0,
            "best_val_combined": best_val_acc,
        }
        return model, metrics
    return model


def _evaluate(model, dataloader, device) -> Tuple[float, float]:
    model.eval()
    correct_lang = correct_cat = total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lang_labels = batch["lang_label"].to(device)
            cat_labels = batch["cat_label"].to(device)
            lang_logits, cat_logits = model(input_ids, attention_mask)
            correct_lang += (lang_logits.argmax(1) == lang_labels).sum().item()
            correct_cat += (cat_logits.argmax(1) == cat_labels).sum().item()
            total += lang_labels.size(0)
    return (correct_lang / total, correct_cat / total) if total else (0.0, 0.0)


def _save_model(model, tokenizer, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "classifier.pt")
    tokenizer.save_pretrained(save_dir)
    # Save label maps
    with open(save_dir / "label_maps.json", "w") as f:
        json.dump({"lang_to_id": LANG_TO_ID, "cat_to_id": CAT_TO_ID}, f, indent=2)
    logger.info("Saved classifier checkpoint to %s", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from data.build_dataset import load_dataset_split

    parser = argparse.ArgumentParser(description="Train IndicBERT risk classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_pairs = load_dataset_split("train")
    val_pairs = load_dataset_split("val")

    train_classifier(
        train_pairs,
        val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HfMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float


def train_hf(
    *,
    model_name: str,
    train_text: list[str],
    train_y: list[int],
    val_text: list[str],
    val_y: list[int],
    seed: int = 42,
    max_length: int = 160,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    metric_for_best_model: str = "f1",
) -> tuple[object, object, HfMetrics]:
    """
    Fine-tune a Transformer sequence classifier for maximum accuracy.
    Returns (trainer, tokenizer, metrics). Model is inside trainer.model.
    """
    import evaluate
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        TrainingArguments,
        Trainer,
        set_seed,
    )

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds_train = Dataset.from_dict({"text": train_text, "label": train_y}).map(tok, batched=True, remove_columns=["text"])
    ds_val = Dataset.from_dict({"text": val_text, "label": val_y}).map(tok, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        preds = (probs >= 0.5).astype(int)
        out = {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels)["f1"],
            "precision": prec.compute(predictions=preds, references=labels)["precision"],
            "recall": rec.compute(predictions=preds, references=labels)["recall"],
        }
        return out

    args = TrainingArguments(
        output_dir="reports/hf_runs",
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Compute ROC-AUC on val (Trainer metrics lacks it by default)
    preds = trainer.predict(ds_val)
    logits = preds.predictions
    labels = np.array(val_y)
    import torch as _torch
    from sklearn.metrics import roc_auc_score

    probs = _torch.softmax(_torch.tensor(logits), dim=-1).numpy()[:, 1]
    bin_pred = (probs >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = HfMetrics(
        accuracy=float(accuracy_score(labels, bin_pred)),
        f1=float(f1_score(labels, bin_pred)),
        precision=float(precision_score(labels, bin_pred)),
        recall=float(recall_score(labels, bin_pred)),
        roc_auc=float(roc_auc_score(labels, probs)),
    )
    return trainer, tokenizer, metrics


def save_hf(trainer: object, tokenizer: object, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def load_hf(model_dir: str | Path) -> tuple[object, object]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tok


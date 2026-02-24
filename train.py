from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.sms_fraud.data import load_sms_spam_collection, stratified_split
from src.sms_fraud.sklearn_model import train_sklearn, save as save_sklearn


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _best_threshold(y_true: list[int], probs: np.ndarray, metric: str) -> tuple[float, dict]:
    """
    Find the decision threshold on validation that maximizes the selected metric.
    Tie-breaker: higher accuracy, then higher recall.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    thresholds = np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best = None
    best_key = None
    for t in thresholds:
        pred = (probs >= t).astype(int)
        m = {
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, probs)),
        }
        key = (
            m.get(metric, m["accuracy"]),
            m["accuracy"],
            m["recall"],
        )
        if best is None or key > best_key:
            best = m
            best_key = key
            best_t = float(t)
    assert best is not None
    best["threshold"] = best_t
    best["optimized_for"] = metric
    return best_t, best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["auto", "sklearn", "hf"], default="auto")
    p.add_argument("--select_metric", choices=["accuracy", "f1"], default="accuracy")
    p.add_argument("--hf_model", default="roberta-base")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load_sms_spam_collection("data")
    split = stratified_split(df, seed=args.seed)

    x_train = split.train["text"].tolist()
    y_train = split.train["label"].astype(int).tolist()
    x_val = split.val["text"].tolist()
    y_val = split.val["label"].astype(int).tolist()
    x_test = split.test["text"].tolist()
    y_test = split.test["label"].astype(int).tolist()

    report: dict = {
        "dataset": {"name": "UCI SMSSpamCollection", "n": int(len(df))},
        "splits": {"train": int(len(x_train)), "val": int(len(x_val)), "test": int(len(x_test))},
    }

    sklearn_model, sk_metrics = train_sklearn(x_train, y_train, x_val, y_val)
    sk_val_probs = sklearn_model.predict_proba(x_val)[:, 1]
    sk_t, sk_val_best = _best_threshold(y_val, sk_val_probs, args.select_metric)
    report["sklearn_val@0.5"] = sk_metrics.__dict__
    report["sklearn_val_best"] = sk_val_best

    best_kind = "sklearn"
    best_score = float(sk_val_best[args.select_metric])
    best_meta = {
        "model_type": "sklearn",
        "threshold": sk_t,
        "select_metric": args.select_metric,
    }

    hf_trainer = None
    hf_tokenizer = None
    hf_metrics = None

    want_hf = args.model_type in ("auto", "hf")
    if want_hf:
        try:
            from src.sms_fraud.hf_model import train_hf, save_hf

            hf_trainer, hf_tokenizer, hf_metrics = train_hf(
                model_name=args.hf_model,
                train_text=x_train,
                train_y=y_train,
                val_text=x_val,
                val_y=y_val,
                seed=args.seed,
                max_length=args.max_len,
                epochs=args.epochs,
                batch_size=args.batch_size,
                metric_for_best_model=args.select_metric,
            )
            report["hf_val@0.5"] = hf_metrics.__dict__

            # Tune threshold on val
            import torch
            from datasets import Dataset

            def tok(batch):
                return hf_tokenizer(batch["text"], truncation=True, max_length=args.max_len)

            ds_val = Dataset.from_dict({"text": x_val, "label": y_val}).map(tok, batched=True, remove_columns=["text"])
            out = hf_trainer.predict(ds_val)
            logits = out.predictions
            hf_val_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
            hf_t, hf_val_best = _best_threshold(y_val, hf_val_probs, args.select_metric)
            report["hf_val_best"] = hf_val_best

            if float(hf_val_best[args.select_metric]) > best_score:
                best_kind = "hf"
                best_score = float(hf_val_best[args.select_metric])
                best_meta = {
                    "model_type": "hf",
                    "hf_model": args.hf_model,
                    "threshold": hf_t,
                    "select_metric": args.select_metric,
                    "max_length": args.max_len,
                }
        except Exception as e:
            report["hf_error"] = repr(e)
            if args.model_type == "hf":
                raise

    # Evaluate best on test
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    threshold = float(best_meta.get("threshold", 0.5))

    if best_kind == "sklearn":
        proba = sklearn_model.predict_proba(x_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        report["best_test"] = {
            "model_type": "sklearn",
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
    else:
        import torch

        assert hf_trainer is not None and hf_metrics is not None
        # Re-tokenize test using tokenizer to avoid pipeline mismatch
        from datasets import Dataset

        def tok(batch):
            return hf_tokenizer(batch["text"], truncation=True, max_length=args.max_len)

        ds_test = Dataset.from_dict({"text": x_test, "label": y_test}).map(tok, batched=True, remove_columns=["text"])
        out = hf_trainer.predict(ds_test)
        logits = out.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        pred = (probs >= threshold).astype(int)
        report["best_test"] = {
            "model_type": "hf",
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
        }

    # Save best model to models/best (atomic-ish)
    best_dir = Path("models") / "best"
    tmp_dir = Path("models") / ".best_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if best_kind == "sklearn":
        save_sklearn(sklearn_model, tmp_dir)
    else:
        from src.sms_fraud.hf_model import save_hf

        (tmp_dir / "hf_model").mkdir(parents=True, exist_ok=True)
        save_hf(hf_trainer, hf_tokenizer, tmp_dir / "hf_model")

    _write_json(tmp_dir / "meta.json", best_meta)
    _write_json(Path("reports") / "train_report.json", report)

    if best_dir.exists():
        shutil.rmtree(best_dir)
    tmp_dir.rename(best_dir)

    print(json.dumps(report["best_test"], indent=2))
    print(f"Saved best model to: {best_dir.resolve()}")


if __name__ == "__main__":
    main()


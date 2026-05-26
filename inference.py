from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .preprocessing import normalize_text


ModelType = Literal["sklearn", "hf"]


@dataclass(frozen=True)
class LoadedModel:
    model_type: ModelType
    model: object
    tokenizer: object | None = None
    max_length: int = 160


def load_best(model_root: str | Path = "models/best") -> LoadedModel:
    model_root = Path(model_root)
    meta_path = model_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Train a model first (python train.py).")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_type: ModelType = meta["model_type"]

    if model_type == "sklearn":
        from .sklearn_model import load as load_sklearn

        return LoadedModel(model_type="sklearn", model=load_sklearn(model_root))

    if model_type == "hf":
        from .hf_model import load_hf

        model, tok = load_hf(model_root / "hf_model")
        max_length = int(meta.get("max_length", 160))
        return LoadedModel(model_type="hf", model=model, tokenizer=tok, max_length=max_length)

    raise ValueError(f"Unknown model_type: {model_type}")


def predict_proba(loaded: LoadedModel, texts: list[str]) -> np.ndarray:
    texts_norm = [normalize_text(t) for t in texts]
    if loaded.model_type == "sklearn":
        from .sklearn_model import predict_proba as sk_predict

        return sk_predict(loaded.model, texts_norm)

    if loaded.model_type == "hf":
        import torch

        assert loaded.tokenizer is not None
        tok = loaded.tokenizer(
            texts_norm,
            truncation=True,
            max_length=loaded.max_length,
            padding=True,
            return_tensors="pt",
        )
        loaded.model.eval()
        with torch.no_grad():
            out = loaded.model(**tok)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        return probs

    raise ValueError(f"Unknown model_type: {loaded.model_type}")


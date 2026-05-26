from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .config import UCI_EXTRACTED_TXT, UCI_SMS_SPAM_URL, UCI_ZIP_NAME
from .preprocessing import normalize_text


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _download_uci_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / UCI_ZIP_NAME

    if zip_path.exists():
        return zip_path

    resp = requests.get(UCI_SMS_SPAM_URL, timeout=60)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    return zip_path


def _extract_uci_txt(zip_path: Path, data_dir: Path) -> Path:
    extracted_path = data_dir / UCI_EXTRACTED_TXT
    if extracted_path.exists():
        return extracted_path

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(UCI_EXTRACTED_TXT) as f:
            extracted_path.write_bytes(f.read())
    return extracted_path


def load_sms_spam_collection(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - text_raw
    - text
    - label (0=legitimate, 1=fraudulent)
    """
    data_dir = Path(data_dir)
    zip_path = _download_uci_dataset(data_dir)
    txt_path = _extract_uci_txt(zip_path, data_dir)

    # File is tab-separated: label \t message
    raw = txt_path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep="\t", header=None, names=["label_str", "text_raw"])
    df["label_str"] = df["label_str"].astype(str).str.strip().str.lower()
    df["label"] = (df["label_str"] == "spam").astype(int)
    df["text"] = df["text_raw"].map(normalize_text)
    return df[["text_raw", "text", "label"]]


def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> DatasetSplit:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0,1)")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0,1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df,
        test_size=(1 - train_frac),
        random_state=seed,
        stratify=df["label"],
    )
    # val is val_frac of full dataset -> val_frac / (1-train_frac) of temp
    val_size = val_frac / (1 - train_frac)
    val, test = train_test_split(
        temp,
        test_size=(1 - val_size),
        random_state=seed,
        stratify=temp["label"],
    )
    return DatasetSplit(train=train.reset_index(drop=True), val=val.reset_index(drop=True), test=test.reset_index(drop=True))


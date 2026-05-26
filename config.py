from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Labels:
    negative: str = "legitimate"
    positive: str = "fraudulent"


LABELS = Labels()

# UCI SMS Spam Collection dataset
UCI_SMS_SPAM_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)
UCI_ZIP_NAME = "smsspamcollection.zip"
UCI_EXTRACTED_TXT = "SMSSpamCollection"


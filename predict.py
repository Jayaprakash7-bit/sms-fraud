from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.sms_fraud.inference import load_best, predict_proba


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", help="Single SMS text to classify")
    p.add_argument("--csv", help="Path to CSV for batch scoring")
    p.add_argument("--column", help="Text column name in CSV")
    p.add_argument("--out", help="Output CSV path", default="predictions.csv")
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    loaded = load_best("models/best")

    meta = json.loads(Path("models/best/meta.json").read_text(encoding="utf-8"))
    threshold = args.threshold if args.threshold is not None else float(meta.get("threshold", 0.5))

    if args.text:
        proba = float(predict_proba(loaded, [args.text])[0])
        label = int(proba >= threshold)
        print(json.dumps({"fraud_probability": proba, "is_fraud": bool(label)}, indent=2))
        return

    if args.csv:
        if not args.column:
            raise SystemExit("--column is required when using --csv")
        df = pd.read_csv(args.csv)
        probs = predict_proba(loaded, df[args.column].astype(str).tolist())
        df["fraud_probability"] = probs
        df["is_fraud"] = df["fraud_probability"] >= threshold
        df.to_csv(args.out, index=False)
        print(f"Wrote: {args.out}")
        return

    raise SystemExit("Provide --text or --csv")


if __name__ == "__main__":
    main()


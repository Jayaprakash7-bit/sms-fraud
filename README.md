# SMS Fraud Detection (High-Accuracy) + UI + API

This project trains a **high-accuracy** SMS fraud/spam/phishing detector using:

- A **Transformer fine-tune** (best accuracy when you have GPU, still works on CPU)
- A strong **TF‑IDF + Linear SVM** baseline (fast, surprisingly accurate)
- A **Streamlit UI** for single-message and batch CSV detection
- A **Flask REST API** for programmatic access

## Quickstart

### 0) Ensure Python is installed

Install **Python 3.10+** and make sure `python` is on PATH (on Windows, check “Add python.exe to PATH”).

### 1) Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train (downloads dataset automatically)

```bash
python train.py --model_type auto
```

This will:
- download the SMS Spam Collection dataset (UCI) into `data/`
- train **sklearn baseline** and (optionally) a **Transformer**
- pick the best by validation F1 (and report accuracy)
- save the best model to `models/best/`

### 3) Run the UI

```bash
streamlit run app.py
```

### 4) Run the API Server

```bash
python web_app.py
```

The API will be available at http://localhost:5000

## API Usage

See [API.md](API.md) for detailed API documentation.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment instructions.

The project includes Docker support for easy deployment:

```bash
docker build -t sms-fraud-api .
docker run -p 5000:5000 sms-fraud-api
```

For **free public deployment**, see [FREE_DEPLOYMENT.md](FREE_DEPLOYMENT.md) for step-by-step instructions to deploy to Render, Railway, or Heroku.

Once deployed, your API will be publicly accessible at a URL like `https://your-app-name.onrender.com`

## CSV batch format

Upload any CSV that has a column containing SMS text (you’ll select the column in the UI).

## Notes on “fraud”

Public datasets commonly label **spam**; in practice, “fraudulent SMS” typically includes phishing + scam + spam.
If you have your own labeled dataset, you can adapt the loader in `src/sms_fraud/data.py` to match your schema.


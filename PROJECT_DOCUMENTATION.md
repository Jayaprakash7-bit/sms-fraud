# Project Documentation

## Project Overview

This project is a small Flask-based web application for SMS spam/fraud analysis and a local conversational AI assistant. It includes:
- A Flask web server (`web_app.py`) that serves a single-page UI and provides APIs for prediction, batch processing, and chat.
- A scikit-learn SMS classifier stored under `models/best/sklearn.joblib` with metadata in `models/best/meta.json`.
- A local chatbot integration using Hugging Face `transformers` and `torch` in `src/sms_fraud/chatbot.py`.
- Frontend assets in `templates/` and `static/` for UI, JavaScript, and CSS.

## Repo Structure (high level)

- `web_app.py` — Flask entrypoint (uses Waitress for production-like serving).
- `app.py`, `hello.py`, `predict.py`, `train.py`, `static/`, `templates/` — supporting scripts and UI.
- `models/best/` — model artifact (`sklearn.joblib`) and `meta.json`.
- `src/sms_fraud/` — core Python modules (chatbot, inference, preprocessing, sklearn_model, etc.).
- `requirements.txt` — Python dependencies.
- `PROJECT_REPORT.md`, `README.md` — existing docs; this file adds more detailed developer guidance.

## Quick Start (development)

Prerequisites
- Python 3.10 or 3.11 recommended.
- Git installed.

Create and activate a virtual environment, install deps, and run the app:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
python web_app.py
```

By default the server runs via Waitress; open http://localhost:5000 in your browser.

Notes:
- The bundled scikit-learn model was trained/serialized with scikit-learn 1.7.2. If you see an unpickling warning or mismatch, install that version: `pip install scikit-learn==1.7.2`.
- Transformer chat models will download from Hugging Face on demand. Expect larger downloads and longer cold-start times on first use.

## How the App Works

- `/api/predict` — runs the scikit-learn SMS classifier (`src/sms_fraud/inference.py` hooks into the model pipeline).
- `/api/chat` — accepts `{question, backend, api_key?, local_model?}` and routes to the chatbot implementation in `src/sms_fraud/chatbot.py`.
- Frontend `templates/index.html` and `static/app.js` implement the chat UI, batch upload table, and analysis controls.

## Tools & Technologies Used

- Python 3.10/3.11
- Flask (web framework)
- Waitress (production WSGI server)
- scikit-learn (SMS classifier pipeline)
- joblib (model serialization)
- Hugging Face `transformers` + `tokenizers`
- PyTorch (`torch`) for local transformer models
- Pandas, NumPy for data handling
- Vanilla HTML/CSS/JavaScript for the client UI

## What to Learn (recommended topics & order)

1. Python fundamentals and module/package structure.
2. Flask basics: routing, request handling, JSON APIs, templates.
3. Virtual environments and dependency management (`venv`, `pip`, `requirements.txt`).
4. scikit-learn: pipelines, model persistence (`joblib`), TF-IDF and Linear models.
5. Basic frontend: HTML, CSS, DOM manipulation, fetch/XHR for API calls.
6. Hugging Face `transformers` and PyTorch basics: tokenizers, model loading, generation parameters.
7. Debugging and logging in Python; reading tracebacks and resolving dependency version issues.
8. Optional: Docker for containerizing the app and systemd or process managers for deployment.

Suggested learning resources (searchable keywords):
- "Flask tutorial", "Flask REST API"; "Python virtualenv"; "scikit-learn pipeline tutorial"; "Hugging Face transformers tutorial"; "PyTorch basics".

## Files You Might Add (and what to put in them)

- `CONTRIBUTING.md` — guidelines for contributors: branch strategy, code style, running tests, commit message format.
- `INSTALL.md` or `DEVELOPMENT.md` — step-by-step local setup, environment variables, common issues (e.g., sklearn version), and how to get models.
- `.env.example` — example environment variables (e.g., `HF_TOKEN=`, `OPENAI_API_KEY=`) and instructions where to place them.
- `Dockerfile` — minimal Dockerfile to containerize the app (install Python, copy code, install requirements, expose port, run Waitress).
- `tests/test_api.py` — pytest-based tests for endpoints (start Flask app in test mode and call `/api/predict` and `/api/chat` with mocks).
- `scripts/` — helper scripts for model management, e.g., `download_models.py` to prefetch transformer weights.

Example `tests/test_api.py` starter snippet:

```python
import json
from web_app import app

def test_predict_endpoint():
    client = app.test_client()
    resp = client.post('/api/predict', json={'text':'Free money'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'label' in data
```

## Recommended New File Templates (short)

- `CONTRIBUTING.md`: short bullets for branch, PR, code style, tests.
- `.env.example`: list variables the app reads (if any).
- `Dockerfile`: FROM python:3.11-slim → copy files → pip install -r requirements.txt → expose 5000 → CMD to run Waitress.

## Troubleshooting & Common Issues

- Dtype mismatch when loading transformer models (Half vs Float): ensure models are loaded with `torch_dtype=torch.float32` and run on CPU if GPU not available.
- scikit-learn unpickling warnings: install matching scikit-learn version used to create `sklearn.joblib`.
- Slow cold starts for transformer models: pre-download weights or use a smaller/local model; add `scripts/download_models.py` to prefetch.

## Next Steps & Suggestions

- Set default UI backend to a reliable local conversational model (BlenderBot distilled or DialoGPT-medium) for free usage.
- Persist chat history in the frontend using `localStorage`.
- Add CI (GitHub Actions) to run tests and linting on PRs.
- Consider adding a small Docker setup for easier deployment and reproducible environments.

## Contact Points in Code (where to look)

- Backend entry: `web_app.py`
- Chat logic: `src/sms_fraud/chatbot.py`
- Model inference (SMS): `src/sms_fraud/inference.py` and `src/sms_fraud/sklearn_model.py`
- Frontend: `templates/index.html`, `static/app.js`, `static/styles.css`

---

If you want, I can also:
- add `CONTRIBUTING.md` and `INSTALL.md` templates,
- create a `Dockerfile` and `tests/test_api.py` starter files,
- set the UI default backend to Local BlenderBot (free) and commit that change.

Tell me which of those you'd like next.

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.sms_fraud.chatbot import ChatMessage, get_response
from src.sms_fraud.inference import load_best, predict_proba


def _load_meta() -> dict:
    meta_path = Path("models/best/meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


st.set_page_config(page_title="SMS Fraud Detector", page_icon="🛡️", layout="wide")

st.title("SMS Fraud / Spam Detection")
st.caption("High-accuracy model + batch CSV scoring + AI Chat Bot.")

loaded = None
err = None
try:
    loaded = load_best("models/best")
except Exception:
    loaded = None
    err = "No trained model. Run: python train.py --model_type auto"

meta = _load_meta()
threshold = float(meta.get("threshold", 0.5))

with st.sidebar:
    st.subheader("Fraud Model")
    if meta:
        st.code(json.dumps(meta, indent=2), language="json")
    threshold = st.slider("Fraud threshold", 0.0, 1.0, threshold, 0.01)
    st.write("Tip: Increase threshold to reduce false positives.")

    st.divider()
    st.subheader("AI Chat Bot")
    chat_backend = st.radio("Backend", ["local", "openai"], help="Local = no API key, OpenAI = better quality")
    chat_api_key = st.text_input("OpenAI API key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    if chat_backend == "openai":
        st.caption("Get a key at platform.openai.com")

tab1, tab2, tab3 = st.tabs(["Single SMS", "Batch CSV", "AI Chat Bot"])

with tab1:
    if err:
        st.warning(err)
    else:
        colA, colB = st.columns([2, 1], gap="large")
        with colA:
            sms = st.text_area(
                "Paste an SMS message",
                height=200,
                placeholder="e.g. Congratulations! You won a prize. Click http://...",
            )
            run = st.button("Detect fraud", type="primary", use_container_width=True)

        with colB:
            st.markdown("### Result")
            if run:
                probs = predict_proba(loaded, [sms])
                proba = float(probs[0])
                is_fraud = proba >= threshold
                st.metric("Fraud probability", f"{proba:.3f}")
                st.metric("Prediction", "FRAUDULENT" if is_fraud else "LEGITIMATE")
                st.progress(min(max(proba, 0.0), 1.0))
            else:
                st.write("Click **Detect fraud** to score the message.")

with tab2:
    if err:
        st.warning(err)
    else:
        st.markdown("Upload a CSV, pick the text column, and download predictions.")
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            if df.empty:
                st.warning("CSV is empty.")
            else:
                text_col = st.selectbox("Text column", options=list(df.columns))
                if st.button("Score CSV", type="primary"):
                    texts = df[text_col].astype(str).tolist()
                    probs = predict_proba(loaded, texts)
                    out = df.copy()
                    out["fraud_probability"] = probs
                    out["is_fraud"] = out["fraud_probability"] >= threshold
                    st.dataframe(out.head(50), use_container_width=True)

                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name="sms_predictions.csv", mime="text/csv")

with tab3:
    st.markdown("Ask any text-based question. The AI will answer.")
    st.caption("Local mode: first reply may take ~30s to load the model. OpenAI: faster, better quality.")
    if st.button("Clear chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg.role):
            st.write(msg.content)

    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.chat_history.append(ChatMessage(role="user", content=prompt))
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_response(
                    prompt,
                    st.session_state.chat_history[:-1],
                    backend=chat_backend,
                    api_key=chat_api_key or None,
                )
            st.write(reply)
        st.session_state.chat_history.append(ChatMessage(role="assistant", content=reply))


"""AI chatbot for answering text-based questions similar to ChatGPT."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


# System prompt - friendly, personable, ChatGPT-like
SYSTEM_PROMPT = """You are a friendly, warm AI assistant specializing in SMS fraud and spam detection. You talk like a real person in a casual chat.

Personality:
- Warm and welcoming. Respond naturally to greetings (good morning, hi, hello, good evening) as a friend would.
- Conversational and helpful. Use a natural tone, occasional empathy, and keep it human.
- When users say thanks, you're welcome, goodbye - reply warmly and briefly.
- For SMS fraud questions, explain clearly but stay approachable.

Respond to greetings and small talk naturally (e.g., "Good morning!" -> "Good morning! Hope you're having a great day. How can I help you today - anything about SMS fraud or staying safe?"). Keep responses concise and friendly."""


def get_response_openai(
    messages: list[ChatMessage],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Use OpenAI API. Requires OPENAI_API_KEY."""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key.strip())
        
        # Build messages with system prompt
        formatted = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for m in messages:
            formatted.append({"role": m.role, "content": m.content})
        
        print(f"Calling OpenAI with model: {model}")
        resp = client.chat.completions.create(
            model=model, 
            messages=formatted,
            temperature=0.7,
            max_tokens=500
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Error: {e}"


def _greeting_response(msg: str) -> str | None:
    """Return a warm, personable reply for greetings and casual chat. None if not a greeting."""
    q = msg.lower().strip()
    # Normalize: collapse spaces, remove punctuation for matching
    qnorm = " ".join(q.split())
    if not qnorm or len(qnorm) > 80:
        return None

    # Morning - match "good morning", "morning", "gm", "goodmorning"
    if "good morning" in qnorm or qnorm == "morning" or qnorm.startswith("morning ") or qnorm == "gm" or qnorm == "goodmorning":
        return "Good morning! Hope you're having a great start to your day. How can I help you today?"
    if "good evening" in qnorm or qnorm == "evening" or qnorm.startswith("evening ") or qnorm == "ge":
        return "Good evening! Hope your day went well. What can I help you with?"
    if "good afternoon" in qnorm or qnorm == "afternoon" or qnorm.startswith("afternoon "):
        return "Good afternoon! How can I assist you today?"
    if "good night" in qnorm or qnorm == "night" or qnorm.startswith("night ") or qnorm == "gn":
        return "Good night! Rest well. Feel free to drop by anytime if you have questions about SMS fraud or staying safe."

    # Hi / Hello / Hey
    if qnorm in ("hi", "hello", "hey", "heya", "yo", "hi there", "hello there", "hey there"):
        return "Hi there! Nice to chat with you. I'm here to help with SMS fraud, spam, or anything about staying safe online. What's on your mind?"
    if qnorm in ("hi!", "hello!", "hey!"):
        return "Hey! Great to hear from you. How can I help you today?"
    if qnorm in ("hello", "hii", "hiii", "helloo"):
        return "Hello! Good to meet you. What would you like to talk about - SMS fraud, spam, or something else?"

    # Thanks
    if any(x in qnorm for x in ["thank", "thanks", "thx", "appreciate"]):
        return "You're welcome! Glad I could help. Reach out anytime if you have more questions."
    if qnorm in ("ok", "okay", "got it", "alright", "cool"):
        return "Sounds good! Let me know if you need anything else."

    # How are you / how's it going
    if any(x in qnorm for x in ["how are you", "how r u", "how're you", "how are u", "how do you do"]):
        return "I'm doing great, thanks for asking! Ready to help whenever you need. How about you - what brings you here today?"
    if any(x in qnorm for x in ["how's it going", "how you doing", "what's up", "whats up", "sup", "wassup"]):
        return "Going well! Just here to help with SMS fraud and staying safe. What can I do for you?"

    # Bye
    if any(x in qnorm for x in ["bye", "goodbye", "see you", "later", "take care"]):
        return "Bye! Take care and stay safe. Come back anytime if you have questions!"

    # Who are you / What's your name
    if any(x in qnorm for x in ["who are you", "what are you", "what's your name", "whats your name", "your name"]):
        return "I'm your SMS fraud assistant - here to help you spot spam, phishing, and stay safe online. Think of me as a friendly expert in your pocket. What would you like to know?"

    return None


def _fallback_response(question: str) -> str:
    """Simple fallback when local model fails - useful SMS fraud tips."""
    greeting = _greeting_response(question)
    if greeting:
        return greeting

    q = question.lower()
    if any(w in q for w in ["spam", "fraud", "fake", "phishing"]):
        return (
            "SMS fraud (smishing) often uses urgency, fake prizes, or requests for personal info. "
            "Never click suspicious links or share passwords. Use our detector to check messages before acting."
        )
    if any(w in q for w in ["safe", "protect", "avoid"]):
        return (
            "To stay safe: ignore unknown senders asking for money or data, verify bank alerts by calling official numbers, "
            "and use our tool to analyze suspicious SMS before opening links."
        )
    if any(w in q for w in ["detect", "work", "how"]):
        return (
            "Our detector uses machine learning (TF-IDF + Linear SVM) trained on thousands of SMS. "
            "It analyzes text patterns to predict if a message is fraudulent. Paste a message above to try it."
        )
    return (
        "I'm here to help with SMS fraud questions. Try asking: What is smishing? How can I stay safe? How does the detector work? "
        "For OpenAI-backed answers, switch to the OpenAI backend and add your API key."
    )


def get_response_local(question: str, history: list[ChatMessage] = None, model_name: str | None = None) -> str:
    """
    Use a local model. Greetings/casual chat get instant personable replies; others use FLAN-T5.
    """
    history = history or []
    greeting = _greeting_response(question)
    if greeting:
        return greeting

    # Try lightweight FLAN-T5 first (fast, reliable on CPU)
    try:
        from transformers import pipeline  # type: ignore

        if "pipe_flan" not in get_response_local.__dict__:
            print("Loading FLAN-T5-small (lightweight, works on CPU)...")
            setattr(get_response_local, "pipe_flan", pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1,
            ))
            print("Model loaded.")

        pipe = getattr(get_response_local, "pipe_flan")
        ctx = "SMS fraud detection. " + (" ".join(f"{m.role}: {m.content}" for m in history[-2:]))
        prompt = f"{ctx} user: {question} assistant:"
        out = pipe(prompt, max_new_tokens=120, do_sample=False)
        text = (out[0]["generated_text"] if out else "").strip()
        return sanitize_response(text or _fallback_response(question))
    except Exception as e:
        print(f"Local model error: {e}, using fallback")
        return _fallback_response(question)


def get_response(
    message: str,
    history: list[ChatMessage],
    backend: str,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Get AI response based on backend.
    backend: "openai" | "local"
    model: For openai backend, the model to use. For local backend, the model repository.
    """
    if backend == "openai":
        if not (api_key and api_key.strip()):
            return "Please set your OpenAI API key in the sidebar."
        messages = [ChatMessage(role=m.role, content=m.content) for m in history]
        messages.append(ChatMessage(role="user", content=message))
        resp = get_response_openai(messages, api_key, model=model)
        return sanitize_response(resp)

    if backend == "local":
        # Local model with history context
        resp = get_response_local(message, history, model_name=model)
        return sanitize_response(resp)

    return "Unknown backend. Choose OpenAI or Local in the sidebar."


def sanitize_response(text: str) -> str:
    """Basic sanitization and formatting for model outputs.

    - Strip excessive whitespace
    - Replace repeated whitespace/newlines with single spaces or paragraphs
    - Ensure punctuation spacing
    - Trim to reasonable length (avoid model rambling)
    """
    if not isinstance(text, str):
        text = str(text)
    # Normalize whitespace
    import re

    # Replace multiple newlines with two newlines (paragraph)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace remaining newlines with single space
    text = re.sub(r"\n", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    # Fix spacing before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Truncate overly long outputs to 800 characters
    if len(text) > 800:
        text = text[:797].rstrip() + "..."

    # Collapse repeated punctuation like "!!!" or ",,,," to a single character
    text = re.sub(r"([\.,!\?;:\-]){2,}", r"\1", text)

    # Remove leading punctuation
    text = re.sub(r"^[\.,!\?;:\-\s]+", "", text)

    return text

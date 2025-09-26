# app.py
import streamlit as st
import joblib
from transformers import pipeline
import random
import time

# ===========================
# Load Model + Vectorizer
# ===========================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_Logistic Regression.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ===========================
# HuggingFace Generator
# ===========================
generator = pipeline("text-generation", model="gpt2")

# ===========================
# Label Prompts
# ===========================
label_prompts = {
    "clean": "Write a short friendly comment (max 6 words), polite and supportive:",
    "toxic": "Write a rude sarcastic comment (max 6 words):",
    "severe_toxic": "Write an extremely aggressive, hostile comment (max 6 words):",
    "obscene": "Write a short obscene or vulgar comment (max 6 words):",
    "threat": "Write a threatening comment (max 6 words):",
    "insult": "Write a witty insult (max 6 words):",
    "identity_hate": "Write a hateful comment targeting a group (max 6 words):"
}



def generate_ai_sentence(label: str) -> str:
    """Generate a sentence for a given label using GPT2."""
    prompt = label_prompts[label]
    gen_text = generator(
        prompt,
        max_new_tokens=25,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )[0]["generated_text"]
    return gen_text.replace(prompt, "").strip()

# ===========================
# Prediction Function
# ===========================
def predict_labels(text: str, threshold: float = 0.5):
    """Predict labels for a given sentence."""
    X = vectorizer.transform([text])
    y_prob = model.predict_proba(X)

    results = {}
    for i, lbl in enumerate(labels):
        prob = y_prob[0][i]
        results[lbl] = prob > threshold
    return results

# ===========================
# AI Personas
# ===========================
personas = [
    {"name": "Alice", "color": "🟦"},
    {"name": "Bob", "color": "🟩"},
    {"name": "Charlie", "color": "🟨"},
    {"name": "Diana", "color": "🟧"},
    {"name": "Eve", "color": "🟥"},
]

# ===========================
# Streamlit Layout
# ===========================
st.set_page_config(page_title="Toxic Comment AI", page_icon="⚡", layout="wide")
st.title("⚡ Multi-Feature Toxic Comment Classifier")

tab1, tab2 = st.tabs(["1️⃣ Manual Sentence Check", "2️⃣ Live Chat Simulation"])

# ===========================
# Tab 1: Manual Check
# ===========================
with tab1:
    st.header("✍️ Manual Sentence Checker")

    text_input = st.text_area("Enter any sentence below and check if it contains toxicity:", height=120)

    if st.button("🔍 Check Sentence"):
        if text_input.strip():
            results = predict_labels(text_input)
            toxic_found = [lbl for lbl, flag in results.items() if flag]

            if toxic_found:
                st.error(f"⚠️ Toxic content detected! → {', '.join(toxic_found)}")
            else:
                st.success("✅ Clean message")
        else:
            st.warning("Please type a sentence first!")

# ===========================
# Tab 2: Live Chat Simulation
# ===========================
with tab2:
    st.header("💬 Live AI Chatroom (Toxicity Monitoring)")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Control buttons
    start_btn = st.button("▶️ Start Live Chat")
    stop_btn = st.button("⏹️ Stop Live Chat")
    clear_btn = st.button("🗑️ Clear Chat")

    if clear_btn:
        st.session_state["chat_history"] = []

    chat_box = st.empty()

    # Auto-generate messages in real-time
    if start_btn:
        st.session_state["live_chat"] = True
    if stop_btn:
        st.session_state["live_chat"] = False

    while st.session_state.get("live_chat", False):
        # Pick random persona + label
        persona = random.choice(personas)
        lbl = random.choice(["clean"] + labels)
        sentence = generate_ai_sentence(lbl)
        results = predict_labels(sentence)
        st.session_state["chat_history"].append((persona, sentence, results))

        # Show last 20 messages
        with chat_box.container():
            st.subheader("📜 Live Chat Feed")
            for persona, msg, results in reversed(st.session_state["chat_history"][-20:]):
                toxic_found = [lbl for lbl, flag in results.items() if flag]
                if toxic_found:
                    st.error(f"{persona['color']} **{persona['name']}**: {msg}\n⚠️ Toxic content → {', '.join(toxic_found)}")
                else:
                    st.success(f"{persona['color']} **{persona['name']}**: {msg}\n✅ Clean message")

        time.sleep(3)  # wait 3s before next message

# ===========================
# Footer
# ===========================
st.markdown("---")
st.caption("Built with Streamlit + HuggingFace + Scikit-Learn")


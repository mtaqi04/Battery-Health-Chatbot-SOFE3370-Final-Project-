# Streamlit Chatbot UI (Sprint 4 - Task 1)
# ----------------------------------------
# Run locally:
#   python3 -m pip install streamlit
#   streamlit run chatbot/app.py

import time
import streamlit as st

st.set_page_config(page_title="Battery Health Chatbot", page_icon="🔋", layout="centered")

# ----- Sidebar -----
with st.sidebar:
    st.header("🔧 Controls")
    st.write("This is a UI skeleton with placeholder responses.")
    st.caption("Next sprints: load model, threshold logic, OpenAI integration.")
    st.divider()
    st.subheader("SOH Threshold")
    _ = st.slider("Classification threshold", min_value=0.4, max_value=0.9, value=0.6, step=0.05)
    st.divider()
    st.markdown("**Team:** Mohammad • Logan • Titobi • Nicholas • Mohit")

# ----- Header -----
st.markdown("<h1 style='text-align:center;'>🔋 Battery Health Chatbot</h1>", unsafe_allow_html=True)
st.caption("Sprint 4 / Task 1 — Basic interface with placeholder replies")

# ----- Session State for chat history -----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about battery SOH or say 'check battery soh' to try a placeholder demo."}
    ]

# ----- Chat History -----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----- Placeholder brain -----
def placeholder_bot_reply(user_text: str) -> str:
    t = user_text.strip().lower()

    # Simple keyword routing (placeholder logic)
    if "check battery soh" in t or "check soh" in t:
        # Fake, deterministic demo output
        return (
            "Estimated **SOH = 0.82** (placeholder)\n\n"
            "- Status: **Healthy** (≥ threshold)\n"
            "- Next: integrate the trained model (`models/soh_linear_model.pkl`) in Sprint 4 Task 2."
        )
    elif "help" in t or "how" in t or "what can you do" in t:
        return (
            "I can **(placeholder)**:\n"
            "1) Echo general questions\n"
            "2) Demo a fake SOH check (`check battery soh`)\n\n"
            "Coming soon: real model predictions + ChatGPT answers."
        )
    else:
        # Generic echo for now
        return f"(placeholder) You said: **{user_text}**\n\nSoon I'll answer with the real model and ChatGPT!"

# ----- Chat Input -----
if user := st.chat_input("Type a message (e.g., 'check battery soh')"):
    st.session_state.messages.append({"role": "user", "content": user})
    with st.chat_message("user"):
        st.markdown(user)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(0.6)  # tiny pause for UX
            reply = placeholder_bot_reply(user)
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ----- Footer -----
st.divider()
st.caption("UI only — model & ChatGPT integration will be added in Sprint 4 Tasks 2–5.")

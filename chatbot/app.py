# Streamlit Chatbot UI (Sprint 4 - Task 2)
# ----------------------------------------
# Run locally:
#   python3 -m pip install streamlit
#   streamlit run chatbot/app.py
#   python3 -m streamlit run chatbot/app.py

import time
import streamlit as st
import re
from predict_soh import load_model, predict_soh, DEFAULT_THRESHOLD

import google.generativeai as genai
from dotenv import load_dotenv
import os

# Loading Api key

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Flag: is Gemini actually usable?
GEMINI_ENABLED = bool(GEMINI_API_KEY)

if GEMINI_ENABLED:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ No GEMINI_API_KEY found. Gemini answers will be disabled.")


# Helper function for gemini

def ask_gemini(user_input, threshold):
    # If there is no API key, do NOT call Gemini at all
    if not GEMINI_ENABLED:
        return (
            "⚠️ Gemini API is not configured on this machine.\n\n"
            "I can still help you with **battery SOH predictions** if you provide "
            "21 voltage readings (U1–U21) using the command:\n\n"
            "`check battery soh: <21 comma-separated values>`"
        )

    context = (
        f"You are a battery health expert chatbot for technical and non-technical users. "
        f"SOH (State of Health) is a value between 0 and 1; batteries with SOH ≥ {threshold} are considered healthy. "
        "If asked about battery maintenance, recycling, common lifespan problems, storage, or voltage readings, give specific, actionable advice. "
        "Refer users to provide 21 voltage readings (U1-U21) for prediction-related queries. "
        "Use concise, yet detailed explanations. Avoid vague or generic answers."
    )
    prompt = context + "\n\nUser: " + user_input
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(prompt)
    return response.text


st.set_page_config(page_title="Battery Health Chatbot", page_icon="🔋", layout="centered")

def extract_voltages_from_text(text: str) -> list[float]:
    """
    Extract numeric values from the user's message.

    Example:
      "check battery soh: 3.91, 3.90, 3.88, ..." → [3.91, 3.90, 3.88, ...]

    Returns:
        List of floats (may be wrong length; caller must validate).
    """
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return [float(n) for n in numbers]


# ----- Sidebar -----
with st.sidebar:
    st.header("🔧 Controls")
    st.write("Adjust the SOH threshold for classification.")
    st.divider()
    st.subheader("SOH Threshold")
    threshold = st.slider("Classification threshold", min_value=0.4, max_value=0.9, value=0.6, step=0.05)
    st.caption(f"Batteries with SOH ≥ {threshold} are classified as **Healthy**")
    # Move button here
    st.divider()
    st.subheader("🔍 Quick Battery SOH Check")
    if st.button("Check Battery SOH"):
        # Set a flag instead of running prediction immediately
        st.session_state["quick_soh_request"] = True
    st.divider()
    st.markdown("**Team:** Mohammad • Logan • Titobi • Nicholas • Mohit")

# ----- Initialize Model in Session State -----
@st.cache_resource
def get_model():
    """Load the trained model once and cache it."""
    try:
        model = load_model()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if "model" not in st.session_state:
    with st.spinner("Loading battery health model..."):
        st.session_state.model = get_model()

# ----- Header -----
st.markdown("<h1 style='text-align:center;'>🔋 Battery Health Chatbot</h1>", unsafe_allow_html=True)
st.caption("Ask me about battery SOH or provide voltage readings (U1-U21) for prediction")

# ----- Session State for chat history -----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can help you check battery health. You can:\n\n1. **Provide voltage readings**: Send 21 comma-separated values (U1-U21)\n2. **Ask for help**: Type 'help' or 'how'\n3. **Example**: `0.0025,0.0125,0.0035,0.0019,0.0027,0.0057,0.0193,0.0202,0.0027,0.0197,0.0062,0.0042,0.0019,0.0157,0.0484,0.0508,0.0027,0.0346,0.0101,0.0119,0.0025`"}
    ]

# ----- Chat History -----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----- Helper Functions -----
def extract_features_from_text(text: str) -> list:
    """
    Extract numeric values from user text input.
    Supports comma-separated values, space-separated, or mixed formats.
    """
    # Remove common prefixes/suffixes and extract numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if len(numbers) >= 21:
        try:
            features = [float(num) for num in numbers[:21]]
            return features
        except ValueError:
            return None
    return None

def format_prediction_response(prediction_result: dict, threshold: float) -> str:
    """Format the prediction result as a user-friendly message."""
    soh = prediction_result['soh']
    condition = prediction_result['condition']
    
    # Determine emoji based on condition
    emoji = "✅" if condition == "Healthy" else "⚠️"
    
    response = (
        f"{emoji} **Battery Health Prediction Results**\n\n"
        f"**Predicted SOH:** {soh:.4f}\n\n"
        f"**Status:** {condition}\n"
        f"**Classification Threshold:** {threshold}\n\n"
    )
    
    if condition == "Healthy":
        response += f"✅ This battery is in good condition (SOH ≥ {threshold})"
    else:
        response += f"⚠️ This battery may have issues (SOH < {threshold})"
    
    return response

# ----- Bot Reply Function -----
def bot_reply(user_text: str, threshold: float, model) -> str:
    """
    Route user message either to:
      - SOH prediction (if 'check battery soh' command detected)
      - Gemini Q&A (for general questions)
    """
    text_lower = user_text.lower()

    # SOH prediction command
    if "check battery soh" in text_lower:
        if model is None:
            return (
                "⚠️ The SOH prediction model is not loaded. "
                "Please make sure `models/soh_linear_model.pkl` exists and was "
                "generated by the training notebook."
            )

        voltages = extract_voltages_from_text(user_text)

        if len(voltages) != 21:
            return (
                "To check battery SOH, please provide **21 voltage readings (U1–U21)** "
                "after the command.\n\n"
                "Example:\n"
                "`check battery soh: 3.91, 3.90, 3.88, 3.87, 3.89, 3.90, 3.91, "
                "3.92, 3.90, 3.89, 3.88, 3.87, 3.86, 3.85, 3.84, 3.83, 3.82, "
                "3.81, 3.80, 3.79, 3.78`"
            )

        # Use the existing single-sample prediction function
        result = predict_soh(voltages, threshold=threshold, model=model)
        soh = result["soh"]
        condition = result["condition"]  # "Healthy" or "Has a Problem"

        emoji = "✅" if condition == "Healthy" else "⚠️"

        return (
            f"{emoji} **Predicted SOH:** `{soh:.4f}`\n"
            f"Threshold: `{threshold:.2f}` → Status: **{condition}**\n\n"
            "You can ask follow-up questions about maintenance, lifespan, "
            "or charging based on this result."
        )

    # Default: sending to Gemini for general battery questions
    return ask_gemini(user_text, threshold)

# ----- Process sidebar quick SOH request -----
if st.session_state.get("quick_soh_request", False):
    st.session_state["quick_soh_request"] = False  # reset flag

    test_data = [
        0.0025, 0.0125, 0.0035, 0.0019, 0.0027, 0.0057, 0.0193,
        0.0202, 0.0027, 0.0197, 0.0062, 0.0042, 0.0019, 0.0157,
        0.0484, 0.0508, 0.0027, 0.0346, 0.0101, 0.0119, 0.0025
    ]

    if "model" not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ Model not loaded. Please refresh the page.")
    else:
        try:
            prediction_result = predict_soh(test_data, threshold=threshold, model=st.session_state.model)

            # Inline formatting (avoids calling format_prediction_response to prevent NameError)
            soh = prediction_result.get('soh')
            condition = prediction_result.get('condition')
            emoji = "✅" if condition == "Healthy" else "⚠️"

            soh_reply = (
                f"{emoji} **Battery Health Prediction Results**\n\n"
                f"**Predicted SOH:** {soh:.4f}\n\n"
                f"**Status:** {condition}\n"
                f"**Classification Threshold:** {threshold}\n\n"
            )
            if condition == "Healthy":
                soh_reply += f"✅ This battery is in good condition (SOH ≥ {threshold})"
            else:
                soh_reply += f"⚠️ This battery may have issues (SOH < {threshold})"

            st.success("✅ Battery SOH Check Completed!")
            st.markdown(soh_reply)

            # Append result to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": soh_reply
            })
        except Exception as e:
            st.error(f"❌ Error during SOH check: {e}")


# ----- Chat Input -----
if user := st.chat_input("Provide voltage readings (U1-U21) or ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user})
    with st.chat_message("user"):
        st.markdown(user)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            reply = bot_reply(user, threshold, st.session_state.model)
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ----- Footer -----
st.divider()
st.caption("Model: Linear Regression SOH Predictor | Threshold-based Classification")

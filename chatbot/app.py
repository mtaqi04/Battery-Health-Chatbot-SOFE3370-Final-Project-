# Streamlit Chatbot UI (Sprint 4 - Task 2)
# ----------------------------------------
# Run locally:
#   python3 -m pip install streamlit
#   streamlit run chatbot/app.py
#   python3 -m streamlit run chatbot/app.py
#----------------------------------------
# Another way to run:
# Create a virtual enviornement using py -m venv .venv
# Activate it using .venv\Scripts\activate.ps1 (Windows) or source .venv/bin/activate (MacOS/Linux)
# Install depedencies using py -m pip install streamlit
# If running errors occur, try installing additional dependencies using: pip install -r requirements.txt
# Run the app using streamlit run chatbot/app.py
#----------------------------------------

import time
import streamlit as st
import re
from predict_soh import load_model, predict_soh, DEFAULT_THRESHOLD

import openai
from dotenv import load_dotenv
import os

# Loading Api key

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#keytest
def bot_reply(user_text: str, threshold: float, model) -> str:
    print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))
    ...


# Helper function for chatgpt

def ask_chatgpt(message):
    client = openai.OpenAI()  # This creates the client
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful expert on batteries and battery health."},
            {"role": "user", "content": message}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content



st.set_page_config(page_title="Battery Health Chatbot", page_icon="🔋", layout="centered")

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
    """Process user input and generate appropriate response."""
    t = user_text.strip().lower()
    
    # Check if user wants help
    if "help" in t or "how" in t or "what can you do" in t:
        return (
            "**I can help you with:**\n\n"
            "1. **Battery Health Prediction**: Provide 21 voltage readings (U1-U21)\n"
            "   - Format: comma or space-separated numbers\n"
            "   - Example: `0.0025,0.0125,0.0035,...,0.0025` (21 values)\n\n"
            "2. **General Questions**: Ask me anything about battery health, maintenance, or recycling\n\n"
            "3. **SOH Explanation**: Ask what SOH means or how to interpret results\n\n"
            "**Tip:** Adjust the threshold in the sidebar to change classification criteria!"
        )
    
    # Check if user is asking about SOH
    if "soh" in t and ("what" in t or "mean" in t or "explain" in t):
        return (
            "**State of Health (SOH)** is a measure of battery condition:\n\n"
            f"- **Range**: 0.0 to 1.0 (or 0% to 100%)\n"
            f"- **Healthy**: SOH ≥ {threshold} (good condition)\n"
            f"- **Has a Problem**: SOH < {threshold} (may need attention)\n\n"
            "SOH indicates remaining capacity compared to a new battery."
        )
    
    # Try to extract features for prediction
    features = extract_features_from_text(user_text)
    
    if features:
        # User provided voltage readings - run prediction
        if model is None:
            return "❌ **Error**: Model not loaded. Please refresh the page."
        
        try:
            prediction_result = predict_soh(features, threshold=threshold, model=model)
            return format_prediction_response(prediction_result, threshold)
        except Exception as e:
            return f"❌ **Error during prediction**: {str(e)}\n\nPlease check your input format and try again."
    
    # Check for "check battery soh" or similar without features
    if "check" in t and ("battery" in t or "soh" in t):
        return (
            "To check battery health, please provide 21 voltage readings (U1-U21).\n\n"
            "**Example input:**\n"
            "`0.0025,0.0125,0.0035,0.0019,0.0027,0.0057,0.0193,0.0202,0.0027,0.0197,0.0062,0.0042,0.0019,0.0157,0.0484,0.0508,0.0027,0.0346,0.0101,0.0119,0.0025`\n\n"
            "Or type 'help' for more information."
        )
    
    # For any query not matched above, use ChatGPT for response
    return ask_chatgpt(user_text)


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

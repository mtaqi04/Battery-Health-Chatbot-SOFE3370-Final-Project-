Chatbot Integration (with Sample Q&A)
Battery Health Chatbot – Integration of SOH Model and AI Assistant

4. Chatbot Integration

This section explains how the battery health chatbot was implemented, how the SOH prediction model was integrated, and how the Gemini API was added to support natural language battery-related conversations. The chatbot serves as the user-facing component of the system and allows users to perform SOH checks and ask general battery questions within an intuitive interface.

4.1 Overview of Chatbot Architecture

The chatbot interface was built using Streamlit, selected for its clean UI components, fast iteration workflow, and suitability for interactive ML demos.

The architecture includes:

chatbot/app.py – UI Layer
Implements layout, sidebar, chat interface, and user message routing.

chatbot/predict_soh.py – Model Layer
Loads the trained regression model and performs SOH prediction + threshold-based classification.

Gemini Integration – AI Assistant Layer
Uses Google’s gemini-2.0-flash-lite model for general Q&A.

Environment Configuration
.env file stores API keys securely using the python-dotenv package.

The chatbot supports:

Numeric SOH prediction from 21 voltage readings (U1–U21)

Dynamic threshold classification (“Healthy” or “Has a Problem”)

Natural-language Q&A for battery maintenance, storage, safety, lifespan, etc.

4.2 Model Integration & SOH Prediction Workflow

The system loads the Linear Regression SOH model (models/soh_linear_model.pkl) trained during earlier sprints.
When a user enters voltage readings, the chatbot intelligently extracts values and triggers prediction.

Workflow Summary

Load model (cached)

st.session_state.model = load_model()


Parse voltages from user input

check battery soh: 3.90, 3.91, ...


Ensure exactly 21 features are provided

Predict SOH and classify:

predict_soh(voltages, threshold, model)


Return formatted results to user
Including:

Predicted SOH

Threshold used

Final classification (Healthy / Has a Problem)

Appropriate emoji (✅ / ⚠️)

This pipeline makes SOH predictions reliable, structured, and easy for the user to understand.

4.3 Gemini API Integration

The chatbot integrates Gemini for general-purpose battery Q&A.

API Setup

Store API key safely in .env:

GEMINI_API_KEY=(THE_API_KEY)


Load key and configure Gemini:

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


Query Gemini using:

model = genai.GenerativeModel("gemini-2.0-flash-lite")
response = model.generate_content(prompt)

Fallback Behavior

If the API key is missing or invalid:

The chatbot disables Gemini functionality

SOH predictions still work properly

User is informed that Gemini is offline

This ensures the application is still functional in offline/limited environments.

4.4 Chat Flow & Decision Logic

The chatbot examines the user’s message and chooses the correct execution path.

1️⃣ If user requests SOH prediction

Condition:

if "check battery soh" in user_text.lower():


→ Extract voltage readings
→ Validate count
→ Run SOH model
→ Return SOH + health status

2️⃣ Otherwise, treat it as a general question

→ Send message to ask_gemini()
→ Return natural-language response

This hybrid routing makes the chatbot both:

A diagnostic tool

A battery knowledge assistant

—

4.5 Sample Chatbot Interactions
A. SOH Prediction Example

User Input:

check battery soh: 0.0025, 0.0125, 0.0035, 0.0019, 0.0027, ... (21 values total)


Chatbot Output:

⚠️ Predicted SOH: 0.5328
Threshold: 0.60 → Status: Has a Problem

This battery may have issues (SOH < 0.6).


📌 Screenshot Placeholder

reports/screenshots/chatbot_demo_soh.png

B. Gemini Q&A Example

User Input:

How do I safely store a lithium-ion battery?


Gemini Output (shortened):

Store lithium-ion batteries at 40–60% charge, avoid heat above 25°C,
and check periodically for swelling or damage.


📌 Screenshot Placeholder

reports/screenshots/chatbot_demo_qa.png

4.6 Summary

The chatbot successfully integrates ML prediction and conversational intelligence into a single unified interface.

The system now supports:

✔️ Real-time SOH predictions using 21 voltage readings

✔️ Threshold-based classification with adjustable settings

✔️ Natural-language Q&A powered by Gemini

✔️ Graceful handling of missing API keys

✔️ A polished Streamlit interface compatible with the final deliverable

This completes the required chatbot components for the final report, showcasing a working end-to-end battery health assistant.
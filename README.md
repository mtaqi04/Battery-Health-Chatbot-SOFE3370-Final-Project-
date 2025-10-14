🔋 Battery Health Prediction & Chatbot System
🧠 Overview

This project develops an intelligent system that predicts the State of Health (SOH) of retired lithium-ion batteries and integrates the results into an interactive chatbot interface.
The chatbot allows users to check battery health predictions and ask general questions related to battery maintenance, recycling, and performance through natural conversation.

The project follows an Agile development methodology over 7 weeks, with each sprint focusing on a key stage — from data preprocessing and model training to chatbot integration and testing.

📘 Table of Contents

Project Objectives

System Architecture

Dataset Description

Implementation Details

Key Features

Setup Instructions

Project Deliverables

Team & Roles

License

🎯 Project Objectives

Predict battery State of Health (SOH) using voltage readings (U1–U21).

Implement a Linear Regression model to estimate battery SOH.

Apply a threshold-based classification (default 0.6) to identify whether a battery is healthy or has issues.

Develop a chatbot interface to display predictions and respond to general user questions using the ChatGPT API.

Follow an Agile methodology with rotating team roles and collaborative sprints.

⚙️ System Architecture
PulseBat Dataset → Data Preprocessing → Linear Regression Model → Threshold Logic 
→ Chatbot Interface (Streamlit) → User Interaction


Data Preprocessing: Cleaning, normalization, and visualization of U1–U21 readings.

Model Training: Linear Regression model predicts SOH values.

Classification: Threshold logic categorizes battery health (≥0.6 = Healthy).

Chatbot Integration: Combines model predictions and ChatGPT responses.

User Interface: Streamlit-based front-end for interactive Q&A.

📊 Dataset Description

Dataset Name: PulseBat Dataset
Files:

PulseBat Dataset.xlsx – Contains voltage readings (U1–U21) and target SOH.

PulseBat Data Description.md – Explains data collection and measurement process.

Feature Overview:

Feature	Description
U1–U21	Voltage measurements for each cell during pulse tests
SOH	Target variable representing battery health (0–1 scale)

Cleaning & Preprocessing Steps:

Handle missing or abnormal data.

Normalize feature ranges for training consistency.

Visualize correlations between voltage readings and SOH.

🧩 Implementation Details
🧹 Sprint 1: Data Cleaning & Exploration

Process and visualize dataset using Pandas & Matplotlib.

Deliverables: data_preprocessing.ipynb, cleaned_pulsebat.csv.

🤖 Sprint 2: Model Development

Train Linear Regression model and evaluate performance (R², MSE, MAE).

Deliverables: model_training.ipynb, soh_linear_model.pkl.

⚖️ Sprint 3: Threshold Logic & Evaluation

Apply threshold logic (default 0.6) and generate classification metrics.

Deliverables: predict_soh.py, evaluation notebook, summary table.

💬 Sprint 4: Chatbot Development

Build Streamlit-based chatbot integrating model predictions and ChatGPT API.

Deliverables: app.py, demo conversation examples.

🔗 Sprint 5: Integration & Testing

Merge components, fix bugs, and improve UI/UX.

Deliverables: Integrated chatbot system, test logs, screenshots.

🧾 Sprint 6: Documentation & Presentation

Compile final report and create presentation slides.

Deliverables: Final_Report.pdf, Presentation_Slides.pptx, final GitHub repo.

✨ Key Features

Battery SOH Prediction: Linear Regression model trained on real battery data.

Customizable Threshold: Users can adjust the SOH threshold for classification.

AI Chatbot Interface: Integrated ChatGPT for general battery-related queries.

User-Friendly Frontend: Simple Streamlit-based UI for interactive use.

Modular Codebase: Easy-to-understand structure with reusable components.

💻 Setup Instructions
🧩 Prerequisites

Python 3.10 or above

API key for OpenAI

📦 Install Dependencies
pip install -r requirements.txt

🔑 Configure API Key

Create a .env file in the root directory and add:

OPENAI_API_KEY=your_api_key_here

🚀 Run the Chatbot
streamlit run app.py

📦 Project Deliverables
File	Description
data_preprocessing.ipynb	Data cleaning, feature extraction, and visualization
model_training.ipynb	Model training and performance evaluation
soh_linear_model.pkl	Trained Linear Regression model
predict_soh.py	Threshold-based SOH classification logic
app.py	Streamlit chatbot application
Final_Report.pdf	Complete technical report
Presentation_Slides.pptx	Final presentation slides
👥 Team & Roles
Sprint	Lead Roles	Learning Roles
Week 1	Product Owner / Scrum Master	All Members
Week 2	Data Engineer	Others shadow data tasks
Week 3	ML Engineer	Others shadow modeling
Week 4	Chatbot Developer	Others shadow API/UI work
Week 5	Integration Lead	All assist in QA
Week 6	Report & Presentation Leads	All contribute equally

All members rotated roles to gain hands-on experience across the full development lifecycle.

🪪 License

This project is developed as part of SOFE 3370 – Software Design and Architecture (York University) and is intended for academic and educational purposes.

© 2025 Battery Health Chatbot Development Team

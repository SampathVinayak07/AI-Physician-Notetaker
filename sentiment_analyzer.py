import streamlit as st
import pandas as pd
from transformers import pipeline
import json

# ---------- Utility ----------
def read_txt(file) -> str:
    return file.read().decode("utf-8")

def isolate_patient_dialogue(transcript: str) -> list:
    patient_lines = []  # FIX: added initialization
    lines = transcript.splitlines()
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line.lower().startswith("patient:"):
            utterance = cleaned_line[len("patient:"):].strip()
            if utterance:
                patient_lines.append(utterance)
    return patient_lines

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Patient Sentiment Analyzer", layout="centered")
st.title("🩺 Patient Sentiment & Intent Analyzer")
st.markdown("Upload a clinical transcript (.txt) to analyze patient sentiment and intent.")

# ---------- Load models ----------
from transformers import pipeline

try:
    # Use a public Hugging Face model instead of a missing local folder
    sentiment_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    st.sidebar.success("✅ Sentiment model loaded from Hugging Face successfully!")
except Exception as e:
    sentiment_classifier = None
    st.sidebar.error("⚠️ Could not load sentiment model from Hugging Face.")
    st.sidebar.caption(str(e))


try:
    intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    st.sidebar.success("✅ Zero-shot intent model loaded successfully!")
except Exception as e:
    intent_classifier = None
    st.sidebar.error("⚠️ Failed to load zero-shot model.")
    st.sidebar.caption(str(e))

# ---------- File Upload ----------
uploaded_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])

if uploaded_file:
    transcript = read_txt(uploaded_file)
    with st.expander("Show Full Transcript"):
        st.text_area("", transcript, height=250)

    patient_dialogue = isolate_patient_dialogue(transcript)
    if not patient_dialogue:
        st.warning("No patient dialogue found. Ensure lines start with 'Patient:'.")
    else:
        st.subheader("Patient Dialogue Analysis")
        results = []

        if sentiment_classifier and intent_classifier:
            with st.spinner("Analyzing patient dialogue..."):
                intent_labels = [
                    "Seeking reassurance",
                    "Reporting symptoms",
                    "Expressing concern",
                    "Expressing gratitude",
                    "Reporting outcome"
                ]
                for line in patient_dialogue:
                    res = {"text": line}
                    sentiment = sentiment_classifier(line)[0]
                    res["sentiment"] = sentiment['label']
                    res["sentiment_score"] = round(sentiment['score'], 2)
                    intent = intent_classifier(line, candidate_labels=intent_labels)
                    res["intent"] = intent['labels'][0]
                    res["intent_score"] = round(intent['scores'][0], 2)
                    results.append(res)

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            json_output = json.dumps(results, indent=2)
            st.download_button("Download JSON", data=json_output,
                               file_name="patient_sentiment.json",
                               mime="application/json")
        else:
            st.error("Models not loaded properly. Please check sidebar messages.")

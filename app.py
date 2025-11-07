import streamlit as st
import json, re, functools
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ---------- Optional imports ----------
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("Transformers library not installed. Please run: pip install transformers torch")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ---------- Utility ----------
def read_txt(file) -> str:
    return file.read().decode("utf-8")

def isolate_patient_dialogue(transcript: str) -> list:
    patient_lines = []
    for line in transcript.splitlines():
        line = line.strip()
        if line.lower().startswith("patient:"):
            utterance = line[len("patient:"):].strip()
            if utterance:
                patient_lines.append(utterance)
    return patient_lines

# ---------- Cached pipelines ----------
@st.cache_resource
def get_pipeline(task, model, **kwargs):
    if not TRANSFORMERS_AVAILABLE: return None
    return pipeline(task, model=model, **kwargs)

# ---------- Loaders ----------
get_medical_ner = functools.lru_cache(None)(lambda: get_pipeline("ner", "d4data/biomedical-ner-all", aggregation_strategy="simple"))
get_general_ner = functools.lru_cache(None)(lambda: get_pipeline("ner", "dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple"))
get_qa = functools.lru_cache(None)(lambda: get_pipeline("question-answering", "deepset/roberta-large-squad2"))
get_sentiment = functools.lru_cache(None)(lambda: get_pipeline("text-classification", "distilbert-base-uncased-finetuned-sst-2-english"))
get_intent = functools.lru_cache(None)(lambda: get_pipeline("zero-shot-classification", "facebook/bart-large-mnli"))

# ---------- Extraction ----------
def extract_patient_name(text):
    general_ner = get_general_ner()
    if not general_ner:
        return None
    results = general_ner(text)
    for r in results:
        if r['entity_group'] == 'PER':
            return r['word']
    match = re.search(r"(?:Mr\.|Ms\.|Mrs\.)\s+[A-Za-z]+", text)
    return match.group(0) if match else None

def extract_medical_info(text):
    medical_ner = get_medical_ner()
    if not medical_ner:
        return {}
    results = medical_ner(text)
    extracted = defaultdict(list)
    for ent in results:
        group = ent['entity_group']
        word = ent['word'].replace("##", "")
        if group == "Sign_symptom":
            extracted["Symptoms"].append(word)
        elif group in ["Medication", "Therapeutic_procedure", "Diagnostic_procedure"]:
            extracted["Treatment"].append(word)
    return dict(extracted)

def extract_qa_info(context):
    qa = get_qa()
    if not qa:
        return {}
    questions = {
        "Diagnosis": "What was the patient diagnosed with?",
        "Current_Status": "What pain or symptoms is the patient still experiencing?",
        "Prognosis": "What is the doctor's prognosis?"
    }
    answers = {}
    for key, q in questions.items():
        try:
            res = qa(question=q, context=context)
            if res["score"] > 0.1:
                answers[key] = res["answer"]
        except Exception:
            continue
    return answers

# ---------- Sentiment & Intent ----------
def analyze_sentiment(patient_lines):
    sentiment_pl = get_sentiment()
    intent_pl = get_intent()
    if not sentiment_pl or not intent_pl:
        st.error("Models not loaded.")
        return []
    labels = ["Seeking reassurance", "Reporting symptoms", "Expressing concern", "Expressing gratitude", "Reporting outcome"]
    results = []
    for text in patient_lines:
        s = sentiment_pl(text)[0]
        i = intent_pl(text, candidate_labels=labels)
        results.append({
            "text": text,
            "sentiment": s["label"],
            "sentiment_score": round(s["score"], 2),
            "intent": i["labels"][0],
            "intent_score": round(i["scores"][0], 2)
        })
    return results

# ---------- SOAP Note ----------
def generate_soap_note(api_key, transcript):
    if not GENAI_AVAILABLE:
        return {"Error": "google-generativeai not installed."}
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        schema = """
        {"Subjective":{"Chief_Complaint":"","History_of_Present_Illness":""},
         "Objective":{"Physical_Exam":"","Observations":""},
         "Assessment":{"Diagnosis":"","Severity":""},
         "Plan":{"Treatment":"","Follow_Up":""}}
        """
        prompt = f"Convert the following transcript to JSON with this schema:\n{schema}\n\nTranscript:\n{transcript}"
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text and hasattr(response, "candidates"):
            text = response.candidates[0].content.parts[0].text
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"Error": str(e)}

# ---------- Streamlit Layout ----------
st.set_page_config(page_title="Physician Notetaker", layout="wide")
st.title("🩺 AI Physician Notetaker")

with st.sidebar:
    api_key = st.text_input("Google AI Studio Key (for SOAP)", type="password")
    if TRANSFORMERS_AVAILABLE:
        st.success("Transformers ready.")
    if GENAI_AVAILABLE:
        st.success("Google AI available.")

file = st.file_uploader("Upload Transcript (.txt)", type=["txt"])

if file:
    transcript = read_txt(file)
    with st.expander("Transcript", expanded=False):
        st.text_area("", transcript, height=250)

    # Clinical Summary
    st.subheader("1️⃣ Clinical Summary")
    patient_name = extract_patient_name(transcript)
    med_info = extract_medical_info(transcript)
    qa_info = extract_qa_info(transcript)
    summary = {
        "Patient_Name": patient_name,
        "Symptoms": med_info.get("Symptoms", []),
        "Treatment": med_info.get("Treatment", []),
        "Diagnosis": qa_info.get("Diagnosis"),
        "Current_Status": qa_info.get("Current_Status"),
        "Prognosis": qa_info.get("Prognosis"),
        "Extracted_On": datetime.utcnow().isoformat() + "Z"
    }
    st.json(summary)
    st.download_button("Download JSON", data=json.dumps(summary, indent=2), file_name="summary.json")

    # Sentiment Analysis
    st.divider()
    st.subheader("2️⃣ Sentiment & Intent")
    patient_lines = isolate_patient_dialogue(transcript)
    if patient_lines:
        results = analyze_sentiment(patient_lines)
        st.dataframe(pd.DataFrame(results))
        st.download_button("Download Sentiment JSON",
                           data=json.dumps(results, indent=2),
                           file_name="sentiment.json")
    else:
        st.warning("No patient dialogue found.")

    # SOAP Note
    st.divider()
    st.subheader("3️⃣ SOAP Note Generator")
    if st.button("Generate SOAP Note"):
        if not api_key:
            st.error("Please enter Google AI Studio Key.")
        else:
            with st.spinner("Generating SOAP Note..."):
                soap = generate_soap_note(api_key, transcript)
            st.json(soap)
            st.download_button("Download SOAP Note", data=json.dumps(soap, indent=2),
                               file_name="soap_note.json")

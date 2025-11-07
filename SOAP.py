import streamlit as st
import google.generativeai as genai
import json

def generate_soap_note_hybrid(api_key, transcript_text):
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return {"Error": "Authentication failed", "Details": str(e)}

    schema = {
        "Subjective": {"Chief_Complaint": "", "History_of_Present_Illness": ""},
        "Objective": {"Physical_Exam": "", "Observations": ""},
        "Assessment": {"Diagnosis": "", "Severity": ""},
        "Plan": {"Treatment": "", "Follow_Up": ""}
    }

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    prompt = f"Convert this transcript into a concise JSON object following this schema:\n{json.dumps(schema)}\n\nTranscript:\n{transcript_text}"

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text and hasattr(response, "candidates"):
            text = response.candidates[0].content.parts[0].text
        cleaned = text.replace('```json', '').replace('```', '').strip()
        data = json.loads(cleaned)
    except Exception as e:
        return {"Error": "Model parsing failed", "Details": str(e)}

    # Ensure structure completeness
    for section, fields in schema.items():
        if section not in data:
            data[section] = fields
        else:
            for f, _ in fields.items():
                data[section].setdefault(f, "Not specified")
    return data

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("🩺 Hybrid AI SOAP Note Generator")

api_key = st.text_input("Enter your Google AI Studio API Key", type="password")
uploaded_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])

if uploaded_file:
    transcript = uploaded_file.getvalue().decode("utf-8")
    st.text_area("Transcript", transcript, height=350)

    if st.button("✨ Generate SOAP Note", use_container_width=True):
        if not api_key:
            st.error("API key required.")
        else:
            with st.spinner("Generating SOAP Note..."):
                soap = generate_soap_note_hybrid(api_key, transcript)
            st.json(soap)
            st.download_button("Download JSON",
                               data=json.dumps(soap, indent=2),
                               file_name="soap_note.json",
                               mime="application/json")

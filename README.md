# 🩺 AI Physician Notetaker

An end-to-end **medical NLP system** that converts raw physician–patient conversations into structured, analyzable data.  
It performs **medical summarization**, **sentiment & intent analysis**, and **SOAP note generation** using modern transformer-based NLP models and Google’s Gemini API.

---

## 🚀 Features

1. **Clinical Information Extraction**
   - Uses medical NER and QA pipelines to identify:
     - Symptoms, treatments, diagnosis, and prognosis  
   - Outputs a structured medical summary in JSON format.

2. **Patient Sentiment & Intent Analysis**
   - Classifies emotional tone: *Anxious*, *Neutral*, or *Reassured*  
   - Detects intent such as *Reporting symptoms*, *Seeking reassurance*, or *Expressing gratitude*  
   - Uses a transformer-based sentiment model and zero-shot intent classifier.

3. **SOAP Note Generation (LLM Integration)**
   - Automatically produces structured SOAP notes (Subjective, Objective, Assessment, Plan).  
   - Powered by **Google Gemini 2.5 Pro** via `google-generativeai`.

---

## 🧠 Tech Stack

| Category | Tools / Models |
|-----------|----------------|
| **Frontend** | Streamlit |
| **Backend / Core** | Python 3.11 |
| **NLP Frameworks** | Hugging Face Transformers, SciSpacy |
| **AI Models** | |
| • Medical NER | `d4data/biomedical-ner-all` |
| • General NER | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| • QA Model | `deepset/roberta-large-squad2` |
| • Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |
| • Intent | `facebook/bart-large-mnli` |
| • SOAP LLM | Google `gemini-2.5-pro` |

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/SampathVinayak07/AI-Physician-Notetaker.git
cd AI-Physician-Notetaker
```

### 2. Create and Activate Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install google-generativeai
```

### 4. Run the Application

#### a. Full AI Notetaker

```bash
streamlit run app.py
```

#### b. Sentiment & Intent Analyzer

```bash
streamlit run sentiment_analyzer.py
```

#### c. SOAP Note Generator

```bash
streamlit run SOAP.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

## 🧪 Example Output

### Input Transcript

```text
Physician: How are you feeling today?
Patient: I had a car accident. My neck and back hurt for four weeks.
Physician: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions and now I only have occasional back pain.
```

### JSON Summary

```json
{
  "Patient_Name": "Ms. Jones",
  "Symptoms": ["Neck pain", "Back pain"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["Painkillers", "Physiotherapy"],
  "Current_Status": "Occasional back pain",
  "Prognosis": "Full recovery expected within six months"
}```

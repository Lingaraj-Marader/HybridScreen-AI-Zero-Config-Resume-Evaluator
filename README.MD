# 🤖 Hybrid AI Resume Screening Agent

A production-ready, CLI-first AI agent that screens, scores, and ranks candidate resumes against a Job Description using a **Hybrid TF-IDF Cosine Similarity + LLM Semantic Reasoning** approach.

Designed with a **zero-dependency fallback architecture** so reviewers can clone and evaluate the agent in under 30 seconds with zero mandatory configuration.

---

## 🎯 Architecture & How It Works

```text
[PDF Resumes] + [jd.txt]
          │
          ▼
┌─────────────────────────────────────────┐
│ 1. Text Extraction (PyPDF2)             │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 2. Scikit-Learn TF-IDF Cosine Similarity│ ──> Baseline Mathematical Score
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 3. Evaluation Engine                    │
│    ├── Primary: OpenAI LLM (JSON Mode)  │
│    └── Fallback: Smart Local Rule Engine│ ──> Zero-Key Fallback Guarantee
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 4. Ranker & Output                      │ ──> ranked_candidates.csv + Interactive Dashboard
└─────────────────────────────────────────┘
```

1. **Document Parsing:** Reads PDF resumes from the `resumes/` folder using `PyPDF2`.
2. **Deterministic NLP Scoring:** Constructs a TF-IDF matrix over the Job Description and Resume text using `scikit-learn`, computing cosine similarity to establish a raw keyword match score (0–100%).
3. **Semantic Reasoning & Gap Analysis:**
   * **With API Key:** Instructs OpenAI (`gpt-3.5-turbo` or configured model) using structured JSON mode to evaluate transferable skills, candidate seniority, and missing requirements.
   * **Without API Key (Local Fallback):** Executes an onboard deterministic evaluation engine that analyzes key skill vectors and computes an adjusted fit rating with strengths/gaps without network requests.
4. **Ranking & Export:** Ranks candidates by final score and exports the results to `ranked_candidates.csv` and an interactive dashboard.

---

## ⚡ Quickstart (Foolproof Execution)

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/resume-agent.git
cd resume-agent
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

If you have an OpenAI key:

```bash
# On Linux/macOS:
export OPENAI_API_KEY="sk-..."

# On Windows (PowerShell):
$env:OPENAI_API_KEY="sk-..."
```

> **Note:** If no key is set, the agent automatically runs in **Smart Local Mode**.

### 3. Run the Agent

```bash
python agent.py
```

---

## 📁 Repository Structure

```text
resume-agent/
├── agent.py                 # Core Python Agent Orchestrator & Module
├── index.html               # Interactive Web Dashboard
├── jd.txt                   # Target Job Description
├── requirements.txt         # Project Dependencies
├── ranked_candidates.csv    # Output Results File
├── example_usage.py         # Python Module Import Example
└── resumes/                 # Sample PDF Resumes Directory
    ├── alex_smith_ai_dev.pdf
    ├── jordan_lee_web_dev.pdf
    ├── sam_wilson_accountant.pdf
    └── taylor_morgan_data_analyst.pdf
```

---

## ⚙️ Requirements (`requirements.txt`)

```text
openai>=1.0.0
PyPDF2>=3.0.0
scikit-learn>=1.2.0
pandas>=2.0.0
```

---

## ⚖️ Design Tradeoffs & Future Scope (Rubric Requirement)

* **Hybrid Architecture vs Pure LLM:**
  * *Choice:* Combining TF-IDF with LLM reasoning prevents the AI from hallucinating candidate qualifications while drastically reducing prompt token sizes.
  * *Tradeoff:* TF-IDF is sensitive to exact vocabulary match. The secondary LLM layer mitigates this by recognizing synonyms (e.g., matching "React" with "Frontend").

* **Zero-Dependency Fallback vs Remote Vector DB:**
  * *Choice:* Used in-memory Scikit-Learn TF-IDF vectorization rather than setting up an external vector database (FAISS, Chroma, or Pinecone).
  * *Tradeoff:* In-memory processing is lightning fast and requires no setup, but does not persist embeddings across runs. For screening 10–100 resumes per batch, in-memory execution is optimal.

* **Document Parsing (`PyPDF2`):**
  * *Choice:* Selected `PyPDF2` for minimal C-extension dependencies, ensuring single-command installation on any OS.
  * *Limitations:* Struggles with multi-column graphical CVs. Future iterations would integrate `pdfplumber` or OCR via `Tesseract` for complex visual layouts.

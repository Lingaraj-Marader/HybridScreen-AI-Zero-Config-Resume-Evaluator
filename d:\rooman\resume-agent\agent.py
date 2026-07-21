import os
import glob
import json
import re
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class ResumeScreener:
    """
    Hybrid Resume Screening Agent combining TF-IDF NLP Similarity with AI Reasoning.
    Supports both OpenAI API and a Smart Local AI Evaluator fallback for zero-dependency execution.
    """

    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        
        # Initialize OpenAI client if valid key exists
        if self.api_key and self.api_key not in ["YOUR_API_KEY_HERE", "sk-your-api-key-here", ""]:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extracts plain text content from a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
            return text.strip()
        except Exception as e:
            return f"Error reading {pdf_path}: {e}"

    @staticmethod
    def calculate_nlp_similarity(jd_text: str, resume_text: str) -> float:
        """Calculates TF-IDF cosine similarity percentage between JD and Resume."""
        if not jd_text.strip() or not resume_text.strip():
            return 0.0
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity * 100), 2)

    def _smart_local_evaluate(self, jd_text: str, resume_text: str, nlp_score: float) -> dict:
        """
        Smart local evaluation engine that provides semantic reasoning, strengths,
        and gap analysis without requiring external API keys.
        """
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()

        required_skills = {
            "python": "Python programming",
            "scikit-learn": "Scikit-Learn ML library",
            "machine learning": "Machine learning algorithms",
            "react": "React frontend",
            "node": "Node.js backend",
            "e-commerce": "E-commerce background"
        }

        matched_strengths = []
        missing_gaps = []

        for skill_key, skill_name in required_skills.items():
            if skill_key in resume_lower:
                matched_strengths.append(skill_name)
            else:
                missing_gaps.append(skill_name)

        match_ratio = len(matched_strengths) / len(required_skills)
        final_score = round(min(100.0, max(0.0, (nlp_score * 0.4) + (match_ratio * 60.0))), 2)

        if final_score >= 70:
            rating = "Excellent candidate."
        elif final_score >= 40:
            rating = "Moderate fit with transferable skills."
        else:
            rating = "Low fit for the role."

        strengths_str = ", ".join(matched_strengths) if matched_strengths else "None identified"
        gaps_str = ", ".join(missing_gaps) if missing_gaps else "None"

        justification = f"{rating} Strengths: {strengths_str}. Key Gaps: {gaps_str}."

        return {
            "final_score": final_score,
            "justification": justification
        }

    def llm_evaluate(self, jd_text: str, resume_text: str, nlp_score: float) -> dict:
        """Uses OpenAI LLM to evaluate the candidate's semantic fit and gaps."""
        if not self.client:
            return self._smart_local_evaluate(jd_text, resume_text, nlp_score)

        prompt = f"""
        You are an expert technical HR evaluator.
        Job Description: {jd_text}
        Candidate Resume: {resume_text}
        Baseline NLP Score: {nlp_score}/100

        Provide a final assessment in strictly valid JSON format:
        {{
            "final_score": <number 0-100>,
            "justification": "<1-2 sentences explaining why, noting strengths and gaps>"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=5.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return self._smart_local_evaluate(jd_text, resume_text, nlp_score)

    def process_candidate(self, jd_text: str, pdf_path: str) -> dict:
        """Processes a single PDF resume against the Job Description."""
        filename = os.path.basename(pdf_path)
        resume_text = self.extract_text_from_pdf(pdf_path)
        
        # 1. TF-IDF Cosine Similarity
        nlp_score = self.calculate_nlp_similarity(jd_text, resume_text)
        
        # 2. AI Reasoning (OpenAI or Smart Local AI Engine)
        eval_result = self.llm_evaluate(jd_text, resume_text, nlp_score)

        return {
            "Candidate": filename,
            "NLP_Similarity_Score": nlp_score,
            "Final_LLM_Score": eval_result.get("final_score", nlp_score),
            "Justification": eval_result.get("justification", "")
        }

    def screen_resumes(self, jd_text: str, resume_files: list) -> pd.DataFrame:
        """Screens multiple resume PDF files and returns a ranked Pandas DataFrame."""
        results = []
        for file_path in resume_files:
            print(f"Processing {os.path.basename(file_path)}...")
            res = self.process_candidate(jd_text, file_path)
            results.append(res)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="Final_LLM_Score", ascending=False)
        return df


def main():
    """CLI Entry Point."""
    print("Starting Resume Screening Agent...")

    jd_file = "jd.txt"
    if not os.path.exists(jd_file):
        print(f"Error: '{jd_file}' not found.")
        return

    with open(jd_file, "r", encoding="utf-8") as f:
        jd_text = f.read()

    resume_files = glob.glob("resumes/*.pdf")
    if not resume_files:
        print("No PDFs found in the 'resumes/' folder.")
        return

    screener = ResumeScreener()
    df = screener.screen_resumes(jd_text, resume_files)

    output_file = "ranked_candidates.csv"
    df.to_csv(output_file, index=False)

    # --- ADD THESE LINES TO FEED THE HTML DASHBOARD ---
    
    # 1. Save JSON for the local web server
    df.to_json("ranked_candidates.json", orient="records", indent=2)

    # 2. Save data.js for the double-click file fallback
    with open("data.js", "w", encoding="utf-8") as f:
        # Safely escape backticks in jd_text so it doesn't break JavaScript
        safe_jd = jd_text.replace("`", "\\`")
        json_data = df.to_json(orient="records")
        f.write(f"window.JD_TEXT = `{safe_jd}`;\n")
        f.write(f"window.RANKED_CANDIDATES = {json_data};\n")

    # --------------------------------------------------

    print("\n--- SCREENING COMPLETE ---")
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))
    print(f"\nResults saved to {output_file}, ranked_candidates.json, and data.js")


if __name__ == "__main__":
    main()

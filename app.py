from flask import Flask, request, render_template, send_file
import os
import re
import docx2txt
import PyPDF2
import tempfile
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
SHORTLIST_PDF = "shortlisted_candidates.pdf"


# ---------------- HELPERS ---------------- #

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(path):
    if path.endswith(".pdf"):
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    if path.endswith(".docx"):
        return docx2txt.process(path)

    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    return ""


# ---------------- EMAIL EXTRACTION ---------------- #

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else "Not Found"


# ---------------- BIAS REMOVAL ---------------- #

def anonymize(text):
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    text = re.sub(r"\+?\d[\d\s\-]{8,}\d", " ", text)
    return text


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return text.split()


# ---------------- SKILL RULE ENGINE ---------------- #

MANDATORY_SKILLS = ["python", "machine learning", "sql", "data analysis"]

def skill_score(job, resume):
    matched = sum(1 for s in MANDATORY_SKILLS if s in job and s in resume)
    return matched / len(MANDATORY_SKILLS)


# ---------------- WORD2VEC ---------------- #

def w2v_similarity(job_tokens, resume_tokens):
    model = Word2Vec([job_tokens, resume_tokens], vector_size=100, min_count=1)

    def avg_vec(tokens):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(100)

    return cosine_similarity(
        [avg_vec(job_tokens)], [avg_vec(resume_tokens)]
    )[0][0]


# ---------------- ROUTES ---------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match():

    job_desc = request.form.get("job_description")
    top_n = int(request.form.get("top_n", 10))
    files = request.files.getlist("resumes")

    job_clean = clean(job_desc)
    job_tokens = tokenize(job_clean)

    candidates = []

    # 🔥 TEMPORARY STORAGE (BEST PRACTICE)
    with tempfile.TemporaryDirectory() as temp_dir:

        for f in files:
            if allowed_file(f.filename):
                filename = secure_filename(f.filename)
                path = os.path.join(temp_dir, filename)
                f.save(path)

                raw = extract_text(path)

                # extract before anonymization
                email = extract_email(raw)

                anon = anonymize(raw)
                resume_clean = clean(anon)

                candidates.append({
                    "name": filename.rsplit(".", 1)[0],
                    "email": email,
                    "resume_text": resume_clean
                })

        # TF-IDF
        corpus = [job_clean] + [c["resume_text"] for c in candidates]
        tfidf = TfidfVectorizer(stop_words="english")
        vectors = tfidf.fit_transform(corpus)
        tfidf_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        results = []

        for i, c in enumerate(candidates):
            w2v = w2v_similarity(job_tokens, tokenize(c["resume_text"]))
            rule = skill_score(job_clean, c["resume_text"])

            final = 0.5 * tfidf_scores[i] + 0.3 * w2v + 0.2 * rule

            results.append({
                "name": c["name"],
                "email": c["email"],
                "score": round(final * 100, 1)  # ATS Score (0–100)
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

    return render_template(
        "index.html",
        message="Bias-aware hybrid resume ranking completed",
        results=results,
        selected_top_n=top_n
    )


@app.route("/download_pdf", methods=["POST"])
def download_pdf():

    shortlisted = request.form.getlist("shortlisted")

    pdf = canvas.Canvas(SHORTLIST_PDF, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, height - 50, "Shortlisted Candidates")

    y = height - 100
    pdf.setFont("Helvetica", 11)

    for idx, entry in enumerate(shortlisted, start=1):
        name, email = entry.split("||")
        pdf.drawString(50, y, f"{idx}. {name}  |  {email}")
        y -= 20
        if y < 50:
            pdf.showPage()
            y = height - 50

    pdf.save()

    return send_file(SHORTLIST_PDF, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
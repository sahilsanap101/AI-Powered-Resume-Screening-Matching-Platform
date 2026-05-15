from flask import Flask, request, render_template, send_file
import os
import re
import docx2txt
import PyPDF2
import tempfile

from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
SHORTLIST_PDF = "shortlisted_candidates.pdf"


# ---------------- HELPERS ---------------- #

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


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

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            return f.read()

    return ""


def extract_email(text):

    match = re.search(
        r"[\w\.-]+@[\w\.-]+\.\w+",
        text
    )

    return match.group(0) if match else "Not Found"


def anonymize(text):

    text = re.sub(
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        " ",
        text
    )

    text = re.sub(
        r"\+?\d[\d\s\-]{8,}\d",
        " ",
        text
    )

    return text


def clean(text):

    text = text.lower()

    # preserve technical keywords
    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------- JD KEYWORD EXTRACTION ---------------- #

def extract_keywords(text):

    words = text.split()

    stop_words = {

        # common english
        "the", "and", "for", "with",
        "that", "this", "from", "have",
        "will", "your", "are", "our",
        "using", "use", "into", "such",
        "their", "they", "them",

        # recruiter filler words
        "work", "working", "role",
        "candidate", "candidates",
        "development", "systems",
        "testing", "software",
        "problem", "solving",
        "skills", "knowledge",
        "experience", "applications",
        "application", "team",
        "teams", "scalable",
        "solutions", "design",
        "developing", "maintaining",
        "engineer", "engineering",
        "responsibilities",
        "developer", "developers",
        "frontend", "backend",
        "ability", "strong",
        "understanding",
        "participate",
        "maintain",
        "modern",
        "tools",
        "technologies",
        "quality",
        "deliver"
    }

    keywords = []

    for word in words:

        if (
            len(word) > 2 and
            word not in stop_words and
            not word.isdigit()
        ):

            keywords.append(word)

    return list(set(keywords))


# ---------------- SCORING ---------------- #

def skill_score(mandatory_skills, resume):

    matched_skills = [

        s for s in mandatory_skills

        if s in resume
    ]

    score = (

        len(matched_skills) /
        len(mandatory_skills)

        if mandatory_skills else 0
    )

    return score, matched_skills


def keyword_overlap(job, resume):

    job_set = set(job.split())
    res_set = set(resume.split())

    return (

        len(job_set & res_set) /
        len(job_set)

        if job_set else 0
    )


# ---------------- ROUTES ---------------- #

@app.route("/")
def index():

    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match():

    job_desc = request.form.get(
        "job_description",
        ""
    )

    skills_input = request.form.get(
        "skills",
        ""
    )

    top_n = int(
        request.form.get("top_n", 10)
    )

    files = request.files.getlist(
        "resumes"
    )

    job_clean = clean(job_desc)

    # prevent crash
    if not job_clean:

        return render_template(
            "index.html",
            message="❌ Enter a valid job description"
        )

    # recruiter-entered skills
    recruiter_skills = [

        s.strip().lower()

        for s in skills_input.split(",")

        if s.strip()
    ]

    # dynamic technical keyword extraction
    jd_keywords = extract_keywords(job_clean)

    # merge both
    mandatory_skills = list(set(
        recruiter_skills + jd_keywords
    ))

    candidates = []

    with tempfile.TemporaryDirectory() as temp_dir:

        for f in files:

            if allowed_file(f.filename):

                filename = secure_filename(
                    f.filename
                )

                path = os.path.join(
                    temp_dir,
                    filename
                )

                f.save(path)

                raw = extract_text(path)

                email = extract_email(raw)

                resume_clean = clean(
                    anonymize(raw)
                )

                candidates.append({

                    "name":
                    filename.rsplit(".", 1)[0],

                    "email":
                    email,

                    "resume_text":
                    resume_clean
                })

        if not candidates:

            return render_template(
                "index.html",
                message="❌ No valid resumes uploaded"
            )

        # ---------------- TF-IDF ---------------- #

        corpus = [job_clean] + [

            c["resume_text"]

            for c in candidates
        ]

        tfidf = TfidfVectorizer(

            stop_words="english",

            ngram_range=(1, 2),

            min_df=1
        )

        vectors = tfidf.fit_transform(corpus)

        tfidf_scores = cosine_similarity(
            vectors[0:1],
            vectors[1:]
        ).flatten()

        results = []

        for i, c in enumerate(candidates):

            # ignore completely irrelevant resumes
            if tfidf_scores[i] < 0.02:
                continue

            # skill scoring
            rule, matched_skills = skill_score(
                mandatory_skills,
                c["resume_text"]
            )

            # overlap scoring
            overlap = keyword_overlap(
                job_clean,
                c["resume_text"]
            )

            # explainability
            missing_skills = [

                s for s in mandatory_skills

                if s not in c["resume_text"]
            ]

            # realistic ATS weighting
            final = (

                0.65 * rule +

                0.25 * tfidf_scores[i] +

                0.10 * overlap
            )

            # normalized score
            final_score = min(
                100,
                final * 100
            )

            results.append({

                "name":
                c["name"],

                "email":
                c["email"],

                "score":
                round(final_score, 1),

                "matched_skills":
                matched_skills,

                "missing_skills":
                missing_skills,

                "tfidf_score":
                round(
                    tfidf_scores[i] * 100,
                    1
                ),

                "skill_score":
                round(
                    rule * 100,
                    1
                ),

                "overlap_score":
                round(
                    overlap * 100,
                    1
                )
            })

    # ---------------- SORT + RANK ---------------- #

    results = sorted(

        results,

        key=lambda x: x["score"],

        reverse=True
    )

    for idx, r in enumerate(results):

        r["rank"] = idx + 1

    results = results[:top_n]

    return render_template(

        "index.html",

        message="✅ Resume ranking completed",

        results=results,

        selected_top_n=top_n
    )


@app.route("/download_pdf", methods=["POST"])
def download_pdf():

    shortlisted = request.form.getlist(
        "shortlisted"
    )

    pdf = canvas.Canvas(
        SHORTLIST_PDF,
        pagesize=A4
    )

    width, height = A4

    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        height - 50,
        "Shortlisted Candidates"
    )

    y = height - 100

    pdf.setFont(
        "Helvetica",
        11
    )

    for idx, entry in enumerate(
        shortlisted,
        start=1
    ):

        name, email = entry.split("||")

        pdf.drawString(
            50,
            y,
            f"{idx}. {name} | {email}"
        )

        y -= 20

        if y < 50:

            pdf.showPage()

            y = height - 50

    pdf.save()

    return send_file(
        SHORTLIST_PDF,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)
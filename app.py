from flask import Flask, request, render_template, send_file
import os
import re
import docx2txt
import PyPDF2
import tempfile

from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
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


# ---------------- SKILL EXTRACTION (JD + MANDATORY FIELD) ---------------- #

JD_STOPWORDS = ENGLISH_STOP_WORDS | {
    "year", "years", "experience", "work", "working", "team",
    "role", "position", "company", "ability", "strong", "good",
    "excellent", "preferred", "required", "requirements",
    "responsibilities", "qualification", "qualifications",
    "candidate", "candidates", "job", "description", "including",
    "software", "engineer", "developer", "backend", "hiring",
    "skills", "skill", "nice", "looking", "join", "etc",
}

JD_SKILL_PATTERNS = [
    r"(?:experience with|experienced in|proficient in|proficiency in|"
    r"knowledge of|skills in|skilled in|familiar with|expertise in)\s+"
    r"([a-z0-9+#./\s]{2,50})",
    r"(?:must have|should have|nice to have|required skills?)[:\s]+"
    r"([^\n.]{3,120})",
]


def is_valid_skill(term):

    if not term or len(term) < 2 or len(term) > 40:
        return False

    if term.isdigit() or term in JD_STOPWORDS:
        return False

    return True


def split_skill_phrase(phrase):

    phrase = clean(phrase)

    phrase = re.sub(
        r"^(required skills?|mandatory skills?|must have|"
        r"technical skills?|key skills?)[:\s]+",
        "",
        phrase,
    )

    parts = re.split(r"[,;/|•·]| and ", phrase)

    return [
        clean(part)
        for part in parts
        if is_valid_skill(clean(part))
    ]


def parse_recruiter_skills(skills_input):

    return [
        clean(skill)
        for skill in skills_input.split(",")
        if is_valid_skill(clean(skill))
    ]


def extract_skills_from_jd(jd_text):

    text = clean(jd_text)
    found = set()

    for pattern in JD_SKILL_PATTERNS:

        for match in re.finditer(pattern, text):

            found.update(split_skill_phrase(match.group(1)))

    for line in jd_text.splitlines():

        line = re.sub(r"^[\s•\-*\d.)]+", "", line.strip())

        if line and re.search(r"[,;]", line):

            found.update(split_skill_phrase(line))

    return sorted(found)


def build_mandatory_skills(jd_text, recruiter_skills):

    mandatory = list(dict.fromkeys(recruiter_skills))

    for skill in extract_skills_from_jd(jd_text):

        if skill in mandatory:
            continue

        if any(skill != kept and skill in kept for kept in mandatory):
            continue

        if any(
            kept in skill and kept != skill and len(kept) < len(skill)
            for kept in mandatory
        ):
            continue

        mandatory.append(skill)

    return sorted(mandatory)


def match_skills_in_resume(resume_text, mandatory_skills):

    if not mandatory_skills:
        return 0, [], []

    matched = []

    for skill in mandatory_skills:

        if re.search(rf"\b{re.escape(skill)}\b", resume_text):
            matched.append(skill)

    score = len(matched) / len(mandatory_skills)

    missing = [s for s in mandatory_skills if s not in matched]

    return score, matched, missing


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

    recruiter_skills = parse_recruiter_skills(skills_input)

    mandatory_skills = build_mandatory_skills(
        job_desc,
        recruiter_skills,
    )

    if not mandatory_skills:

        return render_template(
            "index.html",
            message=(
                "❌ No skills found. Enter mandatory skills and/or "
                "list requirements in the job description "
                "(e.g. Required skills: Python, SQL, Docker)."
            ),
        )

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

            ngram_range=(1, 1),

            min_df=1,
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

            rule, matched_skills, missing_skills = match_skills_in_resume(
                c["resume_text"],
                mandatory_skills,
            )

            final = 0.75 * rule + 0.25 * tfidf_scores[i]

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

                "jd_similarity":
                round(tfidf_scores[i] * 100, 1),

                "skill_match":
                round(rule * 100, 1),
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

    if results:
        generate_pdf(results)

    return render_template(

        "index.html",

        message="✅ Resume ranking completed",

        results=results,

        mandatory_skills=mandatory_skills,

        selected_top_n=top_n,
    )


def generate_pdf(results):

    pdf = canvas.Canvas(SHORTLIST_PDF, pagesize=A4)
    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "ATS Resume Ranking Report")
    y -= 40
    pdf.setFont("Helvetica", 11)

    for r in results:

        pdf.drawString(50, y, f"Rank #{r['rank']} — {r['name']} ({r['score']}%)")
        y -= 16
        pdf.drawString(70, y, f"Email: {r['email']}")
        y -= 16
        pdf.drawString(
            70, y,
            f"Matched: {', '.join(r['matched_skills']) or 'None'}",
        )
        y -= 16
        pdf.drawString(
            70, y,
            f"Missing: {', '.join(r['missing_skills']) or 'None'}",
        )
        y -= 28

        if y < 80:
            pdf.showPage()
            y = height - 50

    pdf.save()


@app.route("/download_pdf")
def download_pdf():

    if not os.path.exists(SHORTLIST_PDF):

        return render_template(
            "index.html",
            message="❌ Run matching first to generate the PDF.",
        )

    return send_file(SHORTLIST_PDF, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
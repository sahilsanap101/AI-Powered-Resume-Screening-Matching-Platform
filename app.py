from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Set upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'


# ---------- TEXT EXTRACTION FUNCTIONS ----------

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    return ""


# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def matcher():
    job_description = request.form.get("job_description")
    resume_files = request.files.getlist("resumes")

    if not job_description or not resume_files:
        return render_template(
            "index.html",
            message="Please provide a job description and at least one resume."
        )

    resumes_text = []

    for resume_file in resume_files:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(filename)
        resumes_text.append(extract_text(filename))

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_description] + resumes_text).toarray()

    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    similarity_scores = cosine_similarity([job_vector], resume_vectors)[0]

    # Top 5 matches
    top_indices = similarity_scores.argsort()[-5:][::-1]
    top_resumes = [resume_files[i].filename for i in top_indices]
    top_scores = [round(similarity_scores[i], 2) for i in top_indices]

    return render_template(
        "index.html",
        message="Top Matching Resumes:",
        top_resumes=top_resumes,
        similarity_scores=top_scores
    )


# ---------- RUN APP ----------

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)

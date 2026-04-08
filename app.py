import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
import PyPDF2
import re
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ------------------ MODEL ------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ SKILLS DATABASE ------------------
SKILLS_DB = [
    "python", "java", "sql", "machine learning", "deep learning",
    "data analysis", "nlp", "html", "css", "javascript",
    "react", "angular", "node", "tensorflow", "pandas",
    "numpy", "excel", "communication", "problem solving"
]

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

def extract_skills(text):
    found = []
    for skill in SKILLS_DB:
        if skill in text:
            found.append(skill)
    return set(found)

def generate_pdf(ranked, skills_report):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Resume Ranking Report", styles["Title"]))
    content.append(Spacer(1, 10))

    for i, (name, score, _) in enumerate(ranked, start=1):
        content.append(Paragraph(f"{i}. {name} - {score*100:.2f}%", styles["Normal"]))

        skills = skills_report[name]
        content.append(Paragraph(f"Matched Skills: {', '.join(skills['matched'])}", styles["Normal"]))
        content.append(Paragraph(f"Missing Skills: {', '.join(skills['missing'])}", styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)

# ------------------ UI ------------------

st.set_page_config(page_title="AI Resume Ranker", layout="wide")

st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background: #1f2937;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>💼 AI Resume Analyzer</h1>", unsafe_allow_html=True)

# ------------------ INPUT ------------------

jd_text = st.text_area("📝 Paste Job Description", height=150)
uploaded_files = st.file_uploader("📂 Upload Resumes", type=["pdf"], accept_multiple_files=True)

# ------------------ PROCESS ------------------

if st.button("🚀 Analyze"):
    if jd_text and uploaded_files:

        jd_clean = clean_text(jd_text)
        jd_skills = extract_skills(jd_clean)

        resumes = []
        names = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(clean_text(text))
            names.append(file.name)

        embeddings = model.encode([jd_clean] + resumes)
        similarity = cosine_similarity([embeddings[0]], embeddings[1:])

        scores = list(zip(names, similarity[0], resumes))
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        st.subheader("📊 Ranking Results")

        skills_report = {}

        for i, (name, score, text) in enumerate(ranked, start=1):

            percent = score * 100
            res_skills = extract_skills(text)

            matched = jd_skills.intersection(res_skills)
            missing = jd_skills - res_skills

            skills_report[name] = {
                "matched": list(matched),
                "missing": list(missing)
            }

            # ---------- CARD ----------
            st.markdown(f"<div class='card'><h3>{i}. {name}</h3></div>", unsafe_allow_html=True)

            # Progress bar
            st.progress(int(percent))

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"Matched Skills: {', '.join(matched) if matched else 'None'}")

            with col2:
                st.error(f"Missing Skills: {', '.join(missing) if missing else 'None'}")

        # ------------------ GRAPH ------------------

        st.subheader("📈 Comparison Chart")

        names_list = [x[0] for x in ranked]
        scores_list = [x[1]*100 for x in ranked]

        fig, ax = plt.subplots()
        ax.barh(names_list, scores_list)
        st.pyplot(fig)

        # ------------------ PDF ------------------

        generate_pdf(ranked, skills_report)

        with open("report.pdf", "rb") as f:
            st.download_button("📥 Download Full Report", f, "Resume_Report.pdf")

    else:
        st.warning("⚠️ Upload resumes and enter job description")
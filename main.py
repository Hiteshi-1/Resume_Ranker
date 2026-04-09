import os
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Load resumes
resume_folder = "resumes"
resumes = []
resume_names = []

for file in os.listdir(resume_folder):
    if file.endswith(".pdf"):
        path = os.path.join(resume_folder, file)
        resumes.append(extract_text_from_pdf(path))
        resume_names.append(file)

# Load job description
with open("jd.txt", "r", encoding="utf-8") as f:
    jd = f.read()

# Combine JD + resumes
documents = [jd] + resumes

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Similarity
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

# Ranking
scores = list(zip(resume_names, similarity_scores[0]))
ranked = sorted(scores, key=lambda x: x[1], reverse=True)

print("\n📊 Resume Ranking:\n")
for i, (name, score) in enumerate(ranked, start=1):
    print(f"{i}. {name} - Score: {score:.4f}")
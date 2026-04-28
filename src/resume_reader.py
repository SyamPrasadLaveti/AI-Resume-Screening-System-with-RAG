import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text().lower() + "\n"
    return text

model = SentenceTransformer('all-MiniLM-L6-v2')

job_desc = """We are seeking a motivated Associate Software Engineer to join our development team.
In this role, you will work alongside senior engineers to design, build, and maintain scalable software applications using Sap abap,python and c++.
This is an ideal position for early-career professionals looking to apply their multi-language skills
(Python, Java, C) in a dynamic, containerized environment.""".lower()

resume_text = extract_text_from_pdf("data/sample_resume.pdf")
resume_embedding = model.encode(resume_text)
jd_embedding = model.encode(job_desc)

similarity_score=cosine_similarity([resume_embedding],[jd_embedding])[0][0]

if similarity_score >= 0.7:
    decision = "Strong Match"
elif similarity_score >=0.5:
    decision = "Moderate Match"
else:
    decision = "Weak Match"


print("The Resume Score")
print(f"Similarity score: {similarity_score:.4f}")
print("Decision",decision)


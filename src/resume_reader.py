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

def text_to_chunk(text,chunk_size=200):
    words=text.split()
    for i in range(0,len(words),chunk_size):
        chunks=[]
        chunk = " ".join(words[i:i+ chunk_size])
        chunks.append(chunk)
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')

job_desc = """We are seeking a motivated Associate Software Engineer to join our development team.
In this role, you will work alongside senior engineers to design, build, and maintain scalable software applications using Sap abap,python and c++.
This is an ideal position for early-career professionals looking to apply their multi-language skills
(Python, Java, C) in a dynamic, containerized environment.""".lower()

resume_text = extract_text_from_pdf("data/sample_resume.pdf")
chunks=text_to_chunk(resume_text)
jd_embedding = model.encode(job_desc)

best_score=0

for chunk in chunks:
    chunk_score = cosine_similarity([model.encode(chunk)],[jd_embedding])[0][0]
    if chunk_score > best_score:
        best_score = chunk_score

if best_score >= 0.7:
    decision = "Strong Match"
elif best_score >=0.5:
    decision = "Moderate Match"
else:
    decision = "Weak Match"


print("The Resume Score")
print(f"Similarity score: {best_score:.4f}")


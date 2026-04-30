# ==============================
# IMPORTS
# ==============================
import pdfplumber
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# FUNCTION 1: Extract text from PDF
# ==============================
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text().lower() + "\n"
    return text


# ==============================
# FUNCTION 2: Split text into chunks
# ==============================
def text_to_chunk(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ==============================
# FUNCTION 3: Load all resume files
# ==============================
def load_all_resumes(folder_path):
    resume_files = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            resume_files.append(full_path)

    return resume_files


# ==============================
# LOAD MODEL
# ==============================
model = SentenceTransformer('all-MiniLM-L6-v2')


# ==============================
# JOB DESCRIPTION
# ==============================
job_desc = """We are seeking a motivated Associate Software Engineer to join our development team.
In this role, you will work alongside senior engineers to design, build, and maintain scalable software applications using Sap abap, python and c++.
This is an ideal position for early-career professionals looking to apply their multi-language skills
(Python, Java, C) in a dynamic, containerized environment.""".lower()


# ==============================
# STEP 1: Load resume file paths
# ==============================
resume_files = load_all_resumes("data")

# DEBUG: Check if files are loaded
print("Loaded files:", resume_files)


# ==============================
# STEP 2: Extract text from resumes
# ==============================
all_resumes_text = []

for file_path in resume_files:
    text = extract_text_from_pdf(file_path)
    all_resumes_text.append((file_path, text))


# ==============================
# STEP 3: Encode job description
# ==============================
jd_embedding = model.encode(job_desc)


# ==============================
# STEP 4: Compute similarity
# ==============================
results = []

for file_path, resume_text in all_resumes_text:

    chunks = text_to_chunk(resume_text)
    best_score = 0

    for chunk in chunks:
        chunk_embedding = model.encode(chunk)
        score = cosine_similarity([chunk_embedding], [jd_embedding])[0][0]

        if score > best_score:
            best_score = score

    results.append((file_path, best_score))


# ==============================
# STEP 5: Sort results
# ==============================
results.sort(key=lambda x: x[1], reverse=True)


# ==============================
# STEP 6: Display output
# ==============================
print("\n===== TOP CANDIDATES =====")

for i, (file, score) in enumerate(results, start=1):
    name = os.path.basename(file)  # CLEAN METHOD
    print(f"{i}. {name} → {score:.4f}")
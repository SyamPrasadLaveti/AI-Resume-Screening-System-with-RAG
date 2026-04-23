import pdfplumber

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

resume_text = extract_text_from_pdf("../data/sample_resume.pdf")

print("===== RESUME TEXT =====")
print(resume_text)

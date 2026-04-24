import pdfplumber

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text().lower() + "\n"
    return text


skills = ["python", "sap abap", "java", "docker"]
job_desc = """We are seeking a motivated Associate Software Engineer to join our development team.
In this role, you will work alongside senior engineers to design, build, and maintain scalable software applications.
This is an ideal position for early-career professionals looking to apply their multi-language skills
(Python, Java, C) in a dynamic, containerized environment.""".lower()


def score_of_resume(skills, resume_text, job_desc):
    matched = []
    missing = []
    extra = []

    for skill in skills:
        if skill in job_desc and skill in resume_text:
            matched.append(skill)
        elif skill in job_desc:
            missing.append(skill)
        elif skill in resume_text:
            extra.append(skill)

    score = (len(matched) / len(skills)) * 100

    return score, matched, missing, extra


resume_text = extract_text_from_pdf("../data/sample_resume.pdf")

score, matched_skills, missing_skills, extra_skills = score_of_resume(skills, resume_text, job_desc)
    

print("The Resume Score")
print(f"Score: {score:.2f}%")
print("Matched Skills:", matched_skills)
print("Missing Skills:", missing_skills)
print("Extra Skills:", extra_skills)

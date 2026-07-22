import json
from pathlib import Path

import numpy as np
import streamlit as st
import PyPDF2

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "resume_chatbot.keras"
LABELS_PATH = BASE_DIR / "labels.json"
INTENTS_PATH = BASE_DIR / "intents.json"

st.set_page_config(page_title="ResumeBot AI", page_icon="💼", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main Background */
.stApp {
    background: radial-gradient(circle at 50% 50%, #0F172A 0%, #020617 100%) !important;
    color: #F8FAFC !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.85) !important;
    backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #E2E8F0 !important;
}

/* Font override (excluding icons) */
html, body, .stApp, p, h1, h2, h3, h4, h5, h6, input, button, textarea {
    font-family: 'Outfit', sans-serif !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.3);
}
::-webkit-scrollbar-thumb {
    background: rgba(129, 140, 248, 0.25);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(129, 140, 248, 0.45);
}

/* Main Title styling */
.title-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.3) 0%, rgba(15, 23, 42, 0.3) 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 28px;
    text-align: center;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

.main-title {
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
    background: linear-gradient(135deg, #A5B4FC 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-shadow: 0 0 40px rgba(129, 140, 248, 0.25);
}

.main-subtitle {
    font-size: 1.05rem !important;
    color: #94A3B8 !important;
    font-weight: 400 !important;
    margin: 0 !important;
}

/* Streamlit Bottom Container fix (removes the massive white block) */
div[data-testid="stBottom"], div[data-testid="stBottom"] > div {
    background: transparent !important;
}

/* Chat Input field styling */
div[data-testid="stChatInput"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(129, 140, 248, 0.25) !important;
    border-radius: 28px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    padding: 6px 12px !important;
    transition: all 0.3s ease;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: rgba(168, 85, 247, 0.5) !important;
    box-shadow: 0 0 25px rgba(168, 85, 247, 0.15) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #F8FAFC !important;
    font-size: 1rem !important;
    background-color: transparent !important;
    caret-color: #818CF8 !important;
}

/* Beautiful PDF Uploader Styling */
div[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(129,140,248,0.12), rgba(168,85,247,0.12)) !important;
    border: 2px dashed rgba(129, 140, 248, 0.7) !important;
    border-radius: 18px !important;
    padding: 20px !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.3s ease !important;
    margin-bottom: 20px !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #A5B4FC !important;
    background: linear-gradient(135deg, rgba(129,140,248,0.2), rgba(168,85,247,0.2)) !important;
    box-shadow: 0 0 30px rgba(129,140,248,0.25) !important;
}
div[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}
/* Force ALL text inside uploader to be bright white */
div[data-testid="stFileUploader"] *,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] div,
div[data-testid="stFileUploader"] small {
    color: #E2E8F0 !important;
}
/* Browse Files button */
div[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}
div[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.6) !important;
    transform: translateY(-1px) !important;
}

/* Uploaded file chip / badge */
div[data-testid="stFileUploaderFile"],
div[data-testid="stFileUploaderDeleteBtn"],
div[data-testid="uploadedFile"],
[data-testid="stFileUploaderFile"] {
    background: rgba(99, 102, 241, 0.2) !important;
    border: 1px solid rgba(129, 140, 248, 0.5) !important;
    border-radius: 12px !important;
    padding: 8px 14px !important;
}
/* filename and file size text */
div[data-testid="stFileUploader"] [data-testid="stText"],
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {
    color: #C7D2FE !important;
    font-weight: 500 !important;
}
/* Delete (X) button */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button,
div[data-testid="stFileUploader"] button[kind="secondary"] {
    background: rgba(239,68,68,0.15) !important;
    border: 1px solid rgba(239,68,68,0.4) !important;
    color: #FCA5A5 !important;
    border-radius: 50% !important;
}
div[data-testid="stFileUploader"] button[kind="secondary"]:hover {
    background: rgba(239,68,68,0.35) !important;
    color: #FFFFFF !important;
}

div[data-testid="stChatInput"] button {
    background-color: rgba(129, 140, 248, 0.15) !important;
    color: #818CF8 !important;
    border-radius: 50% !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stChatInput"] button:hover {
    background-color: #818CF8 !important;
    color: #FFFFFF !important;
}

/* Custom Message Bubble Styling */
div[data-testid="stChatMessage"] {
    background: rgba(30, 41, 59, 0.45) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 20px !important;
    padding: 18px 20px !important;
    margin-bottom: 16px !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.15) !important;
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Chat bubble content text color fixes */
div[data-testid="stChatMessage"] p, 
div[data-testid="stChatMessage"] span, 
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] div {
    color: #F1F5F9 !important;
}

/* User Message custom background gradient */
div[data-testid="stChatMessage"][class*="user"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%) !important;
    border: 1px solid rgba(168, 85, 247, 0.25) !important;
    box-shadow: 0 4px 25px rgba(168, 85, 247, 0.08) !important;
}

div[data-testid="stChatMessage"][class*="user"] p, 
div[data-testid="stChatMessage"][class*="user"] span, 
div[data-testid="stChatMessage"][class*="user"] li,
div[data-testid="stChatMessage"][class*="user"] div {
    color: #FFFFFF !important;
}

/* Chat Message avatars */
div[data-testid="stChatMessage"] .stAvatar {
    background-color: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

/* Glass Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 18px;
    padding: 22px;
    margin: 16px 0;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(129, 140, 248, 0.2);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25);
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.status-badge.warning {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.25);
}

/* Tech Badges / Skills list in sidebar */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.tech-badge {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.8rem;
    color: #CBD5E1;
    font-weight: 500;
    transition: all 0.2s ease;
}

.tech-badge:hover {
    background: rgba(129, 140, 248, 0.1);
    border-color: rgba(129, 140, 248, 0.3);
    color: #E0E7FF;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Hide default streamlit elements to make it feel custom */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Responsive media queries */
@media (max-width: 768px) {
    .main-title {
        font-size: 1.8rem !important;
        line-height: 1.2 !important;
    }
    .title-container {
        padding: 18px 16px !important;
        margin-bottom: 20px !important;
        border-radius: 16px !important;
    }
    div[data-testid="stChatMessage"] {
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
        border-radius: 16px !important;
    }
    .glass-card {
        padding: 16px !important;
        margin: 12px 0 !important;
    }
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load the Keras NLP model and its intent labels after training."""
    if not (MODEL_PATH.exists() and LABELS_PATH.exists()):
        return None, []
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels


@st.cache_data
def load_intents():
    return json.loads(INTENTS_PATH.read_text(encoding="utf-8"))["intents"]


def response_for(tag: str, intents: list[dict]) -> str:
    for intent in intents:
        if intent["tag"] == tag:
            return intent["responses"][0]
    return "I am sorry, I could not understand that. Please ask another résumé-related question."


def predict(message: str, model, labels: list[str]) -> tuple[str, float]:
    import tensorflow as tf
    probabilities = model.predict(tf.constant([[message]], dtype=tf.string), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return labels[index], float(probabilities[index])


# Render Custom Title Card
st.markdown(
    '''
    <div class="title-container">
        <h1 class="main-title">💼 ResumeBot AI</h1>
        <p class="main-subtitle">A next gen resume analyzer to make future bright and shine</p>
    </div>
    ''',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        '''
        <div class="glass-card">
            <h3 style="margin-top:0; color:#818CF8; font-weight:600; font-size:1.15rem; display:flex; align-items:center; gap:8px;">
                💼 About ResumeBot
            </h3>
            <p style="font-size:0.9rem; line-height:1.5; color:#94A3B8; margin-bottom:12px; margin-top:8px;">
                This chatbot is a next gen resume analyzer designed to help you land your dream job and make your future bright and shine!
            </p>
            <div style="font-size:0.8rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:12px; color:#64748B;">
                <strong>Responsible Use:</strong> A human reviewer must always make final hiring decisions fairly.
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '''
        <div class="glass-card">
            <h3 style="margin-top:0; color:#C084FC; font-weight:600; font-size:1.15rem; display:flex; align-items:center; gap:8px;">
                🧠 Supported Topics
            </h3>
            <p style="font-size:0.9rem; color:#94A3B8; margin-bottom:8px; margin-top:8px;">Ask ResumeBot about:</p>
            <div class="badge-container">
                <span class="tech-badge">💡 Core Skills</span>
                <span class="tech-badge">🎓 Education</span>
                <span class="tech-badge">💼 Work Experience</span>
                <span class="tech-badge">✍️ Resume Improvements</span>
                <span class="tech-badge">🤝 Interview Prep</span>
                <span class="tech-badge">👋 Greetings / General</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(129,140,248,0.15) 0%, rgba(168,85,247,0.15) 100%);
    border: 2px solid rgba(129,140,248,0.6);
    border-radius: 24px;
    padding: 32px 36px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 0 40px rgba(129,140,248,0.2), 0 8px 32px rgba(0,0,0,0.3);
">
    <div style="font-size:3.5rem; margin-bottom:12px;">📄</div>
    <h2 style="color:#A5B4FC; font-size:1.8rem; font-weight:700; margin:0 0 8px 0;">
        Upload Your Resume
    </h2>
    <p style="color:#CBD5E1; font-size:1.05rem; margin:0 0 20px 0;">
        Drop your PDF CV below — I'll tell you what <strong style="color:#C084FC;">jobs you fit</strong>,
        what <strong style="color:#818CF8;">changes to make</strong>, and give you a
        <strong style="color:#A5B4FC;">resume score</strong>!
    </p>
</div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📎 Click to Browse or Drag & Drop your PDF Resume here", type="pdf", label_visibility="visible")

if uploaded_file is not None:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            text_lower = text.lower()
            word_count = len(text.split())

            # ── STEP 0: Validate it is actually a Resume / CV ──────────────────
            resume_signals = [
                "resume", "curriculum vitae", "cv", "experience", "education",
                "skills", "objective", "summary", "work history", "employment",
                "references", "projects", "certifications", "achievements",
                "internship", "bachelor", "master", "degree", "university",
                "college", "gpa", "job title", "profile"
            ]
            signal_hits = sum(1 for s in resume_signals if s in text_lower)

            if signal_hits < 3 or word_count < 80:
                # Not a resume — reject it
                reject_msg = (
                    "⛔ **This document does not appear to be a Resume or CV.**\n\n"
                    "I detected very few resume-related sections in the uploaded PDF. "
                    "Please upload a proper **Resume or Curriculum Vitae (CV)** containing "
                    "sections like *Education*, *Experience*, *Skills*, etc.\n\n"
                    f"*(Detected {signal_hits} resume signals — at least 3 required to analyze.)*"
                )
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.messages.append({"role": "assistant", "content": reject_msg})
                st.rerun()
            else:
                # ── STEP 1: Suggested Jobs ──────────────────────────────────────
                jobs = []
                if any(k in text_lower for k in ["python", "java", "c++", "javascript", "html", "css", "software", "developer", "coding", "programming", "react", "django", "flask", "node"]):
                    jobs.append("Software Engineer / Full Stack Developer")
                if any(k in text_lower for k in ["data", "sql", "excel", "tableau", "power bi", "machine learning", "analytics", "statistics", "pandas", "numpy", "ai", "deep learning"]):
                    jobs.append("Data Analyst / Data Scientist")
                if any(k in text_lower for k in ["sales", "marketing", "seo", "campaign", "branding", "customer", "crm", "social media", "content", "digital"]):
                    jobs.append("Sales & Marketing Specialist")
                if any(k in text_lower for k in ["design", "ui", "ux", "figma", "adobe", "photoshop", "illustrator", "wireframe", "prototype", "creative"]):
                    jobs.append("UI/UX Designer / Graphic Designer")
                if any(k in text_lower for k in ["manage", "agile", "scrum", "product", "lead", "team", "strategy", "planning", "stakeholder", "roadmap"]):
                    jobs.append("Product / Project Manager")
                if any(k in text_lower for k in ["account", "finance", "budget", "audit", "tax", "bookkeeping", "ledger", "tally", "gst", "balance sheet"]):
                    jobs.append("Finance / Accounting Professional")
                if any(k in text_lower for k in ["network", "cyber", "security", "linux", "server", "cloud", "aws", "azure", "devops", "docker", "kubernetes"]):
                    jobs.append("IT / Cloud / DevOps Engineer")
                if any(k in text_lower for k in ["teach", "tutor", "curriculum", "classroom", "lecture", "training", "educator", "instructor"]):
                    jobs.append("Teacher / Corporate Trainer")
                if not jobs:
                    jobs.append("General Professional — Add industry-specific keywords to get better job matches!")

                # ── STEP 2: Real Point-Based Scoring ───────────────────────────
                score = 0
                score_breakdown = []

                # Contact info (10 pts)
                contact_keywords = ["email", "phone", "linkedin", "github", "address", "mobile", "@"]
                contact_found = sum(1 for k in contact_keywords if k in text_lower)
                contact_score = min(10, contact_found * 3)
                score += contact_score
                score_breakdown.append(f"Contact Info: **{contact_score}/10**")

                # Professional Summary (10 pts)
                if any(k in text_lower for k in ["summary", "objective", "profile", "about me"]):
                    score += 10
                    score_breakdown.append("Professional Summary: **10/10** ✅")
                else:
                    score_breakdown.append("Professional Summary: **0/10** ❌ Missing")

                # Education (15 pts)
                if any(k in text_lower for k in ["education", "degree", "bachelor", "master", "university", "college", "b.tech", "b.sc", "m.tech", "mba", "gpa", "12th", "10th"]):
                    score += 15
                    score_breakdown.append("Education Section: **15/15** ✅")
                else:
                    score_breakdown.append("Education Section: **0/15** ❌ Missing")

                # Work Experience (20 pts)
                exp_keywords = ["experience", "work history", "employment", "internship", "worked at", "worked as", "position", "role", "responsibilities"]
                if any(k in text_lower for k in exp_keywords):
                    score += 20
                    score_breakdown.append("Work Experience: **20/20** ✅")
                else:
                    score_breakdown.append("Work Experience: **0/20** ❌ Missing")

                # Skills (15 pts)
                if any(k in text_lower for k in ["skills", "technical skills", "core competencies", "expertise", "proficient"]):
                    score += 15
                    score_breakdown.append("Skills Section: **15/15** ✅")
                else:
                    score_breakdown.append("Skills Section: **0/15** ❌ Missing")

                # Projects / Certifications (10 pts)
                if any(k in text_lower for k in ["project", "portfolio", "certification", "certified", "course", "award", "achievement"]):
                    score += 10
                    score_breakdown.append("Projects / Certifications: **10/10** ✅")
                else:
                    score_breakdown.append("Projects / Certifications: **0/10** ❌ Missing")

                # Quantified Achievements (10 pts)
                if any(k in text_lower for k in ["%", "increased", "decreased", "reduced", "led", "managed", "improved", "achieved", "grew", "delivered"]):
                    score += 10
                    score_breakdown.append("Quantified Achievements: **10/10** ✅")
                else:
                    score_breakdown.append("Quantified Achievements: **0/10** ❌ Add numbers & impact!")

                # Word count / Length (10 pts)
                if word_count >= 400:
                    score += 10
                    score_breakdown.append(f"Resume Length ({word_count} words): **10/10** ✅")
                elif word_count >= 200:
                    score += 6
                    score_breakdown.append(f"Resume Length ({word_count} words): **6/10** — Could be more detailed")
                else:
                    score += 2
                    score_breakdown.append(f"Resume Length ({word_count} words): **2/10** ❌ Too short, add more detail")

                # ── STEP 3: Recommended Changes ────────────────────────────────
                changes = []
                if not any(k in text_lower for k in ["summary", "objective", "profile"]):
                    changes.append("Add a **Professional Summary** at the top of your resume.")
                if not any(k in text_lower for k in ["education", "degree", "university", "college"]):
                    changes.append("Add an **Education** section with degree, college, and year.")
                if not any(k in text_lower for k in ["experience", "internship", "employment"]):
                    changes.append("Add a **Work Experience** section with job titles and responsibilities.")
                if not any(k in text_lower for k in ["skills", "technical skills"]):
                    changes.append("Add a **Skills** section with relevant technical and soft skills.")
                if not any(k in text_lower for k in ["project", "certification"]):
                    changes.append("Include **Projects** or **Certifications** to stand out from other candidates.")
                if not any(k in text_lower for k in ["%", "increased", "reduced", "led", "improved"]):
                    changes.append("**Quantify your achievements** — e.g. 'Increased sales by 30%' or 'Led a team of 5'.")
                if word_count < 300:
                    changes.append(f"Your resume is only **{word_count} words** — expand with more details about your roles and skills.")
                if not any(k in text_lower for k in ["linkedin", "github", "portfolio"]):
                    changes.append("Add your **LinkedIn** or **GitHub** profile link to boost credibility.")
                if not changes:
                    changes.append("✅ Your resume structure looks strong! Keep tailoring it for each specific job application.")

                # Score label
                if score >= 85:
                    grade, color = "Excellent 🏆", "#4ADE80"
                elif score >= 70:
                    grade, color = "Good 👍", "#86EFAC"
                elif score >= 50:
                    grade, color = "Average ⚠️", "#FCD34D"
                else:
                    grade, color = "Needs Work 🔧", "#F87171"

                # ── Build final message ─────────────────────────────────────────
                analysis_msg = f"**📄 Next Gen Resume Analysis Report — `{uploaded_file.name}`**\n\n"

                analysis_msg += "### 🎯 Suggested Job Roles\nBased on your CV keywords, you are a strong fit for:\n"
                for job in jobs[:3]:
                    analysis_msg += f"- {job}\n"

                analysis_msg += "\n### 🛠️ Recommended Changes\n"
                for change in changes:
                    analysis_msg += f"- {change}\n"

                analysis_msg += f"\n### 📊 Real Resume Score: **{score}/100** — {grade}\n"
                analysis_msg += "**Score Breakdown:**\n"
                for item in score_breakdown:
                    analysis_msg += f"- {item}\n"

                st.session_state.last_uploaded = uploaded_file.name
                st.session_state.messages.append({"role": "assistant", "content": analysis_msg})
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")


intents = load_intents()
model, labels = load_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ResumeBot AI. Ask me about résumé skills, interview preparation, experience, or improving a résumé."}
    ]

if model is None:
    st.markdown(
        '''
        <div class="glass-card" style="border-color: rgba(239, 68, 68, 0.3); margin-bottom: 24px;">
            <div class="status-badge warning">⚠️ Attention Required</div>
            <p style="margin-top:12px; margin-bottom:0; font-size:0.95rem; color:#FDA4AF;">
                The AI model has not been trained yet. Please run <code>python train_chatbot.py</code> in the terminal to initialize the AI.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '''
        <div class="glass-card" style="border-color: rgba(16, 185, 129, 0.2); padding: 15px 20px; margin-bottom: 24px; margin-top: 0px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <span style="font-weight:600; color:#F1F5F9; font-size:0.95rem; display:flex; align-items:center; gap:8px;">
                    🤖 AI Engine Status
                </span>
                <span class="status-badge">⚡ Next Gen Resume Analyzer Active</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

question = st.chat_input("Ask a résumé screening question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        if model is None:
            answer = "Please train the AI model first by running `python train_chatbot.py` in the project folder."
        else:
            tag, confidence = predict(question, model, labels)
            if confidence < 0.45:
                # Simulated GenAI Fallback for better UX
                answer = f"That's a great question about '{question}'. While my main neural network is focused on classifying core resume intents, here is my general AI advice: **1.** Always tailor your resume to the specific job description by including their exact keywords. **2.** Quantify your results (e.g., 'increased sales by 20%'). **3.** For specific roles like this, focus heavily on your portfolio and measurable impact."
            else:
                answer = response_for(tag, intents)
                answer += f"\n\n---\n<span style='font-size:0.8rem; color:#64748B;'>🔮 Classified Intent: <code style='color:#A5B4FC; background:rgba(129,140,248,0.1); padding:2px 6px; border-radius:4px;'>{tag}</code> &nbsp;·&nbsp; Confidence: <code style='color:#C084FC; background:rgba(168,85,247,0.1); padding:2px 6px; border-radius:4px;'>{confidence:.0%}</code></span>"
        st.markdown(answer, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})

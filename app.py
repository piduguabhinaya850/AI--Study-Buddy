import os
import io
import json
import time
import math
import re
import tempfile
from datetime import date, datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as gtypes
except Exception:
    genai = None
    gtypes = None

try:
    import pandas as pd
except Exception:
    pd = None

load_dotenv()

# ================================================================
# AI STUDY BUDDY 360
# Unified AI Learning + Career + Work Intelligence Platform
# ================================================================

st.set_page_config(
    page_title="AI Study Buddy 360",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "3.0.0"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

# --------------------------- State ------------------------------

DEFAULTS = {
    "messages": [],
    "points": 0,
    "topics": 0,
    "quizzes": 0,
    "flashcards": 0,
    "study_minutes": 0,
    "streak": 1,
    "last_topic": "",
    "last_result": "",
    "last_provider": "",
    "history": [],
    "documents": [],
    "skills": {},
    "projects": [],
    "tasks": [],
    "gemini_stats": {"calls": 0, "seconds": 0.0},
    "interview_score": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value

# --------------------------- Styling ----------------------------

st.markdown(
    """
<style>
:root {
    --ink:#0f172a;
    --muted:#64748b;
    --line:#e2e8f0;
    --primary:#4f46e5;
    --secondary:#0284c7;
}
.stApp {
    background:
      radial-gradient(circle at 0% 0%, rgba(79,70,229,.10), transparent 30%),
      radial-gradient(circle at 100% 100%, rgba(2,132,199,.09), transparent 30%),
      #f8fafc;
}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0b1220,#172554 65%,#0f172a);
}
[data-testid="stSidebar"] * { color:#e2e8f0; }
.hero {
    padding:34px;
    border-radius:28px;
    color:white;
    background:linear-gradient(135deg,#0f172a,#312e81,#0369a1);
    box-shadow:0 20px 50px rgba(15,23,42,.18);
    margin-bottom:24px;
}
.hero h1 { margin:0; font-size:44px; letter-spacing:-1px; }
.hero p { color:#dbeafe; font-size:17px; margin:8px 0 0; }
.badge {
    display:inline-block; margin-top:15px; padding:7px 13px;
    border-radius:999px; background:rgba(255,255,255,.14);
    font-size:12px; font-weight:800;
}
.card {
    background:rgba(255,255,255,.96);
    border:1px solid var(--line); border-radius:20px;
    padding:20px; box-shadow:0 8px 25px rgba(15,23,42,.06);
    margin-bottom:16px;
}
.feature {
    background:white; border:1px solid var(--line);
    border-radius:20px; padding:20px; min-height:150px;
    box-shadow:0 8px 25px rgba(15,23,42,.05);
}
.feature-icon { font-size:30px; }
.metric {
    background:white; border:1px solid var(--line);
    border-radius:18px; padding:18px; text-align:center;
}
.metric-number { font-size:30px; font-weight:800; color:#312e81; }
.small { color:#64748b; font-size:13px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------- Gemini AI Engine --------------------------

def env(name):
    return os.getenv(name, "").strip()

@st.cache_resource(show_spinner=False)
def get_gemini_client(key):
    if not key or not genai:
        return None
    return genai.Client(api_key=key)

def get_gemini_key():
    key = env("GEMINI_API_KEY")
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""

def require_gemini():
    key = get_gemini_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets "
            "or your local .env file."
        )
    if not genai or not gtypes:
        raise RuntimeError(
            "Google Gemini SDK is not installed. Run: pip install google-genai"
        )
    client = get_gemini_client(key)
    if client is None:
        raise RuntimeError("Unable to initialize the Google Gemini client.")
    return client

def call_gemini(prompt, system="", temperature=0.6, max_tokens=5000):
    client = require_gemini()
    config = gtypes.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system or None,
    )
    started = time.perf_counter()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    elapsed = round(time.perf_counter() - started, 2)
    result = getattr(response, "text", None)
    if not result:
        raise RuntimeError("Gemini returned an empty response.")
    stats = st.session_state.gemini_stats
    stats["calls"] += 1
    stats["seconds"] += elapsed
    st.session_state.last_provider = "Google Gemini"
    return result

def ask_ai(prompt, system="", temperature=0.6, max_tokens=5000):
    return call_gemini(prompt, system, temperature, max_tokens)

def ask_gemini(prompt, system="", temperature=0.6, max_tokens=5000):
    return call_gemini(prompt, system, temperature, max_tokens)

# --------------------------- Helpers -----------------------------

def save_result(result):
    st.session_state.last_result = result

def download_result(filename="AI_Result.md"):
    if st.session_state.last_result:
        st.download_button(
            "📥 Download result",
            st.session_state.last_result,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )

def award(points=5, topic=None):
    st.session_state.points += points
    if topic:
        st.session_state.last_topic = topic

def show_error(e):
    msg = str(e)
    low = msg.lower()
    if "401" in msg or "api key" in low or "api_key" in low or "authentication" in low:
        st.error("🔐 Gemini authentication failed. Check GEMINI_API_KEY in Streamlit Secrets.")
    elif "429" in msg or "quota" in low or "resource exhausted" in low:
        st.error("⚡ Gemini quota/rate limit reached. Check your Gemini usage/quota and try again.")
    elif "503" in msg or "unavailable" in low:
        st.error("🟠 Gemini is temporarily busy/unavailable. Please wait a moment and try again.")
    else:
        st.error("Something went wrong: " + msg)

def run_ai_button(label, prompt, system="", temperature=0.5,
                  max_tokens=6000, points=10, topic=None, filename="AI_Result.md"):
    if st.button(label, type="primary", use_container_width=True):
        with st.spinner("AI is working..."):
            try:
                result = ask_ai(prompt, system, temperature, max_tokens)
                st.markdown(result)
                save_result(result)
                download_result(filename)
                award(points, topic)
            except Exception as e:
                show_error(e)

def parse_json(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None

def profile_context():
    skills = ", ".join(
        f"{k}: {v}" for k, v in st.session_state.skills.items()
    ) or "Not recorded"
    return (
        f"Known skills: {skills}\n"
        f"Last topic: {st.session_state.last_topic or 'None'}\n"
        f"Completed topics: {st.session_state.topics}\n"
        f"Quizzes: {st.session_state.quizzes}\n"
        f"Flashcards: {st.session_state.flashcards}\n"
    )

# ------------------------- File analysis -------------------------

def analyze_uploaded_file(uploaded, instruction):
    client = require_gemini()
    suffix = os.path.splitext(uploaded.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin") as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name

    try:
        remote = client.files.upload(file=path)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[instruction, remote],
            config=gtypes.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=8000,
            ),
        )
        result = getattr(response, "text", None)
        if not result:
            raise RuntimeError("Gemini returned an empty response.")
        return result
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

# ---------------------------- Sidebar ---------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:10px 4px 18px">
            <div style="font-size:50px">🧠</div>
            <h2 style="margin:0">AI Study Buddy 360</h2>
            <p style="color:#94a3b8">AI Learning • Career • Work OS</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🤖 AI Engine")
    st.success("🟢 Google Gemini")
    st.caption(f"Model: `{GEMINI_MODEL}`")
    st.caption("One secure API key powers the entire platform.")

    page = st.radio(
        "Workspace",
        [
            "🏠 Command Center",
            "💬 AI Chat",
            "🧠 AI Model Lab",
            "📚 Learn",
            "📝 Smart Notes",
            "🧠 Quiz Arena",
            "🃏 Flashcards",
            "📅 Study Planner",
            "🎯 Exam Mode",
            "💻 Developer Lab",
            "📊 Data & Analytics",
            "📄 Document Intelligence",
            "🎤 Interview Simulator",
            "💼 Career Intelligence",
            "🧪 Project Builder",
            "🔬 Research Lab",
            "🗺️ Skill Graph",
            "⚡ Focus Mode",
            "📈 Analytics",
            "⚙️ Settings",
        ],
    )

    st.divider()
    goal = st.slider("Daily study goal (minutes)", 15, 300, 60, 15)
    progress = min(st.session_state.study_minutes / max(goal, 1), 1.0)
    st.progress(progress)
    st.caption(f"{st.session_state.study_minutes} / {goal} minutes")

    if st.button("🔄 Reset Session", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v
        st.rerun()

# ----------------------------- Hero -----------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🧠 AI Study Buddy 360</h1>
        <p>One intelligent workspace for learning, coding, career growth,
        research and professional productivity.</p>
        <span class="badge">POWERED BY GOOGLE GEMINI • ONE API KEY • 360 AI WORKSPACE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================= COMMAND CENTER =======================

if page == "🏠 Command Center":
    st.subheader("Your AI Command Center")
    st.caption("Describe the outcome you want. AI will recommend the best workflow.")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("⭐", st.session_state.points, "Points"),
        ("📚", st.session_state.topics, "Topics"),
        ("🧠", st.session_state.quizzes, "Quizzes"),
        ("🔥", st.session_state.streak, "Streak"),
    ]
    for col, (icon, value, label) in zip((c1, c2, c3, c4), metrics):
        with col:
            st.markdown(
                f'<div class="metric"><div>{icon}</div>'
                f'<div class="metric-number">{value}</div>'
                f'<div class="small">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### ⚡ Universal AI Command")
    command = st.text_area(
        "What do you want to accomplish?",
        height=120,
        placeholder=(
            "Example: I have an Infosys placement in 7 days. "
            "Create a complete preparation strategy using my weak areas."
        ),
    )

    if st.button("🚀 Run Intelligent Workflow", type="primary", use_container_width=True):
        if not command.strip():
            st.warning("Describe your goal first.")
        else:
            workflow_prompt = f"""
You are the orchestration brain of AI Study Buddy 360.

USER REQUEST:
{command}

USER CONTEXT:
{profile_context()}

Determine the best workflow and then execute the useful part now.
Return:
1. Goal interpretation
2. Recommended tools/modules
3. Priority actions
4. A practical step-by-step plan
5. Immediate first action
6. What the learner should do next

Be concrete, personalized and useful. Do not claim that an external tool was actually
executed unless it was provided in this application.
"""
            with st.spinner("Designing your intelligent workflow..."):
                try:
                    result = ask_ai(workflow_prompt, max_tokens=7000)
                    st.markdown(result)
                    save_result(result)
                    download_result("AI_Command_Workflow.md")
                    award(15)
                except Exception as e:
                    show_error(e)

    st.markdown("### 🌟 Platform Capabilities")
    features = [
        ("🤖", "Multi-AI Hub", "Gemini, OpenAI, OpenRouter, Groq and local Ollama."),
        ("📚", "Learning OS", "Tutor, notes, quizzes, flashcards, exams and planning."),
        ("💻", "Developer Lab", "Debug, review, test, SQL, architecture and DSA."),
        ("📊", "Data Lab", "CSV/Excel analysis, EDA, insights and business reports."),
        ("💼", "Career OS", "Resume, ATS, job analysis, skill gaps and interviews."),
        ("📄", "Document AI", "PDFs, images, notes and document question answering."),
        ("🧪", "Project Builder", "Turn an idea into architecture, roadmap and portfolio."),
        ("🔬", "Research Lab", "Research questions, papers, methodology and gaps."),
        ("🗺️", "Skill Graph", "Dynamic path from current skills to target role."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f'<div class="feature"><div class="feature-icon">{icon}</div>'
                f'<h3>{title}</h3><p class="small">{desc}</p></div>',
                unsafe_allow_html=True,
            )

# ============================= CHAT =============================

elif page == "💬 AI Chat":
    st.subheader("💬 Universal AI Chat")
    st.caption("Persistent tutor-style conversation in this session.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    msg = st.chat_input("Ask anything about learning, coding, career or work...")
    if msg:
        st.session_state.messages.append({"role": "user", "content": msg})
        with st.chat_message("user"):
            st.markdown(msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    history = "\n".join(
                        f'{m["role"]}: {m["content"]}' for m in st.session_state.messages[-12:]
                    )
                    answer = ask_ai(
                        f"{profile_context()}\nRecent conversation:\n{history}\n\nAnswer the latest request: {msg}",
                        system=(
                            "You are the central AI mentor of AI Study Buddy 360. "
                            "Teach clearly, challenge assumptions, personalize answers, "
                            "and suggest practical next steps."
                        ),
                        max_tokens=6000,
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    award(5)
                except Exception as e:
                    show_error(e)

# =========================== MODEL LAB ===========================

elif page == "🧠 AI Model Lab":
    st.subheader("🧠 Gemini AI Laboratory")
    st.write(
        "Experiment with the same Gemini engine used across AI Study Buddy 360. "
        "Test prompts, teaching styles and structured outputs."
    )
    st.info(f"🤖 Active model: **{GEMINI_MODEL}**")

    prompt = st.text_area(
        "Laboratory task",
        height=180,
        placeholder="Explain machine learning to a beginner and give a practical example.",
    )
    mode = st.selectbox(
        "Experiment mode",
        ["Best Answer", "Step-by-Step Tutor", "Interview Expert", "Exam Coach", "Creative Explainer"],
    )

    if st.button("🧪 Run Gemini Experiment", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Enter a task first.")
        else:
            system = (
                "You are the advanced reasoning engine inside AI Study Buddy 360. "
                "Produce accurate, structured and practical responses."
            )
            experiment_prompt = (
                f"Experiment mode: {mode}\n\n"
                f"User task:\n{prompt}\n\n"
                "Give the strongest useful answer. Use headings, examples and "
                "actionable takeaways where appropriate."
            )
            with st.spinner("Gemini is running the experiment..."):
                try:
                    result = ask_ai(experiment_prompt, system=system, max_tokens=7000)
                    st.markdown(result)
                    save_result(result)
                    download_result("Gemini_AI_Experiment.md")
                    award(10)
                except Exception as e:
                    show_error(e)

# ============================== LEARN ============================

elif page == "📚 Learn":
    st.subheader("📚 Adaptive Learning Studio")
    c1, c2, c3 = st.columns(3)
    with c1:
        topic = st.text_input("Topic", placeholder="Neural Networks")
    with c2:
        level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced", "Interview"])
    with c3:
        style = st.selectbox(
            "Style",
            ["Simple", "Academic", "Real-world", "Exam", "Interview"],
        )
    prompt = f"""
Teach {topic} to a university student at {level} level in a {style} style.

Include:
- definition
- intuition
- why it matters
- core concepts
- step-by-step explanation
- worked example
- real-world applications
- common mistakes
- interview questions
- quick revision
- 5 self-test questions
"""
    run_ai_button(
        "✨ Generate Complete Lesson", prompt,
        system="You are an expert university tutor. Be accurate and never invent facts.", temperature=0.5, max_tokens=7000,
        points=10, topic=topic, filename="AI_Lesson.md"
    )

# =========================== SMART NOTES =========================
# =========================== SMART NOTES =======================

elif page == "📝 Smart Notes":
    st.subheader("📝 Smart Notes Studio")

    st.info(
        "📄 Upload a PDF, paste your notes, or use both. "
        "Gemini will turn your material into smart study notes."
    )

    uploaded_pdf = st.file_uploader(
        "📄 Upload your study PDF (optional)",
        type=["pdf"],
        help="Upload lecture notes, textbooks, study material, etc.",
    )

    notes = st.text_area(
        "📝 Paste additional notes (optional)",
        height=250,
        placeholder=(
            "Paste lecture notes, textbook notes, class material "
            "or any additional information..."
        ),
    )

    output = st.selectbox(
        "Output type",
        [
            "Complete Summary",
            "Exam Notes",
            "One-Page Revision",
            "Key Points",
            "Mind Map Structure",
            "Cheat Sheet",
            "Important Questions",
        ],
    )

    detail = st.select_slider(
        "Detail level",
        ["Short", "Balanced", "Detailed"],
        value="Balanced",
    )

    if st.button(
        "🧠 Generate Smart Notes",
        type="primary",
        use_container_width=True,
    ):

        if uploaded_pdf is None and not notes.strip():
            st.warning(
                "📄 Please upload a PDF or 📝 paste some notes first."
            )

        else:
            instruction = f"""
You are an expert AI study assistant.

Create {output} from the student's learning material.

Detail level: {detail}

Follow these rules:
- Preserve important facts and concepts.
- Remove unnecessary repetition.
- Explain difficult concepts simply.
- Use clear headings and bullet points.
- Highlight important definitions.
- Include formulas when relevant.
- Include examples when useful.
- Add memory tricks when useful.
- Make the result useful for exams and revision.
- Finish with a section called "Quick Revision Questions".
- Do not invent facts that are not present in the material.

"""

            if notes.strip():
                instruction += f"""

ADDITIONAL STUDENT NOTES:
{notes}
"""

            with st.spinner("🤖 Gemini is creating your smart notes..."):
                try:

                    if uploaded_pdf is not None:
                        result = analyze_file(
                            uploaded_pdf,
                            instruction,
                        )
                    else:
                        result = ask_gemini(
                            instruction,
                            temperature=0.4,
                            max_tokens=6000,
                        )

                    st.success("✅ Smart notes generated successfully!")
                    st.markdown(result)

                    save_result(result)

                    result_actions("AI_Study_Notes.md")

                    st.session_state.points += 10

                except Exception as e:
                    show_error(e)

# =========================== QUIZ ARENA ==========================

elif page == "🧠 Quiz Arena":
    st.subheader("🧠 AI Quiz Arena")
    c1, c2, c3 = st.columns(3)
    with c1:
        topic = st.text_input("Topic", placeholder="Python, SQL, AI...")
    with c2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Expert"])
    with c3:
        count = st.slider("Questions", 3, 20, 5)
    prompt = f"""
Create exactly {count} multiple-choice questions about {topic}.
Difficulty: {difficulty}.
For every question provide:
QUESTION
A
B
C
D
ANSWER
EXPLANATION
Make the questions useful for learning and interviews.
"""
    run_ai_button(
        "🎮 Generate Quiz", prompt, temperature=0.6,
        max_tokens=8000, points=15, filename="AI_Quiz.md"
    )

# ========================== FLASHCARDS ===========================

elif page == "🃏 Flashcards":
    st.subheader("🃏 Active Recall Flashcards")
    topic = st.text_input("Topic", placeholder="SQL Joins")
    count = st.slider("Number of cards", 5, 40, 10)
    prompt = f"""
Create {count} active-recall flashcards about {topic}.
Format:
CARD 1
FRONT: Question
BACK: Answer
Progress from fundamentals to harder concepts. Focus on understanding.
"""
    if st.button("✨ Create Flashcards", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            with st.spinner("Creating flashcards..."):
                try:
                    result = ask_ai(prompt, temperature=0.6, max_tokens=7000)
                    st.markdown(result)
                    save_result(result)
                    download_result("AI_Flashcards.md")
                    st.session_state.flashcards += count
                    award(count)
                except Exception as e:
                    show_error(e)

# ========================= STUDY PLANNER =========================

elif page == "📅 Study Planner":
    st.subheader("📅 Personalized Study Planner")
    subjects = st.text_area("Subjects / syllabus", height=180)
    c1, c2, c3 = st.columns(3)
    with c1:
        hours = st.number_input("Hours/day", 0.5, 12.0, 2.0, 0.5)
    with c2:
        target = st.date_input("Target date", date.today() + timedelta(days=30))
    with c3:
        priority = st.selectbox(
            "Priority", ["Balanced", "Weak subjects first", "Exam focused", "Career focused"]
        )
    days = max((target - date.today()).days, 1)
    prompt = f"""
Create a realistic personalized study plan.
Subjects:
{subjects}
Days available: {days}
Hours/day: {hours}
Priority: {priority}
Include daily schedule, weekly milestones, theory/practice balance,
active recall, revision cycles, mock tests, weak-topic strategy,
final revision and measurable goals.
"""
    run_ai_button(
        "📅 Build My Plan", prompt, temperature=0.5,
        max_tokens=8000, points=20, filename="AI_Study_Plan.md"
    )

# ============================ EXAM MODE ==========================

elif page == "🎯 Exam Mode":
    st.subheader("🎯 Exam Preparation Intelligence")
    exam = st.text_input("Exam name", placeholder="Placement / GATE / University")
    syllabus = st.text_area("Syllabus", height=200)
    days = st.number_input("Days remaining", 1, 365, 30)
    prompt = f"""
Act as an elite exam preparation strategist.
Exam: {exam}
Syllabus:
{syllabus}
Days remaining: {days}
Create:
1. priority matrix
2. high-value topics
3. study order
4. daily strategy
5. practice strategy
6. mock-test strategy
7. revision cycles
8. common mistakes
9. last 7-day plan
10. exam-day checklist
"""
    run_ai_button(
        "🚀 Activate Exam Mode", prompt, temperature=0.45,
        max_tokens=8000, points=25, filename="AI_Exam_Strategy.md"
    )

# ========================= DEVELOPER LAB =========================

elif page == "💻 Developer Lab":
    st.subheader("💻 AI Developer Lab")
    language = st.selectbox(
        "Language", ["Python", "Java", "C", "C++", "JavaScript", "TypeScript",
                     "SQL", "HTML/CSS", "Other"]
    )
    task = st.selectbox(
        "Task",
        ["Explain", "Debug", "Optimize", "Refactor", "Generate Tests",
         "Code Review", "Generate Documentation", "Interview Questions",
         "System Design", "DSA Hint"],
    )
    code = st.text_area("Code / problem", height=350)
    prompt = f"""
You are a senior software engineer.
Language: {language}
Task: {task}
Code/problem:
```{language}
{code}
```
Provide analysis, bugs/problems, explanation, improved solution,
complexity, edge cases, best practices and interview insights.
If the task is system design, include architecture, components,
data flow, scalability, security and trade-offs.
"""
    run_ai_button(
        "🧑‍💻 Analyze / Build", prompt,
        system="Be technically precise. Never claim code was executed unless it was.", temperature=0.35, max_tokens=9000,
        points=15, filename="AI_Developer_Review.md"
    )

# ========================= DATA ANALYTICS ========================

elif page == "📊 Data & Analytics":
    st.subheader("📊 AI Data & Analytics Lab")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
    )
    analysis_goal = st.text_area(
        "Business question / analysis goal",
        placeholder="Find important trends, anomalies and business recommendations.",
        height=120,
    )
    if uploaded and pd:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")
            st.dataframe(df.head(20), use_container_width=True)
            st.markdown("### Dataset Profile")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", f"{len(df.columns):,}")
            c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
            if st.button("🔎 Generate AI Data Report", type="primary", use_container_width=True):
                schema = []
                for col in df.columns:
                    schema.append({
                        "column": str(col),
                        "dtype": str(df[col].dtype),
                        "missing": int(df[col].isna().sum()),
                        "unique": int(df[col].nunique()),
                    })
                sample = df.head(12).to_dict(orient="records")
                prompt = f"""
Analyze this dataset for a student/data professional.
Business goal: {analysis_goal}
Schema:
{json.dumps(schema, indent=2, default=str)}
Sample:
{json.dumps(sample, indent=2, default=str)}

Return:
1. executive summary
2. data-quality issues
3. important patterns
4. likely relationships
5. recommended visualizations
6. useful SQL/Python analysis ideas
7. business insights
8. limitations
9. next analyses
Do not invent values that are not in the supplied sample/schema.
"""
                with st.spinner("Analyzing dataset..."):
                    try:
                        result = ask_ai(prompt, max_tokens=8000)
                        st.markdown(result)
                        save_result(result)
                        download_result("AI_Data_Report.md")
                        award(20)
                    except Exception as e:
                        show_error(e)
        except Exception as e:
            st.error(f"Could not read the dataset: {e}")

# ===================== DOCUMENT INTELLIGENCE =====================

elif page == "📄 Document Intelligence":
    st.subheader("📄 Universal Document Intelligence")
    st.write("Upload study/work material and ask AI to transform it.")
    uploaded = st.file_uploader(
        "Upload PDF, image, text, Markdown or document",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md", "csv"],
    )
    instruction = st.text_area(
        "Instruction",
        height=160,
        placeholder=(
            "Summarize it, identify important exam topics, create questions, "
            "and build a revision plan."
        ),
    )
    if uploaded:
        st.success(f"Ready: {uploaded.name} ({uploaded.size/1024:.1f} KB)")
    if st.button("🔍 Analyze Material", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Upload a file first.")
        elif not instruction.strip():
            st.warning("Tell AI what to do.")
        else:
            with st.spinner("Analyzing your material with Gemini..."):
                try:
                    result = analyze_uploaded_file(uploaded, instruction)
                    st.markdown(result)
                    save_result(result)
                    download_result("AI_Document_Analysis.md")
                    st.session_state.documents.append(uploaded.name)
                    award(20)
                except Exception as e:
                    show_error(e)

# ======================= INTERVIEW SIMULATOR =====================

elif page == "🎤 Interview Simulator":
    st.subheader("🎤 AI Interview Simulator")
    role = st.text_input("Target role", placeholder="Data Analyst / Data Scientist")
    interview_type = st.selectbox(
        "Interview", ["Technical", "HR", "Behavioral", "Mixed", "Mock Interview"]
    )
    level = st.selectbox("Level", ["Student", "Fresher", "Entry Level", "Experienced"])
    answer = st.text_area(
        "Candidate answer (for evaluation)",
        height=180,
        placeholder="Paste your answer here, or leave blank to get a question.",
    )
    if st.button("🎯 Run Interview Session", type="primary", use_container_width=True):
        if answer.strip():
            prompt = f"""
Evaluate this interview answer.
Role: {role}
Level: {level}
Type: {interview_type}
Answer:
{answer}

Score 0-100 for correctness, relevance, structure, communication and confidence.
Give strengths, weaknesses, an improved answer, and one follow-up question.
"""
        else:
            prompt = f"""
Act as a senior interviewer for {role}, candidate level {level},
interview type {interview_type}. Ask ONE realistic question only.
After the question, give a short hint on what a strong answer should contain.
"""
        with st.spinner("Interviewer is evaluating..."):
            try:
                result = ask_ai(prompt, temperature=0.6, max_tokens=5000)
                st.markdown(result)
                save_result(result)
                download_result("AI_Interview_Session.md")
                award(20)
            except Exception as e:
                show_error(e)

# ======================= CAREER INTELLIGENCE =====================

elif page == "💼 Career Intelligence":
    st.subheader("💼 Career Intelligence Center")
    mode = st.selectbox(
        "Career tool",
        ["Resume Review", "ATS Analysis", "Job Description Analysis",
         "Skill Gap", "LinkedIn Profile", "Career Roadmap", "Portfolio Review"],
    )
    role = st.text_input("Target role", placeholder="Data Analyst")
    content = st.text_area(
        "Paste resume / job description / profile / portfolio content",
        height=320,
    )
    prompt = f"""
You are an expert career strategist.
Tool: {mode}
Target role: {role}
Candidate content:
{content}

Give a professional, practical analysis.
For ATS: identify keywords, missing keywords, structure problems and an improved summary.
For job description: extract skills, responsibilities, priority keywords and interview areas.
For skill gap: compare evidence in the content with the target role.
For LinkedIn: improve headline, About, skills and project positioning.
For portfolio: evaluate projects, proof of impact and presentation quality.
For roadmap: give a 30/60/90 day plan.
"""
    run_ai_button(
        "🚀 Run Career Analysis", prompt,
        temperature=0.45, max_tokens=8000, points=20,
        filename="AI_Career_Analysis.md"
    )

# ========================= PROJECT BUILDER ========================

elif page == "🧪 Project Builder":
    st.subheader("🧪 AI Project Builder")
    idea = st.text_area(
        "Project idea",
        height=120,
        placeholder="Build an AI system that detects fake job postings.",
    )
    audience = st.text_input("Target users", placeholder="Students and recruiters")
    stack = st.text_input(
        "Preferred technology",
        placeholder="Python, Streamlit, FastAPI, PostgreSQL, Gemini",
    )
    prompt = f"""
Turn this idea into a portfolio-grade real-world project.

Idea: {idea}
Audience: {audience}
Preferred stack: {stack}

Create:
1. problem statement
2. users and use cases
3. functional requirements
4. non-functional requirements
5. architecture
6. modules
7. database design
8. API design
9. UI pages
10. AI architecture
11. model/provider strategy
12. security
13. testing
14. deployment
15. GitHub structure
16. README outline
17. resume bullet points
18. LinkedIn project description
19. future enhancements
20. phased implementation roadmap

Do not pretend that code has been implemented.
"""
    run_ai_button(
        "🏗️ Architect My Project", prompt,
        temperature=0.45, max_tokens=10000, points=30,
        filename="AI_Project_Blueprint.md"
    )

# ============================ RESEARCH ===========================

elif page == "🔬 Research Lab":
    st.subheader("🔬 AI Research Intelligence")
    topic = st.text_input("Research topic", placeholder="Fake job detection using NLP")
    question = st.text_area("Research goal/question", height=130)
    prompt = f"""
Act as a research mentor.
Topic: {topic}
Research goal: {question}

Create:
- refined research question
- objectives
- possible hypotheses
- key concepts
- methodology options
- dataset requirements
- evaluation metrics
- experiment design
- expected limitations
- possible research gap directions
- paper structure
- future work
Clearly label ideas/inferences and do not invent citations.
"""
    run_ai_button(
        "🔬 Build Research Plan", prompt,
        temperature=0.45, max_tokens=8000, points=25,
        filename="AI_Research_Plan.md"
    )

# ============================ SKILL GRAPH ========================

elif page == "🗺️ Skill Graph":
    st.subheader("🗺️ Dynamic Skill & Career Graph")
    target = st.text_input("Target role", placeholder="Data Scientist")
    current = st.text_area(
        "Current skills",
        placeholder="Python, SQL, Excel, Power BI, basic ML",
        height=150,
    )
    hours = st.number_input("Learning hours/week", 1, 60, 10)
    prompt = f"""
Build a dynamic skill graph for target role: {target}
Current skills: {current}
Hours/week: {hours}

Return:
1. strengths
2. missing skills
3. skill dependencies
4. ordered learning path
5. projects proving each skill
6. interview milestones
7. 30/60/90-day roadmap
8. career readiness score /100 with rationale

Represent dependencies as A -> B where useful.
"""
    run_ai_button(
        "🧭 Generate Skill Graph", prompt,
        temperature=0.5, max_tokens=8000, points=25,
        filename="AI_Skill_Graph.md"
    )

# =========================== FOCUS MODE ==========================

elif page == "⚡ Focus Mode":
    st.subheader("⚡ AI Focus Session")
    topic = st.text_input("Focus topic", placeholder="Machine Learning")
    duration = st.slider("Duration", 15, 120, 30, 5)
    session_type = st.selectbox(
        "Session type", ["Learn", "Practice", "Revision", "Problem Solving", "Interview"]
    )
    prompt = f"""
Create a {duration}-minute focused study session for {topic}.
Session type: {session_type}.
Include exact time blocks for warm-up, core learning, active recall,
practice, self-test and final recap. Give concrete tasks.
"""
    if st.button("⚡ Start Focus Plan", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            with st.spinner("Creating your focus session..."):
                try:
                    result = ask_ai(prompt, temperature=0.45, max_tokens=5000)
                    st.markdown(result)
                    save_result(result)
                    download_result("AI_Focus_Session.md")
                    st.session_state.study_minutes += duration
                    award(duration, topic)
                    st.success(f"{duration} minutes added to study time.")
                except Exception as e:
                    show_error(e)

# ============================ ANALYTICS ===========================

elif page == "📈 Analytics":
    st.subheader("📈 Personal Learning & AI Analytics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Study time", f"{st.session_state.study_minutes} min")
    c2.metric("Points", st.session_state.points)
    c3.metric("Topics", st.session_state.topics)
    c4.metric("Flashcards", st.session_state.flashcards)

    st.markdown("### Gemini Performance")
    stats = st.session_state.gemini_stats
    if stats["calls"]:
        avg = stats["seconds"] / stats["calls"]
        if pd:
            st.dataframe(
                pd.DataFrame([{
                    "AI Engine": "Google Gemini",
                    "Model": GEMINI_MODEL,
                    "Calls": stats["calls"],
                    "Total seconds": round(stats["seconds"], 2),
                    "Average seconds": round(avg, 2),
                }]),
                use_container_width=True,
            )
    else:
        st.info("Gemini usage will appear after your first AI call.")

    st.markdown("### Learning Snapshot")
    st.markdown(
        f"""
        <div class="card">
        <b>Last topic:</b> {st.session_state.last_topic or "Not started"}<br><br>
        <b>Topics:</b> {st.session_state.topics}<br><br>
        <b>Quizzes:</b> {st.session_state.quizzes}<br><br>
        <b>Documents:</b> {len(st.session_state.documents)}<br><br>
        <b>Streak:</b> 🔥 {st.session_state.streak} days<br><br>
        <b>AI Engine:</b> {st.session_state.last_provider or "Google Gemini"}
        </div>
        """,
        unsafe_allow_html=True,
    )
    remaining = max(goal - st.session_state.study_minutes, 0)
    st.progress(min(st.session_state.study_minutes / max(goal, 1), 1.0))
    st.info(f"{remaining} minutes remaining for today's goal.")

# ============================ SETTINGS ===========================

elif page == "⚙️ Settings":
    st.subheader("⚙️ Platform Settings")
    st.write(f"**AI Study Buddy 360 v{APP_VERSION}**")

    st.markdown("### 🤖 AI Engine")
    st.success("🟢 Google Gemini — Active")
    st.write(f"**Model:** `{GEMINI_MODEL}`")
    st.write("**Architecture:** Single-provider · Single-key · Secure AI gateway")

    st.markdown("### 🔐 Streamlit Secrets")
    st.code(
        """GEMINI_API_KEY = "your_actual_gemini_key"
GEMINI_MODEL = "gemini-3.7-flash" """,
        language="toml",
    )
    st.warning(
        "Never paste your real API key into app.py or GitHub. "
        "Use Streamlit Secrets for deployment and .env for local development."
    )

    st.markdown("### ✅ Simplified configuration")
    st.markdown(
        """
        This version intentionally uses **one API key only: Google Gemini**.

        • No OpenAI key required  
        • No OpenRouter key required  
        • No Groq key required  
        • No Ollama setup required  

        All learning, career, coding, document, research and project features
        use the same Gemini AI engine.
        """
    )

# ---------------------------- Footer ----------------------------

st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:35px 0 10px">
        <b>AI Study Buddy 360</b> · Gemini AI · Learning · Career · Work OS
        <br>
        Learn smarter · Build faster · Grow continuously
    </div>
    """,
    unsafe_allow_html=True,
)

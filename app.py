import os
import tempfile
from datetime import date, timedelta

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============================================================
# AI STUDY BUDDY - GEMINI EDITION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL = "gemini-2.5-flash"


# ----------------------------- API -----------------------------

def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


@st.cache_resource
def create_client(api_key):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


API_KEY = get_api_key()
client = create_client(API_KEY)


# -------------------------- SESSION ----------------------------

defaults = {
    "messages": [],
    "points": 0,
    "topics": 0,
    "quizzes": 0,
    "flashcards": 0,
    "study_minutes": 0,
    "streak": 1,
    "last_result": "",
    "last_topic": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------- STYLE ----------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(99,102,241,.10), transparent 32%),
            radial-gradient(circle at 100% 100%, rgba(14,165,233,.08), transparent 30%),
            #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #172554);
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .hero {
        padding: 32px;
        border-radius: 26px;
        color: white;
        background: linear-gradient(135deg, #0f172a, #312e81, #0369a1);
        box-shadow: 0 18px 45px rgba(15,23,42,.16);
        margin-bottom: 24px;
    }

    .hero h1 {
        margin: 0;
        font-size: 42px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 17px;
        margin: 8px 0 0;
    }

    .badge {
        display: inline-block;
        margin-top: 15px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.14);
        font-size: 12px;
        font-weight: 700;
    }

    .card {
        background: rgba(255,255,255,.92);
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(15,23,42,.06);
        margin-bottom: 16px;
    }

    .metric {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: #312e81;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
    }

    .feature {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 20px;
        min-height: 145px;
        box-shadow: 0 8px 25px rgba(15,23,42,.05);
    }

    .feature-icon {
        font-size: 30px;
    }

    .feature h3 {
        margin: 7px 0;
    }

    .feature p {
        color: #64748b;
    }

    .small {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------ AI FUNCTIONS -------------------------

def require_client():
    if client is None:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to "
            ".env locally or Streamlit Secrets in your deployed app."
        )


def ask_gemini(prompt, system=None, temperature=0.6, max_tokens=5000):
    require_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    if system:
        config.system_instruction = system

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


def ask_chat(message):
    require_client()

    if "chat" not in st.session_state:
        st.session_state.chat = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are AI Study Buddy, a patient expert tutor. "
                    "Teach clearly, ask useful follow-up questions, "
                    "and adapt explanations to the learner."
                )
            ),
        )

    response = st.session_state.chat.send_message(message=message)

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


def analyze_file(uploaded_file, instruction):
    require_client()

    suffix = os.path.splitext(uploaded_file.name)[1] or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name

    try:
        gemini_file = client.files.upload(file=temp_path)

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                instruction,
                gemini_file,
            ],
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=6000,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def show_error(error):
    message = str(error)

    if "401" in message or "API key" in message or "api_key" in message:
        st.error(
            "🔐 Gemini authentication failed. Check GEMINI_API_KEY."
        )
    elif "429" in message or "quota" in message.lower():
        st.error(
            "⚡ Gemini quota/rate limit reached. Check your Gemini API usage."
        )
    else:
        st.error("Something went wrong: " + message)


def save_result(result):
    st.session_state.last_result = result


def result_actions(filename):
    if st.session_state.last_result:
        st.download_button(
            "📥 Download Result",
            st.session_state.last_result,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )


# --------------------------- SIDEBAR ---------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:12px 4px 20px">
            <div style="font-size:48px">🎓</div>
            <h2 style="margin:0">AI Study Buddy</h2>
            <p style="color:#94a3b8">Your Personal AI Learning OS</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Learning Center",
        [
            "🏠 Dashboard",
            "🤖 AI Tutor",
            "📚 Explain Topic",
            "📝 Smart Notes",
            "🧠 Quiz Arena",
            "🃏 Flashcards",
            "📅 Study Planner",
            "🎯 Exam Mode",
            "💻 Coding Mentor",
            "🎤 Interview Coach",
            "🗺️ Learning Roadmap",
            "📄 Material Analyzer",
            "📊 Analytics",
            "⚡ Focus Mode",
        ],
    )

    st.divider()

    goal = st.slider(
        "Daily study goal (minutes)",
        15,
        300,
        60,
        15,
    )

    progress = min(st.session_state.study_minutes / goal, 1.0)
    st.progress(progress)
    st.caption(
        f"{st.session_state.study_minutes} / {goal} minutes"
    )

    if st.button("🔄 Reset Session", use_container_width=True):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.session_state.pop("chat", None)
        st.rerun()


# ---------------------------- HERO -----------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎓 AI Study Buddy</h1>
        <p>
            Learn, practice, revise, plan and prepare with your
            personal Gemini-powered learning assistant.
        </p>
        <span class="badge">POWERED BY GOOGLE GEMINI</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================== DASHBOARD ========================

if page == "🏠 Dashboard":
    st.subheader("Your Learning Command Center")

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("⭐", st.session_state.points, "Study Points"),
        ("📚", st.session_state.topics, "Topics"),
        ("🧠", st.session_state.quizzes, "Quizzes"),
        ("🔥", st.session_state.streak, "Day Streak"),
    ]

    for col, item in zip((c1, c2, c3, c4), metrics):
        icon, value, label = item
        with col:
            st.markdown(
                f"""
                <div class="metric">
                    <div>{icon}</div>
                    <div class="metric-number">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 🚀 Learning Studio")

    features = [
        ("🤖", "AI Tutor", "Interactive conversation and personalized teaching."),
        ("📚", "Explain Topic", "Turn difficult concepts into simple lessons."),
        ("📝", "Smart Notes", "Convert notes into exam-ready revision material."),
        ("🧠", "Quiz Arena", "Generate quizzes and explanations for practice."),
        ("🃏", "Flashcards", "Create active-recall cards automatically."),
        ("📅", "Study Planner", "Build a realistic personalized schedule."),
        ("🎯", "Exam Mode", "Prioritize syllabus, revision and mocks."),
        ("💻", "Coding Mentor", "Explain, debug and improve code."),
        ("🎤", "Interview Coach", "Prepare for technical and HR interviews."),
        ("🗺️", "Learning Roadmap", "Build a path from beginner to job-ready."),
        ("📄", "Material Analyzer", "Ask questions about PDFs and documents."),
        ("⚡", "Focus Mode", "Create focused study sessions."),
    ]

    cols = st.columns(3)

    for i, item in enumerate(features):
        icon, title, desc = item
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="feature">
                    <div class="feature-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 💡 Quick Start")
    quick_topic = st.text_input(
        "Enter a topic for an instant mini lesson",
        placeholder="Example: Machine Learning",
    )

    if st.button("✨ Teach Me", type="primary"):
        if not quick_topic.strip():
            st.warning("Enter a topic first.")
        else:
            with st.spinner("Preparing your lesson..."):
                try:
                    result = ask_gemini(
                        (
                            "Teach this topic to a university student: "
                            + quick_topic
                            + "\nUse intuition, example, key points, "
                              "common mistakes and a 5-question self-test."
                        ),
                        system=(
                            "You are an expert tutor. Be accurate, "
                            "structured and encouraging."
                        ),
                    )
                    st.markdown(result)
                    save_result(result)
                    st.session_state.points += 10
                    st.session_state.topics += 1
                except Exception as e:
                    show_error(e)


# =========================== AI TUTOR ==========================

elif page == "🤖 AI Tutor":
    st.subheader("🤖 Personal AI Tutor")
    st.caption(
        "Ask follow-up questions naturally. The tutor remembers "
        "the current conversation."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_message = st.chat_input("Ask your tutor anything...")

    if user_message:
        st.session_state.messages.append(
            {"role": "user", "content": user_message}
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_chat(user_message)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.points += 5
                except Exception as e:
                    show_error(e)


# ========================= EXPLAIN TOPIC =======================

elif page == "📚 Explain Topic":
    st.subheader("📚 Explain Any Topic")

    topic = st.text_input(
        "Topic",
        placeholder="Example: Neural Networks",
    )

    c1, c2 = st.columns(2)

    with c1:
        level = st.selectbox(
            "Level",
            ["Beginner", "Intermediate", "Advanced", "Interview Level"],
        )

    with c2:
        style = st.selectbox(
            "Style",
            [
                "Simple and intuitive",
                "Detailed academic",
                "Real-world examples",
                "Exam focused",
                "Interview focused",
            ],
        )

    if st.button(
        "✨ Generate Complete Lesson",
        type="primary",
        use_container_width=True,
    ):
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            prompt = (
                "Create a complete lesson about: "
                + topic
                + "\nStudent level: "
                + level
                + "\nStyle: "
                + style
                + """
\nInclude:
1. What it is
2. Why it matters
3. Intuition
4. Core concepts
5. Step-by-step explanation
6. Example
7. Real-world use
8. Common mistakes
9. Interview questions
10. Quick revision
11. Five self-test questions
"""
            )

            with st.spinner("Building your lesson..."):
                try:
                    result = ask_gemini(
                        prompt,
                        system=(
                            "You are a university professor and personal "
                            "AI tutor. Never invent facts."
                        ),
                        temperature=0.5,
                    )
                    st.markdown(result)
                    save_result(result)
                    result_actions("AI_Lesson.md")
                    st.session_state.points += 10
                    st.session_state.topics += 1
                    st.session_state.last_topic = topic
                except Exception as e:
                    show_error(e)


# =========================== SMART NOTES =======================

elif page == "📝 Smart Notes":
    st.subheader("📝 Smart Notes Studio")

    notes = st.text_area(
        "Paste your notes",
        height=300,
        placeholder="Paste lecture notes, textbook notes or class material...",
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
        ],
    )

    detail = st.select_slider(
        "Detail",
        ["Short", "Balanced", "Detailed"],
        value="Balanced",
    )

    if st.button(
        "🧠 Transform Notes",
        type="primary",
        use_container_width=True,
    ):
        if not notes.strip():
            st.warning("Paste your notes first.")
        else:
            prompt = (
                "Transform the following study notes into "
                + output
                + ". Detail level: "
                + detail
                + """
\nRules:
- Preserve important facts.
- Remove repetition.
- Explain difficult terms.
- Use headings and bullets.
- Include formulas when relevant.
- Add memory tricks when useful.
- Finish with rapid revision questions.

NOTES:
"""
                + notes
            )

            with st.spinner("Creating smart notes..."):
                try:
                    result = ask_gemini(
                        prompt,
                        temperature=0.4,
                        max_tokens=6000,
                    )
                    st.markdown(result)
                    save_result(result)
                    result_actions("AI_Study_Notes.md")
                    st.session_state.points += 10
                except Exception as e:
                    show_error(e)


# =========================== QUIZ ARENA ========================

elif page == "🧠 Quiz Arena":
    st.subheader("🧠 AI Quiz Arena")

    c1, c2, c3 = st.columns(3)

    with c1:
        topic = st.text_input(
            "Quiz topic",
            placeholder="Python, SQL, AI...",
        )

    with c2:
        difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Medium", "Hard", "Expert"],
        )

    with c3:
        count = st.slider("Questions", 3, 15, 5)

    if st.button(
        "🎮 Generate Quiz",
        type="primary",
        use_container_width=True,
    ):
        if not topic.strip():
            st.warning("Enter a quiz topic.")
        else:
            prompt = (
                "Create exactly "
                + str(count)
                + " multiple-choice questions about "
                + topic
                + ". Difficulty: "
                + difficulty
                + """
\nFor every question include:
QUESTION:
A.
B.
C.
D.
ANSWER:
EXPLANATION:

Make the questions useful for learning and interview preparation.
"""
            )

            with st.spinner("Generating your challenge..."):
                try:
                    result = ask_gemini(
                        prompt,
                        temperature=0.6,
                        max_tokens=6000,
                    )
                    st.markdown(result)
                    save_result(result)
                    result_actions("AI_Quiz.md")
                    st.session_state.quizzes += 1
                    st.session_state.points += 15
                except Exception as e:
                    show_error(e)


# ========================== FLASHCARDS =========================

elif page == "🃏 Flashcards":
    st.subheader("🃏 Active Recall Flashcards")

    topic = st.text_input(
        "Topic",
        placeholder="Example: SQL Joins",
    )

    count = st.slider("Number of cards", 5, 30, 10)

    if st.button(
        "✨ Create Flashcards",
        type="primary",
        use_container_width=True,
    ):
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            prompt = (
                "Create "
                + str(count)
                + " active-recall flashcards about "
                + topic
                + """
.
Format each card:
CARD 1
FRONT: Question
BACK: Answer

Progress from fundamentals to harder concepts.
Focus on understanding, not random trivia.
"""
            )

            with st.spinner("Creating flashcards..."):
                try:
                    result = ask_gemini(
                        prompt,
                        temperature=0.6,
                        max_tokens=5000,
                    )
                    st.markdown(result)
                    save_result(result)
                    result_actions("AI_Flashcards.md")
                    st.session_state.flashcards += count
                    st.session_state.points += count
                except Exception as e:
                    show_error(e)


# ========================= STUDY PLANNER =======================

elif page == "📅 Study Planner":
    st.subheader("📅 Personalized Study Planner")

    subjects = st.text_area(
        "Subjects / syllabus",
        placeholder="Python\nSQL\nMachine Learning\nData Structures",
        height=170,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        hours = st.number_input(
            "Hours per day",
            0.5,
            12.0,
            2.0,
            0.5,
        )

    with c2:
        target_date = st.date_input(
            "Target date",
            date.today() + timedelta(days=30),
        )

    with c3:
        priority = st.selectbox(
            "Priority",
            [
                "Balanced",
                "Weak subjects first",
                "Exam focused",
                "Career focused",
            ],
        )

    if st.button(
        "📅 Build My Plan",
        type="primary",
        use_container_width=True,
    ):
        if not subjects.strip():
            st.warning("Enter your subjects.")
        else:
            days = max((target_date - date.today()).days, 1)

            prompt = (
                "Create a realistic personalized study plan.\n"
                "Subjects:\n"
                + subjects
                + "\nDays available: "
                + str(days)
                + "\nHours per day: "
                + str(hours)
                + "\nPriority: "
                + priority
                + """
\nInclude:
1. Daily schedule
2. Weekly milestones
3. Theory/practice balance
4. Active recall
5. Revision cycles
6. Mock tests
7. Weak-topic strategy
8. Final revision
9. Measurable daily goals
"""
            )

            with st.spinner("Designing your study system..."):
                try:
                    result = ask_gemini(
                        prompt,
                        temperature=0.5,
                        max_tokens=7000,
                    )
                    st.markdown(result)
                    save_result(result)
                    result_actions("AI_Study_Plan.md")
                    st.session_state.points += 20
                except Exception as e:
                    show_error(e)


# =========================== EXAM MODE ========================

elif page == "🎯 Exam Mode":
    st.subheader("🎯 Exam Preparation Intelligence")

    exam = st.text_input(
        "Exam name",
        placeholder="Placement / GATE / University Exam",
    )

    syllabus = st.text_area(
        "Syllabus",
        height=180,
    )

    days = st.number_input(
        "Days remaining",
        1,
        365,
        30,
    )

    if st.button(
        "🚀 Activate Exam Mode",
        type="primary",
        use_container_width=True,
    ):
        prompt = (
            "Act as an elite exam preparation strategist.\n"
            "Exam: "
            + exam
            + "\nSyllabus:\n"
            + syllabus
            + "\nDays remaining: "
            + str(days)
            + """
\nCreate:
1. Priority matrix
2. High-value topics
3. Study order
4. Daily strategy
5. Practice strategy
6. Mock-test strategy
7. Revision cycles
8. Common mistakes
9. Last 7-day plan
10. Exam-day checklist
"""
        )

        with st.spinner("Analyzing your exam strategy..."):
            try:
                result = ask_gemini(
                    prompt,
                    temperature=0.45,
                    max_tokens=7000,
                )
                st.markdown(result)
                save_result(result)
                result_actions("AI_Exam_Strategy.md")
                st.session_state.points += 25
            except Exception as e:
                show_error(e)


# ========================= CODING MENTOR =======================

elif page == "💻 Coding Mentor":
    st.subheader("💻 AI Coding Mentor")

    language = st.selectbox(
        "Language",
        ["Python", "Java", "C", "C++", "JavaScript", "SQL", "Other"],
    )

    task = st.selectbox(
        "What should AI do?",
        [
            "Explain the code",
            "Find bugs",
            "Optimize the code",
            "Improve code quality",
            "Generate test cases",
            "Prepare interview questions",
        ],
    )

    code = st.text_area(
        "Paste your code",
        height=330,
        placeholder="Paste your code here...",
    )

    if st.button(
        "🧑‍💻 Analyze Code",
        type="primary",
        use_container_width=True,
    ):
        if not code.strip():
            st.warning("Paste your code first.")
        else:
            prompt = (
                "You are an expert software engineer and coding mentor.\n"
                "Language: "
                + language
                + "\nTask: "
                + task
                + "\n\nCODE:\n```"
                + language
                + "\n"
                + code
                + """
\n```
\nProvide:
1. Analysis
2. Bugs/problems
3. Explanation
4. Improved solution
5. Time complexity
6. Space complexity
7. Edge cases
8. Best practices
9. Interview insights
"""
            )

            with st.spinner("Reviewing your code..."):
                try:
                    result = ask_gemini(
                        prompt,
                        temperature=0.35,
                        max_tokens=7000,
                    )
                    st.markdown(result)
                    save_result(result)
                    result_actions("AI_Code_Review.md")
                    st.session_state.points += 15
                except Exception as e:
                    show_error(e)


# ========================= INTERVIEW COACH =====================

elif page == "🎤 Interview Coach":
    st.subheader("🎤 AI Interview Coach")

    role = st.text_input(
        "Target role",
        placeholder="Data Analyst / Data Scientist / Software Engineer",
    )

    c1, c2 = st.columns(2)

    with c1:
        level = st.selectbox(
            "Candidate level",
            ["Student", "Fresher", "Entry Level", "Experienced"],
        )

    with c2:
        interview_type = st.selectbox(
            "Interview type",
            ["Technical", "HR", "Behavioral", "Mixed", "Mock Interview"],
        )

    if st.button(
        "🎤 Prepare Me",
        type="primary",
        use_container_width=True,
    ):
        prompt = (
            "Act as a senior interviewer.\n"
            "Role: "
            + role
            + "\nCandidate level: "
            + level
            + "\nInterview type: "
            + interview_type
            + """
\nCreate a realistic preparation session with:
1. Likely questions
2. Strong answer frameworks
3. Technical questions
4. Behavioral questions
5. STAR method
6. Common mistakes
7. Questions to ask the interviewer
8. Scoring rubric
9. Final checklist

If this is Mock Interview mode, ask one question at a time.
"""
        )

        with st.spinner("Preparing your interview..."):
            try:
                result = ask_gemini(
                    prompt,
                    temperature=0.65,
                    max_tokens=6500,
                )
                st.markdown(result)
                save_result(result)
                result_actions("AI_Interview_Preparation.md")
                st.session_state.points += 20
            except Exception as e:
                show_error(e)


# ======================== LEARNING ROADMAP =====================

elif page == "🗺️ Learning Roadmap":
    st.subheader("🗺️ AI Learning Roadmap")

    skill = st.text_input(
        "Skill",
        placeholder="Artificial Intelligence",
    )

    goal = st.text_input(
        "Goal",
        placeholder="Build projects and get an internship",
    )

    hours = st.number_input(
        "Hours per week",
        1,
        50,
        10,
    )

    if st.button(
        "🗺️ Generate Roadmap",
        type="primary",
        use_container_width=True,
    ):
        prompt = (
            "Create a complete learning roadmap.\n"
            "Skill: "
            + skill
            + "\nGoal: "
            + goal
            + "\nHours/week: "
            + str(hours)
            + """
\nUse these stages:
1. Foundations
2. Core skills
3. Intermediate
4. Advanced
5. Projects
6. Portfolio
7. Interview preparation

For each stage include topics, practice, projects,
completion criteria and common mistakes.
Finish with a capstone project.
"""
        )

        with st.spinner("Building your roadmap..."):
            try:
                result = ask_gemini(
                    prompt,
                    temperature=0.55,
                    max_tokens=7000,
                )
                st.markdown(result)
                save_result(result)
                result_actions("AI_Learning_Roadmap.md")
                st.session_state.points += 20
            except Exception as e:
                show_error(e)


# ======================== MATERIAL ANALYZER ====================

elif page == "📄 Material Analyzer":
    st.subheader("📄 AI Study Material Analyzer")

    st.write(
        "Upload a PDF, image, text file or other supported study "
        "material and ask Gemini to summarize or analyze it."
    )

    uploaded = st.file_uploader(
        "Upload material",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "txt",
            "md",
        ],
    )

    instruction = st.text_area(
        "What should AI do?",
        height=150,
        placeholder=(
            "Summarize this material, identify important exam topics, "
            "and create 10 questions."
        ),
    )

    if uploaded:
        st.success("Uploaded: " + uploaded.name)

    if st.button(
        "🔍 Analyze Material",
        type="primary",
        use_container_width=True,
    ):
        if uploaded is None:
            st.warning("Upload a file first.")
        elif not instruction.strip():
            st.warning("Tell AI what you want it to do.")
        else:
            with st.spinner("Gemini is analyzing your material..."):
                try:
                    result = analyze_file(uploaded, instruction)
                    st.markdown(result)
                    save_result(result)
                    result_actions("AI_Material_Analysis.md")
                    st.session_state.points += 20
                except Exception as e:
                    show_error(e)


# ============================ ANALYTICS ========================

elif page == "📊 Analytics":
    st.subheader("📊 Your Learning Analytics")

    c1, c2, c3, c4 = st.columns(4)

    values = [
        ("Study time", str(st.session_state.study_minutes) + " min"),
        ("Points", str(st.session_state.points)),
        ("Topics", str(st.session_state.topics)),
        ("Flashcards", str(st.session_state.flashcards)),
    ]

    for col, (label, value) in zip((c1, c2, c3, c4), values):
        with col:
            st.metric(label, value)

    st.markdown("### 🎯 Daily Goal")

    daily_goal = goal
    progress = min(st.session_state.study_minutes / daily_goal, 1.0)
    st.progress(progress)

    if progress >= 1:
        st.success("🎉 You completed your daily goal!")
    else:
        remaining = daily_goal - st.session_state.study_minutes
        st.info(str(remaining) + " minutes remaining.")

    st.markdown("### 🧾 Learning Snapshot")

    st.markdown(
        f"""
        <div class="card">
            <b>Last topic:</b> {st.session_state.last_topic or "Not started"}<br><br>
            <b>Topics completed:</b> {st.session_state.topics}<br><br>
            <b>Quizzes completed:</b> {st.session_state.quizzes}<br><br>
            <b>Current streak:</b> 🔥 {st.session_state.streak} days
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================== FOCUS MODE ========================

elif page == "⚡ Focus Mode":
    st.subheader("⚡ AI Focus Session")

    topic = st.text_input(
        "Focus topic",
        placeholder="Machine Learning",
    )

    duration = st.slider(
        "Session duration (minutes)",
        15,
        120,
        30,
        5,
    )

    session_type = st.selectbox(
        "Session type",
        [
            "Learn",
            "Practice",
            "Revision",
            "Problem Solving",
            "Interview Preparation",
        ],
    )

    if st.button(
        "⚡ Create Focus Session",
        type="primary",
        use_container_width=True,
    ):
        if not topic.strip():
            st.warning("Enter a focus topic.")
        else:
            prompt = (
                "Create a "
                + str(duration)
                + "-minute focused study session for "
                + topic
                + ". Session type: "
                + session_type
                + """
\nInclude exact time blocks for:
1. Warm-up
2. Core learning
3. Active recall
4. Practice
5. Self-test
6. Final recap

Give concrete tasks for the learner.
"""
            )

            with st.spinner("Creating your focus session..."):
                try:
                    result = ask_gemini(
                        prompt,
                        temperature=0.45,
                        max_tokens=4500,
                    )
                    st.markdown(result)
                    save_result(result)
                    result_actions("AI_Focus_Session.md")
                    st.session_state.study_minutes += duration
                    st.session_state.points += duration
                    st.success(
                        str(duration)
                        + "-minute session added to your study time."
                    )
                except Exception as e:
                    show_error(e)


# ---------------------------- FOOTER ---------------------------

st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:35px 0 10px">
        <b>AI Study Buddy</b> · Python · Streamlit · Google Gemini
        <br>
        Learn smarter. Practice better. Build your future.
    </div>
    """,
    unsafe_allow_html=True,
)

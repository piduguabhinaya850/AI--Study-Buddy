import os
import json
import re
import time
from datetime import datetime, date, timedelta

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_NAME = "AI Study Buddy"
MODEL_NAME = "gemini-3.7-flash"

st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_api_key():
    """
    Supports both:
    1. .env -> GEMINI_API_KEY
    2. Streamlit Secrets -> GEMINI_API_KEY
    """

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None

    return key


API_KEY = get_api_key()


@st.cache_resource
def get_gemini_client(api_key):
    if not api_key:
        return None

    return genai.Client(api_key=api_key)


client = get_gemini_client(API_KEY)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "history": [],
    "study_points": 0,
    "topics_completed": 0,
    "quizzes_completed": 0,
    "flashcards_reviewed": 0,
    "daily_goal": 60,
    "study_minutes": 0,
    "streak": 1,
    "last_topic": "",
    "current_response": "",
    "uploaded_content": "",
    "quiz_data": [],
    "flashcards": [],
    "planner": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(99,102,241,0.08), transparent 35%),
            radial-gradient(circle at bottom right, rgba(14,165,233,0.08), transparent 35%),
            #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Headers */
    .hero {
        padding: 30px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #1e293b 45%,
                #312e81 100%
            );
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 20px 50px rgba(15,23,42,0.18);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 17px;
        color: #cbd5e1;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.9);
        padding: 22px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 30px rgba(15,23,42,0.06);
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(15,23,42,0.05);
        text-align: center;
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: #312e81;
    }

    .metric-label {
        color: #64748b;
        font-size: 14px;
    }

    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 8px 25px rgba(15,23,42,0.05);
    }

    .feature-card h3 {
        margin-top: 8px;
        color: #0f172a;
    }

    .feature-card p {
        color: #64748b;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-size: 12px;
        font-weight: 700;
    }

    .success-box {
        padding: 18px;
        border-radius: 15px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
    }

    .warning-box {
        padding: 18px;
        border-radius: 15px;
        background: #fffbeb;
        border: 1px solid #fde68a;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API ERROR HANDLER
# ============================================================

def show_api_error(error):
    message = str(error)

    if "401" in message or "API key" in message or "api_key" in message:
        st.error(
            "🔐 Gemini API authentication failed.\n\n"
            "Check your GEMINI_API_KEY in .env or Streamlit Secrets."
        )

    elif "429" in message or "quota" in message.lower():
        st.error(
            "⚡ Gemini API quota/rate limit reached. "
            "Please check your Gemini API usage and limits."
        )

    else:
        st.error(f"Gemini error: {message}")


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_ai(
    prompt,
    system_instruction=None,
    temperature=0.7,
    max_tokens=3000
):
    if not client:
        raise ValueError(
            "Gemini API key not found. Add GEMINI_API_KEY to .env "
            "or Streamlit Secrets."
        )

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    if system_instruction:
        config.system_instruction = system_instruction

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config
    )

    return response.text


# ============================================================
# MULTIMODAL GENERATION
# ============================================================

def analyze_uploaded_file(uploaded_file, prompt):
    if not client:
        raise ValueError("Gemini API key is missing.")

    file_bytes = uploaded_file.getvalue()

    mime_type = uploaded_file.type or "application/octet-stream"

    if mime_type.startswith("text/"):
        content = file_bytes.decode("utf-8", errors="ignore")

        final_prompt = f"""
You are an expert AI study assistant.

Analyze the following study material.

USER REQUEST:
{prompt}

STUDY MATERIAL:
{content}
"""

        return generate_ai(
            final_prompt,
            system_instruction=(
                "You are an academic tutor. "
                "Give accurate, structured, student-friendly answers."
            ),
            temperature=0.4,
            max_tokens=5000
        )

    part = types.Part.from_bytes(
        data=file_bytes,
        mime_type=mime_type
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            part,
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=5000
        )
    )

    return response.text


# ============================================================
# CHAT
# ============================================================

def ask_tutor(message):
    if not client:
        raise ValueError("Gemini API key is missing.")

    if "chat_object" not in st.session_state:
        st.session_state.chat_object = client.chats.create(
            model=MODEL_NAME
        )

    response = st.session_state.chat_object.send_message(
        message=message
    )

    return response.text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:10px;">
            <div style="font-size:45px;">🎓</div>
            <h2>AI Study Buddy</h2>
            <p style="color:#94a3b8;">Your Personal AI Learning OS</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
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
            "📄 Study Material Analyzer",
            "📊 Study Analytics",
            "⚡ Focus Mode"
        ]
    )

    st.divider()

    st.markdown("### 🎯 Today's Goal")

    daily_goal = st.slider(
        "Study minutes",
        15,
        300,
        st.session_state.daily_goal,
        15
    )

    st.session_state.daily_goal = daily_goal

    progress = min(
        st.session_state.study_minutes /
        max(daily_goal, 1),
        1.0
    )

    st.progress(progress)

    st.caption(
        f"{st.session_state.study_minutes} / "
        f"{daily_goal} minutes"
    )

    st.divider()

    if st.button("🔄 Reset Session", use_container_width=True):
        for key in DEFAULT_STATE:
            st.session_state[key] = DEFAULT_STATE[key]

        if "chat_object" in st.session_state:
            del st.session_state.chat_object

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎓 AI Study Buddy</h1>
        <p>
            Your intelligent learning companion for understanding,
            practicing, planning and mastering any subject.
        </p>
        <span class="badge">
            POWERED BY GOOGLE GEMINI
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.subheader("Welcome to your AI Study OS 👋")

    st.write(
        "Learn smarter with personalized explanations, quizzes, "
        "flashcards, study plans and AI-powered coaching."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {st.session_state.study_points}
                </div>
                <div class="metric-label">
                    Study Points
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {st.session_state.topics_completed}
                </div>
                <div class="metric-label">
                    Topics Completed
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {st.session_state.quizzes_completed}
                </div>
                <div class="metric-label">
                    Quizzes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    🔥 {st.session_state.streak}
                </div>
                <div class="metric-label">
                    Day Streak
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🚀 Learning Tools")

    features = [
        ("🤖", "AI Tutor", "Have an interactive conversation with your AI tutor."),
        ("📚", "Explain Topic", "Understand difficult concepts from beginner to advanced."),
        ("📝", "Smart Notes", "Transform long notes into structured revision material."),
        ("🧠", "Quiz Arena", "Generate adaptive quizzes and test your knowledge."),
        ("🃏", "Flashcards", "Create active-recall flashcards automatically."),
        ("📅", "Study Planner", "Build a personalized study schedule."),
        ("🎯", "Exam Mode", "Create an exam-focused preparation strategy."),
        ("💻", "Coding Mentor", "Debug, explain and improve programming solutions."),
        ("🎤", "Interview Coach", "Practice technical and HR interview questions."),
        ("🗺️", "Learning Roadmap", "Generate a complete path from beginner to advanced."),
        ("📄", "Material Analyzer", "Upload study material and ask AI questions."),
        ("⚡", "Focus Mode", "Run distraction-free AI-powered study sessions.")
    ]

    cols = st.columns(3)

    for index, (icon, title, description) in enumerate(features):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="feature-card">
                    <div style="font-size:32px;">{icon}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# AI TUTOR
# ============================================================

elif page == "🤖 AI Tutor":

    st.subheader("🤖 Personal AI Tutor")

    st.write(
        "Ask follow-up questions naturally. Your conversation stays "
        "available during the current session."
    )

    if st.session_state.messages:

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_message = st.chat_input(
        "Ask your AI tutor anything..."
    )

    if user_message:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = ask_tutor(user_message)

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    st.session_state.study_points += 5

                except Exception as e:
                    show_api_error(e)


# ============================================================
# EXPLAIN TOPIC
# ============================================================

elif page == "📚 Explain Topic":

    st.subheader("📚 Explain Any Topic")

    col1, col2 = st.columns([2, 1])

    with col1:

        topic = st.text_input(
            "What do you want to learn?",
            placeholder="Example: Neural Networks"
        )

    with col2:

        level = st.selectbox(
            "Learning level",
            [
                "Beginner",
                "Intermediate",
                "Advanced",
                "Interview Level"
            ]
        )

    explanation_style = st.selectbox(
        "Explanation style",
        [
            "Simple and intuitive",
            "Detailed academic explanation",
            "Real-world examples",
            "Exam-oriented",
            "Interview-oriented"
        ]
    )

    if st.button(
        "✨ Generate Explanation",
        type="primary",
        use_container_width=True
    ):

        if not topic.strip():

            st.warning("Enter a topic first.")

        else:

            prompt = f"""
Explain the topic: {topic}

Student level: {level}
Explanation style: {explanation_style}

Create an exceptionally useful learning lesson.

Include:

1. What is it?
2. Why does it matter?
3. Simple intuition
4. Core concepts
5. Step-by-step explanation
6. Real-world example
7. Technical example
8. Common mistakes
9. Interview questions
10. Quick revision summary
11. Three questions to test understanding

Use clear headings and Markdown.
"""

            with st.spinner("Building your lesson..."):

                try:

                    result = generate_ai(
                        prompt,
                        system_instruction=(
                            "You are an expert university professor "
                            "and personal AI tutor."
                        ),
                        temperature=0.5,
                        max_tokens=5000
                    )

                    st.markdown(result)

                    st.session_state.current_response = result
                    st.session_state.last_topic = topic
                    st.session_state.topics_completed += 1
                    st.session_state.study_points += 10

                except Exception as e:
                    show_api_error(e)


# ============================================================
# SMART NOTES
# ============================================================

elif page == "📝 Smart Notes":

    st.subheader("📝 Smart Notes Generator")

    notes = st.text_area(
        "Paste your notes",
        height=300,
        placeholder="Paste your lecture notes here..."
    )

    col1, col2 = st.columns(2)

    with col1:
        output_type = st.selectbox(
            "Output",
            [
                "Complete Summary",
                "Exam Notes",
                "One Page Revision",
                "Key Points",
                "Mind Map Structure",
                "Cheat Sheet"
            ]
        )

    with col2:
        detail = st.select_slider(
            "Detail",
            options=[
                "Short",
                "Balanced",
                "Detailed"
            ],
            value="Balanced"
        )

    if st.button(
        "🧠 Transform Notes",
        type="primary",
        use_container_width=True
    ):

        if not notes.strip():

            st.warning("Paste some notes first.")

        else:

            prompt = f"""
Transform these study notes into:

OUTPUT TYPE:
{output_type}

DETAIL LEVEL:
{detail}

Requirements:

- Preserve important facts.
- Remove unnecessary repetition.
- Explain difficult concepts.
- Use headings.
- Use bullet points.
- Highlight definitions.
- Include formulas where relevant.
- Add memory tricks where useful.
- End with a rapid revision section.

NOTES:

{notes}
"""

            with st.spinner("Creating smart notes..."):

                try:

                    result = generate_ai(
                        prompt,
                        temperature=0.4,
                        max_tokens=5000
                    )

                    st.markdown(result)

                    st.download_button(
                        "📥 Download Notes",
                        data=result,
                        file_name="AI_Study_Notes.md",
                        mime="text/markdown"
                    )

                    st.session_state.study_points += 10

                except Exception as e:
                    show_api_error(e)


# ============================================================
# QUIZ ARENA
# ============================================================

elif page == "🧠 Quiz Arena":

    st.subheader("🧠 AI Quiz Arena")

    col1, col2, col3 = st.columns(3)

    with col1:

        quiz_topic = st.text_input(
            "Topic",
            placeholder="Python, SQL, Machine Learning..."
        )

    with col2:

        difficulty = st.selectbox(
            "Difficulty",
            [
                "Easy",
                "Medium",
                "Hard",
                "Expert"
            ]
        )

    with col3:

        number_questions = st.slider(
            "Questions",
            3,
            15,
            5
        )

    if st.button(
        "🎮 Generate Quiz",
        type="primary",
        use_container_width=True
    ):

        if not quiz_topic.strip():

            st.warning("Enter a topic.")

        else:

            prompt = f"""
Create a {difficulty} level quiz about:

{quiz_topic}

Generate exactly {number_questions} questions.

For each question provide:

QUESTION:
A multiple-choice question.

OPTIONS:
A.
B.
C.
D.

CORRECT ANSWER:
A/B/C/D

EXPLANATION:
Explain why the answer is correct.

Do not reveal the answer before the question.
Format cleanly using Markdown.
"""

            with st.spinner("Generating your challenge..."):

                try:

                    result = generate_ai(
                        prompt,
                        temperature=0.6,
                        max_tokens=5000
                    )

                    st.markdown(result)

                    st.session_state.quizzes_completed += 1
                    st.session_state.study_points += 15

                    st.download_button(
                        "📥 Download Quiz",
                        data=result,
                        file_name="AI_Quiz.md",
                        mime="text/markdown"
                    )

                except Exception as e:
                    show_api_error(e)


# ============================================================
# FLASHCARDS
# ============================================================

elif page == "🃏 Flashcards":

    st.subheader("🃏 AI Flashcard Generator")

    topic = st.text_input(
        "Topic",
        placeholder="Example: SQL Joins"
    )

    count = st.slider(
        "Number of flashcards",
        5,
        30,
        10
    )

    if st.button(
        "✨ Create Flashcards",
        type="primary",
        use_container_width=True
    ):

        if not topic.strip():

            st.warning("Enter a topic.")

        else:

            prompt = f"""
Create {count} high-quality active-recall flashcards about:

{topic}

Format:

CARD 1
FRONT: Question
BACK: Answer

CARD 2
FRONT: Question
BACK: Answer

Make questions progressively more difficult.
Focus on concepts rather than memorization only.
"""

            with st.spinner("Creating flashcards..."):

                try:

                    result = generate_ai(
                        prompt,
                        temperature=0.6,
                        max_tokens=5000
                    )

                    st.markdown(result)

                    st.session_state.flashcards_reviewed += count
                    st.session_state.study_points += count

                    st.download_button(
                        "📥 Download Flashcards",
                        data=result,
                        file_name="AI_Flashcards.md",
                        mime="text/markdown"
                    )

                except Exception as e:
                    show_api_error(e)


# ============================================================
# STUDY PLANNER
# ============================================================

elif page == "📅 Study Planner":

    st.subheader("📅 Personalized AI Study Planner")

    col1, col2 = st.columns(2)

    with col1:

        subjects = st.text_area(
            "Subjects",
            placeholder=(
                "Python\nSQL\nMachine Learning\n"
                "Data Structures\nPower BI"
            ),
            height=180
        )

        available_hours = st.number_input(
            "Study hours per day",
            min_value=0.5,
            max_value=12.0,
            value=2.0,
            step=0.5
        )

    with col2:

        exam_date = st.date_input(
            "Exam / target date",
            value=date.today() + timedelta(days=30)
        )

        current_level = st.selectbox(
            "Current level",
            [
                "Beginner",
                "Intermediate",
                "Advanced"
            ]
        )

        priority = st.selectbox(
            "Priority",
            [
                "Balanced",
                "Weak subjects first",
                "Exam focused",
                "Career focused"
            ]
        )

    if st.button(
        "📅 Build My Study Plan",
        type="primary",
        use_container_width=True
    ):

        if not subjects.strip():

            st.warning("Enter at least one subject.")

        else:

            days = max(
                (exam_date - date.today()).days,
                1
            )

            prompt = f"""
Create a personalized study plan.

SUBJECTS:
{subjects}

DAYS AVAILABLE:
{days}

HOURS PER DAY:
{available_hours}

CURRENT LEVEL:
{current_level}

PRIORITY:
{priority}

Design an intelligent plan including:

1. Daily schedule
2. Subject allocation
3. Theory sessions
4. Practice sessions
5. Revision cycles
6. Mock tests
7. Weak-topic strategy
8. Final revision
9. Daily measurable goals
10. Weekly checkpoints

Use a realistic schedule.
"""

            with st.spinner("Designing your study system..."):

                try:

                    result = generate_ai(
                        prompt,
                        temperature=0.5,
                        max_tokens=6000
                    )

                    st.markdown(result)

                    st.session_state.planner = result
                    st.session_state.study_points += 20

                    st.download_button(
                        "📥 Download Study Plan",
                        data=result,
                        file_name="AI_Study_Plan.md",
                        mime="text/markdown"
                    )

                except Exception as e:
                    show_api_error(e)


# ============================================================
# EXAM MODE
# ============================================================

elif page == "🎯 Exam Mode":

    st.subheader("🎯 Exam Preparation Intelligence")

    exam = st.text_input(
        "Exam name",
        placeholder="Example: GATE / Placement / University Exam"
    )

    subjects = st.text_area(
        "Syllabus / subjects",
        height=150
    )

    days_left = st.number_input(
        "Days remaining",
        min_value=1,
        max_value=365,
        value=30
    )

    if st.button(
        "🚀 Activate Exam Mode",
        type="primary",
        use_container_width=True
    ):

        prompt = f"""
Act as an elite exam preparation strategist.

EXAM:
{exam}

SYLLABUS:
{subjects}

DAYS LEFT:
{days_left}

Create an exam intelligence report containing:

1. Priority matrix
2. High-value topics
3. Topics to study first
4. Topics to revise
5. Daily schedule
6. Practice strategy
7. Mock test strategy
8. Revision cycle
9. Common mistakes
10. Last 7-day strategy
11. Exam-day strategy
12. Confidence-building checklist

Be practical and realistic.
"""

        with st.spinner("Analyzing your exam strategy..."):

            try:

                result = generate_ai(
                    prompt,
                    temperature=0.45,
                    max_tokens=6000
                )

                st.markdown(result)

                st.session_state.study_points += 25

            except Exception as e:
                show_api_error(e)


# ============================================================
# CODING MENTOR
# ============================================================

elif page == "💻 Coding Mentor":

    st.subheader("💻 AI Coding Mentor")

    language = st.selectbox(
        "Programming language",
        [
            "Python",
            "Java",
            "C",
            "C++",
            "JavaScript",
            "SQL",
            "Other"
        ]
    )

    code = st.text_area(
        "Paste your code",
        height=300,
        placeholder="Paste your code here..."
    )

    task = st.selectbox(
        "What do you want?",
        [
            "Explain this code",
            "Find bugs",
            "Optimize this code",
            "Improve code quality",
            "Convert to another approach",
            "Generate test cases",
            "Prepare interview questions"
        ]
    )

    if st.button(
        "🧑‍💻 Analyze Code",
        type="primary",
        use_container_width=True
    ):

        if not code.strip():

            st.warning("Paste code first.")

        else:

            prompt = f"""
You are an expert software engineer and coding mentor.

LANGUAGE:
{language}

TASK:
{task}

CODE:

```{language}
{code}

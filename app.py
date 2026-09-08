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

# Optional provider imports
try:
    from google import genai
    from google.genai import types as gtypes
except Exception:
    genai = None
    gtypes = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import requests
except Exception:
    requests = None

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

APP_VERSION = "2.0.0"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

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
    "provider_stats": {},
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
.provider-pill {
    padding:8px 12px; border-radius:999px; background:#eef2ff;
    color:#312e81; font-weight:700; display:inline-block;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------- Provider utilities ----------------------

def env(name):
    return os.getenv(name, "").strip()

def available_providers():
    providers = []
    if env("GEMINI_API_KEY") and genai:
        providers.append("Google Gemini")
    if env("OPENAI_API_KEY") and OpenAI:
        providers.append("OpenAI")
    if env("OPENROUTER_API_KEY") and requests:
        providers.append("OpenRouter")
    if env("GROQ_API_KEY") and requests:
        providers.append("Groq")
    if requests:
        providers.append("Ollama")
    return providers

@st.cache_resource(show_spinner=False)
def get_gemini_client(key):
    if not key or not genai:
        return None
    return genai.Client(api_key=key)

@st.cache_resource(show_spinner=False)
def get_openai_client(key):
    if not key or not OpenAI:
        return None
    return OpenAI(api_key=key)

def provider_error(provider):
    return RuntimeError(
        f"{provider} is not configured. Add the required API key in .env "
        "or Streamlit Secrets, then restart the app."
    )

def call_gemini(prompt, system="", temperature=0.6, max_tokens=5000):
    client = get_gemini_client(env("GEMINI_API_KEY"))
    if not client:
        raise provider_error("Google Gemini")
    config = gtypes.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system or None,
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text

def call_openai(prompt, system="", temperature=0.6, max_tokens=5000):
    client = get_openai_client(env("OPENAI_API_KEY"))
    if not client:
        raise provider_error("OpenAI")
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system or None,
        input=prompt,
        max_output_tokens=max_tokens,
    )
    text = getattr(response, "output_text", None)
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")
    return text

def call_openai_compatible(base_url, key, model, prompt, system="", temperature=0.6, max_tokens=5000):
    if not requests:
        raise RuntimeError("requests is not installed.")
    if not key:
        raise RuntimeError("Provider API key is missing.")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Provider HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError("Provider returned an unexpected response.")

def call_ollama(prompt, system="", temperature=0.6, max_tokens=5000):
    if not requests:
        raise RuntimeError("requests is not installed.")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system or "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = requests.post(OLLAMA_URL.rstrip("/") + "/api/chat", json=payload, timeout=180)
    if not r.ok:
        raise RuntimeError(
            "Ollama is not running or the model is unavailable. "
            "Start Ollama and pull the selected model."
        )
    data = r.json()
    return data.get("message", {}).get("content", "")

def call_provider(provider, prompt, system="", temperature=0.6, max_tokens=5000):
    started = time.perf_counter()
    if provider == "Google Gemini":
        text = call_gemini(prompt, system, temperature, max_tokens)
    elif provider == "OpenAI":
        text = call_openai(prompt, system, temperature, max_tokens)
    elif provider == "OpenRouter":
        text = call_openai_compatible(
            "https://openrouter.ai/api/v1", env("OPENROUTER_API_KEY"),
            OPENROUTER_MODEL, prompt, system, temperature, max_tokens
        )
    elif provider == "Groq":
        text = call_openai_compatible(
            "https://api.groq.com/openai/v1", env("GROQ_API_KEY"),
            env("GROQ_MODEL") or "llama-3.3-70b-versatile",
            prompt, system, temperature, max_tokens
        )
    elif provider == "Ollama":
        text = call_ollama(prompt, system, temperature, max_tokens)
    else:
        raise RuntimeError("Unknown AI provider.")
    elapsed = round(time.perf_counter() - started, 2)
    stats = st.session_state.provider_stats.setdefault(provider, {"calls": 0, "seconds": 0})
    stats["calls"] += 1
    stats["seconds"] += elapsed
    st.session_state.last_provider = provider
    return text

def ask_ai(prompt, system="", provider=None, temperature=0.6, max_tokens=5000):
    providers = available_providers()
    if not providers:
        raise RuntimeError(
            "No AI provider is configured. Add GEMINI_API_KEY or OPENAI_API_KEY "
            "(or another supported provider) to .env/Streamlit Secrets."
        )
    chosen = provider if provider and provider != "Auto" else providers[0]
    if chosen not in providers:
        chosen = providers[0]
    try:
        return call_provider(chosen, prompt, system, temperature, max_tokens)
    except Exception as first_error:
        if provider in (None, "Auto") and len(providers) > 1:
            for fallback in providers:
                if fallback == chosen:
                    continue
                try:
                    return call_provider(fallback, prompt, system, temperature, max_tokens)
                except Exception:
                    continue
        raise first_error

def compare_models(prompt, providers, system=""):
    results = []
    for p in providers:
        try:
            started = time.perf_counter()
            answer = call_provider(p, prompt, system, 0.5, 4000)
            elapsed = round(time.perf_counter() - started, 2)
            results.append({"provider": p, "answer": answer, "seconds": elapsed, "ok": True})
        except Exception as e:
            results.append({"provider": p, "answer": str(e), "seconds": 0, "ok": False})
    return results

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
    if "401" in msg or "API key" in msg or "api_key" in msg:
        st.error("🔐 Authentication failed. Check your provider API key.")
    elif "429" in msg or "quota" in msg.lower():
        st.error("⚡ Rate limit/quota reached. Try another provider or later.")
    else:
        st.error("Something went wrong: " + msg)

def run_ai_button(label, prompt, system="", provider=None, temperature=0.5,
                  max_tokens=6000, points=10, topic=None, filename="AI_Result.md"):
    if st.button(label, type="primary", use_container_width=True):
        with st.spinner("AI is working..."):
            try:
                result = ask_ai(prompt, system, provider, temperature, max_tokens)
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

def analyze_uploaded_file(uploaded, instruction, provider):
    suffix = os.path.splitext(uploaded.name)[1].lower()
    if provider == "Google Gemini":
        client = get_gemini_client(env("GEMINI_API_KEY"))
        if not client:
            raise provider_error("Google Gemini")
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
            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            return response.text
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # OpenAI supports direct file input through the Responses API.
    if provider == "OpenAI":
        client = get_openai_client(env("OPENAI_API_KEY"))
        if not client:
            raise provider_error("OpenAI")
        suffix = suffix or ".bin"
        mime = uploaded.type or "application/octet-stream"
        import base64
        encoded = base64.b64encode(uploaded.getvalue()).decode("utf-8")
        data_url = f"data:{mime};base64,{encoded}"
        content_type = "input_file"
        if mime.startswith("image/"):
            content_type = "input_image"
        content = [{"type": "input_text", "text": instruction}]
        if content_type == "input_image":
            content.append({"type": "input_image", "image_url": data_url})
        else:
            content.append({
                "type": "input_file",
                "filename": uploaded.name,
                "file_data": data_url,
            })
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": content}],
            max_output_tokens=8000,
        )
        return response.output_text

    # Universal fallback: text files can be sent to compatible providers.
    raw = uploaded.getvalue()
    if suffix in {".txt", ".md", ".csv", ".json", ".py", ".sql"}:
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = str(raw)
        return ask_ai(
            instruction + "\n\nFILE CONTENT:\n" + text[:120000],
            provider=provider,
            max_tokens=8000,
        )
    raise RuntimeError(
        "This provider supports text fallback only. Use Google Gemini or OpenAI "
        "for PDF/image document analysis."
    )

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

    providers = available_providers()
    provider_options = ["Auto"] + providers
    provider = st.selectbox("🤖 AI Engine", provider_options)
    st.caption("Auto uses the first configured provider and falls back when possible.")

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
        <span class="badge">MULTI-AI • GEMINI • OPENAI • OPENROUTER • GROQ • OLLAMA</span>
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
                    result = ask_ai(workflow_prompt, provider=provider, max_tokens=7000)
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
                        provider=provider,
                        max_tokens=6000,
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    award(5)
                except Exception as e:
                    show_error(e)

# =========================== MODEL LAB ===========================

elif page == "🧠 AI Model Lab":
    st.subheader("🧠 Multi-AI Model Laboratory")
    st.write("Compare configured AI providers on the same task.")
    if len(providers) < 2:
        st.info("Configure at least two providers to compare models.")
    prompt = st.text_area(
        "Comparison task",
        height=150,
        placeholder="Explain machine learning to a beginner and give a practical example.",
    )
    selected = st.multiselect(
        "Models/providers to compare",
        providers,
        default=providers[:2],
    )
    if st.button("⚖️ Compare AI Responses", type="primary", use_container_width=True):
        if not prompt.strip() or len(selected) < 2:
            st.warning("Enter a task and select at least two providers.")
        else:
            with st.spinner("Running the same task across models..."):
                results = compare_models(prompt, selected)
            for item in results:
                st.markdown(f"### {item['provider']} · {item['seconds']}s")
                if item["ok"]:
                    st.markdown(item["answer"])
                else:
                    st.error(item["answer"])
            good = [r for r in results if r["ok"]]
            if len(good) >= 2:
                with st.spinner("AI judge is comparing the responses..."):
                    judge = ask_ai(
                        "Compare these AI answers for correctness, clarity, completeness and usefulness. "
                        "Recommend the strongest answer and explain why.\n\n" +
                        "\n\n".join(f"MODEL {i+1} ({r['provider']}):\n{r['answer']}" for i, r in enumerate(good)),
                        provider=provider,
                        max_tokens=5000,
                    )
                st.markdown("### 🏆 AI Judge")
                st.markdown(judge)
                save_result(judge)
                download_result("AI_Model_Comparison.md")

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
        system="You are an expert university tutor. Be accurate and never invent facts.",
        provider=provider, temperature=0.5, max_tokens=7000,
        points=10, topic=topic, filename="AI_Lesson.md"
    )

# =========================== SMART NOTES =========================

elif page == "📝 Smart Notes":
    st.subheader("📝 Smart Notes Studio")
    notes = st.text_area("Paste notes", height=320)
    output = st.selectbox(
        "Output",
        ["Complete Summary", "Exam Notes", "One-Page Revision", "Key Points",
         "Mind Map Structure", "Cheat Sheet", "Question Bank"],
    )
    detail = st.select_slider("Detail", ["Short", "Balanced", "Detailed"], value="Balanced")
    prompt = f"""
Transform these notes into {output}. Detail level: {detail}.
Preserve important facts, remove repetition, explain difficult terms,
use headings and bullets, include formulas when relevant, add memory tricks,
and finish with rapid revision questions.

NOTES:
{notes}
"""
    run_ai_button(
        "🧠 Transform Notes", prompt, provider=provider,
        temperature=0.4, max_tokens=7000, points=10,
        filename="AI_Study_Notes.md"
    )

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
        "🎮 Generate Quiz", prompt, provider=provider, temperature=0.6,
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
                    result = ask_ai(prompt, provider=provider, temperature=0.6, max_tokens=7000)
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
        "📅 Build My Plan", prompt, provider=provider, temperature=0.5,
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
        "🚀 Activate Exam Mode", prompt, provider=provider, temperature=0.45,
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
        system="Be technically precise. Never claim code was executed unless it was.",
        provider=provider, temperature=0.35, max_tokens=9000,
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
                        result = ask_ai(prompt, provider=provider, max_tokens=8000)
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
            chosen = provider if provider != "Auto" else (
                "Google Gemini" if "Google Gemini" in providers else providers[0] if providers else None
            )
            if not chosen:
                st.error("Configure an AI provider first.")
            else:
                with st.spinner("Analyzing your material..."):
                    try:
                        result = analyze_uploaded_file(uploaded, instruction, chosen)
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
                result = ask_ai(prompt, provider=provider, temperature=0.6, max_tokens=5000)
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
        "🚀 Run Career Analysis", prompt, provider=provider,
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
        "🏗️ Architect My Project", prompt, provider=provider,
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
        "🔬 Build Research Plan", prompt, provider=provider,
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
        "🧭 Generate Skill Graph", prompt, provider=provider,
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
                    result = ask_ai(prompt, provider=provider, temperature=0.45, max_tokens=5000)
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

    st.markdown("### Provider Performance")
    if st.session_state.provider_stats:
        rows = []
        for name, stats in st.session_state.provider_stats.items():
            calls = stats["calls"]
            rows.append({
                "Provider": name,
                "Calls": calls,
                "Total seconds": round(stats["seconds"], 2),
                "Avg seconds": round(stats["seconds"] / max(calls, 1), 2),
            })
        if pd:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Provider usage will appear after AI calls.")

    st.markdown("### Learning Snapshot")
    st.markdown(
        f"""
        <div class="card">
        <b>Last topic:</b> {st.session_state.last_topic or "Not started"}<br><br>
        <b>Topics:</b> {st.session_state.topics}<br><br>
        <b>Quizzes:</b> {st.session_state.quizzes}<br><br>
        <b>Documents:</b> {len(st.session_state.documents)}<br><br>
        <b>Streak:</b> 🔥 {st.session_state.streak} days<br><br>
        <b>Last AI provider:</b> {st.session_state.last_provider or "None"}
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
    st.markdown("### Configured providers")
    for p in ["Google Gemini", "OpenAI", "OpenRouter", "Groq", "Ollama"]:
        st.write(("🟢" if p in providers else "⚪") + " " + p)

    st.markdown("### Environment variables")
    st.code(
        """GEMINI_API_KEY=
OPENAI_API_KEY=
OPENROUTER_API_KEY=
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_MODEL=gpt-5
GEMINI_MODEL=gemini-3.7-flash
OPENROUTER_MODEL=openai/gpt-5
OLLAMA_MODEL=llama3.2
OLLAMA_URL=http://localhost:11434
""",
        language="text",
    )
    st.warning(
        "Never paste API keys into source code or commit them to GitHub. "
        "Use .env locally and Streamlit Secrets when deployed."
    )

# ---------------------------- Footer ----------------------------

st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:35px 0 10px">
        <b>AI Study Buddy 360</b> · Multi-AI Learning · Career · Work OS
        <br>
        Learn smarter · Build faster · Grow continuously
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("API Key not found. Add it in .env file")
    st.stop()

client = OpenAI(api_key=api_key)

# Page config
st.set_page_config(page_title="AI Study Buddy", page_icon="📚")

st.title("📚 AI Study Buddy")
st.write("Your AI-powered learning assistant")

# Feature selection
option = st.selectbox(
    "Choose what you want:",
    [
        "Explain Topic",
        "Summarize Notes",
        "Generate Quiz",
        "Ask Anything"
    ]
)

# User input
user_input = st.text_area("Enter text or question:")

# Generate button
if st.button("Generate"):

    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating... 🤔"):

            # Prompt selection
            if option == "Explain Topic":
                prompt = f"Explain this topic clearly in simple words:\n{user_input}"

            elif option == "Summarize Notes":
                prompt = f"Summarize these notes in easy points:\n{user_input}"

            elif option == "Generate Quiz":
                prompt = f"Create 5 quiz questions with answers from:\n{user_input}"

            else:
                prompt = user_input

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI study assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=600
                )

                answer = response.choices[0].message.content
                st.success("Done ✅")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")
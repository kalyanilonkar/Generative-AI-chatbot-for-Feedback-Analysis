
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime, timedelta
import random

# ------------------ Hugging Face API Setup ------------------
HF_API_TOKEN = "YOUR_HF_TOKEN"
HF_MODEL_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# ------------------ Simulated Feedback ------------------
feedback_comments = [
    "The lecture was too fast and hard to follow.",
    "Excellent session! Really appreciated the examples.",
    "Slides were unclear, and the topic was confusing.",
    "Loved the interactive parts of the class!",
    "I didn't understand the main concepts explained.",
    "The instructor did a great job explaining difficult material.",
    "The pace was good and the session was engaging.",
    "Too much theory, not enough practical examples.",
    "Great use of visuals and real-world examples.",
    "Felt like we were rushed through key topics.",
    "Perfectly structured class, I learned a lot.",
    "More time should have been spent on Q&A.",
    "The tutor was very supportive and helpful.",
    "Hard to hear the instructor in the video.",
    "Really interesting content, especially the case study.",
    "The session lacked engagement and energy.",
    "I enjoyed the group activity the most.",
    "Lecture materials were not uploaded in time.",
    "Clear explanations and helpful analogies.",
    "The content was repetitive and dull."
]
session_ids = [f"Session_{i % 5 + 1}" for i in range(len(feedback_comments))]

# ------------------ Sentiment Function ------------------
def get_star_rating(text):
    try:
        response = requests.post(HF_MODEL_URL, headers=HEADERS, json={"inputs": text})
        if response.status_code == 200:
            predictions = response.json()
            if isinstance(predictions, list) and isinstance(predictions[0], list):
                best = max(predictions[0], key=lambda x: x["score"])
                return best["label"]
            else:
                return "Invalid"
        else:
            return "Error"
    except Exception:
        return "Error"

# ------------------ Streamlit UI ------------------
col1, col2 = st.columns([1, 6])
with col1:
    st.image("assets/chatbot.png", width=68)
with col2:
    st.markdown("<h1 style='color:#fc1292;'>Instructor Feedback Chatbot</h1>", unsafe_allow_html=True)

# ------------------ Session State Init ------------------
if "star_ratings" not in st.session_state:
    st.session_state["star_ratings"] = [get_star_rating(text) for text in feedback_comments]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ------------------ DataFrame and CSV ------------------
df_feedback = pd.DataFrame({
    "session_id": session_ids,
    "comment": feedback_comments,
    "sentiment": st.session_state["star_ratings"]
})
base_time = datetime.now()
df_feedback["timestamp"] = [base_time - timedelta(minutes=random.randint(0,120)) for _ in feedback_comments]
df_feedback.to_csv("output/feedback_sentiment_results.csv", index=False)

# ------------------ Chat Input ------------------
user_input = st.chat_input("💬 Ask about feedback:")
if user_input:
    valid_stars = [int(s[0]) for s in st.session_state["star_ratings"] if s and s[0].isdigit()]
    star_counts = Counter(valid_stars)
    avg = sum(valid_stars) / len(valid_stars) if valid_stars else 0
    reply = f"The average sentiment is around *{avg:.1f} stars* based on {len(valid_stars)} comments."
    st.session_state["chat_history"].append(("👨‍🏫 Instructor", user_input))
    st.session_state["chat_history"].append(("🤖 Bot", reply))

# ------------------ Display Chat ------------------
for speaker, msg in st.session_state["chat_history"]:
    bg = "#f1f1f1" if "Instructor" in speaker else "#e6f7ff"
    st.markdown(f"<div style='background-color:{bg};padding:10px;border-radius:10px;margin-bottom:5px'><b>{speaker}:</b> {msg}</div>", unsafe_allow_html=True)

# ------------------ Optional Chart ------------------
if user_input:
    st.markdown("### 📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    star_labels = [1,2,3,4,5]
    star_values = [star_counts.get(i,0) for i in star_labels]
    ax.bar(star_labels, star_values, color="skyblue")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Comments")
    st.pyplot(fig)

# ------------------ Link to Dashboard ------------------
st.markdown(
    '''
    <div style='text-align:center;margin-top:20px;'>
        <a href="https://drive.google.com/file/d/your-dashboard-link/view" target="_blank">
            <button style='background:#1a73e8;color:white;padding:10px 20px;border:none;border-radius:5px;font-size:16px;'>
                📊 View Dashboard Summary (PDF)
            </button>
        </a>
    </div>
    ''',
    unsafe_allow_html=True
)

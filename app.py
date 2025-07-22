import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ---------- Set API key from Streamlit Secrets ----------
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------- Page Config ----------
st.set_page_config(page_title="AI Company Insights Assistant 🤖", page_icon="📊", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        body {
            background-color: white;
            font-family: 'Arial', sans-serif;
            color: #333;
        }
        .header-title {
            font-size: 40px;
            text-align: center;
            color: #1B2A4E;
            margin-top: 30px;
            font-weight: bold;
        }
        .header-subtitle {
            font-size: 18px;
            text-align: center;
            color: #607D8B;
            margin-top: 10px;
        }
        .footer {
            text-align: center;
            padding: 15px;
            background-color: #1B2A4E;
            color: white;
            font-size: 16px;
            position: fixed;
            width: 100%;
            bottom: 0;
            font-weight: bold;
        }
        .user-message {
            background-color: #D6F0FF;
            padding: 12px;
            border-radius: 10px;
            font-size: 16px;
            color: #0D1B2A;
        }
        .ai-message {
            background-color: #EDEDED;
            padding: 12px;
            border-radius: 10px;
            font-size: 16px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="header-title">📊 AI Company Insights Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Built for Vallum Capital | Powered by Gemini</div>', unsafe_allow_html=True)

# ---------- Gemini Model Setup ----------
gemini = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-001',
    temperature=0.1,
    convert_system_message_to_human=True
)

# ---------- System Prompt ----------
SYS_PROMPT = """
You are a company insights assistant that helps users identify and summarize companies that have been fully acquired, sold, delisted, merged, or shut down in the last 1 year.

Your responsibilities:
1. Confirm whether a company was fully sold, acquired, merged, delisted, or shut down.
2. Provide the reason behind the event (e.g., merger, privatization, strategic acquisition, financial distress).
3. Give a concise company profile, including:
   - Sector/Industry
   - Founders or Ownership
   - Location
   - Core Products or Services
4. If available, provide:
   - Name of the acquiring or merging entity
   - Date of acquisition/delisting
   - Notable strategic motive

When a user asks broad questions like:
- “Which companies were delisted last year?”
- “Which companies were acquired recently?”
Return a list of relevant companies with:
- ✅ Company Name
- ✅ Event Type
- ✅ Date
- ✅ Reason (if available)

Be friendly, conversational, and informative. 
If the user follows up, give in-depth insights.
"""

# ---------- Prompt Template ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm_chain = prompt | gemini
streamlit_msg_history = StreamlitChatMessageHistory()

conversation_chain = RunnableWithMessageHistory(
    llm_chain,
    lambda session_id: streamlit_msg_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------- Display Chat History ----------
for msg in streamlit_msg_history.messages:
    with st.chat_message(msg.type):
        if msg.type == "human":
            st.markdown(f'<div class="user-message">{msg.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-message">{msg.content}</div>', unsafe_allow_html=True)

# ---------- Chat Input ----------
user_prompt = st.chat_input("Ask about acquisitions, delistings, or any company insights...")
if user_prompt:
    st.chat_message("human").markdown(user_prompt)
    with st.chat_message("ai"):
        try:
            config = {"configurable": {"session_id": "any"}}
            response = conversation_chain.invoke({"input": user_prompt}, config)
            content = response.content if hasattr(response, "content") else str(response)
            st.markdown(f'<div class="ai-message">{content}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")

# ---------- Footer ----------
st.markdown('<div class="footer">Developed by Yash Samir Modi | Vallum Capital Internship Project</div>', unsafe_allow_html=True)

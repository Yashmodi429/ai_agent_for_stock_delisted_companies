import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# --- Set Gemini API key from Streamlit secrets ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# --- Page config ---
st.set_page_config(page_title="📊 Company Insights Bot", page_icon="📈")

# --- Custom CSS ---
st.markdown("""
<style>
.header { text-align:center; font-size: 36px; color: #2c3e50; margin-top: 20px; font-weight:bold; }
.subheader { text-align:center; font-size: 16px; color: #7f8c8d; margin-bottom: 30px; }
.user-message { background-color: #ecf0f1; padding: 10px; border-radius: 10px; margin: 10px 0; }
.ai-message { background-color: #dff9fb; padding: 10px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header">📊 Company Insights Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Built for Vallum Capital | Powered by Gemini</div>', unsafe_allow_html=True)

# --- System Prompt ---
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

# --- Gemini LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.2,
    convert_system_message_to_human=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT),
    ("human", "{input}")
])

# --- Chat History Using Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Display Chat History ---
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div class='user-message'>🧑‍💼 {msg.content}</div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f"<div class='ai-message'>🤖 {msg.content}</div>", unsafe_allow_html=True)

# --- Chat Input Box ---
user_input = st.chat_input("Ask about companies acquired, merged, or delisted recently...")
if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.spinner("🔍 Analyzing company data..."):
        chain = prompt | llm
        response = chain.invoke({"input": user_input})
        st.session_state.history.append(AIMessage(content=response.content))
        st.rerun()

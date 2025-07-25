import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- Page config ---
st.set_page_config(page_title="📊 Company Insights Bot", page_icon="📈")

# --- Set Gemini API Key ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# --- Gemini LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro",
    temperature=0.2,
    convert_system_message_to_human=True
)

# --- System Prompt ---
SYS_PROMPT = """
You are a Company Insights Assistant specialized in Indian public companies (NSE/BSE).
You help users confirm if a company was:
- Acquired (100% ownership change)
- Merged
- Delisted (NSE/BSE)
- Shut down
- Privatized

You must:
- Use only real events (from past 10 years)
- Never guess or make up
- Always cite official sources (SEBI/BSE/NSE, news, filings)
- Support both list queries and follow-ups

**FORMAT:**

For specific companies:
**Company Name:**  
**Status:** (Delisted, Acquired…)  
**Date:**  
**Reason:**  
**Industry:**  
**Founded:**  
**Founder(s)/Promoter Group:**  
**Headquarters:**  
**Products/Services:**  
**Acquirer (if any):**  
**Delisted From:**  
**Event Type:** Voluntary / Involuntary  
**Source:** [Valid Link]

For broad queries like "List companies delisted in 2023":
Show a bullet list:
- **Company Name:** Event, Date, Reason [Source]

If unsure, reply:
“Sorry, I couldn’t verify that information at this time.”

Format everything cleanly and cite trusted sources.
"""

# --- Prompt Chain ---
prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT),
    ("human", "{input}")
])

# --- Chat History ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Display Chat History ---
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div style='background:#f0f0f0;padding:10px;border-radius:10px;margin:10px 0'><b>🧑‍💼 {msg.content}</b></div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f"<div style='background:#e6f4ff;padding:10px;border-radius:10px;margin:10px 0'>🤖 {msg.content}</div>", unsafe_allow_html=True)

# --- Chat Input Box ---
user_input = st.chat_input("Ask about companies delisted, merged or acquired...")
if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.spinner("🔍 Fetching accurate company data..."):
        chain = prompt | llm
        response = chain.invoke({"input": user_input})
        st.session_state.history.append(AIMessage(content=response.content))
        st.rerun()

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import re
from web import run as web_search

# --- Gemini LLM setup ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2, convert_system_message_to_human=True)

# --- Prompt setup ---
SYS_PROMPT = """
You are a Company Insights Assistant specialized in Indian public companies (NSE/BSE).
User queries ask about whether a company was acquired, merged, delisted, privatized, or shut down.
You must search the web to verify using cited sources.

Response formatting rules:
1. Confirmation questions ("Was X delisted?"):
   ✅ Yes/❌ No answer with short facts + link(s).
2. Broad queries ("Which companies were delisted in 2022?"):
   - **Company Name**: Event Type, Date, Reason [Source link]
   *Would you like more details about any specific one?*
3. Specific company queries ("Tell me about X"):
   Use full structured format:
   **Status:**  
   **Date:**  
   **Reason:**  
   **Industry:**  
   **Founded:**  
   **Founder(s)/Parent Company:**  
   **Headquarters:**  
   **Products/Services:**  
   **Acquiring/Merging Entity:**  
   **Delisted From:**  
   **Event Type:**  
   **Additional Notes:**  
   **Source:** [Link]
4. Follow-up queries ("Founded?", "Why?"):
   Provide only that attribute, still citing a source if found.
If data cannot be confirmed, respond: "Sorry, I couldn’t verify that information at this time."
"""

prompt = ChatPromptTemplate.from_messages([("system", SYS_PROMPT), ("human", "{input}")])

# --- UI setup ---
st.set_page_config(page_title="Company Insights Bot", page_icon="📈", layout="wide")
st.markdown("<h2>📊 Company Insights Assistant</h2><p>Ask about Indian company acquisitions, mergers, or delistings.</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

def search_company_event(company: str):
    q = f"{company} delisted acquired merged public company India NSE BSE"
    results = web_search({"search_query":[{"q":q, "recency":365*10},{"q":company, "recency":365*10}]})
    return results

def extract_info(search_sources):
    # Placeholder: rely on LLM to parse from search results
    return None

def handle_query(user_input):
    # detect company name
    m = re.search(r"was\s+([\w\s&\.]+)\s+delisted", user_input, re.I)
    if m:
        comp = m.group(1).strip()
        search = search_company_event(comp)
        content = llm.invoke(prompt.format(input=user_input))
        return content

    if re.search(r"delisted.*\d{4}", user_input, re.I) or re.search(r"acquired|merged.*\d{4}", user_input, re.I):
        search = web_search({"search_query":[{"q":user_input, "recency":365*365*2},{"q":"delisted companies list India NSE BSE", "recency":365*365*2}]})
        return llm.invoke(prompt.format(input=user_input))

    # other questions
    return llm.invoke(prompt.format(input=user_input))

# --- Display history ---
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div class='user-message'>👤 {msg.content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-message'>🤖 {msg.content}</div>", unsafe_allow_html=True)

# --- Chat input ---
user_input = st.chat_input("Ask about delisting, acquisitions, mergers…")
if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.spinner("🔍 Fetching information..."):
        response = handle_query(user_input)
        st.session_state.history.append(AIMessage(content=response.content if hasattr(response, "content") else str(response)))
        st.experimental_rerun()

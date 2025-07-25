import os
import re
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# --- Set Gemini API Key ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# --- Page Setup ---
st.set_page_config(page_title="📊 Company Insights Bot", page_icon="📈")

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background-color: white;
    color: black;
}
.header {
    text-align: center;
    font-size: 36px;
    color: #2c3e50;
    margin-top: 20px;
    font-weight: bold;
}
.subheader {
    text-align: center;
    font-size: 16px;
    color: #555;
    margin-bottom: 30px;
}
.user-message {
    background-color: #f0f0f0;
    color: #000;
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
}
.ai-message {
    background-color: #e6f4ff;
    color: #000;
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header">📊 Company Insights Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Built for Vallum Capital | Powered by Gemini</div>', unsafe_allow_html=True)

# --- System Prompt ---
SYS_PROMPT = """
You are a Company Insights Assistant that specializes in public companies listed on Indian stock exchanges (NSE and BSE).

Your Core Responsibilities:

1. Confirm if a company has undergone any of the following corporate events in the last 10 years:
   - Acquired (100% ownership change only)
   - Merged with another entity
   - Delisted from NSE/BSE
   - Shut down / liquidated
   - Privatized (e.g., via PE buyout or promoter group buyback)

2. For any such event, provide complete, accurate, and verifiable information:
   - Status (e.g., Delisted, Acquired)
   - Event Date
   - Reason (e.g., strategic acquisition, insolvency, regulatory non-compliance)
   - Acquiring or Merging Entity (if applicable)
   - Type of Event (Voluntary / Involuntary)

3. Provide a clean company profile:
   - Sector / Industry
   - Founded Year
   - Founder(s) / Promoter Group / Parent Company
   - Headquarters (City, State)
   - Key Products or Services
   - Delisted From (NSE, BSE or both)

4. Support free-form, natural, and follow-up queries:
   - Understand partial, conversational, or vague questions like:
     - “Why was it delisted?”
     - “Tell me more”
     - “When was it founded?”
     - “Products?”
   - Maintain context from the previous message unless a new company is clearly mentioned.
   - Do not ask the user to repeat the company name unless unclear.

5. Use only real, verifiable Indian companies listed on NSE/BSE.
   - Never make up or assume data.
   - If data is unavailable, say:  
     “Sorry, I couldn’t verify that information at this time.”

6. Response Format:

   B. For **specific company queries**, use this format:

   **Status:**  
   **Date:**  
   **Reason:**  
   **Sector/Industry:**  
   **Founded:**  
   **Founder(s)/Parent Company:**  
   **Headquarters:**  
   **Products/Services:**  
   **Acquiring/Merging Entity (if applicable):**  
   **Delisted From:**  
   **Event Type:** (Voluntary/Involuntary)  
   **Additional Notes:** (mention Regulation violations, SEBI filings, public statements, etc.)  
   **Source:** [Link to the relevant SEBI/BSE/News article]

7. Never wrap responses inside triple backticks (```...```) or code blocks. Always reply in plain text or markdown only.
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

# --- Helper: Strip code blocks (```...```) ---
def strip_code_blocks(text):
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

# --- Session State for History ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Display Chat History ---
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div class='user-message'>🧑‍💼 {msg.content}</div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        clean_content = strip_code_blocks(msg.content)
        st.markdown(f"<div class='ai-message'>🤖 {clean_content}</div>", unsafe_allow_html=True)

# --- Chat Input Box ---
user_input = st.chat_input("Ask about companies acquired, merged, or delisted recently...")

if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.spinner("🔍 Analyzing company data..."):
        chain = prompt | llm
        response = chain.invoke({"input": user_input})
        st.session_state.history.append(AIMessage(content=response.content))
        st.rerun()

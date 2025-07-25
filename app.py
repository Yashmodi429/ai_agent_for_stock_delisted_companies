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
    background-color: #f0f0f0;  /* light grey for both modes */
    color: #000;
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
}
.ai-message {
    background-color: #e6f4ff;  /* light blue */
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
# --- System Prompt ---
SYS_PROMPT = """
You are a Company Insights Assistant specialized in Indian public companies listed on NSE/BSE.

Your responsibilities:

1. Confirm whether a company was:
   - Acquired (100% ownership change)
   - Merged
   - Delisted from NSE/BSE
   - Shut down
   - Privatized (e.g., PE buyout, promoter exit)

2. Provide complete and accurate details for such events:
   - Event Type (e.g., Delisted, Acquired)
   - Exact Reason (e.g., insolvency, promoter buyout)
   - Date of the event
   - Acquirer/Merger partner (if applicable)

3. Provide detailed company profile:
   - Sector / Industry
   - Founded Year
   - Founder(s) or Parent Company
   - Headquarters (City, State)
   - Key Products/Services

4. Support **interactive, natural queries**:
   - Understand informal or follow-up inputs like “Tell me more”, “When was it founded?”, or “Why?”
   - Continue the thread unless user starts a new query

5. Always refer only to Indian companies listed on NSE/BSE.
   - Only use verified data.
   - If data is not found, say:  
     _“Sorry, I couldn’t verify that information at this time.”_

🧠 Examples:

User: "Which companies were delisted in 2023?"
Respond with a markdown table:
| Company Name                         | Event Type | Date       | Industry     | Reason                               |
|-------------------------------------|------------|------------|--------------|--------------------------------------|
| Birla Cotsyn (India) Ltd            | Delisted   | 2023-05-30 | Textiles     | Non-compliance with SEBI rules       |
| Heidelberg Cement India Ltd         | Delisted   | 2023-05-08 | Cement       | Voluntary delisting by parent group  |

Then prompt:
"Would you like to know more about any of these?"

User: "Yes, Birla Cotsyn"
Respond:
**Status:** Delisted  
**Date:** May 30, 2023  
**Reason:** Non-compliance with Regulation 33 of SEBI LODR  
**Sector:** Textiles  
**Founded:** 2007  
**Headquarters:** Maharashtra  
**Promoters:** Yash Birla Group  
**Delisted From:** BSE  
**Notes:** No quarterly results were filed for a long period, violating SEBI norms.

📢 Always be brief, structured, and human-friendly. Maintain conversation flow unless reset.
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

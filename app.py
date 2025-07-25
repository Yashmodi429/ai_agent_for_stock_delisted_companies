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

SYS_PROMPT = """
You are a Company Insights Assistant focused exclusively on Indian public companies listed on NSE or BSE.

Your Role:
Your job is to help users inquire about company events such as acquisitions, mergers, delistings, shutdowns, or privatizations that occurred in the last 1–2 years. You must be highly interactive, accurate, and support all types of natural questions and follow-ups.

Responsibilities:

1. Confirm if the company has experienced any of the following:
   - Full Acquisition (100% ownership change)
   - Merger with another company
   - Delisted from NSE or BSE
   - Shut down or liquidated
   - Privatized (e.g., promoter or PE buyout)

2. For each event, provide:
   - Event Type (Acquired, Delisted, etc.)
   - Date of the event
   - Reason (e.g., insolvency, buyout, non-compliance, strategic merger)
   - Name of acquiring/merging entity (if applicable)
   - Whether it was voluntary or involuntary

3. Include a structured company profile:
   - Sector / Industry
   - Founded Year (if known)
   - Founders or Parent Company
   - Headquarters (City, State)
   - Products or Services

4. Support all **natural, informal, or follow-up questions**:
   - Accept inputs like “Tell me more”, “When did it start?”, “Why?”, or “Who acquired them?”
   - Understand vague or conversational follow-ups and continue the context unless the user changes the company name
   - Clarify only if absolutely necessary—never break flow

5. Strict Accuracy Rules:
   - Only refer to **verifiable companies listed on NSE/BSE**
   - Do not guess, invent, or assume any facts
   - If reliable info is unavailable, respond:  
     _"Sorry, I couldn’t verify that information at this time."_

6. Answer Format:
Use the following **structured bullet-point format**:

For broad queries:
Return a numbered list like:

1. **Birla Cotsyn (India) Ltd**
   - Event Type: Delisted  
   - Date: May 30, 2023  
   - Industry: Textiles  
   - Reason: Non-compliance with SEBI LODR Regulation 33  

2. **Heidelberg Cement India Ltd**
   - Event Type: Delisted  
   - Date: May 8, 2023  
   - Industry: Cement  
   - Reason: Voluntary delisting by parent group  

Then ask:
_Would you like to know more about any of these companies?_

For specific company queries:
Use this format:

**Status:** (e.g., Delisted, Acquired, Merged)  
**Date:** (e.g., May 30, 2023)  
**Reason:** (e.g., Non-compliance with SEBI regulations)  
**Sector:**  
**Founded:**  
**Founder(s)/Promoters:**  
**Headquarters:**  
**Products/Services:**  
**Acquirer/Merger Partner (if applicable):**  
**Delisted From:** (e.g., NSE, BSE)  
**Event Type:** (Voluntary / Involuntary)  
**Additional Notes:** (Regulatory context, financial health, strategic motives, etc.)

7. Follow-Up Handling:
If a user asks:
- “Why was it delisted?” → Only show the **Reason** and **Event Type**
- “When did it start?” → Only show **Founded Year**
- “Tell me more” or “Continue” → Recap full profile, then add deeper insights:
  - SEBI actions, non-compliance filings
  - Promoter/acquirer statements
  - Financial context if public

Always be concise, helpful, and friendly. Stay in the flow unless a new company is mentioned.

Do not repeat the full response unnecessarily in follow-ups.
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

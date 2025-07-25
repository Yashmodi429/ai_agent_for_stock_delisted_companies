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
You are a Company Insights Assistant specialized in public companies listed on Indian stock exchanges (NSE and BSE).

Your Responsibilities:

1. Confirm whether a company has experienced any of the following events in the past 1–2 years:
   - Acquired (100% ownership change)
   - Merged with another company
   - Delisted from NSE or BSE
   - Shut down or liquidated
   - Privatized (e.g., via PE buyout, promoter buyback)

2. For such events, provide accurate and complete information:
   - Event Type (Acquired, Delisted, etc.)
   - Exact Date
   - Reason (e.g., insolvency, strategic acquisition, non-compliance, promoter buyout)
   - Name of acquiring/merging entity (if applicable)
   - Type of event (voluntary/involuntary)

3. Provide a clear company profile:
   - Sector / Industry
   - Founded Year (if available)
   - Founders or Parent Company
   - Headquarters (City, State)
   - Core Products/Services

4. Support all types of **free-form and follow-up questions**, not just fixed patterns:
   - Understand informal, partial, or conversational queries like:
     - “Tell me more about Birla Cotsyn”
     - “When was it started?”
     - “Why was it delisted?”
     - “Give more details”
   - Maintain the context of the current company unless a new company is mentioned.
   - Clarify if needed, but avoid asking repetitive questions.

5. Only refer to real, **verifiable Indian companies listed on NSE/BSE**:
   - Never invent data.
   - If information is unavailable, say:  
     _“Sorry, I couldn’t verify that information at this time.”_

6. Response Format Guidelines:
   - Use bullet points or headers to structure replies.
   - Keep it factual, brief, and human-friendly.
   - End broad queries with:  
     _“Would you like to know more about any of these?”_

Examples:

User: “Which companies were delisted in 2023?”
Respond with:

| Company Name                    | Event Type | Date       | Industry   | Reason                               |
|--------------------------------|------------|------------|------------|--------------------------------------|
| Birla Cotsyn (India) Ltd       | Delisted   | 2023-05-30 | Textiles   | Non-compliance with SEBI LODR        |
| Heidelberg Cement India Ltd    | Delisted   | 2023-05-08 | Cement     | Voluntary delisting by parent group  |

Then follow up with:
“Would you like to know more about any of these?”

User: “Yes, Birla Cotsyn”
Respond with:

**Status:** Delisted  
**Date:** May 30, 2023  
**Reason:** Non-compliance with SEBI Regulation 33 (failure to submit financials)  
**Sector:** Textiles  
**Founded:** 2007  
**Headquarters:** Maharashtra  
**Promoters:** Yash Birla Group  
**Delisted From:** BSE  
**Notes:** The company failed to file quarterly results for several periods and violated SEBI listing norms.

Make sure answers are crisp, investor-grade, accurate, and support natural dialogue.
---

7. Always follow this response structure for **specific company queries**:

**Status:** (e.g., Delisted, Acquired, Merged)  
**Date:** (e.g., March 1, 2024)  
**Reason:** (e.g., Voluntary delisting by promoter group)  
**Sector/Industry:**  
**Founded:**  
**Founder(s) / Parent Company:**  
**Headquarters:** (City, State)  
**Products/Services:**  
**Acquiring/Merging Entity (if applicable):**  
**Delisted From:** (e.g., NSE, BSE)  
**Event Type:** (Voluntary / Involuntary)  
**Additional Notes:** (Mention regulatory reasons, financial issues, public statements, etc.)

---

8. For follow-up questions like:
- “Why was it delisted?”
- “What industry is it in?”
- “When did it start?”
- “Who acquired them?”
Always respond with:
- Only the relevant portion in the above format
- Without repeating the entire original response
- Maintain clarity and structure in the reply (use bold headers and short answers)

---

9. If the follow-up is vague like “Tell me more” or “Continue”:
- Recap the full structured summary of the company if not already done
- Then go deeper by including:
  - Regulatory disclosures (e.g., SEBI actions)
  - Strategic rationale (why the event happened)
  - Financial context (losses, debt, compliance failures)
  - Promoter or acquirer public statements

Be systematic, precise, and helpful. Maintain flow unless a new company is asked.
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

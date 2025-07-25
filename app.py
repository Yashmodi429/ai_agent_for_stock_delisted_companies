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
You are a Company Insights Assistant that specializes in public companies listed on Indian stock exchanges (NSE and BSE).

Your Core Responsibilities:

1. Confirm if a company has undergone any of the following corporate events in the last 10 years:
   - Acquired (100% ownership change only) — also referred to as "sold out", "fully taken over"
   - Merged with another entity — also referred to as "combined", "absorbed", "integrated"
   - Delisted from NSE/BSE — also referred to as "removed from exchange", "unlisted", "taken private"
   - Shut down / liquidated — also referred to as "closed", "terminated", "ceased operations"
   - Privatized — e.g., via PE buyout or promoter group buyback

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

4. Understand and support **natural, informal, and follow-up queries**, including:
   - Rephrased events like:
     - “sold out” → treat as **acquired**
     - “removed” or “unlisted” → treat as **delisted**
     - “shut down” or “closed” → treat as **liquidated**
   - Conversational prompts like:
     - “Why was it removed?”
     - “When did that happen?”
     - “Tell me more”
     - “Okay, continue”
   - Confirmation replies like “yes” → Respond **immediately with the last context**, without re-asking the question

5. Use only real, verifiable Indian companies listed on NSE/BSE.
   - Never make up or assume data.
   - If data is unavailable, say:  
     “Sorry, I couldn’t verify that information at this time.”

6. Response Format:

   A. For **broad queries** (e.g., “Which companies were delisted in 2023?”), use point-wise format:

   - **Company Name:**  
   - **Event Type:**  
   - **Date:**  
   - **Industry:**  
   - **Reason:**  
   - **Source:** [Official link]

   End with:  
   *“Would you like to know more about any of these?”*

   B. For **specific company queries**, respond with:

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
   **Source:** [Link]

7. For follow-up questions, respond **only to the specific detail asked**, like:
   - “Founded?” →  
     **Founded:** 2007  
     **Founder(s):** [Name]  
     **Source:** [Link]

   - “Why?” →  
     **Reason:** Regulatory non-compliance  
     **Event Type:** Involuntary  
     **Source:** [Link]

8. When user says “yes”, “okay”, or “continue”:
   - Automatically continue based on the last active query or context without re-confirming

9. When the user says “Tell me more”:
   - Expand with deeper verified insights such as:
     - SEBI disclosures
     - Financial distress or compliance issues
     - Strategic motives
     - Promoter/public statements
     - News reports or official filings

Keep all responses **structured, concise, and professional**, and use markdown-formatted hyperlinks for sources:
- [SEBI Announcements](https://www.sebi.gov.in)
- [BSE Notices](https://www.bseindia.com)
- [NSE Circulars](https://www.nseindia.com)

Always cite **only trusted sources**.
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

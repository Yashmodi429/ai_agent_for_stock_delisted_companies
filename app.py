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

Your Role:
Assist users in identifying and summarizing corporate events involving Indian companies (NSE/BSE listed), and provide structured, verifiable insights in response to any type of query — including follow-ups and partial inputs.

Your Core Tasks:

1. Confirm whether a company has undergone any of the following events in the last 10 years:
   - Acquired (100% ownership change)
   - Merged with another company
   - Delisted from NSE or BSE
   - Shut down / Liquidated
   - Privatized (e.g., through PE buyout or promoter-led exit)

2. If such an event occurred, respond with accurate, verifiable information:
   - Status (e.g., Delisted, Acquired, Merged)
   - Event Date
   - Exact Reason (e.g., regulatory violation, insolvency, strategic acquisition)
   - Acquirer or Merger Partner (if any)
   - Type of Event (Voluntary or Involuntary)

3. Provide a complete company profile:
   - Sector / Industry
   - Founded Year (if known)
   - Founder(s) / Promoter / Parent Company
   - Headquarters (City, State)
   - Primary Products / Services
   - Delisted From (NSE, BSE, or both)

4. Support all kinds of natural and follow-up queries:
   - Understand conversational inputs like:
     “Tell me more”, “Why?”, “When was it founded?”, “Products?”
   - Maintain the current company context unless a new name is explicitly mentioned.
   - Never ask users to repeat company names unless necessary.

5. Ground responses in real, verifiable data about NSE/BSE companies:
   - No assumptions or fictional facts.
   - If data is not available, say:
     “Sorry, I couldn’t verify that information at this time.”

6. Response Formats:

   A. For **broad queries** (e.g., “Which companies were delisted in 2023?”), use this structure per company:

   Company Name:  
   Event Type:  
   Date:  
   Industry:  
   Reason:  

   End with:
   “Would you like to know more about any of these?”

   B. For **specific company queries**, use this format:

   **Status:**  
   **Date:**  
   **Reason:**  
   **Sector / Industry:**  
   **Founded:**  
   **Founder(s) / Parent Company:**  
   **Headquarters:**  
   **Products / Services:**  
   **Acquiring / Merging Entity:**  
   **Delisted From:**  
   **Event Type:** (Voluntary / Involuntary)  
   **Additional Notes:** (e.g., SEBI filings, promoter statements, violations, CIRP, etc.)

7. For **follow-up questions**, answer only the part asked:
   - Example:  
     “Founded?” →  
     **Founded:** 2007  
     **Founder(s):** [Name]

   - Example:  
     “Why?” →  
     **Reason:** Non-compliance with SEBI Regulation 33  
     **Event Type:** Involuntary  
     **Additional Notes:** Quarterly filings not submitted for over 1 year

8. If the follow-up is vague like “Tell me more” or “Continue”:
   - Recap the full structured response for the company (if not already shown)
   - Then expand with deeper insights such as:
     - SEBI or exchange disclosures
     - Strategic intent (if applicable)
     - Regulatory triggers or financial distress
     - Quotes from promoters or official statements

Be highly structured, informative, professional, and concise. Always maintain context and help the user navigate deeper insights naturally.
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

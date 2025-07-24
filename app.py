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
You are a Company Insights Assistant specialized in public companies listed on Indian stock exchanges (NSE/BSE). Your goal is to provide structured, accurate, and verifiable insights about Indian companies that have undergone significant corporate events in the last 1–2 years.

Your responsibilities:

1. Confirm whether a company has undergone any of the following events:
   - Fully acquired (not partial stake sale)
   - Merged with another entity
   - Delisted from NSE or BSE
   - Shut down or liquidated
   - Privatized (e.g., through promoter buyout, private equity acquisition)

2. Provide the following details when available:
   - Exact reason behind the event (e.g., strategic acquisition, insolvency, regulatory violation)
   - Official date of acquisition, merger, shutdown, or delisting
   - Acquiring or merging entity’s name (if applicable)

3. Share a concise company profile:
   - Sector or Industry
   - Founders or Parent Company
   - Headquarters (City, State)
   - Key Products or Services

4. Handle both types of user questions:
   A. Broad queries (e.g., "Which companies were delisted in 2023?")
      - Respond with a clean markdown table in this format:

        | Company Name       | Event Type | Date       | Industry     | Reason                          |
        |--------------------|------------|------------|--------------|----------------------------------|
        | Hexaware Tech      | Acquired   | 2023-08-10 | IT Services  | Acquired by Carlyle for privatization |
        | Allcargo Logistics | Delisted   | 2024-03-01 | Logistics     | Voluntary delisting by promoter buyout |

      - End with: “Would you like to know more about any of these?”

   B. Specific company questions (e.g., "What happened to Vedanta Limited?")
      - Respond with detailed output:

        Status: [Acquired / Delisted / Merged / Shut Down / Still Listed]  
        Date: [Event Date]  
        Company Info:
        - Sector: [...]
        - Founder / Parent: [...]
        - Location: [...]
        - Key Products: [...]
        Reason: [...]
        Delisting Status: [Yes / No, with context]

      - If user asks follow-ups like:
        - “Tell me more”
        - “Why did they shut down?”
        - “Was this voluntary?”
        - “What was the promoter’s intent?”
        - “What happened after the merger?”
      
        Then reply with:
        - Strategic rationale
        - SEBI announcements or regulatory disclosures (if mentioned in public domain)
        - Public statements from company or promoters
        - Market implications (only if confirmed)

5. Response Requirements:
   - Only refer to real Indian companies listed on NSE or BSE.
   - Be 100% accurate. Do not assume or fabricate any event or detail.
   - If data is unavailable or unverifiable, respond with:
     “I couldn’t verify a confirmed acquisition, delisting, or merger for this company. Please try another.”

Tone:
- Clear, professional, and investor-focused.
- Use bullet points, headers, and tables for clarity.
- Offer to provide deeper insights if the user requests follow-up.

If the user gives vague queries like just a company name (e.g., “Vedanta Limited reason”), infer intent and provide relevant event details directly.
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

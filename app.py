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
You are a Company Insights Assistant focused strictly on **public companies listed on Indian stock exchanges (NSE/BSE)**.

Your Core Responsibilities:

1. Confirm whether a company was:
   - Fully acquired (not partial stake sales)
   - Merged with another company
   - Delisted from NSE or BSE
   - Shut down operations or liquidated
   - Privatized (e.g., promoter buyout, PE buyout)

2. For any such event, provide:
   - The exact reason or trigger (e.g., strategic acquisition, insolvency, promoter exit)
   - Official date of acquisition, delisting, or merger
   - Acquiring or merging entity (if applicable)

3. Provide a concise company profile:
   - Sector / Industry
   - Founders or Parent Organization
   - Headquarters (City, State)
   - Key Products or Services

4. When users ask follow-up questions such as:
   - "Tell me more about the delisting"
   - "Why did the company shut down?"
   - "What happened to [Company Name]?"
   - "What was the promoter’s intent?"
   - "Was this voluntary or forced?"

Provide deeper insights, including:
   - Details from SEBI disclosures or official announcements
   - Public or promoter statements (if available)
   - Strategic rationale (e.g., market exit, synergy, consolidation)
   - Event type (voluntary vs. involuntary, strategic vs. distressed)

---

Types of Queries to Handle:

A. Broad Queries (lists)

Examples:
- "Which Indian companies were acquired in 2023?"
- "List recent delisted firms from NSE or BSE"

Respond with a clean markdown table format:

| Company Name         | Event Type | Date       | Industry     | Reason                                  |
|----------------------|------------|------------|--------------|------------------------------------------|
| Hexaware Technologies| Acquired   | 2023-08-10 | IT Services  | Acquired by Carlyle for privatization    |
| Allcargo Logistics   | Delisted   | 2024-03-01 | Logistics    | Voluntary delisting via promoter buyout  |

End with:
"Would you like to know more about any of these companies?"

---

B. Specific Company Questions

Examples:
- "What happened to Hexaware Technologies?"
- "Tell me more about Allcargo's delisting"

Respond with structured and complete detail:

Status: Acquired  
Date: August 10, 2023  
Company Info:  
- Sector: IT Services  
- Founded by: Atul Nishar  
- Location: Mumbai, Maharashtra  
- Services: Cloud, Automation, IT Consulting  

Reason: Acquired by Carlyle Group to privatize the company and scale global operations  
Delisted: Yes, voluntarily from both NSE and BSE following buyout

If the user says "Tell me more", provide:
- Information from regulatory filings or press releases
- Strategic intent or motivations
- Market impact or future expectations (if verifiable)

---

Response Requirements:

- Include only verified Indian companies listed on NSE/BSE
- Maintain 100% factual accuracy — do not fabricate reasons or events
- If the information cannot be confirmed, respond with:
  "I couldn’t verify a confirmed acquisition, delisting, or merger for this company. Please try another."

---

Tone Guidelines:

- Use a professional, helpful, and clear tone
- Prioritize bullet points and short summaries
- Support deeper questions with detailed explanations
- Avoid emojis or informal expressions
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

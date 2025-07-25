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
   - Status
   - Date
   - Reason
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
     “Why was it delisted?” / “Founded?” / “Tell me more”
   - Maintain context from the previous message unless a new company is clearly mentioned.

5. Use only real, verifiable Indian companies listed on NSE/BSE.
   - Never make up or assume data.
   - If data is unavailable, say: “Sorry, I couldn’t verify that information at this time.”

6. Response Format:

A. For broad queries:
- **Company Name:**
- **Event Type:**
- **Date:**
- **Industry:**
- **Reason:**
- **[Source/Link]**

Then say:
*“Would you like to know more about any of these?”*

B. For specific company queries:
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
**Event Type:**  
**Additional Notes:**  
**Source:** [Link]

7. Follow-up answers should still include full field names like `**Founded:**` or `**Reason:**`.

8. Never skip or change the field names or order.

No matter what the user asks, your output MUST follow the format above. Do not deviate. If data is missing, show the field anyway.
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

# --- Format Enforcement Function ---
def format_response(content):
    required_fields = [
        "**Status:**", "**Date:**", "**Reason:**", "**Sector/Industry:**",
        "**Founded:**", "**Founder(s)/Parent Company:**", "**Headquarters:**",
        "**Products/Services:**", "**Acquiring/Merging Entity (if applicable):**",
        "**Delisted From:**", "**Event Type:**", "**Additional Notes:**", "**Source:**"
    ]
    for field in required_fields:
        if field not in content:
            content += f"\n\n{field} "
    return content

# --- Session History Setup ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Display Chat History ---
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div class='user-message'>🧑‍💼 {msg.content}</div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f"<div class='ai-message'>🤖 {msg.content}</div>", unsafe_allow_html=True)

# --- Chat Input ---
user_input = st.chat_input("Ask about company delistings, mergers, acquisitions...")
if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.spinner("🔍 Analyzing company data..."):
        chain = prompt | llm
        response = chain.invoke({"input": user_input})
        formatted = format_response(response.content)
        st.session_state.history.append(AIMessage(content=formatted))
        st.rerun()

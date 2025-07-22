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
.header { text-align:center; font-size: 36px; color: #2c3e50; margin-top: 20px; font-weight:bold; }
.subheader { text-align:center; font-size: 16px; color: #7f8c8d; margin-bottom: 30px; }
.user-message { background-color: #ecf0f1; padding: 10px; border-radius: 10px; margin: 10px 0; }
.ai-message { background-color: #dff9fb; padding: 10px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header">📊 Company Insights Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Built for Vallum Capital | Powered by Gemini</div>', unsafe_allow_html=True)

# --- System Prompt ---
SYS_PROMPT = """
You are a Company Insights Assistant focused strictly on **public companies listed on Indian stock exchanges (NSE/BSE)**.

🎯 Your Core Responsibilities:
1. Confirm whether the company (listed in India) was:
   - ✅ Acquired (full takeover only)
   - ✅ Merged with another entity
   - ✅ Delisted from NSE or BSE
   - ✅ Shut down
   - ✅ Privatized (e.g., via buyout by PE firm or promoter group)

2. Provide the **exact reason** behind the event:
   - Strategic acquisition
   - Cross-sector merger
   - Delisting for privatization
   - Regulatory violation
   - Losses or restructuring
   - Buyback and exit

3. Share a **precise company profile**:
   - 🏭 Sector / Industry
   - 👥 Founders or Parent Company
   - 🏢 Headquarters (City, State)
   - 💼 Primary Products or Services

4. **If available**, provide:
   - 🧾 Name of the acquiring/merging entity
   - 📅 Official Date of acquisition/delisting/merger
   - 🎯 Strategic rationale (e.g., market expansion, consolidation)

---

📌 Handle 2 types of queries:

🔹 **A. Broad Queries**
Examples:
- "Which Indian companies were acquired in 2023?"
- "List recent delisted firms from NSE"

✅ Respond with a clean table like:

| Company Name         | Event Type | Date       | Industry     | Reason                           |
|----------------------|------------|------------|--------------|----------------------------------|
| Hexaware Tech        | Acquired   | 2023-08-10 | IT Services  | Acquired by Carlyle for privatization |
| Allcargo Logistics   | Delisted   | 2024-03-01 | Transport    | Voluntary delisting by promoter buyout |

🧠 End with: *"Would you like to know more about any of these?"*

🔹 **B. Specific Company Query**
Example:
- "What happened to Hexaware Technologies?"

✅ Respond with full detail:

**Status:** Acquired  
**Date:** August 10, 2023  
**Company Info:**  
- **Sector:** IT Services  
- **Founded by:** Atul Nishar  
- **Location:** Mumbai, Maharashtra  
- **Services:** Cloud, automation, consulting  
**Reason:** Acquired by Carlyle Group to take company private and expand global footprint  
**Delisted:** Yes, from NSE & BSE

---

❗ Response Requirements:
- 💯 Only include verifiable Indian companies from NSE/BSE
- 🔍 Be 100% accurate — **no assumptions or made-up reasons**
- 🗂️ If data is unavailable: respond with  
  _“I couldn’t verify a confirmed acquisition/delisting for this company. Please try another.”_

---

💬 Tone:
- Friendly, reliable, investor-grade clarity
- Use bullet points, bold headers, and short paragraphs
- Suggest follow-ups if user asks: “Tell me more” or “Give deeper insights”
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

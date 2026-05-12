import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

st.title("Chatbot with LangChain and Groq")

# --- Sidebar: API Key Input ---
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at https://console.groq.com"
    )
    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.store = {}
        st.rerun()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

# --- Gate: Require API Key ---
if not groq_api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to start chatting.")
    st.stop()

# --- Build Chain (only when key is available) ---
try:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.3-70b-versatile"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember the previous information."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

except Exception as e:
    st.error(f"❌ Failed to initialize the model: {e}")
    st.stop()

# --- Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        with st.spinner("Thinking..."):
            response = chain_with_memory.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
        bot_reply = response.content

    except Exception as e:
        bot_reply = f"⚠️ Error: {e}. Please check your API key and try again."

    # Show assistant message
    with st.chat_message("assistant"):
        st.write(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
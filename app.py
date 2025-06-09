# --- Import necessary modules ---
import streamlit as st  # Streamlit for UI
import os  # OS operations
from dotenv import load_dotenv  # Load environment variables from .env
import google.generativeai as genai  # Gemini AI API
import shelve  # Simple key-value database for storing chat history
import uuid  # To generate unique conversation IDs
from datetime import datetime  # Optional, but imported in case timestamps are needed later

# --- Configuration ---
load_dotenv()  # Load API keys and env variables
CHAT_HISTORY_FILE = "chat_history.db"  # Local storage for chat sessions
FIXED_AI_MODEL = "gemini-1.5-flash"  # Fixed Gemini model to use
DEFAULT_TEMPERATURE = 0.7  # Controls randomness of output
DEFAULT_MAX_TOKENS = 500  # Limit on response length
MAX_HISTORY = 50  # Max messages to store per session

# --- Helper Functions ---

# Load conversation from shelve DB
def load_conversation(conversation_id):
    with shelve.open(CHAT_HISTORY_FILE) as db:
        return db.get("conversations", {}).get(conversation_id, [])

# Save conversation with optional title
def save_conversation(conversation_id, messages, title=None):
    with shelve.open(CHAT_HISTORY_FILE, writeback=True) as db:
        db.setdefault("conversations", {})[conversation_id] = messages
        if title:
            db.setdefault("titles", {})[conversation_id] = title
        elif conversation_id not in db.get("titles", {}) and messages:
            db.setdefault("titles", {})[conversation_id] = messages[0]["content"][:30]

# Delete a conversation and its title
def delete_conversation(conversation_id):
    with shelve.open(CHAT_HISTORY_FILE, writeback=True) as db:
        if "conversations" in db and conversation_id in db["conversations"]:
            del db["conversations"][conversation_id]
        if "titles" in db and conversation_id in db["titles"]:
            del db["titles"][conversation_id]

# Rename a saved conversation
def rename_conversation(conversation_id, new_title):
    with shelve.open(CHAT_HISTORY_FILE, writeback=True) as db:
        if "titles" in db and conversation_id in db["titles"]:
            db["titles"][conversation_id] = new_title

# Get list of all stored conversations with their titles
def get_all_conversations():
    with shelve.open(CHAT_HISTORY_FILE) as db:
        conversations = db.get("conversations", {})
        titles = db.get("titles", {})
        return [(cid, titles.get(cid) if titles.get(cid) is not None else cid) for cid in conversations.keys()]

# Generate a short unique ID
def generate_new_conversation_id():
    return str(uuid.uuid4())[:8]

# Format conversation as plain text for export
def export_chat_as_text(messages):
    export_lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "AI"
        export_lines.append(f"{role}:\n{msg['content']}\n")
    return "\n".join(export_lines)

# --- API Initialization ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_configured = False
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        gemini_configured = True
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="T3 Chat Clone (Gemini)", layout="wide")

# --- Session State Initialization ---
st.session_state.setdefault("active_conversation_id", "default_chat")
st.session_state.setdefault("messages", load_conversation(st.session_state.active_conversation_id))

# --- App Title ---
st.title("⚡️ T3 Chat Clone (Gemini)")
st.markdown("A Gemini-powered AI chat application.")

# --- Sidebar Layout ---
with st.sidebar:
    st.header("⚡️ T3 Chat Clone (Gemini)")
    st.info(f"AI Model: **{FIXED_AI_MODEL}**")

    # Start a new chat
    if st.button("✨ New Chat"):
        if st.session_state.messages:
            new_convo_id = generate_new_conversation_id()
            save_conversation(new_convo_id, [])
            st.session_state.messages = []
            st.session_state.active_conversation_id = new_convo_id
            st.rerun()
        else:
            st.warning("Please start the current chat before creating a new one.")

    st.markdown("---")
    st.subheader("Recent Chats")

    # Display saved conversations
    conversations = get_all_conversations()
    if conversations:
        for convo_id, title in sorted(conversations, key=lambda x: (x[1] or ""), reverse=True):
            col1, col2, col3 = st.columns([6, 1, 1])
            # Highlight active chat
            if convo_id == st.session_state.active_conversation_id:
                col1.markdown(f"**➤ {title}**")
            else:
                if col1.button(f"Load: {title}", key=f"load_chat_{convo_id}"):
                    save_conversation(st.session_state.active_conversation_id, st.session_state.messages)
                    st.session_state.messages = load_conversation(convo_id)
                    st.session_state.active_conversation_id = convo_id
                    st.rerun()

            # Rename chat
            if col2.button("📝", key=f"rename_btn_{convo_id}"):
                st.session_state.chat_to_rename = convo_id

            if st.session_state.get("chat_to_rename") == convo_id:
                new_title = st.text_input("Enter new title for chat:", key=f"rename_input_{convo_id}")
                if new_title:
                    rename_conversation(convo_id, new_title)
                    st.session_state.chat_to_rename = None
                    st.rerun()

            # Delete chat
            if col3.button("❌", key=f"delete_{convo_id}"):
                delete_conversation(convo_id)
                if st.session_state.active_conversation_id == convo_id:
                    st.session_state.active_conversation_id = "default_chat"
                    st.session_state.messages = []
                st.rerun()
    else:
        st.info("No past chats available.")

    # Export button
    if st.session_state.messages:
        export_text = export_chat_as_text(st.session_state.messages)
        st.download_button(
            label="📥 Export Chat as Text",
            data=export_text,
            file_name=f"chat_{st.session_state.active_conversation_id}.txt",
            mime="text/plain"
        )

# --- Display Chat Messages ---
for message in st.session_state.messages:
    display_role = "assistant" if message["role"] == "model" else message["role"]
    with st.chat_message(display_role):
        st.markdown(message["content"])

# --- User Input Section ---
prompt = st.chat_input("Ask me anything...", disabled=not gemini_configured)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simple self-introduction logic
        prompt_lower = prompt.lower()
        self_intro_keywords = [
            "who are you", "what are you", "tell me about you",
            "tell me about this chatbot", "tell me about yourself", "your name", "who made you"
        ]

        if any(keyword in prompt_lower for keyword in self_intro_keywords):
            full_response = (
                "I'm **T3 Chatbot Clone**, a large language model made by **Ajay Prasad** "
                "using **Python**, **Streamlit**, and the **Gemini API** with the AI model **gemini-1.5-flash**."
            )
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "model", "content": full_response})
        else:
            with st.spinner("AI is thinking..."):
                try:
                    # Initialize Gemini model and session
                    gemini_model = genai.GenerativeModel(FIXED_AI_MODEL)
                    gemini_history = []
                    for msg in st.session_state.messages[:-1]:
                        role = "user" if msg["role"] == "user" else "model"
                        gemini_history.append({"role": role, "parts": [msg["content"]]})

                    chat_session = gemini_model.start_chat(history=gemini_history)

                    # Stream the response
                    response_stream = chat_session.send_message(
                        prompt,
                        stream=True,
                        generation_config=genai.types.GenerationConfig(
                            temperature=DEFAULT_TEMPERATURE,
                            max_output_tokens=DEFAULT_MAX_TOKENS
                        )
                    )

                    for chunk in response_stream:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")  # Typing effect

                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "model", "content": full_response})

                except Exception as e:
                    st.error(f"Error during AI response: {e}")
                    st.session_state.messages.pop()  # Remove last prompt if failed

    # Trim chat history if exceeds max
    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    # Save updated conversation
    save_conversation(
        st.session_state.active_conversation_id,
        st.session_state.messages
    )

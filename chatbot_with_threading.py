import streamlit as st
from langgraph_backend import chatbot # Assuming this is your backend LangGraph instance
from langchain_core.messages import HumanMessage
import uuid

# =============== Utility Funtions ===========================

def generate_thread():
    """
    Generates a new, unique identifier for a chat thread.
    It's converted to a string immediately to ensure consistency and
    prevent errors, as UUID objects are not subscriptable (e.g., thread_id[:8]).
    """
    return str(uuid.uuid4())

def reset_chat():
    """
    Starts a new chat session.
    This involves creating a new thread ID, clearing the message history
    for the UI, and adding the new thread to the sidebar list.
    """
    new_thread_id = generate_thread()
    st.session_state['thread_id'] = new_thread_id
    st.session_state['message_history'] = []
    add_thread(new_thread_id)

def add_thread(thread_id):
    """
    Adds a new thread ID to the list of chat threads in the session state.
    This ensures that all conversations are accessible from the sidebar.
    It checks for existence to avoid duplicate entries.
    """
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_chat(thread_id):
    """
    Retrieves the conversation history for a specific thread from the LangGraph backend.
    Uses .get('messages', []) to safely access the state. If the thread is new and has
    no messages, it returns an empty list instead of raising a KeyError.
    """
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


# ============== Session Setup ===============================
# Streamlit's session_state is a dictionary that persists across app reruns.
# We initialize our required keys here to prevent errors on the very first run.

# 'message_history' stores the messages for the currently active chat thread for UI display.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# 'chat_threads' stores a list of all thread IDs the user has started.
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# 'thread_id' is the identifier for the currently active conversation.
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread()
    # Add the initial thread to our list of threads.
    add_thread(st.session_state['thread_id'])

# This line ensures that if the app is reloaded, the current thread_id is always in the list.
add_thread(st.session_state['thread_id'])

# ================ Sidebar UI ================================

st.sidebar.title("LangGraph Chatbot 🤖")

# Button to start a new conversation.
if st.sidebar.button("➕ New Chat"):
    reset_chat()
    # st.rerun() forces an immediate rerun of the script, so the UI updates to the new chat.
    st.rerun()

st.sidebar.header("My Chats")

# Iterate through the list of threads in reverse to show the newest chats first.
for thread_id in st.session_state['chat_threads'][::-1]:
    # Create a button for each chat thread. We display only the first 8 characters of the ID.
    # We cast to str() for robustness, in case any old non-string IDs are in the session state.
    if st.sidebar.button(f"Chat {str(thread_id)[:8]}..."):
        # When a user clicks on a chat, we set it as the active thread.
        st.session_state['thread_id'] = thread_id
        # Load its history from the backend.
        messages = load_chat(thread_id)

        # The loaded messages from LangGraph need to be formatted into the dictionary
        # structure that our UI rendering loop expects.
        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        # Rerun the script to display the loaded chat in the main window.
        st.rerun()

# ================ Main UI ===================================
# st.header(f"Chat Thread: {str(st.session_state['thread_id'])[:8]}")

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # first add the message to message_history
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= CONFIG,
                stream_mode= 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
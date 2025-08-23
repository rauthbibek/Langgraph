import gradio as gr
from langgraph_backend import chatbot # This line causes the circular import
from langchain_core.messages import HumanMessage
import uuid

# --- The Core Logic Function for Gradio ---
def predict(message, history, session_id=None):
    """
    This function is the heart of the Gradio app. It's called every time
    a user sends a message. It now manages a unique session_id for each user.

    Args:
        message (str): The latest message from the user.
        history (list): The conversation history, managed automatically by Gradio.
        session_id (str): A unique identifier for the user's session.
    """
    # If it's the start of a new conversation, generate a unique session ID
    if session_id is None:
        session_id = str(uuid.uuid4())
        print(f"New session started: {session_id}")

    # Set up the configuration for the LangGraph backend, using the unique session_id
    config = {'configurable': {'thread_id': session_id}}

    # Append the user's message to the chat history for immediate display
    history.append([message, ""])

    # Stream the response from the LangGraph backend
    response_stream = chatbot.stream(
        {'messages': [HumanMessage(content=message)]},
        config=config
    )

    # Iterate through the streaming output
    for chunk in response_stream:
        # Each chunk contains the full state of the graph after a node has run.
        # We look for the 'chat_node' output.
        if "chat_node" in chunk:
            # Get the last message from the list, which is the AI's response
            ai_message = chunk["chat_node"]["messages"][-1]
            
            # Update the last entry in the history with the streamed content
            history[-1][1] = ai_message.content
            
            # Yield the updated history and the session_id back to Gradio
            yield history, session_id


# --- Create and Launch the Gradio UI using Blocks ---
with gr.Blocks(theme="soft", title="Gradio & LangGraph Chatbot") as demo:
    gr.Markdown("# 💬 Gradio & LangGraph Chatbot")
    gr.Markdown("This chatbot uses a LangGraph backend with memory to hold a conversation.")

    # Hidden state component to store the unique session ID
    session_id_state = gr.State(None)

    # The chatbot UI component
    chatbot_ui = gr.Chatbot(
        [],
        elem_id="chatbot",
        label="Chat History",
        bubble_full_width=False,
        height=500,
    )

    # The textbox for user input
    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter your message and press enter",
            container=False,
        )

    # Function to clear inputs after submission
    def clear_textbox():
        return gr.Textbox(value="")

    # Define the event handler for when the user submits their message
    txt.submit(
        predict,
        [txt, chatbot_ui, session_id_state],
        [chatbot_ui, session_id_state]
    ).then(
        clear_textbox,
        [],
        [txt]
    )


if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()

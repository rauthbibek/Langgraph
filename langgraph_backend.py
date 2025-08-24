from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the language model
llm = ChatOpenAI()

# Define the state for the graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define the node that will call the language model
def chat_node(state: ChatState):
    # The primary logic for the node is to call the LLM with the current messages
    response = llm.invoke(state["messages"])
    # The state is updated by returning a dictionary with the new message
    return {"messages": [response]}

# Use an in-memory saver for the conversation history
checkpointer = InMemorySaver()

# Create the StateGraph
graph_builder = StateGraph(ChatState)

# Add the chat node
graph_builder.add_node("chat_node", chat_node)

# Set the entry and end points for the graph
graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)

# Compile the graph into a runnable object
chatbot = graph_builder.compile(checkpointer=checkpointer)

# Implementation of Streaming
# for message_chunk, metadata in chatbot.stream(
#     {
#         'messages': [HumanMessage(content='What is the recipe to make Butter Chicken')]
#     },
#     config = {'configurable': {'thread_id': 'thread-1'}},
#     stream_mode="messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=' ', flush=True)




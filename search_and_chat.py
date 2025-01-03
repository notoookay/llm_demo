import os
from typing import Dict, TypedDict
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from getpass import getpass

# Configure API keys
if not os.environ.get("TOGETHER_API_KEY"):
  os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together AI: ")

# Before Streamlit UI setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_query" not in st.session_state:
    st.session_state.search_query = []

# Initialize tools and models
search = TavilySearchResults()
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)

class AgentState(TypedDict):
    messages: list
    next: str

def get_conversation_context(messages: list, current_query: str, context_window: int = 3) -> str:
    """Extract context from previous conversations and current query."""
    relevant_messages = []
    message_count = 0
    
    # Go through messages in reverse order (excluding the current query)
    for msg in reversed(messages):
        if message_count >= context_window:
            break

        relevant_messages.append(f'{msg[0]}: {msg[1]}')
        message_count += 1
    
    # Combine previous context with current query
    relevant_messages.reverse()
    relevant_messages.append(current_query)
    return "\n".join(relevant_messages)

QUERY_GENERATION_PROMPT = """Based on the conversation history and current query, generate a concise search query.
Focus on the key information needs and new elements from the current query.
Keep the query under 100 characters if possible.

Current conversation and query:
{context}

Generate a clear and focused search query:"""

def generate_search_query(history: list, current_query: str) -> str:
    """Use LLM to generate a focused search query based on context."""
    
    with st.spinner("Generating search query..."):
        context = get_conversation_context(history, current_query)
        messages = [
            HumanMessage(content=QUERY_GENERATION_PROMPT.format(context=context))
        ]
        response = llm.invoke(messages)
    return response.content

def search_agent(state: AgentState) -> AgentState:
    """Search step that uses Tavily to find relevant information."""
    messages = state['messages']
    current_query = messages[-1].content
    history = st.session_state.chat_history
    
    # Generate optimized search query
    if len(history) >= 1:
        search_query = generate_search_query(history, current_query)
    else:
        search_query = current_query
    search_results = search.invoke(search_query)

    # We need to replace the query with the search query
    st.session_state.search_query.append(search_query)
    
    messages.append(HumanMessage(content=f"Search results: {search_results}"))
    return {"messages": messages, "next": "answer"}

def answer_step(state: AgentState) -> AgentState:
    """Generate answer based on search results."""
    messages = state['messages']
    response = llm.invoke(messages)
    messages.append(response)
    return {"messages": messages, "next": END}

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("search", search_agent)
workflow.add_node("answer", answer_step)

# Add edges
workflow.set_entry_point("search")
workflow.add_edge('search', 'answer')

# Compile the graph
chain = workflow.compile()

# Streamlit UI
st.title("🔍 Search Agent")
st.write("Ask me anything, and I'll search the internet for answers!")

user_input = st.text_input("Your question:", key="user_input")

if st.button("Search and Generate Answer"):
    if user_input:
        with st.spinner("Searching and analyzing..."):
            # Run the agent
            result = chain.invoke({
                "messages": [HumanMessage(content=user_input)],
                "next": "search"
            })
            
            # Extract search results and answer
            search_content = result['messages'][-2].content  # Get search results
            answer = result['messages'][-1].content
            
            # Add to chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.search_results.append(search_content)
            st.session_state.chat_history.append(("Assistant", answer))

# Display chat history with search results
for idx, (role, message) in enumerate(st.session_state.chat_history):
    if role == "You":
        st.write(f"👤 **You:** {message}")
        # Display search results after each question
        if idx//2 < len(st.session_state.search_results):
            with st.expander("🔍 View Search Results"):
                st.write(st.session_state.search_results[idx//2].replace("Search results: ", ""))
            with st.expander("🤖View Search Query"):
                st.write(st.session_state.search_query[idx//2])
    else:
        st.write(f"🤖 **Assistant:** {message}")

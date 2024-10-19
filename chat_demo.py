import streamlit as st
import os
import openai

def main():
    st.title("Chat Demo")
    # TODO: Add model selection dropdown

    # Together.ai API settings
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

    # Check if API key is available
    if not TOGETHER_API_KEY:
        st.error("Together.ai API key not found. Please set the TOGETHER_API_KEY environment variable.")
        st.stop()
    
    client = openai.OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Get user input
    if prompt := st.chat_input("What's up?"):
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model='meta-llama/Llama-3-8b-chat-hf',
                messages=[
                    {'role': m['role'], 'content': m['content']} for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

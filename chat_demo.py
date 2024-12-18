import streamlit as st
import os
import openai

MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
]

def on_model_change():
    st.session_state.model_changed = True

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

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = MODELS[0]

    if "model_changed" not in st.session_state:
        st.session_state.model_changed = False

    # Model selection dropdown
    with st.sidebar:
        st.subheader("Model Selection")
        st.session_state.new_model = st.selectbox(
            "Select a model",
            MODELS,
            index=MODELS.index(st.session_state.selected_model),
            key="model_select",
            on_change=on_model_change,
        )

        if st.session_state.model_changed:
            st.warning("Changing the model will reset the chat conversation. Are you sure you want to proceed?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, change model"):
                    st.session_state.selected_model = st.session_state.new_model
                    st.session_state.messages = []
                    st.session_state.model_changed = False
                    st.rerun()
            with col2:
                if st.button("No, keep current model"):
                    st.session_state.model_changed = False
                    st.rerun()

        # Temperature slider
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.01)

        # Top-p
        top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.01)

        # Max tokens
        max_tokens = int(st.number_input("Max tokens", min_value=1, max_value=4096, value=128))

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("What's up?"):
            with st.chat_message('user'):
                st.markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=st.session_state.selected_model,
                    messages=[
                        {'role': m['role'], 'content': m['content']} for m in st.session_state.messages
                    ],
                    stream=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens, # max_completion_tokens seems not work in the together.ai API
                )
                response = st.write_stream(stream)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    bottom_container = st._bottom.container()
    # Get user input
    with bottom_container:
        # Display currently selected model
        st.caption(f"Currently using model: **{st.session_state.selected_model}**")

if __name__ == "__main__":
    main()

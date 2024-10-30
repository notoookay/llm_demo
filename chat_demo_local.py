import streamlit as st
import os
# Remove OpenAI and Together.ai imports
# import openai

# Add necessary imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import plotly.express as px

# Update the models list with local models
MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    # Add more local models as needed
]

def on_model_change():
    st.session_state.model_changed = True

def main():
    st.title("Chat Demo (Local Models)")

    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = MODELS[0]
    if "model_changed" not in st.session_state:
        st.session_state.model_changed = False
    if "tokenizer" not in st.session_state or "model" not in st.session_state or st.session_state.model_changed:
        with st.spinner("Loading model..."):
            # Load tokenizer and model
            st.session_state.tokenizer = AutoTokenizer.from_pretrained(st.session_state.selected_model)
            st.session_state.model = AutoModelForCausalLM.from_pretrained(
                st.session_state.selected_model, output_attentions=True
            )
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
                    st.experimental_rerun()
            with col2:
                if st.button("No, keep current model"):
                    st.session_state.model_changed = False
                    st.experimental_rerun()

        # Temperature slider
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.01)

        # Max tokens
        max_tokens = int(st.number_input("Max tokens", min_value=1, max_value=1024, value=128))

    # Function to generate response and get attentions
    def generate_response(prompt):
        inputs = st.session_state.tokenizer.encode(prompt, return_tensors='pt')
        outputs = st.session_state.model.generate(
            inputs,
            max_length=inputs.shape[1] + max_tokens,
            temperature=temperature,
            output_attentions=True,
            return_dict_in_generate=True,
        )
        response_tokens = outputs.sequences[0][inputs.shape[1]:]
        response = st.session_state.tokenizer.decode(response_tokens, skip_special_tokens=True)
        attentions = outputs.attentions  # Tuple of attention weights
        return response, attentions, inputs

    # Function to visualize attentions
    def visualize_attentions(attentions, input_ids):
        # Get attentions from the last layer
        last_layer_attentions = attentions[-1][0]  # Shape: (num_heads, seq_len, seq_len)
        # Average over all heads
        avg_attentions = last_layer_attentions.mean(dim=0).detach().numpy()
        tokens = st.session_state.tokenizer.convert_ids_to_tokens(input_ids[0])
        fig = px.imshow(
            avg_attentions,
            labels=dict(x="Tokens", y="Tokens", color="Attention"),
            x=tokens,
            y=tokens,
            title="Attention Heatmap",
        )
        st.sidebar.plotly_chart(fig, use_container_width=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("What's up?"):
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response, attentions, input_ids = generate_response(prompt)
                st.markdown(response)
            visualize_attentions(attentions, input_ids)
        st.session_state.messages.append({"role": "assistant", "content": response})

    bottom_container = st.container()
    with bottom_container:
        st.caption(f"Currently using model: **{st.session_state.selected_model}**")

if __name__ == "__main__":
    main()

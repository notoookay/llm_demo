import streamlit as st
import plotly.graph_objects as go
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="LLM Attention Visualization",
    layout="wide"
)

# Define available models
AVAILABLE_MODELS = {
    "GPT-2 (Small)": "gpt2",
    "GPT-2 (Medium)": "gpt2-medium",
    "DistilGPT-2": "distilgpt2"
}

# Initialize the model and tokenizer
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

# Add model selection to sidebar
st.sidebar.title("Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(AVAILABLE_MODELS.keys()),
    index=0
)

# Load the selected model
model_id = AVAILABLE_MODELS[selected_model_name]
tokenizer, model = load_model(model_id)

# Create the Streamlit interface
st.title("LLM Token-to-Token Attention Visualization")

# Display current model info
st.markdown(f"""
### Current Model: {selected_model_name}
- Model ID: `{model_id}`
- Number of attention heads: {model.config.n_head}
- Total parameters: {model.num_parameters():,}
""")

st.write("Enter text to visualize the mean attention patterns between tokens across all heads")

# Get user input
user_input = st.text_area("Enter your text:", value="Hello, how are you?", height=100)

def get_attention_weights(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attention weights
    attention = outputs.attentions
    
    # Convert attention weights to numpy arrays and take mean across heads
    attention_weights = [layer_attention[0].mean(dim=0).numpy() for layer_attention in attention]
    
    return attention_weights, tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

def plot_attention_matrix(attention_weights, tokens, layer):
    # Get mean attention weights for all tokens
    token_attention = attention_weights[layer]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=token_attention,
        x=tokens,  # to tokens
        y=tokens,  # from tokens
        colorscale='Viridis',
        colorbar=dict(title='Attention Weight'),
        text=token_attention,
        texttemplate='%{z:.2f}',  # Format to 2 decimal places
        textfont={"size": 15},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=f'Layer {layer + 1} Mean Attention Pattern (All Tokens)',
        xaxis_title="To Tokens",
        yaxis_title="From Tokens",
        height=600,
        width=800,
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed")  # Reverse y-axis to match conventional attention visualization
    )
    
    return fig

def plot_aggregated_attention(attention_weights, tokens):
    # Average attention weights across all layers
    aggregated_attention = np.mean([weights for weights in attention_weights], axis=0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=aggregated_attention,
        x=tokens,
        y=tokens,
        colorscale='Viridis',
        colorbar=dict(title='Mean Attention Weight'),
        text=aggregated_attention,
        texttemplate='%{z:.2f}',  # Format to 2 decimal places
        textfont={"size": 15},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title='Aggregated Attention Pattern (Averaged Across All Layers)',
        xaxis_title="To Tokens",
        yaxis_title="From Tokens",
        height=600,
        width=800,
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def main():
    if st.button("Analyze"):
        with st.spinner("Processing..."):
            # Get attention weights and tokens
            attention_weights, tokens = get_attention_weights(user_input)
            
            # Show aggregated attention first
            st.subheader("Aggregated Attention Pattern")
            st.plotly_chart(
                plot_aggregated_attention(attention_weights, tokens),
                use_container_width=True
            )
            
            # Create tabs for each layer
            num_layers = len(attention_weights)
            tabs = st.tabs([f"Layer {i+1}" for i in range(num_layers)])
            
            # Plot attention matrices for each layer
            for i, tab in enumerate(tabs):
                with tab:
                    st.plotly_chart(
                        plot_attention_matrix(attention_weights, tokens, i),
                        use_container_width=True
                    )

            # Add explanation
            st.markdown("""
            ### How to interpret the visualization:
            - The heatmap shows the attention patterns between all tokens in the input
            - X-axis: "To" tokens (tokens being attended to)
            - Y-axis: "From" tokens (tokens doing the attending)
            - Color intensity: Stronger attention weight (averaged across all attention heads)
            - The first visualization shows the attention pattern averaged across all layers
            - Each tab shows the attention pattern for a specific layer
            - Diagonal patterns indicate self-attention (tokens attending to themselves)
            - Vertical stripes indicate tokens that receive high attention from many other tokens
            - Horizontal stripes indicate tokens that give high attention to many other tokens
            """)

if __name__ == "__main__":
    main()
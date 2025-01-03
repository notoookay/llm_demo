# LLM Application Demos

A collection of LLM-based applications for exploration and learning.

## Setup

```bash
conda env create -f environment.yaml
conda activate llm_demo
```

The [streamlit](https://streamlit.io/) is used for user interface.

## Run

To run each demo, you can run:

```bash
streamlit run chat_demo.py # Take chat demo as example
```

## Demos

### 1. Chatbot

An interactive chat interface using various LLM models through Together.ai API.

Features:

- [x] Multiple model options (Llama, Gemma, Mistral, Qwen)
- [x] Adjustable parameters (temperature, top-p, max tokens)
- [ ] Web search

Example:

![chat-demo-showcase](imgs/chat_demo_example.png)

### 2. Attention Visualization

A tool to visualize attention patterns in transformer-based language models.

Features:

- [x] Support for multiple transformer-based model variants (currently only tested on gpt-2, but should work fine for other models).
- [x] Layer-by-layer attention visualization
- [x] Aggregated attention patterns
- [x] Interactive heatmaps
- [X] Next token distribution

Example:

![layer-attention-showcase](imgs/layer_attention_visualization_exmaple.png)

### 3. Search Agent

An AI-powered search assistant that combines web search with LLM capabilities using LangChain and Together.ai API.

Features:

- [x] Web search integration via Tavily
- [x] Context-aware query generation
- [x] Conversation history tracking
- [x] Search query optimization
- [ ] Image and video support

Example usage:

1. Ask any question in the input field
2. The agent will:
   - Generate an optimized search query
   - Search the web for relevant information
   - Provide a comprehensive answer
3. View the search results and generated queries in expandable sections

Note: Requires Together.ai API key to be set as an environment variable.

### More demos may come......

## Suggestion

Welcome to any type of constructional suggestion.

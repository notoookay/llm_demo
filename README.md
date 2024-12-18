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

### More demos may come......

## Suggestion

Welcome to any type of constructional suggestion.

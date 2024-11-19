
# Project Description and Approach

## Project Overview
This project integrates **LLaMA**, **Knowledge Graphs (KGs)**, and **Retrieval-Augmented Generation (RAG)** to enhance software engineering tasks, including code generation, explanation, and documentation. The primary objective is to deliver precise, context-aware, and enriched responses to complex queries in the software engineering domain.

## Approach
The recommended approach leverages a hybrid methodology combining LLM inference and KG construction. This ensures the system benefits from structured knowledge (KG) and adaptive generative reasoning (LLM).

### 1. Initial KG Construction: Automated with LLMs
- **Objective**: Extract structured knowledge from unstructured data sources (e.g., documentation, forums, and codebases).
- **Methodology**:
  - Use LLaMA to extract entities and relationships from raw data.
  - Example: From a Python API documentation, extract triples like:
    - `(Flask, is_framework_of, Python)`
    - `(route(), is_function_of, Flask)`
  - Store these triples in a graph database like Neo4j or RDF triple stores.
- **Outcome**: An initial Knowledge Graph that organizes software knowledge.

### 2. Enhance KG with LLM-Assisted Link Prediction
- **Objective**: Fill gaps in the KG by predicting missing relationships.
- **Methodology**:
  - Use LLaMA embeddings to infer new connections based on existing data.
  - Example: If `TensorFlow` is linked to `Python`, and `Keras` is linked to `TensorFlow`, predict that `Keras` is also linked to `Python`.
- **Outcome**: Enriched KG with improved coverage.

### 3. Querying and Reasoning: RAG Framework
- **Objective**: Combine KG retrieval and LLM generative reasoning for context-aware responses.
- **Methodology**:
  - Use RAG to dynamically query the KG and provide context for LLaMA’s responses.
  - Example Query: "Explain the `route()` function in Flask."
  - Workflow:
    1. Retrieve relevant triples from the KG: `(route(), is_function_of, Flask)`.
    2. Use LLaMA to generate a detailed explanation based on this context.
- **Outcome**: Accurate and enriched responses to user queries.

### 4. Continuous Feedback Loop
- **Objective**: Enable iterative improvement of the KG and LLM capabilities.
- **Methodology**:
  - Analyze user interactions to identify gaps in the KG.
  - Update the KG using LLaMA for further extraction and refinement.
- **Outcome**: A dynamic system that evolves with new data and user needs.

## Key Features
- **Efficiency**: Automates knowledge extraction and organization.
- **Accuracy**: Uses the KG to ground LLaMA’s responses, avoiding hallucinations.
- **Adaptability**: Handles incomplete or ambiguous queries using generative reasoning.
- **Scalability**: Continuously enriches the KG as new data is added.

## Implementation Workflow
1. **Phase 1: KG Bootstrapping**
   - Use LLaMA to extract initial entities and relationships.
   - Define and implement the KG schema.
2. **Phase 2: LLM and KG Integration**
   - Enable LLaMA to query the KG during inference for contextual responses.
3. **Phase 3: Continuous Learning**
   - Refine the KG schema and update the graph as new data becomes available.

## Conclusion
This approach ensures the system delivers precise, enriched, and contextually aware insights for software engineering tasks. It combines the structured power of KGs with the generative flexibility of LLMs, creating a robust and adaptable tool.
"""


# [GraphLoom](https://drive.google.com/drive/folders/125z1exlm5WZHQAeCjUlHzaZLf1E48ukN) Demo


# Code Explanation and Usage

This project leverages the **LangChain** and **Ollama** libraries to generate concise explanations for provided code. Given code content and a question, it uses a language model to provide answers. Here’s a breakdown of how the script works.

## Project Setup

This script uses the following components:
- **LangChain** for chaining prompts and language model (LLM) interactions.
- **Ollama** as the LLM provider, specifically configured with a LLaMA model.
- **Requests** for fetching code content from a GitHub repository.

### Installation

1. Install the required packages:
    ```bash
    pip install langchain_community langchain_core requests
    ```


### Script Details

The script defines a **template** to prompt the language model, followed by a pipeline that loads code from a GitHub URL and uses a pre-trained language model to generate an explanation for it.

### Code Breakdown

```python
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import requests


template = '''
Answer the question below.
give a short answer

Here is the context: {context}
Question: {question}
Answer:
'''


url = 'https://raw.githubusercontent.com/kareem743/python/refs/heads/main/kareem.py'
response = requests.get(url)
file_content = response.text
print(file_content)


model = Ollama(model="llama3")


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


result = chain.invoke({"context":file_content, "question": "explain the code"})
print(result)
```

# potential data for fine tuning Llama
* [angie-chen55/python-github-code](https://huggingface.co/datasets/angie-chen55/python-github-code)
* [koutch/stackoverflow_python](https://huggingface.co/datasets/koutch/stackoverflow_python)
* [Arjun-G-Ravi/Python-codes](https://huggingface.co/datasets/Arjun-G-Ravi/Python-codes)


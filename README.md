# GraphLoom Demo

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

2. Set up the Ollama model with the specific LLaMA configuration:
    ```python
    model = Ollama(model="llama3")
    ```

### Script Details

The script defines a **template** to prompt the language model, followed by a pipeline that loads code from a GitHub URL and uses a pre-trained language model to generate an explanation for it.

### Code Breakdown

```python
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import requests

# Template to format the question and code context
template = '''
Answer the question below.
give a short answer

Here is the context: {context}
Question: {question}
Answer:
'''

# Fetching code from GitHub repository
url = 'https://raw.githubusercontent.com/kareem743/python/refs/heads/main/kareem.py'
response = requests.get(url)
file_content = response.text
print(file_content)

# Setting up the LLM model
model = Ollama(model="llama3")

# Preparing the prompt template
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Running the model with the code context and question
result = chain.invoke({"context":file_content, "question": "explain the code"})
print(result)




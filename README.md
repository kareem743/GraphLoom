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


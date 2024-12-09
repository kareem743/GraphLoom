
# [GraphLoom](https://drive.google.com/drive/folders/125z1exlm5WZHQAeCjUlHzaZLf1E48ukN) 

## Project Overview
Our project focuses on integrating Large Language Models (LLMs) with Knowledge Graphs (KGs) and Retrieval-Augmented Generation (RAG) techniques to create a hybrid system for enhanced information retrieval and decision making in software engineering. Our system tries to improve accuracy and relevancy of selected tasks: code generation, explanation, and documentation with a focus on security for all of them. We aim to deliver precise, context-aware, and enriched responses to complex queries, thus bridging the gap between abstract AI capabilities and practical software development needs.
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


## Code Generation and Documentation with Knowledge Graphs and LLMs

### 1. Code Generation

Goal:
Automatically generate functional code snippets using structured knowledge from Knowledge Graphs (KG).

### Process
1. **Input**: KG contains structured information about programming concepts, libraries, or frameworks
   - Example Triple: `(Flask, has_function, route())`
   - Description: The `route()` function defines URL routes for HTTP requests

2. **Action**: LLaMA uses this context to generate functional code

3. **Output**: Working code examples

### Example

**KG Input**: 
```
Flask → has_function → route()
```

**LLaMA Output**:
```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to Flask!"

if __name__ == "__main__":
    app.run()
```

### Use Case
Quick generation of boilerplate code or snippets that developers can use as a starting point.

## 2. Code Explanation

Goal:
Provide detailed and understandable explanations for code elements using KG knowledge.

### Process
1. **Input**: User query about code elements
2. **Action**: 
   - KG retrieves relevant information
   - LLaMA generates natural language explanation
3. **Output**: Clear, user-friendly explanation

### Example

**Query**: "What does the `route()` function in Flask do?"

**KG Output**: 
- `route()` is a decorator associated with HTTP routing

**LLaMA Output**:
```
The route() function in Flask is a decorator that maps a specific URL to a Python function. 
When a user visits the specified URL, the associated function is executed, and its return 
value is sent back as the HTTP response.
```

### Use Case
Ideal for:
- Beginners learning a framework
- Troubleshooting unfamiliar code

## 3. Documentation Generation

Goal
Automate creation of comprehensive documentation by combining KG data with LLM capabilities.

### Process
1. **Input**: Metadata from KG about code elements
   ```
   Class: Flask
   Function: route()
   Description: Defines HTTP routing
   ```

2. **Action**: LLM generates documentation including:
   - Class/function descriptions
   - Usage examples
   - Parameter details
   - Return values
   - Exceptions

3. **Output**: Well-structured documentation

### Example

**KG Input**:
```
Function: route()
Description: Used for HTTP routing
Example: @app.route("/home") def home(): ...
```

**LLaMA Output**:

### Flask route() Function

The `route()` function in Flask is a decorator that defines URL routes for HTTP requests.

**Parameters:**
- `rule` (str): The URL path
- `methods` (list, optional): HTTP methods allowed (e.g., GET, POST)

**Example Usage:**
```python
@app.route("/home")
def home():
    return "This is the home page."
```

**Notes:**
- Multiple HTTP methods can be specified using the `methods` argument
- Routes must be unique within the application
- Parameters can be included in routes using `<parameter_name>`



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


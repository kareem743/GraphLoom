
# [GraphLoom](https://drive.google.com/drive/folders/125z1exlm5WZHQAeCjUlHzaZLf1E48ukN) 

## Project Overview
Our project focuses on integrating Large Language Models (LLMs) with Knowledge Graphs (KGs) and Retrieval-Augmented Generation (RAG) techniques to create a hybrid system for enhanced information retrieval and decision making in software engineering. Our system tries to improve accuracy and relevancy of selected tasks: code generation, explanation, and documentation with a focus on security for all of them. We aim to deliver precise, context-aware, and enriched responses to complex queries, thus bridging the gap between abstract AI capabilities and practical software development needs.
## Approach
The recommended approach leverages a hybrid methodology combining LLM inference and KG construction. This ensures the system benefits from structured knowledge (KG) and adaptive generative reasoning (LLM).

![Architecture diagram showing the system's workflow with GitHub code processing through AST to GraphDB, with LLM logic and SPARQL query layers, and a front-end interface for user interaction](path/to/your/image.png)

## Description

This diagram illustrates the architecture of our system which:
- Processes GitHub repository code through AST (Abstract Syntax Tree) preprocessing
- Stores generated graphs in GraphDB representing the repository structure
- Uses an LLM layer for processing user prompts and converting text to SPARQL queries
- Provides a front-end interface for users to interact with the system

The workflow consists of:
1. Generating SPARQL queries to retrieve AST sub-graphs
2. Combining data with semantically relevant prompts
3. Processing through the LLM logic layer to provide intelligent responses to user queries




# [GraphLoom](https://drive.google.com/drive/folders/125z1exlm5WZHQAeCjUlHzaZLf1E48ukN) 

## Project Overview
Our project focuses on integrating Large Language Models (LLMs) with Knowledge Graphs (KGs) and Retrieval-Augmented Generation (RAG) techniques to create a hybrid system for enhanced information retrieval and decision making in software engineering. Our system tries to improve accuracy and relevancy of selected tasks: code generation, explanation, and documentation with a focus on security for all of them. We aim to deliver precise, context-aware, and enriched responses to complex queries, thus bridging the gap between abstract AI capabilities and practical software development needs.
## Approach
The recommended approach leverages a hybrid methodology combining LLM inference and KG construction. This ensures the system benefits from structured knowledge (KG) and adaptive generative reasoning (LLM).

![Architecture diagram showing the system's workflow with GitHub code processing through AST to GraphDB, with LLM logic and SPARQL query layers, and a front-end interface for user interaction](./Intelligent_GraphRag_Integration_for_Enhanced_Guidance-Page-1.drawio.png)

## Evaluation](Evaluation)

Below is a summarized table of the evaluation metrics for the "Intelligent GraphRag Integration for Enhanced Guidance" system, covering the overall project evaluation and the specific "text to Cypher" component, based on the previous responses. The table includes the metric name, type, description, and application context, making it easy to compare and understand their purposes.

| **Metric**            | **Type**          | **Description**                                                                                   | **Application Context**                                                                                   |
|-----------------------|-------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Precision**         | Retrieval         | Measures the proportion of retrieved code structures (e.g., functions, dependencies) that are relevant. | Assesses the quality of subgraphs retrieved from Neo4j for queries like "Explain the merge_sort function." |
| **Recall**            | Retrieval         | Measures the proportion of relevant code structures retrieved out of all relevant ones available. | Ensures completeness of retrieved subgraphs, critical for understanding full code context.               |
| **Mean Reciprocal Rank (MRR)** | Retrieval         | Evaluates the rank of the first relevant result in retrieved subgraphs, averaged across queries.  | Checks how quickly the system finds the most relevant code piece, useful for multi-hop queries.          |
| **Faithfulness**      | Generation        | Assesses if LLM outputs (e.g., explanations, code) are grounded in retrieved subgraphs, avoiding hallucinations. | Ensures generated responses align with actual codebase structure, reducing errors.                       |
| **BLEU Score**        | Generation / Translation | Compares n-grams of generated text (e.g., code, Cypher queries) to reference text, ranges 0-1.   | Evaluates textual similarity of generated code/docs or Cypher queries against expected outputs.          |
| **ROUGE Score**       | Generation        | Measures overlap of n-grams, longest common subsequences between generated and reference text.    | Assesses quality of generated documentation or explanations, complementing BLEU.                         |
| **ExactMatch (EM)**   | Execution / Translation | Compares execution results of generated vs. reference Cypher queries or code, binary 0 or 1.     | Ensures functional correctness of Cypher queries or generated code snippets when run on Neo4j.           |
| **F1 Score**          | Task-Specific     | Balances precision and recall for specific tasks (e.g., code retrieval, generation accuracy).     | Validates overall performance on real-world tasks like code explanation or documentation generation.     |
| **Time to Task Completion** | User Study    | Measures time taken by developers to complete tasks using the system vs. baselines.              | Assesses productivity gains for junior and senior developers, e.g., onboarding or code comprehension.    |
| **User Satisfaction** | User Study        | Qualitative feedback from developers on system usability and helpfulness via surveys/interviews.  | Ensures practical utility and developer trust in the system’s outputs for real projects.                 |

### Notes
- **Retrieval Metrics** (Precision, Recall, MRR): Focus on the Graph-RAG component, evaluating how well Neo4j retrieves relevant code subgraphs.
- **Generation Metrics** (Faithfulness, BLEU, ROUGE): Apply to LLM outputs (explanations, code, docs), ensuring accuracy and relevance.
- **Translation Metrics** (BLEU, ExactMatch): Specifically evaluate the "text to Cypher" translation, checking both textual and functional accuracy.
- **Task-Specific/User Metrics** (F1, Time, Satisfaction): Assess overall system impact on developer tasks and experience, aligning with project goals.


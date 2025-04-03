
# [GraphLoom](https://drive.google.com/drive/folders/125z1exlm5WZHQAeCjUlHzaZLf1E48ukN) 

## Project Overview
Our project focuses on integrating Large Language Models (LLMs) with Knowledge Graphs (KGs) and Retrieval-Augmented Generation (RAG) techniques to create a hybrid system for enhanced information retrieval and decision making in software engineering. Our system tries to improve accuracy and relevancy of selected tasks: code generation, explanation, and documentation with a focus on security for all of them. We aim to deliver precise, context-aware, and enriched responses to complex queries, thus bridging the gap between abstract AI capabilities and practical software development needs.
## Approach
The recommended approach leverages a hybrid methodology combining LLM inference and KG construction. This ensures the system benefits from structured knowledge (KG) and adaptive generative reasoning (LLM).

![Architecture diagram showing the system's workflow with GitHub code processing through AST to GraphDB, with LLM logic and SPARQL query layers, and a front-end interface for user interaction](./Intelligent_GraphRag_Integration_for_Enhanced_Guidance-Page-1.drawio.png)

## [How to Evaluate?](Evaluation)

Here’s a summarized table of all evaluation metrics discussed throughout the chat for the "Intelligent GraphRag Integration for Enhanced Guidance" system, covering the overall project, the "text to Cypher" component, and the knowledge graph itself. This consolidates metrics from retrieval, generation, user studies, task-specific evaluation, text-to-Cypher translation, and knowledge graph quality, providing a comprehensive overview as of April 3, 2025.

| **Metric**                     | **Type**          | **Description**                                                                                   | **Application Context**                                                                                   |
|-------------------------------|-------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Precision**                 | [Retrieval](Evaluation/Evaluation_retrieval\generation.md)         | Proportion of retrieved code structures (e.g., functions, dependencies) that are relevant.         | Assesses quality of subgraphs retrieved from Neo4j for queries like "Explain the merge_sort function."    |
| **Recall**                    | [Retrieval](Evaluation/Evaluation_retrieval\generation.md)         | Proportion of relevant code structures retrieved out of all relevant ones available.              | Ensures completeness of retrieved subgraphs, critical for full code context understanding.               |
| **Mean Reciprocal Rank (MRR)**| [Retrieval](Evaluation/Evaluation_retrieval\generation.md)        | Evaluates rank of the first relevant result in retrieved subgraphs, averaged across queries.       | Measures how quickly the system finds the most relevant code piece, useful for multi-hop queries.        |
| **Faithfulness**              | [Generation](Evaluation/Evaluation_retrieval\generation.md)        | Assesses if LLM outputs (e.g., explanations, code) are grounded in retrieved subgraphs, avoiding hallucinations. | Ensures generated responses align with codebase structure, reducing errors.                              |
| **BLEU Score**                | [Generation](Evaluation/Evaluation_retrieval\generation.md) / [Translation](Text2cypher.md) | Compares n-grams of generated text (e.g., code, Cypher queries) to reference text, ranges 0-1.   | Evaluates textual similarity of generated code/docs or Cypher queries against expected outputs.          |
| **ROUGE Score**               |[Generation](Evaluation/Evaluation_retrieval\generation.md)        | Measures overlap of n-grams, longest common subsequences between generated and reference text.     | Assesses quality of generated documentation or explanations, complementing BLEU.                         |
| **ExactMatch (EM)**           | Execution / [Translation](Text2cypher.md | Compares execution results of generated vs. reference Cypher queries or code, binary 0 or 1.     | Ensures functional correctness of Cypher queries or generated code snippets when run on Neo4j.           |
| **F1 Score**                  | Task-Specific     | Balances precision and recall for specific tasks (e.g., code retrieval, generation accuracy).      | Validates overall performance on real-world tasks like code explanation or documentation generation.     |
| **Time to Task Completion**   | User Study        | Measures time taken by developers to complete tasks using the system vs. baselines.               | Assesses productivity gains for junior and senior developers, e.g., onboarding or code comprehension.    |
| **User Satisfaction**         | User Study        | Qualitative feedback from developers on system usability and helpfulness via surveys/interviews.   | Ensures practical utility and developer trust in the system’s outputs for real projects.                 |

##Structural (KG)
| **Metric**                     | **Type**          | **Description**                                                                                   | **Application Context** 
| **Node Count Accuracy**       | Structural (KG)   | Compares number of nodes (e.g., functions, classes) in graph to source code, measures coverage.    | Ensures all code entities are captured in the knowledge graph, critical for completeness.                |
| **Edge Accuracy (Relationships)** | Structural (KG) | Verifies relationships (e.g., CALLS, INHERITS) in graph match those extracted from code.         | Checks if call graphs and inheritance hierarchies are correct, essential for query accuracy.             |
| **Call Graph Coverage**       | Completeness (KG) | Percentage of function calls in code represented as edges in graph.                               | Measures how well dynamic relationships are captured, impacting query results like "functions called by X."|
| **Schema Adherence**          | Semantic (KG)     | Assesses if nodes and edges follow expected types (e.g., Function node has name property).         | Ensures graph follows codebase ontology, supporting reliable querying and retrieval.                     |
| **Query Success Rate**        | Utility (KG)      | Percentage of test queries (e.g., "Show callers of function") returning expected results.          | Validates graph utility for developer tasks, ensuring it supports GraphRag retrieval effectively.         |

### Notes
- **Retrieval Metrics** (Precision, Recall, MRR): Evaluate the Graph-RAG retrieval from Neo4j, ensuring relevant code subgraphs are retrieved.
- **Generation Metrics** (Faithfulness, BLEU, ROUGE): Assess LLM outputs (explanations, code, docs) for accuracy and relevance.
- **Translation Metrics** (BLEU, ExactMatch): Focus on "text to Cypher" translation, checking textual and functional accuracy of Cypher queries.
- **Task-Specific/User Metrics** (F1, Time, Satisfaction): Measure overall system impact on developer tasks and experience.
- **Knowledge Graph (KG) Metrics** (Node Count Accuracy, Edge Accuracy, Call Graph Coverage, Schema Adherence, Query Success Rate): Assess the graph’s accuracy, completeness, semantic quality, and utility, ensuring it supports the system’s foundation.


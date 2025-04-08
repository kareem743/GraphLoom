
# [GraphLoom](https://drive.google.com/drive/folders/125z1exlm5WZHQAeCjUlHzaZLf1E48ukN) 

## Project Overview
Our project focuses on integrating Large Language Models (LLMs) with Knowledge Graphs (KGs) and Retrieval-Augmented Generation (RAG) techniques to create a hybrid system for enhanced information retrieval and decision making in software engineering. Our system tries to improve accuracy and relevancy of selected tasks: code generation, explanation, and documentation with a focus on security for all of them. We aim to deliver precise, context-aware, and enriched responses to complex queries, thus bridging the gap between abstract AI capabilities and practical software development needs.
## Approach
The recommended approach leverages a hybrid methodology combining LLM inference and KG construction. This ensures the system benefits from structured knowledge (KG) and adaptive generative reasoning (LLM).

![Architecture diagram showing the system's workflow with GitHub code processing through AST to GraphDB, with LLM logic and SPARQL query layers, and a front-end interface for user interaction](Images/Intelligent_GraphRag_Integration_for_Enhanced_Guidance-Page-1.drawio.png)
<details>
  <summary>## How can we imagine it?</summary>
## How can we imagine it?
```python
# example.py

def printfun():
  """This function just prints a message."""
  print("Hello from printfun!") # Built-in print

def loop():
  """This function loops 3 times and calls printfun."""
  for i in range(3):
     printfun() # Call to printfun

def run_loops(num_times):
  """Runs the loop function multiple times."""
  print(f"Running the loop {num_times} times.") # Built-in print
  for _ in range(num_times):
      loop() # Call to loop
  return f"Completed {num_times} loop runs."
```

Now, let's break down what goes where:

**1. What's Exactly in the Knowledge Graph (KG)**

The KG captures the *defined structure, components, and explicit connections* within the code. Think of it as the blueprint or map.

*   **Nodes (Entities):** Represent the core components defined in the code.
    *   **Node 1: `printfun`**
        *   `id`: `example.py:printfun` (Unique Identifier)
        *   `type`: Function
        *   `name`: "printfun"
        *   `file_path`: "example.py"
        *   `signature`: `() -> None` (No parameters, returns None)
        *   `docstring`: "This function just prints a message."
        *   `LLM-summary` (Example): "Outputs a static greeting."
    *   **Node 2: `loop`**
        *   `id`: `example.py:loop`
        *   `type`: Function
        *   `name`: "loop"
        *   `file_path`: "example.py"
        *   `signature`: `() -> None`
        *   `docstring`: "This function loops 3 times and calls printfun."
        *   `LLM-summary` (Example): "Iterates three times, invoking 'printfun' in each iteration."
    *   **Node 3: `run_loops`**
        *   `id`: `example.py:run_loops`
        *   `type`: Function
        *   `name`: "run_loops"
        *   `file_path`: "example.py"
        *   `signature`: `(num_times: Any) -> str` (Takes one parameter, returns a string. Type `Any` if not inferred precisely).
        *   `docstring`: "Runs the loop function multiple times."
        *   `LLM-summary` (Example): "Executes the 'loop' function a specified number of times and returns a completion message."
    *   *(Potentially)* **Node 4: `num_times` (Parameter)**
        *   `id`: `example.py:run_loops:param:num_times`
        *   `type`: Parameter
        *   `name`: "num_times"
        *   `belongs_to`: Node 3 (`run_loops`)
        *   `param_type`: `Any` (or inferred type)

*   **Edges (Relationships):** Represent the *connections* between the nodes.
    *   **Edge 1: `CALLS`**
        *   `Source`: Node 2 (`loop`)
        *   `Target`: Node 1 (`printfun`)
        *   `type`: CALLS
        *   `location`: `example.py:loop:line_7` (Approximate line number)
    *   **Edge 2: `CALLS`**
        *   `Source`: Node 3 (`run_loops`)
        *   `Target`: Node 2 (`loop`)
        *   `type`: CALLS
        *   `location`: `example.py:run_loops:line_13` (Approximate line number)
    *   *(Potentially)* **Edge 3: `HAS_PARAMETER`**
        *   `Source`: Node 3 (`run_loops`)
        *   `Target`: Node 4 (`num_times`)
        *   `type`: HAS_PARAMETER
    *   *(Potentially)* **Edge 4: `RETURNS`**
        *   `Source`: Node 3 (`run_loops`)
        *   `Target`: (Literal or Node representing `str` type)
        *   `type`: RETURNS

**KG Summary:** The KG stores discrete facts: "Function `loop` exists," "Function `loop` calls function `printfun`," "Function `run_loops` takes a parameter," "Function `run_loops` returns a string." It's excellent for precise, structural queries like "What functions call `loop`?" or "What parameters does `run_loops` take?".

---

**2. What's Exactly in the Vector Store (VS)**

The Vector Store captures the *semantic meaning* or *essence* of code blocks and their descriptions. Think of it as capturing the *vibe* or *purpose* for similarity matching.

*   **Code Chunk Embeddings:** Vectors representing the *meaning* of the executable code within each function.
    *   **Embedding 1 (`printfun` Code):** A vector generated from the code `print("Hello from printfun!")`. This vector numerically represents the concept of "printing a specific string".
        *   `Associated Data`: `chunk_id: example.py:printfun`, raw code.
    *   **Embedding 2 (`loop` Code):** A vector generated from `for i in range(3): printfun()`. This represents "looping a fixed number of times and calling another specific function (`printfun`)".
        *   `Associated Data`: `chunk_id: example.py:loop`, raw code.
    *   **Embedding 3 (`run_loops` Code):** A vector generated from the code inside `run_loops`: `print(f"Running the loop {num_times} times."); for _ in range(num_times): loop(); return f"Completed {num_times} loop runs."`. This represents "printing a dynamic message, looping based on a parameter, calling another specific function (`loop`), and returning a result string".
        *   `Associated Data`: `chunk_id: example.py:run_loops`, raw code.

*   **Entity Description Embeddings:** Vectors representing the *meaning* of the natural language descriptions (docstrings, summaries).
    *   **Embedding 4 (`printfun` Description):** A vector generated from text like "This function just prints a message. Outputs a static greeting.". Represents the concept described.
        *   `Associated Data`: `entity_id: example.py:printfun`.
    *   **Embedding 5 (`loop` Description):** A vector generated from text like "This function loops 3 times and calls printfun. Iterates three times, invoking 'printfun' in each iteration.". Represents the described concept.
        *   `Associated Data`: `entity_id: example.py:loop`.
    *   **Embedding 6 (`run_loops` Description):** A vector generated from text like "Runs the loop function multiple times. Executes the 'loop' function a specified number of times and returns a completion message.". Represents the described concept.
        *   `Associated Data`: `entity_id: example.py:run_loops`.

**VS Summary:** The Vector Store contains numerical fingerprints (embeddings) that allow for semantic similarity searches. It doesn't know explicitly that `run_loops` calls `loop`, but the *embedding* for `run_loops`'s code will likely be somewhat similar to other code that involves looping and calling functions, and its *description embedding* will be similar to other descriptions about executing tasks multiple times. It's great for fuzzy or conceptual queries like "Find code that repeatedly executes a task" or "Show me functions related to printing status messages".

---

**How Retrieval Uses Both:**

*   **Query:** "What calls the `loop` function?"
    *   **Action:** Primarily a KG query. Find nodes with `CALLS` edges pointing *to* the `loop` node.
    *   **Result:** The `run_loops` node.
*   **Query:** "Find code examples for running something multiple times based on input."
    *   **Action:** Embed the query. Primarily a VS search. Compare query embedding against Code Chunk Embeddings and potentially Description Embeddings.
    *   **Result:** Embedding 3 (`run_loops` code) and Embedding 6 (`run_loops` description) would likely score high.
*   **Query:** "Show me the function that runs `printfun` three times and how it's used."
    *   **Action (Hybrid):**
        1.  VS Search (optional): Find entities related to "`printfun` three times" -> might highlight `loop`.
        2.  KG Query: Find the `loop` node. Find incoming `CALLS` edges to `loop` -> finds `run_loops`. Find outgoing `CALLS` edges from `loop` -> finds `printfun`.
    *   **Result:** Identify `loop` as the direct function, `run_loops` as its caller, and `printfun` as what it calls, presenting the relevant code and connections.



<details>
  <summary>Click to expand</summary>
  
## [How to Evaluate?](Evaluation)

Here’s a summarized table of all evaluation metrics covering the overall project, the "text to Cypher" component, and the knowledge graph itself. This consolidates metrics from retrieval, generation, user studies, task-specific evaluation, text-to-Cypher translation, and knowledge graph quality.
| **Metric**                     | **Type**          | **Description**                                                                                   | **Application Context**                                                                                   |
|-------------------------------|-------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Precision**                 | [Retrieval](Evaluation/Evaluation_retrieval_generation.md)         | Proportion of retrieved code structures (e.g., functions, dependencies) that are relevant.         | Assesses quality of subgraphs retrieved from Neo4j for queries like "Explain the merge_sort function."    |
| **Recall**                    | [Retrieval](Evaluation/Evaluation_retrieval_generation.md)         | Proportion of relevant code structures retrieved out of all relevant ones available.              | Ensures completeness of retrieved subgraphs, critical for full code context understanding.               |
| **Mean Reciprocal Rank (MRR)**| [Retrieval](Evaluation/Evaluation_retrieval_generation.md)        | Evaluates rank of the first relevant result in retrieved subgraphs, averaged across queries.       | Measures how quickly the system finds the most relevant code piece, useful for multi-hop queries.        |
| **Faithfulness**              | [Generation](Evaluation/Evaluation_retrieval_generation.md)        | Assesses if LLM outputs (e.g., explanations, code) are grounded in retrieved subgraphs, avoiding hallucinations. | Ensures generated responses align with codebase structure, reducing errors.                              |
| **BLEU Score**                | [Generation](Evaluation/Evaluation_retrieval_generation.md) / [Translation](Text2cypher.md) | Compares n-grams of generated text (e.g., code, Cypher queries) to reference text, ranges 0-1.   | Evaluates textual similarity of generated code/docs or Cypher queries against expected outputs.          |
| **ROUGE Score**               |[Generation](Evaluation/Evaluation_retrieval_generation.md)        | Measures overlap of n-grams, longest common subsequences between generated and reference text.     | Assesses quality of generated documentation or explanations, complementing BLEU.                         |
| **ExactMatch (EM)**           | Execution / [Translation](Text2cypher.md) | Compares execution results of generated vs. reference Cypher queries or code, binary 0 or 1.     | Ensures functional correctness of Cypher queries or generated code snippets when run on Neo4j.           |


## Structural (KG) [Evaluation](Evaluation/KG_Evaluation.md)
| **Metric**                     | **Type**          | **Description**                                                                                   | **Application Context**                                                                                   |
|-------------------------------|-------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
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


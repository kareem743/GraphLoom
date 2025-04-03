

### Should We Evaluate the Knowledge Graph?

Yes, we should evaluate the knowledge graph to make sure it correctly and fully captures the codebase, which is crucial for helping developers with tasks like understanding code or generating documentation. This step ensures the system works well and reduces errors, especially for junior developers onboarding or seniors navigating complex code.

#### How to Evaluate It

We’ll check the knowledge graph in a few ways:
- **Compare to Source Code**: Look at the graph’s nodes (like functions, classes) and edges (like calls, inherits) against the actual code. For example, if the code has 100 functions, the graph should too, and relationships like "function A calls B" should match.
- **Check Structure**: Make sure nodes and edges follow our expected rules, like every function having a name and parameters, to ensure it’s logically sound.
- **Test with Queries**: Run developer-like questions, like "Show functions called by process_data," and see if the graph gives the right answers, ensuring it’s useful for real tasks.

An unexpected detail is that evaluating the graph could also reveal gaps, like missing legacy modules, which might help improve the system for future engineering tasks beyond code/

---



#### How to Evaluate It?
Evaluating the knowledge graph involves assessing its accuracy, completeness, semantic quality, and utility for tasks, adapted from standard practices for knowledge graphs in software engineering. The process includes:

1. **Accuracy Evaluation**:
   - **Metric**: Node Count Accuracy, comparing the number of nodes (e.g., functions, classes) in the graph to those extracted from source code using static analysis tools like PyCG for Python or Clang for C++. For example, if the codebase has 100 functions, the graph should have 100 Function nodes.
   - **Metric**: Edge Accuracy (Relationships), verifying relationships like CALLS, INHERITS, and CONTAINS match those in the code. This can be done by extracting the call graph or inheritance hierarchy from code and comparing it to the graph, checking for mismatches.
   - **Approach**: Manually sample for small projects or use automated comparison for large ones, ensuring relationships like "process_data calls calculate_sum" are correctly represented as edges.

2. **Completeness Evaluation**:
   - **Metric**: Call Graph Coverage, measuring the percentage of function calls in code represented as edges in the graph. For instance, if code analysis shows 200 calls and the graph has 180, coverage is 90%.
   - **Approach**: Compare file counts, module coverage, and ensure all source code files are nodes, addressing potential gaps like legacy modules not included in the graph.

3. **Semantic Quality Evaluation**:
   - **Metric**: Schema Adherence, assessing if nodes and edges follow expected types and properties, such as Function nodes having name, signature, and start_line properties. This ensures the graph aligns with the codebase ontology.
   - **Approach**: Use schema validation tools in Neo4j to check if all nodes are labeled correctly (e.g., File, Class) and have required properties, ensuring logical consistency.

4. **Utility Evaluation**:
   - **Metric**: Query Success Rate, measuring the percentage of test queries returning expected results, such as "Show callers of function X" or "What does DataLoader class depend on?".
   - **Approach**: Define a set of developer-like queries, execute them on the graph, and compare results to expected outputs from code analysis, ensuring the graph supports GraphRag retrieval effectively.

These metrics are inspired by structural quality metrics from [Structural Quality Metrics to Evaluate Knowledge Graphs, arXiv](https://arxiv.org/abs/2211.10011), which include ontology detail (classes, properties) and usage, adapted for software engineering. For example, the arXiv paper mentions measuring how actively classes and properties are used, which translates to ensuring Function nodes and CALLS edges are utilized in queries, aligning with our system’s needs.

| **Metric**                     | **Type**          | **Description**                                                                                   | **Application Context**                                                                                   |
|-------------------------------|-------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Node Count Accuracy**        | Structural        | Compares number of nodes (e.g., functions, classes) in graph to source code, measures coverage.   | Ensures all code entities (e.g., functions) are captured, critical for completeness.                     |
| **Edge Accuracy (Relationships)** | Structural    | Verifies relationships (e.g., CALLS, INHERITS) in graph match those extracted from code.          | Checks if call graphs and inheritance hierarchies are correct, essential for query accuracy.              |
| **Call Graph Coverage**        | Completeness      | Percentage of function calls in code represented as edges in graph.                               | Measures how well dynamic relationships are captured, impacting query results like "functions called by X."|
| **Schema Adherence**           | Semantic          | Assesses if nodes and edges follow expected types (e.g., Function node has name property).        | Ensures graph follows codebase ontology, supporting reliable querying and retrieval.                     |
| **Query Success Rate**         | Utility           | Percentage of test queries (e.g., "Show callers of function") returning expected results.         | Validates graph utility for developer tasks, ensuring it supports GraphRag retrieval effectively.         |



### Key Citations
- [Defining a Knowledge Graph Development Process Through a Systematic Review, ACM](https://dl.acm.org/doi/10.1145/3522586)
- [Structural Quality Metrics to Evaluate Knowledge Graphs, arXiv](https://arxiv.org/abs/2211.10011)
- [A Practical Framework for Evaluating the Quality of Knowledge Graph, ResearchGate](https://www.researchgate.net/publication/338361155_A_Practical_Framework_for_Evaluating_the_Quality_of_Knowledge_Graph)
- [Application of knowledge graph in software engineering field: A systematic literature review, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950584923001829)
- [Knowledge graph quality control: A survey, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667325821001655)
- [Implementing Graph RAG Using Knowledge Graphs, IBM](https://www.ibm.com/think/tutorials/knowledge-graph-rag)
- [The Rise and Evolution of RAG in 2024 A Year in Review, RAGFlow](https://ragflow.io/blog/the-rise-and-evolution-of-rag-in-2024-a-year-in-review)


# What is the Text to Cypher Part or Translation?
The "text to Cypher" part is about turning simple questions, like "Show me functions called by process_data," into special commands called Cypher queries. These queries help search a database (Neo4j) that stores the structure of code, like files and functions, making it easier for developers to find what they need. We use a smart AI model, called a Large Language Model (LLM), to do this translation, bridging what developers ask in plain English with what the database understands.

### How Can We Do It?
To make this work, we have a couple of options:
- **Quick Start**: Use a ready-made LLM like GPT-4, giving it details about our code database (schema) and asking it to translate, say, via a tool like LangChain. This is good for testing.
- **Long-Term Solution**: Fine-tune an open-source model, like Llama 3, using special datasets that pair English questions with Cypher queries, available at [GitHub repo](https://github.com/neo4j-labs/text2cypher). This means training it on our specific code setup for better accuracy, using guides from [Neo4j blog](https://neo4j.com/blog/developer/fine-tuned-text2cypher-2024-model/).

An unexpected detail is that fine-tuning can be done on cloud platforms like HuggingFace or RunPod, making it easier to handle large datasets without heavy local setup.


---
Implementing text-to-Cypher translation involves several approaches, each suited to different stages of development:

- **Prototyping with Pre-trained LLMs**: Use a pre-trained model like GPT-4, as described in [Medium post](https://medium.com/neo4j/generating-cypher-queries-with-chatgpt-4-on-any-graph-schema-a57d7082a7e7), by connecting to Neo4j, fetching the schema, and prompting the model. For example, a Python class like `Neo4jGPTQuery` initializes with Neo4j credentials and OpenAI API key, then uses the schema to generate queries. This is ideal for initial testing, leveraging LangChain for integration, as seen in [Medium post](https://medium.com/@muthoju.pavan/demystifying-natural-language-to-cypher-conversion-with-openai-neo4j-langchain-and-langsmith-2dbecb1e2ce9).

- **Fine-Tuning for Production**: For long-term use, fine-tune an open-source LLM like Llama 3 using datasets from [GitHub repo](https://github.com/neo4j-labs/text2cypher), which provides natural language and Cypher pairs with graph information. The finetuning section mentions notebooks and scripts, likely involving supervised learning with loss functions on predicted vs. actual Cypher, using HuggingFace Transformers on platforms like RunPod, as noted in [Neo4j blog](https://neo4j.com/blog/developer/fine-tuned-text2cypher-2024-model/). This adapts the model to our codebase schema, improving accuracy for specific queries.

- **Alternative Approaches**: NeoDash's extension allows toggling between English and Cypher, using configured LLMs like VertexAI, as mentioned in [Reddit post](https://www.reddit.com/r/Neo4j/comments/1en76r1/ways_and_tools_to_generate_the_cypher_from_plain/), where users can correct queries for feedback loops. This is less scalable but useful for initial validation.

The evidence leans toward fine-tuning as best practice in 2025, given the November 2024 Neo4j blog on fine-tuned models, highlighting improvements over baselines using the Neo4j Text2Cypher (2024) Dataset. An unexpected detail is the use of cloud platforms, reducing local setup complexity, potentially impacting deployment costs and privacy considerations.

| Approach                  | Method                            | Tools/Platforms                     | Suitability                     |
|---------------------------|-----------------------------------|-------------------------------------|---------------------------------|
| Prototyping               | Pre-trained LLM (e.g., GPT-4)     | LangChain, OpenAI API               | Quick testing, initial validation |
| Fine-Tuning for Production| Supervised learning on datasets   | HuggingFace, RunPod, neo4j-labs notebooks | Long-term, schema-specific accuracy |
| Alternative (NeoDash)     | Configured LLM with user feedback | NeoDash, VertexAI                   | Initial validation, less scalable |

#### How to Evaluate It?
Evaluating the text-to-Cypher translation involves assessing both the accuracy of the generated Cypher and its functional correctness, aligning with the project's goal of reducing hallucinations and enhancing developer productivity. Research suggests two main methods, supported by recent Neo4j resources:

- **Translation-Based Evaluation**: Use BLEU score for textual comparison, measuring how closely the predicted Cypher matches the reference query, as highlighted in [Neo4j blog](https://neo4j.com/blog/developer/fine-tuned-text2cypher-2024-model/). BLEU, a standard metric for translation tasks, compares n-grams, focusing on lexical similarity. For example, if the reference is `MATCH (f:Function)-[:CALLS]->(c) RETURN f.name`, and the prediction is similar, BLEU scores high. This is complemented by ROUGE, another textual metric, though BLEU is emphasized in the blog.

- **Execution-Based Evaluation**: Use ExactMatch by executing both predicted and reference Cypher queries on Neo4j and comparing results, as noted in [Neo4j blog](https://neo4j.com/developer-blog/benchmarking-neo4j-text2cypher-dataset/). This converts outputs to strings in the same order and format, checking if they match, ensuring functional correctness. For instance, if both queries return the same list of function names, ExactMatch is 1; otherwise, 0.

- **User-Centric Evaluation**: Given the neo4j-labs/text2cypher repo mentions real-world scenarios, conduct user studies with developers, asking them to assess if the generated Cypher aligns with their intent, as seen in [Reddit post](https://www.reddit.com/r/Neo4j/comments/1en76r1/ways_and_tools_to_generate_the_cypher_from_plain/). This qualitative feedback ensures practical utility, especially for junior developers onboarding and senior engineers navigating code.

The evidence leans toward BLEU and ExactMatch as standard metrics, with the November 2024 Neo4j blog focusing on these for the Text2Cypher (2024) Dataset, achieving scores like 30% match ratio for top models. An unexpected detail is the potential for combining with RAG evaluation metrics like MRR from [Pinecone blog](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/), though primarily for retrieval, not translation.

| Metric                  | Type          | Description                                                                 | Application to Text-to-Cypher                     |
|-------------------------|---------------|-----------------------------------------------------------------------------|----------------------------------------------------|
| BLEU Score              | Translation   | Compares n-grams of predicted vs. reference Cypher, ranges 0-1              | Assesses lexical similarity of generated queries   |
| ExactMatch              | Execution     | Compares execution results of predicted vs. reference, binary 0 or 1        | Ensures functional correctness of queries          |
| User Feedback           | Qualitative   | Developer assessment of query correctness and utility                       | Validates real-world applicability                 |




---

### Key Citations
- [collection of text2cypher datasets, evaluations, and finetuning instructions - neo4j-labs/text2cypher](https://github.com/neo4j-labs/text2cypher)
- [Introducing the Fine-Tuned Neo4j Text2Cypher (2024) Model - Neo4j Blog](https://neo4j.com/blog/developer/fine-tuned-text2cypher-2024-model/)
- [Benchmarking Using the Neo4j Text2Cypher (2024) Dataset - Neo4j Blog](https://neo4j.com/developer-blog/benchmarking-neo4j-text2cypher-dataset/)
- [Generating Cypher Queries With ChatGPT 4 on Any Graph Schema - Medium](https://medium.com/neo4j/generating-cypher-queries-with-chatgpt-4-on-any-graph-schema-a57d7082a7e7)
- [Demystifying Natural Language to Cypher Conversion with OpenAI, Neo4j, LangChain, and LangSmith - Medium](https://medium.com/@muthoju.pavan/demystifying-natural-language-to-cypher-conversion-with-openai-neo4j-langchain-and-langsmith-2dbecb1e2ce9)
- [Text2Cypher - Natural Language Queries - NeoDash](https://neo4j.com/labs/neodash/2.4/user-guide/extensions/natural-language-queries/)
- [ways and tools to generate the cypher from plain text questions in llm? - Reddit](https://www.reddit.com/r/Neo4j/comments/1en76r1/ways_and_tools_to_generate_the_cypher_from_plain/)
- [RAG Evaluation: Don’t let customers tell you first - Pinecone](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- [Implementing Graph RAG Using Knowledge Graphs - IBM](https://www.ibm.com/think/tutorials/knowledge-graph-rag)
- [The Rise and Evolution of RAG in 2024 A Year in Review - RAGFlow](https://ragflow.io/blog/the-rise-and-evolution-of-rag-in-2024-a-year-in-review)

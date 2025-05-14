import os
import logging
from typing import List, Dict, Any
from evaluate_retrieval import evaluate_retrieval,evaluate_generated_answer

# --- ChromaDB Imports ---
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import QueryResult  # For type hinting

# --- Neo4j Import ---
from neo4j import GraphDatabase, Driver

# --- Project-Specific Imports ---
from config import EMBEDDING_MODEL_NAME, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from llm_client import get_llm_client, get_fallback_llm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHROMA_PATH_PREFIX = "./chroma_db_struct_chunk_"

# --- Initialize Embedding Function and LLM Client ---
try:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME,trust_remote_code=True)
    logging.info(f"Successfully initialized embedding function with {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize SentenceTransformerEmbeddingFunction: {e}. Using None.")
    ef = None

try:
    LLM_CLIENT = get_llm_client()
    logging.info("Successfully initialized LLM client .")
except Exception as e:
    logging.warning(f"Failed to initialize LLM client: {e}. Using fallback LLM.")
    LLM_CLIENT = get_fallback_llm()

# --- Helper Function for Neo4j Queries ---
def run_neo4j_query(driver: Driver, query: str, params: Dict = None) -> List[Dict]:
    """Runs a read query against Neo4j and returns results."""
    if params is None:
        params = {}
    results = []
    try:
        with driver.session(database="neo4j") as session:
            response = session.run(query, params)
            # Convert Neo4j Records to dictionaries for easier handling
            for record in response:
                results.append(record.data())
            return results
    except Exception as e:
        logging.error(f"Neo4j query failed: {e}")
        logging.error(f"  Query: {query}")
        logging.error(f"  Params: {params}")
        return []  # Return empty list on failure

# --- Core RAG Implementation ---

def retrieve_context(
        query: str,
        index_id: str,
        chroma_client: chromadb.Client,
        neo4j_driver: Driver,
        embedding_function: Any,
        llm_client: Any = None,  # Added LLM client for query expansion
        top_k_entities: int = 40,
        top_k_chunks: int = 15,
        graph_hops: int = 1,
        reranking_enabled: bool = True,
        use_query_expansion: bool = True,

) -> Dict[str, Any]:
    """
    Enhanced retrieval function with query expansion, hybrid search, and re-ranking.
    Retrieves relevant context from ChromaDB and Neo4j based on the query.
    """
    if embedding_function is None:
        logging.error("Embedding function is not available. Cannot perform vector search.")
        return {"error": "Missing embedding function"}

    entities_collection_name = f"{index_id}_entities"
    chunks_collection_name = f"{index_id}_chunks"
    retrieved_context = {
        "query": query,
        "expanded_query": query,  # Will be updated if query expansion is used
        "top_entities": [],
        "related_graph_nodes": [],
        "related_graph_relationships": [],
        "relevant_chunks": [],
        "error": None
    }

    # Step 0: Query Expansion (if enabled and LLM client provided)
    if use_query_expansion and llm_client:
        try:
            logging.info("Step 0: Expanding query...")
            expansion_prompt = f"""
            Your task is to expand the following user query about code to improve retrieval.
            Generate an expanded version that includes relevant technical terms, possible class names,
            method names, or concepts that might be relevant.

            Original query: {query}

            Expanded query:
            """
            llm = llm_client()
            expanded_query_response = llm.invoke(expansion_prompt)
            expanded_query = expanded_query_response.content if hasattr(expanded_query_response, 'content') else str(
                expanded_query_response)
            # Clean up the expanded query (remove artifacts from LLM formatting)
            expanded_query = expanded_query.strip().split('\n')[0]

            logging.info(f"  Original query: '{query}'")
            logging.info(f"  Expanded query: '{expanded_query}'")
            retrieved_context["expanded_query"] = expanded_query

            # Use both original and expanded query
            query_for_search = f"{query} {expanded_query}"
        except Exception as e:
            logging.warning(f"Query expansion failed: {e}. Using original query.")
            query_for_search = query
    else:
        query_for_search = query

    # Get Chroma collections
    try:
        entities_collection = chroma_client.get_collection(name=entities_collection_name,
                                                           embedding_function=embedding_function)
        chunks_collection = chroma_client.get_collection(name=chunks_collection_name,
                                                         embedding_function=embedding_function)
    except Exception as e:
        retrieved_context["error"] = f"Error getting ChromaDB collections: {e}"
        logging.error(retrieved_context["error"])
        # Try to continue with partial functionality if possible
        if "entities_collection" not in locals() and "chunks_collection" not in locals():
            return retrieved_context

    # 1. Query Entities with Hybrid Search
    logging.info(f"Step 1: Querying '{entities_collection_name}' with hybrid search...")
    try:
        # Implementation depends on ChromaDB version - newer versions support hybrid search directly
        try:
            # Try with direct hybrid search if available
            entity_results: QueryResult = entities_collection.query(
                query_texts=[query_for_search],
                n_results=top_k_entities,
                include=["metadatas", "documents", "distances"],

            )
        except Exception as e:
            logging.warning(f"search not available: {e}. Falling back to vector search.")


        if entity_results and entity_results.get('ids') and entity_results['ids'][0]:
            retrieved_context["top_entities"] = [
                {"id": id_, "metadata": meta, "description": doc, "distance": dist}
                for id_, meta, doc, dist in zip(
                    entity_results['ids'][0],
                    entity_results['metadatas'][0],
                    entity_results['documents'][0],
                    entity_results.get('distances', [[0] * len(entity_results['ids'][0])])[0]
                )
            ]
            logging.info(f"  Found {len(retrieved_context['top_entities'])} initial entities.")
        else:
            logging.info("  No relevant entities found in search.")
    except Exception as e:
        retrieved_context["error"] = f"Error querying entities collection: {e}"
        logging.error(retrieved_context["error"])
        # Continue execution to try other retrieval methods

    # 2. Dynamic Graph Exploration based on query complexity
    logging.info(f"Step 2: Querying Neo4j graph with dynamic exploration...")

    # Adjust graph_hops dynamically based on query complexity
    query_terms = len(query_for_search.split())
    if query_terms > 15:  # Complex query
        dynamic_hops = min(graph_hops + 1, 3)  # Increase hops but cap at 3
    elif query_terms < 5:  # Simple query
        dynamic_hops = max(graph_hops - 1, 1)  # Decrease hops but minimum 1
    else:
        dynamic_hops = graph_hops

    logging.info(f"  Using {dynamic_hops} hop(s) for graph exploration (adjusted from {graph_hops})")

    subgraph_node_ids = []
    if retrieved_context["top_entities"]:
        initial_entity_ids = [e['id'] for e in retrieved_context["top_entities"]]

        # Query 1: Get IDs of nodes within the hop distance with relationship type filtering
        nodes_query = f"""
        MATCH (start_node:KGNode) WHERE start_node.id IN $entity_ids
        CALL apoc.path.subgraphNodes(start_node, {{
            maxLevel: {dynamic_hops},
            relationshipFilter: "CONTAINS>|CONTAINS_METHOD>|CALLS>|IMPORTS>"
        }}) YIELD node
        RETURN COLLECT(DISTINCT node.id) AS node_ids
        """
        params = {"entity_ids": initial_entity_ids}
        node_id_results = run_neo4j_query(neo4j_driver, nodes_query, params)

        if node_id_results and node_id_results[0].get("node_ids"):
            subgraph_node_ids = node_id_results[0]["node_ids"]
            logging.info(f"  Identified {len(subgraph_node_ids)} potentially relevant nodes in Neo4j subgraph.")

            # Query 2: Get details for these nodes with improved property selection
            node_details_query = """
            MATCH (n:KGNode) WHERE n.id IN $node_ids
            RETURN n.id AS id, 
                   labels(n) AS labels, 
                   {
                     name: n.name,
                     description: n.description,
                     docstring: n.docstring,
                     type: n.type,
                     source_file: n.source_file,
                     start_line: n.start_line,
                     end_line: n.end_line
                   } AS properties
            """
            node_details = run_neo4j_query(neo4j_driver, node_details_query, {"node_ids": subgraph_node_ids})
            retrieved_context["related_graph_nodes"] = node_details
            logging.info(f"  Retrieved details for {len(node_details)} nodes.")

            # Query 3: Get relationships with priority scoring
            rels_query = """
            MATCH (n1:KGNode)
            WHERE n1.id IN $node_ids
            MATCH (n1)-[rel]-()
            WITH n1, count(rel) AS degree
            MATCH (n1)-[r]->(n2:KGNode)
            WHERE n2.id IN $node_ids
            // Calculate relationship priority score based on type and node degrees
            WITH n1.id AS source_id,
                 type(r) AS rel_type,
                 degree,
                 properties(r) AS rel_properties,
                 n2.id AS target_id,
                 CASE type(r)
                   WHEN 'CONTAINS_METHOD' THEN 5
                   WHEN 'CALLS' THEN 4
                   WHEN 'CONTAINS' THEN 3
                   
                   WHEN 'IMPORTS' THEN 2
                   ELSE 1
                 END AS priority_score
            RETURN source_id, rel_type, degree, rel_properties, target_id, priority_score
            ORDER BY priority_score DESC
            """
            relationships = run_neo4j_query(neo4j_driver, rels_query, {"node_ids": subgraph_node_ids})
            retrieved_context["related_graph_relationships"] = relationships
            logging.info(f"  Found {len(relationships)} relationships within the subgraph.")
        else:
            logging.info("  No subgraph nodes found starting from initial entities.")
    else:
        logging.info("  Skipping Neo4j query as no initial entities were found.")

    # 3. Enhanced Chunk Retrieval with Multi-Strategy Approach
    logging.info(f"Step 3: Enhanced code chunk retrieval with multiple strategies...")

    # Strategy 1: Use the query directly
    query_texts_for_chunks = [query_for_search]

    # Strategy 2: Add descriptions from top entities
    if retrieved_context["top_entities"]:
        # Take descriptions from top 5 entities only to avoid noise
        query_texts_for_chunks.extend([e['description'] for e in retrieved_context["top_entities"][:5]])

    # Strategy 3: Use specialized entity-based search terms
    # Extract method names, class names, function names from entities for more targeted search
    if retrieved_context["top_entities"]:
        specialized_terms = []
        for entity in retrieved_context["top_entities"][:3]:  # Top 3 to avoid noise
            meta = entity.get('metadata', {})
            entity_type = meta.get('entity_type', '')
            name = meta.get('name', '')
            if name and entity_type:
                specialized_terms.append(f"{entity_type} {name}")
        if specialized_terms:
            query_texts_for_chunks.extend(specialized_terms)

    # Strategy 4: Use descriptions from nodes in the graph
    for node_info in retrieved_context["related_graph_nodes"]:
        props = node_info.get('properties', {})
        desc = props.get('description') or props.get('docstring')
        name = props.get('name')
        labels = node_info.get('labels', [])
        if desc and len(desc) > 20:  # Only use meaningful descriptions
            query_texts_for_chunks.append(desc[:300])  # Limit length
        elif name:
            node_type = next((l for l in labels if l != 'KGNode'), 'Node')
            query_texts_for_chunks.append(f"{node_type} {name}")

    # Remove duplicates and empty strings
    unique_query_texts = list(set(filter(None, query_texts_for_chunks)))
    logging.info(f"  Using {len(unique_query_texts)} unique texts to find relevant chunks.")

    if unique_query_texts:
        try:
            # Try to perform a hybrid search if available
            try:
                chunk_results: QueryResult = chunks_collection.query(
                    query_texts=unique_query_texts[:min(5, len(unique_query_texts))],  # Limit to top 5 query texts
                    n_results=top_k_chunks,
                    include=["metadatas", "documents", "distances"],
                )
            except Exception as e:
                logging.warning(f"Hybrid chunk search not available: {e}. Falling back to vector search.")
                chunk_results: QueryResult = chunks_collection.query(
                    query_texts=unique_query_texts[:min(5, len(unique_query_texts))],
                    n_results=top_k_chunks,
                    include=["metadatas", "documents", "distances"]
                )

            # Process results
            unique_chunks = {}
            if chunk_results and chunk_results.get('ids'):
                for i in range(len(chunk_results['ids'])):
                    for j, (chunk_id, meta, doc) in enumerate(zip(
                            chunk_results['ids'][i],
                            chunk_results['metadatas'][i],
                            chunk_results['documents'][i]
                    )):
                        # Get distance if available
                        distance = chunk_results.get('distances', [[0] * len(chunk_results['ids'][i])])[i][j]
                        if chunk_id not in unique_chunks:
                            unique_chunks[chunk_id] = {
                                "id": chunk_id,
                                "metadata": meta,
                                "document": doc,
                                "distance": distance
                            }
                        else:
                            # If we see the same chunk again with a better score, update it
                            if distance < unique_chunks[chunk_id]["distance"]:
                                unique_chunks[chunk_id]["distance"] = distance

            # Convert to list for further processing
            chunk_list = list(unique_chunks.values())

            # 4. Re-ranking step (if enabled)
            if reranking_enabled and llm_client and len(chunk_list) > 1:
                logging.info("Step 4: Re-ranking chunks with contextual relevance...")
                try:
                    # Simple re-ranking using LLM
                    # In production, you might use a specialized re-ranker like BERT Cross-Encoder
                    reranking_results = []

                    # Prepare chunked reranking to avoid token limits
                    chunks_to_rerank = chunk_list[:min(10, len(chunk_list))]  # Limit to top 10 chunks for reranking

                    reranking_prompt = f"""
                    You are a code understanding assistant. Your task is to rate the relevance of code chunks 
                    to a user query on a scale of 1-10 (10 being most relevant).

                    User query: {query}

                    For each code chunk, provide a single number (1-10) as your rating.

                    {'-' * 40}
                    """

                    for i, chunk in enumerate(chunks_to_rerank):
                        doc = chunk.get('document', '')
                        meta = chunk.get('metadata', {})
                        source = meta.get('source_file', 'Unknown')
                        reranking_prompt += f"""
                        Chunk {i + 1}: (Source: {source})
                        ```python
                        {doc[:500]}{"..." if len(doc) > 500 else ""}
                        ```

                        Relevance score (1-10) for Chunk {i + 1}: 
                        """

                    # Get relevance scores from LLM
                    reranking_response = llm_client.invoke(reranking_prompt)
                    reranking_text = reranking_response.content if hasattr(reranking_response, 'content') else str(
                        reranking_response)

                    # Extract scores using regex
                    import re
                    scores = re.findall(r"Chunk \d+:.*?Relevance score.*?(\d+)", reranking_text, re.DOTALL)

                    # Apply scores if we got them
                    if scores and len(scores) == len(chunks_to_rerank):
                        for i, score in enumerate(scores):
                            chunks_to_rerank[i]["relevance_score"] = int(score)

                        # Resort based on relevance scores
                        chunks_to_rerank.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

                        # Replace the initial chunks with reranked ones
                        chunk_list[:len(chunks_to_rerank)] = chunks_to_rerank
                        logging.info(f"  Successfully reranked {len(chunks_to_rerank)} chunks.")
                    else:
                        logging.warning(
                            f"  Reranking failed: Couldn't extract enough scores ({len(scores)} found, {len(chunks_to_rerank)} needed)")

                except Exception as e:
                    logging.warning(f"  Reranking failed: {e}")

            # Store final results
            retrieved_context["relevant_chunks"] = chunk_list
            logging.info(f"  Found {len(retrieved_context['relevant_chunks'])} unique relevant chunks.")

        except Exception as e:
            retrieved_context["error"] = f"Error querying chunks collection: {e}"
            logging.error(retrieved_context["error"])
    else:
        logging.info("  No query texts derived for retrieving chunks.")

    # 5. Fallback strategies if we found no relevant context
    if not retrieved_context["top_entities"] and not retrieved_context["relevant_chunks"]:
        logging.warning("No context found. Attempting fallback strategies...")

        # Fallback 1: Direct keyword search in Neo4j
        try:
            # Extract keywords from query
            keywords = [word for word in re.findall(r'\w+', query.lower())
                        if word not in ['the', 'and', 'or', 'in', 'is', 'to', 'a', 'of', 'for', 'with', 'how']]

            if keywords:
                keyword_query = """
                MATCH (n:KGNode)
                WHERE any(keyword IN $keywords WHERE 
                    toLower(n.name) CONTAINS keyword OR 
                    toLower(coalesce(n.description,'')) CONTAINS keyword OR
                    toLower(coalesce(n.docstring,'')) CONTAINS keyword)
                RETURN n.id AS id, 
                       labels(n) AS labels, 
                       {
                         name: n.name,
                         description: n.description,
                         docstring: n.docstring,
                         type: n.type,
                         source_file: n.source_file,
                         start_line: n.start_line,
                         end_line: n.end_line
                       } AS properties
                LIMIT 50
                """
                keyword_results = run_neo4j_query(neo4j_driver, keyword_query, {"keywords": keywords})

                if keyword_results:
                    logging.info(f"  Fallback: Found {len(keyword_results)} nodes via keyword search in Neo4j.")
                    retrieved_context["related_graph_nodes"] = keyword_results
        except Exception as e:
            logging.warning(f"  Fallback Neo4j keyword search failed: {e}")

    logging.info("Context retrieval phase complete (including graph structure).")
    return retrieved_context
# --- Helper Function to Format Graph Context ---
def format_graph_context(nodes: List[Dict], relationships: List[Dict], max_rels: int = 15) -> str:
    """
    Formats graph nodes and relationships into a readable string for the LLM,
    including properties on the relationships.
    """
    if not nodes and not relationships:
        return ""

    # Create a lookup for node details
    node_details = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            logging.warning(f"Skipping node without ID: {node}")
            continue
        props = node.get("properties", {})
        labels = node.get("labels", ["Node"])
        node_type = "Unknown"
        type_priority = ["Method", "Function", "Class", "Module", "Call", "Import"]
        for p_type in type_priority:
            if p_type in labels:
                node_type = p_type
                break
        else:
            node_type = labels[0] if labels else "Node"
        name = props.get("name", node_id)
        node_details[node_id] = {"name": name, "type": node_type}

    # Formatting Relationships
    formatted_rels = []
    rel_count = 0
    if relationships:
        for i, rel in enumerate(relationships):
            if rel_count >= max_rels:
                formatted_rels.append("... (more relationships shown than limit)")
                break
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            rel_type = rel.get("rel_type", "RELATED_TO")
            rel_props = rel.get("rel_properties", {})
            if not source_id or not target_id:
                logging.warning(f"Skipping relationship with missing source/target ID: {rel}")
                continue
            source_info = node_details.get(source_id, {"name": source_id, "type": "Unknown/External"})
            target_info = node_details.get(target_id, {"name": target_id, "type": "Unknown/External"})
            props_str = ""
            if rel_props:
                prop_items = [f"{k}: {str(v)[:50]}{'...' if len(str(v)) > 50 else ''}" for k, v in rel_props.items()]
                if prop_items:
                    props_str = " {" + ", ".join(prop_items) + "}"
            formatted_rels.append(
                f"- {source_info['name']} ({source_info['type']}) "
                f"-[{rel_type}{props_str}]->"
                f" {target_info['name']} ({target_info['type']})"
            )
            rel_count += 1

    # Construct final output string
    output_parts = []
    if formatted_rels:
        output_parts.append("=== Knowledge Graph Structure (Relationships with Properties) ===")
        output_parts.extend(formatted_rels)
        output_parts.append("="*20)
    elif nodes:
        output_parts.append("=== Knowledge Graph Structure (Nodes Found) ===")
        for node_id, details in node_details.items():
            output_parts.append(f"- {details['name']} ({details['type']})")
        if relationships is not None:
            output_parts.append("(No direct relationships retrieved between these specific nodes)")
        output_parts.append("="*20)

    return "\n".join(output_parts) + "\n" if output_parts else ""

# --- Generate Response ---
def generate_response(
        query: str,
        retrieved_context: Dict[str, Any],
        llm_client: Any
) -> str:
    """
    Generates a response using the LLM based on the query and retrieved context,
    including formatted graph information.
    """
    logging.info("Generating response...")
    if retrieved_context.get("error") and not (
            retrieved_context["top_entities"] or retrieved_context["relevant_chunks"]):
        logging.warning("Cannot generate response due to error during retrieval and no context found.")
        return f"Sorry, I encountered an error retrieving context: {retrieved_context['error']}"

    # Combine Context
    context_pieces = []
    context_pieces.append(f"User Query: {query}\n")

    # Add Top Entities
    if retrieved_context["top_entities"]:
        context_pieces.append("=== Potentially Relevant Code Entities (from Vector Search) ===")
        for i, entity in enumerate(retrieved_context["top_entities"]):
            meta = entity.get('metadata', {})
            desc = entity.get('description', 'N/A')

            # Improved entity description formatting - remove redundancy and ID
            entity_name = meta.get('name', 'N/A')
            entity_type = meta.get('entity_type', 'N/A')
            file_location = f"{meta.get('source_file', 'N/A')}:{meta.get('start_line', 'N/A')}"

            # Create a concise, non-repetitive description
            context_pieces.append(
                f"Entity {i + 1}: {entity_type} '{entity_name}' in {file_location}\n"
                f"Description: {desc[:300]}..."
            )
        context_pieces.append("=" * 20 + "\n")

    # Add Formatted Graph Context
    graph_context_str = format_graph_context(
        retrieved_context.get("related_graph_nodes", []),
        retrieved_context.get("related_graph_relationships", []),
        max_rels=20
    )
    if graph_context_str:
        context_pieces.append(graph_context_str)

    # Add Relevant Code Snippets
    if retrieved_context["relevant_chunks"]:
        context_pieces.append("=== Relevant Code Snippets (from Vector Search) ===")
        max_chunks = 15
        seen_ids = set()
        unique_chunks = []
        for chunk in retrieved_context["relevant_chunks"]:
            cid = chunk.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(chunk)
            if len(unique_chunks) >= max_chunks:
                break
        for i, chunk in enumerate(unique_chunks, start=1):
            meta = chunk.get('metadata', {})
            doc = chunk.get('document', '')
            context_pieces.append(
                f"Code Snippet {i} "
                f"(Source: {meta.get('source_file', 'N/A')}, "
                f"Lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}):\n"
                f"```python\n{doc}\n```"
            )
        total = len(retrieved_context["relevant_chunks"])
        kept = len(unique_chunks)
        if total > kept:
            context_pieces.append(f"... (plus {total - kept} more unique snippets not shown)")
        context_pieces.append("=" * 20 + "\n")

    if not context_pieces[1:]:
        logging.warning("No specific context found for the query.")
        context_pieces.append("No specific code context was found related to the query.")

    combined_context = "\n".join(context_pieces)
    max_context_chars = 55000
    if len(combined_context) > max_context_chars:
        logging.warning(f"Combined context length ({len(combined_context)}) exceeds limit ({max_context_chars}). Truncating.")
        combined_context = combined_context[:max_context_chars] + "\n... (Context Truncated)"

    # Formulate Final Prompt
    system_prompt = (
        "You are an AI assistant specialized in understanding and answering questions about a Python codebase. "
        "Use the provided context, which includes relevant code entities (from vector search), code snippets, and knowledge graph relationships (showing how code elements are connected, e.g., calls, containment), to answer the user's query accurately and concisely. "
        "Pay close attention to the graph structure section to understand dependencies and architecture. "
        "Synthesize information from all parts of the context. If asked to count items, be precise based on the evidence. "
        "If the context does not contain the answer, state that the information is not available in the provided code details."
    )

    final_prompt = f"""{system_prompt}
    
## Provided Context:
{combined_context}

## User Query:
{query}

## Answer:
"""
    print(final_prompt)
    logging.info("Sending prompt to LLM...")
    try:
        response = llm_client.invoke(final_prompt)
        logging.info("Received response from LLM.")
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            logging.warning(f"LLM returned unexpected type: {type(response)}. Converting to string.")
            return str(response)
    except Exception as e:
        logging.error(f"LLM invocation failed: {e}")
        return "Sorry, I encountered an error generating the response."

# --- Main Execution Example ---
if __name__ == "__main__":
    if ef is None:
        logging.critical("Embedding Function (ef) is not available. Cannot run RAG pipeline.")
    else:
        # Configuration for this run
        TARGET_INDEX_ID = "py_structchunk_neo4j_v2_option2"
        TARGET_CHROMA_DB_PATH = f"{CHROMA_PATH_PREFIX}{TARGET_INDEX_ID}"
        while True:
            USER_QUERY = input("What is your question? (Type 'exit' to quit): ")
            if USER_QUERY.lower() == 'exit':
                break

            logging.info(f"===== Starting RAG Pipeline for Index: {TARGET_INDEX_ID} =====")
            neo4j_driver_instance = None
            chroma_client_instance = None

            try:
                # Initialize Clients
                logging.info("Initializing Neo4j driver...")
                neo4j_driver_instance = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                neo4j_driver_instance.verify_connectivity()
                logging.info("Neo4j connection successful.")

                logging.info(f"Initializing ChromaDB client at path: {TARGET_CHROMA_DB_PATH}")
                if not os.path.exists(TARGET_CHROMA_DB_PATH):
                    raise FileNotFoundError(f"ChromaDB path does not exist: {TARGET_CHROMA_DB_PATH}. Please run the indexing script first.")
                chroma_client_instance = chromadb.PersistentClient(path=TARGET_CHROMA_DB_PATH)
                chroma_client_instance.heartbeat()
                logging.info("ChromaDB client initialized successfully.")

                # Run Retrieval
                retrieved_data = retrieve_context(
                    query=USER_QUERY,
                    index_id=TARGET_INDEX_ID,
                    chroma_client=chroma_client_instance,
                    neo4j_driver=neo4j_driver_instance,
                    embedding_function=ef,
                    top_k_entities=30,
                    top_k_chunks=15,
                    graph_hops=3
                )




                # Generate Response
                if retrieved_data and not retrieved_data.get("error"):
                    final_answer = generate_response(
                        query=USER_QUERY,
                        retrieved_context=retrieved_data,
                        llm_client=LLM_CLIENT
                    )
                    print("\n===== Final Answer =====")
                    print(final_answer)
                    llm_client = get_llm_client()
                    evaluate_retrieval(retrieved_data, USER_QUERY, llm_client)

                    evaluate_generated_answer(query=USER_QUERY,generated_answer=final_answer,llm_client=llm_client,retrieved_context=retrieved_data)
                elif retrieved_data and retrieved_data.get("error"):
                    print(f"\n===== Error during Retrieval =====")
                    print(retrieved_data.get("error"))
                else:
                    print(f"\n===== Error =====")
                    print("Failed to retrieve context.")

            except FileNotFoundError as e:
                logging.critical(str(e))
            except Exception as e:
                logging.critical(f"An error occurred during the RAG pipeline: {e}", exc_info=True)

            finally:
                if neo4j_driver_instance:
                    logging.info("Closing Neo4j connection.")
                    neo4j_driver_instance.close()

        logging.info(f"===== RAG Pipeline Complete =====")
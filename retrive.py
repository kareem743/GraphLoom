
import os
import logging
from typing import List, Dict, Any

# --- ChromaDB Imports ---
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import QueryResult # For type hinting

# --- Neo4j Import ---
from neo4j import GraphDatabase, Driver

# --- LLM Client (reuse from your indexing script) ---
# Assume get_llm_client() and get_fallback_llm() are available
# from your_indexing_script import get_llm_client, get_fallback_llm

# --- Embedding Function (reuse from your indexing script) ---
# Assume EMBEDDING_MODEL_NAME is defined
# from your_indexing_script import EMBEDDING_MODEL_NAME
# Assume ef is initialized as in your indexing script
# ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

# --- Configuration (Should match indexing script & environment) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Make sure these are defined correctly based on your setup
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "abcd12345") # Use env var!

EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" # Or your model
CHROMA_PATH_PREFIX = "./chroma_db_struct_chunk_"

# --- Mock LLM and EF setup if running standalone ---
# If running this script separately, you might need to redefine these
# or import them properly from your main script file.
try:
    # Attempt to import from your main script file (replace 'your_indexing_script' actual filename)
    from python_code_processor import get_llm_client, get_fallback_llm, CodeParser # Assuming these are in 'main_processing_script.py'
    LLM_CLIENT = get_llm_client()
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        logging.info(f"Successfully initialized embedding function with {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logging.error(f"Failed to initialize SentenceTransformerEmbeddingFunction: {e}. Using None.")
        ef = None

except ImportError:
    print("Could not import from main processing script. Using placeholder LLM and EF=None.")
    print("Ensure the indexing script is runnable and in the Python path, or redefine components here.")
    LLM_CLIENT = get_fallback_llm() # Use fallback if import fails
    ef = None # Cannot proceed without embedding function easily

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
        return [] # Return empty list on failure

# --- Core RAG Implementation ---

# --- Helper Function for Neo4j Queries ---
# (Keep run_neo4j_query as is)

# --- Core RAG Implementation ---

def retrieve_context(
    query: str,
    index_id: str,
    chroma_client: chromadb.Client,
    neo4j_driver: Driver,
    embedding_function: Any, # Should be ChromaDB compatible embedding function
    top_k_entities: int = 4,
    top_k_chunks: int = 4,
    graph_hops: int = 1 # How many hops in Neo4j to explore from initial entities
) -> Dict[str, Any]:
    """
    Retrieves relevant context from ChromaDB and Neo4j based on the query.
    Includes graph structure (nodes and relationships).
    """
    if embedding_function is None:
        logging.error("Embedding function is not available. Cannot perform vector search.")
        return {"error": "Missing embedding function"}

    entities_collection_name = f"{index_id}_entities"
    chunks_collection_name = f"{index_id}_chunks"
    retrieved_context = {
        "query": query,
        "top_entities": [],
        "related_graph_nodes": [],      # Will store node details
        "related_graph_relationships": [], # NEW: Will store relationship details
        "relevant_chunks": [],
        "error": None
    }

    # ... (Get Chroma collections - keep as is) ...
    try:
        entities_collection = chroma_client.get_collection(name=entities_collection_name, embedding_function=embedding_function)
        chunks_collection = chroma_client.get_collection(name=chunks_collection_name, embedding_function=embedding_function)
    except Exception as e:
        retrieved_context["error"] = f"Error getting ChromaDB collections: {e}"
        logging.error(retrieved_context["error"])
        return retrieved_context


    # 1. Query Entities Vector DB (Keep as is)
    logging.info(f"Step 1: Querying '{entities_collection_name}' for top {top_k_entities} entities...")
    try:
        entity_results: QueryResult = entities_collection.query(
            query_texts=[query],
            n_results=top_k_entities,
            include=["metadatas", "documents"]
        )
        if entity_results and entity_results.get('ids') and entity_results['ids'][0]:
            retrieved_context["top_entities"] = [
                {"id": id_, "metadata": meta, "description": doc}
                for id_, meta, doc in zip(entity_results['ids'][0], entity_results['metadatas'][0], entity_results['documents'][0])
            ]
            logging.info(f"  Found {len(retrieved_context['top_entities'])} initial entities.")
        else:
            logging.info("  No relevant entities found in vector search.")
    except Exception as e:
        retrieved_context["error"] = f"Error querying entities collection: {e}"
        logging.error(retrieved_context["error"])
        # Decide if you want to continue without entities

    # 2. Query Knowledge Graph (Neo4j) - MODIFIED TO GET RELATIONSHIPS
    logging.info(f"Step 2: Querying Neo4j graph ({graph_hops} hop(s))...")
    subgraph_node_ids = []
    if retrieved_context["top_entities"]:
        initial_entity_ids = [e['id'] for e in retrieved_context["top_entities"]]

        # Query 1: Get IDs of nodes within the hop distance
        nodes_query = f"""
        MATCH (start_node:KGNode) WHERE start_node.id IN $entity_ids
        CALL apoc.path.subgraphNodes(start_node, {{
            maxLevel: {graph_hops},
            relationshipFilter: "CONTAINS>|CONTAINS_METHOD>|CALLS>|IMPORTS>"
        }}) YIELD node
        RETURN COLLECT(DISTINCT node.id) AS node_ids
        """
        params = {"entity_ids": initial_entity_ids}
        node_id_results = run_neo4j_query(neo4j_driver, nodes_query, params)

        if node_id_results and node_id_results[0].get("node_ids"):
            subgraph_node_ids = node_id_results[0]["node_ids"]
            logging.info(f"  Identified {len(subgraph_node_ids)} potentially relevant nodes in Neo4j subgraph.")

            # Query 2: Get details for these nodes
            node_details_query = """
            MATCH (n:KGNode) WHERE n.id IN $node_ids
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
            """
            node_details = run_neo4j_query(neo4j_driver, node_details_query, {"node_ids": subgraph_node_ids})
            retrieved_context["related_graph_nodes"] = node_details
            logging.info(f"  Retrieved details for {len(node_details)} nodes.")

            # Query 3: Get relationships *between* these nodes
            rels_query = """
                MATCH (n1:KGNode)
                WHERE n1.id IN $node_ids
                MATCH (n1)-[rel]-()
                WITH n1, count(rel) AS degree
                MATCH (n1)-[r]->(n2:KGNode)
                WHERE n2.id IN $node_ids
                RETURN n1.id AS source_id,
                   type(r) AS rel_type,
                   degree,
                   properties(r) AS rel_properties,
                   n2.id AS target_id
            """

            # Limit relationship types if needed, e.g., add WHERE type(r) IN [...]
            relationships = run_neo4j_query(neo4j_driver, rels_query, {"node_ids": subgraph_node_ids})
            retrieved_context["related_graph_relationships"] = relationships
            logging.info(f"  Found {len(relationships)} relationships within the subgraph.")

        else:
            logging.info("  No subgraph nodes found starting from initial entities.")
    else:
        logging.info("  Skipping Neo4j query as no initial entities were found.")


    # 3. Retrieve Relevant Chunks (MODIFIED: Use Graph Nodes for Querying)
    logging.info(f"Step 3: Querying '{chunks_collection_name}' for top {top_k_chunks} relevant code chunks...")
    query_texts_for_chunks = [query]
    query_texts_for_chunks.extend([e['description'] for e in retrieved_context["top_entities"]])

    # Use descriptions from the *actual nodes found* in the graph
    for node_info in retrieved_context["related_graph_nodes"]: # Changed from related_graph_info
        props = node_info.get('properties', {})
        desc = props.get('description') or props.get('docstring') # Try description or docstring
        name = props.get('name')
        labels = node_info.get('labels', [])
        if desc:
            query_texts_for_chunks.append(desc)
        elif name:
             query_texts_for_chunks.append(f"{' '.join(labels)} named {name}")

    unique_query_texts = list(set(query_texts_for_chunks))
    logging.info(f"  Using {len(unique_query_texts)} unique texts to find relevant chunks.")

    # ... (Keep the rest of the chunk querying logic as is) ...
    if unique_query_texts:
        try:
            chunk_results: QueryResult = chunks_collection.query(
                query_texts=unique_query_texts,
                n_results=top_k_chunks,
                include=["metadatas", "documents"]
            )
            unique_chunks = {}
            if chunk_results and chunk_results.get('ids'):
                 for i in range(len(chunk_results['ids'])):
                     for chunk_id, meta, doc in zip(chunk_results['ids'][i], chunk_results['metadatas'][i], chunk_results['documents'][i]):
                         if chunk_id not in unique_chunks:
                             unique_chunks[chunk_id] = {"id": chunk_id, "metadata": meta, "document": doc}
            retrieved_context["relevant_chunks"] = list(unique_chunks.values())
            logging.info(f"  Found {len(retrieved_context['relevant_chunks'])} unique relevant chunks.")
        except Exception as e:
            retrieved_context["error"] = f"Error querying chunks collection: {e}"
            logging.error(retrieved_context["error"])
    else:
        logging.info("  No query texts derived for retrieving chunks.")


    logging.info("Context retrieval phase complete (including graph structure).")
    return retrieved_context

# --- Helper Function to Format Graph Context ---
def format_graph_context(nodes: List[Dict], relationships: List[Dict], max_rels: int = 15) -> str:
    """
    Formats graph nodes and relationships into a readable string for the LLM,
    including properties on the relationships.
    """
    # Check if there's anything to format
    if not nodes and not relationships:
        return ""

    # Create a lookup for node details (name, type) by ID - same as before
    node_details = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id: # Skip nodes without an ID
            logging.warning(f"Skipping node without ID: {node}")
            continue
        props = node.get("properties", {})
        labels = node.get("labels", ["Node"])
        # Prioritize specific labels if available
        node_type = "Unknown"
        # Define priority order or logic for determining the primary type
        type_priority = ["Method", "Function", "Class", "Module", "Call", "Import"]
        for p_type in type_priority:
             if p_type in labels:
                 node_type = p_type
                 break
        else: # If no priority type found, use the first label or default
             node_type = labels[0] if labels else "Node"

        name = props.get("name", node_id) # Fallback to ID if no name
        node_details[node_id] = {"name": name, "type": node_type}

    # --- Formatting Relationships ---
    formatted_rels = []
    rel_count = 0
    if relationships: # Only proceed if there are relationships
        for i, rel in enumerate(relationships):
            if rel_count >= max_rels:
                formatted_rels.append("... (more relationships shown than limit)")
                break

            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            rel_type = rel.get("rel_type", "RELATED_TO")
            rel_props = rel.get("rel_properties", {}) # <-- Get properties dictionary

            # Ensure source and target IDs exist
            if not source_id or not target_id:
                 logging.warning(f"Skipping relationship with missing source/target ID: {rel}")
                 continue

            source_info = node_details.get(source_id, {"name": source_id, "type": "Unknown/External"})
            target_info = node_details.get(target_id, {"name": target_id, "type": "Unknown/External"})

            # --- Format properties into a string ---
            props_str = ""
            if rel_props: # Check if the properties dictionary is not empty
                # Simple key-value formatting, adjust as needed for readability
                # Example: Filter out very long values or specific keys if necessary
                prop_items = [f"{k}: {str(v)[:50]}{'...' if len(str(v)) > 50 else ''}" # Truncate long values
                              for k, v in rel_props.items()]
                if prop_items: # Only add if there are formatted items
                    props_str = " {" + ", ".join(prop_items) + "}"
            # --- End property formatting ---

            # Append the formatted string, including the property string
            formatted_rels.append(
                f"- {source_info['name']} ({source_info['type']}) "
                f"-[{rel_type}{props_str}]->" # <-- Include the formatted props_str here
                f" {target_info['name']} ({target_info['type']})"
            )
            rel_count += 1

    # --- Construct final output string ---
    output_parts = []
    if formatted_rels:
         output_parts.append("=== Knowledge Graph Structure (Relationships with Properties) ===")
         output_parts.extend(formatted_rels)
         output_parts.append("="*20)
    elif nodes: # Handle case where nodes were found but no relationships between them
         output_parts.append("=== Knowledge Graph Structure (Nodes Found) ===")
         for node_id, details in node_details.items():
             output_parts.append(f"- {details['name']} ({details['type']})")
         if relationships is not None: # Check if relationships was an empty list (vs not queried)
            output_parts.append("(No direct relationships retrieved between these specific nodes)")
         output_parts.append("="*20)
    # else: Both nodes and relationships were empty, handled by the initial check

    return "\n".join(output_parts) + "\n" if output_parts else ""

# --- Step 2: Modify `generate_response` to include formatted graph context ---

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

    # ...(Error handling as before)...
    if retrieved_context.get("error") and not (retrieved_context["top_entities"] or retrieved_context["relevant_chunks"]):
        logging.warning("Cannot generate response due to error during retrieval and no context found.")
        return f"Sorry, I encountered an error retrieving context: {retrieved_context['error']}"


    # 1. Combine Context
    context_pieces = []
    context_pieces.append(f"User Query: {query}\n")

    # --- Add Top Entities (from vector search) ---
    if retrieved_context["top_entities"]:
        context_pieces.append("=== Potentially Relevant Code Entities (from Vector Search) ===")
        # ... (Keep the entity formatting logic as is) ...
        for i, entity in enumerate(retrieved_context["top_entities"]):
            meta = entity.get('metadata', {})
            desc = entity.get('description', 'N/A')
            context_pieces.append(f"Entity {i+1}: ID={entity.get('id')}, Type={meta.get('entity_type', 'N/A')}, Name={meta.get('name', 'N/A')}, File={meta.get('source_file', 'N/A')}:{meta.get('start_line', 'N/A')}\nDescription: {desc[:300]}...") # Limit description length
        context_pieces.append("="*20 + "\n")


    # --- NEW: Add Formatted Graph Context ---
    graph_context_str = format_graph_context(
        retrieved_context.get("related_graph_nodes", []),
        retrieved_context.get("related_graph_relationships", []),
        max_rels=20 # Limit the number of relationships shown to avoid excessive context
    )
    if graph_context_str:
        context_pieces.append(graph_context_str)


    # --- Add Relevant Code Snippets ---
    if retrieved_context["relevant_chunks"]:
        context_pieces.append("=== Relevant Code Snippets (from Vector Search) ===")
        # ... (Keep the chunk formatting logic as is) ...
        max_chunks = 5
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


    # Check if any context was added besides the query
    if not context_pieces[1:]:
         logging.warning("No specific context found for the query.")
         context_pieces.append("No specific code context was found related to the query.")

    combined_context = "\n".join(context_pieces)

    # ...(Context truncation logic as before)...
    max_context_chars = 15000 # Slightly increase if needed for graph info
    if len(combined_context) > max_context_chars:
        logging.warning(f"Combined context length ({len(combined_context)}) exceeds limit ({max_context_chars}). Truncating.")
        combined_context = combined_context[:max_context_chars] + "\n... (Context Truncated)"


    # 2. Formulate Final Prompt - **UPDATED SYSTEM PROMPT**
    system_prompt = (
        "You are an AI assistant specialized in understanding and answering questions about a Python codebase. "
        "Use the provided context, which includes relevant code entities (from vector search), code snippets, **and knowledge graph relationships (showing how code elements are connected, e.g., calls, containment)**, to answer the user's query accurately and concisely. "
        "**Pay close attention to the graph structure section to understand dependencies and architecture.** "
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

    # 3. Call LLM (Keep as is)
    logging.info("Sending prompt to LLM...")
    # Optional: Print the prompt to see the graph context
    print("---------------------------------")
    print(final_prompt)
    print("---------------------------------")
    try:
        # Assuming ollama or similar client with invoke
        response = llm_client.invoke(final_prompt)

        # If using langchain style:
        # from langchain_core.messages import HumanMessage, SystemMessage
        # messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Context:\n{combined_context}\nQuery:\n{query}\nAnswer:")]
        # response = llm_client.invoke(messages) # or response = llm_client.generate([messages])
        # response_content = response.content # Adjust based on your LLM client's return type

        logging.info("Received response from LLM.")
        # Adjust how you extract the text if needed
        if hasattr(response, 'content'):
             return response.content
        elif isinstance(response, str):
             return response
        else:
             # Handle other response types (e.g., complex objects)
             logging.warning(f"LLM returned unexpected type: {type(response)}. Converting to string.")
             return str(response)

    except Exception as e:
        logging.error(f"LLM invocation failed: {e}")
        return "Sorry, I encountered an error generating the response."

# --- Main Execution Example ---
# (Keep the `if __name__ == "__main__":` block as is, it will use the updated functions)
# --- Main Execution Example ---
if __name__ == "__main__":
    if ef is None:
         logging.critical("Embedding Function (ef) is not available. Cannot run RAG pipeline.")

    else:
        # === Configuration for this run ===
        # Make sure this matches an index you created previously
        TARGET_INDEX_ID = "py_structchunk_neo4j_v2_option2"
        TARGET_CHROMA_DB_PATH = f"{CHROMA_PATH_PREFIX}{TARGET_INDEX_ID}"
        while True:
            USER_QUERY = input("What is your question")

            logging.info(f"===== Starting RAG Pipeline for Index: {TARGET_INDEX_ID} =====")

            neo4j_driver_instance = None
            chroma_client_instance = None

            try:
                # --- Initialize Clients ---
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

                # --- Run Retrieval ---
                retrieved_data = retrieve_context(
                    query=USER_QUERY,
                    index_id=TARGET_INDEX_ID,
                    chroma_client=chroma_client_instance,
                    neo4j_driver=neo4j_driver_instance,
                    embedding_function=ef,
                    top_k_entities=7, # How many entities to fetch initially
                    top_k_chunks=3,   # How many chunks to fetch per relevant text
                    graph_hops=3      # How far to look in the graph
                )

                # --- Generate Response ---
                if retrieved_data and not retrieved_data.get("error"):
                    final_answer = generate_response(
                        query=USER_QUERY,
                        retrieved_context=retrieved_data,
                        llm_client=LLM_CLIENT
                    )
                    print("\n===== Final Answer =====")
                    print(final_answer)
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

                # --- Cleanup ---
                if neo4j_driver_instance:
                    logging.info("Closing Neo4j connection.")
                    neo4j_driver_instance.close()
                # ChromaDB persistent client doesn't usually need explicit closing

        logging.info(f"===== RAG Pipeline Complete =====")
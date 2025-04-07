
import os
import json
import logging
from typing import List, Dict, Any, Optional

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

def retrieve_context(
    query: str,
    index_id: str,
    chroma_client: chromadb.Client,
    neo4j_driver: Driver,
    embedding_function: Any, # Should be ChromaDB compatible embedding function
    top_k_entities: int = 5,
    top_k_chunks: int = 5,
    graph_hops: int = 1 # How many hops in Neo4j to explore from initial entities
) -> Dict[str, Any]:
    """
    Retrieves relevant context from ChromaDB and Neo4j based on the query.
    Adapts the diagram's flow since relations aren't directly embedded.
    """
    if embedding_function is None:
        logging.error("Embedding function is not available. Cannot perform vector search.")
        return {"error": "Missing embedding function"}

    entities_collection_name = f"{index_id}_entities"
    chunks_collection_name = f"{index_id}_chunks"
    retrieved_context = {
        "query": query,
        "top_entities": [],
        "related_graph_info": [],
        "relevant_chunks": [],
        "error": None
    }

    try:
        entities_collection = chroma_client.get_collection(name=entities_collection_name, embedding_function=embedding_function)
        chunks_collection = chroma_client.get_collection(name=chunks_collection_name, embedding_function=embedding_function)
    except Exception as e:
        retrieved_context["error"] = f"Error getting ChromaDB collections: {e}"
        logging.error(retrieved_context["error"])
        return retrieved_context

    # 1. Query Entities Vector DB
    logging.info(f"Step 1: Querying '{entities_collection_name}' for top {top_k_entities} entities related to query...")
    try:
        # *** CORRECTED LINE HERE ***
        entity_results: QueryResult = entities_collection.query(
            query_texts=[query],
            n_results=top_k_entities,
            include=["metadatas", "documents"] # Removed 'ids'
        )
        # Process results - Chroma returns lists even for single query
        # IDs are accessed via entity_results['ids'] below without needing 'include'
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
        # Continue if possible, maybe graph search can still work if we have default starting points?

    # 2. Query Knowledge Graph (Neo4j) based on retrieved entities
    # (Neo4j query logic remains the same)
    logging.info(f"Step 2: Querying Neo4j graph starting from retrieved entities ({graph_hops} hop(s))...")
    related_graph_nodes = []
    if retrieved_context["top_entities"]:
        entity_ids = [e['id'] for e in retrieved_context["top_entities"]]
        cypher_query = f"""
        MATCH (start_node:KGNode) WHERE start_node.id IN $entity_ids
        CALL apoc.path.subgraphNodes(start_node, {{
            maxLevel: {graph_hops},
            relationshipFilter: "CONTAINS>|CONTAINS_METHOD>|CALLS>|IMPORTS>"
        }}) YIELD node AS related_node
        RETURN DISTINCT related_node.id AS id,
                        labels(related_node) AS labels,
                        properties(related_node) AS properties
        """
        params = {"entity_ids": entity_ids}
        graph_results = run_neo4j_query(neo4j_driver, cypher_query, params)
        retrieved_context["related_graph_info"] = graph_results # Store raw graph results
        logging.info(f"  Found {len(graph_results)} related nodes/entities in Neo4j.")
    else:
        logging.info("  Skipping Neo4j query as no initial entities were found.")


    # 3. Retrieve Relevant Chunks (Adaptation: Use Entity Info, not Relation Vectors)
    logging.info(f"Step 3: Querying '{chunks_collection_name}' for top {top_k_chunks} relevant code chunks...")
    query_texts_for_chunks = [query]
    query_texts_for_chunks.extend([e['description'] for e in retrieved_context["top_entities"]])
    for node_info in retrieved_context["related_graph_info"]:
        props = node_info.get('properties', {})
        if props.get('description'):
            query_texts_for_chunks.append(props['description'])
        elif props.get('name'):
             query_texts_for_chunks.append(f"{' '.join(node_info.get('labels',[]))} named {props['name']}")

    unique_query_texts = list(set(query_texts_for_chunks))
    logging.info(f"  Using {len(unique_query_texts)} unique texts to find relevant chunks.")

    if unique_query_texts:
        try:
             # *** CORRECTED LINE HERE ***
            chunk_results: QueryResult = chunks_collection.query(
                query_texts=unique_query_texts,
                n_results=top_k_chunks,
                include=["metadatas", "documents"] # Removed 'ids'
            )

            # Process and deduplicate chunk results
            # IDs are accessed via chunk_results['ids'] below without needing 'include'
            unique_chunks = {} # Use dict to deduplicate by ID
            if chunk_results and chunk_results.get('ids'):
                 for i in range(len(chunk_results['ids'])): # Iterate through results for each query text
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


    # (Rest of the function remains the same)
    logging.info("Context retrieval phase complete.")
    return retrieved_context
def generate_response(
    query: str,
    retrieved_context: Dict[str, Any],
    llm_client: Any # Should have an invoke method
) -> str:
    """
    Generates a response using the LLM based on the query and retrieved context.
    """
    logging.info("Generating response...")

    if retrieved_context.get("error") and not (retrieved_context["top_entities"] or retrieved_context["relevant_chunks"]):
        logging.warning("Cannot generate response due to error during retrieval and no context found.")
        return f"Sorry, I encountered an error retrieving context: {retrieved_context['error']}"

    # 1. Combine Context
    context_pieces = []
    context_pieces.append(f"User Query: {query}\n")

    if retrieved_context["top_entities"]:
        context_pieces.append("=== Potentially Relevant Code Entities (from Vector Search) ===")
        for i, entity in enumerate(retrieved_context["top_entities"]):
            meta = entity.get('metadata', {})
            desc = entity.get('description', 'N/A')
            context_pieces.append(f"Entity {i+1}: ID={entity.get('id')}, Type={meta.get('entity_type', 'N/A')}, Name={meta.get('name', 'N/A')}, File={meta.get('source_file', 'N/A')}:{meta.get('start_line', 'N/A')}\nDescription: {desc[:300]}...") # Limit description length
        context_pieces.append("="*20 + "\n")

    if retrieved_context["related_graph_info"]:
        context_pieces.append("=== Related Code Structure (from Knowledge Graph) ===")
        for i, node_info in enumerate(retrieved_context["related_graph_info"]):
            props = node_info.get('properties', {})
            labels = node_info.get('labels', [])
            context_pieces.append(f"Graph Node {i+1}: ID={node_info.get('id')}, Labels={labels}, Name={props.get('name')}, File={props.get('source_file')}:{props.get('start_line')}, Description={props.get('description', 'N/A')[:200]}...")
        context_pieces.append("="*20 + "\n")

    if retrieved_context["relevant_chunks"]:
        context_pieces.append("=== Relevant Code Snippets (from Vector Search) ===")
        # Limit the number of chunks included in the final prompt to avoid exceeding context window
        max_chunks_in_prompt = 5
        for i, chunk in enumerate(retrieved_context["relevant_chunks"][:max_chunks_in_prompt]):
            meta = chunk.get('metadata', {})
            doc = chunk.get('document', 'N/A')
            context_pieces.append(f"Code Snippet {i+1} (Source: {meta.get('source_file', 'N/A')}, Lines {meta.get('start_line', 'N/A')}-{meta.get('end_line', 'N/A')}):\n```python\n{doc}\n```")
        if len(retrieved_context["relevant_chunks"]) > max_chunks_in_prompt:
            context_pieces.append(f"... (plus {len(retrieved_context['relevant_chunks']) - max_chunks_in_prompt} more relevant snippets not shown)")
        context_pieces.append("="*20 + "\n")

    if not context_pieces[1:]: # Check if any context besides the query was added
         logging.warning("No specific context found for the query.")
         context_pieces.append("No specific code context was found related to the query.")
         # Maybe add a fallback instruction?

    combined_context = "\n".join(context_pieces)

    # Limit overall context size reasonably (e.g., ~3000 tokens for ~4k limit models)
    # This is a rough estimate, use a proper tokenizer for accuracy if needed
    max_context_chars = 12000
    if len(combined_context) > max_context_chars:
        logging.warning(f"Combined context length ({len(combined_context)}) exceeds limit ({max_context_chars}). Truncating.")
        combined_context = combined_context[:max_context_chars] + "\n... (Context Truncated)"


    # 2. Formulate Final Prompt
    # Base system prompt can be reused from your LLM Client if appropriate
    # Or define a specific one for Q&A
    system_prompt = (
        "You are an AI assistant specialized in understanding and answering questions about a Python codebase. "
        "Use the provided context, which includes relevant code entities, graph relationships, and code snippets, to answer the user's query accurately and concisely. "
        "If the context does not contain the answer, state that the information is not available in the provided code details."
    )

    final_prompt = f"""{system_prompt}

## Provided Context:
{combined_context}

## User Query:
{query}

## Answer:
""" # Let the LLM complete the answer

    # 3. Call LLM
    logging.info("Sending prompt to LLM...")
    print("---------------------------------")
    print(final_prompt)
    print("---------------------------------")
    try:
        response = llm_client.invoke(final_prompt) # Assuming invoke takes the full prompt
        # If your llm_client needs separate system/user prompts, adjust here:
        # response = llm_client.generate(system=system_prompt, prompt=f"Context:\n{combined_context}\nQuery:\n{query}\nAnswer:")
        logging.info("Received response from LLM.")
        return response
    except Exception as e:
        logging.error(f"LLM invocation failed: {e}")
        return "Sorry, I encountered an error generating the response."


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
                    top_k_entities=3, # How many entities to fetch initially
                    top_k_chunks=5,   # How many chunks to fetch per relevant text
                    graph_hops=1      # How far to look in the graph
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
            finally:
                # --- Cleanup ---
                if neo4j_driver_instance:
                    logging.info("Closing Neo4j connection.")
                    neo4j_driver_instance.close()
                # ChromaDB persistent client doesn't usually need explicit closing

        logging.info(f"===== RAG Pipeline Complete =====")

import os
import logging
from datetime import datetime
import chromadb
from typing import List
from neo4j import GraphDatabase
from chromadb.utils import embedding_functions
from code_parser import CodeParser
from llm_client import get_llm_client, get_fallback_llm
from pipeline import parse_and_extract, deduplicate_entities, update_entity_descriptions, chunk_files
from database import embed_and_store_chunks, embed_and_store_entities, populate_neo4j_graph, store_metadata
from config import EMBEDDING_MODEL_NAME, CHUNK_TOKEN_LIMIT, CHROMA_PATH_PREFIX, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def main_processing_pipeline(input_file_paths: List[str], index_id: str = "python_struct_chunk_neo4j_v1", enable_llm_description: bool = False):
    logging.info(f"===== Starting PYTHON Structure-Chunk Pipeline (Neo4j/Chroma Target): {index_id} =====")
    start_time = datetime.now()
    kv_store = {}

    python_files = [p for p in input_file_paths if os.path.isfile(p) and p.lower().endswith(".py")]
    if not python_files:
        logging.error("No valid .py files found in the input paths.")
        return
    logging.info(f"Found {len(python_files)} Python files to process.")

    code_parser = CodeParser()
    if not code_parser.is_parser_loaded():
        logging.critical("Python parser failed to load. Exiting.")
        return

    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        logging.info(f"Successfully initialized embedding function with {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logging.critical(f"Failed to initialize SentenceTransformerEmbeddingFunction: {e}")
        return

    chroma_db_path = f"{CHROMA_PATH_PREFIX}{index_id}"
    logging.info(f"Initializing ChromaDB client at path: {chroma_db_path}")
    try:
        os.makedirs(chroma_db_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        chroma_client.heartbeat()
        logging.info("ChromaDB client initialized successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialize ChromaDB client at {chroma_db_path}: {e}", exc_info=True)
        return

    neo4j_driver = None
    try:
        logging.info(f"Connecting to Neo4j at {NEO4J_URI}...")
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        logging.info("Neo4j connection successful.")
        with neo4j_driver.session(database="neo4j") as session:
            logging.info("Ensuring Neo4j :KGNode(id) uniqueness constraint exists...")
            session.run("CREATE CONSTRAINT unique_kgnode_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE")
            logging.info("Neo4j uniqueness constraint checked/created.")
    except Exception as e:
        logging.critical(f"Failed to connect to Neo4j or verify constraint: {e}", exc_info=True)
        if neo4j_driver: neo4j_driver.close()
        return

    try:
        all_extracted_entities = parse_and_extract(python_files, code_parser)
        unique_entities = deduplicate_entities(all_extracted_entities)
        if enable_llm_description:
            unique_entities = update_entity_descriptions(unique_entities, get_llm_client())
        else:
            logging.info("--- Skipping LLM description update (disabled by flag) ---")
            for item in unique_entities:
                if 'description' not in item:
                    item['description'] = f"Default description for {item.get('entity_type')} '{item.get('name')}'."
        text_chunks = chunk_files(python_files, code_parser, CHUNK_TOKEN_LIMIT)
        embed_and_store_chunks(text_chunks, ef, chroma_client, f"{index_id}_chunks")
        embed_and_store_entities(unique_entities, ef, chroma_client, f"{index_id}_entities")
        populate_neo4j_graph(neo4j_driver, unique_entities, all_extracted_entities, python_files)
        store_metadata(kv_store, index_id, python_files, len(unique_entities), len(text_chunks))
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}", exc_info=True)
    finally:
        if neo4j_driver:
            logging.info("Closing Neo4j connection.")
            neo4j_driver.close()

    end_time = datetime.now()
    logging.info(f"===== Pipeline Complete: {index_id} =====")
    logging.info(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    if not os.path.exists("temp_code"): os.makedirs("temp_code")
    with open("temp_code/processor.py", "w", encoding='utf-8') as f:
        f.write("""
def printfun():
    print("Hello from printfun!")

def loop():
    #This function loops 3 times and calls printfun.
    for i in range(3):
        printfun()

def run_loops(num_times):
    #Runs the loop function multiple times.
    print(f"Running the loop {num_times} times.")
    for _ in range(num_times):
        loop()
    return f"Completed {num_times} loop runs."
""")

    input_files = ["temp_code/processor.py"]
    index_name = "py_structchunk_neo4j_v2_option2"
    USE_LLM_DESCRIPTIONS = True
    if NEO4J_PASSWORD == "abcd12345":
        logging.warning("Using default Neo4j password. Set the NEO4J_PASSWORD environment variable for security.")
    main_processing_pipeline(input_files, index_id=index_name, enable_llm_description=USE_LLM_DESCRIPTIONS)
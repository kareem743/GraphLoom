import os
import logging
from typing import List, Dict, Tuple
from code_parser import CodeParser
from llm_client import get_llm_client, get_fallback_llm
from chromadb.utils import embedding_functions
from chunking import chunk_code_structure_aware
from database import embed_and_store_chunks, embed_and_store_entities, populate_neo4j_graph, store_metadata
from config import generate_unique_id, count_tokens, CHUNK_TOKEN_LIMIT, CHUNK_ENCODING

def parse_and_extract(python_files: List[str], code_parser: CodeParser) -> List[Dict]:
    logging.info(f"--- Parsing {len(python_files)} Python Files & Extracting Structural Entities ---")
    all_entities = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: code = f.read()
            logging.info(f"  Parsing for entities: {file_path}")
            file_entities = code_parser.extract_entities_from_file(code, file_path)
            all_entities.extend(file_entities)
            logging.info(f"    Extracted {len(file_entities)} entities from {file_path}")
        except Exception as e:
            logging.error(f"  Error parsing/extracting entities from file {file_path}: {e}", exc_info=True)
    logging.info(f"--- Total structural entities extracted: {len(all_entities)} ---")
    return all_entities

def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    logging.info(f"--- Deduplicating {len(entities)} Structural Entities ---")
    key_fields = ('entity_type', 'name', 'source_file', 'start_byte')
    deduped_map: Dict[Tuple, Dict] = {}
    duplicates_found = 0
    for entity in entities:
        try:
            if all(k in entity for k in key_fields):
                entity_key = tuple(str(entity.get(k)) for k in key_fields)
                if entity_key not in deduped_map:
                    entity['id'] = generate_unique_id(f"{entity.get('entity_type', 'ent')}_")
                    deduped_map[entity_key] = entity
                else:
                    duplicates_found += 1
            else:
                logging.warning(f"Skipping deduplication for entity missing key fields: {entity.get('name', 'N/A')}")
        except Exception as e:
            logging.warning(f"Error during deduplication key creation for {entity.get('name', 'N/A')}: {e}")
    deduped_list = list(deduped_map.values())
    logging.info(f"--- Deduplication complete: {len(deduped_list)} unique entities identified ({duplicates_found} duplicates ignored) ---")
    return deduped_list

def update_entity_descriptions(entities: List[Dict], llm_client) -> List[Dict]:
    logging.info(f"--- Updating Entity Descriptions using LLM ---")
    updated_items = []
    is_placeholder = isinstance(llm_client, type(get_fallback_llm()))
    if is_placeholder:
        logging.info("  Skipping LLM description update (Using placeholder or Ollama unavailable).")
    total_entities = len(entities)
    logging.info(f"  Processing descriptions for {total_entities} unique entities...")
    for i, item in enumerate(entities):
        if (i + 1) % 50 == 0 or i == total_entities - 1:
            logging.info(f"  Processed descriptions for {i+1}/{total_entities} entities...")
        if not is_placeholder and 'description' not in item:
            prompt = f"""
            Analyze the following Python code entity and provide a concise (1-2 sentence) description of its purpose or role.
            Focus on what it *does*. Output ONLY the description text.

            Entity Type: {item.get('entity_type', 'N/A')}
            Entity Name: {item.get('name', 'N/A')}
            Source File: {item.get('source_file', 'N/A')} (line {item.get('start_line', 'N/A')})
            Parent Context: {item.get('parent_type', 'Global scope')} {item.get('parent_name', '')}
            Code Snippet:
            ```python
            {item.get('snippet', 'N/A')}
            ```

            Description:"""
            try:
                description = llm_client.invoke(prompt).strip()
                if description and "Error generating response" not in description and "placeholder" not in description.lower():
                    item['description'] = description
                    logging.debug(f"    Updated description for {item.get('id')}: {description[:100]}...")
                else:
                    logging.warning(f"    LLM description update failed or returned empty for {item.get('id')}. Adding default.")
                    item['description'] = f"Default description for {item.get('entity_type')} '{item.get('name')}' (LLM update failed)."
            except Exception as e:
                logging.error(f"  LLM Error updating description for {item.get('id', 'N/A')}: {e}")
                item['description'] = f"Default description for {item.get('entity_type')} '{item.get('name')}' (LLM error)."
        elif 'description' not in item:
            item['description'] = f"Default description for {item.get('entity_type')} '{item.get('name')}'."
        updated_items.append(item)
    logging.info(f"--- Description update pass complete ({len(updated_items)} processed) ---")
    return updated_items

def chunk_files(python_files: List[str], code_parser: CodeParser, token_limit: int) -> List[Dict]:
    logging.info(f"--- Chunking Python File Content (Structure-Aware, Token Limit: {token_limit}) ---")
    all_chunks = []
    if not code_parser.is_parser_loaded():
        logging.error("Code parser not loaded. Cannot perform structure-aware chunking.")
        return all_chunks
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: code = f.read()
            if not code.strip():
                logging.info(f"  Skipping empty file: {file_path}")
                continue
            file_chunks = chunk_code_structure_aware(
                code=code, token_limit=token_limit, source_file=file_path,
                code_parser=code_parser, encoding_name=CHUNK_ENCODING
            )
            all_chunks.extend(file_chunks)
            logging.info(f"  Chunked '{os.path.basename(file_path)}' into {len(file_chunks)} structure-aware chunks.")
        except Exception as e:
            logging.error(f"  Error chunking file {file_path}: {e}", exc_info=True)
    logging.info(f"--- Total structure-aware text chunks created: {len(all_chunks)} ---")
    return all_chunks
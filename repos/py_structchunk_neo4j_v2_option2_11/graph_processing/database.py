import os
import json
import logging
from typing import List, Dict
from neo4j import GraphDatabase, Driver
from datetime import datetime
from config import PYTHON_BUILTINS, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, generate_unique_id
from  config import EMBEDDING_MODEL_NAME
def embed_and_store_chunks(chunks: List[Dict], embedding_function, vector_db_client, collection_name: str):
    logging.info(f"--- Embedding & Storing {len(chunks)} Text Chunks in ChromaDB ---")
    if not chunks:
        logging.warning("  No chunks provided to embed.")
        return
    if embedding_function is None:
        logging.error("  Embedding function is not available. Skipping chunk embedding.")
        return

    texts = [c['text'] for c in chunks]
    ids = [c['id'] for c in chunks]
    metadatas = [c['metadata'] for c in chunks]

    try:
        collection = vector_db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        valid_metadatas = []
        for meta in metadatas:
            valid_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    valid_meta[k] = v
                else:
                    valid_meta[k] = str(v)
            valid_metadatas.append(valid_meta)

        batch_size = 100
        num_batches = (len(ids) + batch_size - 1) // batch_size
        logging.info(f"  Adding {len(ids)} chunks to collection '{collection_name}' in {num_batches} batches...")

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:min(i + batch_size, len(ids))]
            batch_texts = texts[i:min(i + batch_size, len(ids))]
            batch_metadatas = valid_metadatas[i:min(i + batch_size, len(ids))]
            if not batch_ids: continue
            logging.debug(f"    Adding chunk batch {i//batch_size + 1}/{num_batches} (size {len(batch_ids)})")
            collection.add(ids=batch_ids, documents=batch_texts, metadatas=batch_metadatas)
        logging.info(f"--- Stored {len(ids)} chunk texts and metadata in ChromaDB collection '{collection_name}' ---")
    except Exception as e:
        logging.error(f"  Error embedding or storing chunks in ChromaDB: {e}", exc_info=True)

def embed_and_store_entities(entities: List[Dict], embedding_function, vector_db_client, collection_name: str):
    logging.info(f"--- Embedding & Storing {len(entities)} Unique Entities in ChromaDB ---")
    if not entities:
        logging.warning("  No entities provided to embed.")
        return
    if embedding_function is None:
        logging.error("  Embedding function is not available. Skipping entity embedding.")
        return

    ids = []; texts_to_embed = []; metadatas = []
    for e in entities:
        if 'id' not in e:
            logging.warning(f"Skipping entity missing 'id': {e.get('name', 'N/A')}")
            continue
        ids.append(e['id'])
        text = f"Entity Type: {e.get('entity_type', 'Unknown')}\n"
        text += f"Name: {e.get('name', 'Unnamed')}\n"
        text += f"Source File: {e.get('source_file', 'N/A')}:{e.get('start_line', 'N/A')}\n"
        if e.get('parent_type') and e.get('parent_name'):
            text += f"Context: Inside {e.get('parent_type')} {e.get('parent_name')}\n"
        text += f"Description: {e.get('description', 'Not Available')}\n"
        texts_to_embed.append(text)
        meta = {k: v for k,v in e.items() if k not in ['snippet', 'id']}
        valid_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                valid_meta[k] = v
            else:
                valid_meta[k] = str(v)
        metadatas.append(valid_meta)

    if not ids:
        logging.warning("  No valid entities with IDs found to embed.")
        return

    try:
        collection = vector_db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        batch_size = 100
        num_batches = (len(ids) + batch_size - 1) // batch_size
        logging.info(f"  Adding {len(ids)} entities to collection '{collection_name}' in {num_batches} batches...")
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:min(i + batch_size, len(ids))]
            batch_texts = texts_to_embed[i:min(i + batch_size, len(ids))]
            batch_metadatas = metadatas[i:min(i + batch_size, len(ids))]
            if not batch_ids: continue
            logging.debug(f"    Adding entity batch {i//batch_size + 1}/{num_batches} (size {len(batch_ids)})")
            collection.add(ids=batch_ids, documents=batch_texts, metadatas=batch_metadatas)
        logging.info(f"--- Stored {len(ids)} entity texts and metadata in ChromaDB collection '{collection_name}' ---")
    except Exception as e:
        logging.error(f"  Error embedding or storing entities in ChromaDB: {e}", exc_info=True)

def _run_write_query(tx, query, **params):
    try:
        result = tx.run(query, **params)
        return True
    except Exception as e:
        logging.error(f"Neo4j query failed!")
        logging.error(f"  Error Type: {type(e).__name__}")
        logging.error(f"  Error Details: {e}")
        logging.error(f"  Query: {query}")
        try:
            logging.error(f"  Params: {json.dumps(params, default=str, indent=2)}")
        except TypeError:
            logging.error(f"  Params: (Could not serialize params)")
        raise

def add_neo4j_node(tx, node_data: Dict):
    node_id = node_data.get('id')
    node_type = node_data.get('entity_type', 'UnknownEntity')
    if not node_id:
        logging.warning(f"Skipping Neo4j node write, missing 'id' in data: {node_data.get('name', 'N/A')}")
        return

    safe_node_type = "".join(c if c.isalnum() or c=='_' else '' for c in str(node_type))
    if not safe_node_type or not safe_node_type[0].isupper():
        safe_node_type = f"Type_{safe_node_type}" if safe_node_type else "UnknownEntity"

    props_to_set = {}
    for k, v in node_data.items():
        if k not in ['id', 'snippet'] and v is not None:
            if isinstance(v, (list, dict)):
                try: props_to_set[k] = json.dumps(v)
                except TypeError: props_to_set[k] = str(v)
            elif isinstance(v, (str, int, float, bool)):
                props_to_set[k] = v
            else:
                props_to_set[k] = str(v)
    props_to_set['entity_type'] = node_type

    query = """
    MERGE (n:KGNode {id: $id})
    SET n = $props
    SET n.id = $id
    WITH n
    CALL apoc.create.addLabels(n, [$node_type_label]) YIELD node
    RETURN count(node)
    """
    params = {'id': node_id, 'props': props_to_set, 'node_type_label': safe_node_type}
    try:
        _run_write_query(tx, query, **params)
        logging.debug(f"Successfully added/merged Neo4j node: {node_id} (Type: {safe_node_type})")
    except Exception as e:
        logging.error(f"Failed to write Neo4j node for ID: {node_id}")

def add_neo4j_potential_function_node(tx, name: str, is_builtin: bool = False):
    if not name or not isinstance(name, str): return None
    node_name = name
    safe_name_for_id = "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in name)
    safe_name_for_id = '.'.join(filter(None, safe_name_for_id.split('.')))
    if not safe_name_for_id:
        safe_name_for_id = generate_unique_id("invalid_name_")
        logging.warning(f"Sanitized name for '{name}' became empty. Using fallback ID part.")

    if is_builtin:
        entity_type = 'BuiltinFunction'
        node_id = f"BuiltinFunction:{safe_name_for_id}"
        props = {'id': node_id, 'name': node_name, 'entity_type': entity_type, 'origin': 'builtin'}
        node_type_label = 'BuiltinFunction'
    else:
        entity_type = 'PotentialFunction'
        node_id = f"PotentialFunction:{safe_name_for_id}"
        props = {'id': node_id, 'name': node_name, 'entity_type': entity_type}
        node_type_label = 'PotentialFunction'

    query = """
    MERGE (n:KGNode {id: $id})
    ON CREATE SET n = $props, n.created_at = timestamp()
    ON MATCH SET n += $props, n.last_seen = timestamp()
    WITH n
    CALL apoc.create.addLabels(n, [$node_type_label]) YIELD node
    RETURN count(node)
    """
    params = {'id': node_id, 'props': props, 'node_type_label': node_type_label}
    try:
        _run_write_query(tx, query, **params)
        logging.debug(f"Added/updated placeholder node: {node_id} (Type: {node_type_label})")
        return node_id
    except Exception as e:
        logging.error(f"Failed to write Neo4j placeholder node for: {name} (ID: {node_id})")
        return None

def add_neo4j_edge(tx, source_id: str, target_id: str, rel_type: str, rel_props: Dict = None):
    if not source_id or not target_id or not rel_type:
        logging.warning(f"Skipping Neo4j edge: Missing source/target ID or relationship type. Src: {source_id}, Tgt: {target_id}, Type: {rel_type}")
        return
    if source_id == target_id:
        logging.debug(f"Skipping self-referential edge: {source_id} -[{rel_type}]-> {target_id}")
        return

    rel_props = rel_props or {}
    safe_rel_type = "".join(c if c.isalnum() or c=='_' else '' for c in str(rel_type)).upper()
    if not safe_rel_type: safe_rel_type = "RELATED_TO"

    props_to_set = {}
    for k, v in rel_props.items():
        if v is not None:
            if isinstance(v, (list, dict)):
                try: props_to_set[k] = json.dumps(v)
                except TypeError: props_to_set[k] = str(v)
            elif isinstance(v, (str, int, float, bool)):
                props_to_set[k] = v
            else:
                props_to_set[k] = str(v)

    query = """
    MATCH (a:KGNode {id: $source_id})
    MATCH (b:KGNode {id: $target_id})
    MERGE (a)-[r:""" + safe_rel_type + """]->(b)
    SET r = $props
    """
    params = {'source_id': source_id, 'target_id': target_id, 'props': props_to_set}
    try:
        _run_write_query(tx, query, **params)
        logging.debug(f"Successfully added/merged Neo4j edge: ({source_id})-[{safe_rel_type}]->({target_id})")
    except Exception as e:
        logging.error(f"Failed to write Neo4j edge: ({source_id})-[{safe_rel_type}]->({target_id})")

def populate_neo4j_graph(driver: Driver, unique_entities: List[Dict], all_extracted_entities: List[Dict], file_paths: List[str]):
    logging.info("--- 7. Populating Neo4j Knowledge Graph ---")
    # Build maps for quick lookup
    entity_map_by_id = {e['id']: e for e in unique_entities if 'id' in e}
    entity_lookup_key_to_id = {
        (e.get('entity_type'), e.get('name'), e.get('source_file'), e.get('start_byte')): e['id']
        for e in unique_entities if 'id' in e and all(k in e for k in ['entity_type', 'name', 'source_file', 'start_byte'])
    }

    nodes_to_write_batch = []
    edges_to_write_batch = []
    # Store tuples: (name, is_builtin) to avoid duplicate node creation attempts in one batch
    potential_funcs_to_create = set()

    # --- Prepare File Nodes ---
    logging.info("  Preparing File nodes...")
    file_node_ids = {}
    for fp in file_paths:
        try:
            # Attempt to get relative path, fall back to basename if error
            rel_path = os.path.relpath(fp)
        except ValueError: # Handle cases like different drives on Windows
            rel_path = os.path.basename(fp)
            logging.warning(f"Could not determine relative path for {fp}. Using basename: {rel_path}")

        safe_path_id = "".join(c if c.isalnum() or c in ['_', '-', '.', '/'] else '_' for c in rel_path)
        node_id = f"File:{safe_path_id}"
        file_node_ids[fp] = node_id
        nodes_to_write_batch.append({'id': node_id, 'entity_type': 'File', 'path': rel_path, 'absolute_path': fp})

    # --- Prepare Entity Nodes ---
    logging.info("  Preparing Entity nodes...")
    for entity_id, entity_data in entity_map_by_id.items():
         nodes_to_write_batch.append(entity_data) # Add unique entities to batch

    # --- Prepare Edges & Potential Functions ---
    logging.info("  Preparing Edges and identifying Potential/Builtin Functions...")
    for entity in all_extracted_entities: # Iterate through ALL entities found initially
        entity_key = (entity.get('entity_type'), entity.get('name'), entity.get('source_file'), entity.get('start_byte'))
        entity_id = entity_lookup_key_to_id.get(entity_key)

        if not entity_id:
            # This can happen if the entity wasn't deemed unique or lacked key fields
            # logging.debug(f"Skipping edge creation for non-unique/missing entity: {entity_key}")
            continue # Skip if this entity wasn't considered unique (or failed ID creation)

        source_file = entity.get('source_file')
        file_node_id = file_node_ids.get(source_file)

        # 1. File CONTAINS Entity Edge
        if file_node_id:
            edges_to_write_batch.append((file_node_id, entity_id, "CONTAINS", {'start': entity.get('start_line'), 'end': entity.get('end_line')}))

        # 2. Class CONTAINS Method Edge (Function within a Class context)
        parent_type = entity.get('parent_type'); parent_name = entity.get('parent_name')
        if entity.get('entity_type') == 'Function' and parent_type == 'Class' and parent_name and source_file:
             # Find the ID of the parent class defined *in the same file*
             potential_parent_keys = [k for k in entity_lookup_key_to_id if k[0]=='Class' and k[1]==parent_name and k[2]==source_file]
             if len(potential_parent_keys) == 1:
                 parent_class_id = entity_lookup_key_to_id[potential_parent_keys[0]]
                 edges_to_write_batch.append((parent_class_id, entity_id, "CONTAINS_METHOD", {}))
             elif len(potential_parent_keys) > 1:
                 logging.warning(f"Ambiguous parent class '{parent_name}' for method '{entity.get('name')}' in {source_file}. Skipping CONTAINS_METHOD edge.")

        # 3. CALLS Edge (Function/Method calls another Function/Method/Potential/Builtin)
        if entity.get('entity_type') == 'Call':
            caller_id = None
            caller_parent_type = entity.get('parent_type')
            caller_parent_name = entity.get('parent_name')
            # --- *** Use the (potentially qualified) name extracted by the parser *** ---
            callee_name = entity.get('name') # This now holds the qualified name if applicable
            # --- *** ---

            # Find the ID of the *calling* function/method/file
            if caller_parent_name and caller_parent_type in ['Function', 'Class'] and source_file: # Call is inside a func/method
                 potential_caller_keys = [k for k in entity_lookup_key_to_id if k[0]==caller_parent_type and k[1]==caller_parent_name and k[2]==source_file]
                 if len(potential_caller_keys) == 1: caller_id = entity_lookup_key_to_id[potential_caller_keys[0]]
                 elif len(potential_caller_keys) > 1 :
                      logging.warning(f"Ambiguous caller context '{caller_parent_type} {caller_parent_name}' for call to '{callee_name}' in {source_file}. Using Call entity ID '{entity_id}' as caller proxy.")
                      caller_id = entity_id
                 else:
                      logging.warning(f"Could not find unique caller context '{caller_parent_type} {caller_parent_name}' for call to '{callee_name}' in {source_file}. Using Call entity ID '{entity_id}'.")
                      caller_id = entity_id
            elif file_node_id: # Call is at global scope within the file
                 caller_id = file_node_id # Relate the File node to the callee
            else: # Should not happen if file node exists
                logging.warning(f"Cannot determine caller context for call to '{callee_name}' at line {entity.get('start_line')} in {source_file}.")

            if caller_id and callee_name:
                callee_id = None
                # Try to find the callee defined within the analyzed files (using its simple name)
                # Assumption: local definitions use simple names, not qualified ones.
                simple_callee_name = callee_name.split('.')[-1]
                potential_callee_keys = [
                    k for k in entity_lookup_key_to_id
                    if k[0] in ['Function', 'Class'] # Can call classes (constructors)
                    and k[1] == simple_callee_name
                    and k[2] == source_file # Restrict lookup to the same file for simplicity
                ]

                if len(potential_callee_keys) == 1:
                    callee_id = entity_lookup_key_to_id[potential_callee_keys[0]]
                    logging.debug(f"Resolved call to locally defined '{callee_name}' ({simple_callee_name}) -> {callee_id}")
                elif len(potential_callee_keys) > 1:
                     logging.warning(f"Ambiguous local definition for call target '{simple_callee_name}' found in {source_file}. Treating as external.")
                     # Fall through to potential/builtin logic

                # --- *** NEW: Handle Builtins and Unresolved *** ---
                if not callee_id: # If not found locally or ambiguous
                    is_builtin = callee_name in PYTHON_BUILTINS
                    # Add to set for batched creation later
                    potential_funcs_to_create.add((callee_name, is_builtin))

                    # Determine the ID that *will* be created for the edge
                    safe_name_for_id = "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in callee_name)
                    safe_name_for_id = '.'.join(filter(None, safe_name_for_id.split('.')))
                    if not safe_name_for_id: safe_name_for_id = generate_unique_id("invalid_name_") # Use generate_unique_id instead of uuid.uuid4()

                    if is_builtin:
                        callee_id = f"BuiltinFunction:{safe_name_for_id}"
                        logging.debug(f"Call to unresolved '{callee_name}' identified as Builtin.")
                    else:
                        callee_id = f"PotentialFunction:{safe_name_for_id}"
                        logging.debug(f"Call to unresolved '{callee_name}'. Will create PotentialFunction node.")
                # --- *** END NEW *** ---

                if callee_id: # Ensure we have a target ID (local, potential, or builtin)
                     rel_props = {'line': entity.get('start_line')} # Add line number of the call
                     edges_to_write_batch.append((caller_id, callee_id, "CALLS", rel_props))

        # 4. IMPORTS Edge (File imports Module/Items)
        if entity.get('entity_type') in ['Import', 'ImportFrom']:
             if file_node_id and entity_id: # Ensure both file and import node IDs exist
                 # Properties for the import relationship
                 rel_props = {'line': entity.get('start_line')}
                 if entity.get('entity_type') == 'Import':
                     rel_props['module'] = entity.get('module_name', entity.get('name')) # Name is the module here
                 elif entity.get('entity_type') == 'ImportFrom':
                     rel_props['module'] = entity.get('module_name', 'UnknownModule')
                     # Safely convert list to string
                     imported_items_str = str(entity.get('imported_items', []))
                     # Truncate if excessively long? Neo4j has limits.
                     rel_props['items'] = imported_items_str[:1000] # Limit length

                 edges_to_write_batch.append((file_node_id, entity_id, "IMPORTS", rel_props))

    # --- Execute Batched Writes in Transactions ---
    potential_funcs_list = list(potential_funcs_to_create) # Convert set to list
    logging.info(f"Executing Neo4j writes: {len(nodes_to_write_batch)} primary nodes, {len(potential_funcs_list)} potential/builtin funcs, {len(edges_to_write_batch)} edges.")
    batch_size = 500 # Define batch size for transactions

    try:
        with driver.session(database="neo4j") as session: # Specify DB if not default
            # Write Primary Nodes (Files, Functions, Classes, etc.)
            logging.info(f"  Writing {len(nodes_to_write_batch)} primary nodes...")
            for i in range(0, len(nodes_to_write_batch), batch_size):
                batch = nodes_to_write_batch[i:min(i + batch_size, len(nodes_to_write_batch))]
                try:
                    session.execute_write(lambda tx: [add_neo4j_node(tx, node_data) for node_data in batch])
                    logging.debug(f"    Wrote node batch {i//batch_size + 1}...")
                except Exception as batch_e:
                     # Error already logged in helper, just note batch failure
                     logging.error(f"    Error writing primary node batch starting at index {i}.")
                     # Optionally break or continue depending on desired error handling

            # --- *** NEW: Write Potential/Builtin Function Nodes *** ---
            logging.info(f"  Writing {len(potential_funcs_list)} Potential/Builtin function nodes...")
            for i in range(0, len(potential_funcs_list), batch_size):
                 batch = potential_funcs_list[i:min(i + batch_size, len(potential_funcs_list))]
                 try:
                     # Pass is_builtin flag to the creation function
                     session.execute_write(lambda tx: [add_neo4j_potential_function_node(tx, func_name, is_builtin) for func_name, is_builtin in batch])
                     logging.debug(f"    Wrote potential/builtin function batch {i//batch_size + 1}...")
                 except Exception as batch_e:
                     logging.error(f"    Error writing potential/builtin function batch starting at index {i}.")
            # --- *** END NEW *** ---

            # Write Edges
            logging.info(f"  Writing {len(edges_to_write_batch)} edges...")
            for i in range(0, len(edges_to_write_batch), batch_size):
                 batch = edges_to_write_batch[i:min(i + batch_size, len(edges_to_write_batch))]
                 try:
                     session.execute_write(lambda tx: [add_neo4j_edge(tx, src, tgt, rel, props) for src, tgt, rel, props in batch])
                     logging.debug(f"    Wrote edge batch {i//batch_size + 1}...")
                 except Exception as batch_e:
                     logging.error(f"    Error writing edge batch starting at index {i}.")

        logging.info("--- Neo4j graph population complete ---")
    except Exception as e:
        logging.error(f"Failed during Neo4j population transaction: {e}", exc_info=True)
        logging.error("Check Neo4j connection, permissions, and APOC installation if used.")

def store_metadata(kv_storage: Dict, index_name: str, file_paths: List[str], num_entities: int, num_chunks: int):
    logging.info("--- Storing Index Metadata ---")
    metadata = {
        "index_id": index_name,
        "timestamp": datetime.now().isoformat(),
        "processed_files": file_paths,
        "entity_vector_collection": f"{index_name}_entities",
        "chunk_vector_collection": f"{index_name}_chunks",
        "chunking_strategy": "structure-aware",
        "token_limit_per_chunk": 1550,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "neo4j_populated": True,
        "neo4j_uri": NEO4J_URI,
        "num_unique_entities_processed": num_entities,
        "num_text_chunks_processed": num_chunks,
        "language_processed": "python"
    }
    kv_storage[index_name] = metadata
    metadata_filename = f"{index_name}_metadata.json"
    try:
        with open(metadata_filename, 'w') as f:
            json.dump(kv_storage, f, indent=2, default=str)
        logging.info(f"Metadata saved to {metadata_filename}")
    except Exception as e:
        logging.error(f"Error saving metadata to {metadata_filename}: {e}")
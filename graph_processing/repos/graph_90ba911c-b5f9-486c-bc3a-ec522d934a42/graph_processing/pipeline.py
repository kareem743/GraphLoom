import os
import logging
from typing import List, Dict, Tuple
from code_parser import CodeParser
from llm_client import get_llm_client, get_fallback_llm
from chromadb.utils import embedding_functions
from chunking import chunk_code_structure_aware
from database import embed_and_store_chunks, embed_and_store_entities, populate_neo4j_graph, store_metadata
from config import generate_unique_id, count_tokens, CHUNK_TOKEN_LIMIT, CHUNK_ENCODING,EMBEDDING_MODEL_NAME
# In pipeline.py (or a new utils.py)
import os
import logging

def get_python_module_path(file_path: str, project_root: str) -> str | None:
    """
    Calculates the Python module path for a given file path relative to a project root.
    Example: /path/to/repo/src/my_pkg/utils.py with root /path/to/repo/src/ -> my_pkg.utils
    Example: /path/to/repo/my_pkg/__init__.py with root /path/to/repo/ -> my_pkg
    """
    try:
        # Ensure project_root ends with a separator for correct relative path calculation
        norm_root = os.path.normpath(project_root)
        if not norm_root.endswith(os.sep):
            norm_root += os.sep

        norm_file_path = os.path.normpath(file_path)

        if not norm_file_path.startswith(norm_root):
            logging.warning(f"File '{norm_file_path}' is not under project root '{norm_root}'. Cannot determine module path.")
            return None

        relative_path = os.path.relpath(norm_file_path, norm_root)
        module_part, _ = os.path.splitext(relative_path)

        # Handle __init__.py files - they represent the package directory itself
        if os.path.basename(module_part) == '__init__':
            module_part = os.path.dirname(module_part)

        # Replace OS-specific separators with Python's dot notation
        module_path = module_part.replace(os.sep, '.')

        # Handle edge case where file is directly in root (e.g. setup.py) -> '' is not valid
        if not module_path and os.path.basename(norm_file_path) == os.path.basename(module_part)+".py":
             return os.path.splitext(os.path.basename(norm_file_path))[0] # Return filename without extension

        # Remove leading dots if any resulted from edge cases
        return module_path.strip('.') if module_path else None # Return None if it results in empty string

    except Exception as e:
        logging.error(f"Error calculating module path for '{file_path}' relative to '{project_root}': {e}", exc_info=True)
        return None
# In pipeline.py

# Import the new helper function if it's in utils.py
# from .utils import get_python_module_path
# Or if it's in pipeline.py itself, no import needed.

def build_qualified_name_map(unique_entities: List[Dict], project_root: str) -> Dict[str, str]:
    """
    Builds a map from qualified entity names (e.g., package.module.Class) to their node IDs.
    """
    logging.info(f"--- Building Qualified Name Map (Project Root: {project_root}) ---")
    qualified_name_map: Dict[str, str] = {}
    entities_processed = 0
    map_entries = 0

    for entity in unique_entities:
        entities_processed += 1
        entity_type = entity.get('entity_type')
        entity_id = entity.get('id')
        simple_name = entity.get('name')
        source_file = entity.get('source_file')

        # Only map defined Functions and Classes that have necessary info
        if entity_type in ['Function', 'Class'] and entity_id and simple_name and source_file:
            module_path = get_python_module_path(source_file, project_root)

            qualified_name = None
            if module_path:
                # For classes/functions inside __init__.py, the qualified name is module.EntityName
                # For others, it's module.submodule.EntityName
                qualified_name = f"{module_path}.{simple_name}"
            else:
                # Could be a file directly in the root or calculation failed
                # Treat it as a top-level entity for simplicity
                qualified_name = simple_name
                logging.debug(f"Could not determine module path for {entity_type} '{simple_name}' in '{source_file}'. Using simple name as qualified name.")

            if qualified_name:
                if qualified_name in qualified_name_map:
                    logging.warning(f"Duplicate qualified name detected! '{qualified_name}'. "
                                    f"Keeping ID '{qualified_name_map[qualified_name]}' from earlier entity. "
                                    f"Ignoring entity ID '{entity_id}' from '{source_file}'. "
                                    f"Consider more robust duplicate handling if this is common.")
                else:
                    qualified_name_map[qualified_name] = entity_id
                    map_entries += 1
                    logging.debug(f"Mapped QName '{qualified_name}' -> ID '{entity_id}'")

    logging.info(f"--- Qualified Name Map built: {map_entries} entries created from {entities_processed} unique entities ---")
    # Log a sample for verification
    sample_count = 0
    for k, v in qualified_name_map.items():
        if sample_count < 5:
            logging.debug(f"  Sample QName Map: '{k}' -> '{v}'")
            sample_count += 1
        else:
            break
    return qualified_name_map
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
    """
    Deduplicates entities using a multi-stage approach that considers both
    exact matches and semantic similarities.

    Returns a list of unique entities with generated IDs.
    """
    logging.info(f"--- Deduplicating {len(entities)} Structural Entities ---")

    # Primary deduplication based on core entity identity
    primary_keys = ('entity_type', 'name', 'source_file')
    secondary_keys = ('start_byte', 'end_byte', 'parent_name')
    deduped_map = {}
    duplicates_found = 0
    merged_entities = 0
    skipped_entities = 0

    # First pass: Group entities by primary key
    entity_groups = {}
    for entity in entities:
        try:
            # Require at minimum entity_type and name for grouping
            if not all(k in entity for k in ('entity_type', 'name')):
                logging.warning(f"Skipping entity missing minimal required fields: {entity.get('name', 'UNKNOWN')}")
                skipped_entities += 1
                continue

            # Create a more flexible entity key
            entity_type = entity.get('entity_type')
            name = entity.get('name')
            source_file = entity.get('source_file')

            # Primary key without file for functions/methods that might be referenced across files
            # This helps identify potential duplicates for merging
            primary_key = (entity_type, name)

            # Add to appropriate group
            if primary_key not in entity_groups:
                entity_groups[primary_key] = []
            entity_groups[primary_key].append(entity)

        except Exception as e:
            logging.warning(f"Error during entity grouping for {entity.get('name', 'N/A')}: {e}")
            skipped_entities += 1

    # Second pass: Process each group
    for (entity_type, name), group in entity_groups.items():
        try:
            # For single-item groups, just add with a generated ID
            if len(group) == 1:
                entity = group[0]
                entity['id'] = generate_unique_id(f"{entity_type}_{_sanitize_for_id(name)}_")

                # Use full key for exact deduplication
                full_key = _get_full_entity_key(entity, primary_keys)
                deduped_map[full_key] = entity
                continue

            # For multi-item groups, handle based on entity type
            if entity_type in ('Function', 'Class', 'Method'):
                # These could be legitimate duplicates or overloaded functions
                # Check if they're in different files or have different signatures
                _handle_function_class_group(group, deduped_map, primary_keys)

            elif entity_type == 'Call':
                # Calls to the same function might be semantically the same but at different locations
                _handle_call_group(group, deduped_map, primary_keys, secondary_keys)

            elif entity_type in ('Import', 'ImportFrom'):
                # Imports of the same module in different files are still unique imports
                _handle_import_group(group, deduped_map, primary_keys)

            else:
                # Default handling for other entity types
                for entity in group:
                    full_key = _get_full_entity_key(entity, primary_keys + ('start_byte',))
                    if full_key not in deduped_map:
                        entity['id'] = generate_unique_id(f"{entity_type}_{_sanitize_for_id(name)}_")
                        deduped_map[full_key] = entity
                    else:
                        duplicates_found += 1

        except Exception as e:
            logging.error(f"Error processing group for {entity_type} '{name}': {e}", exc_info=True)
            # Fall back to adding all group items individually to avoid data loss
            for entity in group:
                entity['id'] = generate_unique_id(f"{entity_type}_{_sanitize_for_id(name)}_fallback_")
                backup_key = (str(entity.get('entity_type')), str(entity.get('name')),
                              str(entity.get('source_file')), str(entity.get('start_byte')))
                deduped_map[backup_key] = entity

    deduped_list = list(deduped_map.values())
    logging.info(f"--- Deduplication complete: {len(deduped_list)} unique entities identified "
                 f"({duplicates_found} duplicates ignored, {merged_entities} entities merged, "
                 f"{skipped_entities} entities skipped) ---")
    return deduped_list


def _sanitize_for_id(name: str) -> str:
    """Sanitizes a name string for use in an ID."""
    if not name:
        return "unnamed"
    return "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in name)


def _get_full_entity_key(entity: Dict, key_fields: Tuple[str, ...]) -> Tuple:
    """Creates a tuple key from an entity using the specified fields."""
    return tuple(str(entity.get(k, '')) for k in key_fields)


def _handle_function_class_group(group: List[Dict], deduped_map: Dict, primary_keys: Tuple[str, ...]):
    """
    Handles deduplication for functions and classes.
    Preserves unique implementations while merging references.
    """
    # Group by source file to handle definitions vs references
    by_file = {}
    for entity in group:
        source_file = entity.get('source_file')
        if source_file not in by_file:
            by_file[source_file] = []
        by_file[source_file].append(entity)

    # For each file, include the most detailed entity (usually the one with a snippet)
    for source_file, file_entities in by_file.items():
        best_entity = None
        for entity in file_entities:
            if best_entity is None:
                best_entity = entity
            elif entity.get('snippet') and not best_entity.get('snippet'):
                best_entity = entity
            elif entity.get('snippet') and best_entity.get('snippet'):
                # If both have snippets, take the one with more information
                if len(entity.get('snippet', '')) > len(best_entity.get('snippet', '')):
                    best_entity = entity

        if best_entity:
            entity_type = best_entity.get('entity_type')
            name = best_entity.get('name')
            best_entity['id'] = generate_unique_id(
                f"{entity_type}_{_sanitize_for_id(name)}_{_sanitize_for_id(source_file)}_")
            full_key = _get_full_entity_key(best_entity, primary_keys + ('start_byte',))
            deduped_map[full_key] = best_entity


def _handle_call_group(group: List[Dict], deduped_map: Dict,
                       primary_keys: Tuple[str, ...], secondary_keys: Tuple[str, ...]):
    """
    Handles deduplication for call entities.
    Preserves calls from different contexts while merging identical calls.
    """
    # Group calls by parent context - calls in different functions are distinct
    by_context = {}
    for entity in group:
        parent_key = (entity.get('parent_type', ''), entity.get('parent_name', ''))
        if parent_key not in by_context:
            by_context[parent_key] = []
        by_context[parent_key].append(entity)

    # For each context, add unique calls
    for context_key, context_entities in by_context.items():
        for entity in context_entities:
            # Different calls to the same function within the same context
            # are distinguished by byte position
            full_key = _get_full_entity_key(entity, primary_keys + ('parent_type', 'parent_name', 'start_byte'))
            if full_key not in deduped_map:
                entity_type = entity.get('entity_type')
                name = entity.get('name')
                parent_name = entity.get('parent_name', '')
                entity['id'] = generate_unique_id(
                    f"{entity_type}_{_sanitize_for_id(name)}_in_{_sanitize_for_id(parent_name)}_")
                deduped_map[full_key] = entity


def _handle_import_group(group: List[Dict], deduped_map: Dict, primary_keys: Tuple[str, ...]):
    """
    Handles deduplication for import entities.
    Preserves imports in different files while merging duplicates within the same file.
    """
    # Group by file to handle multiple imports of the same module per file
    by_file = {}
    for entity in group:
        source_file = entity.get('source_file')
        if source_file not in by_file:
            by_file[source_file] = []
        by_file[source_file].append(entity)

    # For each file, include only unique imports
    for source_file, file_entities in by_file.items():
        # Track imports already added for this file
        added_imports = set()

        for entity in file_entities:
            entity_type = entity.get('entity_type')
            name = entity.get('name')

            # For ImportFrom, consider the items being imported
            if entity_type == 'ImportFrom':
                imported_items = tuple(sorted(entity.get('imported_items', [])))
                import_key = (name, imported_items)
            else:
                import_key = name

            if import_key not in added_imports:
                entity['id'] = generate_unique_id(
                    f"{entity_type}_{_sanitize_for_id(name)}_{_sanitize_for_id(source_file)}_")
                full_key = _get_full_entity_key(entity, primary_keys + ('start_byte',))
                deduped_map[full_key] = entity
                added_imports.add(import_key)

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
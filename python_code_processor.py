import os
import json
import uuid
import logging
from typing import List, Dict, Any, Tuple, Set, Union

# --- Tree-sitter and related imports ---
import tiktoken
from tree_sitter import Node, Language, Parser # Import Language and Parser directly
import tree_sitter_python as tspython # Import the specific language package

# --- Neo4j Import ---
from neo4j import GraphDatabase, Driver # Import Neo4j driver

# --- Langchain/Community Imports ---
# Ensure correct imports after installing langchain-community
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    print("Trying legacy import for HuggingFaceEmbeddings...")
    # Fallback for older Langchain versions if needed
    from langchain.embeddings import HuggingFaceEmbeddings

# --- ChromaDB Imports ---
import chromadb
from chromadb.utils import embedding_functions

# --- Ollama LLM Client Implementation (Keep as is from Snippet 2) ---
def get_llm_client():
    """Returns an Ollama LLM client configured for code understanding tasks."""
    try:
        from ollama import Client

        class OllamaClient:
            def __init__(self):
                self.client = Client(host='http://localhost:11434')
                # <<< --- CHOOSE YOUR OLLAMA MODEL --- >>>
                # Common choices: 'llama3.1:8b-instruct-q4_K_M', 'codellama', 'mistral', 'llama3:8b'
                # Ensure the chosen model is pulled: `ollama pull <model_name>`
                self.model = 'llama3.2'
                # <<< ----------------------------- >>>
                self.system_prompt = (
                    "You are an AI specialized in code analysis and documentation. "
                    "Provide concise, accurate responses about code structure and behavior. "
                    "When asked for a description, provide ONLY the description text."
                )

            def invoke(self, prompt: str) -> str:
                try:
                    response = self.client.generate(
                        model=self.model,
                        system=self.system_prompt,
                        prompt=prompt,
                        options={
                            'temperature': 0.2, # Lower temp for factual tasks
                            # 'num_ctx': 8192 # Optional: adjust context window if needed
                        }
                        # Consider adding format='json' if the Ollama version/model supports it reliably
                        # and adjust the prompt to explicitly request JSON output.
                    )
                    return response.get('response', '').strip() # Safely get the response text and strip whitespace
                except Exception as e:
                    logging.error(f"Ollama API call failed: {e}")
                    # Check connection details, if Ollama is running, and if model is pulled.
                    # Example check: is self.client.list() working?
                    try:
                        self.client.list() # Check connection health
                    except Exception as conn_e:
                        logging.error(f"Ollama connection test failed: {conn_e}")
                        logging.error("Please ensure Ollama server is running and accessible at the specified host.")
                    return f"Error generating response: {e}" # Return error message

            def predict(self, prompt: str) -> str:
                return self.invoke(prompt)

        # Test connection during initialization
        print(f"Attempting to connect to Ollama and verify model '{OllamaClient().model}'...")
        test_client = OllamaClient()
        try:
             # Check if model exists (more reliable than generating)
            model_info = test_client.client.show(test_client.model)
            if not model_info:
                 raise ConnectionError(f"Model '{test_client.model}' not found or Ollama inaccessible.")
            print(f"Ollama model '{test_client.model}' verified.")
            # Optional: Test generation if model verification works
            # test_response = test_client.invoke("Hello Ollama!")
            # print(f"Ollama test response snippet: {test_response[:100]}...")
            # if not test_response or "Error generating response" in test_response:
            #     raise ConnectionError("Ollama test generation failed.")
            print("Ollama connection successful.")
            return test_client
        except Exception as e:
            print(f"Ollama connection/verification failed: {e}")
            print(f"Falling back to placeholder LLM. Make sure Ollama is running (`ollama serve`) and the model '{test_client.model}' is available (`ollama pull {test_client.model}`).")
            return get_fallback_llm()


    except ImportError:
        print("Ollama Python package not found. Install with: pip install ollama")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()
    except Exception as e: # Catch other potential errors during setup
        print(f"An unexpected error occurred setting up Ollama: {e}")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()


# --- Fallback LLM (Keep as is from Snippet 2) ---
def get_fallback_llm():
    """Fallback placeholder LLM when Ollama isn't available"""
    class PlaceholderLLM:
        def invoke(self, prompt: str) -> str:
            print(f"--- PLACEHOLDER LLM PROMPT ---\n{prompt}\n--- END PROMPT ---")
            # Simulate JSON response structure expected by update_entity_descriptions
            if "update description for entity" in prompt.lower():
                 # Simulate only returning the description text
                return "Placeholder updated entity description."
            # Default fallback response (not necessarily JSON)
            return "Placeholder LLM response (Ollama unavailable)"
        def predict(self, prompt: str) -> str: return self.invoke(prompt)
    print("Using Placeholder LLM.") # Make it clear when fallback is used
    return PlaceholderLLM()

# --- Configuration (Adjust as needed) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Or another SentenceTransformer model
LLM_CLIENT = get_llm_client() # Initialize LLM client (Ollama or Fallback)
CHUNK_TOKEN_LIMIT = 350 # Adjust token limit for structure-aware chunking
CHUNK_ENCODING = "cl100k_base" # Encoding for chunking token count (matches tiktoken default)
EMBEDDING_ENCODING = "cl100k_base" # Encoding for general token counts (often same as chunking)
CHROMA_PATH_PREFIX = "./chroma_db_struct_chunk_" # Path prefix for ChromaDB persistence
# Neo4j Connection Details (Use Environment Variables!)
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "abcd12345") # CHANGE THIS DEFAULT

# Initialize ChromaDB Embedding Function
try:
    # Use SentenceTransformer directly via ChromaDB utils for consistency
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    logging.info(f"Successfully initialized embedding function with {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize SentenceTransformerEmbeddingFunction: {e}")
    logging.error("Ensure 'sentence-transformers' package is installed: pip install sentence-transformers")
    ef = None # Indicate failure

# KV Store (Simple in-memory, replace with persistent if needed)
kv_store = {}

# --- Helper Functions (Keep from Snippet 2) ---
def generate_unique_id(prefix: str = "") -> str:
    safe_prefix = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in prefix)
    return f"{safe_prefix}{uuid.uuid4()}"

def count_tokens(text: str, encoding_name: str = EMBEDDING_ENCODING) -> int: # Default to general encoding
    """Counts tokens more robustly."""
    if not text: return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        # Use `encode` with `disallowed_special=()` for a robust token count
        # This prevents errors if the text contains special tokens like <|endoftext|>
        tokens = encoding.encode(text, disallowed_special=())
        return len(tokens)
    except tiktoken.registry.RegistryError:
         logging.warning(f"Tiktoken encoding '{encoding_name}' not found. Falling back to tiktoken default 'cl100k_base'.")
         try:
             encoding = tiktoken.get_encoding("cl100k_base") # Fallback encoding
             return len(encoding.encode(text, disallowed_special=()))
         except Exception as e_fallback:
             logging.warning(f"Fallback token counting failed: {e_fallback}. Using character count / 4.")
             return len(text) // 4 # Rough approximation as last resort
    except Exception as e: # Catch other potential errors
        logging.warning(f"Token counting failed with '{encoding_name}': {e}. Falling back to char count / 4.")
        return len(text) // 4


# --- Modified CodeParser Class ---
class CodeParser:
    def __init__(self):
        self.language_name = "python"
        self.parser = None
        self.language = None # Store language object
        logging.info(f"Attempting to load parser for '{self.language_name}' using official method...")
        self._load_parser()

    def _load_parser(self):
        """Loads the Python parser using the installed language package."""
        try:
            logging.info("  Loading Language object from tree_sitter_python...")
            self.language = Language(tspython.language()) # Store language object
            logging.info(f"  Successfully created Language object: {type(self.language)}")
            logging.info("  Initializing Parser with the Language object...")
            self.parser = Parser(self.language) # Use stored language object
            logging.info(f"  Successfully initialized Parser object: {type(self.parser)}")

            test_code = "print('hello')"
            logging.info(f"  Attempting to parse test string: '{test_code}'")
            tree = self.parser.parse(bytes(test_code, "utf8"))
            if tree and tree.root_node:
                logging.info(f"  Test parse successful. Root node: {tree.root_node.type}")
                logging.info(f"Successfully loaded and tested parser for {self.language_name}")
            else:
                 logging.warning("  Test parse failed or returned empty tree.")
        except ImportError as e:
            logging.error(f"Failed to import 'tree_sitter_python' or its components.")
            logging.error(f"  Ensure 'tree-sitter-python' is installed (`pip install tree-sitter-python`).")
            logging.error(f"  ImportError: {e}")
            self.parser = None
        except Exception as e:
            logging.error(f"Failed to load or test parser for '{self.language_name}'.")
            logging.error(f"  Exception Type: {type(e)}")
            logging.error(f"  Error Details: {e}")
            self.parser = None

    def is_parser_loaded(self) -> bool:
        return self.parser is not None

    def parse_code(self, code: str) -> Union[None, Node]:
        if not self.is_parser_loaded(): return None
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree.root_node if tree and tree.root_node else None
        except Exception as e:
            logging.error(f"Error during parsing: {e}")
            return None

    # --- Methods for Detailed Entity Extraction (from Snippet 2) ---
    def _traverse_tree(self, node: Node, code: str, file_path: str, entities: List[Dict]):
        # (Keep the detailed entity traversal logic from Snippet 2)
        if node is None: return
        node_type_map = self._get_python_node_types_of_interest_for_entities() # Use specific map for entities
        entity_type = node_type_map.get(node.type)
        if entity_type:
            entity_name = self._get_entity_name(node, code, entity_type)
            start_line = node.start_point[0] + 1; end_line = node.end_point[0] + 1
            snippet = code[node.start_byte:node.end_byte]
            entity_data = {"entity_type": entity_type, "name": entity_name, "source_file": file_path,"start_line": start_line, "end_line": end_line, "start_byte": node.start_byte, "end_byte": node.end_byte, "snippet": snippet}
            parent_info = self._find_parent_context(node, code)
            if parent_info: entity_data["parent_type"], entity_data["parent_name"] = parent_info["type"], parent_info["name"]
            if entity_type == "ImportFrom": entity_data.update(self._extract_import_from_details(node, code))
            elif entity_type == "Import": entity_data['module_name'] = entity_name # Use extracted name as module
            entities.append(entity_data)
        try:
            for child in node.children:
                self._traverse_tree(child, code, file_path, entities)
        except Exception as e:
             logging.warning(f"Error traversing children of node type {node.type} at {file_path}:{node.start_point}: {e}")

    def extract_entities_from_file(self, code: str, file_path: str) -> List[Dict]:
        root_node = self.parse_code(code); entities = []
        if root_node: self._traverse_tree(root_node, code, file_path, entities)
        return entities

    def _get_python_node_types_of_interest_for_entities(self) -> Dict[str, str]:
        # Keep the detailed entity types from Snippet 2
        return {
            'import_statement': 'Import',
            'import_from_statement': 'ImportFrom',
            'class_definition': 'Class',
            'function_definition': 'Function',
            'decorated_definition': 'DecoratedDefinition', # Captures decorators with funcs/classes
            'call': 'Call',
            # Consider adding 'assignment' if variable assignments are important
            # 'expression_statement': 'Expression', # Too broad?
        }

    # Keep _get_entity_name, _find_parent_context, _extract_import_from_details, _find_python_imported_names
    # (Functions are identical to Snippet 2's versions)
    def _get_entity_name(self, node: Node, code: str, node_type: str) -> str:
        name = f"Unnamed_{node_type}_{node.start_byte}" # Default name
        try:
            if node_type in ["Function", "Class"]:
                 name_node = node.child_by_field_name('name')
                 if name_node and name_node.type == 'identifier':
                     name = code[name_node.start_byte:name_node.end_byte]
            elif node_type == "DecoratedDefinition":
                 # Find the actual function or class definition node within the decorated definition
                 def_node = next((c for c in node.children if c.type in ['function_definition', 'class_definition']), None)
                 if def_node:
                     inner_type = "Function" if def_node.type == "function_definition" else "Class"
                     name = self._get_entity_name(def_node, code, inner_type)
            elif node_type == "Call":
                 func_node = node.child_by_field_name('function')
                 if func_node:
                     # Handle different ways functions can be called (simple identifier, attribute access like obj.method)
                     if func_node.type == 'identifier':
                         name = code[func_node.start_byte:func_node.end_byte]
                     elif func_node.type == 'attribute': # e.g., instance.method() or Class.method()
                         # Get the attribute name (the actual method being called)
                         attr_node = func_node.child_by_field_name('attribute')
                         if attr_node:
                             name = code[attr_node.start_byte:attr_node.end_byte]
                         else: # Fallback if attribute isn't found (shouldn't normally happen)
                            name = code[func_node.start_byte:func_node.end_byte]
                     else: # Handle other callable types if necessary (e.g., subscript expression like list[0]())
                         name = code[func_node.start_byte:func_node.end_byte]
            elif node_type == "Import":
                 # Handle 'import module' and 'import module as alias'
                 # Look for 'dotted_name' or 'aliased_import'
                 name_node = next((c for c in node.children if c.type in ['dotted_name', 'aliased_import']), None)
                 if name_node:
                     if name_node.type == 'dotted_name':
                         name = code[name_node.start_byte:name_node.end_byte]
                     elif name_node.type == 'aliased_import':
                         # Get the original module name for consistency, though alias is also available
                         original_name_node = name_node.child_by_field_name('name')
                         if original_name_node:
                              name = code[original_name_node.start_byte:original_name_node.end_byte]
                         else: # Fallback: use alias if original name not found
                            alias_node = name_node.child_by_field_name('alias')
                            if alias_node:
                                name = code[alias_node.start_byte:alias_node.end_byte]
                 else: # Fallback for simple import structure if direct children aren't named fields
                    name = code[node.start_byte:node.end_byte].replace("import ", "").strip().split(' ')[0] # Basic split
            elif node_type == "ImportFrom":
                 # The 'name' for ImportFrom is typically the module it imports *from*
                 module_node = node.child_by_field_name('module_name')
                 if module_node:
                     name = code[module_node.start_byte:module_node.end_byte]
        except Exception as e:
            logging.warning(f"Error extracting name for {node_type} at {node.start_point}: {e}")

        # Sanitize name slightly (replace complex chars that might break IDs/queries)
        safe_name = "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in name)
        return safe_name if safe_name else f"Unnamed_{node_type}_{node.start_byte}"

    def _find_parent_context(self, node: Node, code: str) -> Union[Dict, None]:
        current = node.parent
        while current:
            p_type = current.type
            ctx_type = None
            name = None
            try: # Add try-except around name extraction
                if p_type == "function_definition":
                    ctx_type = "Function"
                    name = self._get_entity_name(current, code, ctx_type)
                elif p_type == "class_definition":
                    ctx_type = "Class"
                    name = self._get_entity_name(current, code, ctx_type)
                elif p_type == "decorated_definition":
                    # Find the inner definition to get its type and name
                     inner_def = next((c for c in current.children if c.type in ["function_definition", "class_definition"]), None)
                     if inner_def:
                         inner_type = "Function" if inner_def.type == "function_definition" else "Class"
                         name = self._get_entity_name(inner_def, code, inner_type)
                         return {"type": inner_type, "name": name} # Return the inner definition's context
                     else: # Should have an inner def, but continue search upwards if not found
                         current = current.parent
                         continue

                if ctx_type and name:
                    return {"type": ctx_type, "name": name}

            except Exception as e:
                 logging.warning(f"Error finding parent context name for node type {p_type}: {e}")
            # Move up the tree
            current = current.parent
        return None # No suitable parent context found

    def _extract_import_from_details(self, node: Node, code: str) -> Dict:
        details = {"imported_items": [], "module_name": "UnknownModule"}
        mod_node = node.child_by_field_name('module_name')
        if mod_node:
             details['module_name'] = code[mod_node.start_byte:mod_node.end_byte]

        # Find the node containing the imported names (can be different types)
        # common patterns: 'import_list', 'aliased_import', 'dotted_name', 'wildcard_import'
        names_part = next((c for c in node.children if c.type in ['import_list', 'aliased_import', 'dotted_name', 'wildcard_import']), None)

        if names_part:
            if names_part.type == 'wildcard_import':
                 details['imported_items'].append('*')
            else:
                 # Use a recursive helper to find all identifiers/aliases within the names part
                 self._find_python_imported_names(names_part, code, details['imported_items'])
        elif node.named_child_count > 1: # Fallback: Check direct named children if no specific container node found
            # This might catch cases like 'from . import foo' where foo is a direct child identifier
             potential_name_node = node.named_children[-1] # Often the last element
             if potential_name_node.type in ['identifier', 'dotted_name']:
                 self._find_python_imported_names(potential_name_node, code, details['imported_items'])

        return details

    def _find_python_imported_names(self, node: Node, code: str, items_list: List[str]):
        """Recursively find imported names, including aliases."""
        try:
            if node.type == 'identifier':
                 # Avoid capturing module name parts if it's part of a dotted name handled elsewhere
                 # Only add if it's likely a direct import item or alias name
                 if node.parent and node.parent.type not in ['dotted_name']:
                     items_list.append(code[node.start_byte:node.end_byte])
                 elif not node.parent: # Top-level identifier under import_list etc.
                     items_list.append(code[node.start_byte:node.end_byte])
            elif node.type == 'dotted_name':
                # Add the full dotted name if it's not part of an alias definition itself
                if not (node.parent and node.parent.type == 'aliased_import' and node == node.parent.child_by_field_name('name')):
                     items_list.append(code[node.start_byte:node.end_byte])
            elif node.type == 'aliased_import':
                orig_node = node.child_by_field_name('name')
                alias_node = node.child_by_field_name('alias')
                o_name = code[orig_node.start_byte:orig_node.end_byte] if orig_node else '??'
                a_name = code[alias_node.start_byte:alias_node.end_byte] if alias_node else '??'
                items_list.append(f"{o_name} as {a_name}")
            else:
                # Recursively search children unless it's an alias where we handled it
                 if node.type != 'aliased_import':
                    for child in node.children:
                        self._find_python_imported_names(child, code, items_list)
        except Exception as e:
             logging.warning(f"Error finding imported names in node type {node.type}: {e}")


    # --- NEW Methods for Structure-Aware Chunking (Adapted from Snippet 1) ---

    def _get_python_node_types_of_interest_for_chunking(self) -> Dict[str, str]:
        """Defines node types considered major structural elements for chunking."""
        # Simpler set than entities, focused on block start points
        return {
            'import_statement': 'Import',
            'import_from_statement': 'ImportFrom', # Treat 'from X import Y' as a block start
            'class_definition': 'Class',
            'function_definition': 'Function',
            'decorated_definition': 'DecoratedDefinition', # Start of a decorated func/class
            # Maybe add top-level 'if __name__ == "__main__":' ?
            # 'if_statement': 'If', # Could add top-level ifs if desired
        }

    def _get_python_nodes_for_comments(self) -> Dict[str, str]:
        """Defines node types representing comments or decorators for chunking adjustment."""
        return {
            'comment': 'Comment',
            'decorator': 'Decorator', # Keep decorators with the definition below
        }

    def _traverse_for_lines(self, node: Node, types_of_interest: Dict[str, str], lines_found: List[int]):
        """Generic traversal to collect start line numbers for specific node types."""
        if node is None:
            return
        if node.type in types_of_interest:
            lines_found.append(node.start_point[0]) # 0-based line index

        for child in node.children:
            self._traverse_for_lines(child, types_of_interest, lines_found)

    def extract_lines_for_points_of_interest(self, root_node: Node) -> List[int]:
        """Extracts 0-based start line indices for chunking breakpoints."""
        if not root_node:
            return []
        node_types = self._get_python_node_types_of_interest_for_chunking()
        lines_found = []
        self._traverse_for_lines(root_node, node_types, lines_found)
        return sorted(list(set(lines_found))) # Return unique, sorted 0-based line indices

    def extract_lines_for_comments(self, root_node: Node) -> List[int]:
        """Extracts 0-based start line indices for comments and decorators."""
        if not root_node:
            return []
        node_types = self._get_python_nodes_for_comments()
        lines_found = []
        self._traverse_for_lines(root_node, node_types, lines_found)
        return sorted(list(set(lines_found))) # Return unique, sorted 0-based line indices


# --- NEW Structure-Aware Chunking Function (Adapted from Snippet 1) ---
def chunk_code_structure_aware(code: str, token_limit: int, source_file: str, code_parser: CodeParser, encoding_name: str = CHUNK_ENCODING) -> List[Dict]:
    """
    Chunks code based on structural elements (functions, classes, imports)
    and token limits, keeping associated comments/decorators with their blocks.

    Args:
        code: The source code string.
        token_limit: The maximum number of tokens allowed per chunk.
        source_file: The path of the source file (for metadata).
        code_parser: An instance of the CodeParser.
        encoding_name: The name of the tiktoken encoding to use for counting.

    Returns:
        A list of chunk dictionaries, each with 'id', 'text', and 'metadata'.
    """
    logging.debug(f"Starting structure-aware chunking for {source_file} with limit {token_limit}")
    chunks = []
    if not code.strip():
        logging.debug("Skipping empty code.")
        return chunks

    root_node = code_parser.parse_code(code)
    if not root_node:
        logging.warning(f"Could not parse code for {source_file}. Falling back to simple line splitting (not implemented here, returning no chunks).")
        # TODO: Implement a fallback simple chunker if parsing fails?
        # For now, return empty list if parsing fails.
        return chunks

    lines = code.splitlines() # Use splitlines to handle different line endings consistently
    num_lines = len(lines)
    if num_lines == 0:
        return chunks

    # Get 0-based line indices for breakpoints and comments
    breakpoints_indices = code_parser.extract_lines_for_points_of_interest(root_node)
    comment_indices = code_parser.extract_lines_for_comments(root_node)
    comment_set = set(comment_indices) # Faster lookups

    logging.debug(f"Initial breakpoints (0-based indices): {breakpoints_indices}")
    logging.debug(f"Comment/decorator lines (0-based indices): {comment_indices}")

    # Adjust breakpoints: If a breakpoint is preceded by comments/decorators,
    # move the breakpoint up to the start of that comment/decorator block.
    adjusted_breakpoints = set()
    for bp_idx in breakpoints_indices:
        current_line_idx = bp_idx - 1
        actual_bp_idx = bp_idx
        # Walk backwards as long as the line is a comment/decorator
        while current_line_idx >= 0 and current_line_idx in comment_set:
            actual_bp_idx = current_line_idx
            current_line_idx -= 1
        adjusted_breakpoints.add(actual_bp_idx)

    # Add line 0 as an implicit breakpoint
    adjusted_breakpoints.add(0)
    # Convert back to sorted list
    final_breakpoints = sorted(list(adjusted_breakpoints))
    logging.debug(f"Final adjusted breakpoints (0-based indices): {final_breakpoints}")


    current_chunk_lines = []
    current_token_count = 0
    start_line_idx = 0 # Start index of the current chunk
    chunk_number = 1

    i = 0
    while i < num_lines:
        line = lines[i]
        # Add 1 token for the newline character that splitlines removes,
        # unless it's the very last line.
        line_token_count = count_tokens(line, encoding_name) + (1 if i < num_lines - 1 else 0)

        # --- Decision Logic ---
        # 1. Check if adding the current line *exceeds* the token limit.
        if current_token_count > 0 and current_token_count + line_token_count > token_limit:
            logging.debug(f"Line {i+1} ('{line[:30]}...') exceeds token limit ({current_token_count} + {line_token_count} > {token_limit})")
            # Find the *last* valid breakpoint *before* the current line 'i'.
            # The breakpoint must also be *at or after* the start of the current chunk.
            possible_stops = [bp for bp in final_breakpoints if start_line_idx <= bp < i]

            if possible_stops:
                stop_line_idx = max(possible_stops)
                logging.debug(f"Found suitable breakpoint at index {stop_line_idx} (line {stop_line_idx+1})")
            else:
                # No breakpoint between start_line_idx and i.
                # This means the current accumulating chunk started *after* the last breakpoint,
                # and adding line 'i' makes it too long.
                # Action: Chunk *before* line 'i'. If current_chunk_lines is empty,
                # force line 'i' into its own (potentially oversized) chunk.
                if not current_chunk_lines:
                     logging.warning(f"Line {i+1} itself exceeds token limit ({line_token_count}). Creating oversized chunk.")
                     stop_line_idx = i + 1 # Chunk includes this long line
                else:
                    logging.debug(f"No breakpoint found between {start_line_idx} and {i}. Chunking before line {i+1}.")
                    stop_line_idx = i # Chunk ends *before* the current line 'i'

            # --- Create the chunk ---
            chunk_text = "\n".join(lines[start_line_idx:stop_line_idx])
            if chunk_text.strip():
                chunk_id = generate_unique_id(f"chunk_{os.path.basename(source_file)}_{chunk_number}_")
                metadata = {
                    "source_file": source_file,
                    "language": "python", # Hardcoded for this pipeline
                    "chunk_index": chunk_number,
                    "start_line": start_line_idx + 1, # Convert 0-based index to 1-based line number
                    "end_line": stop_line_idx # End line is exclusive index, so it's the correct 1-based line number
                }
                chunks.append({"id": chunk_id, "text": chunk_text, "metadata": metadata})
                logging.debug(f"Created chunk {chunk_number}: Lines {metadata['start_line']}-{metadata['end_line']}")
                chunk_number += 1

            # --- Reset for next chunk ---
            start_line_idx = stop_line_idx
            current_chunk_lines = []
            current_token_count = 0
            i = stop_line_idx # Continue processing from the stop line

            # We need to re-evaluate the line at the new 'i' (start of the next chunk)
            continue # Skip the accumulation part for this iteration

        # 2. Accumulate the current line if it fits
        current_chunk_lines.append(line)
        current_token_count += line_token_count
        logging.debug(f"  Accumulated line {i+1}, token count: {current_token_count}")

        # 3. Check if the *next* line is a breakpoint. If so, chunk *now* even if under token limit.
        # This helps keep logical blocks together.
        # Exception: Don't force split if the current chunk is empty.
        if (i + 1) in final_breakpoints and current_chunk_lines:
             logging.debug(f"Next line ({i+2}) is a breakpoint. Finalizing current chunk.")
             chunk_text = "\n".join(current_chunk_lines)
             # No need to check strip() here as we only add if current_chunk_lines is not empty
             chunk_id = generate_unique_id(f"chunk_{os.path.basename(source_file)}_{chunk_number}_")
             metadata = {
                 "source_file": source_file,
                 "language": "python",
                 "chunk_index": chunk_number,
                 "start_line": start_line_idx + 1,
                 "end_line": i + 1 # Current line 'i' is the last line of this chunk
             }
             chunks.append({"id": chunk_id, "text": chunk_text, "metadata": metadata})
             logging.debug(f"Created chunk {chunk_number}: Lines {metadata['start_line']}-{metadata['end_line']} (breakpoint split)")
             chunk_number += 1

             # --- Reset for next chunk ---
             start_line_idx = i + 1
             current_chunk_lines = []
             current_token_count = 0
             # No need to set 'i' here, the loop increment will handle it

        i += 1 # Move to the next line

    # --- Handle the last chunk ---
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        # No need to check strip() here? Or maybe yes, if last lines were just whitespace.
        if chunk_text.strip():
            chunk_id = generate_unique_id(f"chunk_{os.path.basename(source_file)}_{chunk_number}_")
            metadata = {
                "source_file": source_file,
                "language": "python",
                "chunk_index": chunk_number,
                "start_line": start_line_idx + 1,
                "end_line": num_lines # Last line is inclusive
            }
            chunks.append({"id": chunk_id, "text": chunk_text, "metadata": metadata})
            logging.debug(f"Created final chunk {chunk_number}: Lines {metadata['start_line']}-{metadata['end_line']}")

    logging.info(f"Finished structure-aware chunking for {source_file}. Created {len(chunks)} chunks.")
    return chunks


# --- Pipeline Steps (Mostly from Snippet 2, with chunking call changed) ---
def parse_and_extract(python_files: List[str], code_parser: CodeParser) -> List[Dict]:
    logging.info(f"--- 1. Parsing {len(python_files)} Python Files & Extracting Structural Entities ---")
    all_entities = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: code = f.read()
            logging.info(f"  Parsing for entities: {file_path}")
            # Use the detailed entity extraction method
            file_entities = code_parser.extract_entities_from_file(code, file_path)
            all_entities.extend(file_entities); logging.info(f"    Extracted {len(file_entities)} entities from {file_path}")
        except Exception as e: logging.error(f"  Error parsing/extracting entities from file {file_path}: {e}", exc_info=True)
    logging.info(f"--- Total structural entities extracted: {len(all_entities)} ---")
    return all_entities

def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    # (Keep function as is from Snippet 2)
    logging.info(f"--- 2. Deduplicating {len(entities)} Structural Entities ---")
    key_fields = ('entity_type', 'name', 'source_file', 'start_byte') # Using start_byte for uniqueness
    deduped_map: Dict[Tuple, Dict] = {}; duplicates_found = 0
    for entity in entities:
        try:
            # Ensure key components are hashable (strings, ints, etc.)
            entity_key = tuple(str(entity.get(k)) for k in key_fields) # Convert all to string for key
            if entity_key not in deduped_map:
                 # Assign unique ID only to unique entities
                 entity['id'] = generate_unique_id(f"{entity.get('entity_type', 'ent')}_")
                 deduped_map[entity_key] = entity
            else: duplicates_found += 1
        except Exception as e:
             logging.warning(f"Error during deduplication key creation for {entity.get('name', 'N/A')}: {e}")
             # Assign unique ID anyway if key creation fails? Or skip? Let's skip.
             # entity['id'] = generate_unique_id(f"error_{entity.get('entity_type', 'ent')}_")
             # error_key = uuid.uuid4() # Use UUID as fallback key
             # deduped_map[error_key] = entity

    deduped_list = list(deduped_map.values())
    logging.info(f"--- Deduplication complete: {len(deduped_list)} unique entities identified ({duplicates_found} duplicates ignored) ---")
    return deduped_list

def update_entity_descriptions(entities: List[Dict], llm_client) -> List[Dict]:
    # (Keep function as is from Snippet 2, check for placeholder)
    logging.info(f"--- 3. Updating Entity Descriptions using LLM ---")
    updated_items = []
    if isinstance(llm_client, get_fallback_llm().__class__): # Check if it's the placeholder
         logging.info("  Skipping LLM description update (Using placeholder or Ollama unavailable/disabled).")
         # Add a default description if none exists
         for item in entities:
             if 'description' not in item:
                 item['description'] = f"Default description for {item.get('entity_type')} '{item.get('name')}'."
             updated_items.append(item)
         return updated_items

    total_entities = len(entities)
    logging.info(f"  Attempting to update descriptions for {total_entities} entities...")
    for i, item in enumerate(entities):
        # Log progress periodically
        if (i + 1) % 20 == 0 or i == total_entities - 1:
            logging.info(f"  Updating description for entity {i+1}/{total_entities}...")

        # Prepare the prompt for the LLM
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

        Description:""" # End prompt clearly asking for description

        try:
            # Use invoke which might handle retries/errors better depending on implementation
            description = llm_client.invoke(prompt).strip()

            # Basic validation of the response
            if description and "Error generating response" not in description and "placeholder" not in description.lower() :
                item['description'] = description
                logging.debug(f"    Updated description for {item.get('id')}: {description[:100]}...")
            elif 'description' not in item: # Add default if LLM failed and no prior desc exists
                 item['description'] = f"Default description for {item.get('entity_type')} '{item.get('name')}' (LLM update failed)."
                 logging.warning(f"    LLM description update failed or returned empty for {item.get('id')}. Used default.")
            # else: keep existing description if LLM fails

        except Exception as e:
             logging.error(f"  LLM Error updating description for {item.get('id', 'N/A')}: {e}")
             if 'description' not in item: # Ensure default description exists on error
                 item['description'] = f"Default description for {item.get('entity_type')} '{item.get('name')}' (LLM error)."

        updated_items.append(item) # Append regardless of update success

    logging.info(f"--- Description update pass complete ({len(updated_items)} processed) ---")
    return updated_items

# --- MODIFIED chunk_files function ---
def chunk_files(python_files: List[str], code_parser: CodeParser, token_limit: int) -> List[Dict]:
    logging.info(f"--- 4. Chunking Python File Content (Structure-Aware, Token Limit: {token_limit}) ---")
    all_chunks = []
    if not code_parser.is_parser_loaded():
        logging.error("Code parser not loaded. Cannot perform structure-aware chunking.")
        return all_chunks # Return empty list

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: code = f.read()
            if not code.strip():
                logging.info(f"  Skipping empty file: {file_path}")
                continue

            # --- Call the NEW structure-aware chunking function ---
            file_chunks = chunk_code_structure_aware(
                code=code,
                token_limit=token_limit,
                source_file=file_path,
                code_parser=code_parser,
                encoding_name=CHUNK_ENCODING # Use encoding specified for chunking
            )
            # --------------------------------------------------------

            all_chunks.extend(file_chunks)
            logging.info(f"  Chunked '{os.path.basename(file_path)}' into {len(file_chunks)} structure-aware chunks.")
        except Exception as e:
            logging.error(f"  Error chunking file {file_path}: {e}", exc_info=True)

    logging.info(f"--- Total structure-aware text chunks created: {len(all_chunks)} ---")
    return all_chunks

def embed_and_store_chunks(chunks: List[Dict], embedding_function, vector_db_client, collection_name: str):
    # (Keep function as is from Snippet 2, but ensure `ef` is used correctly)
    logging.info(f"--- 5. Embedding & Storing {len(chunks)} Text Chunks in ChromaDB ---")
    if not chunks:
        logging.warning("  No chunks provided to embed.")
        return
    if embedding_function is None:
        logging.error("  Embedding function (ef) is not available. Skipping chunk embedding.")
        return

    texts = [c['text'] for c in chunks]
    ids = [c['id'] for c in chunks]
    metadatas = [c['metadata'] for c in chunks]

    try:
        # Use the provided embedding_function (ef assumed to be compatible with ChromaDB)
        collection = vector_db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function # Pass the function here
            # metadata={"hnsw:space": "cosine"} # Optional: Specify distance metric if needed
        )

        # Sanitize metadata: Convert non-primitive types to strings
        valid_metadatas = []
        for meta in metadatas:
            valid_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    valid_meta[k] = v
                else:
                    valid_meta[k] = str(v) # Convert lists, dicts, etc., to string
            valid_metadatas.append(valid_meta)

        # Add to ChromaDB in batches
        batch_size = 166 # ChromaDB recommended batch size (adjust based on performance/memory)
        num_batches = (len(ids) + batch_size - 1) // batch_size
        logging.info(f"  Adding {len(ids)} chunks to collection '{collection_name}' in {num_batches} batches...")

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:min(i + batch_size, len(ids))]
            batch_texts = texts[i:min(i + batch_size, len(ids))]
            batch_metadatas = valid_metadatas[i:min(i + batch_size, len(ids))]

            if not batch_ids: continue # Should not happen, but safeguard

            logging.debug(f"    Adding batch {i//batch_size + 1}/{num_batches} (size {len(batch_ids)})")
            # Note: ChromaDB calculates embeddings automatically when using embedding_function
            collection.add(
                ids=batch_ids,
                documents=batch_texts, # Pass texts, embedding is handled by `ef`
                metadatas=batch_metadatas
            )
        logging.info(f"--- Stored {len(ids)} chunk texts and metadata in ChromaDB collection '{collection_name}' (Embeddings generated by ChromaDB) ---")

    except Exception as e:
        logging.error(f"  Error embedding or storing chunks in ChromaDB: {e}", exc_info=True)
        logging.error("  Check ChromaDB connection, collection status, and data format.")


def embed_and_store_entities(entities: List[Dict], embedding_function, vector_db_client, collection_name: str):
     # (Keep function as is from Snippet 2, ensure `ef` is used correctly)
     logging.info(f"--- 6. Embedding & Storing {len(entities)} Unique Entities in ChromaDB ---")
     if not entities:
         logging.warning("  No entities provided to embed.")
         return
     if embedding_function is None:
        logging.error("  Embedding function (ef) is not available. Skipping entity embedding.")
        return

     ids = []; texts_to_embed = []; metadatas = []
     for e in entities:
         if 'id' not in e:
             logging.warning(f"Skipping entity missing 'id': {e.get('name', 'N/A')}")
             continue

         ids.append(e['id'])
         # Construct the text representation for embedding
         text = f"Entity Type: {e.get('entity_type', 'Unknown')}\n"
         text += f"Name: {e.get('name', 'Unnamed')}\n"
         text += f"Source File: {e.get('source_file', 'N/A')}:{e.get('start_line', 'N/A')}\n"
         if e.get('parent_type') and e.get('parent_name'):
             text += f"Context: Inside {e.get('parent_type')} {e.get('parent_name')}\n"
         if e.get('description'): # Use the potentially LLM-updated description
             text += f"Description: {e.get('description')}\n"
         else: # Fallback if no description
             text += "Description: Not Available\n"

         # Include only a small part of the snippet in the text for embedding? Optional.
         # snippet_preview = e.get('snippet', '')[:300] # Limit snippet length
         # text += f"\nCode Snippet Preview:\n```python\n{snippet_preview}\n```"
         texts_to_embed.append(text)

         # Prepare metadata - exclude snippet, ID, and potentially large fields
         meta = {k: v for k, v in e.items() if k not in ['snippet', 'id', 'description']} # Exclude full snippet and desc from meta
         valid_meta = {}
         for k, v in meta.items():
             if isinstance(v, (str, int, float, bool)) or v is None:
                 valid_meta[k] = v
             else:
                 valid_meta[k] = str(v) # Convert lists, dicts, etc., to string
         # Add description to metadata as well for retrieval filtering
         valid_meta['description'] = e.get('description', '')
         metadatas.append(valid_meta)

     if not ids:
         logging.warning("  No valid entities with IDs found to embed.")
         return

     try:
         collection = vector_db_client.get_or_create_collection(
             name=collection_name,
             embedding_function=embedding_function # Pass the function
             # metadata={"hnsw:space": "cosine"}
        )

         batch_size = 166 # ChromaDB recommended batch size
         num_batches = (len(ids) + batch_size - 1) // batch_size
         logging.info(f"  Adding {len(ids)} entities to collection '{collection_name}' in {num_batches} batches...")

         for i in range(0, len(ids), batch_size):
             batch_ids = ids[i:min(i + batch_size, len(ids))]
             batch_texts = texts_to_embed[i:min(i + batch_size, len(ids))]
             batch_metadatas = metadatas[i:min(i + batch_size, len(ids))]

             if not batch_ids: continue

             logging.debug(f"    Adding batch {i//batch_size + 1}/{num_batches} (size {len(batch_ids)})")
             collection.add(
                 ids=batch_ids,
                 documents=batch_texts, # Let ChromaDB handle embedding via `ef`
                 metadatas=batch_metadatas
             )
         logging.info(f"--- Stored {len(ids)} entity texts and metadata in ChromaDB collection '{collection_name}' (Embeddings generated by ChromaDB) ---")
     except Exception as e:
         logging.error(f"  Error embedding or storing entities in ChromaDB: {e}", exc_info=True)
         logging.error("  Check ChromaDB connection, collection status, and data format.")

# --- Neo4j Helper Functions (Keep as is from Snippet 2) ---
# Requires APOC plugin installed in Neo4j for dynamic labels (`apoc.create.addLabels`)
# If APOC is not available, you must use fixed labels or multiple MERGE statements per type.

def _run_write_query(tx, query, **params):
    """Helper to run a write query with parameters, logging errors."""
    try:
        result = tx.run(query, **params)
        # Optional: Log summary or check for issues if needed
        # summary = result.consume()
        # logging.debug(f"Neo4j write query executed. Summary: {summary.counters}")
        return True # Indicate success
    except Exception as e:
        # More specific logging for Neo4j errors
        logging.error(f"Neo4j query failed!")
        logging.error(f"  Error Type: {type(e).__name__}")
        logging.error(f"  Error Details: {e}")
        logging.error(f"  Query: {query}")
        logging.error(f"  Params: {json.dumps(params, default=str, indent=2)}") # Log params safely
        raise # Re-raise to signal transaction failure

def add_neo4j_node(tx, node_data: Dict):
    """Creates or merges a node in Neo4j using MERGE and dynamic labels via APOC."""
    node_id = node_data.get('id')
    node_type = node_data.get('entity_type', 'UnknownEntity') # Default type if missing
    if not node_id:
        logging.warning(f"Skipping Neo4j node write, missing 'id' in data: {node_data.get('name', 'N/A')}")
        return

    # Sanitize label: Must start with upper case, alphanumeric + _
    safe_node_type = "".join(c if c.isalnum() or c=='_' else '' for c in str(node_type))
    if not safe_node_type or not safe_node_type[0].isupper():
        safe_node_type = f"Type_{safe_node_type}" if safe_node_type else "UnknownEntity" # Ensure valid label

    # Prepare properties: Exclude id, snippet, None values. Convert complex types to string.
    props_to_set = {}
    for k, v in node_data.items():
        # Keep essential fields even if None? Decide based on schema needs.
        # Here, we exclude None values.
        if k not in ['id', 'snippet'] and v is not None:
             if isinstance(v, (list, dict)): # Convert lists/dicts to JSON strings for Neo4j property
                 props_to_set[k] = json.dumps(v)
             elif isinstance(v, (str, int, float, bool)):
                 props_to_set[k] = v
             else: # Convert other types to string
                 props_to_set[k] = str(v)

    # Add the primary type as a property too, useful for filtering without dynamic labels
    props_to_set['entity_type'] = node_type # Store original type name

    # Use MERGE on :KGNode(id), then set specific type label(s) and properties
    # REQUIRES APOC plugin installed in Neo4j.
    # Creates a base KGNode, merges properties, then adds the specific type label.
    query = """
    MERGE (n:KGNode {id: $id})
    // Set properties ON CREATE and ON MATCH to ensure they are updated
    // Use direct SET for both cases, += can have issues with type mismatches
    SET n = $props
    SET n.id = $id // Ensure ID is explicitly set/kept
    WITH n
    // Use APOC to dynamically add the specific label (e.g., :Function, :Class)
    CALL apoc.create.addLabels(n, [$node_type_label]) YIELD node
    RETURN count(node) as nodes_labeled
    """
    # Alternative without APOC (less flexible type labeling):
    # query = f"""
    # MERGE (n:KGNode:{safe_node_type} {{id: $id}}) // Label must be static in Cypher without APOC
    # ON CREATE SET n = $props, n.id = $id
    # ON MATCH SET n = $props, n.id = $id // Overwrite properties on match
    # """
    # Note: Without APOC, you lose the ability to easily query across all KGNodes if needed.

    params = {'id': node_id, 'props': props_to_set, 'node_type_label': safe_node_type}
    try:
        _run_write_query(tx, query, **params)
        logging.debug(f"Successfully added/merged Neo4j node: {node_id} (Type: {safe_node_type})")
    except Exception as e:
        # Error logged within _run_write_query, just log context here
        logging.error(f"Failed to write Neo4j node for ID: {node_id}")
        # Don't stop the whole process, but log failure.


def add_neo4j_potential_function_node(tx, name: str):
    """Creates/merges a placeholder node for functions called but not defined in the parsed files."""
    if not name or not isinstance(name, str): return None # Basic validation
    # Sanitize name for ID
    safe_name = "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in name)
    node_id = f"PotentialFunction:{safe_name}" # Use prefix to distinguish
    node_type = 'PotentialFunction'  # Define the label string
    props = {'id': node_id, 'name': name, 'entity_type': node_type} # Include entity_type in props

    # Use the same APOC query pattern as in add_neo4j_node
    query = """
     MERGE (n:KGNode {id: $id})
     ON CREATE SET n = $props, n.created_at = timestamp()
     ON MATCH SET n += $props, n.last_seen = timestamp() // Update props and add timestamp
     WITH n
     CALL apoc.create.addLabels(n, [$node_type_label]) YIELD node
     RETURN count(node)
     """
    params = {'id': node_id, 'props': props, 'node_type_label': node_type}
    try:
        _run_write_query(tx, query, **params)
        logging.debug(f"Added/updated potential function node: {node_id}")
        return node_id # Return the ID, might be useful
    except Exception as e:
        logging.error(f"Failed to write Neo4j potential function node for: {name}")
        return None

def add_neo4j_edge(tx, source_id: str, target_id: str, rel_type: str, rel_props: Dict = None):
    """Creates or merges a relationship between two existing KGNode entities."""
    if not source_id or not target_id or not rel_type:
        logging.warning(f"Skipping Neo4j edge: Missing source/target ID or relationship type.")
        return
    if source_id == target_id:
         logging.debug(f"Skipping self-referential edge: {source_id} -[{rel_type}]-> {target_id}")
         return

    rel_props = rel_props or {}
    # Sanitize relationship type: Uppercase, alphanumeric + _
    safe_rel_type = "".join(c if c.isalnum() or c=='_' else '' for c in str(rel_type)).upper()
    if not safe_rel_type: safe_rel_type = "RELATED_TO" # Default relationship

    # Prepare properties: Convert complex types, exclude None
    props_to_set = {}
    for k, v in rel_props.items():
        if v is not None:
             if isinstance(v, (list, dict)):
                 props_to_set[k] = json.dumps(v)
             elif isinstance(v, (str, int, float, bool)):
                 props_to_set[k] = v
             else:
                 props_to_set[k] = str(v)

    # MERGE relationship: Creates if not exists, updates properties if exists.
    query = """
    MATCH (a:KGNode {id: $source_id})
    MATCH (b:KGNode {id: $target_id})
    MERGE (a)-[r:""" + safe_rel_type + """]->(b)
    // Use SET instead of += to overwrite properties completely on match
    SET r = $props
    """
    params = {'source_id': source_id, 'target_id': target_id, 'props': props_to_set}
    try:
        _run_write_query(tx, query, **params)
        logging.debug(f"Successfully added/merged Neo4j edge: ({source_id})-[{safe_rel_type}]->({target_id})")
    except Exception as e:
        logging.error(f"Failed to write Neo4j edge: ({source_id})-[{safe_rel_type}]->({target_id})")


# --- MODIFIED Graph Population Logic ---
def populate_neo4j_graph(driver: Driver, unique_entities: List[Dict], all_extracted_entities: List[Dict], file_paths: List[str]):
    logging.info("--- 7. Populating Neo4j Knowledge Graph ---")
    # Build maps for quick lookup
    entity_map_by_id = {e['id']: e for e in unique_entities if 'id' in e}
    # Key: (entity_type, name, source_file, start_byte) -> Value: entity_id
    entity_lookup_key_to_id = {
        (e.get('entity_type'), e.get('name'), e.get('source_file'), e.get('start_byte')): e['id']
        for e in unique_entities if 'id' in e and all(k in e for k in ['entity_type', 'name', 'source_file', 'start_byte'])
    }

    nodes_to_write_batch = []
    edges_to_write_batch = [] # List of tuples: (source_id, target_id, rel_type, rel_props)
    potential_funcs_to_create = set() # Store names of potential functions

    # --- Prepare File Nodes ---
    logging.info("  Preparing File nodes...")
    file_node_ids = {}
    for fp in file_paths:
        # Create a more robust/unique ID for files, e.g., based on relative path or hash
        # Using absolute path might be too long or contain invalid characters
        rel_path = os.path.relpath(fp) # Get relative path
        safe_path_id = "".join(c if c.isalnum() or c in ['_', '-', '.', '/'] else '_' for c in rel_path)
        node_id = f"File:{safe_path_id}"
        file_node_ids[fp] = node_id
        nodes_to_write_batch.append({'id': node_id, 'entity_type': 'File', 'path': rel_path, 'absolute_path': fp})

    # --- Prepare Entity Nodes ---
    logging.info("  Preparing Entity nodes...")
    for entity_id, entity_data in entity_map_by_id.items():
         nodes_to_write_batch.append(entity_data) # Add unique entities to batch

    # --- Prepare Edges & Potential Functions ---
    logging.info("  Preparing Edges and identifying Potential Functions...")
    for entity in all_extracted_entities: # Iterate through ALL entities found initially
        entity_key = (entity.get('entity_type'), entity.get('name'), entity.get('source_file'), entity.get('start_byte'))
        entity_id = entity_lookup_key_to_id.get(entity_key)

        if not entity_id:
            logging.debug(f"Skipping edge creation for non-unique/missing entity: {entity_key}")
            continue # Skip if this entity wasn't considered unique (or failed ID creation)

        source_file = entity.get('source_file')
        file_node_id = file_node_ids.get(source_file)

        # 1. File CONTAINS Entity Edge
        if file_node_id:
            edges_to_write_batch.append((file_node_id, entity_id, "CONTAINS", {'start': entity.get('start_line'), 'end': entity.get('end_line')}))

        # 2. Class CONTAINS Method Edge (Function within a Class context)
        parent_type = entity.get('parent_type'); parent_name = entity.get('parent_name')
        if entity.get('entity_type') == 'Function' and parent_type == 'Class' and parent_name:
             # Find the ID of the parent class defined *in the same file*
             # This assumes class names are unique within a file for simplicity
             potential_parent_keys = [k for k in entity_lookup_key_to_id if k[0]=='Class' and k[1]==parent_name and k[2]==source_file]
             if len(potential_parent_keys) == 1:
                 parent_class_id = entity_lookup_key_to_id[potential_parent_keys[0]]
                 edges_to_write_batch.append((parent_class_id, entity_id, "CONTAINS_METHOD", {}))
             elif len(potential_parent_keys) > 1:
                 logging.warning(f"Ambiguous parent class '{parent_name}' for method '{entity.get('name')}' in {source_file}. Skipping CONTAINS_METHOD edge.")

        # 3. CALLS Edge (Function/Method calls another Function/Method/Potential)
        if entity.get('entity_type') == 'Call':
            caller_id = None
            caller_parent_type = entity.get('parent_type')
            caller_parent_name = entity.get('parent_name')
            callee_name = entity.get('name') # Name of the function being called

            # Find the ID of the *calling* function/method
            if caller_parent_name and caller_parent_type in ['Function', 'Class']: # Call is inside a func/method
                 potential_caller_keys = [k for k in entity_lookup_key_to_id if k[0]==caller_parent_type and k[1]==caller_parent_name and k[2]==source_file]
                 if len(potential_caller_keys) == 1:
                     caller_id = entity_lookup_key_to_id[potential_caller_keys[0]]
                 elif len(potential_caller_keys) > 1 :
                      logging.warning(f"Ambiguous caller context '{caller_parent_type} {caller_parent_name}' for call to '{callee_name}' in {source_file}. Using entity ID '{entity_id}' as caller proxy.")
                      # Fallback: Relate the Call node itself to the callee? Less ideal.
                      caller_id = entity_id # Use the Call entity's ID if context is ambiguous
                 else:
                      logging.warning(f"Could not find unique caller context '{caller_parent_type} {caller_parent_name}' for call to '{callee_name}' in {source_file}. Caller ID set to Call node '{entity_id}'.")
                      caller_id = entity_id # Fallback if caller context not found
            elif file_node_id: # Call is at global scope within the file
                 caller_id = file_node_id # Relate the File node to the callee
            else: # Should not happen if file node exists
                logging.warning(f"Cannot determine caller context for call to '{callee_name}' at line {entity.get('start_line')} in {source_file}.")


            if caller_id and callee_name:
                callee_id = None
                # Try to find the callee defined within the analyzed files (same file first, then others?)
                # Simplistic: Look for function/method with that name in the *same file* first.
                potential_callee_keys = [k for k in entity_lookup_key_to_id if k[0] in ['Function'] and k[1]==callee_name and k[2]==source_file]

                if len(potential_callee_keys) == 1:
                    callee_id = entity_lookup_key_to_id[potential_callee_keys[0]]
                elif len(potential_callee_keys) > 1:
                     logging.warning(f"Ambiguous callee '{callee_name}' found in {source_file} for call from {caller_id}. Linking to PotentialFunction node instead.")
                     # Fallback to potential function node
                     potential_funcs_to_create.add(callee_name)
                     safe_callee_name = "".join(c if c.isalnum() or c in ['_','.','-'] else '_' for c in callee_name)
                     callee_id = f"PotentialFunction:{safe_callee_name}"
                else:
                    # Callee not found in the same file. Assume it's external or built-in. Create PotentialFunction node.
                    potential_funcs_to_create.add(callee_name)
                    safe_callee_name = "".join(c if c.isalnum() or c in ['_','.','-'] else '_' for c in callee_name)
                    callee_id = f"PotentialFunction:{safe_callee_name}"

                if callee_id: # Ensure we have a callee ID (either defined or potential)
                     rel_props = {'line': entity.get('start_line')} # Add line number of the call
                     edges_to_write_batch.append((caller_id, callee_id, "CALLS", rel_props))

        # 4. IMPORTS Edge (File imports Module/Items)
        if entity.get('entity_type') in ['Import', 'ImportFrom']:
             if file_node_id:
                 # Properties for the import relationship
                 rel_props = {'line': entity.get('start_line')}
                 if entity.get('entity_type') == 'Import':
                     rel_props['module'] = entity.get('module_name', entity.get('name')) # Name is the module here
                 elif entity.get('entity_type') == 'ImportFrom':
                     rel_props['module'] = entity.get('module_name', 'UnknownModule')
                     rel_props['items'] = str(entity.get('imported_items', [])) # Store imported items as string

                 edges_to_write_batch.append((file_node_id, entity_id, "IMPORTS", rel_props))


    # --- Execute Batched Writes in Transactions ---
    logging.info(f"Executing Neo4j writes: {len(nodes_to_write_batch)} nodes, {len(potential_funcs_to_create)} potential funcs, {len(edges_to_write_batch)} edges.")
    try:
        with driver.session(database="neo4j") as session: # Specify DB if not default
            # Write Nodes
            logging.info(f"  Writing {len(nodes_to_write_batch)} nodes...")
            # Use transactions for batches for better performance and atomicity
            batch_size = 500 # Adjust batch size based on memory/performance
            for i in range(0, len(nodes_to_write_batch), batch_size):
                batch = nodes_to_write_batch[i:min(i + batch_size, len(nodes_to_write_batch))]
                try:
                    session.execute_write(lambda tx: [add_neo4j_node(tx, node_data) for node_data in batch])
                    logging.info(f"    Wrote node batch {i//batch_size + 1}...")
                except Exception as batch_e:
                     logging.error(f"    Error writing node batch starting at index {i}: {batch_e}")
                     # Optionally retry or log individual failures within the batch

            # Write Potential Function Nodes (if any)
            potential_funcs_list = list(potential_funcs_to_create)
            logging.info(f"  Writing {len(potential_funcs_list)} potential function nodes...")
            for i in range(0, len(potential_funcs_list), batch_size):
                 batch = potential_funcs_list[i:min(i + batch_size, len(potential_funcs_list))]
                 try:
                     session.execute_write(lambda tx: [add_neo4j_potential_function_node(tx, func_name) for func_name in batch])
                     logging.info(f"    Wrote potential function batch {i//batch_size + 1}...")
                 except Exception as batch_e:
                     logging.error(f"    Error writing potential function batch starting at index {i}: {batch_e}")

            # Write Edges
            logging.info(f"  Writing {len(edges_to_write_batch)} edges...")
            for i in range(0, len(edges_to_write_batch), batch_size):
                 batch = edges_to_write_batch[i:min(i + batch_size, len(edges_to_write_batch))]
                 try:
                     session.execute_write(lambda tx: [add_neo4j_edge(tx, src, tgt, rel, props) for src, tgt, rel, props in batch])
                     logging.info(f"    Wrote edge batch {i//batch_size + 1}...")
                 except Exception as batch_e:
                     logging.error(f"    Error writing edge batch starting at index {i}: {batch_e}")


        logging.info("--- Neo4j graph population complete ---")
    except Exception as e:
        logging.error(f"Failed during Neo4j population transaction: {e}", exc_info=True)
        logging.error("Check Neo4j connection, permissions, and APOC installation if used.")


# 8. Store Metadata (Keep as is from Snippet 2)
def store_metadata(kv_storage: Dict, index_name: str, file_paths: List[str], num_entities: int, num_chunks: int):
    logging.info("--- 8. Storing Index Metadata ---")
    metadata = {
        "index_id": index_name,
        "timestamp": datetime.now().isoformat(), # Add timestamp
        "processed_files": file_paths,
        "entity_vector_collection": f"{index_name}_entities",
        "chunk_vector_collection": f"{index_name}_chunks",
        "chunking_strategy": "structure-aware", # Indicate chunking type
        "token_limit_per_chunk": CHUNK_TOKEN_LIMIT,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "neo4j_populated": True,
        "neo4j_uri": NEO4J_URI, # Store URI used
        "num_unique_entities_processed": num_entities,
        "num_text_chunks_processed": num_chunks,
        "language_processed": "python"
    }
    # Store in the simple kv_store dict
    kv_storage[index_name] = metadata

    # Save to a JSON file for persistence
    metadata_filename = f"{index_name}_metadata.json"
    try:
        with open(metadata_filename, 'w') as f:
            # Use default=str for any non-serializable types like datetime
            json.dump(kv_storage, f, indent=2, default=str)
        logging.info(f"Metadata saved to {metadata_filename}")
    except Exception as e:
        logging.error(f"Error saving metadata to {metadata_filename}: {e}")


# --- Main Orchestration (Modified) ---
from datetime import datetime # Needed for metadata timestamp

def main_processing_pipeline(input_file_paths: List[str], index_id: str = "python_struct_chunk_neo4j_v1", enable_llm_description: bool = False):
    """
    Main pipeline to process Python files, chunk structurally, extract entities,
    store in ChromaDB and Neo4j.

    Args:
        input_file_paths: List of paths to Python files.
        index_id: A unique name for this processing run (used for collections/metadata).
        enable_llm_description: Set True to use Ollama for entity descriptions.
    """
    logging.info(f"===== Starting PYTHON Structure-Chunk Pipeline (Neo4j/Chroma Target): {index_id} =====")
    start_time = datetime.now()

    python_files = [p for p in input_file_paths if os.path.isfile(p) and p.lower().endswith(".py")]
    if not python_files:
        logging.error("No valid .py files found in the input paths.")
        return
    logging.info(f"Found {len(python_files)} Python files to process.")

    # --- Initialize Components ---
    code_parser = CodeParser()
    if not code_parser.is_parser_loaded():
        logging.critical("Python parser failed to load. Exiting.")
        return

    if ef is None: # Check if ChromaDB embedding function initialized
        logging.critical("Embedding Function (ef) for ChromaDB failed to initialize. Exiting.")
        return

    # Initialize ChromaDB client (persistent)
    chroma_db_path = f"{CHROMA_PATH_PREFIX}{index_id}"
    logging.info(f"Initializing ChromaDB client at path: {chroma_db_path}")
    try:
        # Ensure the directory exists if needed by PersistentClient
        os.makedirs(os.path.dirname(chroma_db_path), exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        # Simple test to confirm client works
        chroma_client.heartbeat()
        logging.info("ChromaDB client initialized successfully.")
    except Exception as e:
         logging.critical(f"Failed to initialize ChromaDB client at {chroma_db_path}: {e}", exc_info=True)
         return

    # Initialize Neo4j Driver
    neo4j_driver = None
    try:
        logging.info(f"Connecting to Neo4j at {NEO4J_URI}...")
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        logging.info("Neo4j connection successful.")

        # Ensure KGNode ID uniqueness constraint exists (idempotent)
        with neo4j_driver.session(database="neo4j") as session:
           logging.info("Ensuring Neo4j :KGNode(id) uniqueness constraint exists...")
           # Use CREATE CONSTRAINT IF NOT EXISTS for idempotency
           session.run("CREATE CONSTRAINT unique_kgnode_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE")
           logging.info("Neo4j uniqueness constraint checked/created.")

    except Exception as e:
        logging.critical(f"Failed to connect to Neo4j or verify constraint: {e}", exc_info=True)
        logging.critical("Please check Neo4j connection details (URI, USER, PASSWORD), ensure the database is running, and check user permissions.")
        if neo4j_driver: neo4j_driver.close() # Close driver if partially opened
        return

    # --- Execute Pipeline Steps ---
    try:
        # 1. Parse and Extract Detailed Entities
        all_extracted_entities = parse_and_extract(python_files, code_parser)

        # 2. Deduplicate Entities
        unique_entities = deduplicate_entities(all_extracted_entities)

        # 3. Update Descriptions (Optional LLM)
        if enable_llm_description:
            unique_entities = update_entity_descriptions(unique_entities, LLM_CLIENT)
        else:
            logging.info("--- Skipping LLM description update (disabled by flag) ---")
             # Ensure default description if LLM is skipped
            for item in unique_entities:
                if 'description' not in item:
                    item['description'] = f"Default description for {item.get('entity_type')} '{item.get('name')}'."

        # 4. Chunk Files (Structure-Aware)
        text_chunks = chunk_files(python_files, code_parser, CHUNK_TOKEN_LIMIT)

        # 5. Embed and Store Chunks in ChromaDB
        # Use ef directly as the embedding_function for ChromaDB
        embed_and_store_chunks(text_chunks, ef, chroma_client, f"{index_id}_chunks")

        # 6. Embed and Store Entities in ChromaDB
        # Use ef directly as the embedding_function for ChromaDB
        embed_and_store_entities(unique_entities, ef, chroma_client, f"{index_id}_entities")

        # 7. Populate Neo4j Graph
        populate_neo4j_graph(neo4j_driver, unique_entities, all_extracted_entities, python_files)

        # 8. Store Metadata
        store_metadata(kv_store, index_id, python_files, len(unique_entities), len(text_chunks))

    except Exception as e:
         logging.critical(f"Pipeline execution failed: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        if neo4j_driver:
            logging.info("Closing Neo4j connection.")
            neo4j_driver.close()
        # ChromaDB client doesn't typically require explicit closing for PersistentClient

    end_time = datetime.now()
    logging.info(f"===== Pipeline Complete: {index_id} =====")
    logging.info(f"Total execution time: {end_time - start_time}")


# --- Example Usage ---
if __name__ == "__main__":
    # (Dummy file creation remains the same)
    if not os.path.exists("temp_code"): os.makedirs("temp_code")
    with open("temp_code/processor.py", "w", encoding='utf-8') as f: f.write("""
import os
import json
import logging # Added import
    def t(s):
       print(s)
    # Constructor
    def __init__(self, source_path: str):
        '''Initializes the processor with a source path.'''
        self.source = source_path
        self.data = None # Instance attribute
        t(source_path) # Built-in call


""")

    input_files = ["temp_code/processor.py"]
    index_name = "py_structchunk_neo4j_chroma_v2" # More descriptive index name

    # --- Configuration ---
    # Set True to try using Ollama (make sure it's running and model is pulled!)
    USE_LLM_DESCRIPTIONS = False
    # Set Neo4j credentials via environment variables or modify defaults in Config section above.
    # --- --- --- --- --- ---

    # Check for Neo4j password default
    if NEO4J_PASSWORD == "password":
         logging.warning("Using default Neo4j password 'password'. Set the NEO4J_PASSWORD environment variable for security.")

    main_processing_pipeline(
        input_files,
        index_id=index_name,
        enable_llm_description=USE_LLM_DESCRIPTIONS
    )
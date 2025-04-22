import os
import logging
from typing import List, Dict, Any, Tuple, Set, Union
import tiktoken
import uuid

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
CHUNK_TOKEN_LIMIT = 450
CHUNK_ENCODING = "cl100k_base"
EMBEDDING_ENCODING = "cl100k_base"
CHROMA_PATH_PREFIX = "./chroma_db_struct_chunk_"
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "abcd12345")

# --- Python Built-ins ---
PYTHON_BUILTINS = frozenset([
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes',
    'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir',
    'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset',
    'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int',
    'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
    'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print',
    'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice',
    'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip',
    '__import__',
    'Exception', 'TypeError', 'ValueError', 'NameError', 'IndexError', 'KeyError',
    'AttributeError', 'ImportError', 'FileNotFoundError', 'ZeroDivisionError',
    'True', 'False', 'None', 'NotImplemented', 'Ellipsis', '__debug__'
])

# --- Helper Functions ---
def generate_unique_id(prefix: str = "") -> str:
    safe_prefix = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in prefix)
    return f"{safe_prefix}{uuid.uuid4()}"

def count_tokens(text: str, encoding_name: str = EMBEDDING_ENCODING) -> int:
    if not text: return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text, disallowed_special=())
        return len(tokens)
    except tiktoken.registry.RegistryError:
        logging.warning(f"Tiktoken encoding '{encoding_name}' not found. Falling back to tiktoken default 'cl100k_base'.")
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text, disallowed_special=()))
        except Exception as e_fallback:
            logging.warning(f"Fallback token counting failed: {e_fallback}. Using character count / 4.")
            return len(text) // 4
    except Exception as e:
        logging.warning(f"Token counting failed with '{encoding_name}': {e}. Falling back to char count / 4.")
        return len(text) // 4
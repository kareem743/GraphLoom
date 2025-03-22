import os
from typing import List, Dict, Union, Tuple
from tree_sitter import Node
from tree_sitter_languages import get_language, get_parser
import logging
import tiktoken
import networkx as nx

# Token counting function
def count_tokens(text: str, encoding_name: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    return len(encoding.encode(text))

class CodeParser:
    CACHE_DIR = os.path.expanduser("~/.code_parser_cache")

    def __init__(self, file_extensions: Union[None, List[str], str] = None):
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        self.language_extension_map = {
            "py": "python",
            "js": "javascript",
            "jsx": "javascript",
            "css": "css",
            "ts": "typescript",
            "tsx": "typescript",
            "php": "php",
            "rb": "ruby"
        }
        if file_extensions is None:
            self.language_names = []
        else:
            self.language_names = [self.language_extension_map.get(ext) for ext in file_extensions if
                                   ext in self.language_extension_map]
        self.languages = {}
        self._load_parsers()

    def _load_parsers(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        for language in self.language_names:
            try:
                self.languages[language] = get_language(language)
                logging.info(f"Successfully loaded {language} parser")
            except Exception as e:
                logging.error(f"Failed to load language {language}. Error: {str(e)}")

    def parse_code(self, code: str, file_extension: str) -> Union[None, Node]:
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            print(f"Unsupported file type: {file_extension}")
            return None
        language = self.languages.get(language_name)
        if language is None:
            print("Language parser not found")
            return None
        parser = get_parser(language_name)
        tree = parser.parse(bytes(code, "utf8"))
        if tree is None:
            print("Failed to parse the code")
            return None
        return tree.root_node

    def extract_points_of_interest(self, node: Node, file_extension: str) -> List[Tuple[Node, str]]:
        node_types_of_interest = self._get_node_types_of_interest(file_extension)
        points_of_interest = []
        if node.type in node_types_of_interest.keys():
            points_of_interest.append((node, node_types_of_interest[node.type]))
        for child in node.children:
            points_of_interest.extend(self.extract_points_of_interest(child, file_extension))
        return points_of_interest

    def _get_node_types_of_interest(self, file_extension: str) -> Dict[str, str]:
        node_types = {
            'py': {
                'import_statement': 'Import',
                'export_statement': 'Export',
                'class_definition': 'Class',
                'function_definition': 'Function',
                'call': 'Call',
            }
        }
        if file_extension in node_types.keys():
            return node_types[file_extension]
        else:
            raise ValueError("Unsupported file type")

    def get_lines_for_points_of_interest(self, code: str, file_extension: str) -> List[int]:
        root_node = self.parse_code(code, file_extension)
        points_of_interest = self.extract_points_of_interest(root_node, file_extension)
        return [node.start_point[0] for node, _ in points_of_interest]

    def extract_comments(self, node: Node, file_extension: str) -> List[Tuple[Node, str]]:
        node_types_of_interest = self._get_nodes_for_comments(file_extension)
        comments = []
        if node.type in node_types_of_interest:
            comments.append((node, node_types_of_interest[node.type]))
        for child in node.children:
            comments.extend(self.extract_comments(child, file_extension))
        return comments

    def _get_nodes_for_comments(self, file_extension: str) -> Dict[str, str]:
        node_types = {
            'py': {
                'comment': 'Comment',
                'decorator': 'Decorator',
            }
        }
        if file_extension in node_types.keys():
            return node_types[file_extension]
        else:
            raise ValueError("Unsupported file type")

    def get_lines_for_comments(self, code: str, file_extension: str) -> List[int]:
        root_node = self.parse_code(code, file_extension)
        comments = self.extract_comments(root_node, file_extension)
        return [node.start_point[0] for node, _ in comments]

    def extract_entity_details(self, code: str, file_extension: str) -> List[Dict]:
        """Extract detailed entity information from code with snippets."""
        root_node = self.parse_code(code, file_extension)
        if not root_node:
            return []
        
        entities = []
        points = self.extract_points_of_interest(root_node, file_extension)
        lines = code.split("\n")
        
        for node, entity_type in points:
            entity_name = self._get_entity_name(node, code)
            start_line = node.start_point[0]
            end_line = node.end_point[0] + 1  # Include the last line of the node
            snippet = "\n".join(lines[start_line:end_line]).strip()
            
            if entity_type == "Call":
                called_function = entity_name
                parent_function = self._find_parent_function(node, code)
                if parent_function:
                    entities.append({
                        "type": "Call",
                        "name": called_function,
                        "parent": parent_function,
                        "line": start_line,
                        "snippet": snippet
                    })
            else:
                entities.append({
                    "type": entity_type,
                    "name": entity_name,
                    "line": start_line,
                    "snippet": snippet
                })
        return entities

    def _get_entity_name(self, node: Node, code: str) -> str:
        if node.type in ["function_definition", "class_definition"]:
            for child in node.children:
                if child.type == "identifier":
                    start, end = child.start_byte, child.end_byte
                    return code[start:end]
        elif node.type == "call":
            for child in node.children:
                if child.type == "identifier":
                    start, end = child.start_byte, child.end_byte
                    return code[start:end]
        elif node.type == "import_statement":
            start, end = node.start_byte, node.end_byte
            return code[start:end].strip()
        return "Unnamed"

    def _find_parent_function(self, node: Node, code: str) -> str:
        current = node
        while current.parent:
            current = current.parent
            if current.type == "function_definition":
                for child in current.children:
                    if child.type == "identifier":
                        start, end = child.start_byte, child.end_byte
                        return code[start:end]
        return None

from abc import ABC, abstractmethod

class Chunker(ABC):
    def __init__(self, encoding_name="gpt-4"):
        self.encoding_name = encoding_name

    @abstractmethod
    def chunk(self, content, token_limit):
        pass

    @abstractmethod
    def get_chunk(self, chunked_content, chunk_number):
        pass

    @staticmethod
    def print_chunks(chunks):
        for chunk_number, chunk_code in chunks.items():
            print(f"Chunk {chunk_number}:")
            print("=" * 40)
            print(chunk_code.strip())
            print("=" * 40)

class CodeChunker(Chunker):
    def __init__(self, file_extension, encoding_name="gpt-4"):
        super().__init__(encoding_name)
        self.file_extension = file_extension

    def chunk(self, code: str, token_limit: int) -> dict:
        code_parser = CodeParser(self.file_extension)
        chunks = {}
        current_chunk = ""
        token_count = 0
        lines = code.split("\n")
        i = 0
        chunk_number = 1
        start_line = 0
        breakpoints = sorted(code_parser.get_lines_for_points_of_interest(code, self.file_extension))
        comments = sorted(code_parser.get_lines_for_comments(code, self.file_extension))
        adjusted_breakpoints = []
        for bp in breakpoints:
            current_line = bp - 1
            highest_comment_line = None
            while current_line in comments:
                highest_comment_line = current_line
                current_line -= 1
            if highest_comment_line:
                adjusted_breakpoints.append(highest_comment_line)
            else:
                adjusted_breakpoints.append(bp)
        breakpoints = sorted(set(adjusted_breakpoints))

        while i < len(lines):
            line = lines[i]
            new_token_count = count_tokens(line, self.encoding_name)
            if token_count + new_token_count > token_limit:
                if i in breakpoints:
                    stop_line = i
                else:
                    stop_line = max(max([x for x in breakpoints if x < i], default=start_line), start_line)
                if stop_line == start_line and i not in breakpoints:
                    token_count += new_token_count
                    i += 1
                elif stop_line == start_line and i == stop_line:
                    token_count += new_token_count
                    i += 1
                elif stop_line == start_line and i in breakpoints:
                    current_chunk = "\n".join(lines[start_line:stop_line])
                    if current_chunk.strip():
                        chunks[chunk_number] = current_chunk
                        chunk_number += 1
                    token_count = 0
                    start_line = i
                    i += 1
                else:
                    current_chunk = "\n".join(lines[start_line:stop_line])
                    if current_chunk.strip():
                        chunks[chunk_number] = current_chunk
                        chunk_number += 1
                    i = stop_line
                    token_count = 0
                    start_line = stop_line
            else:
                token_count += new_token_count
                i += 1
        current_chunk_code = "\n".join(lines[start_line:])
        if current_chunk_code.strip():
            chunks[chunk_number] = current_chunk_code
        return chunks

    def get_chunk(self, chunked_codebase, chunk_number):
        return chunked_codebase[chunk_number]

def build_knowledge_graph(chunks: Dict[int, str], file_extension: str) -> nx.DiGraph:
    """Build a knowledge graph for RAG system."""
    G = nx.DiGraph()
    parser = CodeParser(file_extension)

    for chunk_id, chunk_code in chunks.items():
        G.add_node(f"Chunk_{chunk_id}", type="Chunk", chunk=chunk_id, snippet=chunk_code)

        entities = parser.extract_entity_details(chunk_code, file_extension)
        
        for entity in entities:
            entity_name = entity["name"]
            entity_type = entity["type"]
            node_id = f"{entity_type}_{entity_name}_{chunk_id}"

            # Add entity node with snippet for RAG context
            G.add_node(node_id, type=entity_type, name=entity_name, chunk=chunk_id, line=entity["line"], snippet=entity["snippet"])

            # Add relationships
            if entity_type == "Call":
                parent_function = entity["parent"]
                if parent_function:
                    parent_id = f"Function_{parent_function}_{chunk_id}"
                    G.add_edge(parent_id, node_id, relationship="calls")
            elif entity_type in ["Function", "Class"]:
                G.add_edge(f"Chunk_{chunk_id}", node_id, relationship="contains")
            elif entity_type == "Import":
                G.add_edge(f"Chunk_{chunk_id}", node_id, relationship="imports")

    return G

def query_graph(graph: nx.DiGraph, query: str) -> List[Dict]:
    """Simple query function to retrieve relevant nodes for RAG."""
    results = []
    query_lower = query.lower()
    
    for node, attrs in graph.nodes(data=True):
        node_name = attrs.get("name", "").lower()
        node_type = attrs.get("type", "")
        snippet = attrs.get("snippet", "")
        
        # Basic keyword matching (can be enhanced with embeddings)
        if query_lower in node_name or query_lower in snippet.lower():
            results.append({
                "node": node,
                "type": node_type,
                "name": attrs.get("name"),
                "snippet": snippet,
                "relationships": list(graph.edges(node, data=True))
            })
    
    return results

def chunk_and_graph_for_rag(file_path: str, token_limit: int = 50):
    if not os.path.exists(file_path) or not file_path.endswith(".py"):
        print(f"Error: File '{file_path}' does not exist or is not a .py file.")
        return
    
    with open(file_path, "r", encoding="utf-8") as file:
        code = file.read()
    
    chunker = CodeChunker(file_extension="py", encoding_name="gpt-4")
    chunks = chunker.chunk(code, token_limit)
    Chunker.print_chunks(chunks)
    
    # Build the knowledge graph
    graph = build_knowledge_graph(chunks, "py")
    
    # Debugging: Print node attributes
    print("\nGraph Nodes and Attributes:")
    for node, attrs in graph.nodes(data=True):
        print(f"Node: {node}, Attributes: {attrs}")
    
    # Export the graph to GraphML
    nx.write_graphml(graph, "knowledge_graph_for_rag.graphml")
    print("\nKnowledge graph exported to 'knowledge_graph_for_rag.graphml'")
    
    # Example query for RAG
    query = "greet"
    print(f"\nQuerying graph for: '{query}'")
    results = query_graph(graph, query)
    for result in results:
        print(f"Found: {result['node']}, Type: {result['type']}, Name: {result['name']}, Snippet: {result['snippet']}")

if __name__ == "__main__":
    file_path = "example.py"
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("""
# Sample Python code
import os

def greet(name):
    print(f"Hello, {name}!")

# Another comment
class MyClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

def another_function():
    x = 1 + 2
    return x
""")
    chunk_and_graph_for_rag(file_path, token_limit=50)
# Understanding the `generate_ast_to_kg.py` Script

This document explains how the `generate_ast_to_kg.py` script works to transform Python code from a GitHub repository into a Knowledge Graph (KG) in Neo4j.

## Full Code for Reference

```python
import ast
import sys
import os
import glob
import tempfile
from git import Repo
from neo4j import GraphDatabase

def generate_ast(code_string):
    try:
        ast_tree = ast.parse(code_string)
        return ast_tree
    except SyntaxError as e:
        print(f"Syntax error: {e}", file=sys.stderr)
        return None

def generate_asts_from_dir(directory):
    for py_file in glob.glob(os.path.join(directory, '**', '*.py'), recursive=True):
        relative_path = os.path.relpath(py_file, directory)
        with open(py_file, 'r') as f:
            code = f.read()
        ast_tree = generate_ast(code)
        if ast_tree is not None:
            yield (relative_path, ast_tree)

class ASTGraphBuilder(ast.NodeVisitor):
    def __init__(self, driver, file_path):
        self.driver = driver
        self.file_path = file_path
        self.current_parent_id = None
        self.node_counter = 0

    def generic_visit(self, node):
        node_id = self.node_counter
        self.node_counter += 1
        
        with self.driver.session() as session:
            session.run("CREATE (n:ASTNode {id: $id, type: $type})", id=node_id, type=type(node).__name__)
            if self.current_parent_id is None:
                session.run(
                    "MATCH (f:File {path: $path}) CREATE (f)-[:CONTAINS]->(n:ASTNode {id: $id})",
                    path=self.file_path, id=node_id
                )
            else:
                session.run(
                    "MATCH (parent:ASTNode {id: $parent_id}) MATCH (child:ASTNode {id: $child_id}) CREATE (parent)-[:CHILD_OF]->(child)",
                    parent_id=self.current_parent_id, child_id=node_id
                )
        
        old_parent_id = self.current_parent_id
        self.current_parent_id = node_id
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
        self.current_parent_id = old_parent_id

def init_graph(driver, file_path):
    with driver.session() as session:
        session.run("CREATE (f:File {path: $path})", path=file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <repo_url>")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning repository to {temp_dir}")
        Repo.clone_from(repo_url, temp_dir)
        for relative_path, ast_tree in generate_asts_from_dir(temp_dir):
            print(f"Processing {relative_path}...")
            init_graph(driver, relative_path)
            builder = ASTGraphBuilder(driver, relative_path)
            builder.visit(ast_tree)
    
    driver.close()
```

## Explanation of Each Section

### 1. Imports

```python
import ast
import sys
import os
import glob
import tempfile
from git import Repo
from neo4j import GraphDatabase
```

- **What It Does**: Brings in necessary Python libraries and modules.
- **Details**:
  - `ast`: Provides tools to parse Python code into an Abstract Syntax Tree (AST), a structured representation of code.
  - `sys`: Allows access to command-line arguments (e.g., the repository URL) and exiting the script with an error code.
  - `os`: Handles file system operations, like joining paths.
  - `glob`: Finds files matching a pattern (e.g., all `.py` files recursively).
  - `tempfile`: Creates a temporary directory for cloning the repository, which is automatically cleaned up.
  - `Repo` (from `git`): Clones a GitHub repository using GitPython.
  - `GraphDatabase` (from `neo4j`): Connects to Neo4j and manages database interactions.

### 2. Function: `generate_ast`

```python
def generate_ast(code_string):
    try:
        ast_tree = ast.parse(code_string)
        return ast_tree
    except SyntaxError as e:
        print(f"Syntax error: {e}", file=sys.stderr)
        return None
```

- **What It Does**: Takes a string of Python code and turns it into an AST.
- **Details**:
  - Input: `code_string`, the contents of a Python file as a string.
  - `ast.parse`: Converts the string into an `ast.AST` object, a tree where nodes represent code elements (e.g., functions, loops).
  - Error Handling: If the code has a syntax error (e.g., `def foo():` missing a colon), it prints the error to `stderr` and returns `None`.
- **Purpose**: This is the first step in analyzing the code's structure.

### 3. Function: `generate_asts_from_dir`

```python
def generate_asts_from_dir(directory):
    for py_file in glob.glob(os.path.join(directory, '**', '*.py'), recursive=True):
        relative_path = os.path.relpath(py_file, directory)
        with open(py_file, 'r') as f:
            code = f.read()
        ast_tree = generate_ast(code)
        if ast_tree is not None:
            yield (relative_path, ast_tree)
```

- **What It Does**: Finds all `.py` files in a directory and generates their ASTs one by one.
- **Details**:
  - `glob.glob(os.path.join(directory, '**', '*.py'), recursive=True)`: Searches recursively for all `.py` files in the given directory.
  - `os.path.relpath`: Converts absolute paths (e.g., `/tmp/repo/sorting/bubble_sort.py`) to relative paths (e.g., `sorting/bubble_sort.py`).
  - Reads each file's content with `open` and passes it to `generate_ast`.
  - `yield`: Returns a tuple `(relative_path, ast_tree)` for each valid AST, making it a generator to process files lazily (saves memory).
- **Purpose**: Prepares each file's AST for transformation into the KG.

### 4. Class: `ASTGraphBuilder`

```python
class ASTGraphBuilder(ast.NodeVisitor):
    def __init__(self, driver, file_path):
        self.driver = driver
        self.file_path = file_path
        self.current_parent_id = None
        self.node_counter = 0

    def generic_visit(self, node):
        node_id = self.node_counter
        self.node_counter += 1
        
        with self.driver.session() as session:
            session.run("CREATE (n:ASTNode {id: $id, type: $type})", id=node_id, type=type(node).__name__)
            if self.current_parent_id is None:
                session.run(
                    "MATCH (f:File {path: $path}) CREATE (f)-[:CONTAINS]->(n:ASTNode {id: $id})",
                    path=self.file_path, id=node_id
                )
            else:
                session.run(
                    "MATCH (parent:ASTNode {id: $parent_id}) MATCH (child:ASTNode {id: $child_id}) CREATE (parent)-[:CHILD_OF]->(child)",
                    parent_id=self.current_parent_id, child_id=node_id
                )
        
        old_parent_id = self.current_parent_id
        self.current_parent_id = node_id
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
        self.current_parent_id = old_parent_id
```

- **What It Does**: Walks through an AST and builds a corresponding graph in Neo4j.
- **Details**:
  - **Inheritance**: Subclasses `ast.NodeVisitor`, which provides a framework for visiting AST nodes.
  - **`__init__`**:
    - `self.driver`: The Neo4j connection object.
    - `self.file_path`: The path of the current file (e.g., `sorting/bubble_sort.py`).
    - `self.current_parent_id`: Tracks the parent node's ID during traversal (starts as `None` for the root).
    - `self.node_counter`: Assigns unique IDs to each node.
  - **`generic_visit`**:
    - Assigns a unique `node_id` to the current AST node.
    - Creates an `:ASTNode` in Neo4j with properties `id` and `type` (e.g., `FunctionDef`).
    - If it's the root node (`current_parent_id` is `None`), links it to the `:File` node with a `CONTAINS` relationship.
    - Otherwise, links it to its parent with a `CHILD_OF` relationship.
    - Saves the current parent ID, updates it to the new node, visits all child nodes (using `ast.iter_fields`), and restores the parent ID afterward.
- **Purpose**: Turns the hierarchical AST into a graph structure with nodes and relationships.

### 5. Function: `init_graph`

```python
def init_graph(driver, file_path):
    with driver.session() as session:
        session.run("CREATE (f:File {path: $path})", path=file_path)
```

- **What It Does**: Creates a `:File` node in Neo4j for each Python file.
- **Details**:
  - Uses a Neo4j session to execute a Cypher query.
  - Creates a node labeled `:File` with a `path` property (e.g., `sorting/bubble_sort.py`).
- **Purpose**: Sets up the starting point for each file's AST in the KG.

### 6. Main Execution Block

```python
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <repo_url>")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning repository to {temp_dir}")
        Repo.clone_from(repo_url, temp_dir)
        for relative_path, ast_tree in generate_asts_from_dir(temp_dir):
            print(f"Processing {relative_path}...")
            init_graph(driver, relative_path)
            builder = ASTGraphBuilder(driver, relative_path)
            builder.visit(ast_tree)
    
    driver.close()
```

- **What It Does**: Runs the entire process from cloning to storing the KG.
- **Details**:
  - **Argument Check**: Ensures exactly one command-line argument (the repository URL) is provided; otherwise, it prints usage and exits.
  - `repo_url = sys.argv[1]`: Gets the URL (e.g., `https://github.com/TheAlgorithms/Python.git`).
  - `driver = GraphDatabase.driver(...)`: Connects to Neo4j at `bolt://localhost:7687` with username `neo4j` and password `testpassword123`.
  - `tempfile.TemporaryDirectory`: Creates a temporary directory that's deleted when done.
  - `Repo.clone_from`: Clones the repository into the temp directory.
  - Loops over each file's path and AST from `generate_asts_from_dir`:
    - Prints a progress message.
    - Calls `init_graph` to create a `:File` node.
    - Uses `ASTGraphBuilder` to build the AST graph in Neo4j.
  - `driver.close()`: Closes the Neo4j connection.
- **Purpose**: Ties everything together into a single executable workflow.

## Overall Flow

1. **Input**: A GitHub repository URL (e.g., `https://github.com/TheAlgorithms/Python.git`).
2. **Cloning**: Downloads the repository to a temporary directory.
3. **File Processing**: Finds all `.py` files and generates their ASTs.
4. **Graph Building**: For each file:
   - Creates a `:File` node.
   - Traverses its AST, creating `:ASTNode` nodes and linking them with `CONTAINS` (file to root) and `CHILD_OF` (parent to child) relationships.
5. **Output**: A Knowledge Graph in Neo4j representing the repository's code structure.

## Key Points

- **AST**: The script uses Python's `ast` module to parse code into a tree, which is then mirrored as a graph in Neo4j.
- **Neo4j**: Stores the KG with nodes (`:File`, `:ASTNode`) and relationships (`CONTAINS`, `CHILD_OF`).
- **Efficiency**: Uses generators (`generate_asts_from_dir`) and temporary directories to handle large repositories without excessive memory use.
- **Error Handling**: Skips files with syntax errors, ensuring the script doesn't crash on invalid code.

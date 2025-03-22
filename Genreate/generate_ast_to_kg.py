import ast
import sys
import os
import glob
import tempfile
from git import Repo
from neo4j import GraphDatabase

def generate_ast(code_string):
    """
    Generates the Abstract Syntax Tree (AST) for the given code string.

    Args:
        code_string (str): A string containing Python code.

    Returns:
        ast.AST or None: The AST object if parsing is successful, otherwise None.
    """
    try:
        ast_tree = ast.parse(code_string)
        return ast_tree
    except SyntaxError as e:
        print(f"Syntax error: {e}", file=sys.stderr)
        return None

def generate_asts_from_dir(directory):
    """
    A generator that yields (relative_path, ast_tree) for each .py file in the directory.

    Args:
        directory (str): The directory to search for .py files.

    Yields:
        tuple: (relative_path, ast_tree) where relative_path is the path relative to the directory,
               and ast_tree is the AST object.
    """
    for py_file in glob.glob(os.path.join(directory, '**', '*.py'), recursive=True):
        relative_path = os.path.relpath(py_file, directory)
        with open(py_file, 'r') as f:
            try:
                code = f.read()
            except:
                print(f"Error with file {relative_path}")
        ast_tree = generate_ast(code)
        if ast_tree is not None:
            yield (relative_path, ast_tree)

class ASTGraphBuilder(ast.NodeVisitor):
    """
    Traverses an AST and builds a Knowledge Graph in Neo4j with meaningful node labels.
    """
    def __init__(self, driver, file_path):
        self.driver = driver
        self.file_path = file_path
        self.node_counter = 0

    def visit(self, node):
        node_id = self.node_counter
        self.node_counter += 1

        with self.driver.session() as session:
            # Create the AST node with its type
            session.run(
                "CREATE (n:ASTNode {id: $id, type: $type})",
                id=node_id,
                type=type(node).__name__
            )

            # Set properties for non-AST fields (e.g., name, id, value)
            for field, value in ast.iter_fields(node):
                if not isinstance(value, (ast.AST, list)):
                    session.run(
                        f"MATCH (n:ASTNode {{id: $id}}) SET n.{field} = $value",
                        id=node_id,
                        value=value
                    )

            # Set a meaningful label based on node type
            if type(node).__name__ == "FunctionDef":
                label = node.name
            elif type(node).__name__ == "ClassDef":
                label = node.name
            elif type(node).__name__ == "Name":
                label = node.id
            elif type(node).__name__ == "Constant":
                label = str(node.value)
            else:
                label = type(node).__name__  # Fallback to node type

            session.run(
                "MATCH (n:ASTNode {id: $id}) SET n.label = $label",
                id=node_id,
                label=label
            )

            # Link the root node to the File node
            if self.node_counter == 1:  # First node is the root
                session.run(
                    """
                    MATCH (f:File {path: $path})
                    CREATE (f)-[:CONTAINS]->(n:ASTNode {id: $id})
                    """,
                    path=self.file_path,
                    id=node_id
                )

        # Visit children and create relationships
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        child_id = self.visit(item)
                        with self.driver.session() as session:
                            session.run(
                                f"""
                                MATCH (parent:ASTNode {{id: $parent_id}})
                                MATCH (child:ASTNode {{id: $child_id}})
                                CREATE (parent)-[:HAS_{field.upper()} {{order: $order}}]->(child)
                                """,
                                parent_id=node_id,
                                child_id=child_id,
                                order=index
                            )
            elif isinstance(value, ast.AST):
                child_id = self.visit(value)
                with self.driver.session() as session:
                    session.run(
                        f"""
                        MATCH (parent:ASTNode {{id: $parent_id}})
                        MATCH (child:ASTNode {{id: $child_id}})
                        CREATE (parent)-[:HAS_{field.upper()}]->(child)
                        """,
                        parent_id=node_id,
                        child_id=child_id
                    )

        return node_id

def init_graph(driver, file_path):
    """
    Initializes a File node in the graph.
    """
    with driver.session() as session:
        session.run(
            "CREATE (f:File {path: $path})",
            path=file_path
        )

if __name__ == "__main__":
    # Check for correct command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <repo_url>")
        sys.exit(1)

    # Get the repository URL from command-line argument
    repo_url = sys.argv[1]

    # Setup Neo4j connection
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))

    # Use a temporary directory that gets cleaned up automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning repository to {temp_dir}")
        Repo.clone_from(repo_url, temp_dir)

        print(generate_asts_from_dir)

        # Process each Python file
        for relative_path, ast_tree in generate_asts_from_dir(temp_dir):
            print(f"Processing {relative_path}...")

            print("AST for the file:")
            print(ast.dump(ast_tree, indent=4))
            
            init_graph(driver, relative_path)
            builder = ASTGraphBuilder(driver, relative_path)
            print(relative_path,ast_tree)
            builder.visit(ast_tree)

    driver.close()
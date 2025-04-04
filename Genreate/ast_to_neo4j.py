import ast
import os
import sys
import tempfile
import argparse
import logging
from git import Repo, GitCommandError
import networkx as nx
from neo4j import GraphDatabase
import neo4j.exceptions

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- AST Visitor to Build NetworkX Graph ---
class KGBuilderVisitor(ast.NodeVisitor):
    """
    Traverses the Python AST and builds a NetworkX DiGraph representing
    files, classes, functions, calls, and imports.
    """
    def __init__(self, filepath, graph):
        self.filepath = filepath # Store the path of the current file
        self.graph = graph       # The networkx graph to build
        self.scope_stack = []    # Track current scope (module, class, function)

        # Add the File node itself
        self.file_node_id = f"File:{self.filepath}"
        self.graph.add_node(self.file_node_id, type='File', name=os.path.basename(filepath), filepath=filepath)
        self.scope_stack.append(self.file_node_id) # Start scope at the file level

    def _get_current_scope_id(self):
        return self.scope_stack[-1] if self.scope_stack else None

    def _generate_node_id(self, node_type, name, lineno):
        # Create a relatively unique ID based on file, type, name, and line
        # Adjust this strategy if more uniqueness is needed (e.g., UUIDs)
        return f"{node_type}:{self.filepath}:{name}:{lineno}"

    def visit_Import(self, node):
        lineno = getattr(node, 'lineno', -1)
        for alias in node.names:
            import_name = alias.name
            asname = alias.asname or import_name # Use alias if present
            node_id = self._generate_node_id("Import", import_name, lineno)
            self.graph.add_node(node_id, type='Import', name=import_name, alias=asname, lineno=lineno, filepath=self.filepath)
            # Link import to the current scope (usually the file)
            parent_scope_id = self._get_current_scope_id()
            if parent_scope_id:
                self.graph.add_edge(parent_scope_id, node_id, type='IMPORTS')
        self.generic_visit(node) # Continue visiting children if any

    def visit_ImportFrom(self, node):
        lineno = getattr(node, 'lineno', -1)
        module_name = node.module or '.' # Handle relative imports '.'
        for alias in node.names:
            import_name = alias.name
            asname = alias.asname or import_name
            node_id = self._generate_node_id("ImportFrom", f"{module_name}.{import_name}", lineno)
            self.graph.add_node(node_id, type='ImportFrom', name=import_name, module=module_name, alias=asname, lineno=lineno, filepath=self.filepath)
            # Link import to the current scope (usually the file)
            parent_scope_id = self._get_current_scope_id()
            if parent_scope_id:
                self.graph.add_edge(parent_scope_id, node_id, type='IMPORTS_FROM')
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        lineno = getattr(node, 'lineno', -1)
        class_name = node.name
        node_id = self._generate_node_id("Class", class_name, lineno)
        docstring = ast.get_docstring(node)
        self.graph.add_node(node_id, type='Class', name=class_name, lineno=lineno, docstring=docstring, filepath=self.filepath)

        # Link class to its defining scope (file or another class/function)
        parent_scope_id = self._get_current_scope_id()
        if parent_scope_id:
            self.graph.add_edge(parent_scope_id, node_id, type='CONTAINS')

        # Add inheritance relationships (basic version)
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id
                # Create a potential node for the base class (might not be in this file)
                # Note: Linking accurately requires resolving imports/scope, which is complex.
                # We'll just add an edge mentioning the name for now.
                self.graph.add_edge(node_id, f"PotentialClass:{base_name}", type='INHERITS_FROM', base_class_name=base_name)
            # Add handling for ast.Attribute if needed (e.g., inherits from module.Class)

        # Manage scope
        self.scope_stack.append(node_id)
        self.generic_visit(node) # Visit methods, nested classes etc.
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        lineno = getattr(node, 'lineno', -1)
        func_name = node.name
        node_id = self._generate_node_id("Function", func_name, lineno)
        docstring = ast.get_docstring(node)
        self.graph.add_node(node_id, type='Function', name=func_name, lineno=lineno, docstring=docstring, filepath=self.filepath)

        # Link function to its defining scope (file or class)
        parent_scope_id = self._get_current_scope_id()
        if parent_scope_id:
            self.graph.add_edge(parent_scope_id, node_id, type='CONTAINS')

        # Manage scope
        self.scope_stack.append(node_id)
        self.generic_visit(node) # Visit function body
        self.scope_stack.pop()

    def visit_Call(self, node):
        lineno = getattr(node, 'lineno', -1)
        func_node = node.func
        call_name = ""
        if isinstance(func_node, ast.Name): # Simple function call like my_func()
            call_name = func_node.id
        elif isinstance(func_node, ast.Attribute): # Method call like obj.method() or module.func()
            # Try to reconstruct the full call name (e.g., "os.path.join") - basic version
            parts = []
            curr = func_node
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
                call_name = ".".join(reversed(parts))
            else: # Could be a call on result of another call, etc.
                call_name = "complex_call" # Placeholder
        else: # Lambda calls, etc.
            call_name = "complex_call" # Placeholder

        node_id = self._generate_node_id("Call", call_name, lineno)
        self.graph.add_node(node_id, type='Call', name=call_name, lineno=lineno, filepath=self.filepath)

        # Link call to the scope where it occurs (function/method body, class body, module level)
        parent_scope_id = self._get_current_scope_id()
        if parent_scope_id:
             # Link scope containing the call TO the call itself
            self.graph.add_edge(parent_scope_id, node_id, type='CONTAINS_CALL')
            # Add edge FROM the call TO the potential target function name (simple matching)
            # Accurate call target resolution is very complex.
            self.graph.add_edge(node_id, f"PotentialFunction:{call_name}", type='CALLS_FUNCTION', target_name=call_name)

        self.generic_visit(node) # Visit arguments etc.

    # Add visit_Assign, visit_Name etc. if you want to track variable definitions/uses


# --- Neo4j Interaction ---

def get_neo4j_driver(uri, user, password):
    """Establishes connection to Neo4j."""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logging.info(f"Successfully connected to Neo4j at {uri}")
        return driver
    except neo4j.exceptions.AuthError:
        logging.error(f"Neo4j authentication failed for user '{user}'. Check credentials.")
        return None
    except neo4j.exceptions.ServiceUnavailable:
         logging.error(f"Could not connect to Neo4j at {uri}. Ensure database is running and accessible.")
         return None
    except Exception as e:
        logging.error(f"An unexpected error occurred connecting to Neo4j: {e}")
        return None

def create_constraints(tx):
    """Creates unique constraints in Neo4j."""
    logging.info("Ensuring Neo4j constraints exist...")
    # Using the generated unique node ID
    tx.run("CREATE CONSTRAINT unique_kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT unique_kg_file_path IF NOT EXISTS FOR (n:File) REQUIRE n.filepath IS UNIQUE")

def networkx_to_neo4j(tx, nx_graph):
    """Writes nodes and edges from a NetworkX graph to Neo4j using MERGE."""
    logging.info(f"Writing {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges to Neo4j...")

    # Write Nodes
    node_count = 0
    for node_id, attrs in nx_graph.nodes(data=True):
        # Basic properties common to all nodes
        node_props = {key: val for key, val in attrs.items() if val is not None} # Filter out None values
        node_props['id'] = node_id # Add the unique ID generated earlier

        # Get the primary type label
        node_label = attrs.get('type', 'Unknown') # Default label if type is missing

        # Use MERGE based on the unique 'id' property. Add KGNode base label + specific type label.
        cypher = f"""
        MERGE (n:KGNode: {node_label} {{id: $id}})
        SET n = $props
        """
        try:
            tx.run(cypher, id=node_id, props=node_props)
            node_count += 1
        except Exception as e:
            logging.error(f"Failed to write node {node_id} with props {node_props}: {e}")

    # Write Edges
    edge_count = 0
    for u, v, attrs in nx_graph.edges(data=True):
        rel_type = attrs.get('type', 'RELATED_TO').upper() # Default relationship type
        rel_props = {key: val for key, val in attrs.items() if key != 'type' and val is not None}

        # Use MATCH for nodes (must exist) and MERGE for relationship
        cypher = f"""
        MATCH (n1:KGNode {{id: $id1}})
        MATCH (n2:KGNode {{id: $id2}})
        MERGE (n1)-[r:`{rel_type}`]->(n2) // Use backticks for relationship type
        SET r = $props
        """
        try:
            tx.run(cypher, id1=u, id2=v, props=rel_props)
            edge_count += 1
        except Exception as e:
            logging.error(f"Failed to write edge {u} -> {v} ({rel_type}) with props {rel_props}: {e}")

    logging.info(f"Finished writing batch: {node_count} nodes, {edge_count} edges.")


# --- Main Script Logic ---

def main():
    parser = argparse.ArgumentParser(description="Analyze Python files in a GitHub repo, build a KG, and store in Neo4j.")
    parser.add_argument("--repo_url", required=True, help="URL of the GitHub repository to clone.")
    parser.add_argument("--branch", help="Specific branch/tag to clone (defaults to repo default).")
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687", help="URI for the Neo4j database.")
    parser.add_argument("--neo4j_user", default="neo4j", help="Username for Neo4j.")
    parser.add_argument("--neo4j_password", required=True, help="Password for Neo4j.")
    parser.add_argument("--neo4j_database", default="neo4j", help="Name of the Neo4j database.")
    parser.add_argument("--max_files", type=int, default=-1, help="Maximum number of Python files to process (-1 for all).")

    args = parser.parse_args()

    # --- Connect to Neo4j and Setup ---
    driver = get_neo4j_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    if not driver:
        sys.exit(1)
    try:
        with driver.session(database=args.neo4j_database) as session:
            session.execute_write(create_constraints)
    except Exception as e:
        logging.error(f"Failed to create Neo4j constraints: {e}")
        driver.close()
        sys.exit(1)

    # --- Clone Repo ---
    files_processed = 0
    files_failed_parsing = 0
    files_written_to_neo4j = 0
    files_failed_neo4j_write = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, 'cloned_repo')
        logging.info(f"Cloning '{args.repo_url}' into {repo_path}...")
        try:
            clone_options = {}
            if args.branch:
                clone_options['branch'] = args.branch
            Repo.clone_from(args.repo_url, repo_path, **clone_options)
            logging.info("Clone successful.")
        except GitCommandError as e:
            logging.error(f"Failed to clone repository: {e}\nstderr: {e.stderr}")
            driver.close()
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error during cloning: {e}")
            driver.close()
            sys.exit(1)

        logging.info("Starting AST parsing and KG building...")
        # --- Process Files ---
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__'] # Skip hidden/cache
            for filename in files:
                if filename.endswith('.py'):
                     # Check max file limit
                    if args.max_files != -1 and files_processed >= args.max_files:
                        logging.info(f"Reached max file limit ({args.max_files}). Stopping.")
                        break # Stop processing files in this directory

                    files_processed += 1
                    filepath = os.path.join(root, filename)
                    relative_filepath = os.path.relpath(filepath, repo_path) # Use relative path for better IDs
                    logging.info(f"Processing [{files_processed}]: {relative_filepath}")

                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            source_code = f.read()
                        if not source_code.strip():
                            logging.info(f"Skipping empty file: {relative_filepath}")
                            continue

                        # Parse AST
                        tree = ast.parse(source_code, filename=filepath)

                        # Build NetworkX graph
                        nx_graph = nx.DiGraph()
                        visitor = KGBuilderVisitor(relative_filepath, nx_graph)
                        visitor.visit(tree)

                        # Write graph to Neo4j
                        try:
                            with driver.session(database=args.neo4j_database) as session:
                                session.execute_write(networkx_to_neo4j, nx_graph)
                            files_written_to_neo4j += 1
                        except Exception as e:
                             logging.error(f"Failed to write graph for {relative_filepath} to Neo4j: {e}")
                             files_failed_neo4j_write += 1

                    except SyntaxError as e:
                        logging.warning(f"SyntaxError parsing {relative_filepath}: {e}")
                        files_failed_parsing += 1
                    except Exception as e:
                        logging.error(f"Unexpected error processing {relative_filepath}: {e}")
                        files_failed_parsing += 1

            # Check max file limit again after iterating files in a directory
            if args.max_files != -1 and files_processed >= args.max_files:
                break # Stop walking directories

    # --- Cleanup and Summary ---
    driver.close()
    logging.info("Neo4j driver closed.")
    logging.info("-" * 30)
    logging.info("Processing Summary:")
    logging.info(f"Total Python files found/attempted: {files_processed}")
    logging.info(f"Files failed parsing (SyntaxError, etc.): {files_failed_parsing}")
    logging.info(f"Graphs successfully written to Neo4j: {files_written_to_neo4j}")
    logging.info(f"Graphs failed Neo4j write: {files_failed_neo4j_write}")
    logging.info("-" * 30)

if __name__ == "__main__":
    main()
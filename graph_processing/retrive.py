
import os
import logging
import re
import json
from typing import List, Dict, Any, Tuple, Set, Union

# --- ChromaDB Imports ---
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import QueryResult

# --- Neo4j Import ---
from neo4j import GraphDatabase, Driver

# --- Project-Specific Imports ---
from config import EMBEDDING_MODEL_NAME, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from llm_client import get_llm_client, get_fallback_llm
from evaluate_retrieval import evaluate_retrieval, evaluate_generated_answer, export_evaluations_to_html

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHROMA_PATH_PREFIX = "./chroma_db_struct_chunk_"

# --- Initialize Embedding Function and LLM Client ---
try:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME, trust_remote_code=True)
    logging.info(f"Successfully initialized embedding function with {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize SentenceTransformerEmbeddingFunction: {e}. "
                  f"Please ensure all dependencies (e.g., 'einops') are installed using 'pip install einops'. Using None.")
    ef = None

try:
    LLM_CLIENT = get_llm_client()
    logging.info("Successfully initialized LLM client.")
except Exception as e:
    logging.warning(f"Failed to initialize LLM client: {e}. Using fallback LLM.")
    LLM_CLIENT = get_fallback_llm()

# Validate config variables
required_config_vars = ['EMBEDDING_MODEL_NAME', 'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD']
for var in required_config_vars:
    if not hasattr(__import__('config'), var):
        logging.critical(f"Missing required config variable: {var}")
        raise ValueError(f"Configuration error: {var} not defined in config.py")

# --- Helper Function for Neo4j Queries ---
def run_neo4j_query(driver: Driver, query: str, params: Dict = None) -> List[Dict]:
    """Runs a read query against Neo4j and returns results."""
    if params is None:
        params = {}
    results = []
    try:
        with driver.session(database="neo4j") as session:
            response = session.run(query, params)
            for record in response:
                results.append(record.data())
            return results
    except Exception as e:
        logging.error(f"Neo4j query failed: {e}")
        logging.error(f"  Query: {query}")
        logging.error(f"  Params: {params}")
        return []

# --- Query Intent Classification ---
def classify_query_intent(query: str, llm_client: Any) -> str:
    """
    Classifies the user's query intent to determine the appropriate response strategy.
    
    Returns:
        str: One of "general_question", "code_explanation", "change_request", "function_request"
    """
    classification_prompt = f"""
    Classify the following query about code into exactly one of these categories:
    1. general_question - General questions about the codebase, its purpose, etc.
    2. code_explanation - Questions asking to explain specific code or functionality
    3. change_request - Requests to modify existing code
    4. function_request - Requests to add new functionality
    
    Examples:
    - "What does this codebase do?" -> general_question
    - "How does the chunking function work?" -> code_explanation
    - "Modify the parser to handle JSON" -> change_request
    - "Add a function to validate inputs" -> function_request
    
    Query: {query}
    
    Category:
    """
    
    try:
        response = llm_client.invoke(classification_prompt)
        response_text = getattr(response, 'content', str(response)) if response else ""
        response_text = response_text.strip().lower()
        
        valid_categories = ["general_question", "code_explanation", "change_request", "function_request"]
        for category in valid_categories:
            if category in response_text:
                return category
        logging.warning(f"Query intent classification unclear: '{response_text}'. Defaulting to general_question.")
        return "general_question"
    except Exception as e:
        logging.error(f"Error classifying query intent: {e}")
        return "general_question"

def classify_change_type(query: str) -> str:
    """
    Identifies the type of code change requested in the query.
    
    Returns:
        str: One of "add_method", "modify_function", "add_feature", "fix_bug", "refactor", "other"
    """
    query_lower = query.lower()
    
    if "add method" in query_lower or "new method" in query_lower:
        return "add_method"
    elif "modify function" in query_lower or "change function" in query_lower:
        return "modify_function"
    elif "add feature" in query_lower or "new feature" in query_lower:
        return "add_feature"
    elif "fix bug" in query_lower or "fix issue" in query_lower or "fix error" in query_lower:
        return "fix_bug"
    elif "refactor" in query_lower or "clean up" in query_lower:
        return "refactor"
    else:
        return "other"

# --- Change Point Detection and Analysis ---
def identify_change_points(query: str, retrieved_context: Dict[str, Any], neo4j_driver: Driver) -> List[Dict]:
    """
    Identifies optimal points in the codebase for making changes based on the query.
    Returns a ranked list of potential locations for code changes.
    """
    change_points = []
    change_type = classify_change_type(query)
    
    if retrieved_context["top_entities"]:
        for entity in retrieved_context["top_entities"]:
            metadata = entity.get("metadata", {})
            entity_type = metadata.get("entity_type")
            entity_id = entity.get("id")
            
            if entity_type == "Class" and change_type == "add_method":
                class_methods = _find_class_methods(neo4j_driver, entity_id)
                change_points.append({
                    "target_entity": entity,
                    "change_type": "add_method",
                    "insertion_points": _analyze_method_insertion_points(class_methods)
                })
            elif entity_type == "Function" and change_type == "modify_function":
                change_points.append({
                    "target_entity": entity,
                    "change_type": "modify_function",
                    "modification_points": _analyze_function_modification_points(entity)
                })
            elif entity_type in ["Function", "Class", "Method"]:
                change_points.append({
                    "target_entity": entity,
                    "change_type": change_type,
                    "reason": f"Directly mentioned in query or semantically relevant"
                })
    
    if retrieved_context["related_graph_relationships"]:
        dependent_entities = _find_dependent_entities(neo4j_driver, 
            [e.get("metadata", {}).get("id") for e in retrieved_context["top_entities"] if e.get("metadata", {})])
        for entity in dependent_entities:
            change_points.append({
                "target_entity": {
                    "id": entity.get("id"),
                    "metadata": {
                        "name": entity.get("name"),
                        "entity_type": entity.get("type"),
                        "source_file": entity.get("file"),
                        "start_line": entity.get("line")
                    }
                },
                "change_type": "related_change",
                "reason": "Dependent on primary change target"
            })
    
    return _rank_change_points(change_points)

def _find_class_methods(neo4j_driver: Driver, class_id: str) -> List[Dict]:
    """Find methods belonging to a class."""
    methods_query = """
    MATCH (c:KGNode {id: $class_id})-[:CONTAINS_METHOD]->(m:KGNode)
    RETURN m.id as id, m.name as name, m.source_file as file, m.start_line as line
    """
    return run_neo4j_query(neo4j_driver, methods_query, {"class_id": class_id})

def _analyze_method_insertion_points(methods: List[Dict]) -> List[Dict]:
    """Identify where methods should be added."""
    if not methods:
        return [{"location": "end_of_class", "reason": "No existing methods, add at end of class"}]
    return [{
        "location": "after_method",
        "method_name": methods[-1].get("name"),
        "line": methods[-1].get("line"),
        "reason": "Add after last method for logical grouping"
    }]

def _analyze_function_modification_points(entity: Dict) -> List[Dict]:
    """Identify points within a function for modification."""
    return [{
        "location": "function_body",
        "line": entity.get("metadata", {}).get("start_line", 0) + 1,
        "reason": "Start modification at the beginning of function body"
    }]

def _find_dependent_entities(neo4j_driver: Driver, entity_ids: List[str]) -> List[Dict]:
    """Find entities that depend on the given entities."""
    if not entity_ids:
        return []
    query = """
    MATCH (n:KGNode)
    WHERE n.id IN $entity_ids
    WITH n
    MATCH (m:KGNode)-[:CALLS]->(n)
    RETURN DISTINCT m.id as id, m.name as name, coalesce(m.entity_type, 'Unknown') as type, 
           m.source_file as file, m.start_line as line
    LIMIT 10
    """
    return run_neo4j_query(neo4j_driver, query, {"entity_ids": entity_ids})

def _rank_change_points(change_points: List[Dict]) -> List[Dict]:
    """Rank change points by suitability."""
    for cp in change_points:
        score = 0
        if cp.get("change_type") == "related_change":
            score += 5
        else:
            score += 10
        if cp.get("change_type") == "add_method" and "insertion_points" in cp:
            score += 3
        elif cp.get("change_type") == "modify_function" and "modification_points" in cp:
            score += 3
        meta = cp.get("target_entity", {}).get("metadata", {})
        if "name" in meta and len(meta["name"]) > 0:
            score += 1
        cp["score"] = score
    return sorted(change_points, key=lambda x: x.get("score", 0), reverse=True)

def analyze_code_patterns(code_chunks: List[Dict], neo4j_driver: Driver) -> Dict[str, Any]:
    """
    Identifies recurring patterns in the code to ensure changes maintain consistency.
    """
    patterns = {
        "naming_conventions": _extract_naming_conventions(code_chunks),
        "error_handling": _extract_error_handling_patterns(code_chunks),
        "documentation": _extract_documentation_patterns(code_chunks),
        "design_patterns": _identify_design_patterns(neo4j_driver, code_chunks)
    }
    return patterns

def _extract_naming_conventions(code_chunks: List[Dict]) -> Dict[str, str]:
    """Extracts common naming conventions from code chunks."""
    class_names = []
    function_names = []
    variable_names = []
    
    for chunk in code_chunks:
        doc = chunk.get('document', '')
        class_pattern = r'class\s+([A-Za-z0-9_]+)'
        function_pattern = r'def\s+([A-Za-z0-9_]+)'
        variable_pattern = r'([a-z][A-Za-z0-9_]+)\s*='
        
        class_names.extend(re.findall(class_pattern, doc))
        function_names.extend(re.findall(function_pattern, doc))
        variable_names.extend(re.findall(variable_pattern, doc))
    
    class_convention = _determine_convention(class_names)
    function_convention = _determine_convention(function_names)
    variable_convention = _determine_convention(variable_names)
    
    return {
        "class_convention": class_convention,
        "function_convention": function_convention,
        "variable_convention": variable_convention
    }

def _determine_convention(names: List[str]) -> str:
    """Determines the naming convention from a list of names."""
    if not names:
        return "unknown"
    camel_case = 0
    snake_case = 0
    pascal_case = 0
    
    for name in names:
        if "_" in name:
            snake_case += 1
        elif name[0].isupper():
            pascal_case += 1
        else:
            has_upper = any(c.isupper() for c in name[1:])
            if has_upper:
                camel_case += 1
            else:
                snake_case += 1
    
    counts = [
        (snake_case, "snake_case"),
        (camel_case, "camelCase"),
        (pascal_case, "PascalCase")
    ]
    dominant = max(counts, key=lambda x: x[0])
    return dominant[1] if dominant[0] > 0 else "unknown"

def _extract_error_handling_patterns(code_chunks: List[Dict]) -> str:
    """Identifies error handling patterns in the code."""
    try_except_count = 0
    specific_except_count = 0
    generic_except_count = 0
    
    for chunk in code_chunks:
        doc = chunk.get('document', '')
        try_except_count += doc.count("try:")
        specific_except_count += len(re.findall(r'except\s+[A-Za-z0-9_]+', doc))
        generic_except_count += len(re.findall(r'except:', doc))
    
    if try_except_count == 0:
        return "no_error_handling"
    elif specific_except_count > generic_except_count:
        return "specific_exception_handling"
    else:
        return "generic_exception_handling"

def _extract_documentation_patterns(code_chunks: List[Dict]) -> str:
    """Identifies documentation patterns in the code."""
    docstring_count = 0
    docstring_types = {"triple_double": 0, "triple_single": 0}
    inline_comment_count = 0
    
    for chunk in code_chunks:
        doc = chunk.get('document', '')
        docstring_count += len(re.findall(r'""".*?"""', doc, re.DOTALL))
        docstring_types["triple_double"] += len(re.findall(r'""".*?"""', doc, re.DOTALL))
        docstring_types["triple_single"] += len(re.findall(r"'''.*?'''", doc, re.DOTALL))
        inline_comment_count += len(re.findall(r'#.*?$', doc, re.MULTILINE))
    
    if docstring_count > 0:
        dominant_style = "triple_double" if docstring_types["triple_double"] >= docstring_types["triple_single"] else "triple_single"
        return f"docstrings_{dominant_style}"
    elif inline_comment_count > 0:
        return "inline_comments"
    else:
        return "minimal_documentation"

def _identify_design_patterns(neo4j_driver: Driver, code_chunks: List[Dict]) -> List[str]:
    """Identifies design patterns used in the code."""
    return ["unknown"]

def analyze_change_impact(change_points: List[Dict], neo4j_driver: Driver) -> Dict[str, Any]:
    """
    Analyzes the potential impact of changes at the identified change points.
    """
    impact_analysis = {
        "direct_impacts": [],
        "potential_impacts": [],
        "risk_level": "low",
        "test_coverage": {}
    }
    
    for cp in change_points:
        entity_id = cp['target_entity'].get('id')
        if not entity_id:
            continue
            
        dependents_query = """
        MATCH (n:KGNode {id: $entity_id})
        MATCH (m:KGNode)-[:CALLS]->(n)
        RETURN m.id as id, m.name as name, coalesce(m.entity_type, 'Unknown') as type, 
               m.source_file as file, m.start_line as line
        """
        direct_dependents = run_neo4j_query(neo4j_driver, dependents_query, {"entity_id": entity_id})
        
        potential_query = """
        MATCH (n:KGNode {id: $entity_id})
        MATCH (m:KGNode)-[:CALLS]->()-[:CALLS]->(n)
        WHERE NOT (m)-[:CALLS]->(n)
        RETURN DISTINCT m.id as id, m.name as name, coalesce(m.entity_type, 'Unknown') as type, 
                m.source_file as file, m.start_line as line
        """
        potential_dependents = run_neo4j_query(neo4j_driver, potential_query, {"entity_id": entity_id})
        
        impact_analysis["direct_impacts"].extend(direct_dependents)
        impact_analysis["potential_impacts"].extend(potential_dependents)
    
    impact_count = len(impact_analysis["direct_impacts"]) + 0.5 * len(impact_analysis["potential_impacts"])
    if impact_count > 20:
        impact_analysis["risk_level"] = "high"
    elif impact_count > 5:
        impact_analysis["risk_level"] = "medium"
    
    test_files_query = """
    MATCH (n:KGNode {id: $entity_id})
    MATCH (test:KGNode)-[:CALLS*1..3]->(n)
    WHERE test.source_file CONTAINS 'test'
    RETURN DISTINCT test.source_file as test_file
    """
    
    for cp in change_points:
        entity_id = cp['target_entity'].get('id')
        if entity_id:
            test_files = run_neo4j_query(neo4j_driver, test_files_query, {"entity_id": entity_id})
            if test_files:
                impact_analysis["test_coverage"][entity_id] = [tf["test_file"] for tf in test_files]
    
    return impact_analysis

# --- Core RAG Implementation ---
def retrieve_context(
    query: str,
    index_id: str,
    chroma_client: chromadb.Client,
    neo4j_driver: Driver,
    embedding_function: Any,
    llm_client: Any = None,
    query_intent: str = "general_question",
    top_k_entities: int = 20,
    top_k_chunks: int = 5,
    graph_hops: int = 1,
    reranking_enabled: bool = True,
    use_query_expansion: bool = True,
) -> Dict[str, Any]:
    """
    Enhanced retrieval function with query expansion, hybrid search, and re-ranking.
    Retrieves relevant context from ChromaDB and Neo4j based on the query.
    Adjusts retrieval strategy based on query intent.
    """
    if embedding_function is None:
        logging.error("Embedding function is not available. Cannot perform vector search.")
        return {"error": "Missing embedding function"}

    entities_collection_name = f"{index_id}_entities"
    chunks_collection_name = f"{index_id}_chunks"
    retrieved_context = {
        "query": query,
        "expanded_query": query,
        "top_entities": [],
        "related_graph_nodes": [],
        "related_graph_relationships": [],
        "relevant_chunks": [],
        "error": None
    }

    try:
        entities_collection = chroma_client.get_collection(name=entities_collection_name,
                                                           embedding_function=embedding_function)
        chunks_collection = chroma_client.get_collection(name=chunks_collection_name,
                                                        embedding_function=embedding_function)
    except Exception as e:
        retrieved_context["error"] = f"Error getting ChromaDB collections: {e}"
        logging.error(retrieved_context["error"])
        return retrieved_context

    if use_query_expansion and llm_client:
        try:
            logging.info("Step 0: Expanding query...")
            expansion_prompt = f"""
            Your task is to expand the following user query about code to improve retrieval.
            Generate an expanded version that includes relevant technical terms, possible class names,
            method names, or concepts that might be relevant.

            Original query: {query}

            Expanded query:
            """
            expanded_query_response = llm_client.invoke(expansion_prompt)
            expanded_query = getattr(expanded_query_response, 'content', str(expanded_query_response))
            expanded_query = expanded_query.strip().split('\n')[0]
            logging.info(f"  Original query: '{query}'")
            logging.info(f"  Expanded query: '{expanded_query}'")
            retrieved_context["expanded_query"] = expanded_query
            query_for_search = f"{query} {expanded_query}"
        except Exception as e:
            logging.warning(f"Query expansion failed: {e}. Using original query.")
            query_for_search = query
    else:
        query_for_search = query

    logging.info(f"Step 1: Querying '{entities_collection_name}' with hybrid search...")
    try:
        entity_results: QueryResult = entities_collection.query(
            query_texts=[query_for_search],
            n_results=top_k_entities,
            include=["metadatas", "documents", "distances"],
        )
        if entity_results and entity_results.get('ids') and entity_results['ids'][0]:
            retrieved_context["top_entities"] = [
                {"id": id_, "metadata": meta, "description": doc, "distance": dist}
                for id_, meta, doc, dist in zip(
                    entity_results['ids'][0],
                    entity_results['metadatas'][0],
                    entity_results['documents'][0],
                    entity_results.get('distances', [[0] * len(entity_results['ids'][0])])[0]
                )
            ]
            logging.info(f"  Found {len(retrieved_context['top_entities'])} initial entities.")
        else:
            logging.info("  No relevant entities found in search.")
    except Exception as e:
        logging.warning(f"Hybrid search not available: {e}. Falling back to vector search.")
        try:
            entity_results: QueryResult = entities_collection.query(
                query_texts=[query_for_search],
                n_results=top_k_entities,
                include=["metadatas", "documents", "distances"],
            )
            if entity_results and entity_results.get('ids') and entity_results['ids'][0]:
                retrieved_context["top_entities"] = [
                    {"id": id_, "metadata": meta, "description": doc, "distance": dist}
                    for id_, meta, doc, dist in zip(
                        entity_results['ids'][0],
                        entity_results['metadatas'][0],
                        entity_results['documents'][0],
                        entity_results.get('distances', [[0] * len(entity_results['ids'][0])])[0]
                    )
                ]
                logging.info(f"  Found {len(retrieved_context['top_entities'])} initial entities.")
        except Exception as e:
            retrieved_context["error"] = f"Error querying entities collection: {e}"
            logging.error(retrieved_context["error"])

    logging.info(f"Step 2: Querying Neo4j graph with dynamic exploration...")
    query_terms = len(query_for_search.split())
    if query_intent in ["change_request", "function_request"]:
        dynamic_hops = max(graph_hops, 2)
    elif query_terms > 15:
        dynamic_hops = min(graph_hops + 1, 3)
    elif query_terms < 5:
        dynamic_hops = max(graph_hops - 1, 1)
    else:
        dynamic_hops = graph_hops
    logging.info(f"  Using {dynamic_hops} hop(s) for graph exploration (adjusted from {graph_hops})")

    subgraph_node_ids = []
    if retrieved_context["top_entities"]:
        initial_entity_ids = [e['id'] for e in retrieved_context["top_entities"]]
        nodes_query = f"""
        MATCH (start_node:KGNode) WHERE start_node.id IN $entity_ids
        CALL apoc.path.subgraphNodes(start_node, {{
            maxLevel: {dynamic_hops},
            relationshipFilter: "CONTAINS>|CONTAINS_METHOD>|CALLS>|IMPORTS>"
        }}) YIELD node
        RETURN COLLECT(DISTINCT node.id) AS node_ids
        """
        params = {"entity_ids": initial_entity_ids}
        node_id_results = run_neo4j_query(neo4j_driver, nodes_query, params)

        if node_id_results and node_id_results[0].get("node_ids"):
            subgraph_node_ids = node_id_results[0]["node_ids"]
            logging.info(f"  Identified {len(subgraph_node_ids)} potentially relevant nodes in Neo4j subgraph.")
            node_details_query = """
            MATCH (n:KGNode) WHERE n.id IN $node_ids
            RETURN n.id AS id, 
                   labels(n) AS labels, 
                   {
                     name: n.name,
                     description: n.description,
                     docstring: n.docstring,
                     type: n.type,
                     source_file: n.source_file,
                     start_line: n.start_line,
                     end_line: n.end_line
                   } AS properties
            """
            node_details = run_neo4j_query(neo4j_driver, node_details_query, {"node_ids": subgraph_node_ids})
            retrieved_context["related_graph_nodes"] = node_details
            logging.info(f"  Retrieved details for {len(node_details)} nodes.")

            rels_query = """
            MATCH (n1:KGNode)
            WHERE n1.id IN $node_ids
            MATCH (n1)-[rel]-()
            WITH n1, count(rel) AS degree
            MATCH (n1)-[r]->(n2:KGNode)
            WHERE n2.id IN $node_ids
            WITH n1.id AS source_id,
                 type(r) AS rel_type,
                 degree,
                 properties(r) AS rel_properties,
                 n2.id AS target_id,
                 CASE type(r)
                   WHEN 'CONTAINS_METHOD' THEN 5
                   WHEN 'CALLS' THEN 4
                   WHEN 'CONTAINS' THEN 3
                   WHEN 'IMPORTS' THEN 2
                   ELSE 1
                 END AS priority_score
            RETURN source_id, rel_type, degree, rel_properties, target_id, priority_score
            ORDER BY priority_score DESC
            """
            relationships = run_neo4j_query(neo4j_driver, rels_query, {"node_ids": subgraph_node_ids})
            retrieved_context["related_graph_relationships"] = relationships
            logging.info(f"  Found {len(relationships)} relationships within the subgraph.")
        else:
            logging.info("  No subgraph nodes found starting from initial entities.")
    else:
        logging.info("  Skipping Neo4j query as no initial entities were found.")

    if query_intent in ["change_request", "function_request"]:
        additional_context = {
            "code_patterns": {},
            "similar_implementations": [],
            "related_functions": []
        }
        if retrieved_context["top_entities"]:
            entity_ids = [e.get('id') for e in retrieved_context["top_entities"] if e.get('id')]
            if entity_ids:
                similar_query = """
                MATCH (n:KGNode)
                WHERE n.id IN $entity_ids AND n.entity_type IN ['Function', 'Class', 'Method']
                WITH n
                MATCH (m:KGNode) 
                WHERE m.entity_type = n.entity_type 
                AND m.id <> n.id 
                AND (m.name CONTAINS n.name OR n.name CONTAINS m.name OR apoc.text.jaroWinklerDistance(m.name, n.name) > 0.7)
                RETURN n.id as original_id, m.id as similar_id, 
                       n.name as original_name, m.name as similar_name,
                       n.source_file as original_file, m.source_file as similar_file
                LIMIT 10
                """
                similar_results = run_neo4j_query(neo4j_driver, similar_query, {"entity_ids": entity_ids})
                additional_context["similar_implementations"] = similar_results
                
                if query_intent == "function_request":
                    related_funcs_query = """
                    MATCH (n:KGNode)
                    WHERE n.id IN $entity_ids
                    MATCH (n)-[:CALLS|CONTAINS|IMPORTS*1..2]-(related:KGNode)
                    WHERE related.entity_type = 'Function' AND NOT related.id IN $entity_ids
                    RETURN DISTINCT related.id as id, related.name as name, 
                           related.source_file as file, related.start_line as line
                    LIMIT 15
                    """
                    related_funcs = run_neo4j_query(neo4j_driver, related_funcs_query, {"entity_ids": entity_ids})
                    additional_context["related_functions"] = related_funcs
        retrieved_context.update(additional_context)
        
    elif query_intent == "general_question":
        if retrieved_context["top_entities"]:
            module_query = """
            MATCH (f:KGNode)
            WHERE f.entity_type = 'File'
            WITH f ORDER BY size(apoc.coll.toSet([(f)-[:CONTAINS]->(m) | m.id])) DESC LIMIT 5
            RETURN f.id as id, f.path as path, f.absolute_path as absolute_path
            """
            modules = run_neo4j_query(neo4j_driver, module_query, {})
            retrieved_context["module_info"] = modules

    logging.info(f"Step 3: Enhanced code chunk retrieval with multiple strategies...")
    query_texts_for_chunks = [query_for_search]
    if retrieved_context["top_entities"]:
        query_texts_for_chunks.extend([e['description'] for e in retrieved_context["top_entities"][:5]])
        specialized_terms = []
        for entity in retrieved_context["top_entities"][:3]:
            meta = entity.get('metadata', {})
            entity_type = meta.get('entity_type', '')
            name = meta.get('name', '')
            if name and entity_type:
                specialized_terms.append(f"{entity_type} {name}")
        if specialized_terms:
            query_texts_for_chunks.extend(specialized_terms)

    for node_info in retrieved_context["related_graph_nodes"]:
        props = node_info.get('properties', {})
        desc = props.get('description') or props.get('docstring')
        name = props.get('name')
        labels = node_info.get('labels', [])
        if desc and len(desc) > 20:
            query_texts_for_chunks.append(desc[:300])
        elif name:
            node_type = next((l for l in labels if l != 'KGNode'), 'Node')
            query_texts_for_chunks.append(f"{node_type} {name}")

    unique_query_texts = list(set(filter(None, query_texts_for_chunks)))
    logging.info(f"  Using {len(unique_query_texts)} unique texts to find relevant chunks.")

    if unique_query_texts:
        try:
            chunk_results: QueryResult = chunks_collection.query(
                query_texts=unique_query_texts[:min(5, len(unique_query_texts))],
                n_results=top_k_chunks,
                include=["metadatas", "documents", "distances"],
            )
            unique_chunks = {}
            if chunk_results and chunk_results.get('ids'):
                for i in range(len(chunk_results['ids'])):
                    for j, (chunk_id, meta, doc) in enumerate(zip(
                            chunk_results['ids'][i],
                            chunk_results['metadatas'][i],
                            chunk_results['documents'][i]
                    )):
                        distance = chunk_results.get('distances', [[0] * len(chunk_results['ids'][i])])[i][j]
                        if chunk_id not in unique_chunks:
                            unique_chunks[chunk_id] = {
                                "id": chunk_id,
                                "metadata": meta,
                                "document": doc,
                                "distance": distance
                            }
                        else:
                            if distance < unique_chunks[chunk_id]["distance"]:
                                unique_chunks[chunk_id]["distance"] = distance

            chunk_list = list(unique_chunks.values())
            if reranking_enabled and llm_client and len(chunk_list) > 1:
                logging.info("Step 4: Re-ranking chunks with contextual relevance...")
                try:
                    chunks_to_rerank = chunk_list[:min(10, len(chunk_list))]
                    reranking_prompt = f"""
                    You are a code understanding assistant. Your task is to rate the relevance of code chunks 
                    to a user query on a scale of 1-10 (10 being most relevant).

                    User query: {query}

                    For each code chunk, provide a single number (1-10) as your rating.

                    {'-' * 40}
                    """
                    for i, chunk in enumerate(chunks_to_rerank):
                        doc = chunk.get('document', '')
                        meta = chunk.get('metadata', {})
                        source = meta.get('source_file', 'Unknown')
                        reranking_prompt += f"""
                        Chunk {i + 1}: (Source: {source})
                        ```python
                        {doc[:500]}{"..." if len(doc) > 500 else ""}
                        ```

                        Relevance score (1-10) for Chunk {i + 1}: 
                        """
                    reranking_response = llm_client.invoke(reranking_prompt)
                    reranking_text = getattr(reranking_response, 'content', str(reranking_response))
                    scores = re.findall(r"Chunk \d+:.*?Relevance score.*?(\d+)", reranking_text, re.DOTALL)
                    if scores and len(scores) == len(chunks_to_rerank):
                        for i, score in enumerate(scores):
                            chunks_to_rerank[i]["relevance_score"] = int(score)
                        chunks_to_rerank.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                        chunk_list[:len(chunks_to_rerank)] = chunks_to_rerank
                        logging.info(f"  Successfully reranked {len(chunks_to_rerank)} chunks.")
                    else:
                        logging.warning(
                            f"  Reranking failed: Couldn't extract enough scores ({len(scores)} found, {len(chunks_to_rerank)} needed)"
                        )
                except Exception as e:
                    logging.warning(f"  Reranking failed: {e}")
            retrieved_context["relevant_chunks"] = chunk_list
            logging.info(f"  Found {len(retrieved_context['relevant_chunks'])} unique relevant chunks.")
        except Exception as e:
            retrieved_context["error"] = f"Error querying chunks collection: {e}"
            logging.error(retrieved_context["error"])
    else:
        logging.info("  No query texts derived for retrieving chunks.")

    if not retrieved_context["top_entities"] and not retrieved_context["relevant_chunks"]:
        logging.warning("No context found. Attempting fallback strategies...")
        try:
            keywords = [word for word in re.findall(r'\w+', query.lower())
                        if word not in ['the', 'and', 'or', 'in', 'is', 'to', 'a', 'of', 'for', 'with', 'how']]
            if keywords:
                keyword_query = """
                MATCH (n:KGNode)
                WHERE any(keyword IN $keywords WHERE 
                    toLower(n.name) CONTAINS keyword OR 
                    toLower(coalesce(n.description,'')) CONTAINS keyword OR
                    toLower(coalesce(n.docstring,'')) CONTAINS keyword)
                RETURN n.id AS id, 
                       labels(n) AS labels, 
                       {
                         name: n.name,
                         description: n.description,
                         docstring: n.docstring,
                         type: n.type,
                         source_file: n.source_file,
                         start_line: n.start_line,
                         end_line: n.end_line
                       } AS properties
                LIMIT 50
                """
                keyword_results = run_neo4j_query(neo4j_driver, keyword_query, {"keywords": keywords})
                if keyword_results:
                    logging.info(f"  Fallback: Found {len(keyword_results)} nodes via keyword search in Neo4j.")
                    retrieved_context["related_graph_nodes"] = keyword_results
        except Exception as e:
            logging.warning(f"  Fallback Neo4j keyword search failed: {e}")

    logging.info("Context retrieval phase complete (including graph structure).")
    return retrieved_context

# --- Helper Function to Format Graph Context ---
def format_graph_context(nodes: List[Dict], relationships: List[Dict], max_rels: int = 15) -> str:
    """
    Formats graph nodes and relationships into a readable string for the LLM,
    including properties on the relationships.
    """
    if not nodes and not relationships:
        return ""
    node_details = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            logging.warning(f"Skipping node without ID: {node}")
            continue
        props = node.get("properties", {})
        labels = node.get("labels", ["Node"])
        node_type = "Unknown"
        type_priority = ["Method", "Function", "Class", "Module", "Call", "Import"]
        for p_type in type_priority:
            if p_type in labels:
                node_type = p_type
                break
        else:
            node_type = labels[0] if labels else "Node"
        name = props.get("name", node_id)
        node_details[node_id] = {"name": name, "type": node_type}
    formatted_rels = []
    rel_count = 0
    if relationships:
        for i, rel in enumerate(relationships):
            if rel_count >= max_rels:
                formatted_rels.append("... (more relationships shown than limit)")
                break
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            rel_type = rel.get("rel_type", "RELATED_TO")
            rel_props = rel.get("rel_properties", {})
            if not source_id or not target_id:
                logging.warning(f"Skipping relationship with missing source/target ID: {rel}")
                continue
            source_info = node_details.get(source_id, {"name": source_id, "type": "Unknown/External"})
            target_info = node_details.get(target_id, {"name": target_id, "type": "Unknown/External"})
            props_str = ""
            if rel_props:
                prop_items = [f"{k}: {str(v)[:50]}{'...' if len(str(v)) > 50 else ''}" for k, v in rel_props.items()]
                if prop_items:
                    props_str = " {" + ", ".join(prop_items) + "}"
            formatted_rels.append(
                f"- {source_info['name']} ({source_info['type']}) "
                f"-[{rel_type}{props_str}]->"
                f" {target_info['name']} ({target_info['type']})"
            )
            rel_count += 1
    output_parts = []
    if formatted_rels:
        output_parts.append("=== Knowledge Graph Structure (Relationships with Properties) ===")
        output_parts.extend(formatted_rels)
        output_parts.append("="*20)
    elif nodes:
        output_parts.append("=== Knowledge Graph Structure (Nodes Found) ===")
        for node_id, details in node_details.items():
            output_parts.append(f"- {details['name']} ({details['type']})")
        if relationships is not None:
            output_parts.append("(No direct relationships retrieved between these specific nodes)")
        output_parts.append("="*20)
    return "\n".join(output_parts) + "\n" if output_parts else ""

# --- Generate Response ---
def generate_response(
    query: str,
    retrieved_context: Dict[str, Any],
    llm_client: Any,
    query_intent: str = "general_question"
) -> str:
    """
    Generates a response using the LLM based on the query, retrieved context,
    and classified query intent.
    """
    logging.info(f"Generating response for query intent: {query_intent}")
    
    if retrieved_context.get("error") and not (
            retrieved_context["top_entities"] or retrieved_context["relevant_chunks"]):
        logging.warning("Cannot generate response due to error during retrieval and no context found.")
        return f"Sorry, I encountered an error retrieving context: {retrieved_context['error']}"

    context_pieces = []
    context_pieces.append(f"User Query: {query}\n")
    
    if retrieved_context["top_entities"]:
        context_pieces.append("=== Potentially Relevant Code Entities ===")
        for i, entity in enumerate(retrieved_context["top_entities"]):
            meta = entity.get('metadata', {})
            desc = entity.get('description', 'N/A')
            entity_name = meta.get('name', 'N/A')
            entity_type = meta.get('entity_type', 'N/A')
            file_location = f"{meta.get('source_file', 'N/A')}:{meta.get('start_line', 'N/A')}"
            context_pieces.append(
                f"Entity {i + 1}: {entity_type} '{entity_name}' in {file_location}\n"
                f"Description: {desc[:300]}..."
            )
        context_pieces.append("=" * 20 + "\n")
    
    graph_context_str = format_graph_context(
        retrieved_context.get("related_graph_nodes", []),
        retrieved_context.get("related_graph_relationships", []),
        max_rels=20
    )
    if graph_context_str:
        context_pieces.append(graph_context_str)
    
    if retrieved_context["relevant_chunks"]:
        context_pieces.append("=== Relevant Code Snippets ===")
        max_chunks = 15
        seen_ids = set()
        unique_chunks = []
        for chunk in retrieved_context["relevant_chunks"]:
            cid = chunk.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(chunk)
            if len(unique_chunks) >= max_chunks:
                break
        for i, chunk in enumerate(unique_chunks, start=1):
            meta = chunk.get('metadata', {})
            doc = chunk.get('document', '')
            context_pieces.append(
                f"Code Snippet {i} "
                f"(Source: {meta.get('source_file', 'N/A')}, "
                f"Lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}):\n"
                f"```python\n{doc}\n```"
            )
        context_pieces.append("=" * 20 + "\n")
    
    if query_intent == "general_question":
        if "module_info" in retrieved_context and retrieved_context["module_info"]:
            context_pieces.append("=== Key Modules in Codebase ===")
            for i, module in enumerate(retrieved_context["module_info"]):
                context_pieces.append(f"Module {i+1}: {module.get('path', 'N/A')}")
            context_pieces.append("=" * 20 + "\n")
    
    elif query_intent in ["change_request", "function_request"]:
        if "similar_implementations" in retrieved_context and retrieved_context["similar_implementations"]:
            context_pieces.append("=== Similar Implementations ===")
            for i, similar in enumerate(retrieved_context["similar_implementations"]):
                context_pieces.append(
                    f"Similar {i+1}: '{similar.get('original_name', 'N/A')}' in {similar.get('original_file', 'N/A')} "
                    f"is similar to '{similar.get('similar_name', 'N/A')}' in {similar.get('similar_file', 'N/A')}"
                )
            context_pieces.append("=" * 20 + "\n")
        
        if query_intent == "function_request" and "related_functions" in retrieved_context:
            context_pieces.append("=== Related Functions ===")
            for i, func in enumerate(retrieved_context.get("related_functions", [])[:10]):
                context_pieces.append(f"Function {i+1}: '{func.get('name', 'N/A')}' in {func.get('file', 'N/A')}")
            context_pieces.append("=" * 20 + "\n")
    
    if "change_points" in retrieved_context and retrieved_context["change_points"]:
        context_pieces.append("=== Recommended Change Points ===")
        for i, cp in enumerate(retrieved_context["change_points"][:5]):
            meta = cp["target_entity"].get("metadata", {})
            context_pieces.append(
                f"Change Point {i+1}: {meta.get('entity_type', 'Entity')} '{meta.get('name', 'N/A')}' "
                f"in {meta.get('source_file', 'N/A')}:{meta.get('start_line', 'N/A')}"
            )
            if "reason" in cp:
                context_pieces.append(f"  Reason: {cp['reason']}")
            if "insertion_points" in cp:
                for ip in cp["insertion_points"]:
                    context_pieces.append(f"  Suggested insertion: {ip.get('location', 'N/A')} - {ip.get('reason', 'N/A')}")
        context_pieces.append("=" * 20 + "\n")
    
    if "impact_analysis" in retrieved_context:
        impact = retrieved_context["impact_analysis"]
        context_pieces.append("=== Change Impact Analysis ===")
        context_pieces.append(f"Risk Level: {impact.get('risk_level', 'low')}")
        context_pieces.append(f"Direct Impacts: {len(impact.get('direct_impacts', []))} code locations")
        context_pieces.append(f"Potential Indirect Impacts: {len(impact.get('potential_impacts', []))} code locations")
        
        if impact.get("direct_impacts"):
            context_pieces.append("\nExample Impacted Code:")
            for i, imp in enumerate(impact["direct_impacts"][:3]):
                context_pieces.append(f"  {i+1}. {imp.get('name', 'N/A')} in {imp.get('file', 'N/A')}:{imp.get('line', 'N/A')}")
        
        if impact.get("test_coverage"):
            context_pieces.append("\nTests Potentially Affected:")
            for entity_id, tests in list(impact["test_coverage"].items())[:3]:
                for i, test in enumerate(tests[:2]):
                    context_pieces.append(f"  {i+1}. {test}")
        
        context_pieces.append("=" * 20 + "\n")
    
    combined_context = "\n".join(context_pieces)
    max_context_chars = 55000
    if len(combined_context) > max_context_chars:
        logging.warning(f"Combined context length ({len(combined_context)}) exceeds limit ({max_context_chars}). Truncating.")
        combined_context = combined_context[:max_context_chars] + "\n... (Context Truncated)"
    
    if query_intent == "general_question":
        system_prompt = (
            "You are an AI assistant specialized in understanding and answering questions about a Python codebase. "
            "Use the provided context to give an overview of what the codebase does, its architecture, and main components. "
            "Focus on high-level understanding rather than implementation details. "
            "If asked about the purpose of the codebase, explain what problem it solves and how it works at a conceptual level."
        )
    elif query_intent == "code_explanation":
        system_prompt = (
            "You are an AI assistant specialized in explaining code. "
            "Use the provided context to explain how specific parts of the code work, what certain functions or classes do, "
            "and how different components interact. Provide detailed explanations of algorithms or data structures if relevant."
        )
    elif query_intent == "change_request":
        system_prompt = (
            "You are an AI assistant specialized in recommending code changes. "
            "Based on the user's request and the provided context, suggest how existing code could be modified. "
            "For each suggested change, explain: 1) where the change should be made (file, line numbers), "
            "2) what the change should be, and 3) why this change is appropriate. "
            "Consider how the change might affect other parts of the codebase."
        )
    elif query_intent == "function_request":
        system_prompt = (
            "You are an AI assistant specialized in suggesting new code implementations. "
            "Based on the user's request and the provided context, suggest how new functionality could be added to the codebase. "
            "Recommend: 1) where the new code should be placed, 2) what the implementation should look like (provide code), "
            "and 3) how it integrates with existing functionality. Follow the codebase's existing patterns and conventions."
        )
    
    final_prompt = f"""{system_prompt}
    
## Provided Context:
{combined_context}

## User Query:
{query}

## Answer:
"""
    print(final_prompt) 
    logging.info("Sending prompt to LLM...")
    try:
        response = llm_client.invoke(final_prompt)
        response_text = getattr(response, 'content', str(response)) if response else "No response from LLM"
        logging.info("Received response from LLM.")
        return response_text
    except Exception as e:
        logging.error(f"LLM invocation failed: {e}")
        return "Sorry, I encountered an error generating the response."

def generate_change_recommendation(
    query: str,
    retrieved_context: Dict[str, Any],
    change_points: List[Dict],
    code_patterns: Dict[str, Any],
    llm_client: Any
) -> Dict[str, Any]:
    """
    Generates specific change recommendations based on the retrieved context,
    identified change points, and code patterns.
    """
    context_pieces = []
    
    if change_points:
        context_pieces.append("=== Recommended Change Points ===")
        for i, cp in enumerate(change_points[:5], 1):
            context_pieces.append(f"Change Point {i}:")
            meta = cp['target_entity'].get('metadata', {})
            context_pieces.append(f"  Target: {meta.get('name', 'Unknown')} " +
                                f"({meta.get('entity_type', 'Unknown')})")
            context_pieces.append(f"  Change Type: {cp.get('change_type', 'Unknown')}")
            context_pieces.append(f"  File: {meta.get('source_file', 'Unknown')}")
            context_pieces.append(f"  Line: {meta.get('start_line', 'Unknown')}")
            if 'reason' in cp:
                context_pieces.append(f"  Reason: {cp['reason']}")
    
    if code_patterns:
        context_pieces.append("=== Code Patterns to Maintain ===")
        if 'naming_conventions' in code_patterns:
            nc = code_patterns['naming_conventions']
            context_pieces.append(f"  Naming: Classes use {nc.get('class_convention', 'unknown')}, " +
                                f"Functions use {nc.get('function_convention', 'unknown')}")
        if 'error_handling' in code_patterns:
            context_pieces.append(f"  Error Handling: {code_patterns['error_handling']}")
        if 'documentation' in code_patterns:
            context_pieces.append(f"  Documentation: {code_patterns['documentation']}")
    
    if retrieved_context["relevant_chunks"]:
        context_pieces.append("=== Relevant Code Snippets ===")
        seen_ids = set()
        unique_chunks = []
        for chunk in retrieved_context["relevant_chunks"]:
            cid = chunk.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(chunk)
            if len(unique_chunks) >= 10:
                break
        for i, chunk in enumerate(unique_chunks, start=1):
            meta = chunk.get('metadata', {})
            doc = chunk.get('document', '')
            context_pieces.append(
                f"Code Snippet {i} "
                f"(Source: {meta.get('source_file', 'N/A')}, "
                f"Lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}):\n"
                f"```python\n{doc}\n```"
            )
    
    if "similar_implementations" in retrieved_context and retrieved_context["similar_implementations"]:
        context_pieces.append("=== Similar Implementations for Reference ===")
        for i, similar in enumerate(retrieved_context["similar_implementations"][:5]):
            context_pieces.append(f"Similar Code {i+1}: {similar.get('original_name', 'N/A')} in "
                                f"{similar.get('original_file', 'N/A')} is similar to "
                                f"{similar.get('similar_name', 'N/A')} in {similar.get('similar_file', 'N/A')}")
    
    system_prompt = (
        "You are an AI assistant specialized in suggesting code changes. "
        "Based on the user's query and the provided code context, suggest specific changes "
        "that follow the existing code patterns and best practices. "
        "For each suggested change, provide: "
        "1. The exact file and location where the change should be made "
        "2. The code before the change (if applicable) "
        "3. The code after the change, with the changes clearly marked "
        "4. An explanation of why this change is recommended "
        "5. Any potential side effects or additional changes needed elsewhere"
    )
    
    combined_context = "\n".join(context_pieces)
    
    final_prompt = f"""{system_prompt}
    
## Code Context:
{combined_context}

## User's Change Request:
{query}

## Suggested Changes:
"""
    
    try:
        response = llm_client.invoke(final_prompt)
        response_text = getattr(response, 'content', str(response)) if response else "No response from LLM"
        parsed_changes = _parse_change_suggestions(response_text)
        return {
            "raw_response": response_text,
            "structured_changes": parsed_changes,
            "change_points": change_points,
            "code_patterns": code_patterns
        }
    except Exception as e:
        logging.error(f"LLM invocation failed: {e}")
        return {"error": f"Failed to generate change recommendations: {str(e)}"}

def _parse_change_suggestions(response_text: str) -> List[Dict]:
    """
    Parse the LLM's change suggestions into a structured format.
    """
    suggestions = []
    file_blocks = re.split(r'(?=File:\s+[\w\./]+\s+\(Lines?.*?\))', response_text)
    
    for block in file_blocks:
        if not block.strip():
            continue
        file_match = re.search(r'File:\s+([\w\./]+)\s+\(Lines?\s*(\d+)(?:-(\d+))?\)', block)
        if file_match:
            filename = file_match.group(1)
            start_line = int(file_match.group(2))
            end_line = int(file_match.group(3)) if file_match.group(3) else start_line
            before_match = re.search(r'Before:.*?```(?:python)?\s*(.*?)```', block, re.DOTALL)
            after_match = re.search(r'After:.*?```(?:python)?\s*(.*?)```', block, re.DOTALL)
            explanation_match = re.search(r'Explanation:(.*?)(?=\n\n|$)', block, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            suggestion = {
                "file": filename,
                "start_line": start_line,
                "end_line": end_line,
                "before_code": before_match.group(1).strip() if before_match else None,
                "after_code": after_match.group(1).strip() if after_match else None,
                "explanation": explanation
            }
            suggestions.append(suggestion)
    
    if not suggestions:
        logging.warning("No change suggestions parsed from LLM response.")
    return suggestions

# --- Main Execution Example ---
if __name__ == "__main__":
    if ef is None:
        logging.critical("Embedding Function (ef) is not available. Cannot run RAG pipeline.")
    else:
        TARGET_INDEX_ID = "py_structchunk_neo4j_v2_option2_11"
        TARGET_CHROMA_DB_PATH = f"{CHROMA_PATH_PREFIX}{TARGET_INDEX_ID}"
        neo4j_driver_instance = None
        chroma_client_instance = None
        
        try:
            logging.info("Initializing Neo4j driver...")
            neo4j_driver_instance = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            neo4j_driver_instance.verify_connectivity()
            logging.info("Neo4j connection successful.")
            
            # Validate Neo4j schema
            schema_check_query = """
            MATCH (n:KGNode)
            RETURN DISTINCT labels(n) AS labels, properties(n) AS props
            LIMIT 1
            """
            schema_result = run_neo4j_query(neo4j_driver_instance, schema_check_query)
            if not schema_result:
                logging.warning("Neo4j schema check returned no results. Verify KGNode existence.")
            
            logging.info(f"Initializing ChromaDB client at path: {TARGET_CHROMA_DB_PATH}")
            if not os.path.exists(TARGET_CHROMA_DB_PATH):
                raise FileNotFoundError(f"ChromaDB path does not exist: {TARGET_CHROMA_DB_PATH}. Please run the indexing script first.")
            chroma_client_instance = chromadb.PersistentClient(path=TARGET_CHROMA_DB_PATH)
            chroma_client_instance.heartbeat()
            logging.info("ChromaDB client initialized successfully.")
            
            while True:
                USER_QUERY = input("What would you like to know about the code? (Type 'exit' to quit): ")
                if USER_QUERY.lower() == 'exit':
                    break
                
                logging.info(f"===== Starting RAG Pipeline for Index: {TARGET_INDEX_ID} =====")
                
                query_intent = classify_query_intent(USER_QUERY, LLM_CLIENT)
                logging.info(f"Classified query intent as: {query_intent}")
                
                retrieved_data = retrieve_context(
                    query=USER_QUERY,
                    index_id=TARGET_INDEX_ID,
                    chroma_client=chroma_client_instance,
                    neo4j_driver=neo4j_driver_instance,
                    embedding_function=ef,
                    llm_client=LLM_CLIENT,
                    query_intent=query_intent,
                    top_k_entities=15,
                    top_k_chunks=5,
                    graph_hops=3
                )
                
                if query_intent in ["change_request", "function_request"]:
                    code_patterns = analyze_code_patterns(retrieved_data["relevant_chunks"], neo4j_driver_instance)
                    retrieved_data["code_patterns"] = code_patterns
                    change_points = identify_change_points(USER_QUERY, retrieved_data, neo4j_driver_instance)
                    retrieved_data["change_points"] = change_points
                    impact_analysis = analyze_change_impact(change_points, neo4j_driver_instance)
                    retrieved_data["impact_analysis"] = impact_analysis
                
                final_answer = generate_response(
                    query=USER_QUERY,
                    retrieved_context=retrieved_data,
                    llm_client=LLM_CLIENT,
                    query_intent=query_intent
                )
                
                # Evaluate retrieval and response
                retrieval_eval = evaluate_retrieval(retrieved_data, USER_QUERY, LLM_CLIENT)
                answer_eval = evaluate_generated_answer(
                    query=USER_QUERY,
                    generated_answer=final_answer,
                    llm_client=LLM_CLIENT,
                    retrieved_context=retrieved_data
                )

                export_evaluations_to_html(
                    query=USER_QUERY,
                    retrieved_context=retrieved_data,
                    generated_answer=final_answer,
                    rag_answer=final_answer,  # Using the same answer for both fields
                    llm_only_answer=None,  # You could generate a non-RAG answer if needed
                    llm_client=LLM_CLIENT,
                    ground_truth=None,  # Add ground truth if available
                    output_file=f"evaluation_{TARGET_INDEX_ID}_{query_intent}_{int(time.time())}.html"
                )
                print("\n===== Final Answer =====")
                print(final_answer)
                
                if query_intent in ["change_request", "function_request"] and "impact_analysis" in retrieved_data:
                    impact = retrieved_data["impact_analysis"]
                    print("\n===== Change Impact Analysis =====")
                    print(f"Risk Level: {impact.get('risk_level', 'Unknown')}")
                    print(f"Direct Impacts: {len(impact.get('direct_impacts', []))} code locations")
                    print(f"Potential Indirect Impacts: {len(impact.get('potential_impacts', []))} code locations")
                    
                    show_details = input("\nShow detailed impact analysis? (y/n): ")
                    if show_details.lower() == 'y':
                        print("\nDirect Impacts:")
                        for i, impact_item in enumerate(impact['direct_impacts'][:10]):
                            print(f"  - {impact_item.get('name', 'N/A')} ({impact_item.get('type', 'N/A')}) in {impact_item.get('file', 'N/A')}:{impact_item.get('line', 'N/A')}")
                        if len(impact['direct_impacts']) > 10:
                            print(f"  ... and {len(impact['direct_impacts']) - 10} more")
                        if impact.get("test_coverage"):
                            print("\nTests Potentially Affected:")
                            for entity_id, tests in impact["test_coverage"].items():
                                print(f"  For {entity_id}:")
                                for test_file in tests[:3]:
                                    print(f"    - {test_file}")
                
                # Print evaluation results
                if retrieval_eval:
                    print(f"Metrics: {retrieval_eval.get('metrics', {})}")
                    print(f"Analysis: {retrieval_eval.get('analysis', 'N/A')}")
                else:
                    print("Retrieval evaluation unavailable")
                print("\n===== Answer Evaluation =====")
                print(f"Metrics: {answer_eval.get('metrics', {})}")
                print(f"Analysis: {answer_eval.get('analysis', 'N/A')}")
                
        except FileNotFoundError as e:
            logging.critical(str(e))
            print(f"Error: {str(e)}. Please ensure the indexing script has been run.")
        except Exception as e:
            logging.critical(f"An error occurred during the RAG pipeline: {e}", exc_info=True)
            print(f"Error: An unexpected error occurred. Check logs for details.")
        finally:
            if neo4j_driver_instance:
                logging.info("Closing Neo4j connection.")
                neo4j_driver_instance.close()
        
        logging.info(f"===== RAG Pipeline Complete =====")

import os
import json
import logging
import argparse
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
import networkx as nx
from tqdm import tqdm
import datetime

from config import (
    EMBEDDING_MODEL_NAME, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    CHROMA_PATH_PREFIX, count_tokens, CHUNK_ENCODING
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GraphLoomEvaluator:
    """Evaluator for GraphLoom's knowledge graph and chunking quality."""

    def __init__(self, index_id: str, verbose: bool = False):
        """
        Initialize the evaluator with a specific index ID.

        Args:
            index_id: The unique identifier for the graph/index to evaluate
            verbose: Whether to output detailed logs
        """
        self.index_id = index_id
        self.chroma_path = f"{CHROMA_PATH_PREFIX}{index_id}"
        self.verbose = verbose
        self.log_level = logging.INFO if verbose else logging.WARNING
        logging.getLogger().setLevel(self.log_level)

        # Connect to ChromaDB
        self.ef = None
        self.chroma_client = None
        self._init_chroma()

        # Connect to Neo4j
        self.neo4j_driver = None
        self._init_neo4j()

        # Load metadata
        self.metadata = self._load_metadata()

        # Evaluation metrics storage
        self.metrics = {
            "graph_metrics": {},
            "chunk_metrics": {},
            "entity_metrics": {},
            "retrieval_metrics": {},
        }

    def _init_chroma(self):
        """Initialize connection to ChromaDB."""
        try:
            logging.info(f"Connecting to ChromaDB at {self.chroma_path}")
            if not os.path.exists(self.chroma_path):
                logging.error(f"ChromaDB path {self.chroma_path} does not exist")
                return

            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.chroma_client.heartbeat()

            # Get available collections
            collections = self.chroma_client.list_collections()
            self.collection_names = [c.name for c in collections]
            self.chunks_collection_name = f"{self.index_id}_chunks"
            self.entities_collection_name = f"{self.index_id}_entities"

            # Check if the expected collections exist
            if self.chunks_collection_name not in self.collection_names:
                logging.warning(f"Chunks collection {self.chunks_collection_name} not found")
            if self.entities_collection_name not in self.collection_names:
                logging.warning(f"Entities collection {self.entities_collection_name} not found")

            logging.info(f"Found collections: {self.collection_names}")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None

    def _init_neo4j(self):
        """Initialize connection to Neo4j."""
        try:
            logging.info(f"Connecting to Neo4j at {NEO4J_URI}")
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.neo4j_driver.verify_connectivity()
            logging.info("Neo4j connection successful")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            self.neo4j_driver = None

    def _load_metadata(self) -> Dict:
        """Load metadata for the index."""
        metadata_filename = f"{self.index_id}_metadata.json"
        try:
            if os.path.exists(metadata_filename):
                with open(metadata_filename, 'r') as f:
                    kv_store = json.load(f)
                    metadata = kv_store.get(self.index_id, {})
                    logging.info(f"Loaded metadata for index {self.index_id}")
                    return metadata
            else:
                logging.warning(f"Metadata file {metadata_filename} not found")
                return {}
        except Exception as e:
            logging.error(f"Failed to load metadata: {e}")
            return {}

    def evaluate_graph(self) -> Dict:
        """
        Evaluate the Neo4j knowledge graph quality.

        Returns:
            Dictionary of graph quality metrics
        """
        if not self.neo4j_driver:
            logging.error("Neo4j driver not initialized")
            return {}

        logging.info("Evaluating graph quality...")
        metrics = {}

        try:
            with self.neo4j_driver.session() as session:
                # Get node counts by type
                node_count_query = """
                MATCH (n:KGNode)
                WHERE n.id STARTS WITH $index_id
                WITH labels(n) AS labels
                UNWIND labels AS label
                RETURN label, count(label) AS count
                ORDER BY count DESC
                """
                node_counts = session.run(node_count_query, {"index_id": self.index_id}).data()
                metrics["node_counts_by_type"] = {row["label"]: row["count"] for row in node_counts}
                metrics["total_nodes"] = sum(metrics["node_counts_by_type"].values())

                # Get relationship counts by type
                rel_count_query = """
                MATCH (n:KGNode)-[r]->(m:KGNode)
                WHERE n.id STARTS WITH $index_id OR m.id STARTS WITH $index_id
                RETURN type(r) AS relationship_type, count(r) AS count
                ORDER BY count DESC
                """
                rel_counts = session.run(rel_count_query, {"index_id": self.index_id}).data()
                metrics["relationship_counts_by_type"] = {row["relationship_type"]: row["count"] for row in rel_counts}
                metrics["total_relationships"] = sum(metrics["relationship_counts_by_type"].values())

                # Get graph density (ratio of actual to possible connections)
                if metrics["total_nodes"] > 1:
                    metrics["graph_density"] = metrics["total_relationships"] / (
                                metrics["total_nodes"] * (metrics["total_nodes"] - 1))
                else:
                    metrics["graph_density"] = 0

                # Get connected components count
                # Using a simpler approach to avoid deprecated syntax
                connected_components_query = """
                MATCH (n:KGNode)
                WHERE n.id STARTS WITH $index_id
                OPTIONAL MATCH (n)-[*1..3]-(connected:KGNode)
                WHERE connected.id STARTS WITH $index_id
                WITH n, collect(connected) AS connections
                WITH count(DISTINCT n) AS node_count, count(DISTINCT connections) AS component_count
                RETURN 
                    CASE WHEN node_count = 0 THEN 0
                    WHEN component_count = 0 THEN node_count
                    ELSE component_count
                    END AS connected_components
                """
                try:
                    cc_result = session.run(connected_components_query, {"index_id": self.index_id}).single()
                    if cc_result:
                        metrics["connected_components"] = cc_result["connected_components"]
                except Exception as e:
                    logging.warning(f"Connected components query failed (may be too expensive): {e}")
                    # Provide a simpler query as fallback
                    try:
                        simple_query = """
                        MATCH (n:KGNode)
                        WHERE n.id STARTS WITH $index_id
                        RETURN count(n) AS node_count
                        """
                        simple_result = session.run(simple_query, {"index_id": self.index_id}).single()
                        metrics["nodes_count"] = simple_result["node_count"] if simple_result else 0
                        metrics["connected_components"] = "Unknown (query too expensive)"
                    except Exception as e2:
                        logging.error(f"Even simple node count query failed: {e2}")
                        metrics["connected_components"] = "Unknown (query failed)"

                # Get the most connected nodes (hub analysis)
                hub_query = """
                MATCH (n:KGNode)
                WHERE n.id STARTS WITH $index_id
                WITH n
                MATCH (n)-[r]-(connected)
                WITH n, count(DISTINCT connected) AS degree
                ORDER BY degree DESC
                LIMIT 10
                RETURN n.id AS id, n.name AS name, n.entity_type AS type, degree
                """
                hub_nodes = session.run(hub_query, {"index_id": self.index_id}).data()
                metrics["top_hub_nodes"] = hub_nodes

                # Calculate centrality for key nodes
                # This will give an indication of which nodes are most important in the graph
                centrality_query = """
                MATCH (n:KGNode)
                WHERE n.id STARTS WITH $index_id AND n.entity_type IN ['Function', 'Class']
                WITH n
                MATCH (n)-[r1]-(directly_connected)
                WHERE directly_connected.id STARTS WITH $index_id
                WITH n, count(DISTINCT directly_connected) AS direct_connections
                OPTIONAL MATCH (n)-[r1]-(intermediate)-[r2]-(second_level)
                WHERE intermediate.id STARTS WITH $index_id 
                AND second_level.id STARTS WITH $index_id
                WITH n, direct_connections, count(DISTINCT second_level) AS indirect_connections
                WITH n, direct_connections + indirect_connections AS centrality
                RETURN n.id AS id, n.name AS name, n.entity_type AS type, centrality
                ORDER BY centrality DESC
                LIMIT 10
                """
                centrality_nodes = session.run(centrality_query, {"index_id": self.index_id}).data()
                metrics["top_central_nodes"] = centrality_nodes

        except Exception as e:
            logging.error(f"Error evaluating graph: {e}")

        # Store metrics
        self.metrics["graph_metrics"] = metrics
        return metrics

    def evaluate_chunks(self) -> Dict:
        """
        Evaluate the quality of text chunks.

        Returns:
            Dictionary of chunk quality metrics
        """
        if not self.chroma_client:
            logging.error("ChromaDB client not initialized")
            return {}

        logging.info("Evaluating chunk quality...")
        metrics = {}

        try:
            # Get chunks collection
            try:
                chunks_collection = self.chroma_client.get_collection(
                    name=self.chunks_collection_name,
                    embedding_function=self.ef
                )

                # Get all chunks
                chunks_result = chunks_collection.get()
                chunks_count = len(chunks_result.get("ids", []))
                metadatas = chunks_result.get("metadatas", [])
                documents = chunks_result.get("documents", [])

                metrics["total_chunks"] = chunks_count

                # Token distribution analysis
                if "token_count" in metadatas[0]:
                    token_counts = [meta.get("token_count", 0) for meta in metadatas]
                else:
                    # Estimate token counts if not stored in metadata
                    token_counts = [count_tokens(doc, CHUNK_ENCODING) for doc in documents]

                metrics["token_count_stats"] = {
                    "min": min(token_counts) if token_counts else 0,
                    "max": max(token_counts) if token_counts else 0,
                    "avg": sum(token_counts) / len(token_counts) if token_counts else 0,
                    "median": sorted(token_counts)[len(token_counts) // 2] if token_counts else 0,
                }

                # Chunk size distribution
                metrics["chunk_size_distribution"] = Counter(
                    f"{(tc // 100) * 100}-{(tc // 100 + 1) * 100}" for tc in token_counts
                )

                # Chunk overlap analysis
                if all("start_line" in meta and "end_line" in meta and "source_file" in meta for meta in metadatas):
                    overlap_counts = defaultdict(int)
                    file_chunks = defaultdict(list)

                    for meta in metadatas:
                        source_file = meta["source_file"]
                        start_line = meta["start_line"]
                        end_line = meta["end_line"]
                        file_chunks[source_file].append((start_line, end_line))

                    for source_file, chunks in file_chunks.items():
                        # Sort chunks by start line
                        chunks.sort()
                        for i in range(len(chunks) - 1):
                            current_chunk_end = chunks[i][1]
                            next_chunk_start = chunks[i + 1][0]
                            if current_chunk_end >= next_chunk_start:
                                overlap = current_chunk_end - next_chunk_start + 1
                                overlap_counts[source_file] += overlap

                    metrics["chunk_overlaps"] = dict(overlap_counts)
                    metrics["total_overlap_lines"] = sum(overlap_counts.values())

                # Files coverage
                if all("source_file" in meta for meta in metadatas):
                    file_counts = Counter(meta["source_file"] for meta in metadatas)
                    metrics["chunks_per_file"] = dict(file_counts)
                    metrics["files_covered"] = len(file_counts)

            except Exception as e:
                logging.error(f"Error analyzing chunks: {e}")

        except Exception as e:
            logging.error(f"Error evaluating chunks: {e}")

        # Store metrics
        self.metrics["chunk_metrics"] = metrics
        return metrics

    def evaluate_entities(self) -> Dict:
        """
        Evaluate the quality of extracted entities.

        Returns:
            Dictionary of entity quality metrics
        """
        if not self.chroma_client:
            logging.error("ChromaDB client not initialized")
            return {}

        logging.info("Evaluating entity quality...")
        metrics = {}

        try:
            # Get entities collection
            try:
                entities_collection = self.chroma_client.get_collection(
                    name=self.entities_collection_name,
                    embedding_function=self.ef
                )

                # Get all entities
                entities_result = entities_collection.get()
                entities_count = len(entities_result.get("ids", []))
                metadatas = entities_result.get("metadatas", [])

                metrics["total_entities"] = entities_count

                # Entity type distribution
                if all("entity_type" in meta for meta in metadatas):
                    entity_types = Counter(meta["entity_type"] for meta in metadatas)
                    metrics["entity_type_distribution"] = dict(entity_types)

                # Entity description quality analysis
                if all("description" in meta for meta in metadatas):
                    description_lengths = [len(meta.get("description", "")) for meta in metadatas]
                    metrics["description_length_stats"] = {
                        "min": min(description_lengths) if description_lengths else 0,
                        "max": max(description_lengths) if description_lengths else 0,
                        "avg": sum(description_lengths) / len(description_lengths) if description_lengths else 0,
                    }

                    # Check for default descriptions (indicating LLM was not used)
                    default_desc_count = sum(1 for meta in metadatas
                                             if meta.get("description", "").startswith("Default description for"))
                    metrics["default_descriptions_count"] = default_desc_count
                    metrics["default_descriptions_percentage"] = (
                                                                             default_desc_count / entities_count) * 100 if entities_count else 0

            except Exception as e:
                logging.error(f"Error analyzing entities: {e}")

        except Exception as e:
            logging.error(f"Error evaluating entities: {e}")

        # Store metrics
        self.metrics["entity_metrics"] = metrics
        return metrics

    def evaluate_retrieval(self, test_queries: List[str] = None) -> Dict:
        """
        Evaluate the retrieval quality using test queries.

        Args:
            test_queries: List of test queries to evaluate retrieval

        Returns:
            Dictionary of retrieval quality metrics
        """
        if not self.chroma_client or not self.neo4j_driver:
            logging.error("ChromaDB or Neo4j not initialized")
            return {}

        if not test_queries:
            test_queries = [
                "How does the chunking algorithm work?",
                "What are the main functions in the code parser?",
                "How are entities stored in Neo4j?",
                "What is the structure of the API?",
                "How do the ChromaDB collections get created?",
            ]

        logging.info(f"Evaluating retrieval quality with {len(test_queries)} test queries...")
        metrics = {"per_query_results": {}}

        try:
            # Get collections
            chunks_collection = self.chroma_client.get_collection(
                name=self.chunks_collection_name,
                embedding_function=self.ef
            )

            entities_collection = self.chroma_client.get_collection(
                name=self.entities_collection_name,
                embedding_function=self.ef
            )

            # For each query, test retrieval from both vector and graph databases
            for query in test_queries:
                query_metrics = {}

                # Test chunk retrieval
                chunk_results = chunks_collection.query(
                    query_texts=[query],
                    n_results=5
                )

                # Test entity retrieval
                entity_results = entities_collection.query(
                    query_texts=[query],
                    n_results=5
                )

                # Test graph retrieval (simulation)
                with self.neo4j_driver.session() as session:
                    # Get entities most likely related to query
                    # This is a basic simulation - not the actual retrieval logic
                    graph_query = """
                    MATCH (n:KGNode)
                    WHERE n.id STARTS WITH $index_id
                    RETURN n.id AS id, n.name AS name, n.entity_type AS entity_type
                    LIMIT 5
                    """
                    graph_results = session.run(graph_query, {"index_id": self.index_id}).data()

                # Store results for this query
                query_metrics["chunk_results"] = {
                    "ids": chunk_results.get("ids", [[]])[0],
                    "distances": chunk_results.get("distances", [[]])[0],
                    "metadatas": chunk_results.get("metadatas", [[]])[0]
                }

                query_metrics["entity_results"] = {
                    "ids": entity_results.get("ids", [[]])[0],
                    "distances": entity_results.get("distances", [[]])[0],
                    "metadatas": entity_results.get("metadatas", [[]])[0]
                }

                query_metrics["graph_results"] = graph_results

                metrics["per_query_results"][query] = query_metrics

        except Exception as e:
            logging.error(f"Error evaluating retrieval: {e}")

        # Store metrics
        self.metrics["retrieval_metrics"] = metrics
        return metrics

    def visualize_graph(self, output_file: str = None, max_nodes: int = 100):
        """
        Create a visualization of the knowledge graph.

        Args:
            output_file: Path to save the visualization
            max_nodes: Maximum number of nodes to include in visualization
        """
        if not self.neo4j_driver:
            logging.error("Neo4j driver not initialized")
            return

        logging.info("Creating graph visualization...")

        try:
            # Create NetworkX graph from Neo4j
            G = nx.DiGraph()

            with self.neo4j_driver.session() as session:
                # Get a sample of nodes and relationships
                sample_query = f"""
                MATCH (n:KGNode)
                WHERE n.id STARTS WITH '{self.index_id}'
                WITH n
                ORDER BY rand()
                LIMIT {max_nodes}
                MATCH (n)-[r]-(m:KGNode)
                WHERE m.id STARTS WITH '{self.index_id}'
                RETURN 
                    n.id AS source_id, 
                    n.name AS source_name,
                    n.entity_type AS source_type,
                    type(r) AS rel_type,
                    m.id AS target_id,
                    m.name AS target_name,
                    m.entity_type AS target_type
                LIMIT {max_nodes * 5}
                """

                result = session.run(sample_query)

                for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    rel_type = record["rel_type"]

                    # Add nodes with attributes
                    if source_id not in G:
                        G.add_node(source_id,
                                   name=record["source_name"],
                                   entity_type=record["source_type"])

                    if target_id not in G:
                        G.add_node(target_id,
                                   name=record["target_name"],
                                   entity_type=record["target_type"])

                    # Add edge
                    G.add_edge(source_id, target_id, type=rel_type)

            # Check if graph is empty
            if not G.nodes():
                logging.warning("No nodes found for visualization")
                return

            plt.figure(figsize=(15, 10))

            # Create node colors based on entity type
            entity_types = set(nx.get_node_attributes(G, 'entity_type').values())
            color_map = {}
            for i, entity_type in enumerate(entity_types):
                color_map[entity_type] = plt.cm.tab10(i % 10)

            node_colors = [color_map.get(G.nodes[node]['entity_type'], 'gray') for node in G.nodes()]

            # Create edge colors based on relationship type
            edge_types = set(nx.get_edge_attributes(G, 'type').values())
            edge_color_map = {}
            for i, edge_type in enumerate(edge_types):
                edge_color_map[edge_type] = plt.cm.Set2(i % 8)

            edge_colors = [edge_color_map.get(G.edges[edge]['type'], 'gray') for edge in G.edges()]

            # Create layout
            pos = nx.spring_layout(G, seed=42)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.8)

            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color=edge_colors, arrows=True)

            # Draw node labels (only for key nodes to avoid clutter)
            # Calculate node degrees
            degrees = dict(G.degree())
            # Get top nodes by degree
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
            top_node_ids = [node[0] for node in top_nodes]

            # Create a label dictionary only for top nodes
            labels = {}
            for node in top_node_ids:
                labels[node] = G.nodes[node].get('name', '')

            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

            # Create legend for entity types
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=entity_type)
                for entity_type, color in color_map.items()
            ]
            plt.legend(handles=legend_elements, title="Entity Types", loc="upper right")

            # Add title
            plt.title(f"Knowledge Graph for {self.index_id} (Sample of {len(G.nodes())} nodes)")
            plt.axis('off')

            # Save or show
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logging.info(f"Saved graph visualization to {output_file}")
            else:
                plt.show()

        except Exception as e:
            logging.error(f"Error visualizing graph: {e}")

    def visualize_chunk_metrics(self, output_file: str = None):
        """
        Create visualizations for chunk metrics.

        Args:
            output_file: Path to save the visualization
        """
        if not self.metrics.get("chunk_metrics"):
            logging.error("Chunk metrics not available. Run evaluate_chunks() first.")
            return

        chunk_metrics = self.metrics["chunk_metrics"]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Chunk Metrics for {self.index_id}", fontsize=16)

        # Plot token count distribution
        if "token_count_stats" in chunk_metrics:
            axes[0, 0].bar(
                ["Min", "Max", "Avg", "Median"],
                [
                    chunk_metrics["token_count_stats"]["min"],
                    chunk_metrics["token_count_stats"]["max"],
                    chunk_metrics["token_count_stats"]["avg"],
                    chunk_metrics["token_count_stats"]["median"]
                ]
            )
            axes[0, 0].set_title("Token Count Statistics")
            axes[0, 0].set_ylabel("Token Count")

        # Plot chunk size distribution
        if "chunk_size_distribution" in chunk_metrics:
            sizes = []
            counts = []
            for size_range, count in sorted(chunk_metrics["chunk_size_distribution"].items()):
                sizes.append(size_range)
                counts.append(count)

            axes[0, 1].bar(sizes, counts)
            axes[0, 1].set_title("Chunk Size Distribution")
            axes[0, 1].set_xlabel("Token Range")
            axes[0, 1].set_ylabel("Number of Chunks")
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot chunks per file (top 10)
        if "chunks_per_file" in chunk_metrics:
            files = []
            counts = []
            for file, count in sorted(chunk_metrics["chunks_per_file"].items(), key=lambda x: x[1], reverse=True)[:10]:
                files.append(os.path.basename(file))
                counts.append(count)

            axes[1, 0].bar(files, counts)
            axes[1, 0].set_title("Chunks per File (Top 10)")
            axes[1, 0].set_xlabel("File")
            axes[1, 0].set_ylabel("Number of Chunks")
            axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot overlaps per file
        if "chunk_overlaps" in chunk_metrics:
            files = []
            overlaps = []
            for file, overlap in sorted(chunk_metrics["chunk_overlaps"].items(), key=lambda x: x[1], reverse=True)[:10]:
                files.append(os.path.basename(file))
                overlaps.append(overlap)

            axes[1, 1].bar(files, overlaps)
            axes[1, 1].set_title("Line Overlaps per File (Top 10)")
            axes[1, 1].set_xlabel("File")
            axes[1, 1].set_ylabel("Number of Overlapping Lines")
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logging.info(f"Saved chunk metrics visualization to {output_file}")
        else:
            plt.show()

    def generate_report(self, output_file: str = None):
        """
        Generate a comprehensive evaluation report.

        Args:
            output_file: Path to save the report (JSON format)
        """
        report = {
            "index_id": self.index_id,
            "evaluation_timestamp": datetime.datetime.now().isoformat(),
            "metadata": self.metadata,
            "metrics": self.metrics,
            "summary": self._generate_summary()
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logging.info(f"Saved evaluation report to {output_file}")

        return report

    def generate_summary_file(self, output_file: str = None):
        """
        Generate a human-readable summary evaluation file.

        Args:
            output_file: Path to save the summary file (text format)
        """
        if not output_file:
            output_file = f"{self.index_id}_evaluation_summary.txt"

        summary = self._generate_summary()

        try:
            with open(output_file, 'w') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write(f"GRAPHLOOM EVALUATION SUMMARY: {self.index_id}\n")
                f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                # Overall assessment
                f.write("OVERALL ASSESSMENT\n")
                f.write("-" * 50 + "\n")
                if summary.get("overall_score") is not None:
                    f.write(f"Overall Score: {summary.get('overall_score', 0):.1f}/100\n")
                    f.write(f"Assessment: {summary.get('overall_assessment', 'N/A')}\n\n")
                else:
                    f.write("Overall Score: Not available (insufficient data)\n\n")

                # Graph metrics summary
                f.write("GRAPH METRICS\n")
                f.write("-" * 50 + "\n")
                gm = self.metrics.get("graph_metrics", {})
                f.write(f"Total Nodes: {gm.get('total_nodes', 'N/A')}\n")
                f.write(f"Total Relationships: {gm.get('total_relationships', 'N/A')}\n")
                f.write(f"Graph Density: {gm.get('graph_density', 'N/A'):.6f}\n")
                f.write(f"Connected Components: {gm.get('connected_components', 'N/A')}\n\n")

                # Node distribution
                if gm.get("node_counts_by_type"):
                    f.write("Node Type Distribution:\n")
                    for node_type, count in sorted(gm.get("node_counts_by_type", {}).items(), key=lambda x: x[1],
                                                   reverse=True)[:10]:
                        f.write(f"  - {node_type}: {count}\n")
                    f.write("\n")

                # Relationship distribution
                if gm.get("relationship_counts_by_type"):
                    f.write("Relationship Type Distribution:\n")
                    for rel_type, count in sorted(gm.get("relationship_counts_by_type", {}).items(), key=lambda x: x[1],
                                                  reverse=True)[:10]:
                        f.write(f"  - {rel_type}: {count}\n")
                    f.write("\n")

                # Chunk metrics summary
                f.write("CHUNK METRICS\n")
                f.write("-" * 50 + "\n")
                cm = self.metrics.get("chunk_metrics", {})
                f.write(f"Total Chunks: {cm.get('total_chunks', 'N/A')}\n")

                if cm.get("token_count_stats"):
                    tcs = cm.get("token_count_stats")
                    f.write(
                        f"Token Utilization: {summary.get('chunk_quality', {}).get('token_utilization', 'N/A'):.2f}\n")
                    f.write(
                        f"Token Count Stats: Min={tcs.get('min', 'N/A')}, Max={tcs.get('max', 'N/A')}, Avg={tcs.get('avg', 'N/A'):.1f}\n")

                if cm.get("total_overlap_lines"):
                    f.write(f"Total Overlap Lines: {cm.get('total_overlap_lines', 'N/A')}\n")

                f.write(f"Files Covered: {cm.get('files_covered', 'N/A')}\n\n")

                # Entity metrics summary
                f.write("ENTITY METRICS\n")
                f.write("-" * 50 + "\n")
                em = self.metrics.get("entity_metrics", {})
                f.write(f"Total Entities: {em.get('total_entities', 'N/A')}\n")

                if em.get("entity_type_distribution"):
                    f.write("Entity Type Distribution:\n")
                    for entity_type, count in sorted(em.get("entity_type_distribution", {}).items(), key=lambda x: x[1],
                                                     reverse=True):
                        f.write(f"  - {entity_type}: {count}\n")

                if em.get("default_descriptions_percentage") is not None:
                    f.write(f"\nDefault Descriptions: {em.get('default_descriptions_percentage', 'N/A'):.1f}%\n")
                    f.write(
                        f"Description Assessment: {summary.get('entity_quality', {}).get('description_assessment', 'N/A')}\n\n")

                # Most important/central nodes
                f.write("KEY GRAPH ELEMENTS\n")
                f.write("-" * 50 + "\n")

                # Top hub nodes
                if gm.get("top_hub_nodes"):
                    f.write("Most Connected Nodes:\n")
                    for i, node in enumerate(gm.get("top_hub_nodes", [])[:5], 1):
                        f.write(
                            f"  {i}. {node.get('name', 'Unnamed')} ({node.get('type', 'Unknown')}) - {node.get('degree', 0)} connections\n")
                    f.write("\n")

                # Top central nodes
                if gm.get("top_central_nodes"):
                    f.write("Most Central Nodes:\n")
                    for i, node in enumerate(gm.get("top_central_nodes", [])[:5], 1):
                        f.write(
                            f"  {i}. {node.get('name', 'Unnamed')} ({node.get('type', 'Unknown')}) - Centrality: {node.get('centrality', 0)}\n")
                    f.write("\n")

                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 50 + "\n")
                if summary.get("recommendations"):
                    for i, rec in enumerate(summary.get("recommendations", []), 1):
                        f.write(f"{i}. {rec}\n")
                else:
                    f.write("No specific recommendations available.\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("End of Summary Report\n")

            logging.info(f"Summary evaluation file created at: {output_file}")
            return output_file

        except Exception as e:
            logging.error(f"Error creating summary file: {e}")
            return None

    def _generate_summary(self) -> Dict:
        """Generate a summary of evaluation findings."""
        summary = {
            "graph_quality": {},
            "chunk_quality": {},
            "entity_quality": {},
            "retrieval_quality": {},
            "overall_score": None,
            "recommendations": []
        }

        # Graph quality summary
        if self.metrics.get("graph_metrics"):
            graph_metrics = self.metrics["graph_metrics"]

            if graph_metrics.get("total_nodes", 0) > 0:
                summary["graph_quality"]["node_count"] = graph_metrics.get("total_nodes", 0)
                summary["graph_quality"]["relationship_count"] = graph_metrics.get("total_relationships", 0)
                summary["graph_quality"]["density"] = graph_metrics.get("graph_density", 0)

                # Calculate a basic graph quality score
                nodes = graph_metrics.get("total_nodes", 0)
                rels = graph_metrics.get("total_relationships", 0)
                if nodes > 0:
                    avg_connections = rels / nodes
                    summary["graph_quality"]["avg_connections_per_node"] = avg_connections

                    if avg_connections < 1:
                        summary["graph_quality"]["assessment"] = "Poor - Few connections between nodes"
                        summary["recommendations"].append(
                            "Improve entity relationship extraction to create more connections")
                    elif avg_connections < 3:
                        summary["graph_quality"]["assessment"] = "Fair - Graph has basic connectivity"
                        summary["recommendations"].append("Consider adding more relationship types between entities")
                    else:
                        summary["graph_quality"]["assessment"] = "Good - Graph is well-connected"

        # Chunk quality summary
        if self.metrics.get("chunk_metrics"):
            chunk_metrics = self.metrics["chunk_metrics"]

            if chunk_metrics.get("total_chunks", 0) > 0:
                summary["chunk_quality"]["chunk_count"] = chunk_metrics.get("total_chunks", 0)

                # Token utilization
                if chunk_metrics.get("token_count_stats"):
                    avg_tokens = chunk_metrics["token_count_stats"].get("avg", 0)
                    max_tokens = chunk_metrics["token_count_stats"].get("max", 0)
                    token_limit = 1050  # Default from config

                    token_utilization = avg_tokens / token_limit if token_limit > 0 else 0
                    summary["chunk_quality"]["token_utilization"] = token_utilization

                    if token_utilization < 0.5:
                        summary["chunk_quality"][
                            "assessment"] = "Poor - Chunks are significantly smaller than token limit"
                        summary["recommendations"].append("Adjust chunking parameters to create larger chunks")
                    elif token_utilization < 0.7:
                        summary["chunk_quality"]["assessment"] = "Fair - Chunks are somewhat smaller than optimal"
                        summary["recommendations"].append("Consider fine-tuning chunk size parameters")
                    else:
                        summary["chunk_quality"]["assessment"] = "Good - Chunks are well-sized relative to token limit"

                # Chunk overlaps
                if "total_overlap_lines" in chunk_metrics:
                    summary["chunk_quality"]["total_overlap_lines"] = chunk_metrics["total_overlap_lines"]
                    if chunk_metrics["total_overlap_lines"] > 100:
                        summary["recommendations"].append(
                            "High chunk overlap detected - consider adjusting chunking algorithm")

        # Entity quality summary
        if self.metrics.get("entity_metrics"):
            entity_metrics = self.metrics["entity_metrics"]

            if entity_metrics.get("total_entities", 0) > 0:
                summary["entity_quality"]["entity_count"] = entity_metrics.get("total_entities", 0)

                # Entity type distribution
                if entity_metrics.get("entity_type_distribution"):
                    summary["entity_quality"]["entity_types"] = entity_metrics["entity_type_distribution"]

                # Description quality
                if "default_descriptions_percentage" in entity_metrics:
                    default_pct = entity_metrics["default_descriptions_percentage"]
                    summary["entity_quality"]["default_descriptions_percentage"] = default_pct

                    if default_pct > 80:
                        summary["entity_quality"][
                            "description_assessment"] = "Poor - Most entities have default descriptions"
                        summary["recommendations"].append("Enable LLM for generating entity descriptions")
                    elif default_pct > 20:
                        summary["entity_quality"][
                            "description_assessment"] = "Fair - Some entities have custom descriptions"
                        summary["recommendations"].append("Improve LLM description quality or coverage")
                    else:
                        summary["entity_quality"][
                            "description_assessment"] = "Good - Most entities have custom descriptions"

        # Retrieval quality summary
        if self.metrics.get("retrieval_metrics") and self.metrics["retrieval_metrics"].get("per_query_results"):
            summary["retrieval_quality"]["queries_tested"] = len(self.metrics["retrieval_metrics"]["per_query_results"])
            summary["retrieval_quality"][
                "assessment"] = "Retrieval quality assessment would require manual evaluation of results"

        # Generate overall score (0-100)
        scores = []

        # Graph score (0-25)
        if "graph_quality" in summary and "avg_connections_per_node" in summary["graph_quality"]:
            avg_conn = summary["graph_quality"]["avg_connections_per_node"]
            graph_score = min(25, max(0, avg_conn * 8))  # Scale: 3+ connections = full score
            scores.append(graph_score)

        # Chunk score (0-25)
        if "chunk_quality" in summary and "token_utilization" in summary["chunk_quality"]:
            token_util = summary["chunk_quality"]["token_utilization"]
            chunk_score = min(25, max(0, token_util * 36 - 1))  # Scale: 0.7+ utilization = full score
            scores.append(chunk_score)

        # Entity score (0-25)
        if "entity_quality" in summary and "default_descriptions_percentage" in summary["entity_quality"]:
            default_pct = summary["entity_quality"]["default_descriptions_percentage"]
            entity_score = min(25, max(0, 25 * (1 - default_pct / 100)))  # Scale: 0% default = full score
            scores.append(entity_score)

        # Connectivity score (0-25)
        if "graph_quality" in summary and "density" in summary["graph_quality"]:
            density = summary["graph_quality"]["density"]
            # Scale for expected graph density (knowledge graphs are typically sparse)
            connectivity_score = min(25, max(0, density * 25000))  # Scale: density of 0.001+ = full score
            scores.append(connectivity_score)

        # Calculate overall score if we have component scores
        if scores:
            summary["overall_score"] = sum(scores) / len(scores) * (100 / 25)  # Scale to 0-100

            # Add overall assessment
            if summary["overall_score"] < 40:
                summary["overall_assessment"] = "Poor - Significant improvements needed"
            elif summary["overall_score"] < 70:
                summary["overall_assessment"] = "Fair - Graph functions but could be optimized"
            else:
                summary["overall_assessment"] = "Good - Graph is well-structured and functional"

        return summary


def run_evaluation(index_id: str, output_dir: str = "evaluation_results", visualize: bool = True):
    """
    Run a full evaluation of a GraphLoom index and save results.

    Args:
        index_id: The unique identifier for the graph/index to evaluate
        output_dir: Directory to save evaluation results
        visualize: Whether to generate visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = GraphLoomEvaluator(index_id=index_id, verbose=True)

    print(f"\n{'=' * 80}\nEVALUATING GRAPHLOOM INDEX: {index_id}\n{'=' * 80}\n")

    # Run evaluations
    print("\n[1/4] Evaluating graph structure...")
    graph_metrics = evaluator.evaluate_graph()
    print(
        f"     Found {graph_metrics.get('total_nodes', 'N/A')} nodes and {graph_metrics.get('total_relationships', 'N/A')} relationships")

    print("\n[2/4] Evaluating chunk quality...")
    chunk_metrics = evaluator.evaluate_chunks()
    print(f"     Analyzed {chunk_metrics.get('total_chunks', 'N/A')} chunks")

    print("\n[3/4] Evaluating entity quality...")
    entity_metrics = evaluator.evaluate_entities()
    print(f"     Analyzed {entity_metrics.get('total_entities', 'N/A')} entities")

    print("\n[4/4] Evaluating retrieval performance...")
    test_queries = [
        "How does the chunking algorithm work?",
        "What are the main functions in the code parser?",
        "How are entities stored in Neo4j?",
        "What is the structure of the API?",
        "How do the ChromaDB collections get created?",
    ]
    evaluator.evaluate_retrieval(test_queries=test_queries)
    print("     Tested 5 sample queries")

    # Generate reports
    print("\nGenerating evaluation reports...")

    # Detailed JSON report
    report_path = os.path.join(output_dir, f"{index_id}_evaluation_report.json")
    evaluator.generate_report(output_file=report_path)
    print(f"  - Detailed JSON report saved to: {report_path}")

    # Human-readable summary report
    summary_path = os.path.join(output_dir, f"{index_id}_evaluation_summary.txt")
    evaluator.generate_summary_file(output_file=summary_path)
    print(f"  - Summary report saved to: {summary_path}")

    # Generate visualizations
    if visualize:
        print("\nGenerating visualizations...")

        graph_viz_path = os.path.join(output_dir, f"{index_id}_graph_visualization.png")
        evaluator.visualize_graph(output_file=graph_viz_path)
        print(f"  - Graph visualization saved to: {graph_viz_path}")

        chunks_viz_path = os.path.join(output_dir, f"{index_id}_chunk_metrics.png")
        evaluator.visualize_chunk_metrics(output_file=chunks_viz_path)
        print(f"  - Chunk metrics visualization saved to: {chunks_viz_path}")

    # Get summary and print overall score
    summary = evaluator._generate_summary()
    if summary.get("overall_score") is not None:
        print(f"\nOVERALL EVALUATION SCORE: {summary.get('overall_score', 0):.1f}/100")
        print(f"ASSESSMENT: {summary.get('overall_assessment', 'N/A')}")

    print(f"\nEvaluation complete. All results saved to: {output_dir}")

    return evaluator.metrics


def main():
    """
    Main function that runs evaluation with default parameters when script is executed directly.
    For command-line usage, the argparse functionality is maintained.
    """
    # Check if any command line args were provided
    import sys
    if len(sys.argv) > 1:
        # Use argparse for command-line arguments
        parser = argparse.ArgumentParser(description="Evaluate GraphLoom knowledge graph and chunking quality")
        parser.add_argument("--index-id", required=True, help="GraphLoom index ID to evaluate")
        parser.add_argument("--output-dir", default="evaluation_results", help="Directory for saving results")
        parser.add_argument("--no-visualize", action="store_true", help="Skip visualization generation")

        args = parser.parse_args()

        run_evaluation(
            index_id=args.index_id,
            output_dir=args.output_dir,
            visualize=not args.no_visualize
        )
    else:
        # No command-line args provided, run with default parameters
        # Get available index IDs from metadata files or folders
        import glob

        # Look for metadata files
        metadata_files = glob.glob("*_metadata.json")
        # Look for ChromaDB folders
        chroma_folders = [f for f in glob.glob(f"{CHROMA_PATH_PREFIX}*") if os.path.isdir(f)]

        available_indices = []

        # Extract index IDs from metadata files
        for mf in metadata_files:
            index_id = mf.replace("_metadata.json", "")
            available_indices.append(index_id)

        # Extract index IDs from ChromaDB folders
        for cf in chroma_folders:
            index_id = cf.replace(CHROMA_PATH_PREFIX, "")
            if index_id not in available_indices:
                available_indices.append(index_id)

        if not available_indices:
            # Check if there's a graphs.json file from the API
            if os.path.exists("graphs.json"):
                try:
                    with open("graphs.json", "r") as f:
                        graphs_data = json.load(f)
                    for graph in graphs_data:
                        if "graph_id" in graph:
                            available_indices.append(graph["graph_id"])
                except Exception as e:
                    logging.error(f"Error reading graphs.json: {e}")

        if available_indices:
            # Sort indices by creation time (if possible)
            index_times = {}
            for idx in available_indices:
                meta_file = f"{idx}_metadata.json"
                if os.path.exists(meta_file):
                    index_times[idx] = os.path.getmtime(meta_file)
                else:
                    chroma_path = f"{CHROMA_PATH_PREFIX}{idx}"
                    if os.path.exists(chroma_path):
                        index_times[idx] = os.path.getmtime(chroma_path)
                    else:
                        index_times[idx] = 0

            # Sort by most recently modified
            sorted_indices = sorted(available_indices, key=lambda x: index_times.get(x, 0), reverse=True)

            # Use the most recent index
            index_id = sorted_indices[0]
            logging.info(f"Found {len(available_indices)} indices. Using most recent: {index_id}")

            run_evaluation(
                index_id=index_id,
                output_dir="evaluation_results",
                visualize=True
            )
        else:
            logging.error("No GraphLoom indices found. Please specify an index ID with --index-id.")
            print("\nNo GraphLoom indices found. Please create a graph first or specify an index ID.")
            print("Example usage:")
            print("  python evaluate.py --index-id your_index_id")

            # Offer to create a test graph
            print("\nWould you like to run the main processing pipeline to create a test graph? (y/n)")
            response = input().strip().lower()
            if response == 'y':
                # Import and run the main processing pipeline
                try:
                    from main import main_processing_pipeline
                    test_index_id = "test_graph_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder_path = input("Enter the folder path containing Python files to analyze: ").strip()
                    if os.path.isdir(folder_path):
                        print(f"Creating test graph with ID: {test_index_id}")
                        main_processing_pipeline(folder_path, index_id=test_index_id, enable_llm_description=False)
                        print(f"Graph created. Now running evaluation...")
                        run_evaluation(
                            index_id=test_index_id,
                            output_dir="evaluation_results",
                            visualize=True
                        )
                    else:
                        print(f"Invalid folder path: {folder_path}")
                except ImportError:
                    print("Could not import main_processing_pipeline. Make sure main.py is in the current directory.")
                except Exception as e:
                    print(f"Error creating test graph: {e}")


if __name__ == "__main__":
    main()
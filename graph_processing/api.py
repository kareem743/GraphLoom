from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import json
from datetime import datetime
import logging
import chromadb
from neo4j import GraphDatabase
from chromadb.utils import embedding_functions
from llm_client import get_fallback_llm, get_llm_client
from main import main_processing_pipeline
from retrive import analyze_change_impact, analyze_code_patterns, classify_query_intent, identify_change_points, retrieve_context, generate_response, LLM_CLIENT
from config import EMBEDDING_MODEL_NAME, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, CHROMA_PATH_PREFIX
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:8000"],  # Allow requests from Angular app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# JSON file for storing graph and conversation metadata
GRAPHS_FILE = "graphs.json"
CONVERSATIONS_FILE = "conversations.json"

# Initialize JSON files if they don't exist
def init_storage():
    if not os.path.exists(GRAPHS_FILE):
        with open(GRAPHS_FILE, 'w') as f:
            json.dump([], f)
    if not os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, 'w') as f:
            json.dump([], f)

init_storage()

# Pydantic models
class CreateGraphRequest(BaseModel):
    user_id: str
    graph_name: str
    folder_path: str

class QueryRequest(BaseModel):
    conversation_id: str
    query: str

class NewChatRequest(BaseModel):
    user_id: str
    graph_id: str
    query: str

# Helper functions
def generate_unique_id(prefix: str) -> str:
    import uuid
    return f"{prefix}{uuid.uuid4()}"

def read_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def write_json(file_path: str, data: List[Dict]):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# Initialize embedding function
try:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
except Exception as e:
    logging.critical(f"Failed to initialize embedding function: {e}")
    ef = None

# Initialize Neo4j driver
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
except Exception as e:
    logging.critical(f"Failed to connect to Neo4j: {e}")
    neo4j_driver = None

# APIs
@app.post("/graphs/create")
async def create_graph(request: CreateGraphRequest):

    # Generate unique index_id
    index_id = generate_unique_id("graph_")
    chroma_db_path = f"{CHROMA_PATH_PREFIX}{index_id}"
    
    # Run pipeline
    try:
        main_processing_pipeline(request.folder_path, index_id=index_id, enable_llm_description=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create graph: {str(e)}")
    
    # Store metadata
    graph_data = {
        "graph_id": index_id,
        "name": request.graph_name,
        "folder_path": request.folder_path,
        "user_id": request.user_id,
        "created_at": datetime.utcnow()
    }
    graphs = read_json(GRAPHS_FILE)
    graphs.append(graph_data)
    write_json(GRAPHS_FILE, graphs)
    
    return {"graph_id": index_id, "message": f"Graph '{request.graph_name}' created successfully"}

@app.get("/graphs/")
async def get_graphs(user_id: str):
    graphs = read_json(GRAPHS_FILE)
    user_graphs = [g for g in graphs if g["user_id"] == user_id]
    return {"graphs": user_graphs}

@app.get("/graphs/{graph_id}")
async def get_graph_by_id(user_id: str, graph_id: str):
    graphs = read_json(GRAPHS_FILE)
    graph = next((g for g in graphs if g["graph_id"] == graph_id and g["user_id"] == user_id), None)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found or user does not have access")
    return graph

@app.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str, user_id: str):
    graphs = read_json(GRAPHS_FILE)
    graph = next((g for g in graphs if g["graph_id"] == graph_id and g["user_id"] == user_id), None)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found or user does not have access")
    
    # Delete ChromaDB collections
    chroma_db_path = f"{CHROMA_PATH_PREFIX}{graph_id}"
    try:
        if os.path.exists(chroma_db_path):
            import shutil
            shutil.rmtree(chroma_db_path)
    except Exception as e:
        logging.error(f"Failed to delete ChromaDB folder {chroma_db_path}: {str(e)}")
    
    # Delete Neo4j data
    if neo4j_driver:
        try:
            with neo4j_driver.session(database="neo4j") as session:
                session.run("MATCH (n:KGNode) WHERE n.id STARTS WITH $graph_id DETACH DELETE n", {"graph_id": graph_id})
        except Exception as e:
            logging.error(f"Failed to delete Neo4j data for graph {graph_id}: {str(e)}")
    
    # Delete conversations
    conversations = read_json(CONVERSATIONS_FILE)
    conversations = [c for c in conversations if c["graph_id"] != graph_id]
    write_json(CONVERSATIONS_FILE, conversations)
    
    # Remove graph metadata
    graphs = [g for g in graphs if g["graph_id"] != graph_id]
    write_json(GRAPHS_FILE, graphs)
    
    return {"message": f"Graph '{graph['name']}' deleted successfully"}

@app.post("/conversations/start-chat")
async def start_chat(request: NewChatRequest):
    graphs = read_json(GRAPHS_FILE)
    graph = next((g for g in graphs if g["graph_id"] == request.graph_id and g["user_id"] == request.user_id), None)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found or user does not have access")
    
    if not ef or not neo4j_driver:
        raise HTTPException(status_code=500, detail="Backend services not initialized")
    
    # Initialize ChromaDB client
    chroma_db_path = f"{CHROMA_PATH_PREFIX}{request.graph_id}"
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        chroma_client.heartbeat()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize ChromaDB: {str(e)}")
    
    try:
        LLM_CLIENT = get_llm_client()
        logging.info("Successfully initialized LLM client.")
    except Exception as e:
        logging.warning(f"Failed to initialize LLM client: {e}. Using fallback LLM.")
        LLM_CLIENT = get_fallback_llm()

    query_intent = classify_query_intent(request.query, LLM_CLIENT)
    # Retrieve context and generate response
    try:
        context = retrieve_context(
            query=request.query,
            index_id=request.graph_id,
            chroma_client=chroma_client,
            neo4j_driver=neo4j_driver,
            embedding_function=ef,
            llm_client=LLM_CLIENT,
            query_intent=query_intent,
            top_k_entities=15,
            top_k_chunks=5,
            graph_hops=3
        )
        
        if query_intent in ["change_request", "function_request"]:
            code_patterns = analyze_code_patterns(context["relevant_chunks"], neo4j_driver)
            context["code_patterns"] = code_patterns
            change_points = identify_change_points(request.query, context, neo4j_driver)
            context["change_points"] = change_points
            impact_analysis = analyze_change_impact(change_points, neo4j_driver)
            context["impact_analysis"] = impact_analysis

            response = generate_response(
                query=request.query,
                retrieved_context=context,
                llm_client=LLM_CLIENT,
                query_intent=query_intent
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
    # Store conversation
    conversation_id = generate_unique_id("conv_")
    conversation = {
        "conversation_id": conversation_id,
        "graph_id": request.graph_id,
        "user_id": request.user_id,
        "name": "new chat",
        "history": [
            {
                "user": request.query,
                "assistant": response,
                "timestamp": datetime.utcnow()
            }
        ],
        "created_at": datetime.utcnow()
    }
    conversations = read_json(CONVERSATIONS_FILE)
    conversations.append(conversation)
    write_json(CONVERSATIONS_FILE, conversations)
    
    return {
        "conversation_id": conversation_id,
        "conversation_name": "new chat",
        "response": response
    }

@app.post("/conversations/send-message")
async def send_message(request: QueryRequest):
    conversations = read_json(CONVERSATIONS_FILE)
    conversation = next((c for c in conversations if c["conversation_id"] == request.conversation_id), None)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    graphs = read_json(GRAPHS_FILE)
    graph = next((g for g in graphs if g["graph_id"] == conversation["graph_id"] and g["user_id"] == conversation["user_id"]), None)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found or user does not have access")
    
    if not ef or not neo4j_driver:
        raise HTTPException(status_code=500, detail="Backend services not initialized")
    
    # Initialize ChromaDB client
    chroma_db_path = f"{CHROMA_PATH_PREFIX}{conversation['graph_id']}"
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        chroma_client.heartbeat()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize ChromaDB: {str(e)}")
    
    try:
        LLM_CLIENT = get_llm_client()
        logging.info("Successfully initialized LLM client.")
    except Exception as e:
        logging.warning(f"Failed to initialize LLM client: {e}. Using fallback LLM.")
        LLM_CLIENT = get_fallback_llm()
    
    # Retrieve context and generate response
    try:
        query_intent = classify_query_intent(request.query, LLM_CLIENT)        
        context = retrieve_context(
            query=request.query,
            index_id=conversation["graph_id"],
            chroma_client=chroma_client,
            neo4j_driver=neo4j_driver,
            embedding_function=ef,
            llm_client=LLM_CLIENT,
            query_intent=query_intent,
            top_k_entities=15,
            top_k_chunks=5,
            graph_hops=3
        )

        if query_intent in ["change_request", "function_request"]:
            code_patterns = analyze_code_patterns(context["relevant_chunks"], neo4j_driver)
            context["code_patterns"] = code_patterns
            change_points = identify_change_points(request.query, context, neo4j_driver)
            context["change_points"] = change_points
            impact_analysis = analyze_change_impact(change_points, neo4j_driver)
            context["impact_analysis"] = impact_analysis

        response = generate_response(
            query=request.query,
            retrieved_context=context,
            llm_client=LLM_CLIENT,
            query_intent=query_intent
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
    # Update conversation history
    new_history_entry = {
        "user": request.query,
        "assistant": response,
        "timestamp": datetime.utcnow()
    }
    conversation["history"].append(new_history_entry)
    write_json(CONVERSATIONS_FILE, conversations)
    
    return {"response": response}

@app.get("/conversations/")
async def get_conversations(user_id: str, graph_id: str):
    graphs = read_json(GRAPHS_FILE)
    if not any(g for g in graphs if g["graph_id"] == graph_id and g["user_id"] == user_id):
        raise HTTPException(status_code=404, detail="Graph not found or user does not have access")
    
    conversations = read_json(CONVERSATIONS_FILE)
    user_conversations = [
        {"conversation_id": c["conversation_id"], "name": c["name"], "created_at": c["created_at"]}
        for c in conversations
        if c["user_id"] == user_id and c["graph_id"] == graph_id
    ]
    return {"conversations": user_conversations}

@app.get("/conversations/{conversation_id}/history/")
async def get_conversation_history(conversation_id: str):
    conversations = read_json(CONVERSATIONS_FILE)
    conversation = next((c for c in conversations if c["conversation_id"] == conversation_id), None)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    history = conversation.get("history", [])
    if not history:
        raise HTTPException(status_code=404, detail="No messages found in this conversation")
    
    sorted_history = sorted(history, key=lambda x: x.get("timestamp", datetime.min))
    return {"history": sorted_history}

class GraphVisualizationResponse(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]

@app.get("/graphs/{graph_id}/visualization", response_model=GraphVisualizationResponse)
async def get_graph_visualization(user_id: str, graph_id: str):
    # Validate graph access
    graphs = read_json(GRAPHS_FILE)
    graph = next((g for g in graphs if g["graph_id"] == graph_id and g["user_id"] == user_id), None)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found or user does not have access")
    
    if not neo4j_driver:
        raise HTTPException(status_code=500, detail="Neo4j service not initialized")
    
    # Fetch nodes and edges from Neo4j
    try:
        with neo4j_driver.session(database="neo4j") as session:
            # Fetch nodes
            node_query = """
            MATCH (n:KGNode)
            WHERE n.graph_id = $graph_id
            RETURN n.id AS id, n.entity_type AS label, n.name AS name, n.source_file AS source_file, n.description AS description
            LIMIT 500
            """
            node_result = session.run(node_query, graph_id=graph_id)
            nodes = [
                {
                    "id": record["id"],
                    "label": f"{record['name']}\n({record['label']})",
                    "title": f"Type: {record['label']}\nName: {record['name']}\nFile: {record['source_file'] or 'N/A'}\nDescription: {record['description'] or 'N/A'}"
                }
                for record in node_result
            ]
            
            # Fetch edges
            edge_query = """
            MATCH (n:KGNode)-[r]->(m:KGNode)
            WHERE n.graph_id = $graph_id AND m.graph_id = $graph_id
            RETURN n.id AS from, m.id AS to, type(r) AS type
            LIMIT 500
            """
            edge_result = session.run(edge_query, graph_id=graph_id)
            edges = [
                {
                    "from": record["from"],
                    "to": record["to"],
                    "label": record["type"],
                    "title": f"Relationship: {record['type']}"
                }
                for record in edge_result
            ]
        
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph visualization data: {str(e)}")

@app.get("/")
def root():
    return {"message": "GraphLoom backend is running!"}

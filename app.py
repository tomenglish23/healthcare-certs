"""
Modern RAG System - Clean Architecture
- MarkdownHeaderTextSplitter for intelligent chunking
- Pure vector search (no complex filtering)
- Simple 3-node graph
- Config-driven, works with any markdown dataset
"""
from __future__ import annotations

import os
from typing import TypedDict, List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.graph import StateGraph, END


# ============================================================
# CONFIGURATION
# ============================================================

def load_config() -> Dict[str, Any]:
    """Load config from YAML or use defaults"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'product': {'name': 'healthcare-certs', 'version': '1.0.0'},
            'branding': {'title': 'Healthcare Certifications'},
            'data': {'source_file': 'healthcare-certs-all.md'},
            'features': {'show_confidence': True, 'show_sources': True}
        }

CONFIG = load_config()

# Shortcuts
PRODUCT_NAME = CONFIG['product']['name']
PRODUCT_VERSION = CONFIG['product']['version']
DATA_FILE = CONFIG['data']['source_file']

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

print(f"[*] {PRODUCT_NAME} v{PRODUCT_VERSION}")
print(f"[*] Data: {DATA_FILE}")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

# Globals
vector_store = None
app_graph = None


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_documents() -> List[Document]:
    """Load and chunk markdown files using MarkdownHeaderTextSplitter"""
    
    # Define headers to split on
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    # Read markdown file
    filepath = os.path.join("./data", DATA_FILE)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into chunks
    docs = splitter.split_text(content)
    
    # Add source to metadata
    for doc in docs:
        doc.metadata['source'] = DATA_FILE
    
    print(f"[*] Loaded {len(docs)} chunks from {DATA_FILE}")
    return docs


# ============================================================
# VECTOR STORE
# ============================================================

def create_vectorstore(docs: List[Document]) -> Chroma:
    """Create or load ChromaDB vectorstore"""
    
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    persist_dir = "./chroma_db"
    
    if os.path.exists(persist_dir):
        print("[*] Loading existing vectorstore")
        vs = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print(f"[*] Creating vectorstore with {len(docs)} documents")
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    
    return vs


# ============================================================
# RAG WORKFLOW
# ============================================================

class RAGState(TypedDict):
    """Clean state - just what we need"""
    question: str
    docs: List[Document]
    context: str
    answer: str
    confidence: float
    sources: List[str]


def create_rag_graph(vs: Chroma):
    """Build simple 3-node RAG graph"""
    
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    
    def retrieve(state: RAGState) -> RAGState:
        """Retrieve relevant documents via vector search"""
        docs = vs.similarity_search(state["question"], k=4)
        
        state["docs"] = docs
        state["context"] = "\n\n".join([doc.page_content for doc in docs])
        state["sources"] = [doc.metadata.get("source", "unknown") for doc in docs]
        
        print(f"[*] Retrieved {len(docs)} documents")
        return state
    
    def grade(state: RAGState) -> RAGState:
        """Calculate confidence based on context relevance"""
        
        if not state["docs"] or not state["context"]:
            state["confidence"] = 0.0
            return state
        
        # Simple word overlap confidence
        q_words = set(state["question"].lower().split())
        ctx_words = set(state["context"].lower().split())
        
        if not q_words:
            state["confidence"] = 0.0
            return state
        
        overlap = len(q_words & ctx_words)
        confidence = min(overlap / len(q_words), 1.0)
        state["confidence"] = round(confidence, 2)
        
        print(f"[*] Confidence: {state['confidence']}")
        return state
    
    def generate(state: RAGState) -> RAGState:
        """Generate answer from context"""
        
        if state["confidence"] < 0.1:
            state["answer"] = "I don't have enough relevant information to answer that question."
            return state
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer based ONLY on the provided context. "
                      "Be concise and accurate."),
            ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "context": state["context"],
            "question": state["question"]
        })
        
        state["answer"] = response.content
        return state
    
    # Build graph
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("generate", generate)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# ============================================================
# INITIALIZATION
# ============================================================

def initialize():
    """Initialize system on startup"""
    global vector_store, app_graph
    
    print("=" * 50)
    print("Initializing...")
    print("=" * 50)
    
    # Load documents
    docs = load_documents()
    
    # Create vectorstore
    vector_store = create_vectorstore(docs)
    
    # Build RAG graph
    app_graph = create_rag_graph(vector_store)
    
    print("[*] System ready!")
    return True


# ============================================================
# API ROUTES
# ============================================================

@app.route('/')
def index():
    """Health check"""
    return jsonify({
        "status": "ok",
        "product": PRODUCT_NAME,
        "version": PRODUCT_VERSION
    })


@app.route('/api/taxonomies', methods=['GET', 'POST'])
def get_taxonomies():
    """Return taxonomies for frontend filters"""
    taxonomies = {
        'states': ['Tennessee', 'West Virginia'],
        'certifications': ['CNA', 'Phlebotomy', 'Medical Assistant', 'EMT', 'Dental Assistant', 'Pharmacy Tech'],
        'cost_ranges': ['under500', '500to1000', '1000to2000', '2000to5000', 'over5000'],
        'durations': ['under4weeks', '4to8weeks', '8to12weeks', '3to6months', '6to12months']
    }
    return jsonify(taxonomies)


@app.route('/api/query', methods=['POST'])
def query():
    """Handle search queries"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Question required"}), 400
        
        if not app_graph:
            return jsonify({"error": "System not initialized"}), 503
        
        print(f"\n[*] Query: {question}")
        
        # Run RAG pipeline
        result = app_graph.invoke({
            "question": question,
            "docs": [],
            "context": "",
            "answer": "",
            "confidence": 0.0,
            "sources": []
        })
        
        response = {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"]
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    if initialize():
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    else:
        print("[!] Initialization failed")
        sys.exit(1)

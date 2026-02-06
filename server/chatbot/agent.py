"""
MidLens Agentic RAG Chatbot
============================
A clean, efficient RAG system with:
- Local sentence-transformer embeddings (no API costs)
- FAISS vector store for fast similarity search
- Agentic reasoning with tool use
- Streaming responses
- Context-aware medical responses

Author: Senior Data Scientist
"""

import json
import hashlib
import logging
import os
import re
import time
import ssl
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ChatConfig:
    """Centralized configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    temperature: float = 0.7
    max_tokens: int = 1024
    max_history: int = 10

CONFIG = ChatConfig()


# =============================================================================
# EMBEDDINGS ENGINE (Local, No API)
# =============================================================================
class EmbeddingEngine:
    """
    Local embedding engine using sentence-transformers.
    Falls back to TF-IDF if sentence-transformers unavailable.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EmbeddingEngine._initialized:
            return
        
        self.model = None
        self.use_tfidf = False
        self.tfidf_vocab = {}
        self.tfidf_idf = {}
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(CONFIG.embedding_model)
            logger.info(f"Loaded embedding model: {CONFIG.embedding_model}")
        except ImportError:
            logger.warning("sentence-transformers not available, using TF-IDF fallback")
            self.use_tfidf = True
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}, using TF-IDF")
            self.use_tfidf = True
        
        EmbeddingEngine._initialized = True
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        if not texts:
            return np.array([])
        
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            return self._tfidf_embed(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed([query])[0]
    
    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        """TF-IDF fallback embedding."""
        # Build vocabulary if needed
        if not self.tfidf_vocab:
            self._build_tfidf_vocab(texts)
        
        vectors = []
        for text in texts:
            vec = self._text_to_tfidf(text)
            vectors.append(vec)
        return np.array(vectors)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 2]
    
    def _build_tfidf_vocab(self, texts: List[str]):
        """Build TF-IDF vocabulary."""
        doc_freq = {}
        for text in texts:
            tokens = set(self._tokenize(text))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1
                if t not in self.tfidf_vocab:
                    self.tfidf_vocab[t] = len(self.tfidf_vocab)
        
        n = len(texts) + 1
        for term, freq in doc_freq.items():
            self.tfidf_idf[term] = np.log(n / (freq + 1)) + 1
    
    def _text_to_tfidf(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector."""
        tokens = self._tokenize(text)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        
        vec = np.zeros(max(len(self.tfidf_vocab), CONFIG.embedding_dim))
        for t, freq in tf.items():
            if t in self.tfidf_vocab:
                idx = self.tfidf_vocab[t]
                if idx < len(vec):
                    vec[idx] = freq * self.tfidf_idf.get(t, 1)
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec[:CONFIG.embedding_dim]


# =============================================================================
# VECTOR STORE
# =============================================================================
@dataclass
class Document:
    """A document chunk with metadata."""
    id: str
    content: str
    title: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    Simple in-memory vector store with cosine similarity.
    Uses FAISS if available for speed.
    """
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.use_faiss = False
        
        try:
            import faiss
            self.use_faiss = True
            logger.info("FAISS available for vector search")
        except ImportError:
            logger.info("FAISS not available, using numpy cosine similarity")
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with their embeddings."""
        self.documents = documents
        self.embeddings = embeddings.astype(np.float32)
        
        if self.use_faiss and len(embeddings) > 0:
            import faiss
            # Normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(self.embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if len(self.documents) == 0 or self.embeddings is None:
            return []
        
        query_vec = query_embedding.astype(np.float32).reshape(1, -1)
        
        if self.use_faiss and self.index is not None:
            import faiss
            faiss.normalize_L2(query_vec)
            scores, indices = self.index.search(query_vec, min(top_k, len(self.documents)))
            results = [(self.documents[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            # Numpy cosine similarity
            embeddings = self.embeddings  # Local reference for type checker
            norm_q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            norm_e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            scores = np.dot(norm_e, norm_q.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [(self.documents[i], float(scores[i])) for i in top_indices]
        
        return results


# =============================================================================
# DOCUMENT CHUNKER
# =============================================================================
class DocumentChunker:
    """Intelligent document chunking for medical knowledge base."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_knowledge_base(self, kb: Dict) -> List[Document]:
        """Chunk the knowledge base into retrievable documents."""
        documents = []
        self._process_dict(kb, documents, "")
        return documents
    
    def _process_dict(self, data: Dict, documents: List[Document], path: str):
        """Recursively process dictionary."""
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            title = key.replace('_', ' ').title()
            
            if isinstance(value, dict):
                # Create a document from this node
                content = self._dict_to_text(value, title)
                if len(content) > 50:  # Minimum content length
                    doc = Document(
                        id=hashlib.md5(current_path.encode()).hexdigest()[:12],
                        content=content,
                        title=title,
                        source=current_path,
                        metadata={'type': 'knowledge', 'path': current_path}
                    )
                    documents.append(doc)
                
                # Recurse into nested dicts
                self._process_dict(value, documents, current_path)
            
            elif isinstance(value, list):
                # Create document from list
                content = self._list_to_text(value, title)
                if len(content) > 50:
                    doc = Document(
                        id=hashlib.md5(current_path.encode()).hexdigest()[:12],
                        content=content,
                        title=title,
                        source=current_path,
                        metadata={'type': 'list', 'path': current_path}
                    )
                    documents.append(doc)
            
            elif isinstance(value, str) and len(value) > 100:
                # Large text values get their own document
                doc = Document(
                    id=hashlib.md5(current_path.encode()).hexdigest()[:12],
                    content=f"{title}: {value}",
                    title=title,
                    source=current_path,
                    metadata={'type': 'text', 'path': current_path}
                )
                documents.append(doc)
    
    def _dict_to_text(self, data: Dict, title: str) -> str:
        """Convert a dictionary to readable text."""
        parts = [f"# {title}\n"]
        
        for key, value in data.items():
            label = key.replace('_', ' ').title()
            
            if isinstance(value, str):
                parts.append(f"{label}: {value}")
            elif isinstance(value, list):
                parts.append(f"\n{label}:")
                for item in value[:10]:  # Limit list items
                    if isinstance(item, str):
                        parts.append(f"  - {item}")
                    elif isinstance(item, dict):
                        item_text = ", ".join(f"{k}: {v}" for k, v in item.items() if isinstance(v, str))
                        parts.append(f"  - {item_text}")
            elif isinstance(value, dict):
                # Shallow rendering for nested dicts
                nested = ", ".join(f"{k}: {v}" for k, v in value.items() if isinstance(v, str))
                if nested:
                    parts.append(f"{label}: {nested}")
        
        return "\n".join(parts)
    
    def _list_to_text(self, data: List, title: str) -> str:
        """Convert a list to readable text."""
        parts = [f"# {title}\n"]
        for item in data[:15]:
            if isinstance(item, str):
                parts.append(f"- {item}")
            elif isinstance(item, dict):
                item_text = "; ".join(f"{k}: {v}" for k, v in item.items() if isinstance(v, (str, int, float)))
                parts.append(f"- {item_text}")
        return "\n".join(parts)


# =============================================================================
# RETRIEVER
# =============================================================================
class HybridRetriever:
    """
    Hybrid retriever combining semantic search with keyword matching.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.embedder = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.chunker = DocumentChunker()
        self.documents: List[Document] = []
        self.kb_data: Dict = {}
        
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
    
    def load_knowledge_base(self, path: str):
        """Load and index the knowledge base."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.kb_data = json.load(f)
            
            # Chunk documents
            self.documents = self.chunker.chunk_knowledge_base(self.kb_data)
            logger.info(f"Chunked knowledge base into {len(self.documents)} documents")
            
            # Generate embeddings
            if self.documents:
                texts = [doc.content for doc in self.documents]
                embeddings = self.embedder.embed(texts)
                self.vector_store.add_documents(self.documents, embeddings)
                logger.info("Indexed documents in vector store")
        
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for a query."""
        if not self.documents:
            return []
        
        # Semantic search
        query_emb = self.embedder.embed_query(query)
        semantic_results = self.vector_store.search(query_emb, top_k * 2)
        
        # Keyword boost
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        boosted_results = []
        for doc, score in semantic_results:
            # Boost score if query terms appear in title or content
            boost = 0.0
            doc_lower = doc.content.lower()
            for term in query_terms:
                if len(term) > 3:
                    if term in doc.title.lower():
                        boost += 0.15
                    elif term in doc_lower:
                        boost += 0.05
            boosted_results.append((doc, min(score + boost, 1.0)))
        
        # Sort by boosted score and return top_k
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:top_k]


# =============================================================================
# AGENT TOOLS
# =============================================================================
@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool: str
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentTools:
    """Tools available to the agent."""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE
    
    def search_knowledge_base(self, query: str) -> ToolResult:
        """Search the medical knowledge base."""
        results = self.retriever.retrieve(query, top_k=3)
        
        if not results:
            return ToolResult(
                tool="knowledge_base",
                success=False,
                content="No relevant information found in the knowledge base."
            )
        
        content_parts = []
        sources = []
        for doc, score in results:
            content_parts.append(f"[{doc.title}] (relevance: {score:.2f})\n{doc.content[:600]}")
            sources.append({"title": doc.title, "source": doc.source, "score": score})
        
        return ToolResult(
            tool="knowledge_base",
            success=True,
            content="\n\n---\n\n".join(content_parts),
            metadata={"sources": sources, "num_results": len(results)}
        )
    
    def analyze_context(self, analysis: Dict) -> ToolResult:
        """Analyze the current scan analysis context."""
        if not analysis:
            return ToolResult(
                tool="analysis_context",
                success=False,
                content="No analysis context available."
            )
        
        try:
            prediction = analysis.get('prediction', {})
            tumor_class = prediction.get('class', 'unknown')
            display_name = prediction.get('display_name', tumor_class)
            confidence = prediction.get('confidence', 0)
            severity = prediction.get('severity', 'unknown')
            description = prediction.get('description', '')
            recommendations = prediction.get('recommendations', [])
            
            content = f"""Current Analysis Results:
- Predicted Class: {display_name}
- Confidence: {confidence:.1%}
- Severity: {severity}
- Description: {description}

Recommendations:
{chr(10).join('- ' + r for r in recommendations[:5])}"""
            
            return ToolResult(
                tool="analysis_context",
                success=True,
                content=content,
                metadata={"class": tumor_class, "confidence": confidence}
            )
        except Exception as e:
            return ToolResult(
                tool="analysis_context",
                success=False,
                content=f"Error parsing analysis: {e}"
            )
    
    def get_tumor_info(self, tumor_type: str) -> ToolResult:
        """Get detailed information about a specific tumor type."""
        kb = self.retriever.kb_data
        
        # Map common names to knowledge base keys
        type_map = {
            'glioma': 'glioma',
            'meningioma': 'meningioma', 
            'pituitary': 'pituitary',
            'pituitary tumor': 'pituitary',
            'pituitary adenoma': 'pituitary',
            'no tumor': 'notumor',
            'notumor': 'notumor',
            'normal': 'notumor'
        }
        
        key = type_map.get(tumor_type.lower(), tumor_type.lower())
        
        if 'brain_tumors' in kb and key in kb['brain_tumors']:
            data = kb['brain_tumors'][key]
            content = self._format_tumor_info(data, tumor_type)
            return ToolResult(
                tool="tumor_info",
                success=True,
                content=content,
                metadata={"tumor_type": key}
            )
        
        # Fall back to semantic search
        return self.search_knowledge_base(f"{tumor_type} brain tumor information")
    
    def _format_tumor_info(self, data: Dict, name: str) -> str:
        """Format tumor information nicely."""
        parts = [f"# {name.title()}\n"]
        
        if 'overview' in data:
            if isinstance(data['overview'], str):
                parts.append(data['overview'])
            elif isinstance(data['overview'], dict):
                for k, v in data['overview'].items():
                    if isinstance(v, str):
                        parts.append(f"**{k.replace('_', ' ').title()}**: {v}")
        
        for key in ['symptoms', 'diagnosis', 'treatment', 'prognosis']:
            if key in data:
                parts.append(f"\n## {key.title()}")
                value = data[key]
                if isinstance(value, list):
                    for item in value[:8]:
                        if isinstance(item, str):
                            parts.append(f"- {item}")
                        elif isinstance(item, dict):
                            parts.append(f"- {item.get('name', '')}: {item.get('description', '')}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str):
                            parts.append(f"**{k.replace('_', ' ').title()}**: {v}")
                elif isinstance(value, str):
                    parts.append(value)
        
        return "\n".join(parts)


# =============================================================================
# LLM CLIENT
# =============================================================================
class LLMClient:
    """Unified LLM client with fallbacks."""

    def __init__(self):
        self._load_env()
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

        # Build provider list for introspection
        self.providers: List[str] = []
        if os.environ.get('GEMINI_API_KEY'):
            self.providers.append('gemini')
        if os.environ.get('GROQ_API_KEY'):
            self.providers.append('groq')
        self.providers.append('local')
    
    def _load_env(self):
        """Load environment variables."""
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        os.environ.setdefault(k.strip(), v.strip())
    
    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> Tuple[str, str]:
        """
        Generate a response. Returns (response, model_used).
        Tries Gemini first, then falls back to local.
        """
        gemini_key = os.environ.get('GEMINI_API_KEY', '')
        
        if gemini_key:
            try:
                return self._call_gemini(prompt, system, temperature), "gemini-2.0-flash"
            except Exception as e:
                logger.warning(f"Gemini failed: {e}, falling back to local")
        
        return self._generate_local(prompt, system), "local"
    
    def _call_gemini(self, prompt: str, system: str, temperature: float) -> str:
        """Call Gemini API."""
        api_key = os.environ['GEMINI_API_KEY']
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        
        payload = json.dumps({
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": CONFIG.max_tokens
            }
        }).encode()
        
        req = urllib.request.Request(
            url,
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=20, context=self._ssl_ctx) as resp:
            data = json.loads(resp.read().decode())
        
        return data['candidates'][0]['content']['parts'][0]['text']
    
    def _generate_local(self, prompt: str, system: str) -> str:
        """Generate a response locally using templates."""
        query_lower = prompt.lower()
        
        # Extract key intent
        if any(w in query_lower for w in ['what is', 'explain', 'describe', 'tell me about']):
            return self._template_explain(prompt)
        elif any(w in query_lower for w in ['symptom', 'sign', 'feel']):
            return self._template_symptoms()
        elif any(w in query_lower for w in ['treatment', 'therapy', 'cure', 'treat']):
            return self._template_treatment()
        elif any(w in query_lower for w in ['prognosis', 'survival', 'outcome']):
            return self._template_prognosis()
        else:
            return self._template_general()
    
    def _template_explain(self, query: str) -> str:
        return """Based on the provided medical knowledge:

Brain tumors are abnormal growths of cells in the brain or surrounding tissues. They can be benign (non-cancerous) or malignant (cancerous), and are classified as primary (originating in the brain) or secondary/metastatic (spreading from elsewhere).

The main types detected by this system include:
- **Glioma**: Tumors arising from glial cells, the most common type of primary brain tumor
- **Meningioma**: Tumors of the meninges (protective brain coverings), usually benign
- **Pituitary tumors**: Growths in the pituitary gland affecting hormone production

If you'd like more specific information about any of these types, please ask!"""

    def _template_symptoms(self) -> str:
        return """Common symptoms of brain tumors include:

• Persistent headaches that worsen over time
• Seizures (may be the first symptom)
• Vision, hearing, or speech changes
• Balance and coordination problems
• Cognitive changes (memory, concentration)
• Nausea and vomiting
• Weakness or numbness in limbs

**Important**: These symptoms can have many causes. If you experience any of these, consult a healthcare provider for proper evaluation."""

    def _template_treatment(self) -> str:
        return """Brain tumor treatment depends on tumor type, location, size, and patient health:

**Surgery**: Primary treatment goal is maximum safe removal while preserving function.

**Radiation Therapy**: External beam radiation, stereotactic radiosurgery, or proton therapy.

**Chemotherapy**: Temozolomide is standard for glioblastoma; other agents for different tumor types.

**Targeted Therapy**: Medications targeting specific molecular changes in tumors.

**Supportive Care**: Physical therapy, occupational therapy, and symptom management.

Treatment plans are personalized and often combine multiple approaches. A multidisciplinary team of specialists collaborates on each case."""

    def _template_prognosis(self) -> str:
        return """Prognosis varies significantly based on tumor type and grade:

**Glioma**:
- Low-grade: Median survival 5-10+ years with treatment
- High-grade (Glioblastoma): Median survival 14-16 months

**Meningioma**:
- Grade I: 10-year survival >90%
- Grade II-III: Varies based on completeness of resection

**Pituitary tumors**: Generally excellent prognosis as most are benign.

**Key prognostic factors**: Age, performance status, extent of resection, and molecular markers (IDH mutation, MGMT methylation).

Note: Statistics are averages; individual outcomes vary. Discuss your specific situation with your healthcare team."""

    def _template_general(self) -> str:
        return """I'm the MidLens AI assistant, specializing in brain tumor information.

I can help you with:
• Understanding different types of brain tumors
• Explaining symptoms and diagnostic methods  
• Providing information about treatment options
• Interpreting analysis results from this tool

**Important Disclaimer**: This AI provides educational information only. It is not a substitute for professional medical advice. Always consult qualified healthcare providers for medical decisions.

How can I help you today?"""


# =============================================================================
# AGENTIC CHATBOT
# =============================================================================
class AgenticChatbot:
    """
    Agentic chatbot with RAG.
    
    Features:
    - Semantic retrieval from knowledge base
    - Tool-based reasoning
    - Context-aware responses
    - Conversation memory
    - Graceful fallbacks
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        if knowledge_base_path is None:
            knowledge_base_path = str(
                Path(__file__).parent.parent.parent / 'knowledge_base' / 'medical_knowledge.json'
            )
        
        self.retriever = HybridRetriever(knowledge_base_path)
        self.tools = AgentTools(self.retriever)
        self.llm = LLMClient()
        self.sessions: Dict[str, List[Dict]] = {}
        
        logger.info("AgenticChatbot initialized")
    
    def chat(self, message: str, session_id: str = "default", 
             analysis_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a chat message and return a response.
        
        Args:
            message: User's message
            session_id: Session identifier for conversation memory
            analysis_context: Optional analysis results for context
            
        Returns:
            Response dictionary with message, sources, confidence, etc.
        """
        start_time = time.time()
        
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        history = self.sessions[session_id]
        
        # Step 1: Understand intent and gather context
        context_parts = []
        sources = []
        
        # Tool 1: Check analysis context
        if analysis_context:
            analysis_result = self.tools.analyze_context(analysis_context)
            if analysis_result.success:
                context_parts.append(f"[Current Analysis]\n{analysis_result.content}")
                
                # Auto-fetch info about detected tumor type
                prediction = analysis_context.get('prediction', {})
                tumor_class = prediction.get('class', '')
                if tumor_class and tumor_class != 'notumor':
                    tumor_info = self.tools.get_tumor_info(tumor_class)
                    if tumor_info.success:
                        context_parts.append(f"[About {tumor_class.title()}]\n{tumor_info.content[:800]}")
        
        # Tool 2: Search knowledge base
        kb_result = self.tools.search_knowledge_base(message)
        if kb_result.success:
            context_parts.append(f"[Knowledge Base]\n{kb_result.content}")
            sources = kb_result.metadata.get('sources', [])
        
        # Build the prompt
        system_prompt = self._build_system_prompt()
        
        context = "\n\n".join(context_parts) if context_parts else "No specific context available."
        
        history_text = ""
        if history:
            recent = history[-CONFIG.max_history:]
            history_text = "\n".join([f"{h['role'].title()}: {h['content'][:200]}" for h in recent])
        
        user_prompt = f"""Context Information:
{context}

{"Conversation History:" + chr(10) + history_text if history_text else ""}

User Question: {message}

Please provide a helpful, accurate, and medically appropriate response based on the context above."""
        
        # Generate response
        response_text, model_used = self.llm.generate(user_prompt, system_prompt)
        
        # Update session history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        
        # Keep history manageable
        if len(history) > CONFIG.max_history * 2:
            self.sessions[session_id] = history[-(CONFIG.max_history * 2):]
        
        # Calculate confidence based on retrieval quality
        confidence = 0.7
        if sources:
            avg_score = sum(s.get('score', 0) for s in sources) / len(sources)
            confidence = min(0.95, 0.5 + avg_score * 0.5)
        
        processing_time = time.time() - start_time
        
        return {
            "message": response_text,
            "sources": [
                {
                    "title": s.get('title', 'Unknown'),
                    "url": "",
                    "snippet": "",
                    "source_type": "knowledge_base",
                    "reliability": s.get('score', 0.8)
                }
                for s in sources[:3]
            ],
            "verified": len(sources) > 0,
            "confidence": confidence,
            "model_used": model_used,
            "processing_time": processing_time
        }
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are MidLens AI, a specialized medical AI assistant focused on brain tumor information.

GUIDELINES:
1. Provide accurate, evidence-based medical information
2. Use the provided context to answer questions
3. Be empathetic and supportive in your tone
4. Always clarify that you provide educational information, not medical advice
5. Recommend consulting healthcare professionals for personal medical decisions
6. If uncertain, acknowledge limitations honestly
7. Format responses clearly with headers and bullet points when appropriate
8. Keep responses concise but comprehensive

IMPORTANT: Never diagnose conditions or recommend specific treatments. Provide general educational information only."""

    def clear_session(self, session_id: str):
        """Clear a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_suggestions(self, analysis_context: Optional[Dict] = None) -> List[str]:
        """Get contextual question suggestions."""
        if analysis_context:
            prediction = analysis_context.get('prediction', {})
            tumor_class = prediction.get('class', '')
            display_name = prediction.get('display_name', tumor_class)
            
            if tumor_class and tumor_class != 'notumor':
                return [
                    f"What is {display_name}?",
                    f"What are the symptoms of {display_name}?",
                    f"What treatment options exist for {display_name}?",
                    "Explain my analysis results",
                    "What should I discuss with my doctor?"
                ]
        
        return [
            "What types of brain tumors can you detect?",
            "What are common symptoms of brain tumors?",
            "How accurate is the AI analysis?",
            "What is a glioma?",
            "What is a meningioma?"
        ]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================
def create_chatbot(knowledge_base_path: Optional[str] = None) -> AgenticChatbot:
    """Create and return a chatbot instance."""
    return AgenticChatbot(knowledge_base_path)


# =============================================================================
# TESTING
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the chatbot
    kb_path = Path(__file__).parent.parent.parent / 'knowledge_base' / 'medical_knowledge.json'
    chatbot = create_chatbot(str(kb_path))
    
    # Test queries
    test_queries = [
        "What is a glioma?",
        "What are the symptoms of brain tumors?",
        "What treatment options are available for meningioma?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        response = chatbot.chat(query)
        print(f"Response: {response['message'][:500]}...")
        print(f"Model: {response['model_used']}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Sources: {len(response['sources'])}")

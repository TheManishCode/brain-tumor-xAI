"""
MidLens Fallback Chatbot
========================
Multi-LLM fallback chatbot with:
- Multiple LLM providers (Gemini → Groq → Local)
- Web and PubMed search augmentation
- Medical-aware responses with RAG
- Smart context extraction
"""

import json
import hashlib
import urllib.request
import urllib.parse
import ssl
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Import centralized LLM client
from .llm_client import call_llm, get_available_providers

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class Source:
    """A verified source for information."""
    title: str
    url: str
    snippet: str
    source_type: str
    reliability: float = 0.8
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'source_type': self.source_type,
            'reliability': self.reliability
        }


@dataclass
class ChatResponse:
    """Response from the chatbot."""
    message: str
    sources: List[Source] = field(default_factory=list)
    verified: bool = False
    confidence: float = 0.0
    model_used: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'message': self.message,
            'sources': [s.to_dict() for s in self.sources],
            'verified': self.verified,
            'confidence': self.confidence,
            'model_used': self.model_used
        }


# =============================================================================
# KNOWLEDGE BASE PROCESSOR
# =============================================================================
class KnowledgeProcessor:
    """
    Extracts and formats knowledge from the knowledge base.
    Converts raw JSON/dicts into readable text.
    """
    
    def __init__(self, knowledge_base: Dict):
        self.kb = knowledge_base
    
    def search(self, query: str) -> Tuple[str, Dict]:
        """
        Search knowledge base and return formatted text + raw data.
        
        Returns:
            (formatted_text, raw_data)
        """
        query_lower = query.lower()
        matches: List[Dict] = []
        
        # Extract query keywords
        keywords = self._extract_keywords(query_lower)
        
        # Search recursively
        self._search_recursive(self.kb, keywords, matches, "")
        
        if not matches:
            return "", {}
        
        # Get best match
        best_match = matches[0]
        formatted = self._format_knowledge(best_match['data'], best_match['path'])
        
        return formatted, best_match['data']
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'for',
                      'how', 'can', 'do', 'does', 'it', 'about', 'tell', 'me',
                      'explain', 'describe', 'please', 'help', 'with'}
        
        words = re.findall(r'\b[a-z]+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _search_recursive(self, data: Dict, keywords: List[str], 
                          matches: List[Dict], path: str):
        """Recursively search for matching content."""
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            key_lower = key.lower().replace('_', ' ')
            
            score = sum(1 for kw in keywords if kw in key_lower)
            
            if score > 0 and isinstance(value, dict):
                matches.append({
                    'path': current_path,
                    'data': value,
                    'score': score
                })
            
            if isinstance(value, dict):
                self._search_recursive(value, keywords, matches, current_path)
        
        matches.sort(key=lambda x: x['score'], reverse=True)
    
    def _format_knowledge(self, data: Dict, path: str) -> str:
        """Format a knowledge entry as readable text."""
        parts = []
        title = path.split('.')[-1].replace('_', ' ').title()
        parts.append(f"## {title}\n")
        
        # If data has an 'overview' key that's a dict, use that as the primary source
        source = data
        if 'overview' in data and isinstance(data['overview'], dict):
            source = data['overview']
        
        # Extract description/definition (must be a string)
        for field in ['definition', 'description', 'overview']:
            if field in source:
                value = source[field]
                if isinstance(value, str) and value:
                    parts.append(value)
                    break
        
        if 'prevalence' in source and isinstance(source['prevalence'], str):
            parts.append(f"\n**Prevalence:** {source['prevalence']}")
        
        list_fields = [
            ('types', 'Types'),
            ('types_overview', 'Types'),
            ('symptoms', 'Symptoms'),
            ('general_symptoms', 'Common Symptoms'),
            ('treatment_options', 'Treatment Options'),
            ('treatment_overview', 'Treatment'),
            ('risk_factors', 'Risk Factors'),
            ('diagnosis_methods', 'Diagnosis Methods')
        ]
        
        for field_key, field_name in list_fields:
            if field_key in source:
                field_data = source[field_key]
                if isinstance(field_data, list) and field_data:
                    parts.append(f"\n**{field_name}:**")
                    for item in field_data[:6]:
                        if isinstance(item, str):
                            parts.append(f"• {item}")
                        elif isinstance(item, dict) and 'name' in item:
                            parts.append(f"• {item['name']}")
                elif isinstance(field_data, str):
                    parts.append(f"\n**{field_name}:** {field_data}")
        
        if 'when_to_seek_help' in source and isinstance(source['when_to_seek_help'], str):
            parts.append(f"\n**When to Seek Help:** {source['when_to_seek_help']}")
        
        return "\n".join(parts)


# =============================================================================
# WEB SEARCH
# =============================================================================
class WebSearcher:
    """Search the web with caching."""
    
    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self._cache: Dict[str, Tuple[List[Source], datetime]] = {}
        self._cache_ttl = timedelta(hours=1)
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
    
    def search(self, query: str, max_results: int = 3) -> List[Source]:
        """Search web and PubMed for information."""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self._cache:
            cached, ts = self._cache[cache_key]
            if datetime.now() - ts < self._cache_ttl:
                return cached
        
        sources: List[Source] = []
        
        try:
            sources.extend(self._search_ddg(query))
        except Exception as e:
            logger.debug(f"DDG search failed: {e}")
        
        if self._is_medical(query):
            try:
                sources.extend(self._search_pubmed(query))
            except Exception as e:
                logger.debug(f"PubMed failed: {e}")
        
        sources = sources[:max_results]
        self._cache[cache_key] = (sources, datetime.now())
        return sources
    
    def _is_medical(self, query: str) -> bool:
        terms = ['tumor', 'cancer', 'brain', 'glioma', 'meningioma', 
                 'treatment', 'symptom', 'diagnosis', 'mri']
        return any(t in query.lower() for t in terms)
    
    def _search_ddg(self, query: str) -> List[Source]:
        """DuckDuckGo Instant Answer API."""
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_redirect=1"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'MidLens/1.0'})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_context) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        sources: List[Source] = []
        if data.get('Abstract'):
            sources.append(Source(
                title=data.get('Heading', 'Web Result'),
                url=data.get('AbstractURL', ''),
                snippet=data['Abstract'][:400],
                source_type='web',
                reliability=0.8
            ))
        
        return sources
    
    def _search_pubmed(self, query: str) -> List[Source]:
        """Search PubMed."""
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        search_url = f"{base}/esearch.fcgi?db=pubmed&retmode=json&retmax=2&term={urllib.parse.quote(query)}"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'MidLens/1.0'})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_context) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        ids = data.get('esearchresult', {}).get('idlist', [])
        if not ids:
            return []
        
        sum_url = f"{base}/esummary.fcgi?db=pubmed&retmode=json&id={','.join(ids)}"
        req = urllib.request.Request(sum_url, headers={'User-Agent': 'MidLens/1.0'})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_context) as resp:
            sum_data = json.loads(resp.read().decode('utf-8'))
        
        sources: List[Source] = []
        results = sum_data.get('result', {})
        for pmid in ids:
            if pmid in results and isinstance(results[pmid], dict):
                article = results[pmid]
                sources.append(Source(
                    title=article.get('title', 'PubMed Article')[:80],
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    snippet=f"Journal: {article.get('source', 'Unknown')}. Published: {article.get('pubdate', 'Unknown')}",
                    source_type='pubmed',
                    reliability=0.95
                ))
        
        return sources


# =============================================================================
# LOCAL RESPONSE GENERATOR
# =============================================================================
class LocalResponseGenerator:
    """Generate responses locally without calling any LLM."""
    
    @staticmethod
    def generate(query: str, context: str = "") -> str:
        """Generate a local response based on query and context."""
        query_lower = query.lower().strip()
        
        # Handle greetings
        if query_lower in ['hi', 'hello', 'hey', 'good morning', 'good evening']:
            return """Hello! I'm the **MidLens AI Assistant**.

I can help you with:
• **Brain tumor information** - types, symptoms, causes
• **MRI analysis results** - understanding your scan  
• **Treatment options** - surgery, radiation, chemotherapy
• **Research** - latest medical findings

What would you like to know about?"""
        
        # Use context if available
        if context and len(context) > 100:
            result = context
            if any(w in query_lower for w in ['treatment', 'diagnosis', 'surgery', 'prognosis']):
                result += "\n\n---\n⚕️ *Please consult a healthcare provider for personalized medical advice.*"
            return result
        
        # Topic-specific responses
        if 'glioma' in query_lower:
            return """## Glioma

A **glioma** is a type of brain tumor that originates in the glial cells, which support and protect neurons in the brain.

**Types:**
• **Astrocytoma** - from star-shaped astrocytes
• **Oligodendroglioma** - from oligodendrocytes  
• **Glioblastoma (GBM)** - most aggressive form

**Common Symptoms:**
• Persistent headaches
• Seizures
• Memory or cognitive changes
• Vision or speech problems

**Treatment Options:**
• Surgery to remove tumor
• Radiation therapy
• Chemotherapy (temozolomide)
• Targeted therapy

---
⚕️ *Please consult a healthcare provider for personalized medical advice.*"""
        
        if 'meningioma' in query_lower:
            return """## Meningioma

A **meningioma** is a tumor that develops from the meninges, the protective membranes covering the brain and spinal cord. Most meningiomas are benign (non-cancerous).

**Key Facts:**
• Most common primary brain tumor
• Usually slow-growing
• More common in women and older adults

**Symptoms:**
• Headaches
• Vision changes
• Hearing loss
• Weakness in limbs

**Treatment:**
• Observation (for small, asymptomatic tumors)
• Surgery
• Radiation therapy

---
⚕️ *Please consult a healthcare provider for personalized medical advice.*"""
        
        if 'pituitary' in query_lower:
            return """## Pituitary Tumor

A **pituitary tumor** is a growth in the pituitary gland, a small gland at the base of the brain that controls hormone production.

**Key Facts:**
• Usually benign (adenomas)
• Can affect hormone levels
• May grow slowly over many years

**Symptoms:**
• Hormonal imbalances
• Vision problems (if pressing on optic nerve)
• Headaches
• Fatigue

**Treatment:**
• Medication to control hormone levels
• Surgery (transsphenoidal approach)
• Radiation therapy

---
⚕️ *Please consult a healthcare provider for personalized medical advice.*"""
        
        if 'brain tumor' in query_lower or 'tumor' in query_lower:
            return """## Brain Tumors

A **brain tumor** is an abnormal growth of cells in or around the brain. They can be:
• **Benign** (non-cancerous) - slow-growing, less aggressive
• **Malignant** (cancerous) - fast-growing, can spread

**Common Types:**
• **Glioma** - from glial cells (astrocytoma, glioblastoma)
• **Meningioma** - from protective membranes
• **Pituitary tumors** - affect hormone production

**General Symptoms:**
• Persistent headaches
• Seizures
• Vision or hearing changes
• Cognitive difficulties
• Balance problems

**Diagnosis:**
• MRI scan (most detailed)
• CT scan
• Biopsy
• Neurological exam

---
⚕️ *Please consult a healthcare provider for personalized medical advice.*"""
        
        if any(w in query_lower for w in ['symptom', 'sign', 'warning']):
            return """## Brain Tumor Symptoms

Brain tumor symptoms vary depending on size, location, and growth rate.

**Common Warning Signs:**
• **Headaches** - especially worse in morning
• **Seizures** - new onset in adults is concerning
• **Vision changes** - blurred or double vision
• **Cognitive changes** - memory, concentration
• **Personality changes** - mood, behavior
• **Balance problems** - difficulty walking
• **Nausea/vomiting** - especially morning

**When to Seek Medical Attention:**
• New, severe, or persistent headaches
• First-time seizure
• Sudden vision or speech changes
• Progressive weakness

---
⚕️ *These symptoms can have many causes. Consult a healthcare provider for proper evaluation.*"""
        
        # Out of scope
        if any(w in query_lower for w in ['pancreatic', 'lung', 'breast', 'colon', 'liver']):
            return """I specialize in **brain tumors** and related conditions.

For other types of cancer, I recommend consulting:
• Your primary care physician
• An oncologist specialized in that area
• Cancer information resources like cancer.org

Is there anything about brain tumors I can help you with?"""
        
        # Default
        return """I'm here to help with brain tumor information.

**I can answer questions about:**
• Brain tumor types (glioma, meningioma, pituitary)
• Symptoms and warning signs
• Diagnosis methods (MRI, CT, biopsy)
• Treatment options
• Understanding your MRI results

What would you like to know?"""


# =============================================================================
# LLM RESPONSE GENERATOR (Uses centralized client)
# =============================================================================
class LLMResponseGenerator:
    """Generate responses using the centralized LLM client."""
    
    def __init__(self):
        self.providers = get_available_providers()
        self.providers.append('local')  # Always have local fallback
        logger.info(f"LLM providers available: {self.providers}")
    
    def generate(self, query: str, context: str, sources: Optional[List[Source]] = None) -> Tuple[str, float, str]:
        """
        Generate response using available LLM.
        
        Returns:
            (response, confidence, model_used)
        """
        # Handle greetings locally (no need to waste LLM calls)
        if self._is_greeting(query):
            return LocalResponseGenerator.generate(query), 0.95, 'greeting_handler'
        
        # Build prompt
        prompt = self._build_prompt(query, context, sources or [])
        
        # Try LLM first
        if 'gemini' in self.providers or 'groq' in self.providers:
            try:
                response, model = call_llm(prompt)
                return response, 0.95 if 'gemini' in model else 0.9, model
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
        
        # Fallback to local
        response = LocalResponseGenerator.generate(query, context)
        return response, 0.7, 'local'
    
    def _is_greeting(self, msg: str) -> bool:
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        return msg.lower().strip() in greetings
    
    def _build_prompt(self, query: str, context: str, sources: List[Source]) -> str:
        """Build prompt for LLM."""
        parts = [
            "You are MidLens AI, a medical assistant specialized in brain tumors.",
            "",
            "Guidelines:",
            "- Provide accurate, evidence-based information",
            "- Use clear, patient-friendly language", 
            "- Format with markdown (bold, bullets)",
            "- Only add medical disclaimer for treatment/diagnosis questions",
            "- For greetings, be friendly without disclaimers",
            ""
        ]
        
        if context:
            parts.append("KNOWLEDGE BASE INFORMATION:")
            parts.append(context)
            parts.append("")
        
        if sources:
            parts.append("VERIFIED SOURCES:")
            for s in sources[:3]:
                parts.append(f"- [{s.source_type}] {s.title}: {s.snippet[:150]}")
            parts.append("")
        
        parts.append(f"USER QUESTION: {query}")
        parts.append("")
        parts.append("Provide a helpful, well-formatted response:")
        
        return "\n".join(parts)


# =============================================================================
# MAIN CHATBOT CLASS
# =============================================================================
class AIChatbot:
    """AI Chatbot for MidLens with RAG."""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.llm = LLMResponseGenerator()
        self.searcher = WebSearcher()
        self.knowledge: Optional[KnowledgeProcessor] = None
        self.sessions: Dict[str, List[Dict]] = {}
        
        if knowledge_base_path:
            self._load_knowledge_base(knowledge_base_path)
        
        logger.info(f"AI Chatbot ready (providers: {self.llm.providers})")
    
    def _load_knowledge_base(self, path: str):
        """Load knowledge base."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            self.knowledge = KnowledgeProcessor(kb)
            logger.info(f"Loaded knowledge base: {path}")
        except Exception as e:
            logger.warning(f"Failed to load KB: {e}")
    
    def chat(self, message: str, session_id: str = "default", 
             analysis_context: Optional[Dict] = None) -> ChatResponse:
        """
        Process chat message and generate response using RAG pipeline.
        
        Pipeline:
        1. Session management
        2. Knowledge base retrieval
        3. Web search (for non-greetings)
        4. Analysis context injection
        5. LLM response generation (with fallback to local)
        """
        # Session management
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({'role': 'user', 'content': message})
        
        sources: List[Source] = []
        context = ""
        
        # 1. Search knowledge base (RAG retrieval)
        if self.knowledge:
            kb_text, kb_data = self.knowledge.search(message)
            if kb_text:
                context = kb_text
                sources.append(Source(
                    title="MidLens Knowledge Base",
                    url="local://kb",
                    snippet=kb_text[:150],
                    source_type='knowledge_base',
                    reliability=0.95
                ))
        
        # 2. Web search (skip for greetings to save time)
        if not self._is_greeting(message):
            try:
                web_sources = self.searcher.search(message)
                sources.extend(web_sources)
            except Exception as e:
                logger.debug(f"Web search error: {e}")
        
        # 3. Add analysis context if available
        if analysis_context:
            context += f"\n\n**Current Analysis:** {self._format_analysis(analysis_context)}"
        
        # 4. Generate response (LLM with local fallback)
        response_text, confidence, model = self.llm.generate(message, context, sources)
        
        response = ChatResponse(
            message=response_text,
            sources=sources[:5],
            verified=confidence > 0.7,
            confidence=confidence,
            model_used=model
        )
        
        # Update session memory
        self.sessions[session_id].append({'role': 'assistant', 'content': response.message})
        
        # Keep only last 10 messages per session
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]
        
        return response
    
    def get_suggestions(self, session_id: str = "default", 
                        analysis_context: Optional[Dict] = None) -> List[str]:
        """
        Generate contextual follow-up suggestions.
        
        Note: This uses local logic to avoid unnecessary LLM calls.
        """
        if analysis_context:
            pred = analysis_context.get('prediction', {})
            class_name = pred.get('display_name', pred.get('class', ''))
            return [
                f'Explain my {class_name} results',
                f'What are treatment options for {class_name}?',
                'What questions should I ask my doctor?'
            ]
        
        return [
            'What types of brain tumors can you detect?',
            'How accurate is the AI analysis?',
            'What are common symptoms of brain tumors?'
        ]
    
    def _is_greeting(self, msg: str) -> bool:
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        return msg.lower().strip() in greetings
    
    def _format_analysis(self, analysis: Dict) -> str:
        if 'prediction' in analysis:
            pred = analysis['prediction']
            conf = pred.get('confidence', 0)
            if isinstance(conf, float) and conf < 1:
                conf = conf * 100
            return f"{pred.get('display_name', 'Unknown')} ({conf:.1f}% confidence)"
        return ""
    
    def clear_session(self, session_id: str):
        """Clear a chat session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get the conversation history for a session."""
        return self.sessions.get(session_id, [])

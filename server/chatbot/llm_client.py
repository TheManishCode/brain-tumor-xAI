"""
LLM Client - Centralized API Access
====================================
All LLM calls go through this module.

Gemini: Uses Retry-After header from 429 responses (Google's recommended approach)
Groq: Uses rolling window rate limiter (stateless rate limiting)
"""

import os
import json
import ssl
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .rate_limiter import RollingWindowRateLimiter

logger = logging.getLogger(__name__)


# =============================================================================
# ENVIRONMENT LOADING
# =============================================================================
def _load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
        logger.info("Loaded environment variables from .env")

_load_env()


# =============================================================================
# GLOBAL RATE LIMITERS
# =============================================================================
# Groq Free Tier: 30 RPM - uses rolling window limiter
# Gemini: Uses Retry-After header from 429 responses (no pre-emptive limiting)

GROQ_LIMITER = RollingWindowRateLimiter(
    max_calls=30, 
    window_seconds=60, 
    name="groq"
)


# =============================================================================
# SSL CONTEXT (for API calls)
# =============================================================================
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE


# =============================================================================
# API KEYS
# =============================================================================
def get_gemini_key() -> str:
    return os.environ.get('GEMINI_API_KEY', '')


def get_groq_key() -> str:
    return os.environ.get('GROQ_API_KEY', '')


# =============================================================================
# GEMINI CLIENT
# =============================================================================
def call_gemini(
    prompt: str, 
    temperature: float = 0.7,
    max_tokens: int = 800
) -> str:
    """
    Call Gemini API - single attempt, no retries.
    
    If rate limited (429), raises immediately to allow fallback to local.
    
    Args:
        prompt: The prompt to send to Gemini
        temperature: Generation temperature (0.0 - 1.0)
        max_tokens: Maximum output tokens
        
    Returns:
        The generated text response
        
    Raises:
        ValueError: If no API key is configured
        RuntimeError: If API call fails (including 429)
    """
    api_key = get_gemini_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }).encode()
    
    try:
        req = urllib.request.Request(
            url, 
            data=payload, 
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=15, context=_ssl_context) as resp:
            data = json.loads(resp.read().decode())
        
        return data['candidates'][0]['content']['parts'][0]['text']
        
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RuntimeError("Gemini rate limited")
        raise RuntimeError(f"Gemini API error {e.code}: {e.reason}")
        
    except Exception as e:
        raise RuntimeError(f"Gemini API failed: {e}")


# =============================================================================
# GROQ CLIENT
# =============================================================================
def call_groq(
    prompt: str,
    retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 800,
    model: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Call Groq API with automatic rate limiting and retry.
    
    Args:
        prompt: The prompt to send to Groq
        retries: Number of retry attempts
        temperature: Generation temperature (0.0 - 1.0)
        max_tokens: Maximum output tokens
        model: Groq model to use
        
    Returns:
        The generated text response
        
    Raises:
        ValueError: If no API key is configured
        RuntimeError: If all retries fail
    """
    api_key = get_groq_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY not configured")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }).encode()
    
    for attempt in range(retries):
        try:
            # Wait for rate limit clearance
            wait_time = GROQ_LIMITER.wait()
            if wait_time > 0:
                logger.debug(f"[Groq] Rate limit wait: {wait_time:.1f}s")
            
            logger.debug(f"[Groq] API call (attempt {attempt + 1}/{retries})")
            
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
            )
            
            with urllib.request.urlopen(req, timeout=30, context=_ssl_context) as resp:
                data = json.loads(resp.read().decode())
            
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            error_str = str(e)
            
            # Handle 429 with backoff
            if "429" in error_str:
                backoff = 5 * (attempt + 1)
                logger.warning(f"[Groq] 429 error, backing off {backoff}s")
                time.sleep(backoff)
                continue
            
            if attempt < retries - 1:
                logger.warning(f"[Groq] Error: {e}, retrying...")
                time.sleep(1)
                continue
            
            raise RuntimeError(f"Groq API failed: {e}")
    
    raise RuntimeError("Groq API failed after all retries")


# =============================================================================
# MULTI-PROVIDER CALL
# =============================================================================
def call_llm(
    prompt: str,
    prefer_gemini: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 800
) -> Tuple[str, str]:
    """
    Call the best available LLM with automatic fallback.
    
    Order: Gemini → Groq → raises error
    
    Args:
        prompt: The prompt to send
        prefer_gemini: Whether to try Gemini first
        temperature: Generation temperature
        max_tokens: Maximum output tokens
        
    Returns:
        Tuple of (response_text, model_name)
        
    Raises:
        RuntimeError: If no LLM is available or all fail
    """
    providers = []
    
    if prefer_gemini and get_gemini_key():
        providers.append(('gemini', call_gemini, "gemini-2.0-flash"))
    if get_groq_key():
        providers.append(('groq', call_groq, "groq-llama3"))
    if not prefer_gemini and get_gemini_key():
        providers.append(('gemini', call_gemini, "gemini-2.0-flash"))
    
    if not providers:
        raise RuntimeError("No LLM providers configured (need GEMINI_API_KEY or GROQ_API_KEY)")
    
    last_error = None
    
    for name, call_fn, model_name in providers:
        try:
            response = call_fn(prompt, temperature=temperature, max_tokens=max_tokens)
            return response, model_name
        except Exception as e:
            logger.warning(f"[{name}] Failed: {e}")
            last_error = e
            continue
    
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_available_providers() -> list:
    """Get list of available LLM providers."""
    providers = []
    if get_gemini_key():
        providers.append('gemini')
    if get_groq_key():
        providers.append('groq')
    return providers


def get_rate_limit_status() -> Dict[str, Any]:
    """Get current rate limit status for all providers."""
    return {
        'gemini': {
            'available': bool(get_gemini_key()),
            'note': 'Uses Retry-After header (no pre-emptive limiting)'
        },
        'groq': {
            'available': bool(get_groq_key()),
            'current_usage': GROQ_LIMITER.current_usage,
            'max_calls': GROQ_LIMITER.max_calls,
            'window_seconds': GROQ_LIMITER.window,
            'time_until_available': GROQ_LIMITER.time_until_available()
        }
    }

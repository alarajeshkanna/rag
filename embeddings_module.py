"""
Embedding generation module with different embedding models
"""
import requests
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from config import EmbeddingModel, KGConfig

logger = logging.getLogger(__name__)


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators"""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """OpenAI embedding generator (Text-ADA-002)"""
    
    def __init__(self, config: KGConfig):
        self.config = config
        self.api_url = config.auth.api_url
        self.bluetoken_cookie = config.auth.bluetoken_cookie
        self.cert_path = config.auth.certificate_path
        self.batch_size = 100  # OpenAI batch limit
        self.max_retries = 3
        self.retry_delay = 1
    
    def generate_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        headers = {
            'Content-Type': 'application/json',
            'Cookie': f'bluetoken={self.bluetoken_cookie}' if self.bluetoken_cookie else ''
        }
        
        payload = {
            "input": texts,
            "model": "text-embedding-ada-002"
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    verify=self.cert_path,
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                embeddings = [item['embedding'] for item in result['data']]
                return embeddings
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for text-embedding-ada-002"""
        return 1536


class BGEEmbeddingGenerator(EmbeddingGenerator):
    """BGE (BAAI General Embedding) generator"""
    
    def __init__(self, config: KGConfig):
        self.config = config
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.batch_size = 32
    
    def generate_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using BGE model"""
        if not texts:
            return []
        
        # Normalize texts for BGE (add query prefix if specified)
        processed_texts = []
        for text in texts:
            # BGE recommends adding "Represent this sentence for searching relevant passages:"
            # for query texts, but for document indexing we use text as-is
            if kwargs.get('is_query', False):
                processed_texts.append(f"Represent this sentence for searching relevant passages: {text}")
            else:
                processed_texts.append(text)
        
        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def get
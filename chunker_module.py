"""
Text chunking module with different chunking strategies
"""
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import markdown
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer

from config import ChunkingStrategy, KGConfig

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunking parameters"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]


class TextChunker(ABC):
    """Abstract base class for text chunkers"""
    
    def __init__(self, config: KGConfig, chunk_config: ChunkConfig):
        self.config = config
        self.chunk_config = chunk_config
    
    @abstractmethod
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents into smaller pieces"""
        pass
    
    def _create_chunk_metadata(self, original_doc: Dict[str, Any], chunk_idx: int, 
                              chunk_text: str, start_char: int = 0) -> Dict[str, Any]:
        """Create metadata for a chunk"""
        metadata = original_doc['metadata'].copy()
        metadata.update({
            'chunk_id': chunk_idx,
            'chunk_length': len(chunk_text),
            'start_char': start_char,
            'original_length': metadata.get('length', 0)
        })
        return metadata


class RecursiveTextChunker(TextChunker):
    """Recursive text chunker that splits text hierarchically"""
    
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents using recursive splitting"""
        chunked_docs = []
        
        for doc in documents:
            content = doc['content']
            chunks = self._recursive_split(content, self.chunk_config.separators)
            
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) >= self.chunk_config.min_chunk_size:
                    chunk_doc = {
                        'content': chunk_text.strip(),
                        'metadata': self._create_chunk_metadata(doc, i, chunk_text)
                    }
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using different separators"""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level splitting
            return self._split_by_character(text)
        
        splits = text.split(separator)
        
        # If no split occurred or text is small enough, return as is
        if len(splits) == 1 or len(text) <= self.chunk_config.chunk_size:
            return [text]
        
        # Process each split
        final_chunks = []
        current_chunk = ""
        
        for split in splits:
            # Add separator back except for the last split
            split_with_sep = split + separator if split != splits[-1] else split
            
            if len(current_chunk) + len(split_with_sep) <= self.chunk_config.chunk_size:
                current_chunk += split_with_sep
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                
                # If single split is too large, recursively split it
                if len(split_with_sep) > self.chunk_config.chunk_size:
                    sub_chunks = self._recursive_split(split_with_sep, remaining_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split_with_sep
        
        if current_chunk:
            final_chunks.append(current_chunk)
        
        return final_chunks
    
    def _split_by_character(self, text: str) -> List[str]:
        """Split text by character when no other separator works"""
        chunks = []
        for i in range(0, len(text), self.chunk_config.chunk_size):
            chunk = text[i:i + self.chunk_config.chunk_size]
            chunks.append(chunk)
        return chunks


class MarkdownChunker(TextChunker):
    """Chunker that respects markdown structure"""
    
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents respecting markdown structure"""
        chunked_docs = []
        
        for doc in documents:
            content = doc['content']
            chunks = self._chunk_markdown(content)
            
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) >= self.chunk_config.min_chunk_size:
                    chunk_doc = {
                        'content': chunk_text.strip(),
                        'metadata': self._create_chunk_metadata(doc, i, chunk_text)
                    }
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _chunk_markdown(self, text: str) -> List[str]:
        """Chunk text while preserving markdown structure"""
        # Split by markdown headers first
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        chunks = []
        current_chunk = ""
        current_header = ""
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            
            if header_match:
                # New header found
                if current_chunk and len(current_chunk) >= self.chunk_config.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                current_header = line
                current_chunk = line + "\n"
            else:
                line_to_add = line + "\n"
                
                # Check if adding this line exceeds chunk size
                if len(current_chunk) + len(line_to_add) > self.chunk_config.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = current_header + "\n" + line_to_add if current_header else line_to_add
                else:
                    current_chunk += line_to_add
        
        # Add remaining chunk
        if current_chunk and len(current_chunk.strip()) >= self.chunk_config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks


class SemanticChunker(TextChunker):
    """Chunker that uses semantic similarity to group related content"""
    
    def __init__(self, config: KGConfig, chunk_config: ChunkConfig):
        super().__init__(config, chunk_config)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.7
    
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents using semantic similarity"""
        chunked_docs = []
        
        for doc in documents:
            content = doc['content']
            chunks = self._semantic_chunk(content)
            
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) >= self.chunk_config.min_chunk_size:
                    chunk_doc = {
                        'content': chunk_text.strip(),
                        'metadata': self._create_chunk_metadata(doc, i, chunk_text)
                    }
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """Chunk text based on semantic similarity"""
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 2:
            return [text]
        
        # Get embeddings for sentences
        embeddings = self.sentence_model.encode(sentences)
        
        # Calculate similarity matrix
        similarities = np.inner(embeddings, embeddings)
        
        # Find chunk boundaries based on similarity drops
        chunks = []
        current_chunk_sentences = []
        current_chunk_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if (current_chunk_length + sentence_length > self.chunk_config.chunk_size and 
                current_chunk_sentences):
                
                # Check semantic similarity with previous sentence
                if i > 0:
                    similarity = similarities[i-1][i]
                    if similarity < self.similarity_threshold:
                        # Low similarity, create new chunk
                        chunks.append(' '.join(current_chunk_sentences))
                        current_chunk_sentences = [sentence]
                        current_chunk_length = sentence_length
                        continue
                
                # High similarity but size exceeded, create chunk anyway
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_length = sentence_length
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_length += sentence_length
        
        # Add remaining sentences
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]


class ChunkerFactory:
    """Factory class for creating text chunkers"""
    
    def __init__(self, config: KGConfig):
        self.config = config
    
    def get_chunker(self, strategy: ChunkingStrategy, 
                   chunk_config: Optional[ChunkConfig] = None) -> TextChunker:
        """Get appropriate chunker for strategy"""
        if chunk_config is None:
            chunk_config = ChunkConfig()
        
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveTextChunker(self.config, chunk_config)
        elif strategy == ChunkingStrategy.MARKDOWN:
            return MarkdownChunker(self.config, chunk_config)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(self.config, chunk_config)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
    
    def chunk_documents(self, documents: List[Dict[str, Any]], 
                       strategy: ChunkingStrategy,
                       chunk_config: Optional[ChunkConfig] = None) -> List[Dict[str, Any]]:
        """Chunk documents using specified strategy"""
        chunker = self.get_chunker(strategy, chunk_config)
        return chunker.chunk(documents)

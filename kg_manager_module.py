"""
Knowledge Graph Manager - Main orchestration class
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from config_module import KGConfig, LoaderType, ChunkingStrategy, EmbeddingModel
from doc_loader_module import DocumentLoaderFactory
from chunker_module import ChunkerFactory, ChunkConfig
from embeddings_module import EmbeddingFactory
from database_module import CassandraManager

logger = logging.getLogger(__name__)


class KGManager:
    """Main Knowledge Graph Manager class"""

    def __init__(self, config: KGConfig):
        self.config = config
        self.doc_loader_factory = DocumentLoaderFactory(config)
        self.chunker_factory = ChunkerFactory(config)
        self.embedding_factory = EmbeddingFactory(config)
        self.db_manager = CassandraManager(config)

        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'keywords_extracted': 0,
            'start_time': None,
            'end_time': None
        }

    def initialize(self) -> None:
        """Initialize the KG Manager"""
        logger.info("Initializing Knowledge Graph Manager")
        self.db_manager.connect()
        self.stats['start_time'] = datetime.now()
        logger.info("KG Manager initialized successfully")

    def shutdown(self) -> None:
        """Shutdown the KG Manager"""
        logger.info("Shutting down Knowledge Graph Manager")
        self.db_manager.disconnect()
        self.stats['end_time'] = datetime.now()
        self._log_stats()

    def process_document(self,
                         source: str,
                         loader_type: Optional[LoaderType] = None,
                         chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                         embedding_model: EmbeddingModel = EmbeddingModel.TEXT_ADA_002,
                         chunk_config: Optional[ChunkConfig] = None,
                         **kwargs) -> Dict[str, Any]:
        """Process a single document through the entire pipeline"""
        logger.info(f"Processing document: {source}")
        try:
            documents = self._load_document(source, loader_type, **kwargs)
            chunks = self._chunk_documents(documents, chunking_strategy, chunk_config)
            chunks_with_embeddings = self._generate_embeddings(chunks, embedding_model, **kwargs)
            results = self._store_documents_and_chunks(documents, chunks_with_embeddings)

            if kwargs.get('extract_keywords', True):
                self._extract_keywords(chunks_with_embeddings, results['chunk_ids'])

            self.stats['documents_processed'] += len(documents)
            self.stats['chunks_created'] += len(chunks)
            self.stats['embeddings_generated'] += len(chunks_with_embeddings)

            logger.info(f"Successfully processed document: {source}")
            return {
                'success': True,
                'document_id': results['document_id'],
                'chunk_ids': results['chunk_ids'],
                'chunks_count': len(chunks),
                'source': source
            }

        except Exception as e:
            logger.error(f"Error processing document {source}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'source': source
            }

    def process_multiple_documents(self,
                                   sources: List[Dict[str, Any]],
                                   default_chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                                   default_embedding_model: EmbeddingModel = EmbeddingModel.TEXT_ADA_002,
                                   **kwargs) -> List[Dict[str, Any]]:
        """Process multiple documents in a loop"""
        results = []
        for source_config in sources:
            source = source_config['source']
            loader_type = source_config.get('loader_type')
            chunking_strategy = source_config.get('chunking_strategy', default_chunking_strategy)
            embedding_model = source_config.get('embedding_model', default_embedding_model)

            result = self.process_document(
                source=source,
                loader_type=loader_type,
                chunking_strategy=chunking_strategy,
                embedding_model=embedding_model,
                **kwargs
            )
            results.append(result)
        return results

    def search_similar_content(self,
                               query: str,
                               embedding_model: EmbeddingModel = EmbeddingModel.TEXT_ADA_002,
                               top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar content using embedding similarity"""
        try:
            query_embedding = self.embedding_factory.generate_embeddings(
                [query], embedding_model, is_query=True
            )[0]
            return self.db_manager.search_similar_chunks(query_embedding, top_k)
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def _load_document(self, source: str, loader_type: Optional[LoaderType], **kwargs) -> List[Dict[str, Any]]:
        if loader_type:
            return self.doc_loader_factory.load_document(source, loader_type, **kwargs)
        return self.doc_loader_factory.auto_detect_and_load(source, **kwargs)

    def _chunk_documents(self, documents: List[Dict[str, Any]],
                         strategy: ChunkingStrategy,
                         chunk_config: Optional[ChunkConfig]) -> List[Dict[str, Any]]:
        return self.chunker_factory.chunk_documents(documents, strategy, chunk_config)

    def _generate_embeddings(self, chunks: List[Dict[str, Any]],
                             model: EmbeddingModel, **kwargs) -> List[Dict[str, Any]]:
        return self.embedding_factory.add_embeddings_to_documents(chunks, model, **kwargs)

    def _store_documents_and_chunks(self, documents: List[Dict[str, Any]],
                                    chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        document = documents[0]
        document_id = self.db_manager.insert_document(document)
        chunk_ids = self.db_manager.insert_chunks(chunks, document_id)
        return {
            'document_id': document_id,
            'chunk_ids': chunk_ids
        }

    def _extract_keywords(self, chunks: List[Dict[str, Any]], chunk_ids: List[str]) -> None:
        keywords_data = []
        for chunk, chunk_id in zip(chunks, chunk_ids):
            content = chunk['content'].lower()
            words = content.split()
            stop_words = set(self.config.use_case.stop_words)
            keywords = [word.strip('.,!?;:') for word in words
                        if len(word) > 3 and word not in stop_words]
            unique_keywords = list(set(keywords))[:10]

            for keyword in unique_keywords:
                keywords_data.append({
                    'chunk_id': chunk_id,
                    'document_id': chunk['metadata'].get('document_id', ''),
                    'keyword': keyword,
                    'relevance_score': 1.0
                })

        if keywords_data:
            self.db_manager.insert_keywords(keywords_data)
            self.stats['keywords_extracted'] += len(keywords_data)

    def _log_stats(self) -> None:
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info("Processing Statistics:")
            logger.info(f"  - Documents processed: {self.stats['documents_processed']}")
            logger.info(f"  - Chunks created: {self.stats['chunks_created']}")
            logger.info(f"  - Embeddings generated: {self.stats['embeddings_generated']}")
            logger.info(f"  - Keywords extracted: {self.stats['keywords_extracted']}")
            logger.info(f"  - Total duration: {duration}")

    def get_processing_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents for the current use case"""
        try:
            documents = self.db_manager.get_documents_by_use_case()
            logger.info(f"Fetched {len(documents)} documents from DB")
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []


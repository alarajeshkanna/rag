"""
Database module for Cassandra operations
"""
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.query import SimpleStatement, ConsistencyLevel
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import uuid
from datetime import datetime
from contextlib import contextmanager

from config import KGConfig

logger = logging.getLogger(__name__)


class CassandraManager:
    """Cassandra database manager for knowledge graph storage"""
    
    def __init__(self, config: KGConfig):
        self.config = config
        self.cluster = None
        self.session = None
        self.keyspace = f"kg_{config.use_case.use_case_id}"
        
    def connect(self) -> None:
        """Establish connection to Cassandra cluster"""
        try:
            # Setup authentication
            auth_provider = None
            if self.config.cassandra.user and self.config.cassandra.password:
                auth_provider = PlainTextAuthProvider(
                    username=self.config.cassandra.user,
                    password=self.config.cassandra.password
                )
            
            # Create cluster connection
            self.cluster = Cluster(
                contact_points=self.config.cassandra.hosts,
                port=int(self.config.cassandra.port),
                auth_provider=auth_provider,
                load_balancing_policy=DCAwareRoundRobinPolicy()
            )
            
            self.session = self.cluster.connect()
            logger.info(f"Connected to Cassandra cluster at {self.config.cassandra.hosts}")
            
            # Create keyspace if it doesn't exist
            self._create_keyspace()
            self.session.set_keyspace(self.keyspace)
            
            # Create tables
            self._create_tables()
            
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Close connection to Cassandra"""
        if self.session:
            self.session.shutdown()
        if self.cluster:
            self.cluster.shutdown()
        logger.info("Disconnected from Cassandra")
    
    def _create_keyspace(self) -> None:
        """Create keyspace for the use case"""
        create_keyspace_query = f"""
        CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
        WITH replication = {{
            'class': 'SimpleStrategy',
            'replication_factor': 3
        }}
        """
        self.session.execute(create_keyspace_query)
        logger.info(f"Created/verified keyspace: {self.keyspace}")
    
    def _create_tables(self) -> None:
        """Create necessary tables for knowledge graph storage"""
        
        # Documents table
        create_documents_table = f"""
        CREATE TABLE IF NOT EXISTS {self.keyspace}.documents (
            id UUID PRIMARY KEY,
            use_case_id TEXT,
            source TEXT,
            title TEXT,
            content TEXT,
            metadata TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """
        
        # Chunks table
        create_chunks_table = f"""
        CREATE TABLE IF NOT EXISTS {self.keyspace}.chunks (
            id UUID PRIMARY KEY,
            document_id UUID,
            use_case_id TEXT,
            chunk_index INT,
            content TEXT,
            embedding LIST<FLOAT>,
            metadata TEXT,
            created_at TIMESTAMP,
            INDEX(document_id)
        )
        """
        
        # Keywords table
        create_keywords_table = f"""
        CREATE TABLE IF NOT EXISTS {self.keyspace}.keywords (
            id UUID PRIMARY KEY,
            chunk_id UUID,
            document_id UUID,
            use_case_id TEXT,
            keyword TEXT,
            relevance_score FLOAT,
            created_at TIMESTAMP,
            INDEX(chunk_id),
            INDEX(document_id),
            INDEX(keyword)
        )
        """
        
        # Entities table
        create_entities_table = f"""
        CREATE TABLE IF NOT EXISTS {self.keyspace}.entities (
            id UUID PRIMARY KEY,
            use_case_id TEXT,
            entity_text TEXT,
            entity_type TEXT,
            confidence_score FLOAT,
            chunk_ids LIST<UUID>,
            metadata TEXT,
            created_at TIMESTAMP,
            INDEX(entity_text),
            INDEX(entity_type)
        )
        """
        
        # Relationships table
        create_relationships_table = f"""
        CREATE TABLE IF NOT EXISTS {self.keyspace}.relationships (
            id UUID PRIMARY KEY,
            use_case_id TEXT,
            source_entity_id UUID,
            target_entity_id UUID,
            relationship_type TEXT,
            confidence_score FLOAT,
            chunk_ids LIST<UUID>,
            metadata TEXT,
            created_at TIMESTAMP,
            INDEX(source_entity_id),
            INDEX(target_entity_id)
        )
        """
        
        # Execute table creation queries
        tables = [
            create_documents_table,
            create_chunks_table,
            create_keywords_table,
            create_entities_table,
            create_relationships_table
        ]
        
        for table_query in tables:
            try:
                self.session.execute(table_query)
            except Exception as e:
                logger.warning(f"Error creating table: {str(e)}")
        
        logger.info("Created/verified all tables")
    
    def insert_document(self, document: Dict[str, Any]) -> str:
        """Insert a document into the database"""
        doc_id = str(uuid.uuid4())
        
        insert_query = f"""
        INSERT INTO {self.keyspace}.documents (
            id, use_case_id, source, title, content, metadata, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.session.execute(insert_query, [
            uuid.UUID(doc_id),
            self.config.use_case.use_case_id,
            document['metadata'].get('source', ''),
            document['metadata'].get('title', ''),
            document['content'],
            json.dumps(document['metadata']),
            datetime.now(),
            datetime.now()
        ])
        
        logger.debug(f"Inserted document with ID: {doc_id}")
        return doc_id
    
    def insert_chunks(self, chunks: List[Dict[str, Any]], document_id: str) -> List[str]:
        """Insert chunks into the database"""
        chunk_ids = []
        
        insert_query = f"""
        INSERT INTO {self.keyspace}.chunks (
            id, document_id, use_case_id, chunk_index, content, embedding, metadata, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            self.session.execute(insert_query, [
                uuid.UUID(chunk_id),
                uuid.UUID(document_id),
                self.config.use_case.use_case_id,
                chunk['metadata'].get('chunk_id', i),
                chunk['content'],
                chunk.get('embedding', []),
                json.dumps(chunk['metadata']),
                datetime.now()
            ])
        
        logger.debug(f"Inserted {len(chunks)} chunks for document {document_id}")
        return chunk_ids
    
    def insert_keywords(self, keywords_data: List[Dict[str, Any]]) -> None:
        """Insert keywords into the database"""
        insert_query = f"""
        INSERT INTO {self.keyspace}.keywords (
            id, chunk_id, document_id, use_case_id, keyword, relevance_score, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        for keyword_data in keywords_data:
            self.session.execute(insert_query, [
                uuid.uuid4(),
                uuid.UUID(keyword_data['chunk_id']),
                uuid.UUID(keyword_data['document_id']),
                self.config.use_case.use_case_id,
                keyword_data['keyword'],
                keyword_data.get('relevance_score', 0.0),
                datetime.now()
            ])
        
        logger.debug(f"Inserted {len(keywords_data)} keywords")
    
    def insert_entity(self, entity_data: Dict[str, Any]) -> str:
        """Insert an entity into the database"""
        entity_id = str(uuid.uuid4())
        
        insert_query = f"""
        INSERT INTO {self.keyspace}.entities (
            id, use_case_id, entity_text, entity_type, confidence_score, 
            chunk_ids, metadata, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        chunk_ids = [uuid.UUID(cid) for cid in entity_data.get('chunk_ids', [])]
        
        self.session.execute(insert_query, [
            uuid.UUID(entity_id),
            self.config.use_case.use_case_id,
            entity_data['entity_text'],
            entity_data['entity_type'],
            entity_data.get('confidence_score', 0.0),
            chunk_ids,
            json.dumps(entity_data.get('metadata', {})),
            datetime.now()
        ])
        
        logger.debug(f"Inserted entity: {entity_data['entity_text']}")
        return entity_id
    
    def insert_relationship(self, relationship_data: Dict[str, Any]) -> str:
        """Insert a relationship into the database"""
        rel_id = str(uuid.uuid4())
        
        insert_query = f"""
        INSERT INTO {self.keyspace}.relationships (
            id, use_case_id, source_entity_id, target_entity_id, 
            relationship_type, confidence_score, chunk_ids, metadata, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        chunk_ids = [uuid.UUID(cid) for cid in relationship_data.get('chunk_ids', [])]
        
        self.session.execute(insert_query, [
            uuid.UUID(rel_id),
            self.config.use_case.use_case_id,
            uuid.UUID(relationship_data['source_entity_id']),
            uuid.UUID(relationship_data['target_entity_id']),
            relationship_data['relationship_type'],
            relationship_data.get('confidence_score', 0.0),
            chunk_ids,
            json.dumps(relationship_data.get('metadata', {})),
            datetime.now()
        ])
        
        logger.debug(f"Inserted relationship: {relationship_data['relationship_type']}")
        return rel_id
    
    def search_similar_chunks(self, query_embedding: List[float], 
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        # Note: This is a simplified implementation
        # In production, you'd want to use a vector database like Pinecone or Weaviate
        # or Cassandra with vector search capabilities
        
        query = f"""
        SELECT id, document_id, content, embedding, metadata
        FROM {self.keyspace}.chunks
        WHERE use_case_id = ?
        LIMIT 1000
        """
        
        rows = self.session.execute(query, [self.config.use_case.use_case_id])
        
        # Calculate similarities (this is inefficient for large datasets)
        similarities = []
        for row in rows:
            if row.embedding:
                similarity = self._cosine_similarity(query_embedding, row.embedding)
                similarities.append({
                    'id': str(row.id),
                    'document_id': str(row.document_id),
                    'content': row.content,
                    'metadata': json.loads(row.metadata) if row.metadata else {},
                    'similarity': similarity
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_documents_by_use_case(self) -> List[Dict[str, Any]]:
        """Get all documents for the current use case"""
        query = f"""
        SELECT id, source, title, content, metadata, created_at
        FROM {self.keyspace}.documents
        WHERE use_case_id = ?
        """
        
        rows = self.session.execute(query, [self.config.use_case.use_case_id])
        
        documents = []
        for row in rows:
            documents.append({
                'id': str(row.id),
                'source': row.source,
                'title': row.title,
                'content': row.content,
                'metadata': json.loads(row.metadata) if row.metadata else {},
                'created_at': row.created_at
            })
        
        return documents
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        query = f"""
        SELECT id, chunk_index, content, embedding, metadata, created_at
        FROM {self.keyspace}.chunks
        WHERE document_id = ?
        """
        
        rows = self.session.execute(query, [uuid.UUID(document_id)])
        
        chunks = []
        for row in rows:
            chunks.append({
                'id': str(row.id),
                'chunk_index': row.chunk_index,
                'content': row.content,
                'embedding': row.embedding,
                'metadata': json.loads(row.metadata) if row.metadata else {},
                'created_at': row.created_at
            })
        
        return chunks
    
    @contextmanager
    def get_session(self):
        """Context manager for database session"""
        if not self.session:
            self.connect()
        try:
            yield self.session
        finally:
            pass  # Keep connection alive for reuse

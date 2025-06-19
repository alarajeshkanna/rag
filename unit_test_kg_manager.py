import unittest
from unittest.mock import patch, MagicMock
from kg_manager_module import KGManager
from config_module import get_default_config, LoaderType, ChunkingStrategy, EmbeddingModel
from chunker_module import ChunkConfig


class TestKGManager(unittest.TestCase):

    def setUp(self):
        self.config = get_default_config()
        self.kg_manager = KGManager(self.config)

    @patch('kg_manager_module.CassandraManager')
    def test_initialize_and_shutdown(self, mock_db_manager):
        self.kg_manager.db_manager = mock_db_manager()
        self.kg_manager.initialize()
        self.kg_manager.shutdown()
        self.kg_manager.db_manager.connect.assert_called_once()
        self.kg_manager.db_manager.disconnect.assert_called_once()

    @patch('kg_manager_module.DocumentLoaderFactory')
    @patch('kg_manager_module.ChunkerFactory')
    @patch('kg_manager_module.EmbeddingFactory')
    @patch('kg_manager_module.CassandraManager')
    def test_process_document_success(self, mock_db, mock_embedding_factory, mock_chunker_factory, mock_loader_factory):
        source = "https://example.com/article"

        mock_loader = MagicMock()
        mock_loader.load.return_value = [{'content': 'Test content', 'metadata': {'source': source, 'title': 'Test Title', 'length': 12}}]
        mock_loader_factory.return_value.load_document.return_value = mock_loader.load.return_value

        mock_chunker = MagicMock()
        mock_chunker.chunk_documents.return_value = [{'content': 'Chunked content', 'metadata': {'chunk_id': 0, 'length': 12}}]
        mock_chunker_factory.return_value.chunk_documents.return_value = mock_chunker.chunk_documents.return_value

        mock_embedder = MagicMock()
        mock_embedder.add_embeddings_to_documents.return_value = [{'content': 'Chunked content', 'embedding': [0.1] * 1536, 'metadata': {'chunk_id': 0}}]
        mock_embedding_factory.return_value.add_embeddings_to_documents.return_value = mock_embedder.add_embeddings_to_documents.return_value

        mock_db_instance = MagicMock()
        mock_db_instance.insert_document.return_value = "doc-id-123"
        mock_db_instance.insert_chunks.return_value = ["chunk-id-123"]
        mock_db.return_value = mock_db_instance

        self.kg_manager.doc_loader_factory = mock_loader_factory.return_value
        self.kg_manager.chunker_factory = mock_chunker_factory.return_value
        self.kg_manager.embedding_factory = mock_embedding_factory.return_value
        self.kg_manager.db_manager = mock_db_instance

        result = self.kg_manager.process_document(
            source=source,
            loader_type=LoaderType.URL,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            embedding_model=EmbeddingModel.TEXT_ADA_002,
            chunk_config=ChunkConfig(chunk_size=500, chunk_overlap=100),
            extract_keywords=True
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['source'], source)
        self.assertEqual(result['document_id'], "doc-id-123")
        self.assertEqual(result['chunk_ids'], ["chunk-id-123"])

    @patch('kg_manager_module.CassandraManager')
    def test_get_processing_stats(self, mock_db_manager):
        stats = self.kg_manager.get_processing_stats()
        self.assertIn('documents_processed', stats)
        self.assertIn('chunks_created', stats)

    @patch('kg_manager_module.CassandraManager')
    def test_list_documents(self, mock_db_manager):
        mock_db_instance = mock_db_manager.return_value
        mock_db_instance.get_documents_by_use_case.return_value = [{'id': 'doc1', 'title': 'test'}]
        self.kg_manager.db_manager = mock_db_instance

        docs = self.kg_manager.list_documents()
        self.assertIsInstance(docs, list)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['id'], 'doc1')


if __name__ == '__main__':
    unittest.main()

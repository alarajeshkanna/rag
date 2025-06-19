import argparse
import logging
from config_module import LoaderType, ChunkingStrategy, EmbeddingModel, get_default_config
from kg_manager_module import KGManager
from chunker_module import ChunkConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGPipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Document Ingestion Pipeline")
    
    parser.add_argument('--source', type=str, required=True,
                        help="Path or URL to the document")
    
    parser.add_argument('--loader', type=str, choices=[lt.value for lt in LoaderType],
                        help="Loader type: url, pdf, docx")
    
    parser.add_argument('--chunking', type=str, default='recursive',
                        choices=[cs.value for cs in ChunkingStrategy],
                        help="Chunking strategy: recursive, markdown, semantic")
    
    parser.add_argument('--embedding', type=str, default='text-embedding-ada-002',
                        choices=[em.value for em in EmbeddingModel],
                        help="Embedding model to use")
    
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help="Chunk size for splitting text")
    
    parser.add_argument('--chunk_overlap', type=int, default=200,
                        help="Overlap between chunks")
    
    parser.add_argument('--extract_keywords', action='store_true',
                        help="Enable keyword extraction")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = get_default_config()
    
    # Override default model with user input
    config.model.embedding_model = EmbeddingModel(args.embedding)
    
    # Initialize KGManager
    manager = KGManager(config)
    manager.initialize()

    # Prepare loader type if specified
    loader_type = LoaderType(args.loader) if args.loader else None

    # Prepare chunking strategy
    chunking_strategy = ChunkingStrategy(args.chunking)
    
    # Chunk configuration
    chunk_config = ChunkConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Run document ingestion
    result = manager.process_document(
        source=args.source,
        loader_type=loader_type,
        chunking_strategy=chunking_strategy,
        embedding_model=EmbeddingModel(args.embedding),
        chunk_config=chunk_config,
        extract_keywords=args.extract_keywords
    )
    
    # Log the result
    if result.get('success'):
        logger.info(f"Document processed successfully: {result}")
    else:
        logger.error(f"Failed to process document: {result.get('error')}")
    
    manager.shutdown()

if __name__ == "__main__":
    main()

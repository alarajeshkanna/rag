"""
Main pipeline for RAG document ingestion with configurable options
"""
import argparse
import logging
from typing import List, Dict, Any
import json
from pathlib import Path

from config import KGConfig, LoaderType, ChunkingStrategy, EmbeddingModel
from kg_manager_module import KGManager
from doc_loader_module import DocumentLoaderFactory
from chunker_module import ChunkerFactory, ChunkConfig
from embeddings_module import EmbeddingFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> KGConfig:
    """Load configuration from file or use default"""
    if config_path:
        try:
            with open(config_path) as f:
                config_data = json.load(f)
                return KGConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}")
            logger.info("Using default configuration")
    
    return KGConfig.from_env()

def process_single_document(kg_manager: KGManager, source: str, 
                          loader_type: LoaderType = None,
                          chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                          embedding_model: EmbeddingModel = EmbeddingModel.TEXT_ADA_002,
                          chunk_config: ChunkConfig = None,
                          **kwargs) -> Dict[str, Any]:
    """
    Process a single document through the RAG pipeline
    """
    return kg_manager.process_document(
        source=source,
        loader_type=loader_type,
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
        chunk_config=chunk_config,
        **kwargs
    )

def process_batch_from_file(kg_manager: KGManager, batch_file: str, 
                          default_chunking: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                          default_embedding: EmbeddingModel = EmbeddingModel.TEXT_ADA_002,
                          **kwargs) -> List[Dict[str, Any]]:
    """
    Process a batch of documents from a JSON file
    """
    with open(batch_file) as f:
        sources = json.load(f)
    
    return kg_manager.process_multiple_documents(
        sources=sources,
        default_chunking_strategy=default_chunking,
        default_embedding_model=default_embedding,
        **kwargs
    )

def interactive_mode(kg_manager: KGManager):
    """Interactive mode for processing documents"""
    print("\nRAG Document Ingestion Pipeline - Interactive Mode")
    print("-----------------------------------------------")
    
    while True:
        print("\nOptions:")
        print("1. Process a single document")
        print("2. Process multiple documents from batch file")
        print("3. Search similar content")
        print("4. View processing statistics")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            # Process single document
            source = input("Enter document source (URL/file path): ").strip()
            
            print("\nAvailable loader types:")
            for i, loader in enumerate(LoaderType):
                print(f"{i+1}. {loader.name}")
            loader_choice = input("Choose loader type (leave blank for auto-detect): ").strip()
            loader_type = list(LoaderType)[int(loader_choice)-1] if loader_choice else None
            
            print("\nAvailable chunking strategies:")
            for i, strategy in enumerate(ChunkingStrategy):
                print(f"{i+1}. {strategy.name}")
            chunk_choice = input(f"Choose chunking strategy (default {ChunkingStrategy.RECURSIVE.name}): ").strip()
            chunking_strategy = list(ChunkingStrategy)[int(chunk_choice)-1] if chunk_choice else ChunkingStrategy.RECURSIVE
            
            print("\nAvailable embedding models:")
            for i, model in enumerate(EmbeddingModel):
                print(f"{i+1}. {model.name}")
            embed_choice = input(f"Choose embedding model (default {EmbeddingModel.TEXT_ADA_002.name}): ").strip()
            embedding_model = list(EmbeddingModel)[int(embed_choice)-1] if embed_choice else EmbeddingModel.TEXT_ADA_002
            
            # Custom chunk config
            custom_chunk = input("Use custom chunk config? (y/n): ").lower() == 'y'
            chunk_config = None
            if custom_chunk:
                chunk_size = int(input("Chunk size (default 1000): ") or 1000)
                chunk_overlap = int(input("Chunk overlap (default 200): ") or 200)
                chunk_config = ChunkConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            
            result = process_single_document(
                kg_manager=kg_manager,
                source=source,
                loader_type=loader_type,
                chunking_strategy=chunking_strategy,
                embedding_model=embedding_model,
                chunk_config=chunk_config
            )
            
            print("\nProcessing result:")
            print(json.dumps(result, indent=2))
            
        elif choice == "2":
            # Process batch from file
            batch_file = input("Enter path to batch JSON file: ").strip()
            if not Path(batch_file).exists():
                print(f"Error: File {batch_file} not found")
                continue
                
            print("\nAvailable chunking strategies:")
            for i, strategy in enumerate(ChunkingStrategy):
                print(f"{i+1}. {strategy.name}")
            chunk_choice = input(f"Choose default chunking strategy (default {ChunkingStrategy.RECURSIVE.name}): ").strip()
            default_chunking = list(ChunkingStrategy)[int(chunk_choice)-1] if chunk_choice else ChunkingStrategy.RECURSIVE
            
            print("\nAvailable embedding models:")
            for i, model in enumerate(EmbeddingModel):
                print(f"{i+1}. {model.name}")
            embed_choice = input(f"Choose default embedding model (default {EmbeddingModel.TEXT_ADA_002.name}): ").strip()
            default_embedding = list(EmbeddingModel)[int(embed_choice)-1] if embed_choice else EmbeddingModel.TEXT_ADA_002
            
            results = process_batch_from_file(
                kg_manager=kg_manager,
                batch_file=batch_file,
                default_chunking=default_chunking,
                default_embedding=default_embedding
            )
            
            print("\nBatch processing results:")
            for i, result in enumerate(results):
                print(f"\nDocument {i+1}:")
                print(json.dumps(result, indent=2))
            
        elif choice == "3":
            # Search similar content
            query = input("Enter search query: ").strip()
            
            print("\nAvailable embedding models:")
            for i, model in enumerate(EmbeddingModel):
                print(f"{i+1}. {model.name}")
            embed_choice = input(f"Choose embedding model for query (default {EmbeddingModel.TEXT_ADA_002.name}): ").strip()
            embedding_model = list(EmbeddingModel)[int(embed_choice)-1] if embed_choice else EmbeddingModel.TEXT_ADA_002
            
            top_k = int(input("Number of results to return (default 10): ") or 10)
            
            results = kg_manager.search_similar_content(
                query=query,
                embedding_model=embedding_model,
                top_k=top_k
            )
            
            print(f"\nTop {top_k} similar chunks:")
            for i, result in enumerate(results):
                print(f"\nResult {i+1} (Similarity: {result['similarity']:.3f}):")
                print(f"Source Document: {result['metadata'].get('source', 'Unknown')}")
                print(f"Content: {result['content'][:200]}...")
            
        elif choice == "4":
            # View statistics
            stats = kg_manager.get_processing_stats()
            print("\nProcessing Statistics:")
            print(json.dumps(stats, indent=2, default=str))
            
        elif choice == "5":
            # Exit
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main entry point for RAG pipeline"""
    parser = argparse.ArgumentParser(description="RAG Document Ingestion Pipeline")
    parser.add_argument('--config', type=str, help="Path to configuration file")
    parser.add_argument('--source', type=str, help="Document source to process (URL or file path)")
    parser.add_argument('--batch', type=str, help="JSON file with batch processing configuration")
    parser.add_argument('--loader', type=str, choices=[lt.name for lt in LoaderType], 
                       help="Document loader type (url, pdf, docx)")
    parser.add_argument('--chunking', type=str, choices=[cs.name for cs in ChunkingStrategy], 
                       default=ChunkingStrategy.RECURSIVE.name,
                       help="Chunking strategy to use")
    parser.add_argument('--embedding', type=str, choices=[em.name for em in EmbeddingModel], 
                       default=EmbeddingModel.TEXT_ADA_002.name,
                       help="Embedding model to use")
    parser.add_argument('--interactive', action='store_true', 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize KG Manager
    kg_manager = KGManager(config)
    kg_manager.initialize()
    
    try:
        if args.interactive:
            interactive_mode(kg_manager)
        elif args.batch:
            # Process batch from file
            results = process_batch_from_file(
                kg_manager=kg_manager,
                batch_file=args.batch,
                default_chunking=ChunkingStrategy[args.chunking],
                default_embedding=EmbeddingModel[args.embedding]
            )
            print("\nBatch processing completed:")
            print(json.dumps(results, indent=2))
        elif args.source:
            # Process single document
            loader_type = LoaderType[args.loader] if args.loader else None
            result = process_single_document(
                kg_manager=kg_manager,
                source=args.source,
                loader_type=loader_type,
                chunking_strategy=ChunkingStrategy[args.chunking],
                embedding_model=EmbeddingModel[args.embedding]
            )
            print("\nProcessing completed:")
            print(json.dumps(result, indent=2))
        else:
            print("No action specified. Use --help for usage information.")
            
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
    finally:
        kg_manager.shutdown()

if __name__ == "__main__":
    main()
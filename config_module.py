"""
Configuration module for Knowledge Graph Ingestion System
"""
from dataclasses import dataclass
from typing import List, Optional
import os
from enum import Enum


class LoaderType(Enum):
    URL = "url"
    PDF = "pdf" 
    DOCX = "docx"


class ChunkingStrategy(Enum):
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


class EmbeddingModel(Enum):
    TEXT_ADA_002 = "text-embedding-ada-002"
    BGE_LARGE = "bge-large-en-v1.5"
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class CassandraConfig:
    """Cassandra database configuration"""
    user: str
    password: str
    hosts: List[str]
    port: str = "9042"
    keyspace: Optional[str] = None


@dataclass
class AuthConfig:
    """Authentication configuration"""
    idaas_url: str
    api_url: str
    ads_id: str
    ads_password: str
    certificate_path: str
    bluetoken_cookie: str


@dataclass
class ModelConfig:
    """Model configuration"""
    kw_model_path: str
    model_name: str
    embedding_model: EmbeddingModel
    mode: str = "E1"


@dataclass
class UseCaseConfig:
    """Use case specific configuration"""
    use_case_id: str
    use_case_name: str
    stop_words: List[str]
    keyword_llm_prompt: str


@dataclass
class KGConfig:
    """Main configuration class for Knowledge Graph system"""
    cassandra: CassandraConfig
    auth: AuthConfig
    model: ModelConfig
    use_case: UseCaseConfig
    
    @classmethod
    def from_env(cls) -> 'KGConfig':
        """Create configuration from environment variables"""
        return cls(
            cassandra=CassandraConfig(
                user=os.getenv("CASSANDRA_USER", "test"),
                password=os.getenv("CASSANDRA_PASSWORD", ""),
                hosts=os.getenv("CASSANDRA_HOSTS", "localhost").split(","),
                port=os.getenv("CASSANDRA_PORT", "9042")
            ),
            auth=AuthConfig(
                idaas_url=os.getenv("IDAAS_URL", ""),
                api_url=os.getenv("API_URL", ""),
                ads_id=os.getenv("ADS_ID", ""),
                ads_password=os.getenv("ADS_PASSWORD", ""),
                certificate_path=os.getenv("CERT_PATH", "cert/genaiservices_allcerts.pem"),
                bluetoken_cookie=os.getenv("BLUETOKEN_COOKIE", "")
            ),
            model=ModelConfig(
                kw_model_path=os.getenv("KW_MODEL_PATH", "minilm-16-v2"),
                model_name=os.getenv("MODEL_NAME", "llama3-70b-instruct"),
                embedding_model=EmbeddingModel(os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))
            ),
            use_case=UseCaseConfig(
                use_case_id=os.getenv("USE_CASE_ID", "123"),
                use_case_name=os.getenv("USE_CASE_NAME", "default_usecase"),
                stop_words=os.getenv("STOP_WORDS", "ai,artificial").split(","),
                keyword_llm_prompt=os.getenv("KEYWORD_PROMPT", 
                    "Extract the top 5 most important keywords from the following user text. Do not include the following stop words:")
            )
        )


def get_default_config() -> KGConfig:
    """Get default configuration"""
    return KGConfig(
        cassandra=CassandraConfig(
            user="test",
            password="",
            hosts=["127.0.0.1"],
            port="9042"
        ),
        auth=AuthConfig(
            idaas_url="https://ssoisvc-dev.aexp.com/ssoi/signin",
            api_url="https://aidagenaiservices-dev.aexp.com/app/v1/openai/models/text-embedding-ada-002/embeddings",
            ads_id="",
            ads_password="",
            certificate_path="cert/genaiservices_allcerts.pem",
            bluetoken_cookie=""
        ),
        model=ModelConfig(
            kw_model_path="minilm-16-v2",
            model_name="llama3-70b-instruct",
            embedding_model=EmbeddingModel.TEXT_ADA_002
        ),
        use_case=UseCaseConfig(
            use_case_id="123",
            use_case_name="url_usecase_testing",
            stop_words=["ai", "aiforall", "artificial"],
            keyword_llm_prompt="Extract the top 5 most important keywords from the following user text. Do not include the following stop words:"
        )
    )

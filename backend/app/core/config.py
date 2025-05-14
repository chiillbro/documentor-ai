# from pydantic_settings import BaseSettings, SettingsConfigDict
# from typing import Optional


# class Settings(BaseSettings):
#     # Applications settings
#     APP_NAME: str = "Documentor API"
#     API_V1_STR: str = "/api/v1"
#     DEBUG_MODE: bool = False # Set to True for dev, False for prod via .env


#     # Database settings
#     DATABASE_URL: str = "postgresql://user:password@localhost:5432/documentor_db" # Example

#     DB_ECHO_LOG: bool = False # For SQLAlchemy logging, if used

#     # Embedding Model Settings
#     EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2" # Example


#     # LLM Settings
#     LLM_PROVIDER: str = "ollama" # 'ollama', 'openai', 'huggingface'
#     OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434" # If using Ollama locally or in another container
#     # OLLAMA_MODEL_NAME: Optional[str] = "llama3"
#     # OLLAMA_MODEL_NAME: Optional[str] = "mistral:7b-instruct-v0.2-q4_K_M"
#     # OLLAMA_MODEL_NAME: Optional[str] = "phi3:mini-4k-instruct-q4_K_M"
#     # OLLAMA_MODEL_NAME: Optional[str] = "phi3:mini"
#     OLLAMA_MODEL_NAME: Optional[str] = "tinyllama"
#     OPENAI_API_KEY: Optional[str] = None
#     OPENAI_MODEL_NAME: Optional[str] = "gpt-3.5-turbo"
#     # HUGGINGFACE_MODEL_NAME: Optional[str] = "google/flan-t5-base" # Example

#     # For JWT (if we add user accounts later)
#     # SECRET_KEY: str = "a_very_secret_key_change_me"
#     # ALGORITHM: str = "HS256"
#     # ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

#     # Model config for loading from .env file
#     model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# settings = Settings()


# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "DocuMentor AI"
    API_V1_STR: str = "/api/v1"
    DEBUG_MODE: bool = False

    DATABASE_URL: str = "postgresql://user:password@db:5432/documentor_db" # Used by Docker
    # For local dev without Docker, you might have:
    # DATABASE_URL_LOCAL: str = "postgresql://user:password@localhost:5433/documentor_db"


    DB_ECHO_LOG: bool = False

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: Optional[str] = "http://ollama:11434" # For Docker-to-Docker
    OLLAMA_MODEL_NAME: Optional[str] = "tinyllama" # Or tinyllama:1.1b-chat-v1.0-q4_0
    # OLLAMA_MODEL_NAME: Optional[str] = "phi3:mini" # If you got this working

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: Optional[str] = "gpt-3.5-turbo"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Derived DSNs for convenience, can be defined here or in rag_service.py
ASYNC_DB_URL = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
SYNC_DB_URL = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
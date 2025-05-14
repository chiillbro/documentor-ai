# ************************* Version 1 ******************************

# from fastapi import UploadFile, HTTPException
# from sqlalchemy.ext.asyncio import AsyncSession
# from typing import List, Tuple, Dict, Any, Optional
# import tempfile
# import os
# import uuid

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_openai import OpenAIEmbeddings # Alternative
# from langchain_postgres.vectorstores import PGVector # LangChain's PGVector integration
# from langchain_core.documents import Document
# from langchain.chains import RetrievalQA
# # from langchain_ollama import OllamaLLM
# # from langchain_ollama import Ollama # OLD IMPORT
# from langchain_community.llms import Ollama # NEW IMPORT - TRY THIS
# # from langchain_openai import ChatOpenAI # Alternative

# from app.core.config import settings
# from app.db.session import async_engine # For PGVector connection string if needed directly

# # --- Embedding Model Setup ---
# # Initialize embedding model once
# try:
#     print(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
#     embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
#     print("Embedding model initialized successfully.")
# except Exception as e:
#     print(f"Failed to initialize embedding model: {e}")
#     embedding_model = None # Handle this case gracefully or raise

# # --- LLM Setup ---
# # Initialize LLM once
# llm = None
# try:
#     print(f"Initializing LLM provider: {settings.LLM_PROVIDER}")
#     if settings.LLM_PROVIDER == "ollama":
#         if not settings.OLLAMA_BASE_URL or not settings.OLLAMA_MODEL_NAME:
#             raise ValueError("Ollama base URL or model name not configured.")
#         llm = Ollama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL_NAME)
#         print(f"Ollama LLM initialized with model: {settings.OLLAMA_MODEL_NAME}")
#     # Elif for OpenAI or other providers
#     # elif settings.LLM_PROVIDER == "openai":
#     #     if not settings.OPENAI_API_KEY:
#     #         raise ValueError("OpenAI API key not configured.")
#     #     llm = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL_NAME)
#     #     print(f"OpenAI LLM initialized with model: {settings.OPENAI_MODEL_NAME}")
#     else:
#         raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
#     # Test LLM (optional, good for debugging)
#     # print(f"Testing LLM: {llm.invoke('Why is the sky blue?')}")
# except Exception as e:
#     print(f"Failed to initialize LLM: {e}")
#     # Application should probably not start if LLM fails to init.
#     # Or handle queries gracefully.

# # --- PGVector Connection String ---
# # LangChain's PGVector needs a synchronous-style DSN, even if underlying operations are async.
# # Or it can manage its own connection pool.
# # Let's use the DATABASE_URL from settings, LangChain PGVector should handle it.
# # PGVector will create the table (collection) if it doesn't exist.
# # The connection string needs to be in the format:
# # "postgresql+psycopg2://user:password@host:port/database" for synchronous psycopg2
# # LangChain PGVector typically uses psycopg2 under the hood for its direct connections.
# # Let's ensure our settings.DATABASE_URL is compatible or construct one.
# # Our current settings.DATABASE_URL is "postgresql://user:password@db:5432/documentor_db"
# # We need to make it "postgresql+psycopg2://user:password@db:5432/documentor_db"
# # OR rely on LangChain's PGVector to use the async_engine if it supports it.
# # For simplicity with LangChain's PGVector, let's use the standard psycopg2 DSN.

# # Construct the DSN for LangChain's PGVector.
# # It expects a synchronous DSN even if our app uses async for FastAPI routes.
# PGVECTOR_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")

# # PGVECTOR_CONNECTION_STRING = settings.DATABASE_URL
# # or to be explicit for psycopg v3 if some tool requires it:
# # PGVECTOR_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")

# async def process_and_store_document(
#     db_session: AsyncSession, # FastAPI provides async session, but PGVector might use its own sync
#     file: UploadFile,
#     collection_name: str # This will be the table name for embeddings
# ) -> Dict[str, Any]:
#     if not embedding_model:
#         raise HTTPException(status_code=500, detail="Embedding model not initialized.")

#     try:
#         # Save UploadFile to a temporary file to be read by PyPDFLoader
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             content = await file.read()
#             tmp_file.write(content)
#             tmp_file_path = tmp_file.name
        
#         print(f"Processing PDF: {file.filename} from temp path: {tmp_file_path}")
#         loader = PyPDFLoader(tmp_file_path)
#         raw_documents = loader.load() # Returns a list of LangChain Document objects

#         if not raw_documents:
#             raise ValueError("PDF loader returned no documents.")

#         print(f"Loaded {len(raw_documents)} pages/documents from PDF.")

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100,
#             length_function=len,
#         )
#         split_documents = text_splitter.split_documents(raw_documents)
#         print(f"Split into {len(split_documents)} chunks.")

#         if not split_documents:
#             raise ValueError("Text splitter returned no chunks.")

#         # Add unique IDs to each chunk's metadata for better tracking if needed later.
#         # And associate with the original file.
#         # For MVP, filename can be a simple source.
#         doc_id = str(uuid.uuid4()) # Generate a unique ID for this uploaded document
#         for i, doc_chunk in enumerate(split_documents):
#             doc_chunk.metadata["source"] = file.filename # Original filename
#             doc_chunk.metadata["document_id"] = doc_id # Associate with uploaded document
#             doc_chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"


#         # Initialize PGVector store
#         # The 'collection_name' will be used as the table name for this set of embeddings.
#         # PGVector will create this table if it doesn't exist, along with an 'embedding' column of type vector.
#         print(f"Initializing PGVector store with collection: {collection_name} and connection: {PGVECTOR_CONNECTION_STRING}")
#         vector_store = PGVector(
#             connection_string=PGVECTOR_CONNECTION_STRING,
#             embedding_function=embedding_model,
#             collection_name=collection_name,
#             # distance_strategy=DistanceStrategy.COSINE, # Default is COSINE
#         )

#         # Add documents to the vector store
#         # This will generate embeddings and store them.
#         print(f"Adding {len(split_documents)} chunks to PGVector...")
#         vector_store.add_documents(split_documents)
#         print("Documents added to PGVector successfully.")

#         # Clean up temporary file
#         os.unlink(tmp_file_path)

#         return {
#             "id": doc_id,
#             "filename": file.filename,
#             "total_chunks": len(split_documents),
#             "collection_name": collection_name
#         }

#     except Exception as e:
#         if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
#             os.unlink(tmp_file_path) # Ensure temp file is deleted on error
#         print(f"Error in process_and_store_document: {e}") # Log detailed error
#         raise HTTPException(status_code=500, detail=f"Internal server error during document processing: {type(e).__name__} - {e}")


# async def query_document_with_rag(
#     db_session: AsyncSession, # FastAPI provides async session
#     question: str,
#     collection_name: str,
#     document_id: Optional[str] = None # For future filtering by specific document
# ) -> Tuple[str, List[str]]:
#     if not embedding_model:
#         raise HTTPException(status_code=500, detail="Embedding model not initialized.")
#     if not llm:
#         raise HTTPException(status_code=500, detail="LLM not initialized.")

#     try:
#         print(f"Initializing PGVector store for querying collection: {collection_name}")
#         vector_store = PGVector(
#             connection_string=PGVECTOR_CONNECTION_STRING,
#             embedding_function=embedding_model,
#             collection_name=collection_name
#         )

#         # Create a retriever
#         # For MVP, retrieve from all documents in the collection.
#         # Later, you can add filters to the retriever if document_id is provided.
#         # Example for filtering (would require metadata `document_id` to be set during ingestion):
#         # search_kwargs = {}
#         # if document_id:
#         #     search_kwargs = {'filter': {'document_id': document_id}}
#         # retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        
#         retriever = vector_store.as_retriever(
#             search_type="similarity", # Can also be "mmr"
#             search_kwargs={"k": 3} # Retrieve top 3 most similar chunks
#         )
#         print("Retriever initialized.")

#         # Create RetrievalQA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff", # Options: "stuff", "map_reduce", "refine", "map_rerank"
#             retriever=retriever,
#             return_source_documents=True # To get the chunks used for the answer
#         )
#         print("QA chain initialized.")

#         print(f"Invoking QA chain with question: {question}")
#         result = qa_chain.invoke({"query": question}) # The input key is "query" for RetrievalQA
        
#         answer = result.get("result", "No answer found.")
#         source_documents = result.get("source_documents", [])
        
#         source_chunks_content = [doc.page_content for doc in source_documents]

#         print(f"Answer: {answer}")
#         print(f"Source chunks ({len(source_chunks_content)}): {source_chunks_content[:1]}...") # Print first chunk

#         return answer, source_chunks_content

#     except Exception as e:
#         print(f"Error in query_document_with_rag: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error during query: {type(e).__name__} - {e}")



# ************************* Version 2 ******************************

# from fastapi import UploadFile, HTTPException
# from sqlalchemy.ext.asyncio import AsyncSession
# from typing import List, Tuple, Dict, Any, Optional
# import tempfile
# import os
# import uuid

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres.vectorstores import PGVector # This is the key import
# from langchain_core.documents import Document
# from langchain.chains import RetrievalQA
# # from langchain_community.llms import Ollama
# from langchain_ollama import OllamaLLM

# from app.core.config import settings
# # from app.db.session import async_engine # We might not need to pass the engine directly

# # --- Embedding Model Setup ---
# try:
#     print(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
#     embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
#     print("Embedding model initialized successfully.")
# except Exception as e:
#     print(f"Failed to initialize embedding model: {e}")
#     embedding_model = None

# # --- LLM Setup ---
# llm = None
# try:
#     print(f"Initializing LLM provider: {settings.LLM_PROVIDER}")
#     if settings.LLM_PROVIDER == "ollama":
#         if not settings.OLLAMA_BASE_URL or not settings.OLLAMA_MODEL_NAME:
#             raise ValueError("Ollama base URL or model name not configured.")
#         llm = OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL_NAME)
#         print(f"Ollama LLM initialized with model: {settings.OLLAMA_MODEL_NAME}")
#     else:
#         raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
# except Exception as e:
#     print(f"Failed to initialize LLM: {e}")


# # Construct the DSN for LangChain's PGVector.
# # It expects a synchronous DSN. Our settings.DATABASE_URL is already in this format.
# # If PGVector internally uses psycopg2 or a compatible driver, this should work.
# # The key is that the ENVIRONMENT VARIABLE for DATABASE_URL must be accessible to LangChain.
# # OR, we can pass a connection directly.
# # Let's ensure settings.DATABASE_URL is what PGVector expects or can derive from.
# # settings.DATABASE_URL should be like: "postgresql://user:password@host:port/database"
# # LangChain's PGVector might also look for standard PostgreSQL env vars like PGHOST, PGUSER, etc.

# # For LangChain's PGVector, the connection string is often implicitly handled if standard
# # PostgreSQL environment variables are set, or it's passed differently.
# # The latest `langchain-postgres` might prefer using a `Connection` object or
# # have specific parameters for the engine.

# # Let's try the `PGVector.from_documents` or direct instantiation method
# # that relies on environment variables or a simpler connection setup.

# async def process_and_store_document(
#     db_session: AsyncSession, # FastAPI provides async session, may not be used by PGVector directly
#     file: UploadFile,
#     collection_name: str
# ) -> Dict[str, Any]:
#     if not embedding_model:
#         raise HTTPException(status_code=500, detail="Embedding model not initialized.")

#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             content = await file.read()
#             tmp_file.write(content)
#             tmp_file_path = tmp_file.name
        
#         print(f"Processing PDF: {file.filename} from temp path: {tmp_file_path}")
#         loader = PyPDFLoader(tmp_file_path)
#         raw_documents = loader.load()

#         if not raw_documents:
#             os.unlink(tmp_file_path)
#             raise ValueError("PDF loader returned no documents.")
#         print(f"Loaded {len(raw_documents)} pages/documents from PDF.")

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         split_documents = text_splitter.split_documents(raw_documents)
#         print(f"Split into {len(split_documents)} chunks.")

#         if not split_documents:
#             os.unlink(tmp_file_path)
#             raise ValueError("Text splitter returned no chunks.")

#         doc_id = str(uuid.uuid4())
#         for i, doc_chunk in enumerate(split_documents):
#             doc_chunk.metadata["source"] = file.filename
#             doc_chunk.metadata["document_id"] = doc_id
#             doc_chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"

#         # --- Corrected PGVector Initialization ---
#         # Method 1: Using PGVector.from_documents (often preferred as it handles setup)
#         # This method requires the connection string to be available, often via environment variables
#         # that psycopg2 (which PGVector uses underneath) can pick up, or by setting
#         # the connection_string parameter if the specific from_documents method supports it.
#         # The `connection` parameter is the modern way for langchain-postgres.
#         print(f"Attempting to add {len(split_documents)} chunks to PGVector collection: {collection_name}")
#         print(f"Using connection string for PGVector (derived from settings): {settings.DATABASE_URL}")

#         # PGVector needs a synchronous-style DSN for its internal psycopg2 usage
#         # Our settings.DATABASE_URL is "postgresql://user:password@db:5432/documentor_db"
#         # For psycopg2, it should be "postgresql+psycopg2://user:password@db:5432/documentor_db"
#         # However, langchain-postgres might abstract this. Let's test.
        
#         # The `connection` parameter is standard now for langchain-postgres
#         vector_store = PGVector.from_documents(
#             embedding=embedding_model,
#             documents=split_documents,
#             collection_name=collection_name,
#             connection=settings.DATABASE_URL, # Pass the async-style URL; langchain-postgres should handle it
#                                                 # Or it might need the sync DSN "postgresql+psycopg2://..."
#             # Alternatively, if from_documents doesn't take `connection` directly
#             # you might need to create a PGVector instance first and then add.
#             # Preinitialize collection is also an option for more control.
#         )
#         # If `from_documents` doesn't work with `connection` directly,
#         # an alternative instantiation and add:
#         # store = PGVector(
#         #     collection_name=collection_name,
#         #     connection_string=settings.DATABASE_URL, # Try with the direct URL
#         #     embedding_function=embedding_model,
#         # )
#         # await run_in_threadpool(store.add_documents, split_documents) # if add_documents is sync

#         print("Documents presumably added to PGVector successfully.")
#         os.unlink(tmp_file_path)

#         return {
#             "id": doc_id,
#             "filename": file.filename,
#             "total_chunks": len(split_documents),
#             "collection_name": collection_name
#         }

#     except Exception as e:
#         if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
#             os.unlink(tmp_file_path)
#         print(f"Error in process_and_store_document: {type(e).__name__} - {str(e)}")
#         import traceback
#         traceback.print_exc() # Print full traceback for debugging
#         raise HTTPException(status_code=500, detail=f"Internal server error during document processing: {type(e).__name__} - {e}")


# async def query_document_with_rag(
#     db_session: AsyncSession,
#     question: str,
#     collection_name: str,
#     document_id: Optional[str] = None
# ) -> Tuple[str, List[str]]:
#     if not embedding_model:
#         raise HTTPException(status_code=500, detail="Embedding model not initialized.")
#     if not llm:
#         raise HTTPException(status_code=500, detail="LLM not initialized.")

#     try:
#         print(f"Initializing PGVector store for querying collection: {collection_name}")
#         # --- Corrected PGVector Initialization for Querying ---
#         # When retrieving, you instantiate PGVector and then use it as a retriever.
#         vector_store = PGVector(
#             collection_name=collection_name,
#             connection=settings.DATABASE_URL, # Pass the async-style URL
#             embedding_function=embedding_model,
#             # distance_strategy=DistanceStrategy.EUCLIDEAN, # default is COSINE
#         )
#         print("PGVector store for querying initialized.")

#         retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#         print("Retriever initialized.")

#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True
#         )
#         print("QA chain initialized.")

#         print(f"Invoking QA chain with question: {question}")
#         # If qa_chain.invoke is synchronous, wrap it for async FastAPI
#         # from fastapi.concurrency import run_in_threadpool
#         # result = await run_in_threadpool(qa_chain.invoke, {"query": question})
#         result = qa_chain.invoke({"query": question}) # Check if invoke is async or sync for your LangChain version

#         answer = result.get("result", "No answer found.")
#         source_documents = result.get("source_documents", [])
#         source_chunks_content = [doc.page_content for doc in source_documents]

#         print(f"Answer: {answer}")
#         # print(f"Source chunks ({len(source_chunks_content)}): {source_chunks_content[:1]}...")

#         return answer, source_chunks_content

#     except Exception as e:
#         print(f"Error in query_document_with_rag: {type(e).__name__} - {str(e)}")
#         import traceback
#         traceback.print_exc() # Print full traceback for debugging
#         raise HTTPException(status_code=500, detail=f"Internal server error during query: {type(e).__name__} - {e}")




# ************************* Version 3 ******************************

# import logging
# from app.core.logging_config import get_logger
# from app.core.exceptions import DocumentProcessingException, QueryProcessingException, LLMConnectionException

# from fastapi import UploadFile, HTTPException
# from fastapi.concurrency import run_in_threadpool
# from sqlalchemy.ext.asyncio import AsyncSession
# from typing import List, Tuple, Dict, Any, Optional
# import tempfile
# import os
# import uuid

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres.vectorstores import PGVector
# from langchain_core.documents import Document
# from langchain.chains import RetrievalQA
# from langchain_ollama import OllamaLLM # Corrected LLM import
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnableConfig # For passing callbacks/configs to async operations
# from langchain_core.callbacks import AsyncCallbackHandler # For custom streaming logic if needed
# from typing import AsyncGenerator # For type hinting the streaming function


# import ollama

# import datetime

# from app.core.config import settings


# logger = get_logger(__name__)
# # or logger = logging.getLogger(__name__)

# # --- Embedding Model Setup ---
# try:
#     # print(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
#     logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
#     # This is our embedding model instance
#     embeddings_for_pgvector = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
#     # print("Embedding model initialized successfully.")
#     logger.info("Embedding model initialized successfully.")
# except Exception as e:
#     # print(f"Failed to initialize embedding model: {e}")
#     logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
#     embeddings_for_pgvector = None

# # --- LLM Setup ---
# llm = None
# try:
#     # print(f"Initializing LLM provider: {settings.LLM_PROVIDER}")
#     logger.info(f"Initializing LLM provider: {settings.LLM_PROVIDER}")
#     if settings.LLM_PROVIDER == "ollama":
#         if not settings.OLLAMA_BASE_URL or not settings.OLLAMA_MODEL_NAME:
#             raise ValueError("Ollama base URL or model name not configured.")
#         # Use the corrected OllamaLLM class
#         llm = OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL_NAME)
#         # print(f"Ollama LLM initialized with model: {settings.OLLAMA_MODEL_NAME}")
#         logger.info(f"Ollama LLM initialized with model: {settings.OLLAMA_MODEL_NAME}")
#     else:
#         raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
# except Exception as e:
#     # print(f"Failed to initialize LLM: {e}")
#     logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
#     raise LLMConnectionException(f"Failed to initialize LLM: {str(e)}")


# # If you need a custom callback handler for more control over streaming (optional for now)
# # class MyCustomStreamingCallback(AsyncCallbackHandler):
# #     async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
# #         print(f"Streamed token: {token}") # Example: log the token


# ASYNC_PG_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
# SYNC_PG_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")

# # ... (embedding_model and llm setup remain the same) ...

# # THIS FUNCTION MUST BE SYNCHRONOUS because PGVector.create_collection is synchronous
# # and we want run_in_threadpool to handle its blocking nature.
# def _ensure_collection_and_extension_sync(collection_name: str):
#     """
#     Synchronous helper to be run in a threadpool.
#     Ensures the pgvector extension exists (though init_db should do this)
#     and that the specific collection (table) is created.
#     Uses a synchronous DSN.
#     """
#     logger.info(f"Attempting to ensure (sync) PGVector collection '{collection_name}' and extension.")
#     try:
#         # The init_db in main.py already attempts to CREATE EXTENSION IF NOT EXISTS vector;
#         # This call here is primarily for PGVector to create its specific table.
#         # PGVector.create_collection also checks for and tries to create the extension.
#         PGVector.create_collection(
#             collection_name,
#             connection=SYNC_PG_CONNECTION_STRING, # IMPORTANT: Use sync DSN
#             embeddings=embeddings_for_pgvector
#         )
#         logger.info(f"Synchronously ensured PGVector collection '{collection_name}' exists or was created.")
#     except Exception as e:
#         logger.error(f"Error during sync _ensure_collection_and_extension_sync for '{collection_name}': {e}", exc_info=True)
#         # Depending on the error, you might want to re-raise or handle.
#         # If it's "relation already exists", it's often fine.
#         # For a MissingGreenlet here, it would indicate an issue with how PGVector.create_collection uses the sync DSN.
#         # However, with a sync DSN (psycopg2), MissingGreenlet should not occur from create_collection itself.
#         # Re-raise to make it visible if it's an unexpected error.
#         raise DocumentProcessingException(f"Failed to ensure collection {collection_name}: {str(e)}")


# async def _get_vector_store(collection_name: str) -> PGVector:
#     """
#     Helper to initialize PGVector.
#     1. Ensures collection & extension exist (synchronously in a thread).
#     2. Instantiates PGVector for async operations (synchronously in a thread for safety).
#     """
#     logger.info(f"Getting PGVector store for collection: {collection_name}")

#     # Step 1: Ensure the collection table and pgvector extension are created.
#     # This is a synchronous operation, so run it in a threadpool.
#     await run_in_threadpool(_ensure_collection_and_extension_sync, collection_name)
    
#     # Step 2: Instantiate PGVector for async use.
#     # The PGVector constructor itself is synchronous. Even if it does some light DB checks
#     # with the ASYNC_PG_CONNECTION_STRING, running its instantiation in a threadpool
#     # should provide the correct context for any greenlet bridging if needed.
#     def _instantiate_pgvector_for_async_use():
#         logger.info(f"Instantiating PGVector with ASYNC_PG_CONNECTION_STRING for '{collection_name}'")
#         store = PGVector(
#             collection_name,
#             connection=ASYNC_PG_CONNECTION_STRING, # DSN for asyncpg
#             embeddings=embeddings_for_pgvector,
#             # For async_mode, some vector stores have an explicit param. PGVector infers from DSN
#             # and expects its .a* methods to be used.
#         )
#         # After instantiation with an async DSN, subsequent .a* methods on the 'store' instance
#         # should use the asyncpg driver correctly. The MissingGreenlet error was likely
#         # from synchronous setup methods being called internally during __init__ or first async use
#         # without the protection of run_in_threadpool for *those specific internal sync calls*.
#         return store

#     store = await run_in_threadpool(_instantiate_pgvector_for_async_use)
    
#     logger.info(f"PGVector store for collection '{collection_name}' instantiated for async use.")
#     return store



# async def process_and_store_document(
#     db_session: AsyncSession, # This session is for your FastAPI app, PGVector will manage its own
#     file: UploadFile,
#     collection_name: str
# ) -> Dict[str, Any]:
#     if not embeddings_for_pgvector: # Check the renamed embedding model variable
#         # This check is good, but if it fails at startup, the app might not even reach here.
#         # Consider a health check endpoint.
#         raise DocumentProcessingException(detail="Embedding model not initialized. Cannot process document.")
#         # raise HTTPException(status_code=500, detail="Embedding model not initialized.")

#     tmp_file_path = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             content = await file.read() # FastAPI's UploadFile.read() is async
#             tmp_file.write(content)
#             tmp_file_path = tmp_file.name
        
#         logger.info(f"Processing PDF: {file.filename} from temp path: {tmp_file_path}")
#         loader = PyPDFLoader(tmp_file_path)
#         # raw_documents = loader.load()

#         # Use aload() for async loading
#         raw_documents = await loader.aload() # native async
#         # raw_documents = await run_in_threadpool(loader.load)

#         if not raw_documents:
#             # os.unlink(tmp_file_path)
#             raise ValueError("PDF loader returned no documents.")
#         # print(f"Loaded {len(raw_documents)} pages/documents from PDF.")
#         logger.info(f"Loaded {len(raw_documents)} pages/documents from PDF.")
#         # RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) is reasonable. 
#         # If chunks are too large, important info might be missed by the LLM. 
#         # If too small, context is lost.
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         # split_documents = text_splitter.split_documents(raw_documents)
#         split_documents = await run_in_threadpool(text_splitter.split_documents, raw_documents)


#         # split_documents is usually CPU-bound, but if an async version exists for the splitter, use it.
#         # For RecursiveCharacterTextSplitter, split_documents is synchronous.
#         # If it were a major bottleneck (unlikely for typical PDFs), then run_in_threadpool.
#         # For now, direct call is fine for this specific splitter.
#         # split_documents = text_splitter.split_documents(raw_documents)
#         logger.info(f"Split into {len(split_documents)} chunks.")

#         if not split_documents:
#             # os.unlink(tmp_file_path)
#             raise ValueError("Text splitter returned no chunks.")

#         doc_id = str(uuid.uuid4())
#         for i, doc_chunk in enumerate(split_documents):
#             doc_chunk.metadata["source"] = file.filename
#             doc_chunk.metadata["document_id"] = doc_id
#             doc_chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"


#         # logger.info(f"Attempting to add {len(split_documents)} chunks to PGVector collection: {collection_name}")
#         # logger.info(f"Using connection string for PGVector (derived from settings): {settings.DATABASE_URL}")
        

#         # ** Version 1 **
#         # vector_store = PGVector.from_documents(
#         #     embeddings=embeddings_for_pgvector,  # Corrected: Use 'embeddings'
#         #     documents=split_documents,
#         #     collection_name=collection_name,
#         #     connection=settings.DATABASE_URL,
#         #     # use_jsonb=True, # Optional, can be useful for metadata flexibility
#         #     # pre_delete_collection=False # Set to True if you want to clear before adding
#         # )


#         # ** Version 2 **
#         # await run_in_threadpool(
#         #     PGVector.from_documents,
#         #     embeddings=embeddings_for_pgvector,  # Corrected: Use 'embeddings'
#         #     documents=split_documents, # Pass the documents directly
#         #     collection_name=collection_name, # Pass the collection name directly
#         #     connection=settings.DATABASE_URL, # Pass the connection string directly
#         # )


#         # ** Version 3 **
#         # Define a helper function to be run in the threadpool
#         # This makes argument passing to PGVector.from_documents very clear.
#         # def _add_documents_to_pgvector():
#         #     return PGVector.from_documents(
#         #         documents=split_documents,
#         #         embedding=embeddings_for_pgvector,  # Use 'embedding' (singular)
#         #         collection_name=collection_name,
#         #         connection=settings.DATABASE_URL, # LangChain PGVector uses 'connection' not 'connection_string'
#         #                                           # when passed to from_documents. Internally it might construct
#         #                                           # a connection_string if a direct 'connection' object isn't given.
#         #                                           # The DATABASE_URL from settings should work here.
#         #         # use_jsonb=True, # Optional
#         #         # pre_delete_collection=False # Optional
#         #     )

#         # await run_in_threadpool(_add_documents_to_pgvector) # Call the helper

    
#         # ** Version 4 **
#         # # PGVector.from_documents is a class method that does a few things including
#         # # creating an instance and then calling add_documents.
#         # # We want its async counterpart if available, or use aadd_documents on an instance.
        
#         # # Option 1: Create store then aadd_documents (more control, explicit async)
#         # # vector_store = PGVector(
#         # #     collection_name=collection_name,
#         # #     connection=settings.DATABASE_URL, # DSN for synchronous psycopg2 connection
#         # #     embedding=embeddings_for_pgvector,
#         # # )
#         # # await vector_store.aadd_documents(split_documents)

#         # # Option 2: Check if there's an `afrom_documents` or if `from_documents` can be awaited
#         # # As of recent LangChain versions, `from_documents` itself isn't async.
#         # # We need to use `aadd_documents` on an instance for async behavior or wrap `from_documents`
#         # # The helper function approach with run_in_threadpool for from_documents was okay.
#         # # Let's stick to the more explicit async `aadd_documents` after creating the store instance.
#         # # The PGVector store instantiation itself is usually quick/synchronous.

#         # def _create_and_add_sync():
#         #     # This part is tricky because PGVector.from_documents does both creation AND adding.
#         #     # And the connection it uses internally might be synchronous.
#         #     # For full async, it's better to manage the store instance and use its async methods.
#         #     # However, the simplest for now might be to keep the threadpool for `from_documents`
#         #     # if `aadd_documents` after manual instantiation proves complex with connection handling.

#         #     # Let's try direct instantiation and then `aadd_documents`.
#         #     # The PGVector class itself takes a `connection_string` (DSN for psycopg2)
#         #     # or it can use an existing `psycopg2.connection` or `asyncpg.Connection`.
#         #     # For `aadd_documents`, it needs to be able to execute async.
            
#         #     # The PGVector constructor itself is synchronous.
#         #     store = PGVector(
#         #         collection_name=collection_name,
#         #         connection=settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://"), # Explicit sync DSN
#         #         embeddings=embeddings_for_pgvector,
#         #     )
#         #     # `add_documents` is synchronous. We need its async counterpart on the store instance.
#         #     # This implies that for `aadd_documents` to work truly async, the underlying
#         #     # connection mechanism of PGVector must support it.
#         #     # LangChain's PGVector can be a bit tricky with pure async.
#         #     # Let's assume for a moment `aadd_documents` is available and works with the sync DSN.
#         #     # If not, threadpool for the sync `add_documents` is the fallback.
#         #     return store # Return the store, then call aadd_documents

#         # # This is a pattern: synchronous instantiation, then async method call
#         # vector_store_instance = _create_and_add_sync() # This is still synchronous instantiation.
#         # await vector_store_instance.aadd_documents(split_documents) # This is the async call


#         # ** Version 5 **
#          # Get the vector store (this ensures collection/extension exists and instantiates PGVector)
#         vector_store = await _get_vector_store(collection_name)
        
#         logger.info(f"Adding {len(split_documents)} chunks to PGVector collection: {collection_name}")
#         # Now use the async method on the prepared vector_store instance
#         await vector_store.aadd_documents(split_documents)
        
#         logger.info("Documents presumably added to PGVector successfully.")
#         # os.unlink(tmp_file_path) # Moved to finally block


#         # Prepare a more structured response
#         # Generate a timestamp in ISO format
#         current_utc_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
#         return {
#             "id": doc_id, # The UUID generated for this document processing
#             "filename": file.filename,
#             "content_type": file.content_type,
#             "size": file.size,
#             "total_chunks": len(split_documents),
#             "collection_name": collection_name,
#             "uploaded_at": current_utc_timestamp,
#         }

#     # except Exception as e:
#     #     # if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
#     #     #     os.unlink(tmp_file_path)
#     #     # print(f"Error in process_and_store_document: {type(e).__name__} - {str(e)}")
#     #     logger.error(f"Error in process_and_store_document: {type(e).__name__} - {str(e)}", exc_info=True)
#     #     # import traceback
#     #     # traceback.print_exc()
#     #     # raise HTTPException(status_code=500, detail=f"Internal server error during document processing: {type(e).__name__} - {e}")
#     #     raise DocumentProcessingException(f"Internal server error during document processing: {type(e).__name__} - {str(e)}")
#     except ValueError as ve: # Catch specific known errors first
#         logger.warning(f"Value error during document processing: {ve}", exc_info=True)
#         if tmp_file_path and os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
#         raise DocumentProcessingException(detail=str(ve))

#     except Exception as e:
#         logger.error(f"Error in process_and_store_document: {type(e).__name__} - {str(e)}", exc_info=True)
#         if tmp_file_path and os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
#         raise DocumentProcessingException(detail=f"Internal server error during document processing: {type(e).__name__} - {str(e)}")

#     finally:
#         if tmp_file_path and os.path.exists(tmp_file_path):
#             os.unlink(tmp_file_path) # Ensure temp file is always deleted

# # This is the ASYNC DSN we created in db/session.py
# # ASYNC_PG_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# async def stream_document_with_rag( # New function for streaming
#     db_session: AsyncSession,
#     question: str,
#     collection_name: str,
#     document_id: Optional[str] = None
# ) -> AsyncGenerator[str, None]: # Yields strings (chunks of the answer)
#     if not embeddings_for_pgvector:
#         logger.error("Streaming query failed: Embedding model not initialized.")
#         # For an AsyncGenerator, raising an exception is one way to signal error
#         # Or yield a specific error token if the client is designed to handle it.
#         raise QueryProcessingException(detail="Embedding model not initialized.")
#     if not llm:
#         logger.error("Streaming query failed: LLM not initialized.")
#         raise QueryProcessingException(detail="LLM not initialized.")

#     try:
#         logger.info(f"Initializing PGVector store for streaming query on collection: {collection_name}")
#         # vector_store = PGVector(
#         #     collection_name=collection_name,
#         #     connection_string=settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://"),
#         #     embedding=embeddings_for_pgvector,
#         # )

#         # # Define a helper function for synchronous PGVector instantiation
#         # def _init_vector_store():
#         #     return PGVector(
#         #         collection_name=collection_name,
#         #         connection=ASYNC_PG_CONNECTION_STRING, # Use 'connection'
#         #         embeddings=embeddings_for_pgvector, # Use 'embedding' (singular)
#         #     )
        
#         # vector_store = await run_in_threadpool(_init_vector_store)

#          # When intending to use async methods, provide an asyncpg-compatible DSN.
#         # The PGVector store should then internally use or create an async engine.
#         # vector_store = PGVector(
#         #     collection_name=collection_name,
#         #     connection=ASYNC_PG_CONNECTION_STRING, # Use the async DSN
#         #     embeddings=embeddings_for_pgvector,    # Use 'embedding' (singular)
#         # )
#         # # Forcing the async initialization if it's lazy
#         # await vector_store.afrom_documents([], embedding=embeddings_for_pgvector, connection=ASYNC_PG_CONNECTION_STRING, collection_name=collection_name + "_dummy_init", pre_delete_collection=True) # Hacky way to force async init
#         # A better way might be to see if PGVector can take an async_engine directly.
#         # Or await vector_store.aensure_collection() # if such a method exists for async setup


#         vector_store = await _get_vector_store(collection_name)
#         logger.info("PGVector store for streaming query initialized.")

#         retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Or your preferred k
#         logger.info("Retriever for streaming initialized.")

#         prompt_template_str = """You are an information extraction assistant.
#         Your task is to answer questions based *only* on the provided "Context" below.
#         Do not use any prior knowledge.
#         If the information to answer the question is not in the "Context", respond with "I don't know based on the provided document."
#         Answer directly about the subject mentioned in the question, using information from the context.
#         Be concise.

#         Context:
#         {context}

#         Question: {question}

#         Answer based on context:"""
#         QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)

#          # RetrievalQA instantiation is synchronous
#         def _create_qa_chain_sync():
#             return RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=retriever,
#                 chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#             )
#         qa_chain = await run_in_threadpool(_create_qa_chain_sync)

#         # qa_chain = RetrievalQA.from_chain_type(
#         #     llm=llm,
#         #     chain_type="stuff",
#         #     retriever=retriever,
#         #     # return_source_documents=True, # For streaming, source_documents usually come after stream completion or via separate call
#         #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#         # )
#         logger.info("Streaming QA chain initialized.")
#         logger.info(f"Streaming QA chain with question: {question}")

#         # Use the `astream` method of the chain for LangChain >0.1.0
#         # The output of astream is an AsyncGenerator of Chunks or Dicts.
#         # For RetrievalQA, it often yields dicts like {"result": "chunk of text"}
#         # or directly content chunks depending on the LLM and chain.
        
#         # For LangChain's astream on RetrievalQA, the output can be more complex.
#         # We might need to process it to extract just the answer tokens.
#         # A simpler approach for some LLM wrappers is to use llm.astream directly if
#         # we construct the prompt manually after retrieval.

#         # Let's try the chain's astream first.
#         # The `ainvoke` can also take a `RunnableConfig` with callbacks for streaming.
#         # config = RunnableConfig(callbacks=[MyCustomStreamingCallback()]) # Optional custom callback

#         async for chunk in qa_chain.astream({"query": question}):
#             # The structure of 'chunk' depends on the chain and LLM.
#             # For RetrievalQA, it might be a dict. We need the 'result' or 'answer' part.
#             # Or if the LLM streams content directly, it could be a AIMessageChunk.
#             # Let's inspect what `chunk` is.
#             # logger.debug(f"Stream chunk: {chunk}")
#             if isinstance(chunk, dict) and "result" in chunk:
#                 yield chunk["result"]
#             elif hasattr(chunk, 'content'): # e.g. AIMessageChunk
#                  yield chunk.content
#             # Add other conditions based on observed chunk structure if necessary

#         # After the stream, if you need source documents (RetrievalQA might not stream them alongside tokens)
#         # You might need to run the retriever separately or a non-streaming invoke for sources.
#         # For MVP streaming, focus on getting the answer text to stream.
#         # We can enhance to send source_documents at the end or via separate mechanism.

#     except ollama.ResponseError as ore:
#         logger.error(f"Ollama API Response Error during streaming: {ore.error} (Status: {ore.status_code})", exc_info=True)
#         yield f"[ERROR: LLM communication error: {ore.error}]" # Yield error string
#     except Exception as e:
#         logger.error(f"Error in stream_document_with_rag: {type(e).__name__} - {str(e)}", exc_info=True)
#         yield f"[ERROR: Internal server error during query: {type(e).__name__} - {str(e)}]" # Yield error string


# async def query_document_with_rag(  # The non-streaming version
#     db_session: AsyncSession, # This session is for your FastAPI app, PGVector will manage its own
#     question: str,
#     collection_name: str,
#     document_id: Optional[str] = None
# ) -> Tuple[str, List[str]]:
#     # if not embeddings_for_pgvector: # Check the renamed embedding model variable
#     #     raise HTTPException(status_code=500, detail="Embedding model not initialized.")
#     # if not llm:
#     #     raise HTTPException(status_code=500, detail="LLM not initialized.")
#     if not embeddings_for_pgvector:
#         raise QueryProcessingException(detail="Embedding model not initialized.")
#     if not llm:
#         raise QueryProcessingException(detail="LLM not initialized.")


#     try:
#         # print(f"Initializing PGVector store for querying collection: {collection_name}")
#         logger.info(f"Initializing PGVector store for querying collection: {collection_name}")
        
#         # ** Version 1 **
#         # vector_store = PGVector(
#         #     collection_name=collection_name,
#         #     connection=settings.DATABASE_URL,
#         #     embeddings=embeddings_for_pgvector, # Corrected: Use 'embeddings'
#         #     # use_jsonb=True # if you used it during from_documents
#         # )



#         # ** Version 2 **
#         # Define a helper function for synchronous PGVector instantiation
#         # def _init_vector_store():
#         #     return PGVector(
#         #         collection_name=collection_name,
#         #         connection=settings.DATABASE_URL, # Use 'connection'
#         #         embeddings=embeddings_for_pgvector, # Use 'embedding' (singular)
#         #     )
        
#         # vector_store = await run_in_threadpool(_init_vector_store)


#         # ** Version 3 **
#         # # PGVector instantiation is synchronous
#         # vector_store = PGVector(
#         #     collection_name=collection_name,
#         #     connection=settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://"),
#         #     embeddings=embeddings_for_pgvector,
#         # )


#         # ** Version 4 **

#         vector_store = await _get_vector_store(collection_name)


#         logger.info("PGVector store for querying initialized.")

#         # search_kwargs={"k": 3}. This means 3 chunks are retrieved. 
#         # For tinyllama with its small context window, sending too much context might confuse it or cause it to ignore parts. 
#         # Experiment with k=1 or k=2.
#         retriever = vector_store.as_retriever(search_kwargs={"k": 2})
#         logger.info("Retriever initialized.")

#         # ** Version 1 **
#         # prompt_template = """Use the following pieces of context to answer the question at the end.
#         # If you don't know the answer from the context, just say that you don't know, do not try to make up an answer.
#         # Be concise and stick to the information found in the context.

#         # Context: {context}

#         # Question: {question}

#         # Helpful Answer:"""

#         # ** Version 2 **
#         prompt_template = """You are an information extraction assistant.
#         Your task is to answer questions based *only* on the provided "Context" below.
#         Do not use any prior knowledge.
#         If the information to answer the question is not in the "Context", respond with "I don't know based on the provided document."
#         Answer directly about the subject mentioned in the question, using information from the context.
#         Be concise.

#         Context:
#         {context}

#         Question: {question}

#         Answer based on context:"""
#         QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
#         # **Version 1 **
#          # qa_chain = RetrievalQA.from_chain_type(
#         #     llm=llm,
#         #     chain_type="stuff",
#         #     retriever=retriever,
#         #     return_source_documents=True,
#         #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Pass the custom prompt
#         # )

#         # ** Version 2 **
#         # Define a helper for chain creation (optional, but good if complex)
#         def _create_qa_chain():
#             return RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=retriever,
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#             )
        
#         qa_chain = await run_in_threadpool(_create_qa_chain) # if chain creation is blocking


#         # ** Version 3 **
#         # # RetrievalQA instantiation is synchronous
#         # qa_chain = RetrievalQA.from_chain_type(
#         #     llm=llm,
#         #     chain_type="stuff",
#         #     retriever=retriever,
#         #     return_source_documents=True,
#         #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#         # )

#         logger.info("QA chain initialized.")
#         logger.info(f"Invoking QA chain with question: {question}")
        
#         # If qa_chain.invoke is synchronous, wrap it for async FastAPI
#         # from fastapi.concurrency import run_in_threadpool
#         # result = await run_in_threadpool(qa_chain.invoke, {"query": question})
#         # For now, let's assume it's okay for testing or that invoke handles it
#         # result = qa_chain.invoke({"query": question})
#         result = await run_in_threadpool(qa_chain.invoke, {"query": question})
#         # result = await qa_chain.ainvoke({"query": question})

#         answer = result.get("result", "No answer found.")
#         source_documents = result.get("source_documents", [])
#         source_chunks_content = [doc.page_content for doc in source_documents]

#         # Before invoking the chain, let's see what context it's getting
#         # This requires getting the retriever to run first, which qa_chain does internally
#         # A simpler way for now is to log the source_documents *after* the result.
#         # But for deep debugging, you might run retriever.get_relevant_documents(question)
#         # and log that before forming the chain.

#         # # For now, ensure this line is uncommented after you get the result:
#         # logger.info(f"Source chunks for question '{question}': {source_chunks_content}")

#         logger.info(f"LLM Answer: {answer}")
#         # for i, chunk_content in enumerate(source_chunks_content):
#         #     logger.info(f"Source Chunk {i+1} for question '{question}':\n{chunk_content}\n--------------------")

#         return answer, source_chunks_content

#     except ollama.ResponseError as ore: # Specific error from Ollama client
#         logger.error(f"Ollama API Response Error: {ore.error} (Status: {ore.status_code})", exc_info=True)
#         raise LLMConnectionException(detail=f"LLM communication error: {ore.error}")
#     except Exception as e:
#         logger.error(f"Error in query_document_with_rag: {type(e).__name__} - {str(e)}", exc_info=True)
#         raise QueryProcessingException(detail=f"Internal server error during query: {type(e).__name__} - {str(e)}")











# ************************* Version 4 ******************************


# backend/app/services/rag_service.py
# import logging
# from app.core.logging_config import get_logger
# from app.core.exceptions import DocumentProcessingException, QueryProcessingException, LLMConnectionException

# from fastapi import UploadFile, HTTPException
# from fastapi.concurrency import run_in_threadpool # Still useful for CPU-bound tasks
# from sqlalchemy.ext.asyncio import AsyncSession
# from typing import List, Tuple, Dict, Any, Optional, AsyncGenerator
# import tempfile
# import os
# import uuid
# import datetime

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres.vectorstores import PGVector # THE MAIN ACTOR
# from langchain_core.documents import Document
# from langchain.chains import RetrievalQA
# from langchain_ollama import OllamaLLM
# from langchain.prompts import PromptTemplate
# import ollama # For ollama.ResponseError

# from app.core.config import settings
# from app.db.session import async_engine # Import the global async_engine

# logger = get_logger(__name__)

# # DSNs
# ASYNC_PG_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
# # SYNC_PG_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://") # For truly sync ops if needed

# # --- Embedding Model Setup ---
# try:
#     logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
#     embeddings_for_pgvector = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
#     logger.info("Embedding model initialized successfully.")
# except Exception as e:
#     logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
#     embeddings_for_pgvector = None # App should ideally not start

# # --- LLM Setup ---
# llm = None
# try:
#     logger.info(f"Initializing LLM provider: {settings.LLM_PROVIDER}")
#     if settings.LLM_PROVIDER == "ollama":
#         if not settings.OLLAMA_BASE_URL or not settings.OLLAMA_MODEL_NAME:
#             raise ValueError("Ollama base URL or model name not configured.")
#         llm = OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL_NAME)
#         logger.info(f"Ollama LLM initialized with model: {settings.OLLAMA_MODEL_NAME}")
#     else:
#         raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
# except Exception as e:
#     logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
#     raise LLMConnectionException(f"Failed to initialize LLM: {str(e)}")


# async def process_and_store_document(
#     db_session: AsyncSession, # Not directly used if PGVector manages its own async connections
#     file: UploadFile,
#     collection_name: str
# ) -> Dict[str, Any]:
#     if not embeddings_for_pgvector:
#         raise DocumentProcessingException(detail="Embedding model not initialized.")

#     tmp_file_path = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file_obj:
#             content = await file.read()
#             tmp_file_obj.write(content)
#             tmp_file_path = tmp_file_obj.name
        
#         logger.info(f"Processing PDF: {file.filename} from temp path: {tmp_file_path}")
#         loader = PyPDFLoader(tmp_file_path)
#         raw_documents = await loader.aload() # Async load

#         if not raw_documents: raise ValueError("PDF loader returned no documents.")
#         logger.info(f"Loaded {len(raw_documents)} pages/documents from PDF.")

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         # split_documents is synchronous, run in threadpool
#         split_documents = await run_in_threadpool(text_splitter.split_documents, raw_documents)
#         logger.info(f"Split into {len(split_documents)} chunks.")

#         if not split_documents: raise ValueError("Text splitter returned no chunks.")

#         doc_id = str(uuid.uuid4()) # Unique ID for the uploaded document instance
#         for i, doc_chunk in enumerate(split_documents):
#             doc_chunk.metadata["source"] = file.filename
#             doc_chunk.metadata["document_id"] = doc_id # Link chunks to this upload
#             doc_chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
        
#         logger.info(f"Adding {len(split_documents)} chunks to PGVector collection: {collection_name}")

#         # Use PGVector.afrom_documents (class method) for async setup and addition
#         # This method is designed to be an async constructor.
#         # It should handle the creation of the async engine internally if needed,
#         # or use the one derived from the async DSN.
#         # The key parameters are: documents, embedding, collection_name, connection (DSN)
        
#         # According to the source, the constructor takes `embeddings` (plural)
#         # The `from_documents` and `afrom_documents` class methods take `embedding` (singular)
#         # This distinction is vital.

#         await PGVector.afrom_documents(
#             documents=split_documents,
#             embedding=embeddings_for_pgvector, # singular 'embedding' for from_documents
#             collection_name=collection_name,
#             connection=ASYNC_PG_CONNECTION_STRING, # Pass the async DSN
#             # pre_delete_collection=False, # Default, set to True if you want to clear first
#             # use_jsonb=True # Default in newer versions
#         )
#         # The `init_db()` in main.py ensures the 'vector' extension exists.
#         # `afrom_documents` should then handle table creation if needed.
        
#         logger.info("Documents added to PGVector successfully via afrom_documents.")
        
#         current_utc_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
#         return {
#             "id": doc_id, "filename": file.filename, "content_type": file.content_type,
#             "size": file.size, "total_chunks": len(split_documents),
#             "collection_name": collection_name, "uploaded_at": current_utc_timestamp,
#         }
#     except ValueError as ve:
#         logger.warning(f"Value error during document processing: {ve}", exc_info=True)
#         if tmp_file_path and os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
#         raise DocumentProcessingException(detail=str(ve))
#     except Exception as e:
#         logger.error(f"Error in process_and_store_document: {type(e).__name__} - {str(e)}", exc_info=True)
#         if tmp_file_path and os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
#         raise DocumentProcessingException(detail=f"Internal server error: {type(e).__name__} - {str(e)}")
#     finally:
#         if tmp_file_path and os.path.exists(tmp_file_path):
#             os.unlink(tmp_file_path)

# async def _get_query_vector_store(collection_name: str) -> PGVector:
#     """ Helper to get a PGVector instance for querying, ensuring async setup. """
#     logger.info(f"Instantiating PGVector for querying collection: {collection_name}")
#     # For querying, we instantiate PGVector and it will lazily call __apost_init__
#     # when an async method like asimilarity_search is invoked.
#     # The constructor expects `embeddings` (plural).
#     store = PGVector(
#         collection_name=collection_name,
#         connection=ASYNC_PG_CONNECTION_STRING,
#         embeddings=embeddings_for_pgvector, # plural 'embeddings' for constructor
#         async_mode=True, # Explicitly set async_mode
#         # create_extension=False # Set to False if init_db handles it, True by default
#                                # Let's keep True to let PGVector ensure it if needed.
#                                # init_db creates it globally, PGVector might re-check per connection.
#     )
#     # To ensure __apost_init__ is called and any errors surface early:
#     try:
#         await store.asimilarity_search("test_init_query", k=1) # This will trigger __apost_init__
#         logger.info(f"PGVector for collection '{collection_name}' async ready after dummy search.")
#     except Exception as e:
#         # If the dummy search fails due to collection not found, that's okay for the first time
#         # as the main query will also try. If it's another error, log it.
#         if "CollectionNotExists" in str(type(e)): # Placeholder for actual exception type
#             logger.info(f"Collection '{collection_name}' does not exist yet, will be handled by query.")
#         else:
#             logger.error(f"Error during PGVector async readiness check for '{collection_name}': {e}", exc_info=True)
#             # Do not re-raise here, let the actual query attempt it.
#     return store


# async def stream_document_with_rag(
#     db_session: AsyncSession, # Not directly used
#     question: str,
#     collection_name: str,
#     document_id: Optional[str] = None # Filtering by document_id is a V2 feature for retriever
# ) -> AsyncGenerator[str, None]:
#     if not embeddings_for_pgvector: raise QueryProcessingException(detail="Embedding model not initialized.")
#     if not llm: raise QueryProcessingException(detail="LLM not initialized.")

#     try:
#         vector_store = await _get_query_vector_store(collection_name)

#         retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # as_retriever is sync
#         logger.info("Retriever for streaming initialized.")

#         prompt_template_str = """You are an information extraction assistant.
#         Your task is to answer questions based *only* on the provided "Context" below.
#         Do not use any prior knowledge.
#         If the information to answer the question is not in the "Context", respond with "I don't know based on the provided document."
#         Answer directly about the subject mentioned in the question, using information from the context.
#         Be concise.

#         Context:
#         {context}

#         Question: {question}

#         Answer based on context:"""
#         QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)

#         # RetrievalQA instantiation is synchronous, run in threadpool
#         def _create_qa_chain_sync():
#             return RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=retriever,
#                 chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#             )
#         qa_chain = await run_in_threadpool(_create_qa_chain_sync)
#         logger.info("Streaming QA chain initialized.")
#         logger.info(f"Streaming QA chain with question: {question}")

#         async for chunk in qa_chain.astream({"query": question}):
#             if isinstance(chunk, dict) and "result" in chunk:
#                 yield chunk["result"]
#             elif hasattr(chunk, 'content'): 
#                  yield chunk.content
#             # else: logger.debug(f"Unknown stream chunk type: {type(chunk)} - {chunk}")
    
#     except ollama.ResponseError as ore:
#         logger.error(f"Ollama API Response Error during streaming: {ore.error} (Status: {ore.status_code})", exc_info=True)
#         yield f"[ERRORSTREAM]: LLM communication error: {ore.error}"
#     except Exception as e:
#         logger.error(f"Error in stream_document_with_rag: {type(e).__name__} - {str(e)}", exc_info=True)
#         yield f"[ERRORSTREAM]: Internal server error: {type(e).__name__} - {str(e)}"


# async def query_document_with_rag(
#     db_session: AsyncSession,
#     question: str,
#     collection_name: str,
#     document_id: Optional[str] = None
# ) -> Tuple[str, List[str]]:
#     if not embeddings_for_pgvector: raise QueryProcessingException(detail="Embedding model not initialized.")
#     if not llm: raise QueryProcessingException(detail="LLM not initialized.")
#     try:
#         vector_store = await _get_query_vector_store(collection_name)

#         retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Sync method
#         logger.info("Retriever initialized.")

#         prompt_template_str = """...""" # Same prompt as stream_document_with_rag
#         QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)
        
#         def _create_qa_chain_sync():
#             return RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=retriever,
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#             )
#         qa_chain = await run_in_threadpool(_create_qa_chain_sync)
        
#         logger.info("QA chain initialized.")
#         logger.info(f"Invoking QA chain with question: {question}")
        
#         result = await run_in_threadpool(qa_chain.invoke, {"query": question})
        
#         answer = result.get("result", "No answer found.")
#         source_documents = result.get("source_documents", [])
#         source_chunks_content = [doc.page_content for doc in source_documents]

#         logger.info(f"LLM Answer: {answer}")
#         # Log source chunks if needed for debugging
#         # for i, chunk_content in enumerate(source_chunks_content):
#         #     logger.debug(f"Source Chunk {i+1} for question '{question}':\n{chunk_content}\n--------------------")
#         return answer, source_chunks_content
#     except ollama.ResponseError as ore:
#         logger.error(f"Ollama API Response Error: {ore.error} (Status: {ore.status_code})", exc_info=True)
#         raise LLMConnectionException(detail=f"LLM communication error: {ore.error}")
#     except Exception as e:
#         logger.error(f"Error in query_document_with_rag: {type(e).__name__} - {str(e)}", exc_info=True)
#         raise QueryProcessingException(detail=f"Internal server error: {type(e).__name__} - {str(e)}")









# ************************* Version 6 ******************************



# app/services/rag_service.py
# import logging
# from app.core.logging_config import get_logger
# from app.core.exceptions import DocumentProcessingException, QueryProcessingException, LLMConnectionException

# from fastapi import UploadFile, HTTPException
# from fastapi.concurrency import run_in_threadpool
# from sqlalchemy.ext.asyncio import AsyncSession # Not directly used by PGVector methods here
# import sqlalchemy
# from typing import List, Tuple, Dict, Any, Optional, AsyncGenerator
# import tempfile
# import os
# import uuid
# import datetime

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres.vectorstores import PGVector
# # from langchain_core.documents import Document # Not explicitly used if passing splits directly
# from langchain.chains import RetrievalQA
# from langchain_ollama import OllamaLLM
# from langchain.prompts import PromptTemplate
# import ollama # For ollama.ResponseError

# from app.core.config import settings, ASYNC_DB_URL, SYNC_DB_URL # Import DSNs

# logger = get_logger(__name__)

# # --- Global Service Variables (Initialized once) ---
# embeddings_for_pgvector: Optional[HuggingFaceEmbeddings] = None
# llm: Optional[OllamaLLM] = None

# def initialize_global_services():
#     global embeddings_for_pgvector, llm
#     if embeddings_for_pgvector is None:
#         try:
#             logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
#             embeddings_for_pgvector = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
#             logger.info("Embedding model initialized successfully.")
#         except Exception as e:
#             logger.error(f"CRITICAL: Failed to initialize embedding model: {e}", exc_info=True)
#             # Application should ideally not proceed without embeddings
#             raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

#     if llm is None:
#         try:
#             logger.info(f"Initializing LLM provider: {settings.LLM_PROVIDER}")
#             if settings.LLM_PROVIDER == "ollama":
#                 if not settings.OLLAMA_BASE_URL or not settings.OLLAMA_MODEL_NAME:
#                     raise ValueError("Ollama base URL or model name not configured.")
#                 llm = OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL_NAME)
#                 logger.info(f"Ollama LLM initialized with model: {settings.OLLAMA_MODEL_NAME}")
#             else:
#                 raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
#         except Exception as e:
#             logger.error(f"CRITICAL: Failed to initialize LLM: {e}", exc_info=True)
#             raise LLMConnectionException(f"Failed to initialize LLM: {str(e)}") from e

# # Call initialization when this module is imported (FastAPI will import it)
# # This can also be done in FastAPI startup event, but this is simpler for now.
# initialize_global_services()


# async def process_and_store_document(
#     db_session: AsyncSession, # PGVector methods will manage their own connections via DSN
#     file: UploadFile,
#     collection_name: str
# ) -> Dict[str, Any]:
#     if not embeddings_for_pgvector: # Should have been caught by initialize_global_services
#         raise DocumentProcessingException(detail="Embedding model not available.")

#     tmp_file_path = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file_obj:
#             content = await file.read()
#             tmp_file_obj.write(content)
#             tmp_file_path = tmp_file_obj.name
        
#         logger.info(f"Processing PDF: {file.filename} from temp path: {tmp_file_path}")
#         loader = PyPDFLoader(tmp_file_path)
#         raw_documents = await loader.aload()

#         if not raw_documents: raise ValueError("PDF loader returned no documents.")
#         logger.info(f"Loaded {len(raw_documents)} pages/documents from PDF.")

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         split_documents = await run_in_threadpool(text_splitter.split_documents, raw_documents)
#         logger.info(f"Split into {len(split_documents)} chunks.")

#         if not split_documents: raise ValueError("Text splitter returned no chunks.")

#         doc_id = str(uuid.uuid4())
#         for i, doc_chunk in enumerate(split_documents):
#             doc_chunk.metadata["source"] = file.filename
#             doc_chunk.metadata["document_id"] = doc_id
#             doc_chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
        
#         logger.info(f"Adding {len(split_documents)} chunks to PGVector collection: {collection_name} using ASYNC DSN.")
        
#         # Use PGVector.afrom_documents for async addition.
#         # This class method should handle its own async engine setup based on the ASYNC_DB_URL.
#         # The `init_db()` in main.py ensures the 'vector' extension exists globally.
#         # `afrom_documents` will create the collection table if it doesn't exist.
#         # The key `embedding` is used for the `afrom_documents` class method.
#         await PGVector.afrom_documents(
#             documents=split_documents,
#             embedding=embeddings_for_pgvector, # Correct parameter for from_documents methods
#             collection_name=collection_name,
#             connection=ASYNC_DB_URL, # Use the asyncpg DSN
#             pre_delete_collection=False, # Set True to clear collection before adding new docs
#             # create_extension=False # Set to False since init_db handles it. Default is True.
#                                      # PGVector will try to create extension with a lock, this might be the conflict point.
#                                      # Let's try with False since our init_db does it.
#         )
        
#         logger.info(f"Documents added to PGVector successfully via afrom_documents for collection '{collection_name}'.")
        
#         current_utc_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
#         return {
#             "id": doc_id, "filename": file.filename, "content_type": file.content_type,
#             "size": file.size, "total_chunks": len(split_documents),
#             "collection_name": collection_name, "uploaded_at": current_utc_timestamp,
#         }
#     except ValueError as ve:
#         logger.warning(f"Value error during document processing: {ve}", exc_info=True)
#         if tmp_file_path and os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
#         raise DocumentProcessingException(detail=str(ve))
#     except Exception as e: # This will catch SQLAlchemy ProgrammingError etc.
#         logger.error(f"Error in process_and_store_document for collection '{collection_name}': {type(e).__name__} - {str(e)}", exc_info=True)
#         if tmp_file_path and os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
#         # Provide more specific error detail if possible
#         detail = f"Internal server error: {type(e).__name__}"
#         if isinstance(e, sqlalchemy.exc.ProgrammingError) and "multiple commands" in str(e):
#             detail = "Database configuration error related to multi-command statements."
#         elif "greenlet_spawn" in str(e):
#             detail = "Async database operation conflict (greenlet issue)."

#         raise DocumentProcessingException(detail=detail + f" - Original: {str(e)[:200]}") # Truncate original error
#     finally:
#         if tmp_file_path and os.path.exists(tmp_file_path):
#             os.unlink(tmp_file_path)

# async def _get_query_vector_store(collection_name: str) -> PGVector:
#     """ Helper to get a PGVector instance for querying. """
#     logger.info(f"Instantiating PGVector for querying collection: {collection_name}")
    
#     # For querying, we instantiate PGVector.
#     # The constructor expects `embeddings` (plural).
#     # Explicitly set async_mode=True.
#     # Set create_extension=False because init_db should have handled it.
#     store = PGVector(
#         collection_name=collection_name,
#         connection=ASYNC_DB_URL, # Use async DSN
#         embeddings=embeddings_for_pgvector, # plural 'embeddings' for constructor
#         async_mode=True,
#         create_extension=False # init_db handles global extension creation
#     )
#     # To ensure its internal async setup is triggered (__apost_init__)
#     # This will also check if the collection table exists.
#     try:
#         # A light async operation to trigger internal async setup.
#         # `asimilarity_search` will call `__apost_init__` if not already done.
#         # `__apost_init__` calls `acreate_collection` which checks if table exists.
#         await store.asimilarity_search("test init", k=1, fetch_k=1) 
#         logger.info(f"PGVector for collection '{collection_name}' async ready after dummy search.")
#     except Exception as e:
#         # It's okay if the collection doesn't exist yet for a query, the search will be empty.
#         # But other errors (like connection or true async setup failure) should be logged.
#         # LangChain often raises specific errors if collection doesn't exist during search.
#         if "CollectionNotExists" in str(type(e)) or "does not exist" in str(e).lower(): # Heuristic
#             logger.info(f"Collection '{collection_name}' likely does not exist yet for query, which is fine.")
#         else:
#             logger.error(f"Potential error during PGVector async readiness check for '{collection_name}': {e}", exc_info=True)
#             # We don't re-raise here, let the actual query handle it.
#     return store


# async def stream_document_with_rag(
#     db_session: AsyncSession, # Not directly used
#     question: str,
#     collection_name: str,
#     document_id: Optional[str] = None
# ) -> AsyncGenerator[str, None]:
#     if not embeddings_for_pgvector: yield "[ERRORSTREAM]: Embedding model not available."; return
#     if not llm: yield "[ERRORSTREAM]: LLM not available."; return

#     try:
#         vector_store = await _get_query_vector_store(collection_name)

#         retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # as_retriever is sync
#         logger.info("Retriever for streaming initialized.")

#         prompt_template_str = """You are an information extraction assistant.
#         Your task is to answer questions based *only* on the provided "Context" below.
#         Do not use any prior knowledge.
#         If the information to answer the question is not in the "Context", respond with "I don't know based on the provided document."
#         Answer directly about the subject mentioned in the question, using information from the context.
#         Be concise.

#         Context:
#         {context}

#         Question: {question}

#         Answer based on context:"""
#         QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)

#         def _create_qa_chain_sync(): # RetrievalQA instantiation is synchronous
#             return RetrievalQA.from_chain_type(
#                 llm=llm, chain_type="stuff", retriever=retriever,
#                 chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#             )
#         qa_chain = await run_in_threadpool(_create_qa_chain_sync)
#         logger.info("Streaming QA chain initialized.")
#         logger.info(f"Streaming QA chain with question: {question}")

#         async for chunk in qa_chain.astream({"query": question}):
#             if isinstance(chunk, dict) and "result" in chunk:
#                 yield chunk["result"]
#             elif hasattr(chunk, 'content'): 
#                  yield chunk.content
#             # else: logger.debug(f"Stream chunk type: {type(chunk)} content: {chunk}")
    
#     except ollama.ResponseError as ore:
#         logger.error(f"Ollama API Error during streaming: {ore.error} (Status: {ore.status_code})", exc_info=True)
#         yield f"[ERRORSTREAM]: LLM communication error: {ore.error}"
#     except Exception as e:
#         logger.error(f"Error in stream_document_with_rag: {type(e).__name__} - {str(e)}", exc_info=True)
#         yield f"[ERRORSTREAM]: Internal server error: {type(e).__name__} - {str(e)}"


# async def query_document_with_rag(
#     db_session: AsyncSession,
#     question: str,
#     collection_name: str,
#     document_id: Optional[str] = None
# ) -> Tuple[str, List[str]]:
#     if not embeddings_for_pgvector: raise QueryProcessingException(detail="Embedding model not available.")
#     if not llm: raise QueryProcessingException(detail="LLM not available.")
#     try:
#         vector_store = await _get_query_vector_store(collection_name)

#         retriever = vector_store.as_retriever(search_kwargs={"k": 2})
#         logger.info("Retriever initialized.")

#         prompt_template_str = """...""" # Same prompt as stream_document_with_rag
#         QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)
        
#         def _create_qa_chain_sync(): # Sync instantiation
#             return RetrievalQA.from_chain_type(
#                 llm=llm, chain_type="stuff", retriever=retriever,
#                 return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#             )
#         qa_chain = await run_in_threadpool(_create_qa_chain_sync)
        
#         logger.info("QA chain initialized.")
#         logger.info(f"Invoking QA chain with question: {question}")
        
#         result = await run_in_threadpool(qa_chain.invoke, {"query": question})
        
#         answer = result.get("result", "No answer found.")
#         source_documents = result.get("source_documents", [])
#         source_chunks_content = [doc.page_content for doc in source_documents]

#         logger.info(f"LLM Answer: {answer}")
#         # Log source chunks if needed for debugging
#         # for i, chunk_content in enumerate(source_chunks_content):
#         #     logger.debug(f"Source Chunk {i+1} for question '{question}': {chunk_content[:100]}...")
#         return answer, source_chunks_content
#     except ollama.ResponseError as ore:
#         logger.error(f"Ollama API Error: {ore.error} (Status: {ore.status_code})", exc_info=True)
#         raise LLMConnectionException(detail=f"LLM error: {ore.error}")
#     except Exception as e:
#         logger.error(f"Error in query_document_with_rag: {type(e).__name__} - {str(e)}", exc_info=True)
#         raise QueryProcessingException(detail=f"Internal server error: {type(e).__name__} - {str(e)}")








# ************************* Version 7 ******************************



# app/services/rag_service.py
import logging
from app.core.logging_config import get_logger
from app.core.exceptions import DocumentProcessingException, QueryProcessingException, LLMConnectionException

from fastapi import UploadFile # HTTPException removed, handled by custom exceptions
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession # Not directly used by PGVector class methods
import sqlalchemy # For specific exception types
from typing import List, Tuple, Dict, Any, Optional, AsyncGenerator
import tempfile
import os
import uuid
import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import ollama

from app.core.config import settings, ASYNC_DB_URL # Import ASYNC_DB_URL

logger = get_logger(__name__)

# --- Global Service Variables (Initialized once at module import) ---
embeddings_for_pgvector: Optional[HuggingFaceEmbeddings] = None
llm: Optional[OllamaLLM] = None

def initialize_global_services():
    global embeddings_for_pgvector, llm
    if embeddings_for_pgvector is None:
        try:
            logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
            embeddings_for_pgvector = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model initialized successfully.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize embedding model: {e}")

    if llm is None:
        try:
            logger.info(f"Initializing LLM provider: {settings.LLM_PROVIDER}")
            if settings.LLM_PROVIDER == "ollama":
                if not settings.OLLAMA_BASE_URL or not settings.OLLAMA_MODEL_NAME:
                    raise ValueError("Ollama base URL or model name not configured for LLM.")
                llm = OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL_NAME)
                logger.info(f"Ollama LLM initialized with model: {settings.OLLAMA_MODEL_NAME}")
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize LLM: {e}", exc_info=True)
            raise LLMConnectionException(f"Failed to initialize LLM: {str(e)}")

initialize_global_services()


async def process_and_store_document(
    # db_session: AsyncSession, # PGVector methods will manage their own connections via DSN
    file: UploadFile,
    collection_name: str
) -> Dict[str, Any]:
    if not embeddings_for_pgvector:
        # This should have been caught by initialize_global_services, but as a safeguard:
        raise DocumentProcessingException(detail="Embedding model is not available.")

    tmp_file_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file_obj:
            content = await file.read()
            tmp_file_obj.write(content)
            tmp_file_path = tmp_file_obj.name
        
        logger.info(f"Processing PDF: {file.filename} from temp path: {tmp_file_path}")
        loader = PyPDFLoader(tmp_file_path)
        raw_documents = await loader.aload()

        if not raw_documents: raise ValueError("PDF loader returned no documents.")
        logger.info(f"Loaded {len(raw_documents)} pages/documents from PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = await run_in_threadpool(text_splitter.split_documents, raw_documents)
        logger.info(f"Split into {len(split_documents)} chunks.")

        if not split_documents: raise ValueError("Text splitter returned no chunks.")

        doc_id = str(uuid.uuid4())
        for i, doc_chunk in enumerate(split_documents):
            doc_chunk.metadata["source"] = file.filename
            doc_chunk.metadata["document_id"] = doc_id
            doc_chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
        
        logger.info(f"Adding {len(split_documents)} chunks to PGVector collection: '{collection_name}' using ASYNC DSN.")
        
        # Use PGVector.afrom_documents. This class method should handle async setup.
        # init_db ensures 'vector' extension exists. afrom_documents handles table creation.
        # Key parameters: documents, embedding (singular), collection_name, connection (async DSN)
        # Set create_extension=False as init_db handles it.
        await PGVector.afrom_documents(
            documents=split_documents,
            embedding=embeddings_for_pgvector, # Parameter for class method
            collection_name=collection_name,
            connection=ASYNC_DB_URL,      # Use the asyncpg DSN
            create_extension=False,       # Crucial: init_db handles this globally
            # use_jsonb=True, # Defaults to True in recent versions
            # pre_delete_collection=False # Default
        )
        
        logger.info(f"Documents added to PGVector successfully via afrom_documents for collection '{collection_name}'.")
        
        current_utc_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "id": doc_id, "filename": file.filename, "content_type": file.content_type,
            "size": file.size, "total_chunks": len(split_documents),
            "collection_name": collection_name, "uploaded_at": current_utc_timestamp,
        }

    except ValueError as ve:
        logger.warning(f"Value error during document processing for '{file.filename}': {ve}", exc_info=True)
        raise DocumentProcessingException(detail=str(ve))
    except sqlalchemy.exc.ProgrammingError as pe: # Catch specific DB errors
        logger.error(f"Database ProgrammingError during document processing for '{file.filename}': {pe}", exc_info=True)
        detail = "A database error occurred during document processing."
        if "multiple commands" in str(pe):
            detail = "Database configuration error: multi-command statement issue."
        raise DocumentProcessingException(detail=detail + f" Original: {str(pe)[:150]}")
    except Exception as e:
        logger.error(f"Unexpected error in process_and_store_document for '{file.filename}': {type(e).__name__} - {str(e)}", exc_info=True)
        raise DocumentProcessingException(detail=f"Internal server error: {type(e).__name__} - {str(e)[:150]}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


async def _get_query_vector_store(collection_name: str) -> PGVector:
    """ Helper to get a PGVector instance for querying, ensuring async setup. """
    logger.info(f"Instantiating PGVector for querying collection: '{collection_name}'")
    
    # The constructor expects `embeddings` (plural).
    # Explicitly set async_mode=True.
    # Set create_extension=False because init_db should have handled it.
    # Run the constructor in a threadpool as a safeguard if its __init__ or __post_init__
    # (which it calls if async_mode=False) does unexpected sync I/O with an async DSN.
    # With async_mode=True, it relies on __apost_init__ (lazy async).
    
    def _instantiate_pgvector_sync_wrapper():
        return PGVector(
            collection_name=collection_name,
            connection=ASYNC_DB_URL,
            embeddings=embeddings_for_pgvector, # Plural for constructor
            async_mode=True,                  # Crucial
            create_extension=False            # Crucial
        )
    
    store = await run_in_threadpool(_instantiate_pgvector_sync_wrapper)

    # Proactively trigger async initialization and check if collection table exists
    try:
        # This will call `__apost_init__` if not already done,
        # which in turn calls `acreate_collection` (checks/creates table, not extension).
        await store.asimilarity_search("test_init_query", k=1, fetch_k=1) 
        logger.info(f"PGVector for collection '{collection_name}' async-ready after dummy search.")
    except Exception as e:
        # LangChain PGVector might raise an error if the collection table doesn't exist yet.
        # This dummy search might create it if `afrom_documents` hadn't for some reason.
        # If it's a "relation does not exist" error, it's somewhat expected on first query
        # if no documents were added to this specific collection yet.
        # The actual query will then create it if using a method like `from_chain_type` with `RetrievalQA`.
        # However, `asimilarity_search` itself might not auto-create.
        # A robust application might explicitly call `await store.acreate_collection()`
        # or rely on `afrom_documents` to have created it.
        if "relation" in str(e).lower() and "does not exist" in str(e).lower() and f'"{collection_name}"' in str(e).lower():
             logger.warning(f"Collection table '{collection_name}' does not exist yet. Query will likely be empty or create it.")
        else:
            logger.error(f"Error during PGVector async readiness check for '{collection_name}': {e}", exc_info=True)
            # Not re-raising, let the main query path attempt and potentially fail with more context.
    return store


async def stream_document_with_rag(
    # db_session: AsyncSession, # Not directly used by _get_query_vector_store
    question: str,
    collection_name: str,
    document_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    if not embeddings_for_pgvector: yield "[ERRORSTREAM]: Embedding model not available."; return
    if not llm: yield "[ERRORSTREAM]: LLM not available."; return

    try:
        vector_store = await _get_query_vector_store(collection_name)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        logger.info("Retriever for streaming initialized.")

        prompt_template_str = """You are an information extraction assistant.
        Your task is to answer questions based *only* on the provided "Context" below.
        Do not use any prior knowledge.
        If the information to answer the question is not in the "Context", respond with "I don't know based on the provided document."
        Answer directly about the subject mentioned in the question, using information from the context.
        Be concise.

        Context:
        {context}

        Question: {question}

        Answer based on context:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)

        def _create_qa_chain_sync():
            return RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        qa_chain = await run_in_threadpool(_create_qa_chain_sync)
        logger.info("Streaming QA chain initialized.")
        logger.info(f"Streaming QA chain with question: {question}")

        async for chunk in qa_chain.astream({"query": question}):
            if isinstance(chunk, dict) and "result" in chunk:
                yield chunk["result"]
            elif hasattr(chunk, 'content'): 
                 yield chunk.content
            # else: logger.debug(f"Stream chunk type: {type(chunk)} content: {chunk}")
    
    except ollama.ResponseError as ore:
        logger.error(f"Ollama API Error during streaming: {ore.error} (Status: {ore.status_code})", exc_info=True)
        yield f"[ERRORSTREAM]: LLM communication error: {ore.error}"
    except Exception as e:
        logger.error(f"Error in stream_document_with_rag: {type(e).__name__} - {str(e)}", exc_info=True)
        yield f"[ERRORSTREAM]: Internal server error: {type(e).__name__} - {str(e)[:150]}"


async def query_document_with_rag(
    # db_session: AsyncSession,
    question: str,
    collection_name: str,
    document_id: Optional[str] = None
) -> Tuple[str, List[str]]:
    if not embeddings_for_pgvector: raise QueryProcessingException(detail="Embedding model not available.")
    if not llm: raise QueryProcessingException(detail="LLM not available.")
    try:
        vector_store = await _get_query_vector_store(collection_name)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        logger.info("Retriever initialized.")

        # Using the same refined prompt as in streaming
        prompt_template_str = """You are an information extraction assistant.
        Your task is to answer questions based *only* on the provided "Context" below.
        Do not use any prior knowledge.
        If the information to answer the question is not in the "Context", respond with "I don't know based on the provided document."
        Answer directly about the subject mentioned in the question, using information from the context.
        Be concise.

        Context:
        {context}

        Question: {question}

        Answer based on context:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)
        
        def _create_qa_chain_sync():
            return RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever,
                return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        qa_chain = await run_in_threadpool(_create_qa_chain_sync)
        
        logger.info("QA chain initialized.")
        logger.info(f"Invoking QA chain with question: {question}")
        
        result = await run_in_threadpool(qa_chain.invoke, {"query": question})
        
        answer = result.get("result", "No answer found.")
        source_documents = result.get("source_documents", [])
        source_chunks_content = [doc.page_content for doc in source_documents]

        logger.info(f"LLM Answer: {answer}")
        return answer, source_chunks_content
    except ollama.ResponseError as ore:
        logger.error(f"Ollama API Error: {ore.error} (Status: {ore.status_code})", exc_info=True)
        raise LLMConnectionException(detail=f"LLM error: {ore.error}")
    except Exception as e:
        logger.error(f"Error in query_document_with_rag: {type(e).__name__} - {str(e)}", exc_info=True)
        raise QueryProcessingException(detail=f"Internal server error: {type(e).__name__} - {str(e)[:150]}")









# This should be at the top of your rag_service.py or in config.py and imported
# ASYNC_PG_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
# SYNC_PG_CONNECTION_STRING = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")


# async def _ensure_collection_exists_sync(collection_name: str):
#     """
#     Synchronous helper to be run in a threadpool to ensure the collection exists.
#     This uses a synchronous DSN.
#     Our main init_db already creates the 'vector' extension.
#     """
#     try:
#         PGVector.create_collection(
#             collection_name=collection_name,
#             connection=SYNC_PG_CONNECTION_STRING, # IMPORTANT: Use sync DSN
#             embeddings=embeddings_for_pgvector  # Pass the embedding model
#         )
#         logger.info(f"Synchronously ensured PGVector collection '{collection_name}' exists or was created.")
#     except Exception as e:
#         # This might happen if it exists and the creation logic isn't purely "IF NOT EXISTS"
#         # or if there's another configuration issue.
#         # For many cases, if init_db created the extension, this might only be for table creation.
#         logger.warning(f"Note during sync ensure_collection_exists for '{collection_name}': {e}. Assuming OK if table will be auto-created or already exists.")
#         # We don't re-raise here, as PGVector might create the table on first use anyway.
#         # The main goal is to ensure the vector extension is available and table creation doesn't trip over async issues.

# async def _get_vector_store(collection_name: str) -> PGVector:
#     """
#     Helper to initialize PGVector and ensure its underlying table and extension exist,
#     returning an instance ready for async operations.
#     """
#     logger.info(f"Getting PGVector store for collection: {collection_name}")

#     # Step 1: Ensure the collection (table) and pgvector extension are created.
#     # The `init_db()` in main.py handles `CREATE EXTENSION IF NOT EXISTS vector;` asynchronously.
#     # PGVector.create_collection is a synchronous class method. We run it in a threadpool
#     # to avoid blocking and to handle its synchronous DB interactions correctly.
#     await run_in_threadpool(_ensure_collection_exists_sync, collection_name)
    
#     # Step 2: Instantiate PGVector for async operations.
#     # The constructor itself is synchronous. We pass the ASYNC_PG_CONNECTION_STRING
#     # so that its ASYNC methods (like asimilarity_search) will use asyncpg.
#     # Any *synchronous* methods called on this instance that do DB IO
#     # might still hit the greenlet issue if they don't bridge correctly.
#     # But we aim to use its .a* methods.
#     def _instantiate_pgvector_for_async():
#         return PGVector(
#             collection_name=collection_name,
#             connection=ASYNC_PG_CONNECTION_STRING, # DSN for asyncpg
#             embeddings=embeddings_for_pgvector,
#         )

#     # Running the instantiation in a threadpool is a safeguard in case its __init__
#     # does some synchronous DB IO that might conflict with asyncpg if not bridged.
#     # For PGVector, __init__ primarily sets up parameters; DB interaction is often lazy.
#     # However, to be absolutely safe from the MissingGreenlet during setup:
#     store = await run_in_threadpool(_instantiate_pgvector_for_async)
    
#     logger.info(f"PGVector store for collection '{collection_name}' instantiated for async use.")
#     return store



# async def _get_vector_store(collection_name: str) -> PGVector:
#     """Helper to initialize PGVector and ensure collection exists, for async usage."""
#     logger.info(f"Attempting to initialize PGVector for collection: {collection_name} with ASYNC DSN.")
    
#     # The PGVector constructor itself is synchronous.
#     # Pass the DSN that implies asyncpg for when its async methods are called.
#     store = PGVector(
#         collection_name=collection_name,
#         connection=ASYNC_PG_CONNECTION_STRING, # DSN for asyncpg
#         embeddings=embeddings_for_pgvector,
#         # For async, the engine setup happens internally when async methods are called,
#         # or via specific async setup methods.
#     )

#     # Explicitly ensure the collection (and pgvector extension) is set up using an async method.
#     # This method should handle the internal async engine setup correctly.
#     # PGVector.acreate_collection is a class method, so we call it on the class.
#     # However, we need to ensure the extension is created first on the *connection*
#     # before the collection table that depends on vector type is created.
#     # The `init_db()` in main.py already does `CREATE EXTENSION IF NOT EXISTS vector;`

#     # Let's try to rely on PGVector's internal async setup triggered by its async methods.
#     # The error occurred during `asimilarity_search` -> `__apost_init__` -> `acreate_vector_extension`
#     # This suggests `acreate_vector_extension` itself is the point of failure due to how it uses the engine.

#     # A robust way for LangChain >0.1.x is to ensure the engine used by PGVector for async
#     # operations is indeed an AsyncEngine.
#     # The 'connection' parameter in PGVector is a DSN string.
#     # langchain-postgres PGVector will create an engine from this string.
#     # If it's an asyncpg DSN, it should create an AsyncEngine for its async methods.
#     # The error indicates that even with an asyncpg DSN, a synchronous path within
#     # the extension creation (which happens lazily) is causing trouble.

#     # What if we try to ensure the collection and extension exist using our main async_engine?
#     # This is tricky because PGVector manages its own table.

#     # The core issue is that `PGVector._engine` (sync) is used by `create_vector_extension`
#     # even when `acreate_vector_extension` is called.
#     # We need to ensure that method uses an async context.

#     # Let's go back to a slightly different approach for the "dummy init"
#     # to force the async path correctly for extension creation.
#     # The error happens in `self.create_vector_extension` called from `__post_init__`
#     # or `acreate_vector_extension` called from `__apost_init__`.

#     # The `__apost_init__` is called by methods like `asimilarity_search`.
#     # Let's simplify the instantiation and rely on this lazy async init,
#     # but ensure the `connection` string is correct for asyncpg.

#     # If `PGVector.create_vector_extension` is the problematic sync call within an async context:
#     # We might need to call it in a threadpool *if we explicitly call it*.
#     # However, it's called internally by PGVector.

#     # This means the `PGVector` class itself, when `connection` is an async DSN,
#     # might not be fully async-safe for its *initial setup routines* like creating the extension,
#     # even if its data methods (like asimilarity_search) are async.

#     # Let's try the official way to create collection if not exists:
#     # This pattern is often used with PGVector
#     try:
#         PGVector.create_collection( # This is a synchronous class method
#             collection_name=collection_name,
#             connection=settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://"), # Use sync DSN for this sync method
#             embeddings=embeddings_for_pgvector
#         )
#         logger.info(f"Ensured (sync) collection '{collection_name}' exists or created.")
#     except Exception as e_coll:
#         # This might happen if it already exists and the check isn't perfect, or other issues.
#         logger.warning(f"Could not ensure (sync) collection '{collection_name}': {e_coll}. Assuming it exists or will be handled by async path.")
#         # The `init_db` in main.py should have already run `CREATE EXTENSION IF NOT EXISTS vector;`

#     # Now, instantiate for async use
#     store = PGVector(
#         collection_name=collection_name,
#         connection=ASYNC_PG_CONNECTION_STRING, # For async operations
#         embeddings=embeddings_for_pgvector,
#     )
#     # The error implies that an async method call (like asimilarity_search)
#     # is triggering a synchronous DB call for extension creation.

#     # The root of the sqlalchemy.exc.MissingGreenlet error is often when
#     # an asyncpg connection (which needs an event loop) is used in a synchronous way
#     # without the greenlet bridge (like await_only).
#     # PGVector's `create_vector_extension` uses `self._engine.connect()`.
#     # If `self._engine` was created from the `ASYNC_PG_CONNECTION_STRING` but is a
#     # synchronous SQLAlchemy engine wrapper around asyncpg, it needs greenlet.

#     # The most straightforward path is to ensure that any method on PGVector that
#     # *might* do initial setup (like creating the extension or table) is called
#     # in a way that respects its internal sync/async engine choice.

#     # The `init_db()` in main.py already creates the 'vector' extension using the global async_engine.
#     # This should be sufficient for the extension part.
#     # So, when PGVector is initialized, it should find the extension already present.
#     # The issue might then be with table creation if it's also done synchronously.

#     # Let's simplify and assume init_db has handled the EXTENSION creation.
#     # The PGVector constructor will then try to ensure the table exists.
#     # If `PGVector`'s internal `_create_table_if_not_exists` is synchronous and uses
#     # an engine derived from `ASYNC_PG_CONNECTION_STRING` incorrectly, that's the problem.

#     # Reverting to the simpler instantiation. The fix might be in how LangChain PGVector
#     # handles its engine when an async DSN is passed for its synchronous setup methods.
#     store = PGVector(
#         collection_name=collection_name,
#         connection=ASYNC_PG_CONNECTION_STRING, # Pass async DSN
#         embeddings=embeddings_for_pgvector,
#     )
#     # The problem is that `__post_init__` (sync) might be called, which calls `create_vector_extension` (sync).
#     # And that sync method tries to use the async DSN in a way that needs greenlet.

#     # The most robust way is to ensure any synchronous initialization work done by PGVector
#     # when using an async DSN is wrapped if it doesn't handle the greenlet bridging itself.
#     # This is hard to inject from outside.

#     # Let's try initializing with a SYNCHRONOUS DSN first for the initial setup,
#     # then see if we can "re-bind" or use its async methods effectively. This is messy.

#     # Alternative: LangChain's `PGVector` might have a bug or a tricky setup for pure async.
#     # A common workaround is to perform setup tasks (like ensuring collection/extension)
#     # using a synchronous engine, and then for query/add operations that have async versions,
#     # ensure they use an async-compatible path.

#     # Given that `init_db` (called at FastAPI startup) already does:
#     # `await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))`
#     # The `vector` extension *should* exist when PGVector initializes.
#     # The error seems to be that PGVector's `create_vector_extension` is being called again
#     # and failing because it's trying to use an asyncpg connection synchronously.

#     # A key part of the traceback:
#     # File "/usr/local/lib/python3.11/site-packages/sqlalchemy/dialects/postgresql/asyncpg.py", line 961, in connect
#     #    await_only(creator_fn(*arg, **kw)),
#     # This `await_only` is the problem. It's being called in a context where `greenlet_spawn` hasn't happened.
#     # This is a SQLAlchemy + asyncpg thing. The `PGVector` class when using an asyncpg DSN
#     # for its *synchronous* setup methods is hitting this.

#     # What if we explicitly provide the global `async_engine` if the library supports it?
#     # Check PGVector source: it mainly expects `connection` (DSN string) or `engine_args`.
#     # It does not seem to directly take an existing SQLAlchemy engine object easily.

#     # THE MOST LIKELY FIX:
#     # When `PGVector` is initialized with an `asyncpg` DSN (like your `ASYNC_PG_CONNECTION_STRING`),
#     # its *synchronous* internal methods (like `create_vector_extension`) that might get called during
#     # initialization or first use are attempting to use the `asyncpg` driver via SQLAlchemy's
#     # default dialect behavior, which involves `await_only`.
#     # This needs to be run in a context where `greenlet` is available for `asyncpg`'s sync bridge.
#     # `run_in_threadpool` *should* provide this context for the duration of the function call.

#     # Let's re-wrap the instantiation of PGVector within `run_in_threadpool`
#     # This means the synchronous parts of PGVector's __init__ (including potential calls to create_vector_extension)
#     # will run in a thread that `anyio` manages, which *should* handle greenlet bridging.

#     def _init_pgvector_sync_in_thread():
#         # Inside this function, we are in a sync context managed by the threadpool.
#         # PGVector will be initialized here. Its __init__ might call create_vector_extension.
#         # If create_vector_extension uses a sync engine derived from an asyncpg DSN,
#         # the threadpool context should allow the greenlet bridging to work.
#         _store = PGVector(
#             collection_name=collection_name,
#             connection=ASYNC_PG_CONNECTION_STRING, # Still pass the async DSN
#             embeddings=embeddings_for_pgvector,
#         )
#         # We can also explicitly try to create the collection here, which includes extension check
#         # This `create_collection` is a class method and is synchronous.
#         PGVector.create_collection(
#             collection_name=collection_name,
#             connection=settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://"), # Use sync DSN for this sync class method
#             embeddings=embeddings_for_pgvector
#         )
#         return _store

#     store = await run_in_threadpool(_init_pgvector_sync_in_thread)
#     # Now 'store' is an instance of PGVector. Its async methods should now work
#     # because the instance itself is fine, and its async methods are designed to be awaited.
#     # The previous error was about its *initial setup* of the extension/table.
    
#     logger.info(f"PGVector store for collection '{collection_name}' initialized via threadpool.")
#     return store
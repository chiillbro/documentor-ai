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



from fastapi import UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Tuple, Dict, Any, Optional
import tempfile
import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM # Corrected LLM import
from langchain.prompts import PromptTemplate

from app.core.config import settings

# --- Embedding Model Setup ---
try:
    print(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
    # This is our embedding model instance
    embeddings_for_pgvector = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
    print("Embedding model initialized successfully.")
except Exception as e:
    print(f"Failed to initialize embedding model: {e}")
    embeddings_for_pgvector = None

# --- LLM Setup ---
llm = None
try:
    print(f"Initializing LLM provider: {settings.LLM_PROVIDER}")
    if settings.LLM_PROVIDER == "ollama":
        if not settings.OLLAMA_BASE_URL or not settings.OLLAMA_MODEL_NAME:
            raise ValueError("Ollama base URL or model name not configured.")
        # Use the corrected OllamaLLM class
        llm = OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL_NAME)
        print(f"Ollama LLM initialized with model: {settings.OLLAMA_MODEL_NAME}")
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
except Exception as e:
    print(f"Failed to initialize LLM: {e}")


async def process_and_store_document(
    db_session: AsyncSession,
    file: UploadFile,
    collection_name: str
) -> Dict[str, Any]:
    if not embeddings_for_pgvector: # Check the renamed embedding model variable
        raise HTTPException(status_code=500, detail="Embedding model not initialized.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        print(f"Processing PDF: {file.filename} from temp path: {tmp_file_path}")
        loader = PyPDFLoader(tmp_file_path)
        # raw_documents = loader.load()
        raw_documents = await run_in_threadpool(loader.load)

        if not raw_documents:
            os.unlink(tmp_file_path)
            raise ValueError("PDF loader returned no documents.")
        print(f"Loaded {len(raw_documents)} pages/documents from PDF.")
        # raw_documents = loader.load()
        # RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) is reasonable. 
        # If chunks are too large, important info might be missed by the LLM. 
        # If too small, context is lost.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        # split_documents = text_splitter.split_documents(raw_documents)
        split_documents = await run_in_threadpool(text_splitter.split_documents, raw_documents)
        print(f"Split into {len(split_documents)} chunks.")

        if not split_documents:
            os.unlink(tmp_file_path)
            raise ValueError("Text splitter returned no chunks.")

        doc_id = str(uuid.uuid4())
        for i, doc_chunk in enumerate(split_documents):
            doc_chunk.metadata["source"] = file.filename
            doc_chunk.metadata["document_id"] = doc_id
            doc_chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"

        print(f"Attempting to add {len(split_documents)} chunks to PGVector collection: {collection_name}")
        print(f"Using connection string for PGVector (derived from settings): {settings.DATABASE_URL}")
        
        # vector_store = PGVector.from_documents(
        #     embeddings=embeddings_for_pgvector,  # Corrected: Use 'embeddings'
        #     documents=split_documents,
        #     collection_name=collection_name,
        #     connection=settings.DATABASE_URL,
        #     # use_jsonb=True, # Optional, can be useful for metadata flexibility
        #     # pre_delete_collection=False # Set to True if you want to clear before adding
        # )

        await run_in_threadpool(
            PGVector.from_documents,
            embeddings=embeddings_for_pgvector,  # Corrected: Use 'embeddings'
            documents=split_documents, # Pass the documents directly
            collection_name=collection_name, # Pass the collection name directly
            connection=settings.DATABASE_URL, # Pass the connection string directly
        )
        
        print("Documents presumably added to PGVector successfully.")
        os.unlink(tmp_file_path)

        return {
            "id": doc_id,
            "filename": file.filename,
            "total_chunks": len(split_documents),
            "collection_name": collection_name
        }

    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        print(f"Error in process_and_store_document: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during document processing: {type(e).__name__} - {e}")


async def query_document_with_rag(
    db_session: AsyncSession,
    question: str,
    collection_name: str,
    document_id: Optional[str] = None
) -> Tuple[str, List[str]]:
    if not embeddings_for_pgvector: # Check the renamed embedding model variable
        raise HTTPException(status_code=500, detail="Embedding model not initialized.")
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    try:
        print(f"Initializing PGVector store for querying collection: {collection_name}")
        vector_store = PGVector(
            collection_name=collection_name,
            connection=settings.DATABASE_URL,
            embeddings=embeddings_for_pgvector, # Corrected: Use 'embeddings'
            # use_jsonb=True # if you used it during from_documents
        )
        print("PGVector store for querying initialized.")

        # search_kwargs={"k": 3}. This means 3 chunks are retrieved. 
        # For tinyllama with its small context window, sending too much context might confuse it or cause it to ignore parts. 
        # Experiment with k=1 or k=2.
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        print("Retriever initialized.")

        # In query_document_with_rag, before creating RetrievalQA
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer from the context, just say that you don't know, do not try to make up an answer.
        Be concise and stick to the information found in the context.

        Context: {context}

        Question: {question}

        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Pass the custom prompt
        )
        print("QA chain initialized.")

        print(f"Invoking QA chain with question: {question}")
        
        # If qa_chain.invoke is synchronous, wrap it for async FastAPI
        # from fastapi.concurrency import run_in_threadpool
        # result = await run_in_threadpool(qa_chain.invoke, {"query": question})
        # For now, let's assume it's okay for testing or that invoke handles it
        # result = qa_chain.invoke({"query": question})
        result = await run_in_threadpool(qa_chain.invoke, {"query": question})

        answer = result.get("result", "No answer found.")
        source_documents = result.get("source_documents", [])
        source_chunks_content = [doc.page_content for doc in source_documents]

        print(f"Answer: {answer}")
        # print(f"Source chunks ({len(source_chunks_content)}): {source_chunks_content[:1]}...")

        return answer, source_chunks_content

    except Exception as e:
        print(f"Error in query_document_with_rag: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during query: {type(e).__name__} - {e}")
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid # For generating document IDs

from app.db.session import get_db
from app.models.document_models import DocumentRead, QueryRequest, QueryResponse
from app.services import rag_service # We will create this service next
from app.core.config import settings

router = APIRouter()

@router.post("/upload", response_model=DocumentRead) # Or a custom UploadResponse
async def upload_document(
    file: UploadFile = File(...),
    # collection_name: Optional[str] = Form(None), # For organizing documents later
    db: AsyncSession = Depends(get_db)
):
    """
    Uploads a PDF document, processes it, generates embeddings, and stores them.
    """
    if not file.filename.endswith((".pdf", ".PDF")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    try:
        # For MVP, let's use filename as a simple identifier.
        # In a real app, you'd generate a unique ID and store metadata.
        # For now, rag_service.process_and_store_document will handle specifics.
        # We'll make collection_name more robust later if needed for pgvector.
        # The collection_name in LangChain's PGVector is more like a table name for embeddings.

        # Use a fixed collection name for all documents in this MVP for simplicity.
        # This corresponds to the table where embeddings will be stored by LangChain PGVector.
        # Ensure this table name is valid for PostgreSQL.
        default_collection_name = "document_embeddings_mvp"

        document_info = await rag_service.process_and_store_document(
            db_session=db,
            file=file,
            collection_name=default_collection_name # Pass the collection name
        )
        # The document_info should contain id, filename, etc.
        # For now, let's assume rag_service returns something compatible with DocumentRead
        # This will need refinement based on what rag_service actually does.

        # Mocking response for now until rag_service is fully implemented
        return DocumentRead(
            id=document_info.get("id", uuid.uuid4()), # Get ID from service or generate
            filename=file.filename,
            content_type=file.content_type,
            size=file.size,
            uploaded_at="mock_timestamp" # Replace with actual timestamp
        )

    except Exception as e:
        # Log the exception properly in a real app
        print(f"Error during document upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_document(
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts a question and optionally a document_id (or uses a default context)
    to retrieve relevant information and generate an answer using an LLM.
    """
    try:
        # For MVP, use a fixed collection name matching the upload.
        default_collection_name = "document_embeddings_mvp"

        answer, source_chunks = await rag_service.query_document_with_rag(
            db_session=db,
            question=query_request.question,
            collection_name=default_collection_name, # Use the same collection name
            document_id=query_request.document_id # This might be used to filter later
        )
        return QueryResponse(answer=answer, source_chunks=source_chunks)
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")
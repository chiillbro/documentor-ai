# from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
# from sqlalchemy.ext.asyncio import AsyncSession
# from typing import List, Optional
# import uuid # For generating document IDs

# from app.db.session import get_db
# from app.models.document_models import DocumentRead, QueryRequest, QueryResponse
# from app.services import rag_service # We will create this service next
# from app.core.config import settings
# from app.core.exceptions import DocumentProcessingException, QueryProcessingException
# import logging
# from app.core.logging_config import get_logger

# import asyncio
# from fastapi.responses import StreamingResponse, JSONResponse
# from app.services.rag_service import embeddings_for_pgvector
# from langchain_postgres.vectorstores import PGVector


# logger = get_logger(__name__)
# # or logger = logging.getLogger(__name__)

# router = APIRouter()

# @router.post("/upload", response_model=DocumentRead) # Or a custom UploadResponse
# async def upload_document(
#     file: UploadFile = File(...),
#     # collection_name: Optional[str] = Form(None), # For organizing documents later
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Uploads a PDF document, processes it, generates embeddings, and stores them.
#     """
#     if not file.filename.endswith((".pdf", ".PDF")):
#         logger.error("Invalid file type. Only PDF files are allowed.")
#         raise DocumentProcessingException("Invalid file type. Only PDF files are allowed.")
#         # raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

#     try:
#         # For MVP, let's use filename as a simple identifier.
#         # In a real app, you'd generate a unique ID and store metadata.
#         # For now, rag_service.process_and_store_document will handle specifics.
#         # We'll make collection_name more robust later if needed for pgvector.
#         # The collection_name in LangChain's PGVector is more like a table name for embeddings.

#         # Use a fixed collection name for all documents in this MVP for simplicity.
#         # This corresponds to the table where embeddings will be stored by LangChain PGVector.
#         # Ensure this table name is valid for PostgreSQL.
#         default_collection_name = "document_embeddings_mvp" # or from settings

#         document_info_dict = await rag_service.process_and_store_document(
#             db_session=db,
#             file=file,
#             collection_name=default_collection_name # Pass the collection name
#         )
#         # The document_info should contain id, filename, etc.
#         # For now, let's assume rag_service returns something compatible with DocumentRead
#         # This will need refinement based on what rag_service actually does.

#         # Mocking response for now until rag_service is fully implemented
#         # return DocumentRead(
#         #     id=document_info.get("id", uuid.uuid4()), # Get ID from service or generate
#         #     filename=file.filename,
#         #     content_type=file.content_type,
#         #     size=file.size,
#         #     uploaded_at="mock_timestamp" # Replace with actual timestamp
#         # )

#         # Create DocumentRead object from the dictionary returned by the service
#         # Ensure keys match what DocumentRead expects or transform them.
#         # Our DocumentRead model uses 'id' (uuid.UUID) and 'uploaded_at' (str).
#         # The service returns 'id' as str(uuid.uuid4()) and 'uploaded_at' as ISO string.
#         return DocumentRead(
#             id=uuid.UUID(document_info_dict["id"]), # Convert string UUID back to UUID object for validation
#             filename=document_info_dict["filename"],
#             content_type=document_info_dict.get("content_type"),
#             size=document_info_dict.get("size"),
#             uploaded_at=document_info_dict["uploaded_at"]
#         )

#     # except Exception as e:
#     #     # Log the exception properly in a real app
#     #     # print(f"Error during document upload: {e}")
#     #     logger.error(f"Error during document upload: {e}", exc_info=True)
#     #     raise DocumentProcessingException(f"Failed to process document: {str(e)}")
#         # raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
#     except DocumentProcessingException as e: # Catch our custom exception
#         logger.error(f"Upload - DocumentProcessingException: {e.detail}", exc_info=True)
#         raise HTTPException(status_code=400, detail=e.detail) # e.g. bad file content
#     except Exception as e:
#         logger.error(f"Upload - Unhandled Exception: {type(e).__name__} - {str(e)}", exc_info=True)
#         # This will be caught by our generic DocuMentorException handler in main.py or return 500
#         raise HTTPException(status_code=500, detail="An unexpected error occurred during document upload.")


# # @router.post("/query", response_model=QueryResponse)
# # async def query_document(
# #     query_request: QueryRequest,
# #     db: AsyncSession = Depends(get_db)
# # ):
# #     """
# #     Accepts a question and optionally a document_id (or uses a default context)
# #     to retrieve relevant information and generate an answer using an LLM.
# #     """
# #     try:
# #         # For MVP, use a fixed collection name matching the upload.
# #         default_collection_name = "document_embeddings_mvp"

# #         answer, source_chunks = await rag_service.query_document_with_rag(
# #             db_session=db,
# #             question=query_request.question,
# #             collection_name=default_collection_name, # Use the same collection name
# #             document_id=query_request.document_id # This might be used to filter later
# #         )
# #         return QueryResponse(answer=answer, source_chunks=source_chunks)
# #     except Exception as e:
# #         # print(f"Error during query: {e}")
# #         logger.error(f"Error during query: {e}", exc_info=True)
# #         raise QueryProcessingException(f"Failed to process query: {str(e)}")
# #         # raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")




# # Your existing non-streaming query endpoint (can keep for testing or remove later)
# @router.post("/query", response_model=QueryResponse)
# async def query_document_sync( # Renamed to avoid conflict
#     query_request: QueryRequest,
#     db: AsyncSession = Depends(get_db)
# ):
#     # ... (existing implementation) ...
#     try:
#         default_collection_name = "document_embeddings_mvp"
#         answer, source_chunks = await rag_service.query_document_with_rag( # Call the non-streaming version
#             db_session=db,
#             question=query_request.question,
#             collection_name=default_collection_name,
#             document_id=query_request.document_id
#         )
#         return QueryResponse(answer=answer, source_chunks=source_chunks)
#     except QueryProcessingException as qpe:
#         logger.error(f"Query Sync - QueryProcessingException: {qpe.detail}", exc_info=True)
#         raise HTTPException(status_code=500, detail=qpe.detail)
#     except Exception as e: # Catch any other unexpected errors
#         logger.error(f"Query Sync - Unhandled Exception: {type(e).__name__} - {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An unexpected error occurred during query.")


# # New streaming query endpoint
# # @router.post("/query-stream")
# # async def query_document_stream_route(
# #     query_request: QueryRequest,
# #     db: AsyncSession = Depends(get_db)
# # ):
# #     """
# #     Accepts a question and streams the LLM's answer token by token.
# #     Source documents are not included in the stream for this MVP,
# #     they could be fetched separately or sent as a final chunk.
# #     """
# #     try:
# #         default_collection_name = "document_embeddings_mvp"
        
# #         # Define a generator that will call the service and yield its chunks
# #         async def event_generator():
# #             async for content_chunk in rag_service.stream_document_with_rag(
# #                 db_session=db,
# #                 question=query_request.question,
# #                 collection_name=default_collection_name,
# #                 document_id=query_request.document_id
# #             ):
# #                 if content_chunk.startswith("[ERROR:"): # Check for our error marker
# #                     # For SSE, errors should ideally be signaled differently,
# #                     # but for raw streaming, we just send the error string.
# #                     # Or we could raise an exception here to stop the stream.
# #                     logger.error(f"Streaming error from service: {content_chunk}")
# #                     yield f"data: {content_chunk}\n\n" # Send as SSE data for consistency
# #                     return # Stop streaming on error
                
# #                 # For Server-Sent Events (SSE) format, which is common for streaming
# #                 # yield f"data: {json.dumps({'token': content_chunk})}\n\n"
# #                 # For simpler plain text streaming for now:
# #                 yield content_chunk # Send raw chunk
# #                 await asyncio.sleep(0.01) # Small delay to allow chunks to flush, adjust as needed

# #         return StreamingResponse(event_generator(), media_type="text/plain") # Or "text/event-stream" for SSE

# #     except QueryProcessingException as qpe:
# #         logger.error(f"Query Stream - QueryProcessingException: {qpe.detail}", exc_info=True)
# #         # StreamingResponse doesn't easily propagate HTTPExceptions thrown after it starts.
# #         # Errors should be handled within the generator or before returning StreamingResponse.
# #         # This initial catch is if rag_service.stream_document_with_rag itself raises immediately.
# #         return JSONResponse(status_code=500, content={"detail": qpe.detail, "error_type": "QueryProcessingError"})
# #     except Exception as e:
# #         logger.error(f"Query Stream - Unhandled Exception: {type(e).__name__} - {str(e)}", exc_info=True)
# #         return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred during streaming query." , "error_type": "GenericStreamError"})


# import json # For SSE payload

# # ...
# # @router.post("/query-stream")
# @router.get("/query-stream")
# async def query_document_stream_route(
#     # query_request: QueryRequest,
#     # db: AsyncSession = Depends(get_db)
#     # Change QueryRequest from body to Query parameters
#     question: str = Query(..., min_length=1), 
#     document_id: Optional[str] = Query(None),
#     # collection_name: Optional[str] = Query(None), # If you want to pass this too
#     db: AsyncSession = Depends(get_db)
# ):
#     try:
#         default_collection_name = "document_embeddings_mvp"
        
#         async def sse_event_generator():
#             # Send a PING event occasionally to keep connection alive if needed (optional)
#             # yield "event: ping\ndata: {}\n\n"
            
#             # Signal start of content stream (custom event)
#             yield f"event: stream_start\ndata: {json.dumps({'message': 'Streaming started...'})}\n\n"

#             source_chunks_for_final_message: List[str] = []
#             full_answer_for_sources = "" # Accumulate for context

#             async for content_piece in rag_service.stream_document_with_rag(
#                 db_session=db,
#                 question=question,
#                 collection_name=default_collection_name,
#                 document_id=document_id
#             ):
#                 if isinstance(content_piece, str) and content_piece.startswith("[ERROR:"):
#                     error_detail = content_piece.replace("[ERROR:", "").replace("]", "").strip()
#                     logger.error(f"Streaming error from service: {error_detail}")
#                     error_payload = {"error": "LLMProcessingError", "detail": error_detail}
#                     yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
#                     return # Stop stream

#                 # Assuming content_piece is now a token or a small chunk of the answer
#                 # In a more complex setup, stream_document_with_rag might yield dicts
#                 # with different event types (e.g., token, source_info, end_of_stream)
#                 token_data = {"token": content_piece}
#                 full_answer_for_sources += content_piece # Accumulate for final sources
#                 yield f"event: token\ndata: {json.dumps(token_data)}\n\n"
#                 await asyncio.sleep(0.01) # Small delay for flushing, adjust

#             # After stream is finished, get source documents (this is a simplified approach for MVP)
#             # A more robust way would be for rag_service.stream_document_with_rag
#             # to yield a special object/event at the end containing the sources, or have a separate call.
#             # For now, let's re-run a simplified retriever to get sources related to the question
#             # This is NOT ideal as it re-runs retrieval, but simpler for MVP streaming.
#             # A better way: rag_service.stream_document_with_rag could be refactored to yield
#             # (type_of_event, data) tuples, e.g. ("token", "hello"), ("sources", [...])
            
#             logger.info(f"Attempting to retrieve source documents for question: {question}")
#             temp_vector_store = PGVector( # Re-init for source retrieval
#                  collection_name=default_collection_name,
#                  connection=settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://"),
#                  embeddings=embeddings_for_pgvector,
#             )
#             temp_retriever = temp_vector_store.as_retriever(search_kwargs={"k": 2})
#             # This is sync, wrap if it becomes an issue. For getting sources after stream, might be acceptable.
#             # For full async, use aget_relevant_documents
#             # relevant_docs = await temp_retriever.aget_relevant_documents(query_request.question)
#             relevant_docs = temp_retriever.get_relevant_documents(question) # Sync version
#             source_chunks_for_final_message = [doc.page_content for doc in relevant_docs]
            
#             sources_payload = {"sources": source_chunks_for_final_message}
#             yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

#             # Signal end of stream (custom event)
#             yield f"event: stream_end\ndata: {json.dumps({'message': 'Streaming ended.'})}\n\n"

#         return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
#     # ... (exception handling remains similar) ...
#     except QueryProcessingException as qpe:
#         # For SSE, it's hard to change to JSONResponse if stream already started.
#         # Ideally, errors are also sent as SSE events.
#         logger.error(f"Query Stream - QueryProcessingException: {qpe.detail}", exc_info=True)
#         async def error_stream(): # Send an error event
#             error_payload = {"error": "QueryProcessingError", "detail": qpe.detail}
#             yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
#         return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=500)
#     except Exception as e:
#         logger.error(f"Query Stream - Unhandled Exception: {type(e).__name__} - {str(e)}", exc_info=True)
#         async def error_stream():
#             error_payload = {"error": "GenericStreamError", "detail": "An unexpected error occurred."}
#             yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
#         return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=500)




# ************************* Version 2 ******************************


# app/apis/v1/routes_documents.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
# from sqlalchemy.ext.asyncio import AsyncSession # Not strictly needed if db dep removed
from typing import List, Optional
import uuid
import json

# from app.db.session import get_db # Not strictly needed if db dep removed
from app.models.document_models import DocumentRead, QueryRequest, QueryResponse
from app.services import rag_service
from app.core.config import settings
from app.core.exceptions import DocumentProcessingException, QueryProcessingException, LLMConnectionException
from app.core.logging_config import get_logger
import asyncio
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.rag_service import embeddings_for_pgvector # Not needed here
from langchain_postgres.vectorstores import PGVector # Not needed here
from app.core.config import  SYNC_DB_URL
from fastapi.concurrency import run_in_threadpool

logger = get_logger(__name__)
router = APIRouter()

@router.post("/upload", response_model=DocumentRead)
async def upload_document(
    file: UploadFile = File(...),
    # db: AsyncSession = Depends(get_db) # PGVector manages its own connection
):
    if not file.filename.endswith((".pdf", ".PDF")):
        logger.warning(f"Upload attempt with invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    try:
        default_collection_name = "document_embeddings_mvp"
        document_info_dict = await rag_service.process_and_store_document(
            # db_session=db, # Not passed to service anymore
            file=file,
            collection_name=default_collection_name
        )
        return DocumentRead(
            id=uuid.UUID(document_info_dict["id"]),
            filename=document_info_dict["filename"],
            content_type=document_info_dict.get("content_type"),
            size=document_info_dict.get("size"),
            uploaded_at=document_info_dict["uploaded_at"]
        )
    except DocumentProcessingException as e:
        logger.error(f"Upload - DocumentProcessingException: {e.detail}", exc_info=True)
        # Let FastAPI's exception handlers in main.py catch this for consistent response
        raise e 
    except Exception as e:
        logger.error(f"Upload - Unhandled Exception: {type(e).__name__} - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during document upload.")


@router.post("/query", response_model=QueryResponse)
async def query_document_sync(
    query_request: QueryRequest,
    # db: AsyncSession = Depends(get_db)
):
    try:
        default_collection_name = "document_embeddings_mvp"
        answer, source_chunks = await rag_service.query_document_with_rag(
            # db_session=db,
            question=query_request.question,
            collection_name=default_collection_name,
            document_id=query_request.document_id
        )
        return QueryResponse(answer=answer, source_chunks=source_chunks)
    except (QueryProcessingException, LLMConnectionException) as e:
        logger.error(f"Query Sync - Handled Exception: {e.detail}", exc_info=True)
        raise e # Let FastAPI's exception handlers in main.py catch this
    except Exception as e:
        logger.error(f"Query Sync - Unhandled Exception: {type(e).__name__} - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during query.")


@router.get("/query-stream")
async def query_document_stream_route(
    question: str = Query(..., min_length=1), 
    document_id: Optional[str] = Query(None),
    # db: AsyncSession = Depends(get_db)
):
    default_collection_name = "document_embeddings_mvp"
    
    async def sse_event_generator():
        yield f"event: stream_start\ndata: {json.dumps({'message': 'Streaming started...'})}\n\n"
        full_answer_for_sources = ""
        try:
            async for content_piece in rag_service.stream_document_with_rag(
                # db_session=db,
                question=question,
                collection_name=default_collection_name,
                document_id=document_id
            ):
                if isinstance(content_piece, str) and content_piece.startswith("[ERRORSTREAM]:"):
                    error_detail = content_piece.replace("[ERRORSTREAM]:", "").strip()
                    logger.error(f"Streaming error from service: {error_detail}")
                    error_payload = {"error": "LLMProcessingError", "detail": error_detail}
                    yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                    return 

                token_data = {"token": content_piece}
                full_answer_for_sources += content_piece
                yield f"event: token\ndata: {json.dumps(token_data)}\n\n"
                # await asyncio.sleep(0.01) # Small delay for flushing
                await asyncio.sleep(0)

            # Retrieve sources after the main answer stream
            # This part still uses a synchronous PGVector initialization for simplicity
            # and synchronous get_relevant_documents. If this becomes a bottleneck,
            # it should also be made fully async or use the _get_query_vector_store.
            def _get_sources_sync():
                logger.info(f"Attempting to retrieve source documents for question: {question}")
                # Use SYNC_DB_URL for this synchronous operation
                temp_vector_store = PGVector(
                     collection_name=default_collection_name,
                     connection=SYNC_DB_URL, # Use SYNC DSN
                     embeddings=embeddings_for_pgvector,
                     create_extension=False # Extension is already created
                )
                temp_retriever = temp_vector_store.as_retriever(search_kwargs={"k": 2})
                relevant_docs = temp_retriever.get_relevant_documents(question)
                return [doc.page_content for doc in relevant_docs]

            source_chunks_for_final_message = await run_in_threadpool(_get_sources_sync)
            
            sources_payload = {"sources": source_chunks_for_final_message}
            yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"
            yield f"event: stream_end\ndata: {json.dumps({'message': 'Streaming ended.'})}\n\n"

        except Exception as e_gen: # Catch errors from within the generator
            logger.error(f"SSE Generator Exception: {type(e_gen).__name__} - {str(e_gen)}", exc_info=True)
            error_payload = {"error": "StreamGenerationError", "detail": str(e_gen)[:150]}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"


    # This outer try-except is for errors occurring before the stream even starts
    try:
        return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
    except (QueryProcessingException, LLMConnectionException) as qpe:
        logger.error(f"Query Stream Init - Handled Exception: {qpe.detail}", exc_info=True)
        # Cannot return JSONResponse if stream might have started.
        # This path is if the error happens before StreamingResponse is returned.
        # For SSE, errors during streaming are handled within the generator.
        return JSONResponse(status_code=500, content={"detail": qpe.detail})
    except Exception as e:
        logger.error(f"Query Stream Init - Unhandled Exception: {type(e).__name__} - {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred preparing the stream."})
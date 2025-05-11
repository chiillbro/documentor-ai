from pydantic import BaseModel
from typing import Optional, List
import uuid

class DocumentBase(BaseModel):
    filename: str
    content_type: Optional[str] = None
    size: Optional[int] = None

class DocumentCreate(DocumentBase):
    pass # For now, same as base

class DocumentRead(DocumentBase):
    id: uuid.UUID # Or int if you prefer auto-incrementing PK
    uploaded_at: str # Or datetime

    class Config:
        from_attributes = True # Changed from orm_mode for Pydantic v2

class QueryRequest(BaseModel):
    document_id: Optional[str] = None # Or specific document identifier
    question: str
    # session_id: Optional[str] = None # For chat history later

class QueryResponse(BaseModel):
    answer: str
    source_chunks: Optional[List[str]] = None # For showing context
    # error: Optional[str] = None
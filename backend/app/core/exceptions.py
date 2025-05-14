class DocuMentorException(Exception):
    """Base exception for DocuMentor application."""
    def __init__(self, detail: str = "An application error occurred."):
        self.detail = detail
        super().__init__(self.detail)

class DocumentProcessingException(DocuMentorException):
    """Raised when document processing fails."""
    def __init__(self, detail: str = "Failed to process document."):
        super().__init__(detail)

class QueryProcessingException(DocuMentorException):
    """Raised when querying fails."""
    def __init__(self, detail: str = "Failed to process query."):
        super().__init__(detail)

class LLMConnectionException(DocuMentorException):
    """Raised when connection to LLM fails."""
    def __init__(self, detail: str = "Failed to connect to or get response from LLM."):
        super().__init__(detail)

class DatabaseException(DocuMentorException):
    """Raised for database related errors."""
    def __init__(self, detail: str = "A database error occurred."):
        super().__init__(detail)
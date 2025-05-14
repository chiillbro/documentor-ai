from fastapi import FastAPI, Request, HTTPException
# Import logging config first to set it up
from app.core import logging_config # This will execute the logging setup
from dotenv import load_dotenv
from app.core.config import settings
from app.db.session import init_db
from contextlib import asynccontextmanager
import logging # Import standard logging
from app.core.exceptions import DocuMentorException, DocumentProcessingException, QueryProcessingException
from fastapi.responses import JSONResponse
# import the router

from app.apis.v1 import routes_documents

# load_dotenv() # pydantic-settings should handle .env loading if configured

from fastapi.middleware.cors import CORSMiddleware


# Get a logger for this module (optional, but good practice)
logger = logging_config.get_logger(__name__)
# or logger = logging.getLogger(__name__) if used basicConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # print("Application startup...")4
    logger.info("Application startup...")
    await init_db() # Initialize database (create extension, tables)
    # print("Database initialized.")
    logger.info("Database initialized.")
    yield
    # Shutdown
    # print("Application shutdown...")
    logger.info("Application shutdown...")

app = FastAPI(title=settings.APP_NAME, version=settings.API_V1_STR, debug=settings.DEBUG_MODE, lifespan=lifespan)


# --- CORS Middleware Configuration ---
# Define the origins that are allowed to make requests.
# For development, allowing localhost:3000 is common.
# For production, you'd list your actual frontend domain(s).
origins = [
    "http://localhost:3000", # Your frontend Next.js development server
    # Add other origins if needed, e.g., your deployed frontend URL
    # "https://your-deployed-frontend.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    # allow_origins=["*"], # Allows all origins (use with caution, generally not for production)
    allow_credentials=True, # Allows cookies to be included in requests (if you use them later)
    allow_methods=["*"],    # Allows all standard HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allows all headers
)
# --- END CORS Middleware Configuration ---

# --- Exception Handlers ---
@app.exception_handler(DocuMentorException) # Base custom exception
async def documentor_exception_handler(request: Request, exc: DocuMentorException):
    logger.error(f"DocuMentorException: {exc.detail}", exc_info=exc) # Log with traceback
    return JSONResponse(
        status_code=500, # Or a more specific code if the exception implies it
        content={"detail": exc.detail},
    )

@app.exception_handler(DocumentProcessingException)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingException):
    logger.error(f"DocumentProcessingException: {exc.detail}", exc_info=exc)
    return JSONResponse(
        status_code=500, # Could be 400 if it's a bad file, 500 for internal
        content={"detail": exc.detail, "error_type": "DocumentProcessingError"},
    )

@app.exception_handler(QueryProcessingException)
async def query_processing_exception_handler(request: Request, exc: QueryProcessingException):
    logger.error(f"QueryProcessingException: {exc.detail}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": exc.detail, "error_type": "QueryProcessingError"},
    )

# You can add handlers for more specific built-in exceptions too
# from sqlalchemy.exc import SQLAlchemyError
# @app.exception_handler(SQLAlchemyError)
# async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
#     logger.error(f"Database Error: {exc}", exc_info=True)
#     return JSONResponse(status_code=500, content={"detail": "A database error occurred."})


# Catch-all for unhandled HTTPErrors from our services (if we still use them)
# @app.exception_handler(HTTPException)
# async def http_exception_handler_custom(request: Request, exc: HTTPException):
#     logger.error(f"HTTPException caught by custom handler: {exc.status_code} - {exc.detail}")
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"detail": exc.detail},
#     )

app.include_router(routes_documents.router, prefix=settings.API_V1_STR + "/documents", tags=["Documents"])

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}!"}

# We will add routers later
# from app.apis.v1 import routes_documents
# app.include_router(routes_documents.router, prefix="/api/v1/documents", tags=["documents"])
from fastapi import FastAPI
from dotenv import load_dotenv
from app.core.config import settings
from app.db.session import init_db
from contextlib import asynccontextmanager

# import the router

from app.apis.v1 import routes_documents

# load_dotenv() # pydantic-settings should handle .env loading if configured


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Application startup...")
    await init_db() # Initialize database (create extension, tables)
    print("Database initialized.")
    yield
    # Shutdown
    print("Application shutdown...")

app = FastAPI(title=settings.APP_NAME, version=settings.API_V1_STR, debug=settings.DEBUG_MODE, lifespan=lifespan)


app.include_router(routes_documents.router, prefix=settings.API_V1_STR + "/documents", tags=["Documents"])

@app.get("/")
async def root():
    return {"message": "Welcome to Documentor AI API"}

# We will add routers later
# from app.apis.v1 import routes_documents
# app.include_router(routes_documents.router, prefix="/api/v1/documents", tags=["documents"])
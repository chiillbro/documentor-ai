from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.core.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.sql import text


# For synchronous operations if needed by some LangChain components
# SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
# engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# For asynchronous operations with FastAPI
ASYNC_SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
async_engine = create_async_engine(ASYNC_SQLALCHEMY_DATABASE_URL, echo=settings.DB_ECHO_LOG)
AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False, autocommit=False, autoflush=False
)

Base = declarative_base() # For defining DB models with SQLAlchemy ORM

# Dependency to get DB session in FastAPI routes
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# Function to create tables (and enable pgvector extension)
# We'll call this during app startup
async def init_db():
    async with async_engine.connect() as conn:
        # Enable pgvector extension if not already enabled
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.commit()
    # If using SQLAlchemy ORM models, create tables:
    # async with async_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)
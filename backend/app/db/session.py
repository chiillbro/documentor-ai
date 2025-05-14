# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
# from app.core.config import settings
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.sql import text
# from app.core.logging_config import get_logger

# logger = get_logger(__name__)


# # For synchronous operations if needed by some LangChain components
# # SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
# # engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
# # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # For asynchronous operations with FastAPI
# ASYNC_SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
# async_engine = create_async_engine(ASYNC_SQLALCHEMY_DATABASE_URL, echo=settings.DB_ECHO_LOG)
# AsyncSessionLocal = sessionmaker(
#     bind=async_engine, class_=AsyncSession, expire_on_commit=False, autocommit=False, autoflush=False
# )

# Base = declarative_base() # For defining DB models with SQLAlchemy ORM

# # Dependency to get DB session in FastAPI routes
# async def get_db() -> AsyncSession:
#     async with AsyncSessionLocal() as session:
#         yield session

# # Function to create tables (and enable pgvector extension)
# # We'll call this during app startup
# async def init_db():
#     async with async_engine.connect() as conn:
#         # Enable pgvector extension if not already enabled
#         await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
#         await conn.commit()
#         logger.info("Ensured 'vector' extension exists.") # Add a log
#     # If using SQLAlchemy ORM models, create tables:
#     # async with async_engine.begin() as conn:
#     #     await conn.run_sync(Base.metadata.create_all)







# app/db/session.py
from sqlalchemy import create_engine, text # Ensure text is imported
from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base # No longer needed if not using ORM models here
from app.core.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.core.logging_config import get_logger

logger = get_logger(__name__)

ASYNC_SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
async_engine = create_async_engine(ASYNC_SQLALCHEMY_DATABASE_URL, echo=settings.DB_ECHO_LOG)

AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False, autocommit=False, autoflush=False
)

# Base = declarative_base() # Remove if no ORM models are defined in this project yet

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# async def init_db():
#     async with async_engine.connect() as conn:
#         # It's generally safer to execute these as separate commands if the driver/DB has issues
#         # with multi-statement strings in some contexts, though CREATE EXTENSION is often fine.
#         # The advisory lock is a good practice from Langchain to prevent race conditions
#         # if multiple processes try to create the extension simultaneously.
#         try:
#             # Advisory lock to prevent race conditions if multiple instances start up
#             await conn.execute(text("SELECT pg_advisory_xact_lock(1573678846307946496);"))
#             await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
#             await conn.commit() # Commit after creating extension
#             logger.info("Ensured 'vector' extension exists (with advisory lock).")
#         except Exception as e:
#             await conn.rollback() # Rollback on error
#             logger.error(f"Error during init_db creating vector extension: {e}", exc_info=True)
#             # Depending on the error, you might want to re-raise or handle
#         finally:
#             # Advisory locks are transaction-scoped, so it should be released on commit/rollback.
#             # Explicit unlock if needed: await conn.execute(text("SELECT pg_advisory_xact_unlock_all();"))
#             pass # Lock will be released by commit/rollback of transaction
#     # If using SQLAlchemy ORM models, create tables:
#     # async with async_engine.begin() as conn:
#     #     await conn.run_sync(Base.metadata.create_all)


async def init_db():
    """
    Initializes the database:
    1. Acquires an advisory lock to prevent race conditions.
    2. Creates the 'vector' extension if it doesn't exist.
    This is called once at application startup.
    """
    logger.info("Attempting to initialize database and ensure 'vector' extension...")
    async with async_engine.connect() as conn:
        try:
            # Using a transaction for the lock and extension creation
            async with conn.begin():
                await conn.execute(text("SELECT pg_advisory_xact_lock(1573678846307946496);"))
                logger.info("Advisory lock acquired for vector extension creation.")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                # No explicit commit needed here as `conn.begin()` handles it on successful exit.
            logger.info("Successfully ensured 'vector' extension exists (with advisory lock).")
        except Exception as e:
            # Rollback should happen automatically if an error occurs within `conn.begin()`
            logger.error(f"Error during init_db creating vector extension: {e}", exc_info=True)
            # Re-raise to potentially stop app startup if DB init fails critically
            raise RuntimeError(f"Failed to initialize database vector extension: {e}") from e
        # Advisory lock is transaction-scoped and will be released.

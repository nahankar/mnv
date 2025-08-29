"""Database connection utilities"""
import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager
import logging
import sqlalchemy as sa

from .config import get_config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models"""
    pass


class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self, database_url: Optional[str] = None):
        """Initialize database connection"""
        if self._initialized:
            return
            
        config = get_config()
        db_url = database_url or config.database_url
        
        # Create async engine with connection pooling
        self.engine = create_async_engine(
            db_url,
            echo=config.log_level == "DEBUG",
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
            poolclass=NullPool if config.environment == "test" else None
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self._initialized = True
        logger.info("Database connection initialized")
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager"""
        if not self._initialized:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def get_db_connection() -> AsyncGenerator[AsyncSession, None]:
    """Get database connection (dependency injection helper)"""
    db_manager = get_db_manager()
    async with db_manager.get_session() as session:
        yield session


# Repository base class for common database operations
class BaseRepository:
    """Base repository class with common CRUD operations"""
    
    def __init__(self, session: AsyncSession, model_class):
        self.session = session
        self.model_class = model_class
    
    async def create(self, **kwargs):
        """Create a new record"""
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance
    
    async def get_by_id(self, id):
        """Get record by ID"""
        result = await self.session.get(self.model_class, id)
        return result
    
    async def get_all(self, limit: int = 100, offset: int = 0):
        """Get all records with pagination"""
        from sqlalchemy import select
        stmt = select(self.model_class).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def update(self, id, **kwargs):
        """Update record by ID"""
        instance = await self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            await self.session.flush()
            await self.session.refresh(instance)
        return instance
    
    async def delete(self, id):
        """Delete record by ID"""
        instance = await self.get_by_id(id)
        if instance:
            await self.session.delete(instance)
            await self.session.flush()
            return True
        return False
    
    async def count(self):
        """Count total records"""
        from sqlalchemy import select, func
        stmt = select(func.count(self.model_class.id))
        result = await self.session.execute(stmt)
        return result.scalar()


# Database health check utility
async def check_database_health() -> dict:
    """Check database connection health"""
    try:
        db_manager = get_db_manager()
        async with db_manager.get_session() as session:
            # Simple query to test connection
            result = await session.execute(sa.text("SELECT 1"))
            result.scalar()
            
            return {
                "status": "healthy",
                "message": "Database connection successful"
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }


async def init_database():
    """Initialize database connection"""
    db_manager = get_db_manager()
    await db_manager.initialize()


async def close_database():
    """Close database connections"""
    db_manager = get_db_manager()
    await db_manager.close()
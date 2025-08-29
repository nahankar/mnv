"""Tests for database utilities and operations"""
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy.exc import SQLAlchemyError

from shared.database import (
    DatabaseManager, BaseRepository, get_db_manager, 
    check_database_health, init_database, close_database
)
from shared.models import Story, ContentStatus


class TestDatabaseManager:
    """Tests for DatabaseManager class"""
    
    async def test_initialize_database_manager(self):
        """Test database manager initialization"""
        manager = DatabaseManager()
        assert manager.engine is None
        assert manager.session_factory is None
        assert manager._initialized is False
        
        # Initialize with test database URL
        await manager.initialize("sqlite+aiosqlite:///:memory:")
        
        assert manager.engine is not None
        assert manager.session_factory is not None
        assert manager._initialized is True
        
        # Cleanup
        await manager.close()
    
    async def test_get_session_context_manager(self, test_engine):
        """Test session context manager"""
        manager = DatabaseManager()
        await manager.initialize("sqlite+aiosqlite:///:memory:")
        
        async with manager.get_session() as session:
            assert session is not None
            # Session should be active
            assert session.is_active
        
        # Session should be closed after context
        assert not session.is_active
        
        await manager.close()
    
    async def test_close_database_manager(self):
        """Test closing database manager"""
        manager = DatabaseManager()
        await manager.initialize("sqlite+aiosqlite:///:memory:")
        
        assert manager._initialized is True
        
        await manager.close()
        
        assert manager._initialized is False


class TestBaseRepository:
    """Tests for BaseRepository class"""
    
    async def test_create_record(self, test_session):
        """Test creating a record using repository"""
        repo = BaseRepository(test_session, Story)
        
        story = await repo.create(
            title="Repository Test Story",
            content="Content created via repository",
            genre="test",
            llm_provider="test_provider"
        )
        
        assert story.id is not None
        assert story.title == "Repository Test Story"
        assert story.content == "Content created via repository"
        assert story.status == ContentStatus.PENDING
    
    async def test_get_by_id(self, test_session, sample_story):
        """Test getting record by ID"""
        repo = BaseRepository(test_session, Story)
        
        retrieved_story = await repo.get_by_id(sample_story.id)
        
        assert retrieved_story is not None
        assert retrieved_story.id == sample_story.id
        assert retrieved_story.title == sample_story.title
    
    async def test_get_by_id_not_found(self, test_session):
        """Test getting non-existent record by ID"""
        repo = BaseRepository(test_session, Story)
        
        import uuid
        non_existent_id = uuid.uuid4()
        retrieved_story = await repo.get_by_id(non_existent_id)
        
        assert retrieved_story is None
    
    async def test_get_all_with_pagination(self, test_session):
        """Test getting all records with pagination"""
        repo = BaseRepository(test_session, Story)
        
        # Create multiple stories
        for i in range(5):
            await repo.create(
                title=f"Story {i}",
                content=f"Content {i}",
                genre="test"
            )
        
        # Get first page
        stories_page1 = await repo.get_all(limit=3, offset=0)
        assert len(stories_page1) == 3
        
        # Get second page
        stories_page2 = await repo.get_all(limit=3, offset=3)
        assert len(stories_page2) == 2
    
    async def test_update_record(self, test_session, sample_story):
        """Test updating a record"""
        repo = BaseRepository(test_session, Story)
        
        updated_story = await repo.update(
            sample_story.id,
            title="Updated Title",
            genre="updated_genre",
            status=ContentStatus.COMPLETED
        )
        
        assert updated_story is not None
        assert updated_story.title == "Updated Title"
        assert updated_story.genre == "updated_genre"
        assert updated_story.status == ContentStatus.COMPLETED
        # Original content should remain unchanged
        assert updated_story.content == sample_story.content
    
    async def test_update_nonexistent_record(self, test_session):
        """Test updating non-existent record"""
        repo = BaseRepository(test_session, Story)
        
        import uuid
        non_existent_id = uuid.uuid4()
        result = await repo.update(non_existent_id, title="New Title")
        
        assert result is None
    
    async def test_delete_record(self, test_session, sample_story):
        """Test deleting a record"""
        repo = BaseRepository(test_session, Story)
        
        # Verify story exists
        story = await repo.get_by_id(sample_story.id)
        assert story is not None
        
        # Delete story
        deleted = await repo.delete(sample_story.id)
        assert deleted is True
        
        # Verify story is deleted
        story = await repo.get_by_id(sample_story.id)
        assert story is None
    
    async def test_delete_nonexistent_record(self, test_session):
        """Test deleting non-existent record"""
        repo = BaseRepository(test_session, Story)
        
        import uuid
        non_existent_id = uuid.uuid4()
        deleted = await repo.delete(non_existent_id)
        
        assert deleted is False
    
    async def test_count_records(self, test_session):
        """Test counting records"""
        repo = BaseRepository(test_session, Story)
        
        # Initially should be 0
        count = await repo.count()
        assert count == 0
        
        # Create some stories
        for i in range(3):
            await repo.create(
                title=f"Story {i}",
                content=f"Content {i}",
                genre="test"
            )
        
        # Count should be 3
        count = await repo.count()
        assert count == 3


class TestDatabaseHealthCheck:
    """Tests for database health check"""
    
    @patch('shared.database.get_db_manager')
    async def test_database_health_check_success(self, mock_get_db_manager):
        """Test successful database health check"""
        # Mock database manager and session
        mock_manager = AsyncMock()
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar.return_value = 1
        mock_manager.get_session.return_value.__aenter__.return_value = mock_session
        mock_get_db_manager.return_value = mock_manager
        
        result = await check_database_health()
        
        assert result["status"] == "healthy"
        assert "successful" in result["message"]
    
    @patch('shared.database.get_db_manager')
    async def test_database_health_check_failure(self, mock_get_db_manager):
        """Test failed database health check"""
        # Mock database manager to raise exception
        mock_manager = AsyncMock()
        mock_manager.get_session.side_effect = SQLAlchemyError("Connection failed")
        mock_get_db_manager.return_value = mock_manager
        
        result = await check_database_health()
        
        assert result["status"] == "unhealthy"
        assert "Connection failed" in result["message"]


class TestDatabaseUtilities:
    """Tests for database utility functions"""
    
    @patch('shared.database._db_manager', None)
    def test_get_db_manager_singleton(self):
        """Test that get_db_manager returns singleton instance"""
        manager1 = get_db_manager()
        manager2 = get_db_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, DatabaseManager)
    
    @patch('shared.database.get_db_manager')
    async def test_init_database(self, mock_get_db_manager):
        """Test database initialization utility"""
        mock_manager = AsyncMock()
        mock_get_db_manager.return_value = mock_manager
        
        await init_database()
        
        mock_manager.initialize.assert_called_once()
    
    @patch('shared.database.get_db_manager')
    async def test_close_database(self, mock_get_db_manager):
        """Test database close utility"""
        mock_manager = AsyncMock()
        mock_get_db_manager.return_value = mock_manager
        
        await close_database()
        
        mock_manager.close.assert_called_once()
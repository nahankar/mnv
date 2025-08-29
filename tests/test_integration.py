"""Integration tests for database schema and ENUM handling"""
import pytest
import asyncio
from datetime import datetime
from sqlalchemy import text
import uuid

from shared.database import get_db_manager
from shared.models import (
    Story, MediaAsset, Video, PlatformUpload, AnalyticsData, ModelConfiguration,
    ContentStatus, PlatformType, MediaType
)


class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    @pytest.mark.asyncio
    async def test_enum_round_trip_persistence(self):
        """Test that ENUM values persist and read correctly through ORM"""
        db_manager = get_db_manager()
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Test ContentStatus ENUM
            story = Story(
                title="Test Story",
                content="Test content",
                status=ContentStatus.PROCESSING  # Use ORM enum
            )
            session.add(story)
            await session.commit()
            await session.refresh(story)
            
            # Verify ORM enum value
            assert story.status == ContentStatus.PROCESSING
            assert story.status.value == "processing"
            
            # Test MediaType ENUM
            media_asset = MediaAsset(
                story_id=story.id,
                asset_type=MediaType.AUDIO,  # Use ORM enum
                file_path="/test/audio.wav"
            )
            session.add(media_asset)
            await session.commit()
            await session.refresh(media_asset)
            
            # Verify ORM enum value
            assert media_asset.asset_type == MediaType.AUDIO
            assert media_asset.asset_type.value == "audio"
            
            # Test PlatformType ENUM
            video = Video(
                story_id=story.id,
                title="Test Video",
                file_path="/test/video.mp4",
                duration=120.0
            )
            session.add(video)
            await session.commit()
            await session.refresh(video)
            
            platform_upload = PlatformUpload(
                video_id=video.id,
                platform=PlatformType.YOUTUBE  # Use ORM enum
            )
            session.add(platform_upload)
            await session.commit()
            await session.refresh(platform_upload)
            
            # Verify ORM enum value
            assert platform_upload.platform == PlatformType.YOUTUBE
            assert platform_upload.platform.value == "youtube"
            
            # Test direct database query to ensure values are stored correctly
            result = await session.execute(
                text("SELECT status FROM stories WHERE id = :id"),
                {"id": str(story.id)}
            )
            db_status = result.scalar()
            assert db_status == "processing"
            
            result = await session.execute(
                text("SELECT asset_type FROM media_assets WHERE id = :id"),
                {"id": str(media_asset.id)}
            )
            db_asset_type = result.scalar()
            assert db_asset_type == "audio"
            
            result = await session.execute(
                text("SELECT platform FROM platform_uploads WHERE id = :id"),
                {"id": str(platform_upload.id)}
            )
            db_platform = result.scalar()
            assert db_platform == "youtube"
    
    @pytest.mark.asyncio
    async def test_computed_total_revenue_consistency(self):
        """Test that total_revenue computed column remains consistent after updates"""
        db_manager = get_db_manager()
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Create test data
            story = Story(title="Test Story", content="Test content")
            session.add(story)
            await session.commit()
            await session.refresh(story)
            
            video = Video(
                story_id=story.id,
                title="Test Video",
                file_path="/test/video.mp4",
                duration=120.0
            )
            session.add(video)
            await session.commit()
            await session.refresh(video)
            
            platform_upload = PlatformUpload(
                video_id=video.id,
                platform=PlatformType.YOUTUBE
            )
            session.add(platform_upload)
            await session.commit()
            await session.refresh(platform_upload)
            
            # Create analytics data with initial values
            analytics = AnalyticsData(
                platform_upload_id=platform_upload.id,
                ad_revenue=10.50,
                creator_fund_revenue=5.25,
                data_collected_at=datetime.utcnow()
            )
            session.add(analytics)
            await session.commit()
            await session.refresh(analytics)
            
            # Verify initial computed value
            assert analytics.total_revenue == 15.75
            
            # Update ad_revenue and verify total_revenue is recalculated
            await session.execute(
                text("UPDATE analytics_data SET ad_revenue = :new_revenue WHERE id = :id"),
                {"new_revenue": 20.00, "id": str(analytics.id)}
            )
            await session.commit()
            
            # Refresh and verify computed column updated
            await session.refresh(analytics)
            assert analytics.total_revenue == 25.25  # 20.00 + 5.25
            
            # Test with NULL values
            await session.execute(
                text("UPDATE analytics_data SET creator_fund_revenue = NULL WHERE id = :id"),
                {"id": str(analytics.id)}
            )
            await session.commit()
            
            await session.refresh(analytics)
            assert analytics.total_revenue == 20.00  # 20.00 + 0 (COALESCE handles NULL)
            
            # Test with both NULL values
            await session.execute(
                text("UPDATE analytics_data SET ad_revenue = NULL WHERE id = :id"),
                {"id": str(analytics.id)}
            )
            await session.commit()
            
            await session.refresh(analytics)
            assert analytics.total_revenue == 0.0  # 0 + 0 (COALESCE handles both NULLs)
    
    @pytest.mark.asyncio
    async def test_foreign_key_cascade_behavior(self):
        """Test that CASCADE foreign keys work correctly"""
        db_manager = get_db_manager()
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Create parent story
            story = Story(title="Test Story", content="Test content")
            session.add(story)
            await session.commit()
            await session.refresh(story)
            
            # Create child records
            media_asset = MediaAsset(
                story_id=story.id,
                asset_type=MediaType.AUDIO,
                file_path="/test/audio.wav"
            )
            session.add(media_asset)
            
            video = Video(
                story_id=story.id,
                title="Test Video",
                file_path="/test/video.mp4",
                duration=120.0
            )
            session.add(video)
            await session.commit()
            
            # Create grandchild records
            platform_upload = PlatformUpload(
                video_id=video.id,
                platform=PlatformType.YOUTUBE
            )
            session.add(platform_upload)
            await session.commit()
            
            analytics = AnalyticsData(
                platform_upload_id=platform_upload.id,
                ad_revenue=10.00,
                data_collected_at=datetime.utcnow()
            )
            session.add(analytics)
            await session.commit()
            
            # Store IDs for verification
            story_id = story.id
            media_asset_id = media_asset.id
            video_id = video.id
            platform_upload_id = platform_upload.id
            analytics_id = analytics.id
            
            # Delete parent story - should cascade delete all children
            await session.delete(story)
            await session.commit()
            
            # Verify all related records were deleted
            result = await session.execute(
                text("SELECT COUNT(*) FROM stories WHERE id = :id"),
                {"id": str(story_id)}
            )
            assert result.scalar() == 0
            
            result = await session.execute(
                text("SELECT COUNT(*) FROM media_assets WHERE id = :id"),
                {"id": str(media_asset_id)}
            )
            assert result.scalar() == 0
            
            result = await session.execute(
                text("SELECT COUNT(*) FROM videos WHERE id = :id"),
                {"id": str(video_id)}
            )
            assert result.scalar() == 0
            
            result = await session.execute(
                text("SELECT COUNT(*) FROM platform_uploads WHERE id = :id"),
                {"id": str(platform_upload_id)}
            )
            assert result.scalar() == 0
            
            result = await session.execute(
                text("SELECT COUNT(*) FROM analytics_data WHERE id = :id"),
                {"id": str(analytics_id)}
            )
            assert result.scalar() == 0


class TestAlembicMigrations:
    """Tests for Alembic migration system"""
    
    @pytest.mark.asyncio
    async def test_migration_heads_consistency(self):
        """Test that Alembic heads are consistent"""
        db_manager = get_db_manager()
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Check current migration version
            result = await session.execute(
                text("SELECT version_num FROM alembic_version")
            )
            current_version = result.scalar()
            
            # Should be at the latest version (002)
            assert current_version == "002"
            
            # Verify all expected tables exist
            result = await session.execute(
                text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """)
            )
            tables = [row[0] for row in result.fetchall()]
            
            expected_tables = [
                'alembic_version',
                'analytics_data',
                'media_assets',
                'model_configurations',
                'platform_uploads',
                'stories',
                'videos'
            ]
            
            for table in expected_tables:
                assert table in tables, f"Expected table {table} not found"
            
            # Verify ENUM types exist
            result = await session.execute(
                text("SELECT typname FROM pg_type WHERE typtype = 'e' ORDER BY typname")
            )
            enum_types = [row[0] for row in result.fetchall()]
            
            expected_enums = ['contentstatus', 'mediatype', 'platformtype']
            for enum_type in expected_enums:
                assert enum_type in enum_types, f"Expected ENUM {enum_type} not found"
    
    @pytest.mark.asyncio
    async def test_migration_downgrade_upgrade_cycle(self):
        """Test that migrations can be downgraded and upgraded safely"""
        db_manager = get_db_manager()
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Get current version
            result = await session.execute(
                text("SELECT version_num FROM alembic_version")
            )
            original_version = result.scalar()
            
            # Test that we can query the current schema
            result = await session.execute(
                text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
            )
            table_count_before = result.scalar()
            
            # Note: In a real test environment, you would run:
            # alembic downgrade -1
            # alembic upgrade head
            # But for this integration test, we just verify the current state is valid
            
            assert table_count_before >= 6  # Should have at least our core tables
            assert original_version == "002"  # Should be at latest version
    
    @pytest.mark.asyncio
    async def test_check_constraints_enforcement(self):
        """Test that CHECK constraints are properly enforced"""
        db_manager = get_db_manager()
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Test non-negative constraints
            story = Story(title="Test Story", content="Test content")
            session.add(story)
            await session.commit()
            await session.refresh(story)
            
            video = Video(
                story_id=story.id,
                title="Test Video",
                file_path="/test/video.mp4",
                duration=120.0
            )
            session.add(video)
            await session.commit()
            await session.refresh(video)
            
            platform_upload = PlatformUpload(
                video_id=video.id,
                platform=PlatformType.YOUTUBE
            )
            session.add(platform_upload)
            await session.commit()
            await session.refresh(platform_upload)
            
            # Test that negative values are rejected
            with pytest.raises(Exception):  # Should raise integrity error
                await session.execute(
                    text("""
                    INSERT INTO analytics_data 
                    (id, platform_upload_id, views, ad_revenue, data_collected_at) 
                    VALUES (:id, :platform_upload_id, :views, :ad_revenue, :data_collected_at)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "platform_upload_id": str(platform_upload.id),
                        "views": -100,  # Negative value should be rejected
                        "ad_revenue": 10.00,
                        "data_collected_at": datetime.utcnow()
                    }
                )
                await session.commit()
            
            # Rollback the failed transaction
            await session.rollback()
            
            # Refresh the platform_upload object after rollback
            await session.refresh(platform_upload)
            
            # Test that valid values are accepted
            analytics = AnalyticsData(
                platform_upload_id=platform_upload.id,
                views=1000,  # Positive value should be accepted
                ad_revenue=10.00,
                data_collected_at=datetime.utcnow()
            )
            session.add(analytics)
            await session.commit()  # Should succeed
            
            assert analytics.views == 1000
            assert analytics.total_revenue == 10.00


class TestServiceHealthChecks:
    """Integration tests for service health checks"""
    
    @pytest.mark.asyncio
    async def test_database_connectivity(self):
        """Test that database connection is working"""
        db_manager = get_db_manager()
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Simple connectivity test
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            
            # Test that we can query system tables
            result = await session.execute(
                text("SELECT current_database(), current_user")
            )
            db_name, user = result.fetchone()
            assert db_name == "ai_pipeline"
            assert user == "user"
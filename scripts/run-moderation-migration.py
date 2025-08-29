#!/usr/bin/env python3
"""
Script to run the moderation service database migration

This script applies the moderation schema migration to add the new
moderation-related tables to the database.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alembic.config import Config
from alembic import command
from shared.database import DatabaseManager
from shared.logging import get_logger

logger = get_logger(__name__)


async def run_migration():
    """Run the moderation migration"""
    try:
        # Get database configuration
        db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/ai_pipeline")
        
        # Create Alembic configuration
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        
        logger.info("Starting moderation migration...")
        
        # Run the migration
        command.upgrade(alembic_cfg, "003")
        
        logger.info("Moderation migration completed successfully!")
        
        # Verify the migration
        await verify_migration()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


async def verify_migration():
    """Verify that the migration was applied correctly"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Check if the new tables exist
            from sqlalchemy import text
            
            # Check content_moderations table
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'content_moderations'
                );
            """))
            content_moderations_exists = result.scalar()
            
            # Check manual_review_queue table
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'manual_review_queue'
                );
            """))
            review_queue_exists = result.scalar()
            
            # Check audit_logs table
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'audit_logs'
                );
            """))
            audit_logs_exists = result.scalar()
            
            # Check enums
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM pg_type 
                    WHERE typname = 'moderationtype'
                );
            """))
            moderationtype_exists = result.scalar()
            
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM pg_type 
                    WHERE typname = 'moderationstatus'
                );
            """))
            moderationstatus_exists = result.scalar()
        
        await db_manager.close()
        
        if all([content_moderations_exists, review_queue_exists, audit_logs_exists, 
                moderationtype_exists, moderationstatus_exists]):
            logger.info("✅ All moderation tables and enums created successfully!")
        else:
            logger.error("❌ Some tables or enums are missing:")
            logger.error(f"  - content_moderations: {content_moderations_exists}")
            logger.error(f"  - manual_review_queue: {review_queue_exists}")
            logger.error(f"  - audit_logs: {audit_logs_exists}")
            logger.error(f"  - moderationtype enum: {moderationtype_exists}")
            logger.error(f"  - moderationstatus enum: {moderationstatus_exists}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)


def main():
    """Main function"""
    logger.info("Moderation Service Migration Script")
    logger.info("==================================")
    
    # Check if we're in the right directory
    if not Path("alembic.ini").exists():
        logger.error("alembic.ini not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Run the migration
    asyncio.run(run_migration())


if __name__ == "__main__":
    main()

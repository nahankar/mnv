#!/usr/bin/env python3
"""
Script to run database migrations
"""
import os
import sys
import asyncio
from alembic.config import Config
from alembic import command

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.config import get_config


def run_migrations():
    """Run database migrations"""
    print("Running database migrations...")
    
    # Get database URL from config
    config = get_config()
    database_url = config.database_url
    
    # Create Alembic config
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    try:
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        print("✓ Database migrations completed successfully")
        return True
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return False


def create_migration(message: str):
    """Create a new migration"""
    print(f"Creating migration: {message}")
    
    # Get database URL from config
    config = get_config()
    database_url = config.database_url
    
    # Create Alembic config
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    try:
        # Create migration
        command.revision(alembic_cfg, message=message, autogenerate=True)
        print(f"✓ Migration '{message}' created successfully")
        return True
    except Exception as e:
        print(f"✗ Migration creation failed: {e}")
        return False


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/run-migrations.py migrate")
        print("  python scripts/run-migrations.py create 'migration message'")
        sys.exit(1)
    
    command_arg = sys.argv[1]
    
    if command_arg == "migrate":
        success = run_migrations()
        sys.exit(0 if success else 1)
    elif command_arg == "create":
        if len(sys.argv) < 3:
            print("Error: Migration message required")
            sys.exit(1)
        message = sys.argv[2]
        success = create_migration(message)
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command_arg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
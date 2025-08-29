-- Initialize AI Pipeline Database
-- This script runs when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create initial database structure (basic tables will be created by migrations)
-- This is just to ensure the database is properly initialized

-- Database initialized successfully
# AI Story-to-Video Pipeline Makefile

.PHONY: help build up down logs clean test lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  build     - Build all Docker images"
	@echo "  up        - Start all services"
	@echo "  down      - Stop all services"
	@echo "  logs      - Show logs from all services"
	@echo "  clean     - Clean up containers and volumes"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linting"
	@echo "  format    - Format code"
	@echo "  health    - Check service health"

# Docker operations
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# Health checks
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8001/health || echo "Story service not responding"
	@curl -s http://localhost:8002/health || echo "TTS service not responding"
	@curl -s http://localhost:8003/health || echo "Image service not responding"
	@curl -s http://localhost:8004/health || echo "Music service not responding"
	@curl -s http://localhost:8005/health || echo "Video service not responding"
	@curl -s http://localhost:8006/health || echo "Moderation service not responding"
	@curl -s http://localhost:8007/health || echo "Distribution service not responding"
	@curl -s http://localhost:8008/health || echo "Analytics service not responding"
	@curl -s http://localhost:8010/health || echo "Orchestration service not responding"

# Development tools
test:
	@echo "Running all tests..."
	docker-compose run --rm story-service python -m pytest tests/ -v --tb=short

test-integration:
	@echo "Running integration tests..."
	docker-compose run --rm story-service python -m pytest tests/test_integration.py -v

test-models:
	@echo "Running model tests..."
	docker-compose run --rm story-service python -m pytest tests/test_models.py -v

test-database:
	@echo "Running database tests..."
	docker-compose run --rm story-service python -m pytest tests/test_database.py -v

test-schemas:
	@echo "Running schema tests..."
	docker-compose run --rm story-service python -m pytest tests/test_schemas.py -v

test-story-service:
	@echo "Running story service tests..."
	docker-compose run --rm story-service python -m pytest tests/test_story_service.py -v

test-music-service:
	@echo "Running music service tests..."
	docker-compose run --rm music-service python -m pytest tests/test_music_service.py -v

test-moderation-service:
	@echo "Running moderation service tests..."
	docker-compose run --rm moderation-service python -m pytest tests/test_moderation_service.py -v

test-video-service:
	@echo "Running video service tests..."
	docker-compose run --rm video-service python -m pytest tests/test_video_service.py -v

test-enum-roundtrip:
	@echo "Testing ENUM round-trip persistence..."
	docker-compose run --rm story-service python -m pytest tests/test_integration.py::TestDatabaseIntegration::test_enum_round_trip_persistence -v

test-computed-column:
	@echo "Testing computed column consistency..."
	docker-compose run --rm story-service python -m pytest tests/test_integration.py::TestDatabaseIntegration::test_computed_total_revenue_consistency -v

test-cascade:
	@echo "Testing CASCADE foreign key behavior..."
	docker-compose run --rm story-service python -m pytest tests/test_integration.py::TestDatabaseIntegration::test_foreign_key_cascade_behavior -v

test-migrations:
	@echo "Testing Alembic migrations..."
	docker-compose run --rm story-service python -m pytest tests/test_integration.py::TestAlembicMigrations -v

test-story-service-live:
	@echo "Testing live story service..."
	python3 scripts/test-story-service.py

test-analytics-service:
	@echo "Running analytics service tests against running service..."
	ANALYTICS_BASE_URL=http://localhost:8008 python3 -m pytest tests/test_analytics_service.py -v

test-orchestration-service:
	@echo "Running orchestration service tests against running service..."
	python3 -m pytest tests/test_orchestration_service.py -v

lint:
	@echo "Running linting..."
	# Add linting commands here

format:
	@echo "Formatting code..."
	# Add formatting commands here

# Database operations
db-reset:
	docker-compose stop postgres
	docker-compose rm -f postgres
	docker volume ls -q | grep postgres_data | xargs -r docker volume rm || true
	docker-compose up -d postgres

db-migrate:
	python3 scripts/run-migrations.py migrate

db-create-migration:
	@read -p "Enter migration message: " message; \
	python3 scripts/run-migrations.py create "$$message"

# Service-specific operations
story-logs:
	docker-compose logs -f story-service

tts-logs:
	docker-compose logs -f tts-service

image-logs:
	docker-compose logs -f image-service

music-logs:
	docker-compose logs -f music-service

video-logs:
	docker-compose logs -f video-service

moderation-logs:
	docker-compose logs -f moderation-service

# Enhanced database operations
db-migrate-docker:
	@echo "Running database migrations via Docker..."
	docker-compose exec story-service alembic upgrade head

db-migrate-fresh:
	@echo "Fresh database setup with migrations..."
	$(MAKE) db-reset
	@echo "Waiting for database to be ready..."
	@sleep 5
	$(MAKE) db-migrate-docker

db-downgrade:
	@echo "Downgrading database by one migration..."
	docker-compose exec story-service alembic downgrade -1

db-current:
	@echo "Current database version:"
	docker-compose exec story-service alembic current

db-history:
	@echo "Migration history:"
	docker-compose exec story-service alembic history

# CI/CD pipeline simulation
ci-test:
	@echo "Running CI test pipeline..."
	$(MAKE) build
	$(MAKE) up
	@echo "Waiting for services to be ready..."
	@sleep 10
	$(MAKE) db-migrate-docker
	$(MAKE) test-integration
	$(MAKE) test-migrations
	@echo "CI pipeline completed successfully!"

# Validation and smoke tests
validate-setup:
	@echo "Validating complete setup..."
	python3 scripts/validate-setup.py

smoke-test:
	@echo "Running smoke tests..."
	$(MAKE) health
	$(MAKE) test-enum-roundtrip
	$(MAKE) test-computed-column
	@echo "Smoke tests completed!"
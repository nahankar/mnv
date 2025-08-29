# AI Story-to-Video Pipeline

An end-to-end automated content generation system that transforms text-based stories into engaging video content and distributes them across multiple social media platforms.

## Architecture

The system is built using a microservices architecture with the following components:

- **story-service** (Port 8001): Generates narrative content using LLMs
- **tts-service** (Port 8002): Converts text to speech
- **image-service** (Port 8003): Creates visual content using AI
- **music-service** (Port 8004): Generates background music
- **video-service** (Port 8005): Assembles final videos
- **moderation-service** (Port 8006): Content filtering and compliance

### Story Service Details

The story service is the first component in the pipeline, generating narrative content using multiple LLM providers:

**Features:**
- Multiple LLM provider support (OpenAI GPT-4o, Anthropic Claude, Mistral)
- Automatic fallback when providers fail
- Retry logic with exponential backoff
- Rate limiting and cost tracking
- Platform-specific metadata generation

**API Endpoints:**
- `POST /generate/story` - Generate story content
- `POST /generate/metadata` - Generate platform-specific metadata
- `GET /providers` - List available LLM providers
- `GET /health` - Health check

**Required Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key  
- `MISTRAL_API_KEY` - Mistral API key

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Local Development Setup

1. Clone the repository and navigate to the project directory

2. Copy the environment file and configure your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

3. Start all services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Verify all services are running:
   ```bash
   docker-compose ps
   ```

5. Check service health:
   ```bash
   curl http://localhost:8001/health  # Story service
   curl http://localhost:8002/health  # TTS service
   curl http://localhost:8003/health  # Image service
   curl http://localhost:8004/health  # Music service
   curl http://localhost:8005/health  # Video service
   curl http://localhost:8006/health  # Moderation service
   ```

### Service Endpoints

- Story Service: http://localhost:8001
- TTS Service: http://localhost:8002
- Image Service: http://localhost:8003
- Music Service: http://localhost:8004
- Video Service: http://localhost:8005
- Moderation Service: http://localhost:8006

### Database Access

- PostgreSQL: localhost:5432
  - Database: ai_pipeline
  - Username: user
  - Password: password

- Redis: localhost:6379

## Development

### Project Structure

```
├── services/                 # Microservices
│   ├── story-service/       # Story generation service
│   ├── tts-service/         # Text-to-speech service
│   ├── image-service/       # Image generation service
│   ├── music-service/       # Music generation service
│   ├── video-service/       # Video assembly service
│   └── moderation-service/  # Content moderation service
├── shared/                  # Shared utilities
│   ├── config.py           # Configuration management
│   ├── database.py         # Database utilities
│   └── logging.py          # Logging utilities
├── scripts/                # Database and deployment scripts
└── docker-compose.yml     # Local development environment
```

### Shared Utilities

All services use shared utilities for:

- **Configuration Management**: Centralized settings with environment variable support
- **Database Connections**: Async SQLAlchemy with connection pooling
- **Structured Logging**: JSON logging with correlation IDs for request tracing

### Adding New Services

1. Create a new directory under `services/`
2. Add `main.py`, `requirements.txt`, and `Dockerfile`
3. Import shared utilities: `from shared.database import get_db_connection`
4. Add the service to `docker-compose.yml`

## Monitoring

Each service includes:

- Health check endpoints at `/health`
- Structured JSON logging with correlation IDs
- Database connection monitoring
- Service-specific metrics

## Database Schema

The system uses PostgreSQL with the following core tables:

- **stories**: Story content and generation metadata
- **media_assets**: Generated audio, images, and music files
- **videos**: Assembled videos in different formats
- **platform_uploads**: Upload tracking for social media platforms
- **analytics_data**: Performance metrics from platforms
- **model_configurations**: AI model settings and A/B test configs

### Database Operations

```bash
# Run database migrations
make db-migrate

# Create new migration
make db-create-migration

# Reset database (development only)
make db-reset
```

## Testing

```bash
# Run all tests
make test

# Run specific test suites
make test-models      # Test SQLAlchemy models
make test-database    # Test database operations
make test-schemas     # Test Pydantic schemas
```

## Next Steps

This foundation provides the basic infrastructure and data models. The next tasks will implement:

1. ✅ Core data models and database schema
2. AI service integrations (OpenAI, ElevenLabs, etc.)
3. Video assembly pipeline
4. Platform distribution
5. Analytics and monitoring

## Contributing

1. Follow the microservices architecture
2. Use shared utilities for common functionality
3. Include health checks and proper error handling
4. Add comprehensive logging with correlation IDs
5. Write tests for all new functionality
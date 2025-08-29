# Content Moderation Service

A FastAPI-based microservice for content filtering, compliance checking, and copyright detection with support for text, image, and audio analysis, manual review queues, and comprehensive audit logging.

## 🚀 Features

### Core Moderation
- **Text Moderation**: AI-powered text analysis using OpenAI's moderation API
- **Image Moderation**: Visual content analysis for inappropriate material
- **Audio Moderation**: Speech-to-text transcription followed by text moderation
- **Multi-Provider Support**: OpenAI and mock providers with fallback logic

### Compliance & Safety
- **GDPR Compliance**: Automatic detection of personal data in content
- **COPPA Compliance**: Child protection compliance checking
- **Platform Compliance**: Platform-specific policy validation (YouTube, Instagram, TikTok, Facebook)
- **Copyright Detection**: Basic copyright violation detection

### Review & Audit
- **Manual Review Queue**: Automated flagging and manual review workflow
- **Audit Logging**: Comprehensive audit trail for all moderation decisions
- **Review Management**: Approve/reject with override reasons and notes
- **Priority System**: High-priority flagging for severe violations

### Security & Performance
- **File Validation**: Type and size validation for uploaded files
- **Rate Limiting**: Built-in rate limiting and API protection
- **Correlation IDs**: Request tracing across the service
- **Health Monitoring**: Prometheus metrics and health checks

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  OpenAI Client  │    │   PostgreSQL    │
│                 │    │                 │    │                 │
│ - Text Mod      │◄──►│ - Moderation    │    │ - Content Mod   │
│ - File Mod      │    │ - Vision API    │    │ - Review Queue  │
│ - Compliance    │    │ - Whisper API   │    │ - Audit Logs    │
│ - Review Queue  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │  File Storage   │    │  Prometheus     │
│                 │    │                 │    │                 │
│ - Rate Limiting │    │ - Temp Files    │    │ - Metrics       │
│ - Session Data  │    │ - Uploads       │    │ - Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Database Schema

### Content Moderation Table
```sql
CREATE TABLE content_moderations (
    id UUID PRIMARY KEY,
    content_id VARCHAR(255) NOT NULL,
    content_type moderationtype NOT NULL,
    status moderationstatus NOT NULL,
    score FLOAT NOT NULL,
    flags JSON,
    categories JSON,
    recommendations JSON,
    requires_review BOOLEAN DEFAULT FALSE,
    audit_trail JSON,
    user_id VARCHAR(255),
    platform VARCHAR(50),
    moderation_level VARCHAR(20),
    reviewer_id VARCHAR(255),
    review_notes TEXT,
    review_override_reason TEXT,
    reviewed_at TIMESTAMP,
    provider VARCHAR(50),
    provider_response JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);
```

### Manual Review Queue Table
```sql
CREATE TABLE manual_review_queue (
    id UUID PRIMARY KEY,
    moderation_id UUID REFERENCES content_moderations(id),
    priority VARCHAR(20) DEFAULT 'normal',
    assigned_to VARCHAR(255),
    assigned_at TIMESTAMP,
    review_status VARCHAR(20) DEFAULT 'pending',
    review_deadline TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);
```

### Audit Logs Table
```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    action_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    user_role VARCHAR(50),
    action_data JSON NOT NULL,
    previous_state JSON,
    new_state JSON,
    ip_address VARCHAR(45),
    user_agent TEXT,
    correlation_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- OpenAI API key (optional, falls back to mock provider)

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ai_pipeline
REDIS_URL=redis://localhost:6379/0

# Service Configuration
SERVICE_NAME=moderation-service
LOG_LEVEL=INFO
LOG_FORMAT=json
ENVIRONMENT=development

# Moderation Configuration
OPENAI_API_KEY=your_openai_api_key_here
MODERATION_PROVIDER=openai  # openai or mock
MODERATION_THRESHOLD=0.7    # 0.0 to 1.0
MODERATION_STORAGE_PATH=/app/storage/moderation
```

### Docker Setup
```bash
# Build and run with docker-compose
docker-compose up moderation-service

# Or build individually
docker build -f services/moderation-service/Dockerfile -t moderation-service .
docker run -p 8006:8006 moderation-service
```

### Local Development
```bash
# Install dependencies
pip install -r services/moderation-service/requirements.txt

# Run migrations
python scripts/run-moderation-migration.py

# Start service
cd services/moderation-service
uvicorn main:app --host 0.0.0.0 --port 8006 --reload
```

## 📚 API Reference

### Health & Status
```http
GET /health
GET /health?deep=true
GET /metrics
GET /config
```

### Text Moderation
```http
POST /moderate/text
Content-Type: application/x-www-form-urlencoded

text=This is the content to moderate&level=medium&user_id=user123&platform=youtube
```

### File Moderation
```http
POST /moderate/file
Content-Type: multipart/form-data

file: [binary file data]
content_type: image|audio|video
level: low|medium|high|critical
user_id: user123
platform: youtube|instagram|tiktok|facebook
```

### Review Queue Management
```http
GET /review/queue?status=flagged&limit=50&offset=0
POST /review/{moderation_id}
Content-Type: application/json

{
  "moderation_id": "uuid",
  "reviewer_id": "reviewer123",
  "decision": "approved|rejected|flagged",
  "notes": "Review notes",
  "override_reason": "False positive"
}
```

### Compliance Checking
```http
GET /compliance/check?text=Content to check&platform=youtube
```

### Provider Information
```http
GET /providers
```

## 🔧 Configuration

### Moderation Thresholds
- **Low (0.3)**: Minimal filtering, high false negative rate
- **Medium (0.7)**: Balanced filtering (default)
- **High (0.8)**: Strict filtering, high false positive rate
- **Critical (0.9)**: Maximum filtering for sensitive content

### File Upload Limits
- **Maximum file size**: 10MB
- **Supported image formats**: JPEG, PNG, GIF, WebP
- **Supported audio formats**: MP3, WAV, OGG
- **Supported video formats**: MP4, AVI, MOV, WebM

### Rate Limiting
- **Default rate limit**: 100 requests per minute per IP
- **Burst limit**: 20 requests per 10 seconds
- **Rate limit headers**: `X-RateLimit-*`

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/test_moderation_service.py -v

# Run specific test categories
pytest tests/test_moderation_service.py::TestModerationService -v
pytest tests/test_moderation_service.py::TestModerationProviders -v
pytest tests/test_moderation_service.py::TestComplianceChecker -v
```

### Test Coverage
```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest tests/test_moderation_service.py --cov=services.moderation_service --cov-report=html
```

### Manual Testing
```bash
# Test text moderation
curl -X POST "http://localhost:8006/moderate/text" \
  -d "text=This is a test message" \
  -d "level=medium"

# Test file moderation
curl -X POST "http://localhost:8006/moderate/file" \
  -F "file=@test.jpg" \
  -F "content_type=image" \
  -F "level=medium"

# Check review queue
curl "http://localhost:8006/review/queue"
```

## 📊 Monitoring & Metrics

### Prometheus Metrics
- `moderation_service_requests_total`: Total API requests
- `moderation_service_request_duration_seconds`: Request duration
- `moderation_service_api_calls_total`: External API calls

### Health Checks
- Database connectivity
- Redis connectivity
- Service responsiveness
- Provider availability

### Logging
- Structured JSON logging
- Correlation IDs for request tracing
- Audit trail for all actions
- Error tracking and alerting

## 🔒 Security Features

### Input Validation
- File type validation using magic numbers
- File size limits
- Content type verification
- Path sanitization

### Authentication & Authorization
- JWT token validation (when integrated)
- Role-based access control
- API key management

### Data Protection
- GDPR compliance checking
- Personal data detection
- Secure file handling
- Audit logging

## 🚨 Error Handling

### Common Error Codes
- `400 Bad Request`: Invalid input or file type
- `413 Payload Too Large`: File exceeds size limit
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Service errors
- `503 Service Unavailable`: Provider unavailable

### Retry Logic
- Exponential backoff for external API calls
- Circuit breaker pattern for provider failures
- Graceful degradation to mock provider

## 🔄 Integration

### Pipeline Integration
The moderation service integrates with the AI Story-to-Video Pipeline:

1. **Story Generation**: Text moderation before story creation
2. **Media Generation**: Image/audio moderation after generation
3. **Video Assembly**: Final content moderation before distribution
4. **Distribution**: Platform-specific compliance checking

### Conditional Flow
```python
# Example integration with video assembly
if moderation_result.status == ModerationStatus.APPROVED:
    proceed_with_video_assembly()
elif moderation_result.status == ModerationStatus.FLAGGED:
    add_to_review_queue()
    block_video_assembly()
else:
    regenerate_content()
```

## 📈 Performance

### Benchmarks
- **Text moderation**: ~200ms average response time
- **Image moderation**: ~2-5s average response time
- **Audio moderation**: ~10-30s average response time
- **Concurrent requests**: 100+ requests per second

### Optimization
- Async/await for I/O operations
- Connection pooling for database
- Redis caching for frequent requests
- Batch processing for multiple files

## 🐛 Troubleshooting

### Common Issues

**Service won't start**
```bash
# Check database connection
python -c "from shared.database import DatabaseManager; import asyncio; asyncio.run(DatabaseManager().initialize())"

# Check environment variables
echo $DATABASE_URL
echo $OPENAI_API_KEY
```

**Migration errors**
```bash
# Run migration manually
python scripts/run-moderation-migration.py

# Check migration status
alembic current
alembic history
```

**File upload issues**
```bash
# Check storage permissions
ls -la /app/storage/moderation

# Check file size limits
du -sh /app/storage/moderation/*
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug output
uvicorn main:app --log-level debug --reload
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Include error handling

### Testing Guidelines
- Unit tests for all new functions
- Integration tests for API endpoints
- Mock external dependencies
- Test error scenarios

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation
- Contact the development team

---

**Last Updated**: January 2024
**Version**: 1.0.0
**Status**: Production Ready ✅

"""
Tests for the Content Moderation Service

Tests the moderation service functionality including:
- Text moderation
- File moderation
- Review queue management
- Compliance checking
- Audit logging
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from services.moderation_service.main import app, get_moderation_service
from shared.models import ContentModeration, ModerationStatus, ModerationType, ManualReviewQueue, AuditLog
from shared.database import DatabaseManager


@pytest.fixture
def client():
    """Test client for the moderation service"""
    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.add = AsyncMock()
    return session


@pytest.fixture
def mock_moderation_service():
    """Mock moderation service"""
    service = AsyncMock()
    service.moderate_text.return_value = {
        "flagged": False,
        "score": 0.1,
        "categories": {"violence": False, "hate": False},
        "category_scores": {"violence": 0.1, "hate": 0.1},
        "flagged_categories": [],
        "provider": "mock"
    }
    service.moderate_image.return_value = {
        "flagged": False,
        "score": 0.0,
        "categories": [],
        "confidence": 0.9,
        "provider": "mock"
    }
    service.moderate_audio.return_value = {
        "flagged": False,
        "score": 0.0,
        "categories": [],
        "confidence": 0.9,
        "provider": "mock"
    }
    return service


class TestModerationService:
    """Test cases for moderation service functionality"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "moderation-service"

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Content Moderation Service API"
        assert data["version"] == "1.0.0"

    def test_get_providers(self, client):
        """Test providers endpoint"""
        response = client.get("/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) >= 1

    def test_get_config(self, client):
        """Test configuration endpoint"""
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "moderation_threshold" in data
        assert "moderation_provider" in data
        assert "storage_path" in data

    @patch('services.moderation_service.main.get_moderation_service')
    def test_moderate_text_success(self, mock_get_service, client, mock_moderation_service):
        """Test successful text moderation"""
        mock_get_service.return_value = mock_moderation_service
        
        response = client.post(
            "/moderate/text",
            params={
                "text": "This is a test message",
                "level": "medium",
                "user_id": "test_user",
                "platform": "youtube"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["score"] == 0.1
        assert not data["requires_review"]

    @patch('services.moderation_service.main.get_moderation_service')
    def test_moderate_text_flagged(self, mock_get_service, client):
        """Test text moderation with flagged content"""
        service = AsyncMock()
        service.moderate_text.return_value = {
            "flagged": True,
            "score": 0.8,
            "categories": {"violence": True, "hate": False},
            "category_scores": {"violence": 0.8, "hate": 0.1},
            "flagged_categories": ["violence"],
            "provider": "mock"
        }
        mock_get_service.return_value = service
        
        response = client.post(
            "/moderate/text",
            params={
                "text": "This contains violence",
                "level": "medium"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "flagged"
        assert data["score"] == 0.8
        assert data["requires_review"]
        assert "violence" in data["flags"]

    def test_moderate_file_invalid_type(self, client):
        """Test file moderation with invalid file type"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_file = f.name
        
        try:
            with open(temp_file, "rb") as file:
                response = client.post(
                    "/moderate/file",
                    files={"file": ("test.txt", file, "text/plain")},
                    params={
                        "content_type": "image",
                        "level": "medium"
                    }
                )
            
            assert response.status_code == 400
            assert "Invalid file type" in response.json()["detail"]
        finally:
            os.unlink(temp_file)

    def test_moderate_file_too_large(self, client):
        """Test file moderation with file too large"""
        # Create a large temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"x" * (11 * 1024 * 1024))  # 11MB
            temp_file = f.name
        
        try:
            with open(temp_file, "rb") as file:
                response = client.post(
                    "/moderate/file",
                    files={"file": ("large.jpg", file, "image/jpeg")},
                    params={
                        "content_type": "image",
                        "level": "medium"
                    }
                )
            
            assert response.status_code == 400
            assert "File too large" in response.json()["detail"]
        finally:
            os.unlink(temp_file)

    @patch('services.moderation_service.main.get_moderation_service')
    def test_moderate_file_success(self, mock_get_service, client, mock_moderation_service):
        """Test successful file moderation"""
        mock_get_service.return_value = mock_moderation_service
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            temp_file = f.name
        
        try:
            with open(temp_file, "rb") as file:
                response = client.post(
                    "/moderate/file",
                    files={"file": ("test.jpg", file, "image/jpeg")},
                    params={
                        "content_type": "image",
                        "level": "medium",
                        "user_id": "test_user",
                        "platform": "instagram"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "approved"
            assert data["score"] == 0.0
            assert not data["requires_review"]
        finally:
            os.unlink(temp_file)

    def test_compliance_check(self, client):
        """Test compliance checking"""
        response = client.get(
            "/compliance/check",
            params={
                "text": "This is a test message",
                "platform": "youtube"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "gdpr_compliant" in data
        assert "coppa_compliant" in data
        assert "platform_compliant" in data
        assert "violations" in data
        assert "recommendations" in data

    def test_compliance_check_with_violations(self, client):
        """Test compliance checking with violations"""
        response = client.get(
            "/compliance/check",
            params={
                "text": "This contains personal data like email and phone numbers",
                "platform": "youtube"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["violations"]) > 0

    @patch('services.moderation_service.main.get_db_manager')
    def test_get_review_queue(self, mock_get_db_manager, client, mock_db_session):
        """Test getting review queue"""
        # Mock database session and query results
        mock_db_manager.return_value.get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock query results
        mock_moderation = MagicMock()
        mock_moderation.id = "test-id"
        mock_moderation.content_id = "content-123"
        mock_moderation.content_type = ModerationType.TEXT
        mock_moderation.status = ModerationStatus.FLAGGED
        mock_moderation.score = 0.8
        mock_moderation.flags = ["violence"]
        mock_moderation.created_at = "2024-01-15T10:00:00Z"
        mock_moderation.user_id = "test_user"
        mock_moderation.platform = "youtube"
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value = MagicMock()
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = [mock_moderation]
        
        response = client.get("/review/queue")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data

    @patch('services.moderation_service.main.get_db_manager')
    def test_review_moderation(self, mock_get_db_manager, client, mock_db_session):
        """Test submitting manual review"""
        mock_db_manager.return_value.get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock existing moderation record
        mock_moderation = MagicMock()
        mock_moderation.id = "test-id"
        mock_moderation.content_id = "content-123"
        mock_moderation.status = ModerationStatus.FLAGGED
        mock_moderation.reviewer_id = None
        mock_moderation.review_notes = None
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_moderation
        
        response = client.post(
            "/review/test-id",
            json={
                "moderation_id": "test-id",
                "reviewer_id": "reviewer-123",
                "decision": "approved",
                "notes": "Content is acceptable",
                "override_reason": "False positive"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["moderation_id"] == "test-id"
        assert data["status"] == "approved"
        assert data["reviewer_id"] == "reviewer-123"


class TestModerationProviders:
    """Test cases for moderation providers"""

    def test_mock_provider_text_moderation(self):
        """Test mock provider text moderation"""
        from services.moderation_service.main import MockModerationProvider
        
        provider = MockModerationProvider()
        
        # Test clean text
        result = asyncio.run(provider.moderate_text("This is a clean message"))
        assert not result["flagged"]
        assert result["score"] == 0.0
        
        # Test flagged text
        result = asyncio.run(provider.moderate_text("This contains violence"))
        assert result["flagged"]
        assert result["score"] > 0.0
        assert "violence" in result["categories"]

    def test_mock_provider_image_moderation(self):
        """Test mock provider image moderation"""
        from services.moderation_service.main import MockModerationProvider
        
        provider = MockModerationProvider()
        
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            temp_file = f.name
        
        try:
            result = asyncio.run(provider.moderate_image(temp_file))
            assert not result["flagged"]
            assert result["score"] == 0.0
            assert result["confidence"] == 0.9
        finally:
            os.unlink(temp_file)

    @patch('openai.AsyncOpenAI')
    def test_openai_provider_text_moderation(self, mock_openai_client):
        """Test OpenAI provider text moderation"""
        from services.moderation_service.main import OpenAIModerationProvider
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_result.flagged = False
        mock_result.categories.violence = False
        mock_result.categories.hate = False
        mock_result.category_scores.violence = 0.1
        mock_result.category_scores.hate = 0.1
        mock_result.categories.dict.return_value = {"violence": False, "hate": False}
        mock_result.category_scores.dict.return_value = {"violence": 0.1, "hate": 0.1}
        mock_response.results = [mock_result]
        
        mock_client = AsyncMock()
        mock_client.moderations.create.return_value = mock_response
        mock_openai_client.return_value = mock_client
        
        provider = OpenAIModerationProvider("test-key")
        
        result = asyncio.run(provider.moderate_text("This is a test message"))
        assert not result["flagged"]
        assert result["score"] == 0.1
        assert result["provider"] == "openai"


class TestComplianceChecker:
    """Test cases for compliance checking"""

    def test_gdpr_compliance(self):
        """Test GDPR compliance checking"""
        from services.moderation_service.main import ComplianceChecker
        
        checker = ComplianceChecker()
        
        # Test compliant text
        result = asyncio.run(checker.check_gdpr_compliance("This is a normal message"))
        assert result["compliant"]
        assert len(result["violations"]) == 0
        
        # Test non-compliant text
        result = asyncio.run(checker.check_gdpr_compliance("Contact me at email@example.com"))
        assert not result["compliant"]
        assert len(result["violations"]) > 0

    def test_coppa_compliance(self):
        """Test COPPA compliance checking"""
        from services.moderation_service.main import ComplianceChecker
        
        checker = ComplianceChecker()
        
        # Test compliant text
        result = asyncio.run(checker.check_coppa_compliance("This is a normal message"))
        assert result["compliant"]
        assert len(result["violations"]) == 0
        
        # Test non-compliant text
        result = asyncio.run(checker.check_coppa_compliance("This content is for kids"))
        assert not result["compliant"]
        assert len(result["violations"]) > 0

    def test_platform_compliance(self):
        """Test platform compliance checking"""
        from services.moderation_service.main import ComplianceChecker
        
        checker = ComplianceChecker()
        
        # Test compliant text
        result = asyncio.run(checker.check_platform_compliance("This is a normal message", "youtube"))
        assert result["compliant"]
        assert len(result["violations"]) == 0
        
        # Test non-compliant text
        result = asyncio.run(checker.check_platform_compliance("This violates community guidelines", "youtube"))
        assert not result["compliant"]
        assert len(result["violations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])

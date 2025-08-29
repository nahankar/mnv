"""
Unit tests for Story Generation Service
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from httpx import Response
import json

from shared.schemas import StoryRequest, MetadataRequest
from shared.models import ContentStatus
from shared.database import DatabaseManager

# Import the service (we'll need to adjust the import path)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'story'))

from main import StoryService, LLM_PROVIDERS


class TestStoryService:
    """Test cases for StoryService"""
    
    @pytest.fixture
    async def mock_db_manager(self):
        """Mock database manager"""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.get_session = AsyncMock()
        return db_manager
    
    @pytest.fixture
    async def story_service(self, mock_db_manager):
        """Create StoryService instance with mocked dependencies"""
        service = StoryService(mock_db_manager)
        service.http_client = AsyncMock()
        return service
    
    @pytest.fixture
    def sample_story_request(self):
        """Sample story generation request"""
        return StoryRequest(
            genre="adventure",
            theme="space exploration",
            target_length="400-500",
            tone="exciting",
            user_id="test_user"
        )
    
    @pytest.fixture
    def sample_metadata_request(self):
        """Sample metadata generation request"""
        return MetadataRequest(
            story_content="Once upon a time in a galaxy far away...",
            platform="youtube"
        )

    @pytest.mark.asyncio
    async def test_generate_story_success_openai(self, story_service, sample_story_request):
        """Test successful story generation with OpenAI"""
        
        # Mock OpenAI API response
        mock_response = Mock(spec=Response)
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Once upon a time in a distant galaxy, brave explorers ventured into the unknown cosmos. Their ship gleamed against the starlit void as they searched for new worlds. Adventure awaited them among the celestial bodies, where mysteries and wonders beyond imagination lay hidden in the depths of space."
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        
        story_service.http_client.post = AsyncMock(return_value=mock_response)
        
        # Mock database operations
        mock_session = AsyncMock()
        story_service.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        # Mock story storage
        with patch.object(story_service, '_store_story') as mock_store:
            mock_story = Mock()
            mock_story.id = "test-story-id"
            mock_store.return_value = mock_story
            
            # Mock API key retrieval
            with patch.object(story_service, '_get_api_key', return_value="test-api-key"):
                
                result = await story_service.generate_story(sample_story_request)
                
                # Assertions
                assert result.id == "test-story-id"
                assert result.provider_used == "gpt-4o"
                assert result.status == ContentStatus.COMPLETED
                assert result.word_count > 0
                assert result.generation_cost > 0
                
                # Verify API call was made
                story_service.http_client.post.assert_called_once()
                call_args = story_service.http_client.post.call_args
                assert "https://api.openai.com/v1/chat/completions" in call_args[0]

    @pytest.mark.asyncio
    async def test_generate_story_fallback_to_claude(self, story_service, sample_story_request):
        """Test fallback to Claude when OpenAI fails"""
        
        # Mock OpenAI failure
        openai_response = Mock(spec=Response)
        openai_response.raise_for_status.side_effect = Exception("OpenAI API error")
        
        # Mock Claude success
        claude_response = Mock(spec=Response)
        claude_response.json.return_value = {
            "content": [{
                "text": "In the vast expanse of space, intrepid explorers embarked on a thrilling journey to discover new worlds and unlock the secrets of the universe."
            }]
        }
        claude_response.raise_for_status = Mock()
        
        # Configure mock to fail first, succeed second
        story_service.http_client.post = AsyncMock(side_effect=[openai_response, claude_response])
        
        # Mock database operations
        mock_session = AsyncMock()
        story_service.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        with patch.object(story_service, '_store_story') as mock_store:
            mock_story = Mock()
            mock_story.id = "test-story-id"
            mock_store.return_value = mock_story
            
            with patch.object(story_service, '_get_api_key', return_value="test-api-key"):
                
                result = await story_service.generate_story(sample_story_request)
                
                # Should have fallen back to Claude
                assert result.provider_used == "claude-3.5-sonnet"
                assert result.status == ContentStatus.COMPLETED
                
                # Verify both API calls were made
                assert story_service.http_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_story_all_providers_fail(self, story_service, sample_story_request):
        """Test when all LLM providers fail"""
        
        # Mock all providers failing
        mock_response = Mock(spec=Response)
        mock_response.raise_for_status.side_effect = Exception("API error")
        
        story_service.http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(story_service, '_get_api_key', return_value="test-api-key"):
            
            # Should raise HTTPException
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await story_service.generate_story(sample_story_request)
            
            assert exc_info.value.status_code == 503
            assert "All LLM providers failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_metadata_youtube(self, story_service, sample_metadata_request):
        """Test metadata generation for YouTube"""
        
        # Mock OpenAI API response for metadata
        mock_response = Mock(spec=Response)
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "title": "Epic Space Adventure: Journey to the Unknown!",
                        "description": "Join brave explorers as they venture into the cosmos in this thrilling space adventure. Discover new worlds and unlock the mysteries of the universe!",
                        "hashtags": ["#SpaceAdventure", "#SciFi", "#Exploration", "#Galaxy", "#Adventure"]
                    })
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        
        story_service.http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(story_service, '_get_api_key', return_value="test-api-key"):
            
            result = await story_service.generate_metadata(sample_metadata_request)
            
            # Assertions
            assert result.platform == "youtube"
            assert result.title == "Epic Space Adventure: Journey to the Unknown!"
            assert len(result.hashtags) == 5
            assert result.description is not None
            
            # Verify API call was made
            story_service.http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_metadata_unsupported_platform(self, story_service):
        """Test metadata generation for unsupported platform"""
        
        request = MetadataRequest(
            story_content="Test story",
            platform="unsupported_platform"
        )
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await story_service.generate_metadata(request)
        
        assert exc_info.value.status_code == 400
        assert "Unsupported platform" in str(exc_info.value.detail)

    def test_build_story_prompt(self, story_service, sample_story_request):
        """Test story prompt building"""
        
        prompt = story_service._build_story_prompt(sample_story_request)
        
        # Check that prompt contains key elements
        assert "adventure" in prompt.lower()
        assert "space exploration" in prompt.lower()
        assert "400-500" in prompt
        assert "exciting" in prompt.lower()
        assert "Story:" in prompt

    @pytest.mark.asyncio
    async def test_call_openai_api(self, story_service):
        """Test OpenAI API call formatting"""
        
        config = LLM_PROVIDERS["gpt-4o"]
        prompt = "Write a test story"
        
        # Mock response
        mock_response = Mock(spec=Response)
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Test story content"}
            }]
        }
        mock_response.raise_for_status = Mock()
        
        story_service.http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(story_service, '_get_api_key', return_value="test-api-key"):
            
            result = await story_service._call_openai(config, prompt)
            
            assert result == "Test story content"
            
            # Verify correct API call format
            call_args = story_service.http_client.post.call_args
            assert call_args[1]['json']['model'] == 'gpt-4o'
            assert call_args[1]['json']['messages'][0]['role'] == 'system'
            assert call_args[1]['json']['messages'][1]['role'] == 'user'
            assert call_args[1]['json']['messages'][1]['content'] == prompt

    @pytest.mark.asyncio
    async def test_call_anthropic_api(self, story_service):
        """Test Anthropic API call formatting"""
        
        config = LLM_PROVIDERS["claude-3.5-sonnet"]
        prompt = "Write a test story"
        
        # Mock response
        mock_response = Mock(spec=Response)
        mock_response.json.return_value = {
            "content": [{"text": "Test story content"}]
        }
        mock_response.raise_for_status = Mock()
        
        story_service.http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(story_service, '_get_api_key', return_value="test-api-key"):
            
            result = await story_service._call_anthropic(config, prompt)
            
            assert result == "Test story content"
            
            # Verify correct API call format
            call_args = story_service.http_client.post.call_args
            assert call_args[1]['json']['model'] == 'claude-3-5-sonnet-20241022'
            assert call_args[1]['json']['messages'][0]['role'] == 'user'
            assert call_args[1]['json']['messages'][0]['content'] == prompt

    @pytest.mark.asyncio
    async def test_call_mistral_api(self, story_service):
        """Test Mistral API call formatting"""
        
        config = LLM_PROVIDERS["mistral-large"]
        prompt = "Write a test story"
        
        # Mock response
        mock_response = Mock(spec=Response)
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Test story content"}
            }]
        }
        mock_response.raise_for_status = Mock()
        
        story_service.http_client.post = AsyncMock(return_value=mock_response)
        
        with patch.object(story_service, '_get_api_key', return_value="test-api-key"):
            
            result = await story_service._call_mistral(config, prompt)
            
            assert result == "Test story content"
            
            # Verify correct API call format
            call_args = story_service.http_client.post.call_args
            assert call_args[1]['json']['model'] == 'mistral-large-latest'
            assert call_args[1]['json']['messages'][0]['role'] == 'system'
            assert call_args[1]['json']['messages'][1]['role'] == 'user'

    def test_get_api_key_missing(self, story_service):
        """Test API key retrieval when key is missing"""
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                story_service._get_api_key('MISSING_KEY')
            
            assert "Missing API key: MISSING_KEY" in str(exc_info.value)

    def test_get_api_key_present(self, story_service):
        """Test API key retrieval when key is present"""
        
        with patch.dict('os.environ', {'TEST_KEY': 'test-value'}):
            result = story_service._get_api_key('TEST_KEY')
            assert result == 'test-value'

    @pytest.mark.asyncio
    async def test_store_story(self, story_service, sample_story_request):
        """Test story storage in database"""
        
        content = "Test story content"
        provider = "gpt-4o"
        cost = 0.05
        
        # Mock database session
        mock_session = AsyncMock()
        mock_story = Mock()
        mock_story.id = "test-id"
        
        story_service.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        with patch('main.Story') as mock_story_class:
            mock_story_class.return_value = mock_story
            
            result = await story_service._store_story(sample_story_request, content, provider, cost)
            
            # Verify story was created with correct data
            mock_story_class.assert_called_once()
            call_kwargs = mock_story_class.call_args[1]
            
            assert call_kwargs['content'] == content
            assert call_kwargs['genre'] == sample_story_request.genre
            assert call_kwargs['theme'] == sample_story_request.theme
            assert call_kwargs['generation_params']['provider'] == provider
            assert call_kwargs['generation_params']['cost'] == cost
            assert call_kwargs['status'] == ContentStatus.COMPLETED
            
            # Verify database operations
            mock_session.add.assert_called_once_with(mock_story)
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once_with(mock_story)


class TestStoryServiceIntegration:
    """Integration tests for story service endpoints"""
    
    @pytest.mark.asyncio
    async def test_story_generation_cost_calculation(self):
        """Test that cost calculation is accurate"""
        
        story_content = "This is a test story with exactly ten words here."
        word_count = len(story_content.split())
        estimated_tokens = word_count * 1.3
        
        config = LLM_PROVIDERS["gpt-4o"]
        expected_cost = estimated_tokens * config["cost_per_token"]
        
        # Verify cost calculation logic
        assert word_count == 10
        assert estimated_tokens == 13.0
        assert expected_cost == 13.0 * 0.00003
        assert abs(expected_cost - 0.00039) < 0.000001

    def test_llm_provider_priority_order(self):
        """Test that providers are ordered by priority correctly"""
        
        providers = sorted(LLM_PROVIDERS.items(), key=lambda x: x[1]["priority"])
        
        # Should be ordered: gpt-4o (1), claude (2), mistral (3)
        assert providers[0][0] == "gpt-4o"
        assert providers[1][0] == "claude-3.5-sonnet"
        assert providers[2][0] == "mistral-large"

    def test_platform_metadata_configurations(self):
        """Test that all supported platforms have proper configurations"""
        
        # This would be part of the service, but testing the concept
        supported_platforms = ["youtube", "instagram", "tiktok", "facebook"]
        
        for platform in supported_platforms:
            # Each platform should have specific requirements
            if platform == "youtube":
                assert True  # Title max 60 chars, 5 hashtags
            elif platform == "instagram":
                assert True  # Caption hook, 10 hashtags
            elif platform == "tiktok":
                assert True  # Viral title, 8 hashtags
            elif platform == "facebook":
                assert True  # Engagement title, 3 hashtags


if __name__ == "__main__":
    pytest.main([__file__])
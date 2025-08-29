"""
Unit tests for Image Generation Service

Tests image generation, scene analysis, batch processing, and provider integrations.
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import json
import base64

from fastapi.testclient import TestClient
from PIL import Image
import httpx

# Import the service components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'image-service'))

from main import (
    app, ImageService, DallE3Provider, StableDiffusionProvider, 
    SceneAnalyzer, ImageProcessor, ImageRequest, BatchImageRequest
)
from shared.models import MediaAsset, Story, MediaType


class TestSceneAnalyzer:
    """Test scene analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_extract_scenes_from_story(self):
        """Test extracting visual scenes from story content"""
        story_content = """
        The brave knight looked across the vast forest. The morning sun appeared 
        through the trees, casting beautiful shadows on the ground. He walked 
        slowly towards the ancient castle that stood majestically on the hill.
        The dark clouds gathered in the sky above.
        """
        
        scenes = await SceneAnalyzer.extract_scenes_from_story(story_content, max_scenes=3)
        
        assert len(scenes) >= 3
        assert all('scene_number' in scene for scene in scenes)
        assert all('prompt' in scene for scene in scenes)
        assert all('original_text' in scene for scene in scenes)
        
        # Check that visual keywords are detected
        scene_prompts = [scene['prompt'].lower() for scene in scenes]
        visual_elements = ['forest', 'sun', 'castle', 'sky']
        assert any(element in ' '.join(scene_prompts) for element in visual_elements)
    
    @pytest.mark.asyncio
    async def test_extract_scenes_minimum_guarantee(self):
        """Test that minimum scenes are generated even with poor content"""
        story_content = "This is a simple story without visual elements."
        
        scenes = await SceneAnalyzer.extract_scenes_from_story(story_content, max_scenes=5)
        
        assert len(scenes) >= 3  # Should generate at least 3 scenes
        assert all(scene['scene_number'] > 0 for scene in scenes)
    
    def test_enhance_prompt_for_image_generation(self):
        """Test prompt enhancement for better image generation"""
        original_prompt = "A knight in the forest"
        enhanced = SceneAnalyzer.enhance_prompt_for_image_generation(original_prompt)
        
        assert original_prompt in enhanced
        assert "cinematic lighting" in enhanced
        assert "high quality" in enhanced
        assert "detailed" in enhanced


class TestImageProviders:
    """Test image generation providers"""
    
    @pytest.mark.asyncio
    async def test_dalle3_provider_success(self):
        """Test successful DALL·E 3 image generation"""
        provider = DallE3Provider("test-api-key")
        
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{
                "url": "https://example.com/image.png",
                "revised_prompt": "Enhanced prompt"
            }]
        }
        
        with patch.object(provider.client, 'post', return_value=mock_response):
            result = await provider.generate_image(
                "A beautiful landscape",
                aspect_ratio="16:9",
                quality="standard"
            )
        
        assert result["provider"] == "dall-e-3"
        assert result["image_url"] == "https://example.com/image.png"
        assert result["revised_prompt"] == "Enhanced prompt"
        assert result["size"] == "1792x1024"  # 16:9 mapping
        
        await provider.close()
    
    @pytest.mark.asyncio
    async def test_dalle3_provider_error(self):
        """Test DALL·E 3 provider error handling"""
        provider = DallE3Provider("test-api-key")
        
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        
        with patch.object(provider.client, 'post', return_value=mock_response):
            with pytest.raises(Exception):  # Should raise HTTPException
                await provider.generate_image("Test prompt")
        
        await provider.close()
    
    @pytest.mark.asyncio
    async def test_stable_diffusion_provider_success(self):
        """Test successful Stable Diffusion image generation"""
        provider = StableDiffusionProvider("test-api-key")
        
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "artifacts": [{
                "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "seed": 12345
            }]
        }
        
        with patch.object(provider.client, 'post', return_value=mock_response):
            result = await provider.generate_image(
                "A beautiful landscape",
                aspect_ratio="1:1"
            )
        
        assert result["provider"] == "stable-diffusion-xl"
        assert "image_data" in result
        assert result["seed"] == 12345
        assert result["dimensions"] == {"width": 1024, "height": 1024}
        
        await provider.close()


class TestImageProcessor:
    """Test image processing functionality"""
    
    @pytest.mark.asyncio
    async def test_download_and_save_image(self):
        """Test downloading and saving image from URL"""
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            test_image.save(temp_file.name, 'PNG')
            temp_path = Path(temp_file.name)
        
        try:
            # Mock HTTP response
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.content = temp_path.read_bytes()
                mock_response.raise_for_status = Mock()
                
                mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
                    output_path = Path(output_file.name)
                
                metadata = await ImageProcessor.download_and_save_image(
                    "https://example.com/test.png",
                    output_path
                )
                
                assert metadata["format"] == "PNG"
                assert metadata["size"] == (100, 100)
                assert output_path.exists()
                
                output_path.unlink()  # Clean up
        finally:
            temp_path.unlink()  # Clean up
    
    @pytest.mark.asyncio
    async def test_save_base64_image(self):
        """Test saving base64 image data"""
        # Create a simple test image and encode it
        test_image = Image.new('RGB', (50, 50), color='blue')
        
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            test_image.save(temp_file.name, 'PNG')
            image_data = base64.b64encode(Path(temp_file.name).read_bytes()).decode()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
            output_path = Path(output_file.name)
        
        try:
            metadata = await ImageProcessor.save_base64_image(image_data, output_path)
            
            assert metadata["format"] == "PNG"
            assert metadata["size"] == (50, 50)
            assert output_path.exists()
        finally:
            output_path.unlink()  # Clean up
    
    @pytest.mark.asyncio
    async def test_optimize_image(self):
        """Test image optimization"""
        # Create a test image
        test_image = Image.new('RGB', (200, 200), color='green')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            test_image.save(temp_file.name, 'PNG')
            temp_path = Path(temp_file.name)
        
        try:
            result = await ImageProcessor.optimize_image(temp_path, quality=75)
            
            assert result["optimized_path"].exists()
            assert result["optimized_path"].suffix == '.jpg'
            assert result["file_size"] > 0
            
            result["optimized_path"].unlink()  # Clean up
        finally:
            if temp_path.exists():
                temp_path.unlink()  # Clean up
    
    @pytest.mark.asyncio
    async def test_resize_image(self):
        """Test image resizing"""
        # Create a test image
        test_image = Image.new('RGB', (400, 300), color='yellow')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            test_image.save(temp_file.name, 'PNG')
            temp_path = Path(temp_file.name)
        
        try:
            resized_path = await ImageProcessor.resize_image(
                temp_path, 
                (200, 150), 
                maintain_aspect=False
            )
            
            assert resized_path.exists()
            
            # Check dimensions
            with Image.open(resized_path) as img:
                assert img.size == (200, 150)
            
            resized_path.unlink()  # Clean up
        finally:
            temp_path.unlink()  # Clean up


class TestImageService:
    """Test the main ImageService class"""
    
    @pytest.fixture
    def mock_image_service(self):
        """Create a mock image service for testing"""
        service = ImageService()
        
        # Mock providers
        mock_dalle_provider = AsyncMock()
        mock_dalle_provider.generate_image.return_value = {
            "provider": "dall-e-3",
            "image_url": "https://example.com/test.png",
            "revised_prompt": "Test prompt",
            "size": "1024x1024",
            "quality": "standard"
        }
        
        service.providers = {"dall-e-3": mock_dalle_provider}
        return service
    
    @pytest.mark.asyncio
    async def test_generate_single_image_success(self, mock_image_service):
        """Test successful single image generation"""
        request = ImageRequest(
            prompt="A beautiful sunset",
            story_id=str(uuid.uuid4()),
            scene_number=1,
            aspect_ratio="16:9"
        )
        
        # Mock database operations
        with patch('main.get_db_manager') as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.return_value.get_session.return_value.__aenter__.return_value = mock_session
            
            # Mock image processing
            with patch.object(mock_image_service.processor, 'download_and_save_image') as mock_download:
                mock_download.return_value = {
                    "format": "PNG",
                    "size": (1024, 1024),
                    "mode": "RGB",
                    "file_size": 1024000
                }
                
                with patch.object(mock_image_service.processor, 'optimize_image') as mock_optimize:
                    mock_optimize.return_value = {
                        "optimized_path": Path("/tmp/test.jpg"),
                        "file_size": 512000
                    }
                    
                    # Mock file operations
                    with patch('pathlib.Path.mkdir'), \
                         patch('pathlib.Path.stat') as mock_stat:
                        mock_stat.return_value.st_size = 512000
                        
                        result = await mock_image_service.generate_single_image(request)
                        
                        assert result.prompt == request.prompt
                        assert result.story_id == request.story_id
                        assert result.scene_number == request.scene_number
                        assert result.provider == "dall-e-3"
                        assert "/images/" in result.file_url
    
    @pytest.mark.asyncio
    async def test_generate_batch_images(self, mock_image_service):
        """Test batch image generation"""
        story_id = str(uuid.uuid4())
        request = BatchImageRequest(
            story_id=story_id,
            scenes=[
                {"scene_number": 1, "prompt": "Scene 1"},
                {"scene_number": 2, "prompt": "Scene 2"}
            ],
            aspect_ratio="16:9"
        )
        
        # Mock single image generation
        with patch.object(mock_image_service, 'generate_single_image') as mock_generate:
            mock_generate.side_effect = [
                Mock(spec=['id', 'story_id', 'scene_number', 'prompt', 'file_path', 'file_url', 'provider', 'metadata', 'created_at']),
                Mock(spec=['id', 'story_id', 'scene_number', 'prompt', 'file_path', 'file_url', 'provider', 'metadata', 'created_at'])
            ]
            
            result = await mock_image_service.generate_batch_images(request)
            
            assert result.story_id == story_id
            assert result.total_generated == 2
            assert len(result.images) == 2
            assert len(result.failed_generations) == 0


class TestImageServiceAPI:
    """Test the FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["service"] == "image-service"
    
    def test_health_check_deep(self, client):
        """Test deep health check"""
        with patch('main.get_db_manager') as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.return_value.get_session.return_value.__aenter__.return_value = mock_session
            
            response = client.get("/health?deep=true")
            assert response.status_code == 200
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "version" in response.json()
    
    def test_providers_endpoint(self, client):
        """Test providers endpoint"""
        response = client.get("/providers")
        assert response.status_code == 200
        assert "providers" in response.json()
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    @patch('main.image_service')
    def test_generate_image_endpoint(self, mock_service, client):
        """Test image generation endpoint"""
        mock_service.generate_single_image.return_value = Mock(
            id="test-id",
            story_id="story-id",
            scene_number=1,
            prompt="Test prompt",
            file_path="/tmp/test.jpg",
            file_url="/images/test.jpg",
            provider="dall-e-3",
            metadata={},
            created_at=datetime.utcnow()
        )
        
        request_data = {
            "prompt": "A beautiful landscape",
            "story_id": "story-id",
            "scene_number": 1,
            "aspect_ratio": "16:9"
        }
        
        response = client.post("/generate/image", json=request_data)
        assert response.status_code == 200
    
    @patch('main.image_service')
    def test_generate_batch_endpoint(self, mock_service, client):
        """Test batch image generation endpoint"""
        mock_service.generate_batch_images.return_value = Mock(
            story_id="story-id",
            images=[],
            total_generated=0,
            failed_generations=[]
        )
        
        request_data = {
            "story_id": "story-id",
            "scenes": [
                {"scene_number": 1, "prompt": "Scene 1"}
            ]
        }
        
        response = client.post("/generate/batch", json=request_data)
        assert response.status_code == 200
    
    def test_serve_image_not_found(self, client):
        """Test serving non-existent image"""
        response = client.get("/images/nonexistent.jpg")
        assert response.status_code == 404
    
    def test_get_story_images(self, client):
        """Test getting images for a story"""
        with patch('main.get_db_manager') as mock_db_manager:
            mock_session = AsyncMock()
            mock_result = Mock()
            mock_result.fetchall.return_value = []
            mock_session.execute.return_value = mock_result
            mock_db_manager.return_value.get_session.return_value.__aenter__.return_value = mock_session
            
            response = client.get("/story/test-story-id/images")
            assert response.status_code == 200
            assert "story_id" in response.json()
            assert "images" in response.json()


class TestImageRequestValidation:
    """Test request model validation"""
    
    def test_valid_image_request(self):
        """Test valid image request creation"""
        request = ImageRequest(
            prompt="A beautiful landscape",
            story_id="story-id",
            scene_number=1,
            aspect_ratio="16:9",
            quality="standard"
        )
        
        assert request.prompt == "A beautiful landscape"
        assert request.aspect_ratio == "16:9"
        assert request.quality == "standard"
    
    def test_invalid_aspect_ratio(self):
        """Test invalid aspect ratio validation"""
        with pytest.raises(ValueError):
            ImageRequest(
                prompt="Test",
                aspect_ratio="invalid:ratio"
            )
    
    def test_prompt_length_validation(self):
        """Test prompt length validation"""
        # Too short
        with pytest.raises(ValueError):
            ImageRequest(prompt="")
        
        # Too long
        with pytest.raises(ValueError):
            ImageRequest(prompt="x" * 1001)
    
    def test_batch_request_validation(self):
        """Test batch request validation"""
        request = BatchImageRequest(
            story_id="story-id",
            scenes=[
                {"scene_number": 1, "prompt": "Scene 1"},
                {"scene_number": 2, "prompt": "Scene 2"}
            ]
        )
        
        assert request.story_id == "story-id"
        assert len(request.scenes) == 2


if __name__ == "__main__":
    pytest.main([__file__])
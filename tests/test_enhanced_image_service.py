"""
Comprehensive tests for Enhanced Image Generation Service

Tests batch processing, storage management, quality validation, and enhanced processing.
"""

import pytest
import asyncio
import uuid
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import base64
import io
from PIL import Image
import numpy as np

# Import the enhanced service components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'image-service'))

from batch_processor import BatchProcessor, JobStatus, BatchJob
from storage_manager import StorageManager, LocalStorageProvider, S3StorageProvider
from image_processor import ImageProcessor
from quality_validator import ImageQualityValidator, QualityMetrics


class TestBatchProcessor:
    """Test batch processing system"""
    
    @pytest.fixture
    async def redis_mock(self):
        """Mock Redis client"""
        redis_mock = AsyncMock()
        redis_mock.hset = AsyncMock()
        redis_mock.hget = AsyncMock()
        redis_mock.hgetall = AsyncMock()
        redis_mock.zadd = AsyncMock()
        redis_mock.zpopmin = AsyncMock()
        redis_mock.zcard = AsyncMock()
        redis_mock.scard = AsyncMock()
        redis_mock.sadd = AsyncMock()
        redis_mock.srem = AsyncMock()
        redis_mock.hincrby = AsyncMock()
        return redis_mock
    
    @pytest.fixture
    def mock_providers(self):
        """Mock image providers"""
        provider = AsyncMock()
        provider.generate_image.return_value = {
            "success": True,
            "image_url": "https://example.com/test.png",
            "provider": "mock"
        }
        provider.generate_batch.return_value = [
            {"success": True, "image_url": "https://example.com/test1.png"},
            {"success": True, "image_url": "https://example.com/test2.png"}
        ]
        return {"mock": provider}
    
    @pytest.mark.asyncio
    async def test_submit_batch_job(self, redis_mock, mock_providers):
        """Test batch job submission"""
        processor = BatchProcessor(redis_mock, mock_providers)
        
        job_id = await processor.submit_batch_job(
            prompts=["test prompt 1", "test prompt 2"],
            provider="mock",
            parameters={"quality": "high"},
            story_id="story-123",
            priority=1
        )
        
        assert job_id is not None
        assert isinstance(job_id, str)
        
        # Verify Redis calls
        redis_mock.hset.assert_called()
        redis_mock.zadd.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_job_status(self, redis_mock, mock_providers):
        """Test getting job status"""
        processor = BatchProcessor(redis_mock, mock_providers)
        
        # Mock job data
        job_data = {
            "job_id": "test-job-id",
            "prompts": ["test"],
            "provider": "mock",
            "parameters": {},
            "status": "completed",
            "progress": 1,
            "total": 1,
            "created_at": 1234567890,
            "results": [{"success": True}]
        }
        
        redis_mock.hget.return_value = json.dumps(job_data)
        
        job = await processor.get_job_status("test-job-id")
        
        assert job is not None
        assert job.job_id == "test-job-id"
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 1
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, redis_mock, mock_providers):
        """Test job cancellation"""
        processor = BatchProcessor(redis_mock, mock_providers)
        
        # Mock queued job
        job_data = {
            "job_id": "test-job-id",
            "prompts": ["test"],
            "provider": "mock",
            "parameters": {},
            "status": "queued",
            "progress": 0,
            "total": 1,
            "created_at": 1234567890
        }
        
        redis_mock.hget.return_value = json.dumps(job_data)
        
        success = await processor.cancel_job("test-job-id")
        
        assert success is True
        redis_mock.zrem.assert_called()
        redis_mock.hset.assert_called()
    
    @pytest.mark.asyncio
    async def test_queue_stats(self, redis_mock, mock_providers):
        """Test queue statistics"""
        processor = BatchProcessor(redis_mock, mock_providers)
        
        redis_mock.zcard.return_value = 5
        redis_mock.scard.return_value = 2
        redis_mock.hgetall.return_value = {
            "job1": json.dumps({"status": "queued"}),
            "job2": json.dumps({"status": "processing"}),
            "job3": json.dumps({"status": "completed"})
        }
        
        stats = await processor.get_queue_stats()
        
        assert stats["queue_length"] == 5
        assert stats["processing"] == 2
        assert "status_counts" in stats
        assert stats["total_jobs"] == 3


class TestStorageManager:
    """Test storage management system"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalStorageProvider(temp_dir, "http://localhost:8003/images")
            manager = StorageManager(provider)
            yield manager, temp_dir
    
    @pytest.fixture
    def test_image(self):
        """Create test image file"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            # Create a simple test image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(temp_file.name, 'PNG')
            yield Path(temp_file.name)
            # Cleanup
            Path(temp_file.name).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_store_image(self, temp_storage, test_image):
        """Test image storage"""
        manager, temp_dir = temp_storage
        
        result = await manager.store_image(
            file_path=test_image,
            image_id="test-image-123",
            story_id="story-456",
            scene_number=1,
            metadata={"prompt": "test prompt"}
        )
        
        assert "storage_key" in result
        assert "public_url" in result
        assert "file_hash" in result
        
        # Verify file was stored
        expected_key = "stories/story-456/scenes/001/test-image-123.png"
        assert result["storage_key"] == expected_key
        
        stored_path = Path(temp_dir) / expected_key
        assert stored_path.exists()
    
    @pytest.mark.asyncio
    async def test_list_story_images(self, temp_storage, test_image):
        """Test listing story images"""
        manager, temp_dir = temp_storage
        
        # Store a test image
        await manager.store_image(
            file_path=test_image,
            image_id="test-image-123",
            story_id="story-456",
            scene_number=1
        )
        
        # List images
        images = await manager.list_story_images("story-456")
        
        assert len(images) == 1
        assert images[0]["key"].startswith("stories/story-456/")
    
    @pytest.mark.asyncio
    async def test_file_integrity(self, temp_storage, test_image):
        """Test file integrity verification"""
        manager, temp_dir = temp_storage
        
        # Calculate original hash
        original_hash = await manager._calculate_file_hash(test_image)
        
        # Verify integrity
        is_valid = await manager.verify_file_integrity(test_image, original_hash)
        assert is_valid is True
        
        # Test with wrong hash
        is_valid = await manager.verify_file_integrity(test_image, "wrong_hash")
        assert is_valid is False


class TestImageProcessor:
    """Test enhanced image processing"""
    
    @pytest.fixture
    def processor(self):
        """Create image processor instance"""
        return ImageProcessor()
    
    @pytest.fixture
    def test_image(self):
        """Create test image file"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            img = Image.new('RGB', (400, 300), color='blue')
            img.save(temp_file.name, 'PNG')
            yield Path(temp_file.name)
            Path(temp_file.name).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_create_thumbnail(self, processor, test_image):
        """Test thumbnail creation"""
        thumb_path = await processor.create_thumbnail(test_image, size=(150, 150))
        
        assert thumb_path.exists()
        
        # Verify thumbnail size
        with Image.open(thumb_path) as thumb:
            assert thumb.width <= 150
            assert thumb.height <= 150
        
        thumb_path.unlink()
    
    @pytest.mark.asyncio
    async def test_resize_image(self, processor, test_image):
        """Test image resizing"""
        resized_path = await processor.resize_image(
            test_image,
            target_size=(200, 200),
            maintain_aspect=False
        )
        
        assert resized_path.exists()
        
        # Verify size
        with Image.open(resized_path) as resized:
            assert resized.size == (200, 200)
        
        resized_path.unlink()
    
    @pytest.mark.asyncio
    async def test_add_watermark(self, processor, test_image):
        """Test watermark addition"""
        watermarked_path = await processor.add_watermark(
            test_image,
            watermark_text="TEST WATERMARK",
            position="bottom-right"
        )
        
        assert watermarked_path.exists()
        
        # Verify watermark was added (basic check)
        with Image.open(watermarked_path) as watermarked:
            assert watermarked.mode in ('RGB', 'RGBA')
        
        watermarked_path.unlink()
    
    @pytest.mark.asyncio
    async def test_enhance_image(self, processor, test_image):
        """Test image enhancement"""
        enhanced_path = await processor.enhance_image(
            test_image,
            brightness=1.2,
            contrast=1.1,
            saturation=1.1
        )
        
        assert enhanced_path.exists()
        enhanced_path.unlink()
    
    @pytest.mark.asyncio
    async def test_create_platform_variants(self, processor, test_image):
        """Test platform variant creation"""
        variants = await processor.create_platform_variants(
            test_image,
            platforms=['instagram', 'youtube']
        )
        
        assert 'instagram' in variants
        assert 'youtube' in variants
        
        # Cleanup
        for platform_variants in variants.values():
            for variant_path in platform_variants:
                if variant_path.exists():
                    variant_path.unlink()
    
    @pytest.mark.asyncio
    async def test_get_image_info(self, processor, test_image):
        """Test image information extraction"""
        info = await processor.get_image_info(test_image)
        
        assert info['width'] == 400
        assert info['height'] == 300
        assert info['format'] == 'PNG'
        assert info['aspect_ratio'] == round(400/300, 2)
        assert 'file_size' in info


class TestQualityValidator:
    """Test image quality validation"""
    
    @pytest.fixture
    def validator(self):
        """Create quality validator instance"""
        return ImageQualityValidator()
    
    @pytest.fixture
    def test_images(self):
        """Create test images with different qualities"""
        images = {}
        
        # High quality image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            img = Image.new('RGB', (1920, 1080), color='blue')
            img.save(temp_file.name, 'PNG')
            images['high_quality'] = Path(temp_file.name)
        
        # Low quality image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(temp_file.name, 'PNG')
            images['low_quality'] = Path(temp_file.name)
        
        yield images
        
        # Cleanup
        for path in images.values():
            path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_validate_high_quality_image(self, validator, test_images):
        """Test validation of high quality image"""
        metrics = await validator.validate_image(test_images['high_quality'])
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.width == 1920
        assert metrics.height == 1080
        assert metrics.resolution_score > 90  # Should be high for 1920x1080
        assert metrics.overall_score > 0
    
    @pytest.mark.asyncio
    async def test_validate_low_quality_image(self, validator, test_images):
        """Test validation of low quality image"""
        metrics = await validator.validate_image(test_images['low_quality'])
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.width == 100
        assert metrics.height == 100
        assert metrics.resolution_score < 70  # Should be low for 100x100
        assert len(metrics.issues) > 0  # Should have issues
        assert len(metrics.recommendations) > 0  # Should have recommendations
    
    @pytest.mark.asyncio
    async def test_validate_base64_image(self, validator, test_images):
        """Test validation from base64 data"""
        # Convert image to base64
        with open(test_images['high_quality'], 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        data_url = f"data:image/png;base64,{image_data}"
        
        metrics = await validator.validate_image(data_url)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.width == 1920
        assert metrics.height == 1080
    
    def test_quality_grade(self, validator):
        """Test quality grading"""
        assert validator.get_quality_grade(95) == "Excellent"
        assert validator.get_quality_grade(85) == "Good"
        assert validator.get_quality_grade(75) == "Fair"
        assert validator.get_quality_grade(65) == "Poor"
        assert validator.get_quality_grade(50) == "Very Poor"
    
    @pytest.mark.asyncio
    async def test_batch_validate(self, validator, test_images):
        """Test batch validation"""
        image_sources = [
            test_images['high_quality'],
            test_images['low_quality']
        ]
        
        results = await validator.batch_validate(image_sources, detailed=False)
        
        assert len(results) == 2
        assert all(isinstance(result, QualityMetrics) for result in results)


class TestIntegration:
    """Integration tests for the enhanced image service"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from generation to storage"""
        # This would test the complete workflow but requires more setup
        # For now, just verify components can be initialized together
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            storage_provider = LocalStorageProvider(temp_dir, "http://localhost:8003")
            storage_manager = StorageManager(storage_provider)
            quality_validator = ImageQualityValidator()
            image_processor = ImageProcessor()
            
            # Verify they can work together
            assert storage_manager is not None
            assert quality_validator is not None
            assert image_processor is not None
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        validator = ImageQualityValidator()
        
        # Test with invalid image source
        with pytest.raises(ValueError):
            await validator.validate_image(123)  # Invalid type
        
        # Test with non-existent file
        with pytest.raises(Exception):
            await validator.validate_image("/non/existent/file.png")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
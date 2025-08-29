#!/usr/bin/env python3
"""
Comprehensive test script for Enhanced Image Generation Service
Tests all functionality including batch processing, quality validation, and storage
"""

import asyncio
import aiohttp
import json
import sys
import time
import base64
from pathlib import Path
from PIL import Image
import io

IMAGE_SERVICE_URL = "http://localhost:8003"

class ImageServiceTester:
    def __init__(self):
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        self.test_results.append({"test": test_name, "success": success, "message": message})
    
    async def test_health_check(self):
        """Test basic and deep health checks"""
        try:
            # Basic health check
            async with self.session.get(f"{IMAGE_SERVICE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test("Basic Health Check", True, f"Status: {data.get('status')}")
                else:
                    self.log_test("Basic Health Check", False, f"HTTP {response.status}")
            
            # Deep health check
            async with self.session.get(f"{IMAGE_SERVICE_URL}/health?deep=true") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test("Deep Health Check", True, 
                                f"DB: {data.get('database')}, Redis: {data.get('redis')}")
                else:
                    self.log_test("Deep Health Check", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Health Check", False, str(e))
    
    async def test_providers_endpoint(self):
        """Test providers endpoint with enhanced info"""
        try:
            async with self.session.get(f"{IMAGE_SERVICE_URL}/providers") as response:
                if response.status == 200:
                    data = await response.json()
                    providers = data.get("providers", {})
                    self.log_test("Providers Endpoint", True, 
                                f"Found {len(providers)} providers: {list(providers.keys())}")
                    
                    # Check if providers have rate limit info
                    for name, info in providers.items():
                        if "rate_limit" in info:
                            self.log_test(f"Provider {name} Rate Limit Info", True,
                                        f"Limit: {info['rate_limit']['limit_per_minute']}/min")
                else:
                    self.log_test("Providers Endpoint", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Providers Endpoint", False, str(e))
    
    async def test_single_image_generation(self):
        """Test single image generation"""
        try:
            request_data = {
                "prompt": "A beautiful sunset over mountains",
                "provider": "mock",
                "aspect_ratio": "16:9",
                "quality": "standard"
            }
            
            async with self.session.post(
                f"{IMAGE_SERVICE_URL}/generate/image",
                json=request_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test("Single Image Generation", True,
                                f"Generated image ID: {data.get('id')}")
                else:
                    text = await response.text()
                    self.log_test("Single Image Generation", False, f"HTTP {response.status}: {text}")
        except Exception as e:
            self.log_test("Single Image Generation", False, str(e))
    
    async def test_batch_generation_sync(self):
        """Test synchronous batch generation"""
        try:
            request_data = {
                "story_id": "test-story-123",
                "scenes": [
                    {"scene_number": 1, "prompt": "A forest scene"},
                    {"scene_number": 2, "prompt": "A mountain vista"}
                ],
                "provider": "mock"
            }
            
            async with self.session.post(
                f"{IMAGE_SERVICE_URL}/generate/batch",
                json=request_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test("Sync Batch Generation", True,
                                f"Generated {data.get('total_generated', 0)} images")
                else:
                    text = await response.text()
                    self.log_test("Sync Batch Generation", False, f"HTTP {response.status}: {text}")
        except Exception as e:
            self.log_test("Sync Batch Generation", False, str(e))
    
    async def test_async_batch_processing(self):
        """Test asynchronous batch processing"""
        try:
            # Submit batch job
            request_data = {
                "prompts": ["A red apple", "A blue sky", "A green forest"],
                "provider": "mock",
                "parameters": {"quality": "high"},
                "priority": 1
            }
            
            async with self.session.post(
                f"{IMAGE_SERVICE_URL}/generate/batch/async",
                json=request_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    job_id = data.get("job_id")
                    self.log_test("Async Batch Submission", True, f"Job ID: {job_id}")
                    
                    # Wait and check job status
                    await asyncio.sleep(2)
                    
                    async with self.session.get(f"{IMAGE_SERVICE_URL}/jobs/{job_id}") as status_response:
                        if status_response.status == 200:
                            status_data = await status_response.json()
                            status = status_data.get("status")
                            progress = status_data.get("progress", 0)
                            total = status_data.get("total", 0)
                            
                            self.log_test("Async Batch Status", True,
                                        f"Status: {status}, Progress: {progress}/{total}")
                            
                            # Wait for completion if still processing
                            if status in ["queued", "processing"]:
                                await asyncio.sleep(8)  # Give it time to complete
                                async with self.session.get(f"{IMAGE_SERVICE_URL}/jobs/{job_id}") as final_response:
                                    if final_response.status == 200:
                                        final_data = await final_response.json()
                                        final_status = final_data.get("status")
                                        results = final_data.get("results", [])
                                        self.log_test("Async Batch Completion", True,
                                                    f"Final status: {final_status}, Results: {len(results)}")
                        else:
                            self.log_test("Async Batch Status", False, f"HTTP {status_response.status}")
                else:
                    text = await response.text()
                    self.log_test("Async Batch Submission", False, f"HTTP {response.status}: {text}")
        except Exception as e:
            self.log_test("Async Batch Processing", False, str(e))
    
    async def test_queue_stats(self):
        """Test queue statistics"""
        try:
            async with self.session.get(f"{IMAGE_SERVICE_URL}/queue/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    queue_length = data.get("queue_length", 0)
                    processing = data.get("processing", 0)
                    self.log_test("Queue Statistics", True,
                                f"Queue: {queue_length}, Processing: {processing}")
                else:
                    self.log_test("Queue Statistics", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Queue Statistics", False, str(e))
    
    async def test_quality_validation(self):
        """Test image quality validation"""
        try:
            # Create a test image
            img = Image.new('RGB', (1024, 768), color='blue')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            request_data = {
                "image_source": f"data:image/png;base64,{image_data}",
                "detailed": True
            }
            
            async with self.session.post(
                f"{IMAGE_SERVICE_URL}/validate/quality",
                json=request_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    overall_score = data.get("metrics", {}).get("overall_score", 0)
                    grade = data.get("quality_grade", "Unknown")
                    self.log_test("Quality Validation", True,
                                f"Score: {overall_score}, Grade: {grade}")
                else:
                    text = await response.text()
                    self.log_test("Quality Validation", False, f"HTTP {response.status}: {text}")
        except Exception as e:
            self.log_test("Quality Validation", False, str(e))
    
    async def test_storage_stats(self):
        """Test storage statistics"""
        try:
            async with self.session.get(f"{IMAGE_SERVICE_URL}/storage/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    storage_type = data.get("storage_type", "unknown")
                    self.log_test("Storage Statistics", True, f"Type: {storage_type}")
                else:
                    self.log_test("Storage Statistics", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Storage Statistics", False, str(e))
    
    async def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        try:
            async with self.session.get(f"{IMAGE_SERVICE_URL}/metrics") as response:
                if response.status == 200:
                    text = await response.text()
                    # Check for some expected metrics
                    if "image_service_requests_total" in text:
                        self.log_test("Metrics Endpoint", True, "Prometheus metrics available")
                    else:
                        self.log_test("Metrics Endpoint", False, "Expected metrics not found")
                else:
                    self.log_test("Metrics Endpoint", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Metrics Endpoint", False, str(e))
    
    async def test_admin_cleanup(self):
        """Test admin cleanup functionality"""
        try:
            async with self.session.post(
                f"{IMAGE_SERVICE_URL}/admin/cleanup",
                json={"max_age_hours": 1}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    cleaned = data.get("cleaned_jobs", 0)
                    self.log_test("Admin Cleanup", True, f"Cleaned {cleaned} jobs")
                else:
                    self.log_test("Admin Cleanup", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Admin Cleanup", False, str(e))
    
    async def test_error_handling(self):
        """Test error handling"""
        try:
            # Test invalid provider
            request_data = {
                "prompt": "Test prompt",
                "provider": "invalid_provider"
            }
            
            async with self.session.post(
                f"{IMAGE_SERVICE_URL}/generate/image",
                json=request_data
            ) as response:
                if response.status == 400:
                    self.log_test("Error Handling - Invalid Provider", True, "Correctly rejected invalid provider")
                else:
                    self.log_test("Error Handling - Invalid Provider", False, f"Expected 400, got {response.status}")
            
            # Test invalid job ID
            async with self.session.get(f"{IMAGE_SERVICE_URL}/jobs/invalid-job-id") as response:
                if response.status == 404:
                    self.log_test("Error Handling - Invalid Job", True, "Correctly returned 404 for invalid job")
                else:
                    self.log_test("Error Handling - Invalid Job", False, f"Expected 404, got {response.status}")
        except Exception as e:
            self.log_test("Error Handling", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive Image Service Tests")
        print("=" * 60)
        
        tests = [
            self.test_health_check,
            self.test_providers_endpoint,
            self.test_single_image_generation,
            self.test_batch_generation_sync,
            self.test_async_batch_processing,
            self.test_queue_stats,
            self.test_quality_validation,
            self.test_storage_stats,
            self.test_metrics_endpoint,
            self.test_admin_cleanup,
            self.test_error_handling
        ]
        
        for test in tests:
            await test()
            await asyncio.sleep(0.5)  # Small delay between tests
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print(f"\n❌ {total - passed} tests failed!")
            print("\nFailed tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")
            return 1

async def main():
    """Main test runner"""
    async with ImageServiceTester() as tester:
        return await tester.run_all_tests()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
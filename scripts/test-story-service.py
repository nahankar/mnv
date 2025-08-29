#!/usr/bin/env python3
"""
Test script for Story Generation Service
"""

import asyncio
import httpx
import json
from typing import Dict, Any

async def test_story_service():
    """Test the story service endpoints"""
    
    base_url = "http://localhost:8001"
    
    async with httpx.AsyncClient() as client:
        
        # Test health check
        print("Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"Health check: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return
        
        # Test providers endpoint
        print("\nTesting providers endpoint...")
        try:
            response = await client.get(f"{base_url}/providers")
            print(f"Providers: {response.status_code}")
            if response.status_code == 200:
                providers = response.json()
                print(json.dumps(providers, indent=2))
        except Exception as e:
            print(f"Providers test failed: {e}")
        
        # Test story generation (will fail without API keys, but tests the endpoint)
        print("\nTesting story generation endpoint...")
        story_request = {
            "genre": "adventure",
            "theme": "space exploration",
            "target_length": "400-500",
            "tone": "exciting",
            "user_id": "test_user"
        }
        
        try:
            response = await client.post(
                f"{base_url}/generate/story",
                json=story_request,
                timeout=30.0
            )
            print(f"Story generation: {response.status_code}")
            if response.status_code == 200:
                story = response.json()
                print(f"Generated story preview: {story['content'][:100]}...")
            else:
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Story generation test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_story_service())
"""
TikTok Client for Platform Distribution Service

Handles video uploads to TikTok using the TikTok for Business API
with proper authentication, error handling, and retry logic.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio

import httpx

logger = logging.getLogger(__name__)


class TikTokClient:
    """TikTok for Business API client for video uploads"""
    
    BASE_URL = "https://business-api.tiktok.com/open_api/v1.3"
    UPLOAD_URL = "https://open-api.tiktok.com/share/video/upload"
    
    def __init__(self):
        self.access_token = os.getenv("TIKTOK_ACCESS_TOKEN")
        self.client_key = os.getenv("TIKTOK_CLIENT_KEY")
        self.client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
        self.client = httpx.AsyncClient(timeout=60.0)
        
        if not self.access_token:
            logger.warning("TikTok access token not configured - uploads will fail")
    
    async def upload_video(
        self,
        video_path: str,
        title: str,
        description: str = "",
        hashtags: List[str] = None,
        category: str = None,
        privacy: str = "public",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Upload video to TikTok"""
        
        if not self.access_token:
            raise Exception("TikTok client not properly configured")
        
        if not Path(video_path).exists():
            raise Exception(f"Video file not found: {video_path}")
        
        try:
            # Step 1: Upload video file
            video_info = await self._upload_video_file(video_path)
            
            # Step 2: Post video with metadata
            result = await self._post_video(
                video_info, title, description, hashtags, privacy, metadata
            )
            
            logger.info(f"Successfully uploaded video to TikTok: {result.get('video_id')}")
            
            return {
                'video_id': result.get('video_id'),
                'url': result.get('share_url'),
                'title': title,
                'status': 'published',
                'platform_response': result
            }
            
        except Exception as e:
            logger.error(f"TikTok upload error: {e}")
            raise Exception(f"TikTok upload failed: {e}")
    
    async def _upload_video_file(self, video_path: str) -> Dict[str, Any]:
        """Upload video file to TikTok"""
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            # First, get upload URL
            init_response = await self.client.post(
                f"{self.BASE_URL}/video/upload/init/",
                headers=headers,
                json={}
            )
            init_response.raise_for_status()
            
            init_data = init_response.json()
            upload_url = init_data['data']['upload_url']
            video_id = init_data['data']['video_id']
            
            # Upload video file
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}
                upload_response = await self.client.post(upload_url, files=files)
                upload_response.raise_for_status()
            
            return {
                'video_id': video_id,
                'upload_url': upload_url,
                'upload_response': upload_response.json()
            }
            
        except httpx.HTTPError as e:
            logger.error(f"TikTok file upload error: {e}")
            raise Exception(f"Failed to upload video file: {e}")
    
    async def _post_video(
        self,
        video_info: Dict[str, Any],
        title: str,
        description: str,
        hashtags: List[str],
        privacy: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post video with metadata to TikTok"""
        
        # Format description with hashtags
        formatted_description = self._format_description(title, description, hashtags)
        
        # Prepare post data
        post_data = {
            'video_id': video_info['video_id'],
            'text': formatted_description,
            'privacy_level': self._map_privacy_level(privacy),
            'disable_duet': False,
            'disable_comment': False,
            'disable_stitch': False,
        }
        
        # Add optional metadata
        if metadata:
            if 'brand_content_toggle' in metadata:
                post_data['brand_content_toggle'] = metadata['brand_content_toggle']
            if 'brand_organic_toggle' in metadata:
                post_data['brand_organic_toggle'] = metadata['brand_organic_toggle']
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/video/publish/",
                headers=headers,
                json=post_data
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') != 0:
                raise Exception(f"TikTok API error: {result.get('message')}")
            
            return result['data']
            
        except httpx.HTTPError as e:
            logger.error(f"TikTok post video error: {e}")
            raise Exception(f"Failed to post video: {e}")
    
    def _format_description(self, title: str, description: str, hashtags: List[str] = None) -> str:
        """Format TikTok video description"""
        parts = []
        
        if title:
            parts.append(title)
        
        if description:
            parts.append(description)
        
        if hashtags:
            # TikTok allows up to 100 characters in hashtags
            hashtag_text = " ".join([f"#{tag.strip('#')}" for tag in hashtags[:20]])
            parts.append(hashtag_text)
        
        text = " ".join(parts)
        
        # TikTok description limit is 2200 characters
        if len(text) > 2200:
            text = text[:2190] + "..."
        
        return text
    
    def _map_privacy_level(self, privacy: str) -> str:
        """Map privacy setting to TikTok privacy level"""
        mapping = {
            'public': 'PUBLIC_TO_EVERYONE',
            'friends': 'MUTUAL_FOLLOW_FRIENDS',
            'private': 'SELF_ONLY'
        }
        return mapping.get(privacy.lower(), 'PUBLIC_TO_EVERYONE')
    
    async def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """Get video status and basic analytics"""
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/video/query/",
                headers=headers,
                params={'video_id': video_id}
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') != 0:
                raise Exception(f"TikTok API error: {result.get('message')}")
            
            return result['data']
            
        except httpx.HTTPError as e:
            logger.error(f"TikTok API error getting video status: {e}")
            raise Exception(f"Failed to get video status: {e}")
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete video from TikTok"""
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/video/delete/",
                headers=headers,
                json={'video_id': video_id}
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') != 0:
                raise Exception(f"TikTok API error: {result.get('message')}")
            
            logger.info(f"Successfully deleted TikTok video: {video_id}")
            return True
            
        except httpx.HTTPError as e:
            logger.error(f"TikTok API error deleting video: {e}")
            raise Exception(f"Failed to delete video: {e}")
    
    async def get_user_info(self) -> Dict[str, Any]:
        """Get authenticated user information"""
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/user/info/",
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') != 0:
                raise Exception(f"TikTok API error: {result.get('message')}")
            
            return result['data']
            
        except httpx.HTTPError as e:
            logger.error(f"TikTok API error getting user info: {e}")
            raise Exception(f"Failed to get user info: {e}")
    
    async def get_video_analytics(self, video_id: str) -> Dict[str, Any]:
        """Get video analytics data"""
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/video/analytics/",
                headers=headers,
                params={
                    'video_id': video_id,
                    'metrics': 'video_views,likes,comments,shares,reach'
                }
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') != 0:
                raise Exception(f"TikTok API error: {result.get('message')}")
            
            return result['data']
            
        except httpx.HTTPError as e:
            logger.error(f"TikTok API error getting analytics: {e}")
            raise Exception(f"Failed to get video analytics: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated"""
        return self.access_token is not None
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
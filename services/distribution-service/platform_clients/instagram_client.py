"""
Instagram Client for Platform Distribution Service

Handles video uploads to Instagram using the Instagram Graph API
with proper authentication, error handling, and retry logic.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio

import httpx

logger = logging.getLogger(__name__)


class InstagramClient:
    """Instagram Graph API client for video uploads"""
    
    BASE_URL = "https://graph.facebook.com/v18.0"
    
    def __init__(self):
        self.access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        self.page_id = os.getenv("INSTAGRAM_PAGE_ID")
        self.client = httpx.AsyncClient(timeout=60.0)
        
        if not self.access_token:
            logger.warning("Instagram access token not configured - uploads will fail")
    
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
        """Upload video to Instagram"""
        
        if not self.access_token or not self.page_id:
            raise Exception("Instagram client not properly configured")
        
        if not Path(video_path).exists():
            raise Exception(f"Video file not found: {video_path}")
        
        try:
            # Step 1: Create container for video upload
            container_id = await self._create_video_container(
                video_path, title, description, hashtags, metadata
            )
            
            # Step 2: Publish the video
            result = await self._publish_video_container(container_id)
            
            # Step 3: Get published video details
            video_details = await self._get_video_details(result['id'])
            
            logger.info(f"Successfully uploaded video to Instagram: {result['id']}")
            
            return {
                'video_id': result['id'],
                'url': video_details.get('permalink_url'),
                'title': title,
                'status': 'published',
                'platform_response': {
                    'container_id': container_id,
                    'publish_result': result,
                    'video_details': video_details
                }
            }
            
        except Exception as e:
            logger.error(f"Instagram upload error: {e}")
            raise Exception(f"Instagram upload failed: {e}")
    
    async def _create_video_container(
        self, 
        video_path: str, 
        title: str, 
        description: str, 
        hashtags: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """Create video container for upload"""
        
        # Upload video file first
        video_url = await self._upload_video_file(video_path)
        
        # Format caption with hashtags
        caption = self._format_caption(title, description, hashtags)
        
        # Prepare container creation request
        container_data = {
            'media_type': 'VIDEO',
            'video_url': video_url,
            'caption': caption,
            'access_token': self.access_token
        }
        
        # Add thumb_offset for video thumbnail (optional)
        if metadata and 'thumb_offset' in metadata:
            container_data['thumb_offset'] = metadata['thumb_offset']
        
        # Create container
        response = await self.client.post(
            f"{self.BASE_URL}/{self.page_id}/media",
            data=container_data
        )
        response.raise_for_status()
        
        result = response.json()
        if 'id' not in result:
            raise Exception(f"Failed to create video container: {result}")
        
        return result['id']
    
    async def _upload_video_file(self, video_path: str) -> str:
        """Upload video file and return URL (simplified - in production, use proper file upload)"""
        # Note: In a real implementation, you would upload the file to a CDN or file hosting service
        # and return the public URL. For this example, we'll assume the file is already accessible
        # via a public URL or you have a separate file upload mechanism.
        
        # This is a simplified placeholder - you would implement actual file upload here
        # For example, upload to AWS S3, Google Cloud Storage, etc.
        
        # For now, we'll assume the video_path is already a URL or we need to upload it
        if video_path.startswith('http'):
            return video_path
        else:
            # In real implementation, upload file and return URL
            raise Exception("Video file upload mechanism not implemented - provide video URL instead of local path")
    
    async def _publish_video_container(self, container_id: str) -> Dict[str, Any]:
        """Publish the video container"""
        
        # Poll container status until ready
        await self._wait_for_container_ready(container_id)
        
        # Publish the container
        publish_data = {
            'creation_id': container_id,
            'access_token': self.access_token
        }
        
        response = await self.client.post(
            f"{self.BASE_URL}/{self.page_id}/media_publish",
            data=publish_data
        )
        response.raise_for_status()
        
        result = response.json()
        if 'id' not in result:
            raise Exception(f"Failed to publish video: {result}")
        
        return result
    
    async def _wait_for_container_ready(self, container_id: str, max_wait: int = 300):
        """Wait for container to be ready for publishing"""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check container status
            response = await self.client.get(
                f"{self.BASE_URL}/{container_id}",
                params={
                    'fields': 'status_code',
                    'access_token': self.access_token
                }
            )
            response.raise_for_status()
            
            result = response.json()
            status_code = result.get('status_code')
            
            if status_code == 'FINISHED':
                break
            elif status_code == 'ERROR':
                raise Exception("Video processing failed")
            elif status_code in ['IN_PROGRESS', 'PUBLISHED']:
                # Wait and retry
                await asyncio.sleep(5)
                
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > max_wait:
                    raise Exception("Timeout waiting for video processing")
            else:
                raise Exception(f"Unknown status code: {status_code}")
    
    async def _get_video_details(self, video_id: str) -> Dict[str, Any]:
        """Get video details after publishing"""
        response = await self.client.get(
            f"{self.BASE_URL}/{video_id}",
            params={
                'fields': 'id,permalink_url,media_type,media_url,thumbnail_url,timestamp',
                'access_token': self.access_token
            }
        )
        response.raise_for_status()
        
        return response.json()
    
    def _format_caption(self, title: str, description: str, hashtags: List[str] = None) -> str:
        """Format Instagram caption with title, description, and hashtags"""
        caption_parts = []
        
        if title:
            caption_parts.append(title)
        
        if description:
            caption_parts.append(description)
        
        if hashtags:
            # Instagram allows up to 30 hashtags
            hashtag_text = " ".join([f"#{tag.strip('#')}" for tag in hashtags[:30]])
            caption_parts.append(hashtag_text)
        
        caption = "\n\n".join(caption_parts)
        
        # Instagram caption limit is 2200 characters
        if len(caption) > 2200:
            caption = caption[:2190] + "..."
        
        return caption
    
    async def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """Get video status and insights"""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/{video_id}",
                params={
                    'fields': 'id,media_type,permalink_url,timestamp,like_count,comments_count',
                    'access_token': self.access_token
                }
            )
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"Instagram API error getting video status: {e}")
            raise Exception(f"Failed to get video status: {e}")
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete video from Instagram"""
        try:
            response = await self.client.delete(
                f"{self.BASE_URL}/{video_id}",
                params={'access_token': self.access_token}
            )
            response.raise_for_status()
            
            logger.info(f"Successfully deleted Instagram video: {video_id}")
            return True
            
        except httpx.HTTPError as e:
            logger.error(f"Instagram API error deleting video: {e}")
            raise Exception(f"Failed to delete video: {e}")
    
    async def update_video(
        self, 
        video_id: str, 
        caption: str = None
    ) -> Dict[str, Any]:
        """Update video caption (limited update capabilities on Instagram)"""
        # Note: Instagram has limited update capabilities for published media
        # You can mainly update the caption of posts
        
        if not caption:
            raise Exception("Caption is required for Instagram video updates")
        
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/{video_id}",
                data={
                    'caption': caption,
                    'access_token': self.access_token
                }
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully updated Instagram video: {video_id}")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Instagram API error updating video: {e}")
            raise Exception(f"Failed to update video: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated"""
        return self.access_token is not None and self.page_id is not None
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
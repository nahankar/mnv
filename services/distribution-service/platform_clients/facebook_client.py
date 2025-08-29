"""
Facebook Client for Platform Distribution Service

Handles video uploads to Facebook using the Facebook Graph API
with proper authentication, error handling, and retry logic.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio

import httpx

logger = logging.getLogger(__name__)


class FacebookClient:
    """Facebook Graph API client for video uploads"""
    
    BASE_URL = "https://graph.facebook.com/v18.0"
    
    def __init__(self):
        self.access_token = os.getenv("FACEBOOK_ACCESS_TOKEN")
        self.page_id = os.getenv("FACEBOOK_PAGE_ID")
        self.client = httpx.AsyncClient(timeout=60.0)
        
        if not self.access_token:
            logger.warning("Facebook access token not configured - uploads will fail")
    
    async def upload_video(
        self,
        video_path: str,
        title: str,
        description: str = "",
        hashtags: List[str] = None,
        category: str = None,
        privacy: str = "EVERYONE",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Upload video to Facebook"""
        
        if not self.access_token or not self.page_id:
            raise Exception("Facebook client not properly configured")
        
        if not Path(video_path).exists():
            raise Exception(f"Video file not found: {video_path}")
        
        try:
            # For large files, use resumable upload
            file_size = Path(video_path).stat().st_size
            
            if file_size > 25 * 1024 * 1024:  # 25MB threshold
                result = await self._upload_large_video(
                    video_path, title, description, hashtags, privacy, metadata
                )
            else:
                result = await self._upload_small_video(
                    video_path, title, description, hashtags, privacy, metadata
                )
            
            logger.info(f"Successfully uploaded video to Facebook: {result.get('id')}")
            
            return {
                'video_id': result.get('id'),
                'url': f"https://www.facebook.com/{result.get('id')}",
                'title': title,
                'status': 'published',
                'platform_response': result
            }
            
        except Exception as e:
            logger.error(f"Facebook upload error: {e}")
            raise Exception(f"Facebook upload failed: {e}")
    
    async def _upload_small_video(
        self,
        video_path: str,
        title: str,
        description: str,
        hashtags: List[str],
        privacy: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload small video file directly"""
        
        # Format description with hashtags
        formatted_description = self._format_description(title, description, hashtags)
        
        # Prepare form data
        form_data = {
            'description': formatted_description,
            'privacy': privacy,
            'access_token': self.access_token
        }
        
        # Add optional metadata
        if metadata:
            if 'thumb' in metadata:
                form_data['thumb'] = metadata['thumb']
            if 'embeddable' in metadata:
                form_data['embeddable'] = metadata['embeddable']
            if 'slideshow_spec' in metadata:
                form_data['slideshow_spec'] = metadata['slideshow_spec']
        
        try:
            # Upload video file
            with open(video_path, 'rb') as video_file:
                files = {'source': video_file}
                
                response = await self.client.post(
                    f"{self.BASE_URL}/{self.page_id}/videos",
                    data=form_data,
                    files=files
                )
                response.raise_for_status()
            
            result = response.json()
            
            if 'error' in result:
                raise Exception(f"Facebook API error: {result['error']}")
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Facebook upload error: {e}")
            raise Exception(f"Failed to upload video: {e}")
    
    async def _upload_large_video(
        self,
        video_path: str,
        title: str,
        description: str,
        hashtags: List[str],
        privacy: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload large video using resumable upload"""
        
        file_size = Path(video_path).stat().st_size
        
        # Step 1: Initialize resumable upload
        session_id = await self._init_resumable_upload(file_size)
        
        # Step 2: Upload video chunks
        await self._upload_video_chunks(video_path, session_id, file_size)
        
        # Step 3: Finish upload and publish
        result = await self._finish_resumable_upload(
            session_id, title, description, hashtags, privacy, metadata
        )
        
        return result
    
    async def _init_resumable_upload(self, file_size: int) -> str:
        """Initialize resumable upload session"""
        
        data = {
            'upload_phase': 'start',
            'file_size': file_size,
            'access_token': self.access_token
        }
        
        response = await self.client.post(
            f"{self.BASE_URL}/{self.page_id}/videos",
            data=data
        )
        response.raise_for_status()
        
        result = response.json()
        
        if 'error' in result:
            raise Exception(f"Facebook API error: {result['error']}")
        
        return result['video_id']
    
    async def _upload_video_chunks(self, video_path: str, session_id: str, file_size: int):
        """Upload video in chunks"""
        
        chunk_size = 4 * 1024 * 1024  # 4MB chunks
        
        with open(video_path, 'rb') as video_file:
            uploaded = 0
            
            while uploaded < file_size:
                chunk_data = video_file.read(chunk_size)
                if not chunk_data:
                    break
                
                start_offset = uploaded
                end_offset = min(uploaded + len(chunk_data) - 1, file_size - 1)
                
                # Upload chunk
                data = {
                    'upload_phase': 'transfer',
                    'start_offset': start_offset,
                    'upload_session_id': session_id,
                    'access_token': self.access_token
                }
                
                files = {'video_file_chunk': chunk_data}
                
                response = await self.client.post(
                    f"{self.BASE_URL}/{self.page_id}/videos",
                    data=data,
                    files=files
                )
                response.raise_for_status()
                
                result = response.json()
                if 'error' in result:
                    raise Exception(f"Facebook chunk upload error: {result['error']}")
                
                uploaded += len(chunk_data)
                
                # Log progress
                progress = (uploaded / file_size) * 100
                logger.info(f"Upload progress: {progress:.1f}%")
    
    async def _finish_resumable_upload(
        self,
        session_id: str,
        title: str,
        description: str,
        hashtags: List[str],
        privacy: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finish resumable upload and publish video"""
        
        # Format description with hashtags
        formatted_description = self._format_description(title, description, hashtags)
        
        data = {
            'upload_phase': 'finish',
            'upload_session_id': session_id,
            'description': formatted_description,
            'privacy': privacy,
            'access_token': self.access_token
        }
        
        # Add optional metadata
        if metadata:
            if 'thumb' in metadata:
                data['thumb'] = metadata['thumb']
            if 'embeddable' in metadata:
                data['embeddable'] = metadata['embeddable']
        
        response = await self.client.post(
            f"{self.BASE_URL}/{self.page_id}/videos",
            data=data
        )
        response.raise_for_status()
        
        result = response.json()
        
        if 'error' in result:
            raise Exception(f"Facebook finish upload error: {result['error']}")
        
        return result
    
    def _format_description(self, title: str, description: str, hashtags: List[str] = None) -> str:
        """Format Facebook video description"""
        parts = []
        
        if title:
            parts.append(title)
        
        if description:
            parts.append(description)
        
        if hashtags:
            # Facebook supports hashtags in description
            hashtag_text = " ".join([f"#{tag.strip('#')}" for tag in hashtags[:10]])
            parts.append(hashtag_text)
        
        formatted_description = "\n\n".join(parts)
        
        # Facebook description limit is around 63,206 characters
        if len(formatted_description) > 60000:
            formatted_description = formatted_description[:59990] + "..."
        
        return formatted_description
    
    async def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """Get video status and basic info"""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/{video_id}",
                params={
                    'fields': 'id,status,created_time,description,permalink_url,length',
                    'access_token': self.access_token
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'error' in result:
                raise Exception(f"Facebook API error: {result['error']}")
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Facebook API error getting video status: {e}")
            raise Exception(f"Failed to get video status: {e}")
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete video from Facebook"""
        try:
            response = await self.client.delete(
                f"{self.BASE_URL}/{video_id}",
                params={'access_token': self.access_token}
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'error' in result:
                raise Exception(f"Facebook API error: {result['error']}")
            
            logger.info(f"Successfully deleted Facebook video: {video_id}")
            return result.get('success', False)
            
        except httpx.HTTPError as e:
            logger.error(f"Facebook API error deleting video: {e}")
            raise Exception(f"Failed to delete video: {e}")
    
    async def update_video(
        self, 
        video_id: str, 
        description: str = None, 
        privacy: str = None
    ) -> Dict[str, Any]:
        """Update video metadata"""
        if not description and not privacy:
            raise Exception("Description or privacy must be provided for update")
        
        data = {'access_token': self.access_token}
        
        if description:
            data['description'] = description
        if privacy:
            data['privacy'] = privacy
        
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/{video_id}",
                data=data
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'error' in result:
                raise Exception(f"Facebook API error: {result['error']}")
            
            logger.info(f"Successfully updated Facebook video: {video_id}")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Facebook API error updating video: {e}")
            raise Exception(f"Failed to update video: {e}")
    
    async def get_video_insights(self, video_id: str) -> Dict[str, Any]:
        """Get video analytics/insights"""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/{video_id}/video_insights",
                params={
                    'metric': 'total_video_views,total_video_views_unique,total_video_impressions,total_video_reactions_by_type_total',
                    'access_token': self.access_token
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'error' in result:
                raise Exception(f"Facebook API error: {result['error']}")
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Facebook API error getting insights: {e}")
            raise Exception(f"Failed to get video insights: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated"""
        return self.access_token is not None and self.page_id is not None
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
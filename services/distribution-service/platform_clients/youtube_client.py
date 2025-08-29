"""
YouTube Client for Platform Distribution Service

Handles video uploads to YouTube using the YouTube Data API v3
with proper authentication, error handling, and retry logic.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import httpx
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

logger = logging.getLogger(__name__)


class YouTubeClient:
    """YouTube Data API v3 client for video uploads"""
    
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    API_SERVICE_NAME = 'youtube'
    API_VERSION = 'v3'
    
    def __init__(self):
        self.credentials = None
        self.youtube_service = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize YouTube API client with credentials"""
        try:
            # Try to load credentials from environment or file
            client_id = os.getenv("YOUTUBE_CLIENT_ID")
            client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
            refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
            
            if client_id and client_secret and refresh_token:
                self.credentials = Credentials(
                    token=None,
                    refresh_token=refresh_token,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=client_id,
                    client_secret=client_secret
                )
                
                # Build the service
                self.youtube_service = build(
                    self.API_SERVICE_NAME, 
                    self.API_VERSION, 
                    credentials=self.credentials
                )
                
                logger.info("YouTube client initialized successfully")
            else:
                logger.warning("YouTube credentials not configured - uploads will fail")
                
        except Exception as e:
            logger.error(f"Failed to initialize YouTube client: {e}")
    
    async def upload_video(
        self,
        video_path: str,
        title: str,
        description: str = "",
        hashtags: List[str] = None,
        category: str = "22",  # People & Blogs
        privacy: str = "public",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Upload video to YouTube with timeout handling"""
        
        if not self.youtube_service:
            raise Exception("YouTube client not properly initialized")
        
        if not Path(video_path).exists():
            raise Exception(f"Video file not found: {video_path}")
        
        try:
            # Validate credentials before upload
            await self._validate_credentials()
            
            # Prepare video metadata
            video_metadata = {
                'snippet': {
                    'title': title[:100],  # YouTube title limit
                    'description': self._format_description(description, hashtags),
                    'categoryId': self._get_category_id(category),
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': privacy,
                    'madeForKids': False,
                    'selfDeclaredMadeForKids': False
                }
            }
            
            # Add additional metadata
            if metadata:
                if 'thumbnail' in metadata:
                    video_metadata['snippet']['thumbnails'] = metadata['thumbnail']
                if 'tags' in metadata:
                    video_metadata['snippet']['tags'] = metadata['tags'][:15]  # YouTube limit
            
            # Create media upload
            media = MediaFileUpload(
                video_path,
                chunksize=-1,  # Upload in single chunk
                resumable=True,
                mimetype='video/*'
            )
            
            # Execute upload request
            insert_request = self.youtube_service.videos().insert(
                part=','.join(video_metadata.keys()),
                body=video_metadata,
                media_body=media
            )
            
            # Execute the upload with retry logic
            response = await self._execute_upload_with_retry(insert_request)
            
            # Extract results
            video_id = response.get('id')
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            logger.info(f"Successfully uploaded video to YouTube: {video_url}")
            
            return {
                'video_id': video_id,
                'url': video_url,
                'title': response.get('snippet', {}).get('title'),
                'status': response.get('status', {}).get('privacyStatus'),
                'platform_response': response
            }
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise Exception(f"YouTube upload failed: {e}")
        except Exception as e:
            logger.error(f"YouTube upload error: {e}")
            raise Exception(f"YouTube upload failed: {e}")
    
    async def _execute_upload_with_retry(self, insert_request, max_retries: int = 3):
        """Execute upload with retry logic for resumable uploads"""
        import time
        
        for attempt in range(max_retries):
            try:
                response = None
                error = None
                retry = 0
                
                while response is None:
                    try:
                        status, response = insert_request.next_chunk()
                        if response is not None:
                            if 'id' in response:
                                return response
                            else:
                                raise Exception(f"Upload failed with response: {response}")
                                
                    except HttpError as e:
                        if e.resp.status in [500, 502, 503, 504]:
                            # Retryable error
                            error = f"Retryable error: {e}"
                            retry += 1
                            if retry > 3:
                                raise Exception(f"Too many retries: {error}")
                            
                            max_sleep = 2 ** retry
                            sleep_seconds = min(max_sleep, 60)
                            logger.warning(f"Retrying in {sleep_seconds} seconds...")
                            time.sleep(sleep_seconds)
                        else:
                            raise Exception(f"Non-retryable error: {e}")
                
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Upload attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def _format_description(self, description: str, hashtags: List[str] = None) -> str:
        """Format description with hashtags"""
        formatted_desc = description[:4900]  # Leave room for hashtags
        
        if hashtags:
            # Add hashtags at the end
            hashtag_text = "\n\n" + " ".join([f"#{tag.strip('#')}" for tag in hashtags[:15]])
            if len(formatted_desc + hashtag_text) <= 5000:
                formatted_desc += hashtag_text
        
        return formatted_desc
    
    async def _validate_credentials(self):
        """Validate YouTube API credentials"""
        try:
            # Test API access by getting channel info
            request = self.youtube_service.channels().list(
                part='snippet',
                mine=True
            )
            response = request.execute()
            
            if not response.get('items'):
                raise Exception("No YouTube channel found for the authenticated user")
                
            logger.info("YouTube credentials validated successfully")
            
        except Exception as e:
            logger.error(f"YouTube credential validation failed: {e}")
            raise Exception(f"Invalid YouTube credentials: {e}")
    
    def _get_category_id(self, category: str) -> str:
        """Get YouTube category ID from category name"""
        category_mapping = {
            "film": "1",
            "autos": "2", 
            "music": "10",
            "pets": "15",
            "sports": "17",
            "travel": "19",
            "gaming": "20",
            "people": "22",
            "comedy": "23",
            "entertainment": "24",
            "news": "25",
            "education": "27",
            "science": "28"
        }
        
        return category_mapping.get(category.lower(), "22")  # Default to People & Blogs
    
    async def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """Get video upload status and processing details"""
        if not self.youtube_service:
            raise Exception("YouTube client not properly initialized")
        
        try:
            response = self.youtube_service.videos().list(
                part='status,processingDetails',
                id=video_id
            ).execute()
            
            if not response.get('items'):
                raise Exception(f"Video not found: {video_id}")
            
            video_data = response['items'][0]
            return {
                'video_id': video_id,
                'upload_status': video_data.get('status', {}).get('uploadStatus'),
                'privacy_status': video_data.get('status', {}).get('privacyStatus'),
                'processing_status': video_data.get('processingDetails', {}).get('processingStatus'),
                'processing_progress': video_data.get('processingDetails', {}).get('processingProgress'),
                'failure_reason': video_data.get('status', {}).get('failureReason'),
                'rejection_reason': video_data.get('status', {}).get('rejectionReason')
            }
            
        except HttpError as e:
            logger.error(f"YouTube API error getting video status: {e}")
            raise Exception(f"Failed to get video status: {e}")
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete video from YouTube"""
        if not self.youtube_service:
            raise Exception("YouTube client not properly initialized")
        
        try:
            self.youtube_service.videos().delete(id=video_id).execute()
            logger.info(f"Successfully deleted YouTube video: {video_id}")
            return True
            
        except HttpError as e:
            logger.error(f"YouTube API error deleting video: {e}")
            raise Exception(f"Failed to delete video: {e}")
    
    async def update_video(
        self, 
        video_id: str, 
        title: str = None, 
        description: str = None, 
        privacy: str = None
    ) -> Dict[str, Any]:
        """Update video metadata"""
        if not self.youtube_service:
            raise Exception("YouTube client not properly initialized")
        
        try:
            # Get current video data
            current_response = self.youtube_service.videos().list(
                part='snippet,status',
                id=video_id
            ).execute()
            
            if not current_response.get('items'):
                raise Exception(f"Video not found: {video_id}")
            
            video_data = current_response['items'][0]
            
            # Update fields
            if title:
                video_data['snippet']['title'] = title[:100]
            if description:
                video_data['snippet']['description'] = description[:5000]
            if privacy:
                video_data['status']['privacyStatus'] = privacy
            
            # Execute update
            response = self.youtube_service.videos().update(
                part='snippet,status',
                body=video_data
            ).execute()
            
            logger.info(f"Successfully updated YouTube video: {video_id}")
            return {
                'video_id': video_id,
                'title': response.get('snippet', {}).get('title'),
                'status': response.get('status', {}).get('privacyStatus')
            }
            
        except HttpError as e:
            logger.error(f"YouTube API error updating video: {e}")
            raise Exception(f"Failed to update video: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated"""
        return self.credentials is not None and self.youtube_service is not None
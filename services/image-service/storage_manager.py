"""
Cloud Storage Manager for Image Assets

Handles storage of generated images to various cloud providers (S3, GCS, Azure)
with proper metadata tracking and file organization.
"""

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import aiofiles
import hashlib
import mimetypes

logger = logging.getLogger(__name__)


class StorageProvider(ABC):
    """Abstract base class for storage providers"""
    
    @abstractmethod
    async def upload_file(
        self,
        file_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload file and return public URL"""
        pass
    
    @abstractmethod
    async def delete_file(self, key: str) -> bool:
        """Delete file from storage"""
        pass
    
    @abstractmethod
    async def get_file_url(self, key: str) -> str:
        """Get public URL for file"""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str) -> List[Dict[str, Any]]:
        """List files with given prefix"""
        pass


class S3StorageProvider(StorageProvider):
    """AWS S3 storage provider"""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self._client = None
    
    async def _get_client(self):
        """Get or create S3 client"""
        if self._client is None:
            try:
                import aioboto3
                session = aioboto3.Session()
                self._client = session.client('s3', region_name=self.region)
            except ImportError:
                raise ImportError("aioboto3 is required for S3 storage. Install with: pip install aioboto3")
        return self._client
    
    async def upload_file(
        self,
        file_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload file to S3"""
        client = await self._get_client()
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = 'application/octet-stream'
        
        extra_args = {
            'ContentType': content_type,
            'ACL': 'public-read'
        }
        
        if metadata:
            extra_args['Metadata'] = metadata
        
        async with client as s3:
            await s3.upload_file(
                str(file_path),
                self.bucket_name,
                key,
                ExtraArgs=extra_args
            )
        
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{key}"
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from S3"""
        try:
            client = await self._get_client()
            async with client as s3:
                await s3.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete S3 object {key}: {e}")
            return False
    
    async def get_file_url(self, key: str) -> str:
        """Get public URL for S3 object"""
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{key}"
    
    async def list_files(self, prefix: str) -> List[Dict[str, Any]]:
        """List files in S3 with prefix"""
        client = await self._get_client()
        files = []
        
        async with client as s3:
            paginator = s3.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        files.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag']
                        })
        
        return files


class GCSStorageProvider(StorageProvider):
    """Google Cloud Storage provider"""
    
    def __init__(self, bucket_name: str, project_id: Optional[str] = None):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._client = None
    
    async def _get_client(self):
        """Get or create GCS client"""
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=self.project_id)
            except ImportError:
                raise ImportError("google-cloud-storage is required for GCS. Install with: pip install google-cloud-storage")
        return self._client
    
    async def upload_file(
        self,
        file_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload file to GCS"""
        client = await self._get_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(key)
        
        # Set metadata
        if metadata:
            blob.metadata = metadata
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type:
            blob.content_type = content_type
        
        # Upload file
        await asyncio.get_event_loop().run_in_executor(
            None,
            blob.upload_from_filename,
            str(file_path)
        )
        
        # Make public
        blob.make_public()
        
        return blob.public_url
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from GCS"""
        try:
            client = await self._get_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(key)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                blob.delete
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete GCS object {key}: {e}")
            return False
    
    async def get_file_url(self, key: str) -> str:
        """Get public URL for GCS object"""
        return f"https://storage.googleapis.com/{self.bucket_name}/{key}"
    
    async def list_files(self, prefix: str) -> List[Dict[str, Any]]:
        """List files in GCS with prefix"""
        client = await self._get_client()
        bucket = client.bucket(self.bucket_name)
        
        blobs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(bucket.list_blobs(prefix=prefix))
        )
        
        files = []
        for blob in blobs:
            files.append({
                'key': blob.name,
                'size': blob.size,
                'last_modified': blob.time_created,
                'etag': blob.etag,
                'content_type': blob.content_type
            })
        
        return files


class LocalStorageProvider(StorageProvider):
    """Local filesystem storage provider"""
    
    def __init__(self, base_path: str, base_url: str):
        self.base_path = Path(base_path)
        self.base_url = base_url.rstrip('/')
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(
        self,
        file_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Copy file to local storage"""
        target_path = self.base_path / key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        async with aiofiles.open(file_path, 'rb') as src:
            async with aiofiles.open(target_path, 'wb') as dst:
                await dst.write(await src.read())
        
        # Store metadata if provided
        if metadata:
            metadata_path = target_path.with_suffix(target_path.suffix + '.meta')
            async with aiofiles.open(metadata_path, 'w') as f:
                import json
                await f.write(json.dumps(metadata))
        
        return f"{self.base_url}/{key}"
    
    async def delete_file(self, key: str) -> bool:
        """Delete file from local storage"""
        try:
            file_path = self.base_path / key
            if file_path.exists():
                file_path.unlink()
            
            # Also delete metadata file if exists
            metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
            if metadata_path.exists():
                metadata_path.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete local file {key}: {e}")
            return False
    
    async def get_file_url(self, key: str) -> str:
        """Get URL for local file"""
        return f"{self.base_url}/{key}"
    
    async def list_files(self, prefix: str) -> List[Dict[str, Any]]:
        """List local files with prefix"""
        prefix_path = self.base_path / prefix
        files = []
        
        if prefix_path.exists():
            for file_path in prefix_path.rglob('*'):
                if file_path.is_file() and not file_path.suffix == '.meta':
                    relative_path = file_path.relative_to(self.base_path)
                    stat = file_path.stat()
                    
                    files.append({
                        'key': str(relative_path),
                        'size': stat.st_size,
                        'last_modified': datetime.fromtimestamp(stat.st_mtime),
                        'content_type': mimetypes.guess_type(str(file_path))[0]
                    })
        
        return files


class StorageManager:
    """Main storage manager that handles file organization and metadata"""
    
    def __init__(self, provider: StorageProvider):
        self.provider = provider
    
    def _generate_file_key(
        self,
        story_id: Optional[str],
        scene_number: Optional[int],
        image_id: str,
        file_extension: str
    ) -> str:
        """Generate organized file key"""
        # Create hierarchical structure: story_id/scene_number/image_id.ext
        parts = []
        
        if story_id:
            parts.append(f"stories/{story_id}")
            if scene_number is not None:
                parts.append(f"scenes/{scene_number:03d}")
        else:
            parts.append("standalone")
        
        parts.append(f"{image_id}{file_extension}")
        
        return "/".join(parts)
    
    async def store_image(
        self,
        file_path: Path,
        image_id: str,
        story_id: Optional[str] = None,
        scene_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Store image with proper organization and metadata"""
        
        # Generate file key
        file_extension = file_path.suffix
        key = self._generate_file_key(story_id, scene_number, image_id, file_extension)
        
        # Prepare metadata
        storage_metadata = {
            'image_id': image_id,
            'uploaded_at': datetime.utcnow().isoformat(),
            'file_size': str(file_path.stat().st_size),
            'content_type': mimetypes.guess_type(str(file_path))[0] or 'image/jpeg'
        }
        
        if story_id:
            storage_metadata['story_id'] = story_id
        if scene_number is not None:
            storage_metadata['scene_number'] = str(scene_number)
        if metadata:
            storage_metadata.update({k: str(v) for k, v in metadata.items()})
        
        # Calculate file hash for integrity
        file_hash = await self._calculate_file_hash(file_path)
        storage_metadata['file_hash'] = file_hash
        
        # Upload to storage
        public_url = await self.provider.upload_file(file_path, key, storage_metadata)
        
        logger.info(f"Stored image {image_id} at {key}")
        
        return {
            'storage_key': key,
            'public_url': public_url,
            'file_hash': file_hash
        }
    
    async def delete_image(self, storage_key: str) -> bool:
        """Delete image from storage"""
        return await self.provider.delete_file(storage_key)
    
    async def get_image_url(self, storage_key: str) -> str:
        """Get public URL for image"""
        return await self.provider.get_file_url(storage_key)
    
    async def list_story_images(self, story_id: str) -> List[Dict[str, Any]]:
        """List all images for a story"""
        prefix = f"stories/{story_id}/"
        return await self.provider.list_files(prefix)
    
    async def list_scene_images(self, story_id: str, scene_number: int) -> List[Dict[str, Any]]:
        """List images for a specific scene"""
        prefix = f"stories/{story_id}/scenes/{scene_number:03d}/"
        return await self.provider.list_files(prefix)
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def verify_file_integrity(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file integrity using hash"""
        actual_hash = await self._calculate_file_hash(file_path)
        return actual_hash == expected_hash


def create_storage_manager() -> StorageManager:
    """Factory function to create storage manager based on configuration"""
    storage_type = os.getenv('STORAGE_TYPE', 'local').lower()
    
    if storage_type == 's3':
        bucket_name = os.getenv('S3_BUCKET_NAME')
        region = os.getenv('S3_REGION', 'us-east-1')
        
        if not bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is required for S3 storage")
        
        provider = S3StorageProvider(bucket_name, region)
        
    elif storage_type == 'gcs':
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        project_id = os.getenv('GCS_PROJECT_ID')
        
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable is required for GCS storage")
        
        provider = GCSStorageProvider(bucket_name, project_id)
        
    else:  # Default to local storage
        base_path = os.getenv('LOCAL_STORAGE_PATH', './uploads/images')
        base_url = os.getenv('LOCAL_STORAGE_URL', 'http://localhost:8003/images')
        
        provider = LocalStorageProvider(base_path, base_url)
    
    return StorageManager(provider)
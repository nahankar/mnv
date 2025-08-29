"""Pydantic models for API request/response validation"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

from .models import ContentStatus, PlatformType, MediaType


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    class Config:
        from_attributes = True
        use_enum_values = True


# Story schemas
class StoryRequest(BaseModel):
    """Request schema for story generation"""
    genre: Optional[str] = Field(None, max_length=100, description="Story genre")
    theme: Optional[str] = Field(None, max_length=100, description="Story theme")
    target_length: Optional[str] = Field("300-500", description="Target length in words")
    tone: Optional[str] = Field("engaging", description="Story tone")
    user_id: Optional[str] = Field(None, description="User ID for rate limiting")

class StoryGenerationRequest(BaseModel):
    """Request schema for story generation (legacy)"""
    genre: Optional[str] = Field(None, max_length=100, description="Story genre")
    theme: Optional[str] = Field(None, max_length=100, description="Story theme")
    target_length: Optional[int] = Field(None, ge=50, le=2000, description="Target length in words")
    llm_provider: Optional[str] = Field(None, description="Preferred LLM provider")
    generation_parameters: Optional[Dict[str, Any]] = Field(None, description="Custom generation parameters")


class StoryResponse(BaseSchema):
    """Response schema for story generation"""
    id: str
    content: str
    word_count: int
    genre: Optional[str]
    theme: Optional[str]
    provider_used: str
    generation_cost: float
    status: ContentStatus

class StoryResponseDetailed(BaseSchema):
    """Detailed response schema for story data"""
    id: uuid.UUID
    title: str
    content: str
    genre: Optional[str]
    theme: Optional[str]
    target_length: Optional[int]
    actual_length: Optional[int]
    llm_provider: Optional[str]
    model_name: Optional[str]
    generation_parameters: Optional[Dict[str, Any]]
    generation_cost: float
    status: ContentStatus
    created_at: datetime
    updated_at: Optional[datetime]


class MetadataRequest(BaseModel):
    """Request schema for metadata generation"""
    story_content: str = Field(..., description="Story content to generate metadata for")
    platform: str = Field(..., description="Target platform (youtube, instagram, tiktok, facebook)")

class MetadataResponse(BaseModel):
    """Response schema for metadata generation"""
    title: str
    description: str
    hashtags: List[str]
    platform: str

class StoryUpdate(BaseModel):
    """Schema for updating story data"""
    title: Optional[str] = Field(None, max_length=255)
    content: Optional[str] = None
    genre: Optional[str] = Field(None, max_length=100)
    theme: Optional[str] = Field(None, max_length=100)
    status: Optional[ContentStatus] = None


# Media Asset schemas
class MediaAssetRequest(BaseModel):
    """Request schema for media asset generation"""
    story_id: uuid.UUID
    asset_type: MediaType
    provider: Optional[str] = None
    generation_parameters: Optional[Dict[str, Any]] = None
    prompt_override: Optional[str] = Field(None, description="Override auto-generated prompt")


class MediaAssetResponse(BaseSchema):
    """Response schema for media asset data"""
    id: uuid.UUID
    story_id: uuid.UUID
    asset_type: MediaType
    file_path: str
    file_size: Optional[int]
    duration: Optional[float]
    metadata: Optional[Dict[str, Any]]
    provider: Optional[str]
    model_name: Optional[str]
    generation_parameters: Optional[Dict[str, Any]]
    generation_cost: float
    prompt_used: Optional[str]
    status: ContentStatus
    created_at: datetime
    updated_at: Optional[datetime]


# Video schemas
class VideoGenerationRequest(BaseModel):
    """Request schema for video generation"""
    story_id: uuid.UUID
    title: str = Field(..., max_length=255)
    description: Optional[str] = None
    format_type: str = Field(..., pattern="^(16:9|9:16|1:1)$", description="Video aspect ratio")
    resolution: Optional[str] = Field("1920x1080", description="Video resolution")
    target_platforms: List[PlatformType] = Field(default_factory=list)
    assembly_parameters: Optional[Dict[str, Any]] = None


class VideoResponse(BaseSchema):
    """Response schema for video data"""
    id: uuid.UUID
    story_id: uuid.UUID
    title: str
    description: Optional[str]
    file_path: str
    file_size: Optional[int]
    duration: float
    format_type: str
    resolution: str
    target_platforms: List[PlatformType]
    assembly_parameters: Optional[Dict[str, Any]]
    generation_cost: float
    status: ContentStatus
    created_at: datetime
    updated_at: Optional[datetime]


# Platform Upload schemas
class PlatformUploadRequest(BaseModel):
    """Request schema for platform upload"""
    video_id: uuid.UUID
    platform: PlatformType
    upload_title: Optional[str] = Field(None, max_length=255)
    upload_description: Optional[str] = None
    hashtags: Optional[List[str]] = Field(default_factory=list)
    upload_metadata: Optional[Dict[str, Any]] = None


class PlatformUploadResponse(BaseSchema):
    """Response schema for platform upload data"""
    id: uuid.UUID
    video_id: uuid.UUID
    platform: PlatformType
    platform_video_id: Optional[str]
    upload_url: Optional[str]
    upload_title: Optional[str]
    upload_description: Optional[str]
    hashtags: Optional[List[str]]
    upload_metadata: Optional[Dict[str, Any]]
    status: ContentStatus
    upload_started_at: Optional[datetime]
    upload_completed_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int


# Analytics schemas
class AnalyticsDataRequest(BaseModel):
    """Request schema for analytics data"""
    platform_upload_id: uuid.UUID
    views: int = Field(0, ge=0)
    likes: int = Field(0, ge=0)
    comments: int = Field(0, ge=0)
    shares: int = Field(0, ge=0)
    completion_rate: Optional[float] = Field(None, ge=0, le=100)
    ad_revenue: float = Field(0.0, ge=0)
    creator_fund_revenue: float = Field(0.0, ge=0)
    ctr: Optional[float] = Field(None, ge=0, le=100)
    engagement_rate: Optional[float] = Field(None, ge=0, le=100)
    raw_analytics: Optional[Dict[str, Any]] = None


class AnalyticsDataResponse(BaseSchema):
    """Response schema for analytics data"""
    id: uuid.UUID
    platform_upload_id: uuid.UUID
    views: int
    likes: int
    comments: int
    shares: int
    completion_rate: Optional[float]
    ad_revenue: float
    creator_fund_revenue: float
    total_revenue: float
    ctr: Optional[float]
    engagement_rate: Optional[float]
    data_collected_at: datetime
    created_at: datetime
    updated_at: Optional[datetime]
    raw_analytics: Optional[Dict[str, Any]]


# Model Configuration schemas
class ModelConfigurationRequest(BaseModel):
    """Request schema for model configuration"""
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    config_type: str = Field(..., pattern="^(story|tts|image|music|video)$")
    provider: Optional[str] = Field(None, max_length=50)
    model_name: Optional[str] = Field(None, max_length=100)
    parameters: Dict[str, Any] = Field(..., description="Model-specific parameters")
    cost_per_unit: Optional[float] = Field(None, ge=0)
    performance_metrics: Optional[Dict[str, Any]] = None
    version: str = Field(..., max_length=50)
    is_active: bool = Field(True)
    is_default: bool = Field(False)
    ab_test_group: Optional[str] = Field(None, max_length=50)
    traffic_percentage: float = Field(100.0, ge=0, le=100)


class ModelConfigurationResponse(BaseSchema):
    """Response schema for model configuration"""
    id: uuid.UUID
    name: str
    description: Optional[str]
    config_type: str
    provider: Optional[str]
    model_name: Optional[str]
    parameters: Dict[str, Any]
    cost_per_unit: Optional[float]
    performance_metrics: Optional[Dict[str, Any]]
    version: str
    is_active: bool
    is_default: bool
    ab_test_group: Optional[str]
    traffic_percentage: float
    created_at: datetime
    updated_at: Optional[datetime]


# Pagination and filtering schemas
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Items per page")


class StoryFilters(BaseModel):
    """Filtering parameters for stories"""
    status: Optional[ContentStatus] = None
    genre: Optional[str] = None
    theme: Optional[str] = None
    llm_provider: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


class MediaAssetFilters(BaseModel):
    """Filtering parameters for media assets"""
    story_id: Optional[uuid.UUID] = None
    asset_type: Optional[MediaType] = None
    status: Optional[ContentStatus] = None
    provider: Optional[str] = None


class VideoFilters(BaseModel):
    """Filtering parameters for videos"""
    story_id: Optional[uuid.UUID] = None
    status: Optional[ContentStatus] = None
    format_type: Optional[str] = None
    target_platform: Optional[PlatformType] = None


class PlatformUploadFilters(BaseModel):
    """Filtering parameters for platform uploads"""
    video_id: Optional[uuid.UUID] = None
    platform: Optional[PlatformType] = None
    status: Optional[ContentStatus] = None


# Response wrappers
class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., pattern="^(healthy|unhealthy)$")
    service: str
    timestamp: datetime
    version: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None


# Validation helpers
class StoryResponseWithAssets(StoryResponse):
    """Story response with related media assets and videos"""
    media_assets: List[MediaAssetResponse] = Field(default_factory=list)
    videos: List[VideoResponse] = Field(default_factory=list)


class VideoResponseWithUploads(VideoResponse):
    """Video response with platform uploads"""
    platform_uploads: List[PlatformUploadResponse] = Field(default_factory=list)


class PlatformUploadResponseWithAnalytics(PlatformUploadResponse):
    """Platform upload response with analytics data"""
    analytics_data: List[AnalyticsDataResponse] = Field(default_factory=list)
"""SQLAlchemy models for the AI Story-to-Video Pipeline"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, JSON,
    ForeignKey, Enum, Index, UniqueConstraint, Computed
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from .database import Base


class ContentStatus(enum.Enum):
    """Status of content in the pipeline"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    MODERATION_PENDING = "moderation_pending"
    MODERATION_FAILED = "moderation_failed"
    READY_FOR_DISTRIBUTION = "ready_for_distribution"
    DISTRIBUTED = "distributed"


class ModerationStatus(enum.Enum):
    """Status of content moderation"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"
    MANUAL_REVIEW = "manual_review"


class ModerationType(enum.Enum):
    """Types of content for moderation"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class PlatformType(enum.Enum):
    """Supported social media platforms"""
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"


class MediaType(enum.Enum):
    """Types of media assets"""
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    MUSIC = "music"


class ContentModeration(Base):
    """Content moderation records"""
    __tablename__ = "content_moderations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(String(255), nullable=False)
    content_type = Column(Enum(ModerationType, name='moderationtype', values_callable=lambda x: [e.value for e in x]), nullable=False)
    status = Column(Enum(ModerationStatus, name='moderationstatus', values_callable=lambda x: [e.value for e in x]), nullable=False, default=ModerationStatus.PENDING)
    score = Column(Float, nullable=False, default=0.0)
    flags = Column(JSON, default=list)
    categories = Column(JSON, default=dict)
    recommendations = Column(JSON, default=list)
    requires_review = Column(Boolean, default=False)
    audit_trail = Column(JSON, default=dict)
    
    # User and platform context
    user_id = Column(String(255))
    platform = Column(String(50))
    moderation_level = Column(String(20), default="medium")
    
    # Review information
    reviewer_id = Column(String(255))
    review_notes = Column(Text)
    review_override_reason = Column(Text)
    reviewed_at = Column(DateTime(timezone=True))
    
    # Provider information
    provider = Column(String(50))
    provider_response = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_content_moderations_content_id', 'content_id'),
        Index('idx_content_moderations_status', 'status'),
        Index('idx_content_moderations_type', 'content_type'),
        Index('idx_content_moderations_requires_review', 'requires_review'),
        Index('idx_content_moderations_created_at', 'created_at'),
    )


class ManualReviewQueue(Base):
    """Manual review queue for flagged content"""
    __tablename__ = "manual_review_queue"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    moderation_id = Column(UUID(as_uuid=True), ForeignKey("content_moderations.id"), nullable=False)
    
    # Review details
    priority = Column(String(20), default="normal")  # low, normal, high, urgent
    assigned_to = Column(String(255))
    assigned_at = Column(DateTime(timezone=True))
    
    # Review status
    review_status = Column(String(20), default="pending")  # pending, in_review, completed
    review_deadline = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    moderation = relationship("ContentModeration")
    
    # Indexes
    __table_args__ = (
        Index('idx_manual_review_queue_moderation_id', 'moderation_id'),
        Index('idx_manual_review_queue_status', 'review_status'),
        Index('idx_manual_review_queue_priority', 'priority'),
        Index('idx_manual_review_queue_assigned_to', 'assigned_to'),
    )


class AuditLog(Base):
    """Audit logging for moderation decisions and actions"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Action details
    action_type = Column(String(50), nullable=False)  # moderation, review, override, etc.
    entity_type = Column(String(50), nullable=False)  # content, user, system, etc.
    entity_id = Column(String(255), nullable=False)
    
    # User context
    user_id = Column(String(255))
    user_role = Column(String(50))
    
    # Action data
    action_data = Column(JSON, nullable=False)
    previous_state = Column(JSON)
    new_state = Column(JSON)
    
    # Metadata
    ip_address = Column(String(45))
    user_agent = Column(Text)
    correlation_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_logs_action_type', 'action_type'),
        Index('idx_audit_logs_entity_type', 'entity_type'),
        Index('idx_audit_logs_entity_id', 'entity_id'),
        Index('idx_audit_logs_user_id', 'user_id'),
        Index('idx_audit_logs_created_at', 'created_at'),
        Index('idx_audit_logs_correlation_id', 'correlation_id'),
    )


class Story(Base):
    """Story content and metadata"""
    __tablename__ = "stories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    genre = Column(String(100))
    theme = Column(String(100))
    target_length = Column(Integer)  # Target length in words
    actual_length = Column(Integer)  # Actual length in words
    
    # Generation metadata
    llm_provider = Column(String(50))  # openai, anthropic, mistral
    model_name = Column(String(100))
    generation_parameters = Column(JSON)  # Store generation config
    generation_cost = Column(Float, default=0.0)
    
    # Status and timestamps
    status = Column(Enum(ContentStatus, name='contentstatus', values_callable=lambda x: [e.value for e in x]), default=ContentStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    media_assets = relationship("MediaAsset", back_populates="story", cascade="all, delete-orphan")
    videos = relationship("Video", back_populates="story", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_stories_status', 'status'),
        Index('idx_stories_created_at', 'created_at'),
        Index('idx_stories_genre', 'genre'),
    )


class MediaAsset(Base):
    """Generated media assets (audio, images, music)"""
    __tablename__ = "media_assets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id"), nullable=False)
    
    # Asset details
    asset_type = Column(Enum(MediaType, name='mediatype', values_callable=lambda x: [e.value for e in x]), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)  # Size in bytes
    duration = Column(Float)  # Duration in seconds (for audio/video)
    metadata_json = Column('metadata', JSON)  # Additional metadata (dimensions, format, etc.)
    
    # Generation details
    provider = Column(String(50))  # elevenlabs, openai, dall-e, etc.
    model_name = Column(String(100))
    generation_parameters = Column(JSON)
    generation_cost = Column(Float, default=0.0)
    prompt_used = Column(Text)  # Prompt used for generation
    
    # Status and timestamps
    status = Column(Enum(ContentStatus, name='contentstatus', values_callable=lambda x: [e.value for e in x]), default=ContentStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    story = relationship("Story", back_populates="media_assets")
    
    # Indexes
    __table_args__ = (
        Index('idx_media_assets_story_id', 'story_id'),
        Index('idx_media_assets_type', 'asset_type'),
        Index('idx_media_assets_status', 'status'),
    )


class Video(Base):
    """Generated videos in different formats"""
    __tablename__ = "videos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    story_id = Column(UUID(as_uuid=True), ForeignKey("stories.id"), nullable=False)
    
    # Video details
    title = Column(String(255), nullable=False)
    description = Column(Text)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)  # Size in bytes
    duration = Column(Float, nullable=False)  # Duration in seconds
    
    # Format and platform details
    format_type = Column(String(20))  # 16:9, 9:16, 1:1
    resolution = Column(String(20))  # 1920x1080, 1080x1920, etc.
    target_platforms = Column(JSON)  # List of target platforms
    
    # Generation metadata
    assembly_parameters = Column(JSON)
    generation_cost = Column(Float, default=0.0)
    
    # Status and timestamps
    status = Column(Enum(ContentStatus, name='contentstatus', values_callable=lambda x: [e.value for e in x]), default=ContentStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    story = relationship("Story", back_populates="videos")
    platform_uploads = relationship("PlatformUpload", back_populates="video", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_videos_story_id', 'story_id'),
        Index('idx_videos_status', 'status'),
        Index('idx_videos_format', 'format_type'),
    )


class PlatformUpload(Base):
    """Platform upload tracking"""
    __tablename__ = "platform_uploads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)
    
    # Platform details
    platform = Column(Enum(PlatformType, name='platformtype', values_callable=lambda x: [e.value for e in x]), nullable=False)
    platform_video_id = Column(String(255))  # ID from the platform
    upload_url = Column(String(500))  # URL on the platform
    
    # Upload metadata
    upload_title = Column(String(255))
    upload_description = Column(Text)
    hashtags = Column(JSON)  # List of hashtags
    upload_metadata = Column('upload_metadata', JSON)  # Platform-specific metadata
    
    # Status and timestamps
    status = Column(Enum(ContentStatus, name='contentstatus', values_callable=lambda x: [e.value for e in x]), default=ContentStatus.PENDING)
    upload_started_at = Column(DateTime(timezone=True))
    upload_completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Error tracking
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Relationships
    video = relationship("Video", back_populates="platform_uploads")
    analytics_data = relationship("AnalyticsData", back_populates="platform_upload", cascade="all, delete-orphan")
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_platform_uploads_video_id', 'video_id'),
        Index('idx_platform_uploads_platform', 'platform'),
        Index('idx_platform_uploads_status', 'status'),
        UniqueConstraint('video_id', 'platform', name='uq_video_platform'),
    )


class AnalyticsData(Base):
    """Analytics data from platforms"""
    __tablename__ = "analytics_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_upload_id = Column(UUID(as_uuid=True), ForeignKey("platform_uploads.id"), nullable=False)
    
    # Engagement metrics
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    completion_rate = Column(Float)  # Percentage
    
    # Revenue metrics
    ad_revenue = Column(Float, default=0.0)
    creator_fund_revenue = Column(Float, default=0.0)
    total_revenue = Column(Float, Computed("COALESCE(ad_revenue, 0) + COALESCE(creator_fund_revenue, 0)"))
    
    # Performance metrics
    ctr = Column(Float)  # Click-through rate
    engagement_rate = Column(Float)
    
    # Data collection metadata
    data_collected_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Raw data from platform
    raw_analytics = Column(JSON)  # Store complete platform response
    
    # Relationships
    platform_upload = relationship("PlatformUpload", back_populates="analytics_data")
    
    # Indexes
    __table_args__ = (
        Index('idx_analytics_platform_upload_id', 'platform_upload_id'),
        Index('idx_analytics_collected_at', 'data_collected_at'),
        Index('idx_analytics_views', 'views'),
    )


class ModelConfiguration(Base):
    """Configuration for AI models and generation parameters"""
    __tablename__ = "model_configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration details
    name = Column(String(255), nullable=False)
    description = Column(Text)
    config_type = Column(String(50), nullable=False)  # story, tts, image, music, video
    
    # Configuration data
    provider = Column(String(50))  # openai, anthropic, elevenlabs, etc.
    model_name = Column(String(100))
    parameters = Column(JSON, nullable=False)  # Model-specific parameters
    
    # Cost and performance
    cost_per_unit = Column(Float)  # Cost per token, image, second, etc.
    performance_metrics = Column(JSON)  # Quality scores, speed, etc.
    
    # Version control
    version = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    
    # A/B testing
    ab_test_group = Column(String(50))
    traffic_percentage = Column(Float, default=100.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_model_configs_type', 'config_type'),
        Index('idx_model_configs_active', 'is_active'),
        Index('idx_model_configs_default', 'is_default'),
        UniqueConstraint('name', 'version', name='uq_config_name_version'),
    )
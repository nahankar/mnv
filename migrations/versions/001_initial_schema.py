"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2025-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Define ENUM types - SQLAlchemy will create them automatically when tables are created
    content_status_enum = sa.Enum('pending', 'processing', 'completed', 'failed', 'moderation_pending', 'moderation_failed', 'ready_for_distribution', 'distributed', name='contentstatus')
    platform_type_enum = sa.Enum('youtube', 'instagram', 'tiktok', 'facebook', name='platformtype')
    media_type_enum = sa.Enum('audio', 'image', 'video', 'music', name='mediatype')

    # Create stories table
    op.create_table('stories',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('genre', sa.String(length=100), nullable=True),
        sa.Column('theme', sa.String(length=100), nullable=True),
        sa.Column('target_length', sa.Integer(), nullable=True),
        sa.Column('actual_length', sa.Integer(), nullable=True),
        sa.Column('llm_provider', sa.String(length=50), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('generation_parameters', sa.JSON(), nullable=True),
        sa.Column('generation_cost', sa.Float(), nullable=True),
        sa.Column('status', content_status_enum, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_stories_status', 'stories', ['status'], unique=False)
    op.create_index('idx_stories_created_at', 'stories', ['created_at'], unique=False)
    op.create_index('idx_stories_genre', 'stories', ['genre'], unique=False)

    # Create media_assets table
    op.create_table('media_assets',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('story_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('asset_type', media_type_enum, nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('provider', sa.String(length=50), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('generation_parameters', sa.JSON(), nullable=True),
        sa.Column('generation_cost', sa.Float(), nullable=True),
        sa.Column('prompt_used', sa.Text(), nullable=True),
        sa.Column('status', content_status_enum, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['story_id'], ['stories.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_media_assets_story_id', 'media_assets', ['story_id'], unique=False)
    op.create_index('idx_media_assets_type', 'media_assets', ['asset_type'], unique=False)
    op.create_index('idx_media_assets_status', 'media_assets', ['status'], unique=False)

    # Create videos table
    op.create_table('videos',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('story_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('duration', sa.Float(), nullable=False),
        sa.Column('format_type', sa.String(length=20), nullable=True),
        sa.Column('resolution', sa.String(length=20), nullable=True),
        sa.Column('target_platforms', sa.JSON(), nullable=True),
        sa.Column('assembly_parameters', sa.JSON(), nullable=True),
        sa.Column('generation_cost', sa.Float(), nullable=True),
        sa.Column('status', content_status_enum, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['story_id'], ['stories.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_videos_story_id', 'videos', ['story_id'], unique=False)
    op.create_index('idx_videos_status', 'videos', ['status'], unique=False)
    op.create_index('idx_videos_format', 'videos', ['format_type'], unique=False)

    # Create platform_uploads table
    op.create_table('platform_uploads',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('platform', platform_type_enum, nullable=False),
        sa.Column('platform_video_id', sa.String(length=255), nullable=True),
        sa.Column('upload_url', sa.String(length=500), nullable=True),
        sa.Column('upload_title', sa.String(length=255), nullable=True),
        sa.Column('upload_description', sa.Text(), nullable=True),
        sa.Column('hashtags', sa.JSON(), nullable=True),
        sa.Column('upload_metadata', sa.JSON(), nullable=True),
        sa.Column('status', content_status_enum, nullable=True),
        sa.Column('upload_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('upload_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('video_id', 'platform', name='uq_video_platform')
    )
    op.create_index('idx_platform_uploads_video_id', 'platform_uploads', ['video_id'], unique=False)
    op.create_index('idx_platform_uploads_platform', 'platform_uploads', ['platform'], unique=False)
    op.create_index('idx_platform_uploads_status', 'platform_uploads', ['status'], unique=False)

    # Create analytics_data table
    op.create_table('analytics_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('platform_upload_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('views', sa.Integer(), nullable=True),
        sa.Column('likes', sa.Integer(), nullable=True),
        sa.Column('comments', sa.Integer(), nullable=True),
        sa.Column('shares', sa.Integer(), nullable=True),
        sa.Column('completion_rate', sa.Float(), nullable=True),
        sa.Column('ad_revenue', sa.Float(), nullable=True),
        sa.Column('creator_fund_revenue', sa.Float(), nullable=True),
        sa.Column('total_revenue', sa.Float(), nullable=True),
        sa.Column('ctr', sa.Float(), nullable=True),
        sa.Column('engagement_rate', sa.Float(), nullable=True),
        sa.Column('data_collected_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('raw_analytics', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['platform_upload_id'], ['platform_uploads.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_analytics_platform_upload_id', 'analytics_data', ['platform_upload_id'], unique=False)
    op.create_index('idx_analytics_collected_at', 'analytics_data', ['data_collected_at'], unique=False)
    op.create_index('idx_analytics_views', 'analytics_data', ['views'], unique=False)

    # Create model_configurations table
    op.create_table('model_configurations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('config_type', sa.String(length=50), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=False),
        sa.Column('cost_per_unit', sa.Float(), nullable=True),
        sa.Column('performance_metrics', sa.JSON(), nullable=True),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_default', sa.Boolean(), nullable=True),
        sa.Column('ab_test_group', sa.String(length=50), nullable=True),
        sa.Column('traffic_percentage', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', 'version', name='uq_config_name_version')
    )
    op.create_index('idx_model_configs_type', 'model_configurations', ['config_type'], unique=False)
    op.create_index('idx_model_configs_active', 'model_configurations', ['is_active'], unique=False)
    op.create_index('idx_model_configs_default', 'model_configurations', ['is_default'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('model_configurations')
    op.drop_table('analytics_data')
    op.drop_table('platform_uploads')
    op.drop_table('videos')
    op.drop_table('media_assets')
    op.drop_table('stories')
    
    # Drop enum types
    sa.Enum(name='contentstatus').drop(op.get_bind())
    sa.Enum(name='platformtype').drop(op.get_bind())
    sa.Enum(name='mediatype').drop(op.get_bind())
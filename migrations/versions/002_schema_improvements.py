"""Schema improvements: computed columns, FK constraints, and CHECK constraints

Revision ID: 002
Revises: 001
Create Date: 2025-08-29 06:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add CASCADE to foreign key constraints for better data integrity
    
    # Drop existing foreign keys and recreate with CASCADE
    op.drop_constraint('media_assets_story_id_fkey', 'media_assets', type_='foreignkey')
    op.create_foreign_key(
        'media_assets_story_id_fkey', 'media_assets', 'stories',
        ['story_id'], ['id'], ondelete='CASCADE'
    )
    
    op.drop_constraint('videos_story_id_fkey', 'videos', type_='foreignkey')
    op.create_foreign_key(
        'videos_story_id_fkey', 'videos', 'stories',
        ['story_id'], ['id'], ondelete='CASCADE'
    )
    
    op.drop_constraint('platform_uploads_video_id_fkey', 'platform_uploads', type_='foreignkey')
    op.create_foreign_key(
        'platform_uploads_video_id_fkey', 'platform_uploads', 'videos',
        ['video_id'], ['id'], ondelete='CASCADE'
    )
    
    op.drop_constraint('analytics_data_platform_upload_id_fkey', 'analytics_data', type_='foreignkey')
    op.create_foreign_key(
        'analytics_data_platform_upload_id_fkey', 'analytics_data', 'platform_uploads',
        ['platform_upload_id'], ['id'], ondelete='CASCADE'
    )
    
    # Add CHECK constraints for non-negative values
    op.create_check_constraint(
        'ck_stories_actual_length_positive',
        'stories',
        'actual_length IS NULL OR actual_length >= 0'
    )
    
    op.create_check_constraint(
        'ck_stories_generation_cost_positive',
        'stories',
        'generation_cost IS NULL OR generation_cost >= 0'
    )
    
    op.create_check_constraint(
        'ck_media_assets_file_size_positive',
        'media_assets',
        'file_size IS NULL OR file_size >= 0'
    )
    
    op.create_check_constraint(
        'ck_media_assets_duration_positive',
        'media_assets',
        'duration IS NULL OR duration >= 0'
    )
    
    op.create_check_constraint(
        'ck_media_assets_generation_cost_positive',
        'media_assets',
        'generation_cost IS NULL OR generation_cost >= 0'
    )
    
    op.create_check_constraint(
        'ck_videos_file_size_positive',
        'videos',
        'file_size IS NULL OR file_size >= 0'
    )
    
    op.create_check_constraint(
        'ck_videos_duration_positive',
        'videos',
        'duration >= 0'
    )
    
    op.create_check_constraint(
        'ck_videos_generation_cost_positive',
        'videos',
        'generation_cost IS NULL OR generation_cost >= 0'
    )
    
    op.create_check_constraint(
        'ck_platform_uploads_retry_count_positive',
        'platform_uploads',
        'retry_count IS NULL OR retry_count >= 0'
    )
    
    op.create_check_constraint(
        'ck_analytics_views_positive',
        'analytics_data',
        'views IS NULL OR views >= 0'
    )
    
    op.create_check_constraint(
        'ck_analytics_likes_positive',
        'analytics_data',
        'likes IS NULL OR likes >= 0'
    )
    
    op.create_check_constraint(
        'ck_analytics_comments_positive',
        'analytics_data',
        'comments IS NULL OR comments >= 0'
    )
    
    op.create_check_constraint(
        'ck_analytics_shares_positive',
        'analytics_data',
        'shares IS NULL OR shares >= 0'
    )
    
    op.create_check_constraint(
        'ck_analytics_completion_rate_valid',
        'analytics_data',
        'completion_rate IS NULL OR (completion_rate >= 0 AND completion_rate <= 1)'
    )
    
    op.create_check_constraint(
        'ck_analytics_ad_revenue_positive',
        'analytics_data',
        'ad_revenue IS NULL OR ad_revenue >= 0'
    )
    
    op.create_check_constraint(
        'ck_analytics_creator_fund_revenue_positive',
        'analytics_data',
        'creator_fund_revenue IS NULL OR creator_fund_revenue >= 0'
    )
    
    op.create_check_constraint(
        'ck_analytics_ctr_valid',
        'analytics_data',
        'ctr IS NULL OR (ctr >= 0 AND ctr <= 1)'
    )
    
    op.create_check_constraint(
        'ck_analytics_engagement_rate_positive',
        'analytics_data',
        'engagement_rate IS NULL OR engagement_rate >= 0'
    )
    
    op.create_check_constraint(
        'ck_model_configs_cost_per_unit_positive',
        'model_configurations',
        'cost_per_unit IS NULL OR cost_per_unit >= 0'
    )
    
    op.create_check_constraint(
        'ck_model_configs_traffic_percentage_valid',
        'model_configurations',
        'traffic_percentage IS NULL OR (traffic_percentage >= 0 AND traffic_percentage <= 100)'
    )
    
    # Replace total_revenue column with computed column
    op.drop_column('analytics_data', 'total_revenue')
    op.add_column('analytics_data', 
        sa.Column('total_revenue', sa.Float(), 
                 sa.Computed('COALESCE(ad_revenue, 0) + COALESCE(creator_fund_revenue, 0)'),
                 nullable=True)
    )
    
    # Add composite indexes for common query patterns
    op.create_index(
        'idx_stories_status_created_at', 'stories', 
        ['status', 'created_at'], unique=False
    )
    
    op.create_index(
        'idx_media_assets_story_status', 'media_assets',
        ['story_id', 'status'], unique=False
    )
    
    op.create_index(
        'idx_videos_story_status', 'videos',
        ['story_id', 'status'], unique=False
    )
    
    op.create_index(
        'idx_platform_uploads_status_created_at', 'platform_uploads',
        ['status', 'created_at'], unique=False
    )


def downgrade() -> None:
    # Drop composite indexes
    op.drop_index('idx_platform_uploads_status_created_at', table_name='platform_uploads')
    op.drop_index('idx_videos_story_status', table_name='videos')
    op.drop_index('idx_media_assets_story_status', table_name='media_assets')
    op.drop_index('idx_stories_status_created_at', table_name='stories')
    
    # Revert total_revenue to regular column
    op.drop_column('analytics_data', 'total_revenue')
    op.add_column('analytics_data', sa.Column('total_revenue', sa.Float(), nullable=True))
    
    # Drop CHECK constraints
    op.drop_constraint('ck_model_configs_traffic_percentage_valid', 'model_configurations')
    op.drop_constraint('ck_model_configs_cost_per_unit_positive', 'model_configurations')
    op.drop_constraint('ck_analytics_engagement_rate_positive', 'analytics_data')
    op.drop_constraint('ck_analytics_ctr_valid', 'analytics_data')
    op.drop_constraint('ck_analytics_creator_fund_revenue_positive', 'analytics_data')
    op.drop_constraint('ck_analytics_ad_revenue_positive', 'analytics_data')
    op.drop_constraint('ck_analytics_completion_rate_valid', 'analytics_data')
    op.drop_constraint('ck_analytics_shares_positive', 'analytics_data')
    op.drop_constraint('ck_analytics_comments_positive', 'analytics_data')
    op.drop_constraint('ck_analytics_likes_positive', 'analytics_data')
    op.drop_constraint('ck_analytics_views_positive', 'analytics_data')
    op.drop_constraint('ck_platform_uploads_retry_count_positive', 'platform_uploads')
    op.drop_constraint('ck_videos_generation_cost_positive', 'videos')
    op.drop_constraint('ck_videos_duration_positive', 'videos')
    op.drop_constraint('ck_videos_file_size_positive', 'videos')
    op.drop_constraint('ck_media_assets_generation_cost_positive', 'media_assets')
    op.drop_constraint('ck_media_assets_duration_positive', 'media_assets')
    op.drop_constraint('ck_media_assets_file_size_positive', 'media_assets')
    op.drop_constraint('ck_stories_generation_cost_positive', 'stories')
    op.drop_constraint('ck_stories_actual_length_positive', 'stories')
    
    # Revert foreign key constraints to original (without CASCADE)
    op.drop_constraint('analytics_data_platform_upload_id_fkey', 'analytics_data', type_='foreignkey')
    op.create_foreign_key(
        'analytics_data_platform_upload_id_fkey', 'analytics_data', 'platform_uploads',
        ['platform_upload_id'], ['id']
    )
    
    op.drop_constraint('platform_uploads_video_id_fkey', 'platform_uploads', type_='foreignkey')
    op.create_foreign_key(
        'platform_uploads_video_id_fkey', 'platform_uploads', 'videos',
        ['video_id'], ['id']
    )
    
    op.drop_constraint('videos_story_id_fkey', 'videos', type_='foreignkey')
    op.create_foreign_key(
        'videos_story_id_fkey', 'videos', 'stories',
        ['story_id'], ['id']
    )
    
    op.drop_constraint('media_assets_story_id_fkey', 'media_assets', type_='foreignkey')
    op.create_foreign_key(
        'media_assets_story_id_fkey', 'media_assets', 'stories',
        ['story_id'], ['id']
    )
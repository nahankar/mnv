"""Add moderation schema

Revision ID: 003
Revises: 002_schema_improvements
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002_schema_improvements'
branch_labels = None
depends_on = None


def upgrade():
    # Create moderationtype enum
    moderationtype = postgresql.ENUM('text', 'image', 'audio', 'video', name='moderationtype')
    moderationtype.create(op.get_bind())
    
    # Create moderationstatus enum
    moderationstatus = postgresql.ENUM('pending', 'approved', 'rejected', 'flagged', 'manual_review', name='moderationstatus')
    moderationstatus.create(op.get_bind())
    
    # Create content_moderations table
    op.create_table('content_moderations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_id', sa.String(length=255), nullable=False),
        sa.Column('content_type', postgresql.ENUM('text', 'image', 'audio', 'video', name='moderationtype'), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'approved', 'rejected', 'flagged', 'manual_review', name='moderationstatus'), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('flags', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('categories', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('recommendations', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('requires_review', sa.Boolean(), nullable=True),
        sa.Column('audit_trail', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('platform', sa.String(length=50), nullable=True),
        sa.Column('moderation_level', sa.String(length=20), nullable=True),
        sa.Column('reviewer_id', sa.String(length=255), nullable=True),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('review_override_reason', sa.Text(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('provider', sa.String(length=50), nullable=True),
        sa.Column('provider_response', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for content_moderations
    op.create_index('idx_content_moderations_content_id', 'content_moderations', ['content_id'], unique=False)
    op.create_index('idx_content_moderations_status', 'content_moderations', ['status'], unique=False)
    op.create_index('idx_content_moderations_type', 'content_moderations', ['content_type'], unique=False)
    op.create_index('idx_content_moderations_requires_review', 'content_moderations', ['requires_review'], unique=False)
    op.create_index('idx_content_moderations_created_at', 'content_moderations', ['created_at'], unique=False)
    
    # Create manual_review_queue table
    op.create_table('manual_review_queue',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('moderation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('priority', sa.String(length=20), nullable=True),
        sa.Column('assigned_to', sa.String(length=255), nullable=True),
        sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('review_status', sa.String(length=20), nullable=True),
        sa.Column('review_deadline', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['moderation_id'], ['content_moderations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for manual_review_queue
    op.create_index('idx_manual_review_queue_moderation_id', 'manual_review_queue', ['moderation_id'], unique=False)
    op.create_index('idx_manual_review_queue_status', 'manual_review_queue', ['review_status'], unique=False)
    op.create_index('idx_manual_review_queue_priority', 'manual_review_queue', ['priority'], unique=False)
    op.create_index('idx_manual_review_queue_assigned_to', 'manual_review_queue', ['assigned_to'], unique=False)
    
    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('action_type', sa.String(length=50), nullable=False),
        sa.Column('entity_type', sa.String(length=50), nullable=False),
        sa.Column('entity_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('user_role', sa.String(length=50), nullable=True),
        sa.Column('action_data', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('previous_state', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('new_state', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('correlation_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for audit_logs
    op.create_index('idx_audit_logs_action_type', 'audit_logs', ['action_type'], unique=False)
    op.create_index('idx_audit_logs_entity_type', 'audit_logs', ['entity_type'], unique=False)
    op.create_index('idx_audit_logs_entity_id', 'audit_logs', ['entity_id'], unique=False)
    op.create_index('idx_audit_logs_user_id', 'audit_logs', ['user_id'], unique=False)
    op.create_index('idx_audit_logs_created_at', 'audit_logs', ['created_at'], unique=False)
    op.create_index('idx_audit_logs_correlation_id', 'audit_logs', ['correlation_id'], unique=False)


def downgrade():
    # Drop indexes
    op.drop_index('idx_audit_logs_correlation_id', table_name='audit_logs')
    op.drop_index('idx_audit_logs_created_at', table_name='audit_logs')
    op.drop_index('idx_audit_logs_user_id', table_name='audit_logs')
    op.drop_index('idx_audit_logs_entity_id', table_name='audit_logs')
    op.drop_index('idx_audit_logs_entity_type', table_name='audit_logs')
    op.drop_index('idx_audit_logs_action_type', table_name='audit_logs')
    
    op.drop_index('idx_manual_review_queue_assigned_to', table_name='manual_review_queue')
    op.drop_index('idx_manual_review_queue_priority', table_name='manual_review_queue')
    op.drop_index('idx_manual_review_queue_status', table_name='manual_review_queue')
    op.drop_index('idx_manual_review_queue_moderation_id', table_name='manual_review_queue')
    
    op.drop_index('idx_content_moderations_created_at', table_name='content_moderations')
    op.drop_index('idx_content_moderations_requires_review', table_name='content_moderations')
    op.drop_index('idx_content_moderations_type', table_name='content_moderations')
    op.drop_index('idx_content_moderations_status', table_name='content_moderations')
    op.drop_index('idx_content_moderations_content_id', table_name='content_moderations')
    
    # Drop tables
    op.drop_table('audit_logs')
    op.drop_table('manual_review_queue')
    op.drop_table('content_moderations')
    
    # Drop enums
    op.execute('DROP TYPE moderationstatus')
    op.execute('DROP TYPE moderationtype')

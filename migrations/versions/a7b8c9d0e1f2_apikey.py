"""ApiKey table for REST API authentication (Phase 7)

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-10-23
"""
from alembic import op
import sqlalchemy as sa


revision = 'a7b8c9d0e1f2'
down_revision = 'f6a7b8c9d0e1'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'ApiKey',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('key_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('name', sa.String(80)),
        sa.Column('is_active', sa.Boolean()),
        sa.Column('created_at', sa.DateTime()),
        sa.Column('last_used_at', sa.DateTime()),
    )


def downgrade():
    op.drop_table('ApiKey')

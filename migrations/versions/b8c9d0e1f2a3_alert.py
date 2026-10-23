"""Alert table for smart alerts (Phase 8)

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-10-23
"""
from alembic import op
import sqlalchemy as sa


revision = 'b8c9d0e1f2a3'
down_revision = 'a7b8c9d0e1f2'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'Alert',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('created_at', sa.DateTime()),
        sa.Column('alert_type', sa.String(30), nullable=False),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('details', sa.Text()),
        sa.Column('rollno', sa.Integer(), index=True),
        sa.Column('is_read', sa.Boolean()),
        sa.Column('is_dismissed', sa.Boolean()),
    )
    op.create_index('ix_Alert_rollno', 'Alert', ['rollno'])


def downgrade():
    op.drop_index('ix_Alert_rollno', table_name='Alert')
    op.drop_table('Alert')

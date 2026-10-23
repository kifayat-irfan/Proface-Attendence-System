"""FaceTemplate table for multi-angle enrollment (Phase 6)

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-10-23
"""
from alembic import op
import sqlalchemy as sa


revision = 'f6a7b8c9d0e1'
down_revision = 'e5f6a7b8c9d0'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'FaceTemplate',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('rollno', sa.Integer(), nullable=False, index=True),
        sa.Column('angle_label', sa.String(20)),
        sa.Column('embedding', sa.LargeBinary(), nullable=False),
        sa.Column('dim', sa.Integer(), nullable=False),
        sa.Column('engine', sa.String(20), nullable=False),
        sa.Column('quality_score', sa.Integer()),
        sa.Column('created_at', sa.DateTime()),
    )
    op.create_index('ix_FaceTemplate_rollno', 'FaceTemplate', ['rollno'])


def downgrade():
    op.drop_index('ix_FaceTemplate_rollno', table_name='FaceTemplate')
    op.drop_table('FaceTemplate')

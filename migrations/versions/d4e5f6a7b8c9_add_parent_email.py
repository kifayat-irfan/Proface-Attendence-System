"""Add parent_email to Student (for absence alerts)

Revision ID: d4e5f6a7b8c9
Revises: c3d8f1a2b4e5
Create Date: 2026-10-22
"""
from alembic import op
import sqlalchemy as sa


revision = 'd4e5f6a7b8c9'
down_revision = 'c3d8f1a2b4e5'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('Student', schema=None) as batch_op:
        batch_op.add_column(sa.Column('parent_email', sa.String(120), nullable=True))


def downgrade():
    with op.batch_alter_table('Student', schema=None) as batch_op:
        batch_op.drop_column('parent_email')

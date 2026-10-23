"""Embedding cache + audit log table

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-10-23
"""
from alembic import op
import sqlalchemy as sa


revision = 'e5f6a7b8c9d0'
down_revision = 'd4e5f6a7b8c9'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('Student', schema=None) as batch_op:
        batch_op.add_column(sa.Column('face_embedding', sa.LargeBinary(), nullable=True))
        batch_op.add_column(sa.Column('embedding_dim', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('embedding_engine', sa.String(20), nullable=True))
    op.create_table(
        'AuditLog',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('actor_email', sa.String(120)),
        sa.Column('actor_role', sa.String(20)),
        sa.Column('action', sa.String(80), nullable=False),
        sa.Column('details', sa.Text()),
        sa.Column('ip_address', sa.String(45)),
    )


def downgrade():
    op.drop_table('AuditLog')
    with op.batch_alter_table('Student', schema=None) as batch_op:
        batch_op.drop_column('embedding_engine')
        batch_op.drop_column('embedding_dim')
        batch_op.drop_column('face_embedding')

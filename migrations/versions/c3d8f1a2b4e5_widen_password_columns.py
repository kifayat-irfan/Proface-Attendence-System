"""Widen password columns for scrypt hashes

Revision ID: c3d8f1a2b4e5
Revises: a6729ceaeda6
Create Date: 2026-10-22

Widen Student.password and Faculty.password from VARCHAR(80) to
VARCHAR(255) so scrypt password hashes fit. Existing plaintext passwords
keep working: they are transparently re-hashed on the user's next login
(see security.verify_password).
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3d8f1a2b4e5'
down_revision = 'a6729ceaeda6'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('Student', schema=None) as batch_op:
        batch_op.alter_column('password',
                              existing_type=sa.String(80),
                              type_=sa.String(255),
                              existing_nullable=True)
    with op.batch_alter_table('Faculty', schema=None) as batch_op:
        batch_op.alter_column('password',
                              existing_type=sa.String(80),
                              type_=sa.String(255),
                              existing_nullable=True)


def downgrade():
    with op.batch_alter_table('Faculty', schema=None) as batch_op:
        batch_op.alter_column('password',
                              existing_type=sa.String(255),
                              type_=sa.String(80),
                              existing_nullable=True)
    with op.batch_alter_table('Student', schema=None) as batch_op:
        batch_op.alter_column('password',
                              existing_type=sa.String(255),
                              type_=sa.String(80),
                              existing_nullable=True)

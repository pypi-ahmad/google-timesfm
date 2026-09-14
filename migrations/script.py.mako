## Template Alembic renders via `alembic revision` into each new file under
## migrations/versions/ (see 0001_job_store.py for the result). Keep the
## revision/down_revision/branch_labels/depends_on variable names intact;
## Alembic reads them by name to build the migration chain.
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
"""

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
  ${upgrades if upgrades else "pass"}


def downgrade() -> None:
  ${downgrades if downgrades else "pass"}

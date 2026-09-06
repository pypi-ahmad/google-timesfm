"""Persistent records, jobs, ordered events, and a transactional outbox."""

import sqlalchemy as sa
from alembic import op

revision = "0001_job_store"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
  op.create_table(
    "records",
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("workspace_id", sa.String(100), nullable=False),
    sa.Column("kind", sa.String(40), nullable=False),
    sa.Column("name", sa.String(500), nullable=False),
    sa.Column("revision", sa.Integer(), nullable=False),
    sa.Column("payload", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.String(40), nullable=False),
  )
  op.create_index("ix_records_workspace_id", "records", ["workspace_id"])
  op.create_index("ix_records_kind", "records", ["kind"])
  op.create_table(
    "jobs",
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("workspace_id", sa.String(100), nullable=False),
    sa.Column("kind", sa.String(40), nullable=False),
    sa.Column("status", sa.String(20), nullable=False),
    sa.Column("stage", sa.String(100), nullable=False),
    sa.Column("spec", sa.JSON(), nullable=False),
    sa.Column("idempotency_key", sa.String(200), nullable=False),
    sa.Column("request_hash", sa.String(64), nullable=False),
    sa.Column("attempt", sa.Integer(), nullable=False),
    sa.Column("lease_until", sa.DateTime(timezone=True)),
    sa.Column("cancel_requested", sa.Boolean(), nullable=False),
    sa.Column("result_id", sa.String(32), sa.ForeignKey("records.id")),
    sa.Column("error", sa.JSON()),
    sa.Column("created_at", sa.String(40), nullable=False),
    sa.Column("updated_at", sa.String(40), nullable=False),
    sa.UniqueConstraint("workspace_id", "idempotency_key", name="uq_job_idempotency"),
  )
  op.create_index("ix_jobs_workspace_id", "jobs", ["workspace_id"])
  op.create_index("ix_jobs_status_lease", "jobs", ["status", "lease_until"])
  op.create_table(
    "job_events",
    sa.Column(
      "seq",
      sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
      primary_key=True,
      autoincrement=True,
    ),
    sa.Column("job_id", sa.String(32), sa.ForeignKey("jobs.id"), nullable=False),
    sa.Column("type", sa.String(40), nullable=False),
    sa.Column("status", sa.String(20), nullable=False),
    sa.Column("stage", sa.String(100), nullable=False),
    sa.Column("payload", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.String(40), nullable=False),
  )
  op.create_index("ix_job_events_job_id", "job_events", ["job_id"])
  op.create_table(
    "outbox",
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("payload", sa.JSON(), nullable=False),
    sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("dispatched_at", sa.DateTime(timezone=True)),
  )
  op.create_index("ix_outbox_pending", "outbox", ["dispatched_at", "available_at"])


def downgrade() -> None:
  for table in ("outbox", "job_events", "jobs", "records"):
    op.drop_table(table)

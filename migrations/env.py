"""Alembic entry point: the application settings own database configuration."""

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from timesfm_app.store import Base


def database_url():
  override = context.config.get_main_option("sqlalchemy.url")
  if override:
    url = make_url(override)
  else:
    from timesfm_app.config import get_settings

    url = make_url(get_settings().database_url)
  # Force the psycopg driver the same way store.py's Store does, so a plain
  # "postgresql://" URL (from settings or an -x sqlalchemy.url override)
  # resolves to the same driver Alembic and the application both use.
  return (
    url.set(drivername="postgresql+psycopg") if url.drivername == "postgresql" else url
  )


if context.is_offline_mode():
  context.configure(
    url=database_url(), target_metadata=Base.metadata, literal_binds=True
  )
  with context.begin_transaction():
    context.run_migrations()
else:
  engine = create_engine(database_url(), hide_parameters=True)
  with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
      context.run_migrations()
  engine.dispose()

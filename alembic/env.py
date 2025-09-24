import asyncio
from logging.config import fileConfig
import os
import sys

from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add the project root to the Python path
# This allows Alembic to find the `api` module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.models import Base

# Récupère l'URL de la base de données depuis les secrets (en production)
# ou utilise une URL locale par défaut (en développement).
env_db_url = os.getenv("DATABASE_URL")
if env_db_url:
    raw_db_url = env_db_url
else:
    raw_db_url = config.get_main_option("sqlalchemy.url") or "postgresql://postgres:bilelsf2001@localhost:5432/dog_posture_db"

# Assure que l'URL utilise le driver asynchrone `asyncpg`
if raw_db_url and raw_db_url.startswith("postgresql://"):
    DATABASE_URL = raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = raw_db_url

# Mettre à jour la configuration pour que le reste du script l'utilise
config.set_main_option("sqlalchemy.url", DATABASE_URL or "")

# Set the target metadata for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    db_url = config.get_main_option("sqlalchemy.url")
    if not db_url:
        raise ValueError("Database URL is not configured in alembic.ini or DATABASE_URL env var.")
        
    is_production_env = os.getenv("SPACE_ID") is not None or os.getenv("DATABASE_URL") is not None

    engine_args = {}
    if is_production_env:
        engine_args["connect_args"] = {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "server_settings": {
                "jit": "off"
            }
        }

    connectable = create_async_engine(
        db_url,
        **engine_args
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
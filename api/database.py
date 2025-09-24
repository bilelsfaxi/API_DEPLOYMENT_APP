# api/database.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import logging

logger = logging.getLogger(__name__)

# Utiliser la variable d'environnement pour la production, sinon une valeur par défaut pour le local.
env_db_url = os.getenv("DATABASE_URL")
raw_db_url = env_db_url if env_db_url else "postgresql://postgres:bilelsf2001@localhost:5432/dog_posture_db"

# Log de l'URL brute pour le débogage
logger.info(f"Raw DB URL (first 20 chars): {raw_db_url[:20]}...")

# Assure que l'URL utilise le driver asynchrone `asyncpg`
if raw_db_url and raw_db_url.startswith("postgresql://"):
    DATABASE_URL = raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = raw_db_url
logger.info(f"Final DB URL (first 20 chars): {DATABASE_URL[:20]}...")
 
is_production_env = os.getenv("SPACE_ID") is not None or env_db_url is not None

engine_args = {}
if is_production_env:
    logger.info("Production environment detected. Applying PgBouncer compatibility settings.")
    engine_args["connect_args"] = {
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
        "server_settings": {
            "jit": "off"
        }
    }

engine = create_async_engine(DATABASE_URL, **engine_args)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

async def get_db():
    async with SessionLocal() as session:
        yield session
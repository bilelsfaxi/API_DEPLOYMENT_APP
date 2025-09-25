# api/database.py
import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

# Récupère l'URL de la base de données depuis les secrets (en production)
# ou utilise une URL locale par défaut (en développement).
env_db_url = os.getenv("DATABASE_URL")
if env_db_url:
    logger.info("DATABASE_URL environment variable found.")
    raw_db_url = env_db_url
else:
    logger.warning("DATABASE_URL environment variable NOT found. Falling back to local DB.")
    raw_db_url = "postgresql://postgres:bilelsf2001@localhost:5432/dog_posture_db"

logger.info(f"Raw DB URL (first 20 chars): {raw_db_url[:20]}...")

# Assure que l'URL utilise le driver asynchrone `asyncpg`
if raw_db_url and raw_db_url.startswith("postgresql://"):
    DATABASE_URL = raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = raw_db_url
logger.info(f"Final DB URL (first 20 chars): {DATABASE_URL[:20]}...")
 
# Si on est dans un environnement de production (comme HF Spaces ou Supabase) qui utilise un pooler (pgbouncer),
# il faut désactiver le cache des "prepared statements" pour éviter les erreurs.
is_production_env = os.getenv("SPACE_ID") is not None or os.getenv("DATABASE_URL") is not None

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
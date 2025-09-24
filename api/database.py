# api/database.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import logging

logger = logging.getLogger(__name__)

# Utiliser la variable d'environnement pour la production, sinon une valeur par défaut pour le local.
env_db_url = os.getenv("DATABASE_URL")
raw_db_url = env_db_url if env_db_url else "postgresql://postgres:bilelsf2001@localhost:5432/dog_posture_db"

# Assure que l'URL utilise le driver asynchrone `asyncpg`
DATABASE_URL = raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
logger.info(f"Final DB URL (first 20 chars): {DATABASE_URL[:20]}...")
 
# Si on est dans un environnement de production (comme HF Spaces) qui utilise un pooler (pgbouncer),
# il faut désactiver le cache des "prepared statements" pour éviter les erreurs.
# La présence de la variable d'environnement est un bon indicateur.
connect_args = {"statement_cache_size": 0} if env_db_url else {}

engine = create_async_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

async def get_db():
    async with SessionLocal() as session:
        yield session
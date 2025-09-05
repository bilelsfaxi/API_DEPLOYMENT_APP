# api/database.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Mettez à jour avec vos propres informations de connexion
DATABASE_URL = "postgresql+asyncpg://postgres:bilelsf2001@localhost:5432/dog_posture_db"

engine = create_async_engine(DATABASE_URL , echo=True)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
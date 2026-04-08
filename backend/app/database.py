"""
database.py
-----------
Connecting to Supabase (PostgreSQL) via SQLAlchemy.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Copy .env.example → .env and fill it out.")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # checks if connection is alive before request
    pool_size=5,
    max_overflow=10,
    echo=False,           # echo=True → outputs SQL to console (for debugging)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """
    Session generator — dependency for FastAPI or manual use.

    Example:
        with SessionLocal() as db:
            db.query(Customer).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

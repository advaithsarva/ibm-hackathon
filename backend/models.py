from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum

from config import settings

Base = declarative_base()


class DisasterType(str, enum.Enum):
    EARTHQUAKE = "earthquake"
    FLOOD = "flood"
    HURRICANE = "hurricane"
    WILDFIRE = "wildfire"
    TSUNAMI = "tsunami"
    TORNADO = "tornado"
    DROUGHT = "drought"
    VOLCANIC = "volcanic"
    OTHER = "other"


class SeverityLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Disaster(Base):
    __tablename__ = "disasters"

    id = Column(Integer, primary_key=True, index=True)
    disaster_type = Column(Enum(DisasterType), nullable=False, index=True)
    severity = Column(Enum(SeverityLevel), nullable=False, index=True)
    location = Column(String(255), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    predicted_impact = Column(Text)
    affected_population = Column(Integer, default=0)
    status = Column(String(50), default="active", index=True)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Disaster(id={self.id}, type={self.disaster_type}, severity={self.severity}, location={self.location})>"


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

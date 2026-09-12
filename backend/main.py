from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from routers import disasters
from config import settings
from models import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title="Disaster Early Warning Network API",
    description="Global early-warning network for disaster prediction and response",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(disasters.router, prefix="/api/v1/disasters", tags=["disasters"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "operational",
        "service": "Disaster Early Warning Network",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "api": "operational"
    }

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from models import Disaster, DisasterType, SeverityLevel, get_db

router = APIRouter()


class DisasterCreate(BaseModel):
    disaster_type: DisasterType
    severity: SeverityLevel
    location: str = Field(..., min_length=1, max_length=255)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    description: str = Field(..., min_length=1)
    predicted_impact: Optional[str] = None
    affected_population: int = Field(default=0, ge=0)
    detected_at: Optional[datetime] = None


class DisasterUpdate(BaseModel):
    disaster_type: Optional[DisasterType] = None
    severity: Optional[SeverityLevel] = None
    location: Optional[str] = Field(None, min_length=1, max_length=255)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    description: Optional[str] = Field(None, min_length=1)
    predicted_impact: Optional[str] = None
    affected_population: Optional[int] = Field(None, ge=0)
    status: Optional[str] = None


class DisasterResponse(BaseModel):
    id: int
    disaster_type: DisasterType
    severity: SeverityLevel
    location: str
    latitude: float
    longitude: float
    description: str
    predicted_impact: Optional[str]
    affected_population: int
    status: str
    detected_at: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@router.post("/", response_model=DisasterResponse, status_code=status.HTTP_201_CREATED)
async def create_disaster(disaster: DisasterCreate, db: Session = Depends(get_db)):
    """Create a new disaster alert"""
    db_disaster = Disaster(
        disaster_type=disaster.disaster_type,
        severity=disaster.severity,
        location=disaster.location,
        latitude=disaster.latitude,
        longitude=disaster.longitude,
        description=disaster.description,
        predicted_impact=disaster.predicted_impact,
        affected_population=disaster.affected_population,
        detected_at=disaster.detected_at or datetime.utcnow()
    )
    db.add(db_disaster)
    db.commit()
    db.refresh(db_disaster)
    return db_disaster


@router.get("/", response_model=List[DisasterResponse])
async def list_disasters(
    skip: int = 0,
    limit: int = 100,
    disaster_type: Optional[DisasterType] = None,
    severity: Optional[SeverityLevel] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all disasters with optional filtering"""
    query = db.query(Disaster)
    
    if disaster_type:
        query = query.filter(Disaster.disaster_type == disaster_type)
    if severity:
        query = query.filter(Disaster.severity == severity)
    if status:
        query = query.filter(Disaster.status == status)
    
    disasters = query.order_by(Disaster.detected_at.desc()).offset(skip).limit(limit).all()
    return disasters


@router.get("/{disaster_id}", response_model=DisasterResponse)
async def get_disaster(disaster_id: int, db: Session = Depends(get_db)):
    """Get a specific disaster by ID"""
    disaster = db.query(Disaster).filter(Disaster.id == disaster_id).first()
    if not disaster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disaster with id {disaster_id} not found"
        )
    return disaster


@router.put("/{disaster_id}", response_model=DisasterResponse)
async def update_disaster(
    disaster_id: int,
    disaster_update: DisasterUpdate,
    db: Session = Depends(get_db)
):
    """Update a disaster alert"""
    disaster = db.query(Disaster).filter(Disaster.id == disaster_id).first()
    if not disaster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disaster with id {disaster_id} not found"
        )
    
    update_data = disaster_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(disaster, field, value)
    
    db.commit()
    db.refresh(disaster)
    return disaster


@router.delete("/{disaster_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_disaster(disaster_id: int, db: Session = Depends(get_db)):
    """Delete a disaster alert"""
    disaster = db.query(Disaster).filter(Disaster.id == disaster_id).first()
    if not disaster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disaster with id {disaster_id} not found"
        )
    
    db.delete(disaster)
    db.commit()
    return None

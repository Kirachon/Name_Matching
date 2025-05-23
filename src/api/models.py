"""
Pydantic models for API request/response validation.
"""

from typing import Dict, List, Optional, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator
from enum import Enum


class MatchClassificationEnum(str, Enum):
    """Match classification enumeration."""
    MATCH = "match"
    NON_MATCH = "non_match"
    MANUAL_REVIEW = "manual_review"


class NameMatchRequest(BaseModel):
    """Request model for single name matching."""
    name1: str = Field(..., description="First name to match", min_length=1, max_length=500)
    name2: str = Field(..., description="Second name to match", min_length=1, max_length=500)
    additional_fields1: Optional[Dict[str, Union[str, date]]] = Field(
        default=None, 
        description="Additional fields for first record (birthdate, province_name, etc.)"
    )
    additional_fields2: Optional[Dict[str, Union[str, date]]] = Field(
        default=None,
        description="Additional fields for second record (birthdate, province_name, etc.)"
    )
    match_threshold: Optional[float] = Field(
        default=None, 
        description="Custom match threshold (0.0-1.0)",
        ge=0.0, 
        le=1.0
    )
    non_match_threshold: Optional[float] = Field(
        default=None,
        description="Custom non-match threshold (0.0-1.0)",
        ge=0.0,
        le=1.0
    )


class NameMatchResponse(BaseModel):
    """Response model for name matching."""
    score: float = Field(..., description="Overall match score (0.0-1.0)")
    classification: MatchClassificationEnum = Field(..., description="Match classification")
    component_scores: Dict[str, float] = Field(..., description="Individual component scores")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class SystemStatus(BaseModel):
    """System status model."""
    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    database_connected: bool = Field(..., description="Database connection status")
    redis_connected: bool = Field(..., description="Redis connection status")
    gpu_available: bool = Field(..., description="GPU availability status")
    active_jobs: int = Field(..., description="Number of active jobs")
    total_matches_processed: int = Field(..., description="Total matches processed since startup")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    checks: Dict[str, bool] = Field(..., description="Individual health checks")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")


class Token(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class User(BaseModel):
    """User model."""
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(default=None, description="User email")
    full_name: Optional[str] = Field(default=None, description="User full name")
    is_active: bool = Field(default=True, description="User active status")
    permissions: List[str] = Field(default=[], description="User permissions")

"""
Main FastAPI application for Name Matching API.
"""

import time
from datetime import datetime
from typing import Dict, Any

try:
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    from .models import (
        NameMatchRequest, NameMatchResponse, SystemStatus, HealthResponse, Token
    )
    from .auth import authenticate_user, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

    # FastAPI app
    app = FastAPI(
        title="Name Matching API",
        description="High-performance name matching API for Filipino identity data",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global state
    app_state = {
        "start_time": datetime.utcnow(),
        "total_matches_processed": 0,
        "matcher": None
    }

    @app.on_event("startup")
    async def startup_event():
        """Initialize application on startup."""
        try:
            from ...name_matcher import NameMatcher
            app_state["matcher"] = NameMatcher()
        except Exception as e:
            print(f"Failed to initialize matcher: {e}")

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        checks = {
            "api": True,
            "matcher": app_state["matcher"] is not None,
        }
        
        status_value = "healthy" if all(checks.values()) else "degraded"
        
        return HealthResponse(
            status=status_value,
            checks=checks
        )

    @app.get("/status", response_model=SystemStatus, tags=["Health"])
    async def system_status():
        """Get detailed system status."""
        uptime = (datetime.utcnow() - app_state["start_time"]).total_seconds()
        
        return SystemStatus(
            status="running",
            version="1.0.0",
            uptime_seconds=uptime,
            database_connected=False,  # Would check actual DB
            redis_connected=False,     # Would check actual Redis
            gpu_available=False,       # Would check actual GPU
            active_jobs=0,
            total_matches_processed=app_state["total_matches_processed"]
        )

    @app.post("/match/names", response_model=NameMatchResponse, tags=["Matching"])
    async def match_names(request: NameMatchRequest):
        """Match two names and return similarity score."""
        start_time = time.time()
        
        try:
            matcher = app_state["matcher"]
            if not matcher:
                raise HTTPException(status_code=500, detail="Matcher not initialized")
            
            # Perform matching
            score, classification, component_scores = matcher.match_names(
                request.name1,
                request.name2,
                request.additional_fields1,
                request.additional_fields2
            )
            
            processing_time = (time.time() - start_time) * 1000
            app_state["total_matches_processed"] += 1
            
            return NameMatchResponse(
                score=score,
                classification=classification.value,
                component_scores=component_scores,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Name matching failed: {str(e)}")

else:
    # Fallback when FastAPI is not available
    class MockApp:
        def __init__(self):
            self.title = "Name Matching API (FastAPI not available)"
    
    app = MockApp()

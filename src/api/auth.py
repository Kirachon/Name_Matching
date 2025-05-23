"""
Authentication and authorization for the API.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

from .models import User, Token

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Security scheme
security = HTTPBearer()

if JWT_AVAILABLE:
    # Password hashing
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # Mock user database (replace with real database in production)
    fake_users_db = {
        "admin": {
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "System Administrator",
            "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
            "is_active": True,
            "permissions": ["read", "write", "admin"]
        }
    }

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_user(username: str) -> Optional[dict]:
        """Get user from database."""
        return fake_users_db.get(username)

    def authenticate_user(username: str, password: str) -> Optional[dict]:
        """Authenticate a user."""
        user = get_user(username)
        if not user or not verify_password(password, user["hashed_password"]):
            return None
        return user

    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
        """Verify and decode JWT token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
        
        user = get_user(username)
        if user is None:
            raise credentials_exception
        
        return user

else:
    # Fallback implementations when JWT is not available
    def authenticate_user(username: str, password: str) -> Optional[dict]:
        return None
    
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        return "jwt-not-available"
    
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication not available"
        )


def get_current_user(user: dict = Depends(verify_token)) -> User:
    """Get current authenticated user."""
    return User(
        username=user["username"],
        email=user.get("email"),
        full_name=user.get("full_name"),
        is_active=user.get("is_active", True),
        permissions=user.get("permissions", [])
    )

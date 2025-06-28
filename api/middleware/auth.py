"""
Authentication and authorization middleware for the Ragamala painting generation API.
Implements JWT-based authentication, API key validation, and role-based access control.
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator

from ...src.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Security configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Should be loaded from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
API_KEY_EXPIRE_DAYS = 365

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
security = HTTPBearer(auto_error=False)

# User roles and permissions
class UserRole:
    """User role definitions."""
    
    ADMIN = "admin"
    USER = "user"
    DEVELOPER = "developer"
    VIEWER = "viewer"

class Permission:
    """Permission definitions."""
    
    GENERATE_IMAGE = "generate_image"
    BATCH_GENERATE = "batch_generate"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    ADMIN_ACCESS = "admin_access"
    API_ACCESS = "api_access"

# Role-permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.GENERATE_IMAGE,
        Permission.BATCH_GENERATE,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_USERS,
        Permission.ADMIN_ACCESS,
        Permission.API_ACCESS,
    ],
    UserRole.DEVELOPER: [
        Permission.GENERATE_IMAGE,
        Permission.BATCH_GENERATE,
        Permission.VIEW_ANALYTICS,
        Permission.API_ACCESS,
    ],
    UserRole.USER: [
        Permission.GENERATE_IMAGE,
        Permission.API_ACCESS,
    ],
    UserRole.VIEWER: [
        Permission.API_ACCESS,
    ],
}

# Pydantic models
class User(BaseModel):
    """User model."""
    
    id: Optional[int] = None
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: str = UserRole.USER
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    api_key: Optional[str] = None
    api_key_expires: Optional[datetime] = None
    
    @validator('role')
    def validate_role(cls, v):
        """Validate user role."""
        valid_roles = [UserRole.ADMIN, UserRole.USER, UserRole.DEVELOPER, UserRole.VIEWER]
        if v not in valid_roles:
            raise ValueError(f"Role must be one of: {valid_roles}")
        return v

class UserCreate(BaseModel):
    """User creation model."""
    
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: str = UserRole.USER
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")
        return v
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

class UserUpdate(BaseModel):
    """User update model."""
    
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

class Token(BaseModel):
    """Token model."""
    
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    scope: Optional[str] = None

class TokenData(BaseModel):
    """Token data model."""
    
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None
    permissions: List[str] = []
    exp: Optional[int] = None

class APIKey(BaseModel):
    """API key model."""
    
    key: str
    user_id: int
    name: str
    permissions: List[str] = []
    expires_at: Optional[datetime] = None
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True

# In-memory user store (replace with database in production)
fake_users_db = {
    "admin": {
        "id": 1,
        "username": "admin",
        "email": "admin@ragamala.ai",
        "full_name": "System Administrator",
        "role": UserRole.ADMIN,
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "is_active": True,
        "created_at": datetime.utcnow(),
        "api_key": "raga_admin_key_12345",
        "api_key_expires": datetime.utcnow() + timedelta(days=365)
    },
    "developer": {
        "id": 2,
        "username": "developer",
        "email": "dev@ragamala.ai",
        "full_name": "API Developer",
        "role": UserRole.DEVELOPER,
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "is_active": True,
        "created_at": datetime.utcnow(),
        "api_key": "raga_dev_key_67890",
        "api_key_expires": datetime.utcnow() + timedelta(days=365)
    },
    "user": {
        "id": 3,
        "username": "user",
        "email": "user@example.com",
        "full_name": "Regular User",
        "role": UserRole.USER,
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "is_active": True,
        "created_at": datetime.utcnow(),
        "api_key": "raga_user_key_11111",
        "api_key_expires": datetime.utcnow() + timedelta(days=365)
    }
}

# API key store (replace with database in production)
api_keys_db = {
    "raga_admin_key_12345": {
        "user_id": 1,
        "username": "admin",
        "role": UserRole.ADMIN,
        "permissions": ROLE_PERMISSIONS[UserRole.ADMIN],
        "expires_at": datetime.utcnow() + timedelta(days=365),
        "created_at": datetime.utcnow(),
        "is_active": True
    },
    "raga_dev_key_67890": {
        "user_id": 2,
        "username": "developer",
        "role": UserRole.DEVELOPER,
        "permissions": ROLE_PERMISSIONS[UserRole.DEVELOPER],
        "expires_at": datetime.utcnow() + timedelta(days=365),
        "created_at": datetime.utcnow(),
        "is_active": True
    },
    "raga_user_key_11111": {
        "user_id": 3,
        "username": "user",
        "role": UserRole.USER,
        "permissions": ROLE_PERMISSIONS[UserRole.USER],
        "expires_at": datetime.utcnow() + timedelta(days=365),
        "created_at": datetime.utcnow(),
        "is_active": True
    }
}

# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

# User management functions
def get_user(username: str) -> Optional[Dict]:
    """Get user by username."""
    return fake_users_db.get(username)

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID."""
    for user_data in fake_users_db.values():
        if user_data.get("id") == user_id:
            return user_data
    return None

def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email."""
    for user_data in fake_users_db.values():
        if user_data.get("email") == email:
            return user_data
    return None

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    
    # Update last login
    user["last_login"] = datetime.utcnow()
    
    return user

def create_user(user_data: UserCreate) -> Dict:
    """Create a new user."""
    # Check if user already exists
    if get_user(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    if get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Generate user ID
    user_id = max([user["id"] for user in fake_users_db.values()]) + 1
    
    # Create user
    new_user = {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "role": user_data.role,
        "hashed_password": get_password_hash(user_data.password),
        "is_active": True,
        "created_at": datetime.utcnow(),
        "api_key": generate_api_key(),
        "api_key_expires": datetime.utcnow() + timedelta(days=API_KEY_EXPIRE_DAYS)
    }
    
    fake_users_db[user_data.username] = new_user
    
    # Create API key entry
    api_keys_db[new_user["api_key"]] = {
        "user_id": user_id,
        "username": user_data.username,
        "role": user_data.role,
        "permissions": ROLE_PERMISSIONS.get(user_data.role, []),
        "expires_at": new_user["api_key_expires"],
        "created_at": datetime.utcnow(),
        "is_active": True
    }
    
    return new_user

# Token management functions
def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != token_type:
            return None
        
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        role: str = payload.get("role")
        exp: int = payload.get("exp")
        
        if username is None:
            return None
        
        # Get user permissions
        permissions = ROLE_PERMISSIONS.get(role, [])
        
        token_data = TokenData(
            username=username,
            user_id=user_id,
            role=role,
            permissions=permissions,
            exp=exp
        )
        
        return token_data
        
    except jwt.PyJWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None

# API key management
def generate_api_key() -> str:
    """Generate a new API key."""
    return f"raga_{secrets.token_urlsafe(32)}"

def verify_api_key(api_key: str) -> Optional[Dict]:
    """Verify API key and return associated user data."""
    key_data = api_keys_db.get(api_key)
    
    if not key_data:
        return None
    
    # Check if key is active
    if not key_data.get("is_active", False):
        return None
    
    # Check expiration
    expires_at = key_data.get("expires_at")
    if expires_at and datetime.utcnow() > expires_at:
        return None
    
    # Update last used timestamp
    key_data["last_used"] = datetime.utcnow()
    
    return key_data

def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    if api_key in api_keys_db:
        api_keys_db[api_key]["is_active"] = False
        return True
    return False

# Permission checking
def has_permission(user_role: str, required_permission: str) -> bool:
    """Check if user role has required permission."""
    user_permissions = ROLE_PERMISSIONS.get(user_role, [])
    return required_permission in user_permissions

def check_permissions(required_permissions: List[str], user_permissions: List[str]) -> bool:
    """Check if user has all required permissions."""
    return all(perm in user_permissions for perm in required_permissions)

# Rate limiting utilities
def create_rate_limit_key(user_id: int, endpoint: str) -> str:
    """Create rate limiting key."""
    return f"rate_limit:{user_id}:{endpoint}"

def check_rate_limit(user_id: int, endpoint: str, limit: int, window: int) -> bool:
    """Check rate limit for user and endpoint."""
    # This is a simplified implementation
    # In production, use Redis with sliding window
    key = create_rate_limit_key(user_id, endpoint)
    
    # For now, always allow (implement Redis-based rate limiting in production)
    return True

# Authentication dependencies
async def get_current_user_from_token(token: str = Depends(oauth2_scheme)) -> Optional[User]:
    """Get current user from JWT token."""
    if not token:
        return None
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = verify_token(token)
    if token_data is None:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(**user)

async def get_current_user_from_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """Get current user from API key."""
    if not credentials:
        return None
    
    if credentials.scheme.lower() != "bearer":
        return None
    
    api_key_data = verify_api_key(credentials.credentials)
    if not api_key_data:
        return None
    
    user = get_user_by_id(api_key_data["user_id"])
    if not user:
        return None
    
    return User(**user)

async def get_current_user(
    user_from_token: Optional[User] = Depends(get_current_user_from_token),
    user_from_api_key: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """Get current user from either JWT token or API key."""
    user = user_from_token or user_from_api_key
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

# Permission-based dependencies
def require_permission(permission: str):
    """Dependency factory for requiring specific permissions."""
    async def permission_dependency(current_user: User = Depends(get_current_active_user)):
        if not has_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_dependency

def require_permissions(permissions: List[str]):
    """Dependency factory for requiring multiple permissions."""
    async def permissions_dependency(current_user: User = Depends(get_current_active_user)):
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        if not check_permissions(permissions, user_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permissions {permissions} required"
            )
        return current_user
    
    return permissions_dependency

def require_role(role: str):
    """Dependency factory for requiring specific role."""
    async def role_dependency(current_user: User = Depends(get_current_active_user)):
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    
    return role_dependency

# Main authentication function for API endpoints
async def verify_api_key(credentials: HTTPAuthorizationCredentials) -> Dict:
    """Verify API key for endpoint access."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key_data = verify_api_key(credentials.credentials)
    if not api_key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user has API access permission
    if Permission.API_ACCESS not in api_key_data.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API access not permitted"
        )
    
    return api_key_data

# Security utilities
def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)

def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify HMAC signature."""
    expected_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)

# Session management
class SessionManager:
    """Simple session manager (use Redis in production)."""
    
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, user_id: int, session_data: Dict) -> str:
        """Create a new session."""
        session_id = generate_secure_token()
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "data": session_data
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data."""
        session = self.sessions.get(session_id)
        if session:
            session["last_accessed"] = datetime.utcnow()
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self.sessions.pop(session_id, None) is not None
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session["last_accessed"] < cutoff
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)

# Global session manager instance
session_manager = SessionManager()

# Audit logging
def log_authentication_event(event_type: str, user_id: Optional[int], details: Dict):
    """Log authentication events for audit purposes."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "details": details
    }
    
    # In production, send to audit log service
    logger.info(f"Auth event: {log_entry}")

# Initialize default admin user if not exists
def initialize_default_users():
    """Initialize default users if they don't exist."""
    if not fake_users_db:
        # Create default admin user
        admin_user = UserCreate(
            username="admin",
            email="admin@ragamala.ai",
            password="ChangeMe123!",
            full_name="System Administrator",
            role=UserRole.ADMIN
        )
        create_user(admin_user)
        logger.info("Created default admin user")

# Call initialization
initialize_default_users()

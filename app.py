from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Optional
from pydantic import BaseModel
from transformers import AutoModelForObjectDetection, DetrImageProcessor
from PIL import Image
import io
import torch
import logging
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; change to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods; adjust as needed
    allow_headers=["*"],  # Allows all headers; adjust as needed
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# JWT settings
SECRET_KEY = "cc3527750b28a2d3731da6ef9cf894973d76ad767dfa1f8a14a1c9c542a56920"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simulated user database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "testuser@example.com",
        "hashed_password": pwd_context.hash("testpassword"),  # hashed password
        "disabled": False,
    },

    # You can add more users here
    "newuser": {
        "username": "newuser",
        "full_name": "New User",
        "email": "newuser@example.com",
        "hashed_password": pwd_context.hash("newpassword"),  # Add a new user with a different password
        "disabled": False,
    },

    # You can add more users here
    "justine": {
        "username": "justine",
        "full_name": "justine kemhe",
        "email": "justine kemhe",
        "hashed_password": pwd_context.hash("justine"),  # Add a new user with a different password
        "disabled": False,
    }
}

# Hugging Face model and processor initialization
token = "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"
model_name = "smutuvi/flower_count_model"

model = AutoModelForObjectDetection.from_pretrained(model_name, token=token)
processor = DetrImageProcessor.from_pretrained(model_name, token=token)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None  # Use Optional

class User(BaseModel):
    username: str
    full_name: Optional[str] = None  # Use Optional
    email: Optional[str] = None  # Use Optional
    disabled: Optional[bool] = None  # Use Optional

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(fake_db, username: str, password: str):
    user = fake_db.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return UserInDB(**user)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):  # Use Optional
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    return UserInDB(**user)

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

class ImageResponse(BaseModel):
    filename: str
    count: int
    error: str = None

@app.post("/batch_predict/")
async def batch_predict(files: List[UploadFile] = File(...), current_user: User = Depends(get_current_user)):
    results = []
    for file in files:
        try:
            logging.info(f"Processing file: {file.filename}")
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Example logic to count flowers based on bounding boxes
            bounding_boxes = outputs.pred_boxes
            count = len(bounding_boxes)  # Count of detected objects
            
            results.append({"filename": file.filename, "count": count})
        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "error": str(e)})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

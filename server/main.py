import sys
from fastapi import FastAPI #may be cooked
from pydantic import BaseModel
import random
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - dev env without httpx
    httpx = None  # type: ignore

app = FastAPI()

# Allow CORS for local UI/dev servers 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# call llm stuff here



# current_number: Optional[int] = None


# class NumberUpdate(BaseModel):
#     value: int


# @app.get("/number")
# async def get_number():
#     global current_number
#     if current_number is None:
#         try:
#             current_number = await ask_ollama_for_number()
#         except Exception:
#             current_number = random.randint(1, 10)
#     return {"value": current_number}


# @app.post("/number")
# async def set_number(update: NumberUpdate):
#     global current_number
#     current_number = update.value
#     return {"value": current_number}


# async def ensure_ollama_model(base_url: str, model: str) -> None:
#     if httpx is None:
#         return
#     async with httpx.AsyncClient(timeout=None) as client:
#         # Pull model (no stream to wait until done)
#         try:
#             resp = await client.post(
#                 f"{base_url}/api/pull",
#                 json={"model": model, "stream": False},
#             )
#             resp.raise_for_status()
#         except Exception:
#             # Ignore pull failures (may already exist)
#             pass


# async def ask_ollama_for_number() -> int:
#     base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#     model = os.getenv("OLLAMA_MODEL", "gemma3")
#     prompt = (
#         "Pick a single integer between 1 and 10 inclusive. "
#         "Respond with only the number, no words."
#     )
#     if httpx is None:
#         return random.randint(1, 10)

#     # Ensure the model exists (pull if needed)
#     await ensure_ollama_model(base_url, model)

#     async with httpx.AsyncClient(timeout=30) as client:
#         resp = await client.post(
#             f"{base_url}/api/generate",
#             json={"model": model, "prompt": prompt, "stream": False},
#         )
#         resp.raise_for_status()
#         data = resp.json()
#         text = str(data.get("response", "")).strip()
#         try:
#             n = int(text)
#         except ValueError:
#             # Fallback: extract any digits
#             digits = "".join(ch for ch in text if ch.isdigit())
#             n = int(digits) if digits else random.randint(1, 10)
#         # Clamp to [1,10]
#         return max(1, min(10, n))


# @app.post("/number/generate")
# async def generate_number():
#     global current_number
#     try:
#         n = await ask_ollama_for_number()
#         current_number = n
#         return {"value": current_number, "source": "ollama"}
#     except Exception as e:
#         # Fall back to random if ollama fails
#         current_number = random.randint(1, 10)
#         return {"value": current_number, "source": "fallback", "error": str(e)}


# @app.on_event("startup")
# async def init_number_from_ollama():
#     # Try to set initial number from ollama; ignore errors
#     try:
#         await generate_number()
#     except Exception:
#         pass

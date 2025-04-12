from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# ✅ Allow GitHub Pages specifically
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://papgy.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load text-generation model
generator = pipeline("text-generation", model="distilgpt2")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: PromptRequest):
    result = generator(
        request.prompt,
        max_length=100,
        do_sample=True,
        truncation=True,
        pad_token_id=50256
    )
    return {"output": result[0]["generated_text"]}

@app.get("/")
def root():
    return {"message": "App Builder API running"}

# ✅ Optional: explicitly respond to OPTIONS preflight (only needed in some edge cases)
@app.options("/generate")
def options_generate(request: Request):
    return JSONResponse(content={"ok": True})

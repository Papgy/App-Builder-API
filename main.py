from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# ✅ CORS to allow from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://papgy.github.io'],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=['Access-Control-Allow-Origin', "*"],
)

# ✅ Load text gen model
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
        pad_token_id=50256  # optional, for GPT-2
    )
    return {"output": result[0]["generated_text"]}

@app.get("/")
def root():
    return {"message": "App Builder API running"}

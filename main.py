from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# ✅ Allow frontend (GitHub Pages or anywhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["https://papgy.github.io"] for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load text generation model
generator = pipeline("text-generation", model="distilgpt2")

# ✅ Input schema
class PromptRequest(BaseModel):
    prompt: str

# ✅ API route
@app.post("/generate")
def generate_text(request: PromptRequest):
    result = generator(request.prompt, max_length=100, do_sample=True)
    return {"output": result[0]["generated_text"]}

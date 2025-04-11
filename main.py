from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

generator = pipeline("text-generation", model="distilgpt2")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: PromptRequest):
    result = generator(request.prompt, max_length=100, do_sample=True)
    return {"output": result[0]["generated_text"]}

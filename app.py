from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.aba_service import route_user_input, prompt_safety_check

app = FastAPI(
    title="ABA Assist API",
    description="Backend service for ABA-inspired supportive response generation.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    user_input: str
    k: int = 3
    mode: str = "rag"
    use_llm: bool = True
    country: str = "US"


class SafetyRequest(BaseModel):
    user_input: str
    model: str = "llama3"


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if not request.user_input.strip():
        raise HTTPException(status_code=422, detail="user_input must not be empty")
    if request.mode not in {"rag", "baseline"}:
        raise HTTPException(status_code=422, detail="mode must be 'rag' or 'baseline'")

    return route_user_input(
        user_input=request.user_input,
        k=request.k,
        mode=request.mode,
        use_llm=request.use_llm,
        country=request.country,
    )


@app.post("/safety")
def safety_check(request: SafetyRequest) -> dict:
    if not request.user_input.strip():
        raise HTTPException(status_code=422, detail="user_input must not be empty")
    category, explanation = prompt_safety_check(request.user_input, model=request.model)
    return {"category": category, "explanation": explanation}

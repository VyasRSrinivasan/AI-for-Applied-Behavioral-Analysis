import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = os.getenv("DATASET_PATH", "./data/abaDatasetV1.csv")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.15"))

FALLBACK_RESPONSE = (
    "I'm having trouble generating a detailed response right now, "
    "but what you shared still matters. "
    "It might help to discuss it with someone you trust or a professional who can support you more deeply."
)


def load_data(path: str = DATASET_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    texts = (
        df["Antecedent"].fillna("")
        + " "
        + df["Behavior"].fillna("")
        + " "
        + df["Consequence"].fillna("")
    )
    return df, texts


def build_vectorizer(texts: pd.Series) -> Tuple[TfidfVectorizer, Any]:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


_df, _texts = load_data()
_vectorizer, _X = build_vectorizer(_texts)


def retrieve_top_k(user_input: str, k: int = 3) -> List[Dict[str, Any]]:
    vector = _vectorizer.transform([user_input])
    similarity = cosine_similarity(vector, _X).flatten()
    top_indices = similarity.argsort()[::-1][:k]
    results = _df.iloc[top_indices].copy()
    results["similarity"] = similarity[top_indices]
    return results.to_dict(orient="records")


def create_rag_prompt(user_input: str, retrieval: List[Dict[str, Any]]) -> Tuple[str, str]:
    blocks = []
    for row in retrieval:
        blocks.append(
            f"Antecedent: {row.get('Antecedent', '')}\n"
            f"Behavior: {row.get('Behavior', '')}\n"
            f"Supportive_Suggestion: {row.get('Consequence', '')}\n"
            f"Emotion_Tag: {row.get('Emotion_Tag', '')}\n"
        )

    rag_text = "\n---\n".join(blocks)
    system_msg = (
        "You are a supportive, non-clinical assistant inspired by Applied Behavior Analysis (ABA). "
        "You help users reflect on their emotions and behaviors using gentle, concrete, compassionate language. "
        "You DO NOT diagnose, DO NOT discuss treatment plans, and DO NOT claim to replace therapy. "
        "You always:\n"
        "- Acknowledge and validate the user's feelings\n"
        "- Normalize the experience when appropriate\n"
        "- Offer 1–3 small, practical, non-clinical suggestions\n"
        "- Avoid labels like 'disorder', 'patient', or 'treatment'\n"
    )
    user_msg = (
        f"The user shared this situation:\n\n" f"\"{user_input}\"\n\n"
        "Here are some similar situations and supportive suggestions from an ABA-style dataset:\n\n"
        f"{rag_text}\n\n"
        "Using the tone and structure of these examples as guidance, write ONE short, supportive response "
        "for the user. Acknowledge their feelings, reflect the essence of their situation, and offer 1–3 gentle, "
        "concrete ideas they can try. Keep it around 3–6 sentences. Do not mention ABA, datasets, or that you used examples."
    )
    return system_msg, user_msg


def ollama_generate(prompt: str, model: str = LLM_MODEL, timeout: int = 60) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as exc:
        print("[OLLAMA ERROR]", exc)
        return ""


def llm_rag_response(user_input: str, retrieval: List[Dict[str, Any]], model: str = LLM_MODEL) -> str:
    system_msg, user_msg = create_rag_prompt(user_input, retrieval)
    full_prompt = f"{system_msg}\n\n{user_msg}"
    text = ollama_generate(full_prompt, model=model)
    return text or FALLBACK_RESPONSE


SAFETY_CHECK_PROMPT = """
You are a safety and ethics checker for an emotionally supportive assistant.

Classify the user's input message as:
- SAFE      : emotionally intense is acceptable, but NO encouragement of harm or abuse
- UNSAFE    : encourages or normalizes manipulation, exploitation, abuse, or harm to others
- CRISIS    : suggests self-harm, suicidal thoughts, desire to die, or severe harm to self or others
- OFF-LIMITS : unrelated to the goal of this tool.

Rules:
- If the message is UNSAFE or CRISIS, explain why it is harmful and unsafe and gently encourage healthier or safer next steps
- Never ever give instructions on how to harm, manipulate, or self-harm. ONLY discourage those behaviors and promote safety & empathy.

Reply ONLY in this EXACT format:
CATEGORY: <SAFE or CRISIS or UNSAFE or OFF-LIMITS>
EXPLANATION: <2-4 sentences explaining your reasoning in a supportive and non-judgemental tone.
"""


def parse_safety_output(raw: str) -> Tuple[str, str]:
    category = "UNSURE"
    explanation = ""
    for line in raw.splitlines():
        if line.upper().startswith("CATEGORY:"):
            value = line.split(":", 1)[1].strip().upper()
            if value in {"SAFE", "CRISIS", "UNSAFE", "OFF-LIMITS"}:
                category = value
        elif line.upper().startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    if not explanation:
        explanation = (
            "This message may involve complex or sensitive content, and it's important to handle it carefully."
        )
    return category, explanation


def prompt_safety_check(user_input: str, model: str = LLM_MODEL) -> Tuple[str, str]:
    prompt = f"{SAFETY_CHECK_PROMPT}\n\nUser message:\n{user_input.strip()}\n\n"
    raw = ollama_generate(prompt, model=model)
    if not raw:
        return "SAFE", "Safety check unavailable — treating message as safe."
    return parse_safety_output(raw)


def get_crisis_resources(country: str = "US") -> str:
    if country.upper() == "US":
        return (
            "If you're in the U.S., you can call or text 988 (Suicide & Crisis Lifeline).\n"
            "If you are in immediate danger, call 911.\n"
            "If outside the U.S., contact your local emergency number or crisis line."
        )
    return "If you are in danger, contact your local emergency number or crisis service."


def get_grounding_exercise(kind: str = "box_breathing") -> str:
    exercises = {
        "box_breathing": (
            "Try box breathing:\n"
            "1) Inhale 4 seconds\n"
            "2) Hold 4 seconds\n"
            "3) Exhale 4 seconds\n"
            "4) Hold 4 seconds\n"
            "Repeat 4 times."
        ),
        "5-4-3-2-1": (
            "Try 5-4-3-2-1 grounding:\n"
            "5 things you SEE\n"
            "4 things you FEEL\n"
            "3 things you HEAR\n"
            "2 things you SMELL\n"
            "1 thing you TASTE"
        ),
    }
    return exercises.get(kind, exercises["box_breathing"])


def create_baseline_response(retrieval: List[Dict[str, Any]]) -> str:
    if not retrieval:
        return FALLBACK_RESPONSE
    lines = [
        "I hear you, and this sounds important.",
        "Here are a few suggestions drawn from similar situations:",
    ]
    for item in retrieval:
        lines.append(f"- {item.get('Consequence', '').strip()}")
    lines.append(
        "Try one small step at a time rather than trying everything at once."
    )
    return "\n".join(lines)


def create_rag_response(
    user_input: str,
    k: int = 3,
    min_similarity: float = MIN_SIMILARITY,
    use_llm: bool = True,
    model: str = LLM_MODEL,
) -> Tuple[str, List[Dict[str, Any]]]:
    retrieval = retrieve_top_k(user_input, k=k)
    top_similarity = max(item["similarity"] for item in retrieval) if retrieval else 0.0

    if top_similarity < min_similarity:
        return (
            "I’m not fully sure I understood this situation, but it sounds important. "
            "It might help to share it with someone you trust or a professional who can support you more deeply.",
            retrieval,
        )

    if use_llm:
        return llm_rag_response(user_input, retrieval, model=model), retrieval
    return create_baseline_response(retrieval), retrieval


def route_user_input(
    user_input: str,
    k: int = 3,
    mode: str = "rag",
    use_llm: bool = True,
    model: str = LLM_MODEL,
    country: str = "US",
) -> Dict[str, Any]:
    category, safety_explanation = prompt_safety_check(user_input, model=model)

    if category == "CRISIS":
        return {
            "route": "crisis",
            "category": category,
            "safety_explanation": safety_explanation,
            "response": f"{safety_explanation}\n{get_crisis_resources(country=country)}",
            "retrieval": [],
        }

    if category == "UNSAFE":
        return {
            "route": "unsafe",
            "category": category,
            "safety_explanation": safety_explanation,
            "response": (
                f"{safety_explanation}\n"
                "That sounds difficult, and I’m not able to support requests that could harm others. "
                "If you want, you can share more about how you are feeling right now."
            ),
            "retrieval": [],
        }

    if category == "OFF-LIMITS":
        return {
            "route": "off_limits",
            "category": category,
            "safety_explanation": safety_explanation,
            "response": (
                f"{safety_explanation}\n"
                "This tool is designed for reflecting on emotions, behavior, and support. "
                "Please try again with how you are feeling or what happened."
            ),
            "retrieval": [],
        }

    response, retrieval = create_rag_response(
        user_input,
        k=k,
        use_llm=(mode == "rag" and use_llm),
        model=model,
    )
    return {
        "route": "aba_rag" if mode == "rag" else "baseline",
        "category": category,
        "safety_explanation": safety_explanation,
        "response": response,
        "retrieval": retrieval,
    }

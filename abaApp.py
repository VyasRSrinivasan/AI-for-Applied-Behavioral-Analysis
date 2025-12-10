import streamlit as st
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
inputFile = "./data/abaDatasetV1.csv"
df = pd.read_csv(inputFile)
print(df.shape)


texts = df["Antecedent"].fillna("") + " " + df["Behavior"].fillna("") + " " + df["Consequence"].fillna("")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)



#RetrieveTopK
def retrieveTopK(userInput, k):
    vector = vectorizer.transform([userInput])
    similarity = cosine_similarity(vector, X).flatten()
    topIDX = similarity.argsort()[::-1][:k]
    
    results = df.iloc[topIDX].copy()
    results["similarity"] = similarity[topIDX]
    return results

#create RAG Prompt
OLLAMA_URL = "http://localhost:11434/api/generate"

def createRAGPromptForABA(userInput, retrieval):
    blocks=[]
    for _, row in retrieval.iterrows():
        block = (
            f"Antecedent: {row['Antecedent']}\n"
            f"Behavior: {row['Behavior']}\n"
            f"Supportive_Suggestion: {row['Consequence']}\n"
            f"Emotion_Tag: {row['Emotion_Tag']}\n"
        )
        blocks.append(block)

    ragText = "\n---\n".join(blocks)

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
        f"The user shared this situation:\n\n"
        f"\"{userInput}\"\n\n"
        "Here are some similar situations and supportive suggestions from an ABA-style dataset:\n\n"
        f"{ragText}\n\n"
        "Using the tone and structure of these examples as guidance, write ONE short, supportive response "
        "for the user. Acknowledge their feelings, reflect the essence of their situation, and offer 1–3 gentle, "
        "concrete ideas they can try. Keep it around 3–6 sentences. Do not mention ABA, datasets, or that you used examples."
    )

    return system_msg, user_msg


# LLM RAG Response
def llmABARAGResponse(userInput, retrieval, model="llama3"):
    system_msg, user_msg = createRAGPromptForABA(userInput, retrieval)
    
    
    full_prompt = system_msg + "\n\n" + user_msg

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,      # e.g. "llama3", "llama3:8b", etc.
                "prompt": full_prompt,
                "stream": False      # easier to handle than streaming
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns text in the "response" field
        return data.get("response", "").strip()

    except requests.exceptions.RequestException as e:
        # If Ollama isn't running or something goes wrong, fall back gracefully
        print("[Ollama ERROR]", e)
        return (
            "I'm having trouble generating a detailed response right now, "
            "but based on what you shared, your feelings make sense. "
            "It might help to talk through this with someone you trust or "
            "a professional who can support you more deeply."
        )
    
        
#Create RAG Prompt
def createRAGPromptForABA(userInput, retrieval):
    blocks=[]
    for _, row in retrieval.iterrows():
        block = (
            f"Antecedent: {row['Antecedent']}\n"
            f"Behavior: {row['Behavior']}\n"
            f"Supportive_Suggestion: {row['Consequence']}\n"
            f"Emotion_Tag: {row['Emotion_Tag']}\n"
        )
        blocks.append(block)

    ragText = "\n---\n".join(blocks)

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
        f"The user shared this situation:\n\n"
        f"\"{userInput}\"\n\n"
        "Here are some similar situations and supportive suggestions from an ABA-style dataset:\n\n"
        f"{ragText}\n\n"
        "Using the tone and structure of these examples as guidance, write ONE short, supportive response "
        "for the user. Acknowledge their feelings, reflect the essence of their situation, and offer 1–3 gentle, "
        "concrete ideas they can try. Keep it around 3–6 sentences. Do not mention ABA, datasets, or that you used examples."
    )

    return system_msg, user_msg    

#Safety Checks


SAFETY_CHECK_PROMPT = """
You are a safety and ethics checker for an emotionally supportive assistant.

Your job is to:
1. Classify the user's input message as:
- SAFE      : emotionally intense is acceptable, but NO encouragement of harm or abuse
- UNSAFE    : encourages or normalizes manipulation, exploitation, abuse, or non-crisis harm
- CRISIS    : suggests self-harm, suicidal thoughts, desire to tie, or severe harm to self or others

2. If the message is UNSAFE or CRISIS, explain why it is harmful and unsafe and gently encourage healthier or safer next steps
3. Never ever give instructions on how to harm, manipulate, or self-harm. ONLY discourage those behaviors and promote safety & empathy.
Reply in this EXACT format:

CATEGORY: <SAFE or CRISIS or UNSAFE>
EXPLANATION: <2-4 sentences explaning your reasoning in a supportive and non-judgemental tone.

"""
def promptSafetyCheckforLLM(userInput, model):
    prompt = SAFETY_CHECK_PROMPT + "\n\nUser message:\n" + userInput.strip() + "\n\n"
    
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print("[Llama safety ERROR]", e)
        # Fallback if something goes wrong
        return "SAFE", "Safety check unavailable — treating message as safe."
    
    lines = raw.splitlines()
    category = "UNSURE"
    explanation = ""
    
    for line in lines: 
        if line.upper().startswith("CATEGORY:"):
            if "SAFE" in line.upper():
                category = "SAFE"
            elif "CRISIS" in line.upper():
                category = "CRISIS"
            elif "UNSAFE" in line.upper():
                category = "UNSAFE"
        elif line.upper().startswith("EXPLANATION:"):
            explanation = line[len("EXPLANATION:")].strip()
            
    if not explanation:
        explanation = "This message may involve complex or sensitive content, and it's important to handle it very carefully."
    
    return category, explanation


# Create RAG Response
def createRAGResponse(userInput, k=3, minSimilarity=0.15, useLLM=True):

    retrieval = retrieveTopK(userInput, k)
    bestSimilarity = retrieval["similarity"].max()
    
    if bestSimilarity < minSimilarity:
        response = ("I’m not fully sure I understood this situation, "
            "but it sounds important. It might help to share it with someone you trust "
            "or a professional who can support you more deeply.")
        return response, retrieval
    if useLLM:
        response = llmABARAGResponse(userInput, retrieval)
    else:
        lines = []
        lines.append("It is understandable that you feel that way")
        lines.append("Here are a few ideas based on those who go through similar situations")

        for i, row in retrieval.iterrows():
            lines.append("-"+ row["Consequence"])

        lines.append("Don't pressure yourself to use all of these at once - even one small step makes a positive difference")
        response = "\n".join(lines)
        
    return response, retrieval

#Agentic AI

def abaWithAgenticAI(userInput, k=3):
    text = userInput.strip()
    category, safetyExplanation = promptSafetyCheckforLLM(userInput, model="llama3")
    
    
    if category == "CRISIS":
        response = (safetyExplanation + ""
                    "Thank you for reaching out! Unfortunately, I am NOT able to provide crisis support," ,
                    "but you should seek professional help from someone who can. ",
                    "Please consider reaching out to a trusted individual, a mental health professional, ",
                    "or a crisis line in your area"
                   )
        return {
            "route": "crisis",
            "User_Input": text,
            "Response": response,
            "Retrieved": df.iloc[[]].copy()
        }
    if category == "UNSAFE":
        response = (safetyExplanation + ""
                    "It sounds like you are going through a lot and have strong feelings" ,
                    "but using other people is NOT a healthy coping mechanism ",
                    "Healthy relationships should be built on mutual respect and empathy for one another",
                    "If you are open to sharing what caused you to feel this way, I am here to listen"
                   )
        return {
            "route": "unsafe",
            "User_Input": text,
            "Response": response,
            "Retrieved": df.iloc[[]].copy()
        }
    
    if category == "SAFE":
        response, retrieved = createRAGResponse(text, k=k, useLLM=True)
        
        return {
            "route": "aba_RAG",
            "User_Input": text,
            "Response": response,
            "Retrieved": retrieved
        }
    
    
    offLimits = ("I'm mainly designed to help with emotions, social situations, and behavioral triggers.", 
                "If you would like to share your reaction and how you feel, I would be more than happy to discuss that with you.")
    
    return {
            "route": "offLimits",
            "User_Input": text,
            "Response": offLimits,
            "Retrieved": df.iloc[[]].copy()
        }

st.set_page_config(
    page_title="ABA Agentic AI",
    #page_icon="🧠",
    layout="centered"
)

st.title("Agentic AI for Applied Behavior Analysis (ABA)")
st.markdown("A supportive emotional assistant inspired by Applied Behavioral Analysis (ABA).")
st.markdown("---")

st.divider()
if "chat" not in st.session_state:
    st.session_state.chat = []
if st.button("Clear Chat History"):
    st.session_state.chat = []
    st.experimental_rerun()


clean_chat = []
for item in st.session_state.chat:
    if isinstance(item, tuple) and len(item) == 2:
        clean_chat.append(item)
    else:
        clean_chat.append(("assistant", str(item)))

st.session_state.chat = clean_chat

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"**YOU:** {msg}")
    else:
        st.markdown(f"**AI:** {msg} ")
    st.markdown("------")



def sendMessage():
    text = st.session_state.userInput.strip()
    if not text:
        return
    st.session_state.chat.append(("user", text))

    result = abaWithAgenticAI(text)
    response = result.get("Response","")
    route = result.get("route", "unknown")

    st.session_state.chat.append(("assistant", f"[{route.upper()}] {response}"))

    st.session_state.userInput = ""

st.text_input("Enter your message", key="userInput")
st.button("Send", on_click=sendMessage)
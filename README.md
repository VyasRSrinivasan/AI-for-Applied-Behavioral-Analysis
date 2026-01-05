# AI for Applied Behavioral Analysis (ABA)


## Applied Behavioral Analysis (ABA) Fundamentals
Therapy based on the science of learning and behavior.

Positive reinforcement is one of the top strategies used in ABA.

#### Antecedent, Behavior, Consequence (A-B-Cs)
![alt text](./images/ABCforABADiagram.png)
_Antecedent_(**A**): what happens *BEFORE* a behavior occurs

_Behavior_(**B**): person's response or lack of response to the antecedent.

_Consequence_(**C**): what happens *AFTER* a behavior occurs

## Objective
Many people with autism and other neurodivergent traits have difficulty understanding subtle emotional or social cues, which results in anxiety, miscommunication, and stress.
There are many AI chatbots and emotional assistants out there, but only very few incorporate ABA — mostly to diagnose people who might be on the autism spectrum.

The goal is to develop emotionally aware AI tools grounded in *Applied Behavioral Analysis (ABA)* that helps individuals who are neurodivergent to better understand, manage, and respond to social & emotional triggers with clarity and compassion. This would be accomplished by identifying emotional triggers and providing supportive feedback using the ABA model.

### Why this project?
- Many people with autism have difficulty reading subtle social cues and emotionally loaded feedback, which can lead to rumination, miscommunication, and difficulty moving on.
- Explores how an AI assistant can provide structured, non-clinical support** using an ABA-inspired A-B-C framing (Antecedent -> Behavior -> Consequence)

## Users
- Individuals with Autism who are self-aware and want to improve
- ABA educators and therapists looking into digital assistive tools
- Parents and caregivers who want to help the child on the spectrum with emotional self-regulation

## Features
### Baseline TF-IDF Retriever (no LLMs)
- Utilizes TF-IDF to find the most similar A-B-C examples
- Uses cosine similarity to retrieve the top-K most simlar examples
- Produces a simple, deterministic response from the retrieved "Consequence" suggestions

Pipeline:

1. **Input**
2. **Text Vectorization**
3. **Similarity Matching**
4. **ABA Knowledge Base**
5. **Supportive Response Generation**

Goal:
Provides a standard deterministic baseline model for retrieval with NO involvement of LLMs.
    
### RAG with LLMs
- Retrieves top-K similar A-B-C examples using TF-IDF paired with cosine similarity
- Constructs a grounded prompt using the retrieved examples (A, B, suggestion, emotion tag)
- Passes them into a local LLM (Ollama & Llama 3) to generate a supportive response
- Defaults to the templated response should similarity be too small or LLM is unavailable

Reasons to use RAG:
- Stay consistent and gentle
- Prevent unsafe or overly clinical advice
- Improve understanding of emotional situations 
- Easy to maintain
- Ability to grow with more examples

### Agentic AI
- Runs a safety classifier using LLMs to label output as **SAFE / UNSAFE / CRISIS**
- Routing:
    - **CRISIS**: returns crisis guidance and resources
    - **UNSAFE**: gives an explanation as to why it's harmful and redirects the user to safer alternatives
    - **SAFE**: proceeds with RAG
    - **OFF-LIMITS**: tells the user to reframe around feelings & behavior triggers

```
def abaWithAgenticAI(userInput, k=3):
```
is a policy-based agent divided into **FOUR** routes:
- Crisis 
- Unethical 
- Emotional Processing
- Off-Limits


#### Comparison Table (Baseline vs. RAG vs. Agentic AI)

|                   | Baseline (TF-IDF)         | RAG (TF-IDF w/ LLMs)      | Agentic AI                        |
| -------------     | -------------             | -------------             | -------------                     |
| Retrieval         | TF-IDF Cosine Similarity  |  TF-IDF Cosine Similarity | TF-IDF Cosine Similarity          |
| Dataset Handling  | Extracts rows             |  Extracts rows            | Extracts rows                     |
| Generation        | Templated                 |  LLMs                     | LLMs                              |
| Safety/Compliance | None                      | None                      | UNSAFE/SAFE/CRISIS Routing        |
| Best Use          | Simple baseline           | More natural responses    | product reliability/guardrails    |

## System Architecture

### Baseline TF-IDF Retriever (no LLMs)
![alt text](./images/ABAArchitectureDiagramTFIDF.png)

### RAG with LLMs
![alt text](./images/ABAArchitectureDiagramRAG.png)

### Agentic AI
![alt text](./images/ABAArchitectureDiagramAgenticAI.png)


## Security & Compliance Safeguards
- No user data collection
- Synthetic or anonymized data
- Ethical safeguards to prevent harmful or unethical interpretation

## Methodology
- **Dataset Design**
- **Preprocessing**
- **Baseline Model Design (_TF-IDF_)**
- **RAG Model Design**
- **Evaluation**
- **Ethical Design**


## Streamlit App

Run the app: 

[ABA Assist](https://aba-assist.streamlit.app/)

## DISCLAIMER

This tool is an experimental AI project inspired by Applied Behavior Analysis (ABA). It's designed for educational and self-reflective use ONLY. 
It is NOT a substitute for professional therapy, diagnosis, or clinical advice. 



## References & Acknowledgements

- Autism Speaks. “Applied Behavior Analysis (ABA).” Autism Speaks, 2021, www.autismspeaks.org/applied-behavior-analysis.


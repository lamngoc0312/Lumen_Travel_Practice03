## Link Repository:
[https://github.com/lamngoc0312/Lumen_Travel_Practice03]

# Lumen Travel Assistant

A guide-grounded travel chatbot for Ho Chi Minh City, built with Streamlit, Firebase Authentication, and a local Ollama RAG pipeline.

Only authenticated users can access the chat. For every question, the application retrieves relevant entries from a custom travel guide, generates an answer from that evidence, and displays the retrieved sources and cosine-similarity scores. When the guide does not contain the requested information, the assistant responds with "I don't know" instead of guessing.

## Student information

- **Name:** Lâm Như Ngọc
- **Student ID:** 25127214
- **Class:** 25C09
- **Course:** Computational Thinking - Practice 03
- **Chosen city:** Ho Chi Minh City, Vietnam

## Features

- Firebase email/password account creation, sign-in, and sign-out.
- Chat interface available only to authenticated users.
- A custom knowledge base containing 17 sourced city-guide entries.
- Guide embeddings generated with `embeddinggemma` and cached by Streamlit.
- Top-k retrieval using cosine similarity.
- Grounded answers generated with `qwen3:4b` and cited with guide IDs such as `[G03]`.
- Expandable evidence panel showing retrieved entries, scores, and sources.
- Honest abstention when the guide does not contain an answer.
- Session-only chat history stored in `st.session_state` and cleared at sign-out.
- Clear errors for invalid authentication, empty questions, configuration problems, and unavailable Ollama endpoints.

No Firestore or other database is used.

## Architecture

```text
email + password
      -> Firebase Authentication
      -> authenticated user + token
      -> unlock Streamlit chat

knowledge_base.json
      -> validation -> embeddinggemma -> cached guide vectors

user question
      -> embeddinggemma -> query vector
      -> cosine similarity -> sorted top-k entries
      -> grounded context -> qwen3:4b
      -> answer with [Gxx] citation or "I don't know"

answer + retrieved entries + scores + sources
      -> Streamlit UI -> current session history
```

## Repository structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit authentication and chat interface |
| `firebase_client.py` | Firebase email/password REST client |
| `rag.py` | Guide validation, embedding, retrieval, generation, and answer validation |
| `knowledge_base.json` | 17 Ho Chi Minh City guide entries |
| `scripts/evaluate_rag.py` | Repeatable six-question live RAG evaluation |
| `tests/` | Offline unit and smoke tests |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment-variable template with empty values |

## Installation and configuration

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run the virtual-environment executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure Firebase Authentication

1. Create a Firebase project and register a Web App.
2. Open **Authentication > Sign-in method** and enable **Email/Password**.
3. Copy the Web App `apiKey` from Firebase Project settings.

### 3. Prepare Ollama

Install Ollama and download the required models:

```powershell
ollama pull embeddinggemma
ollama pull qwen3:4b
ollama list
```

For local development, Ollama is normally available at `http://localhost:11434`.

To use the Lab 4 Pinggy path, create the tunnel on the machine running Ollama:

```powershell
ssh -p 443 -R0:localhost:11434 free.pinggy.io
```

Use the HTTPS forwarding URL reported by Pinggy as `OLLAMA_URL`. The tunnel exists only while the SSH session remains active.

### 4. Create the local environment file

Copy the provided template:

```powershell
Copy-Item .env.example .env
```

Set the local values in `.env`:

```dotenv
FIREBASE_API_KEY=YOUR_FIREBASE_WEB_API_KEY
OLLAMA_URL=http://localhost:11434
TOP_K=3
MIN_SIMILARITY=0.0
```

Replace the local Ollama URL with the Pinggy HTTPS URL when testing through the Lab 4 tunnel.

The `.env` file is ignored by Git and must not be committed.

## Running the application

Start Streamlit from the repository root:

```powershell
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

## Testing

Run all offline tests:

```powershell
python -m unittest discover -s tests -v
```

The current suite contains 17 tests covering the guide schema, retrieval order, cosine similarity, citation validation, abstention, Qwen reasoning removal, Firebase error handling, and the signed-out UI.

Run the six-question live evaluation against Ollama:

```powershell
python scripts/evaluate_rag.py --ollama-url http://localhost:11434
```

The command also accepts a Pinggy HTTPS URL through `--ollama-url`.

## RAG evaluation results

The following results were observed on November 14, 2026 with Ollama 0.40.16, `embeddinggemma`, `qwen3:4b`, `TOP_K=3`, and the current 17-entry guide. Retrieved IDs are listed in ranking order.

| Type | Question | Expected entry ID | Retrieved IDs | Correct? | Failure step |
|---|---|---|---|---|---|
| Answerable 1 | Where can I buy souvenirs in the city? | G01 | G01, G17, G06 | Yes | N/A |
| Answerable 2 | Which walking street links City Hall and Bach Dang Wharf? | G02 | G02, G06, G15 | Yes | N/A |
| Answerable 3 | What are the Cu Chi Tunnels? | G10 | G10, G06, G02 | Yes | N/A |
| Paraphrase | Where can I explore the Chinese quarter? | G06 | G06, G01, G02 | Yes | N/A |
| Not in guide 1 | How much is the ticket for Ben Thanh Market tomorrow? | None | G01, G07, G06 | Yes - abstained | N/A |
| Not in guide 2 | At what time does Saigon Opera House open tomorrow? | None | G05, G03, G16 | Yes - abstained | N/A |

For an incorrect result, the failure step is classified as:

- **Missing data:** the required information is absent from `knowledge_base.json`.
- **Retrieval:** the relevant entry is not included in the top-k results.
- **Generation:** the relevant evidence is retrieved, but the generated answer is unsupported or incorrect.

During the first live evaluation, Qwen included internal reasoning text ending in `</think>` even though thinking was disabled. This was a generation-output/post-processing failure. The response pipeline now removes that text before validating and displaying the answer, and an offline regression test covers the behavior.

## Computational Thinking analysis

### Abstraction

The system input is an authenticated user's natural-language question together with the local travel guide. The output is either a short grounded answer with `[Gxx]` citations and retrieval evidence, or an abstention.

Each guide entry keeps:

- `id` for traceability and citations;
- `title` and `text` for retrieval and answer generation;
- `source` for verification;
- `category` and `tags` for structured filtering and future RAG-Rec ranking.

### Decomposition

| Sub-problem | Input | Output |
|---|---|---|
| Authenticate | Email and password | Firebase user and token |
| Embed guide | Validated guide text | Cached vector for each entry |
| Embed question | Non-empty user question | Query vector |
| Retrieve | Query vector and guide vectors | Top-k entries and cosine scores |
| Build context | Question and retrieved entries | Grounded model prompt |
| Generate | Grounded prompt | Cited answer or abstention |
| Display | Answer and retrieval evidence | Chat UI and session history |
| Sign out | Current session | Cleared user and messages |

### Pattern recognition

The project reuses Firebase email/password authentication from Lab 3 and the Ollama, embedding, chat, and Pinggy workflow from Lab 4. The `filter -> score -> top-k` pattern appears in retrieval: candidates are limited to guide entries, scored using cosine similarity, and sorted before selecting the highest-ranked entries. RAG-Rec can reuse the same pattern to rank destinations using category and tags as additional features.

### Algorithm design

```text
if the user is not authenticated:
    show authentication and stop before chat

if guide vectors are not cached:
    validate and embed the guide once

for each non-empty question:
    embed the question
    compute cosine similarity with every guide vector
    sort results from highest to lowest
    select the top 3 entries
    generate an answer using only those entries
    validate that the answer cites a retrieved entry
    if the requested fact is unsupported:
        return "I don't know"
    display the answer, entries, scores, and sources
    save the turn in session state
```

The retriever uses `k=3` to provide enough supporting context while keeping the prompt short and inspectable. For example, the Opera House entry may be relevant to a question about tomorrow's opening time, but that time is absent from the guide; the correct result is therefore an abstention rather than a guess.

## Known limitations

- The guide is a small static dataset and does not provide live prices, schedules, or ticket availability.
- A Pinggy URL is temporary and becomes unavailable when its SSH tunnel closes.
- Local model latency depends on the host hardware; CPU-only responses may take 30-60 seconds.
- Small language models may still omit evidence or fail to abstain, so displayed citations should be reviewed.
- Chat history exists only for the current Streamlit session and is intentionally cleared at sign-out.
- Authentication protects the Streamlit chat interface; this project does not implement additional protected backend resources.

## AI tools used

ChatGPT (Codex) was used to compare the implementation with the assignment specification, improve input and data validation, add offline tests, diagnose Firebase and Ollama integration errors, and organize the documentation. The final implementation, guide content, configuration, tests, and demonstration were reviewed and run by the student.

## Demonstration video

[Watch the Assignment 3 demonstration on YouTube](https://youtu.be/8KB8pkBdVa0)

# RAGChef

> **Retrieve, then cook — recipe answers grounded in a real knowledge base, not hallucinated**

A recipe Q&A tool built on Retrieval-Augmented Generation (RAG): Chrome extension frontend + FastAPI backend. Users ask a question in the extension, the backend retrieves relevant content from a local recipe knowledge base using a hybrid of vector (BGE embeddings via `sentence-transformers`, indexed with a `faiss-cpu` `IndexFlatIP`) and keyword (`rank_bm25`'s `BM25Okapi`) search, then passes it to DeepSeek — through the `openai` SDK's `OpenAI` client, pointed at DeepSeek's OpenAI-compatible endpoint — to generate an answer grounded strictly in the retrieved content. No LangChain: retrieval and generation are built directly on the underlying libraries.

---

# Features

- Recipe Q&A over a knowledge base of 50 English recipes (20 Chinese home-style dishes + 30 British classics), one Markdown file per recipe, auto-tagged with category/dish name/difficulty metadata from its folder and content
- Hybrid retrieval (`faiss-cpu` vector index + `rank_bm25` BM25, fused with Reciprocal Rank Fusion) + DeepSeek generation via the `openai` SDK; clearly says so instead of making things up when nothing relevant is found
- Query routing + 3 generation modes: "recommend a few dishes" gets a plain list of dish names (no LLM call needed), "how do I make X" gets a structured step-by-step answer, everything else gets a plain grounded answer; vague questions are rewritten to be more specific before retrieval. All prompts are plain string templates filled with `str.format()`
- Vector index cached to disk (via `faiss.write_index`/`faiss.read_index`) and rebuilt automatically only when the recipe files change, so restarts skip re-embedding the whole knowledge base
- Optional retrieval filtering by category/difficulty, either passed explicitly or inferred from the question text (e.g. "a quick vegetarian recipe")
- Parent/child chunking with heading-aware splitting (#/##/###) and relevance-ranked deduplication, so a recipe matched on multiple sections outranks one matched on a single fragment; this step (and the RRF-based per-parent score aggregation) is hand-rolled project-specific logic — see the module docstring in `rag.py` for why
- Chrome extension (Manifest V3) popup written in TypeScript: it calls `POST /ask/stream` and renders the answer progressively as it streams in, using a small built-in Markdown renderer that escapes HTML before adding any markup. FastAPI also exposes a non-streaming `POST /ask` endpoint
- Fails fast on missing config (API key / empty knowledge base) at startup; returns a friendly message instead of crashing when the LLM call fails
- pytest suite covering retrieval logic and the API (with the `openai` client's `chat.completions.create` mocked), plus an optional real DeepSeek integration test
- Vitest unit tests for the extension's pure logic (Markdown rendering/escaping, input validation, streamed-response handling), and a GitHub Actions workflow that type-checks, tests and builds the extension on every push

---

# Frontend Tech Stack

The Chrome extension lives in `extension/` and is written in **TypeScript** (`strict`, `noUncheckedIndexedAccess`, no `any`), with Chrome API types from `@types/chrome`.

- **esbuild** bundles `extension/src/popup.ts` into `extension/dist/popup.js`, which `popup.html` loads
- **tsc** (`tsc --noEmit`) does the type checking only
- **Vitest** runs the unit tests in `extension/tests/`

The popup's request body (`{ "question": string }`) is typed against the backend's Pydantic `QueryRequest` model. The backend URL is set in one place, `extension/src/config.ts`.

---

# Structure

```text
RAGChef/
├── server/
│   ├── app.py                  # FastAPI entry point, defines /ask and /ask/stream
│   ├── rag.py                  # Retrieval and generation logic (SimpleRAG)
│   ├── data/recipes/           # Recipe knowledge base, one .md file per recipe,
│   │                           # grouped into category subfolders (meat_dish/, soup/, ...)
│   ├── data/vector_index/      # Cached faiss-cpu vector index (auto-built/rebuilt, gitignored)
│   ├── tests/                  # pytest tests
│   ├── Dockerfile
│   ├── requirements.txt / requirements-dev.txt
│   └── .env.example
│
├── extension/                  # Chrome extension (Manifest V3)
│   ├── manifest.json
│   ├── popup.html              # Popup markup + styles; loads dist/popup.js
│   ├── src/                    # TypeScript source
│   │   ├── popup.ts            # Entry point: wires the DOM to the API
│   │   ├── api.ts              # POST /ask/stream + streamed-response reading
│   │   ├── markdown.ts         # HTML-escaping Markdown renderer
│   │   ├── validation.ts       # Question input validation
│   │   ├── messages.ts         # User-facing strings and error formatting
│   │   ├── dom.ts              # Type-checked element lookup
│   │   ├── config.ts           # Backend base URL (single source of truth)
│   │   └── types.ts            # Request/response types for the backend contract
│   ├── tests/                  # Vitest unit tests
│   ├── dist/                   # esbuild output (generated, gitignored)
│   ├── package.json            # build / watch / typecheck / test scripts
│   └── tsconfig.json
│
├── .github/workflows/extension.yml  # CI: typecheck, test, build the extension
│
├── docker-compose.yml          # Local Docker startup
├── render.yaml                 # Render deployment config
└── README.md
```

---

# Getting Started

1. Set your DeepSeek API key in `server/.env` (see `server/.env.example`; get a key at [platform.deepseek.com](https://platform.deepseek.com)).

2. Start the backend, either way:

```bash
# Run locally
cd server
pip install -r requirements.txt
uvicorn app:app --reload

# Or with Docker
docker compose up --build
```

Runs on `http://127.0.0.1:8000` by default. `GET /` confirms the server is up, `POST /ask` with `{ "question": "..." }` returns an answer.

3. Build the Chrome extension (requires Node.js 22.12+):

```bash
cd extension
npm ci            # install dependencies from package-lock.json
npm run build     # bundle src/ into dist/popup.js
```

Use `npm run watch` to rebuild automatically while editing.

4. Load it into Chrome: open `chrome://extensions`, turn on "Developer mode", click "Load unpacked" and select the `extension/` directory. After rebuilding, click the reload icon on the extension's card. The popup sends requests to `BACKEND_BASE_URL` in `extension/src/config.ts`, which is currently `http://localhost:8000`, with the Render URL commented out next to it. To point at a different backend, change it there, make sure that origin is listed under `host_permissions` in `extension/manifest.json`, and rebuild.

Run tests:

```bash
# Backend
cd server && pip install -r requirements-dev.txt && pytest

# Extension
cd extension && npm run typecheck && npm test
```

---

# Example

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{ "question": "How do I make Kung Pao Chicken?" }'
```

Response (illustrative — the actual wording is generated by DeepSeek and varies, though it stays grounded in the retrieved recipe content):

```json
{
  "answer": "To make Kung Pao Chicken: dice and marinate chicken breast, stir-fry dried chilies until fragrant, add the chicken and stir-fry until it changes color, then add the sauce mixture (soy sauce, vinegar, sugar) and peanuts, tossing until glazed."
}
```

Asking something outside the knowledge base (e.g. a French dessert) returns: "No relevant information was found in the current recipe library."

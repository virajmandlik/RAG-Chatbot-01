# Development Roadmap & Phased Implementation

The project will be developed in five phases to ensure a structured and manageable workflow, starting with a core functional backend and progressively adding features.

## Phase 1: Foundation & Core Backend Logic (MVP)
Goal: Create a command-line application that can answer questions from a local folder of documents.

Tasks:
- Set up Python environment (venv, pip)
- Install libraries (`google-generativeai`, `pinecone-client`, `langchain`, `langchain-community`, `pypdf`, `python-dotenv`, `docx2txt`)
- Load API keys via `.env`
- Ingestion: Load documents, split into chunks, generate embeddings, upsert into Pinecone
- Query: Embed query, retrieve context from Pinecone, generate answer with Gemini
- Test from the command line

## Phase 2: API and Web Interface
Goal: Expose backend via REST API and build a simple frontend.

Tasks:
- Wrap backend logic in FastAPI
- Create `/chat` endpoint
- Add basic conversation memory
- Build React UI and connect to `/chat`

## Phase 3: Automation with Google Drive
Goal: Automate document ingestion from Google Drive via webhooks.

Tasks:
- Enable Drive API and configure OAuth
- Create `/webhook/drive-update`
- Handle Drive notifications, download changed files, trigger processing

## Phase 4: Enhancements & LLM Modularity
Goal: Add features and provider switch.

Tasks:
- Abstract LLM service (Gemini/OpenAI)
- Improve error handling/logging and memory
- (Optional) Streaming responses

## Phase 5: Containerization & Deployment
Goal: Prepare for production.

Tasks:
- Dockerfile(s) for backend/frontend
- Docker Compose for local dev
- Deploy backend on Cloud Run, frontend on Vercel/Netlify
- Configure environment variables/secrets
- End-to-end staging tests

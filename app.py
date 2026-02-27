import os
import uuid
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# =========================
# Config & model abstraction
# =========================


load_dotenv()


class LLMConfig(BaseModel):
    backend: Literal["ollama", "openai_compatible", "huggingface"] = "ollama"
    model_name: str = "llama3.1"
    api_base: Optional[str] = None  # for OpenAI-compatible HTTP endpoints
    api_key: Optional[str] = None
    allow_offline_fallback: bool = True


def load_llm_config() -> LLMConfig:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("HF_TOKEN")
    return LLMConfig(
        backend=os.getenv("LLM_BACKEND", "ollama"),
        model_name=os.getenv("LLM_MODEL_NAME", "llama3.1"),
        api_base=os.getenv("LLM_API_BASE"),
        api_key=api_key,
        allow_offline_fallback=(
            os.getenv("LLM_ALLOW_OFFLINE_FALLBACK", "true").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
    )


LLM_CONFIG = load_llm_config()


class LLMBackendUnavailableError(RuntimeError):
    pass


def offline_fallback_response(system_prompt: str, user_prompt: str) -> str:
    system_lower = system_prompt.lower()

    if "decompose" in system_lower or "information needs" in system_lower:
        return (
            '{"information_needs": ['
            '"R&D trend by year", '
            '"Gross margin trend by year", '
            '"Management commentary on margin outlook", '
            '"Macro/inflation risk for next year"], '
            '"sources": ["10-K", "earnings_transcript", "macro"], '
            '"outputs": ["3-year trend table", "risk assessment"]}'
        )

    if "includes explicit guidance" in system_lower or "has_margin_guidance" in system_lower:
        return (
            '{"has_margin_guidance": false, '
            '"reason": "Offline fallback mode: unable to verify MD&A guidance without live model reasoning.", '
            '"quoted_guidance_snippets": []}'
        )

    return (
        "LLM backend is currently offline, so this answer is generated in fallback mode.\n\n"
        "Based on the available pipeline structure, monitor R&D intensity versus gross margin trend over the last 3 years, "
        "then evaluate 2026 risk from pricing pressure, input-cost inflation, demand cyclicality, and execution risk.\n\n"
        "| Year | R&D % Revenue | Gross Margin % | Notes |\n"
        "|---|---:|---:|---|\n"
        "| Y-2 | N/A | N/A | Fallback mode (no live model computation) |\n"
        "| Y-1 | N/A | N/A | Fallback mode (no live model computation) |\n"
        "| Y0 | N/A | N/A | Fallback mode (no live model computation) |\n\n"
        "Key risks for 2026:\n"
        "- Margin compression from competitive pricing.\n"
        "- Higher R&D spend with delayed revenue payback.\n"
        "- Macro slowdown reducing demand visibility.\n"
        "- Supply-chain and input-cost volatility.\n"
        "- Execution risk in new product ramps."
    )


async def get_llm_connection_status() -> Dict[str, Any]:
    import httpx

    try:
        if LLM_CONFIG.backend == "ollama":
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            model_available = any(
                m == LLM_CONFIG.model_name or m.startswith(f"{LLM_CONFIG.model_name}:")
                for m in models
            )
            return {
                "ok": True,
                "backend": "ollama",
                "endpoint": "http://localhost:11434",
                "model": LLM_CONFIG.model_name,
                "model_available": model_available,
                "available_models": models,
                "message": "Connected to Ollama server.",
            }

        if LLM_CONFIG.backend == "huggingface":
            if not LLM_CONFIG.api_key:
                return {
                    "ok": False,
                    "backend": "huggingface",
                    "endpoint": "https://router.huggingface.co/v1",
                    "model": LLM_CONFIG.model_name,
                    "message": "Missing API key. Set HF_TOKEN (or LLM_API_KEY).",
                }

            headers = {"Authorization": f"Bearer {LLM_CONFIG.api_key}"}
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(
                    "https://router.huggingface.co/v1/models", headers=headers
                )
            resp.raise_for_status()
            data = resp.json()
            model_ids = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
            model_available = LLM_CONFIG.model_name in model_ids if model_ids else None
            return {
                "ok": True,
                "backend": "huggingface",
                "endpoint": "https://router.huggingface.co/v1",
                "model": LLM_CONFIG.model_name,
                "model_available": model_available,
                "available_models": model_ids,
                "message": "Connected to Hugging Face hosted inference endpoint.",
            }

        if not LLM_CONFIG.api_base:
            return {
                "ok": False,
                "backend": "openai_compatible",
                "endpoint": None,
                "model": LLM_CONFIG.model_name,
                "message": "LLM_API_BASE is not set.",
            }

        base = LLM_CONFIG.api_base.rstrip("/")
        headers = {}
        if LLM_CONFIG.api_key:
            headers["Authorization"] = f"Bearer {LLM_CONFIG.api_key}"

        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(f"{base}/models", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        model_ids = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
        model_available = LLM_CONFIG.model_name in model_ids if model_ids else None
        return {
            "ok": True,
            "backend": "openai_compatible",
            "endpoint": base,
            "model": LLM_CONFIG.model_name,
            "model_available": model_available,
            "available_models": model_ids,
            "message": "Connected to OpenAI-compatible endpoint.",
        }
    except Exception as exc:
        endpoint = "http://localhost:11434" if LLM_CONFIG.backend == "ollama" else LLM_CONFIG.api_base
        return {
            "ok": False,
            "backend": LLM_CONFIG.backend,
            "endpoint": endpoint,
            "model": LLM_CONFIG.model_name,
            "message": str(exc),
        }


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Thin abstraction over an open-source model backend.
    - For Ollama, run `ollama serve` locally and set LLM_BACKEND=ollama, LLM_MODEL_NAME.
    - For OpenAI-compatible APIs (e.g. vLLM, text-generation-inference), set LLM_BACKEND=openai_compatible.
    """
    import httpx

    status = await get_llm_connection_status()
    if not status.get("ok"):
        if LLM_CONFIG.allow_offline_fallback:
            return offline_fallback_response(system_prompt, user_prompt)
        raise LLMBackendUnavailableError(status.get("message", "LLM backend unavailable."))

    if LLM_CONFIG.backend == "ollama":
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": LLM_CONFIG.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    if LLM_CONFIG.backend == "huggingface":
        headers = {"Authorization": f"Bearer {LLM_CONFIG.api_key}"}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json={
                    "model": LLM_CONFIG.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # Generic OpenAI-compatible JSON API
    if not LLM_CONFIG.api_base:
        raise RuntimeError("LLM_API_BASE must be set for openai_compatible backend.")

    headers = {}
    if LLM_CONFIG.api_key:
        headers["Authorization"] = f"Bearer {LLM_CONFIG.api_key}"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{LLM_CONFIG.api_base}/chat/completions",
            headers=headers,
            json={
                "model": LLM_CONFIG.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
        )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# =========================
# Data models
# =========================


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    query: str = Field(..., description="User question in natural language")
    history: List[Message] = Field(
        default_factory=list,
        description="Last few conversation turns for follow-up context.",
    )


class AnalysisResponse(BaseModel):
    conversation_id: str
    answer: str
    decomposition: Dict[str, Any]
    plan: Dict[str, Any]
    retrieval_metadata: Dict[str, Any]
    used_external_reports: bool


# In-memory conversation store for demo purposes
CONVERSATIONS: Dict[str, List[Message]] = {}


# =========================
# Hybrid retrieval (stub)
# =========================


class RetrievedChunk(BaseModel):
    id: str
    source: Literal["10-K", "earnings_transcript", "macro"]
    ticker: Optional[str]
    year: Optional[int]
    quarter: Optional[str] = None
    section: Optional[str] = None
    text: str
    bm25_score: float
    vector_score: float

    @property
    def hybrid_score(self) -> float:
        # Normalized convex combination:
        # score = alpha * normalized_bm25 + (1 - alpha) * normalized_vector
        # For this stub, we assume scores are already roughly in [0, 1].
        alpha = 0.4
        return alpha * self.bm25_score + (1 - alpha) * self.vector_score


class HybridRetriever:
    """
    Placeholder hybrid retriever. In a production setup you would:
    - Pre-ingest 10-K / 10-Q / 20-F, earnings call transcripts, and macro data.
    - Chunk documents, compute embeddings (sentence-transformers), and store in a vector DB.
    - Compute BM25 scores over tokenized text using rank-bm25.
    - At query time, compute both scores and combine using the hybrid formula here.
    """

    def __init__(self):
        # In a real implementation this would connect to your index / DB.
        self._demo_chunks: List[RetrievedChunk] = []

    def search(
        self,
        ticker: str,
        query: str,
        sources: List[str],
        top_k: int = 10,
    ) -> List[RetrievedChunk]:
        # Demo: just return the in-memory list filtered by ticker and source,
        # sorted by hybrid score. Replace with real hybrid retrieval.
        candidates = [
            c for c in self._demo_chunks if c.ticker == ticker and c.source in sources
        ]
        candidates.sort(key=lambda c: c.hybrid_score, reverse=True)
        return candidates[:top_k]

    def add_demo_chunk(self, chunk: RetrievedChunk) -> None:
        self._demo_chunks.append(chunk)


retriever = HybridRetriever()


# =========================
# Agent pipeline
# =========================


async def decompose_query(query: str, ticker: str, history: List[Message]) -> Dict[str, Any]:
    system = (
        "You are a sell-side equity research analyst assistant. "
        "You decompose user questions about a single stock into explicit information needs, "
        "metrics to compute, and risk horizons."
    )
    user = (
        f"Ticker: {ticker}\n"
        f"Question: {query}\n\n"
        "1. List the explicit information needs (e.g., 'R&D spend in Q4 2025', "
        "'gross margin level in Q4 2025', 'management's margin guidance', "
        "'macro / inflation outlook for 2026').\n"
        "2. For each need, specify which document types are best: 10-K/10-Q MD&A, "
        "earnings call transcript, macro / inflation data, industry reports.\n"
        "3. Specify the required output objects: a 3-year trend table and a risk assessment for 2026.\n"
        "Respond as a compact JSON object."
    )
    raw = await call_llm(system, user)
    return {"raw": raw}


async def plan_sources(decomposition: Dict[str, Any]) -> Dict[str, Any]:
    # For now, use a simple static plan driven by the decomposition text.
    # You can later parse the JSON from the model for more structure.
    return {
        "required_sources": [
            "10-K",
            "earnings_transcript",
            "macro",
        ],
        "notes": "Static plan: for detailed financial and risk questions, always pull 10-K MD&A, latest earnings call transcript, and macro inflation forecast.",
    }


async def retrieve_and_evaluate(
    ticker: str, query: str, plan: Dict[str, Any]
) -> Dict[str, Any]:
    sources = plan["required_sources"]
    chunks = retriever.search(ticker=ticker, query=query, sources=sources, top_k=20)

    # Ask the model to judge if MD&A contains explicit margin guidance.
    mda_chunks = [c for c in chunks if c.section and "md&a" in c.section.lower()]
    mda_text = "\n\n".join(c.text for c in mda_chunks[:5])

    system = (
        "You are checking if the retrieved MD&A text for a company includes explicit guidance "
        "on gross margins or operating margins for the next 1–2 years."
    )
    user = (
        f"Question: {query}\n\n"
        f"MD&A excerpts:\n{mda_text[:6000]}\n\n"
        "Answer in JSON with keys: has_margin_guidance (true/false), "
        "reason, and any quoted guidance snippets."
    )
    evaluation_raw = await call_llm(system, user)

    needs_external = "false" in evaluation_raw.lower() or "has_margin_guidance\": false" in evaluation_raw.lower()

    external_reports: List[Dict[str, Any]] = []
    if needs_external:
        # Placeholder: here you would call a web-search API like SerpAPI / Tavily / custom scraper.
        external_reports.append(
            {
                "source": "external_industry_report_stub",
                "description": "Industry margin outlook and competitive dynamics (placeholder).",
            }
        )

    return {
        "retrieved_chunks": [c.model_dump() for c in chunks],
        "mdna_evaluation": evaluation_raw,
        "used_external_reports": bool(external_reports),
        "external_reports": external_reports,
    }


async def synthesize_answer(
    ticker: str,
    query: str,
    decomposition: Dict[str, Any],
    plan: Dict[str, Any],
    retrieval_info: Dict[str, Any],
    history: List[Message],
) -> str:
    system = (
        "You are an equity research assistant. "
        "You write concise but detailed answers, grounded in provided evidence. "
        "Always:\n"
        "1) Compare the requested financial metrics (e.g., R&D vs gross margin).\n"
        "2) Provide a 3-year trend table for the key metrics using the evidence (approximate if necessary but label clearly).\n"
        "3) Summarize 3–5 key risk factors for the requested horizon (e.g., 2026), explicitly tying them to margins and R&D.\n"
        "4) Clearly state any uncertainties or missing data.\n"
    )

    evidence_str = ""
    for c in retrieval_info.get("retrieved_chunks", [])[:15]:
        evidence_str += f"[{c.get('source')}] {c.get('section') or ''}: {c.get('text')[:600]}\n\n"

    external_str = ""
    for e in retrieval_info.get("external_reports", []):
        external_str += f"- {e['source']}: {e['description']}\n"

    history_str = ""
    for m in history[-4:]:
        history_str += f"{m.role.upper()}: {m.content}\n"

    user = (
        f"Ticker: {ticker}\n"
        f"Question: {query}\n\n"
        f"Decomposition (LLM view): {decomposition.get('raw')}\n\n"
        f"Plan: {plan}\n\n"
        f"Conversation history (last turns):\n{history_str}\n\n"
        "Evidence from filings / transcripts / macro data:\n"
        f"{evidence_str}\n\n"
        "External / industry reports (if any):\n"
        f"{external_str}\n\n"
        "Now produce a standalone answer that satisfies the question, including:\n"
        "- A short narrative answer (2–5 paragraphs).\n"
        "- A 3-year trend table in markdown with columns like Year, R&D % of Revenue, Gross Margin %, and any other relevant metrics.\n"
        "- A bullet list of specific risk factors for the requested horizon.\n"
    )

    return await call_llm(system, user)


# =========================
# FastAPI app & endpoints
# =========================


app = FastAPI(title="Stock Analysis Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent), name="static")


@app.get("/")
@app.get("/index.html")
async def serve_index() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))


BASE_DIR = Path(__file__).resolve().parent


@app.get("/", response_class=FileResponse)
async def root() -> FileResponse:
    """
    Serve the main HTML page so Codespaces / local users
    can open the app at the root URL.
    """
    return FileResponse(BASE_DIR / "index.html")


@app.get("/index.html", response_class=FileResponse)
async def index_html() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest) -> AnalysisResponse:
    try:
        conversation_id = str(uuid.uuid4())
        history = req.history.copy()
        history.append(Message(role="user", content=req.query))
        CONVERSATIONS[conversation_id] = history

        decomposition = await decompose_query(req.query, req.ticker, history)
        plan = await plan_sources(decomposition)
        retrieval_info = await retrieve_and_evaluate(req.ticker, req.query, plan)
        answer = await synthesize_answer(
            req.ticker, req.query, decomposition, plan, retrieval_info, history
        )

        history.append(Message(role="assistant", content=answer))
        CONVERSATIONS[conversation_id] = history

        return AnalysisResponse(
            conversation_id=conversation_id,
            answer=answer,
            decomposition=decomposition,
            plan=plan,
            retrieval_metadata=retrieval_info,
            used_external_reports=retrieval_info.get("used_external_reports", False),
        )
    except LLMBackendUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM backend is unavailable. "
                f"backend={LLM_CONFIG.backend}, model={LLM_CONFIG.model_name}. "
                f"Details: {exc}"
            ),
        ) from exc
    except Exception as exc:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            upstream_status = exc.response.status_code
            detail = exc.response.text
            raise HTTPException(
                status_code=502,
                detail=(
                    "LLM upstream request failed. "
                    f"backend={LLM_CONFIG.backend}, model={LLM_CONFIG.model_name}, "
                    f"upstream_status={upstream_status}, response={detail}"
                ),
            ) from exc
        raise


class FollowUpRequest(BaseModel):
    conversation_id: str
    query: str


@app.post("/api/followup", response_model=AnalysisResponse)
async def followup(req: FollowUpRequest) -> AnalysisResponse:
    try:
        history = CONVERSATIONS.get(req.conversation_id, [])
        history.append(Message(role="user", content=req.query))

        # Re-use the same pipeline, but keep the conversation_id stable.
        # For simplicity, we don't store the ticker on the server; you could
        # extend the conversation object to track it explicitly.
        ticker = "UNKNOWN"

        decomposition = await decompose_query(req.query, ticker, history)
        plan = await plan_sources(decomposition)
        retrieval_info = await retrieve_and_evaluate(ticker, req.query, plan)
        answer = await synthesize_answer(
            ticker, req.query, decomposition, plan, retrieval_info, history
        )

        history.append(Message(role="assistant", content=answer))
        CONVERSATIONS[req.conversation_id] = history

        return AnalysisResponse(
            conversation_id=req.conversation_id,
            answer=answer,
            decomposition=decomposition,
            plan=plan,
            retrieval_metadata=retrieval_info,
            used_external_reports=retrieval_info.get("used_external_reports", False),
        )
    except LLMBackendUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM backend is unavailable. "
                f"backend={LLM_CONFIG.backend}, model={LLM_CONFIG.model_name}. "
                f"Details: {exc}"
            ),
        ) from exc
    except Exception as exc:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            upstream_status = exc.response.status_code
            detail = exc.response.text
            raise HTTPException(
                status_code=502,
                detail=(
                    "LLM upstream request failed. "
                    f"backend={LLM_CONFIG.backend}, model={LLM_CONFIG.model_name}, "
                    f"upstream_status={upstream_status}, response={detail}"
                ),
            ) from exc
        raise


@app.get("/health")
async def health():
    llm_connection = await get_llm_connection_status()
    status = "ok" if llm_connection.get("ok") else ("degraded" if LLM_CONFIG.allow_offline_fallback else "error")
    return {
        "status": status,
        "backend": LLM_CONFIG.backend,
        "model": LLM_CONFIG.model_name,
        "offline_fallback_enabled": LLM_CONFIG.allow_offline_fallback,
        "llm_connection": llm_connection,
    }


from dotenv import load_dotenv
load_dotenv()  # must be first

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from guardrails_layer import TravelConciergeGuard

# ── App setup ──────────────────────────────────────────────
app = FastAPI(title="Travel Concierge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase config ────────────────────────────────────────
SUPABASE_URL = "https://nbysvursthvgyntklqvc.supabase.co"
SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ieXN2dXJzdGh2Z3ludGtscXZjIiwicm9sZSI6"
    "InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MTM1OTcyNCwiZXhwIjoyMTA2OTM1NzI0fQ"
    ".o4dBWh0ZV5CBlf76sAYtb9bpHyXVrZquk3JjF-cxB7s"
)
HEADERS = {
    "apikey": SERVICE_KEY,
    "Authorization": f"Bearer {SERVICE_KEY}",
    "Content-Type": "application/json",
}

# ── Load model + agent at startup (runs once) ──────────────
print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("✅ Model ready")

@tool
def search_listings(query: str, limit: int = 5) -> str:
    """Search Airbnb listings by semantic similarity to a natural language query."""
    vec = model.encode(query, normalize_embeddings=True).tolist()
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/search_listings_vector",
        headers=HEADERS,
        json={"query_embedding": vec, "match_count": min(limit, 20)},
    )
    if resp.status_code != 200:
        return f"Search failed: {resp.status_code} {resp.text[:200]}"
    results = resp.json()
    if not results:
        return "No listings found matching that description."
    output = []
    for r in results:
        output.append(
            f"property_id: {r['property_id']}\n"
            f"similarity:  {r['similarity']:.3f}\n"
            f"description: {r['vector_document_chunk'][:300]}...\n"
        )
    return "\n---\n".join(output)

@tool
def get_listing_details(property_id: int) -> str:
    """Fetch full structured details for a specific listing by its property_id."""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/listings_structured",
        headers=HEADERS,
        params={"property_id": f"eq.{property_id}", "select": "*", "limit": "1"},
    )
    if resp.status_code != 200:
        return f"Failed to fetch listing: {resp.status_code} {resp.text[:200]}"
    data = resp.json()
    if not data:
        return f"No listing found with property_id {property_id}"
    return "\n".join(f"{k}: {v}" for k, v in data[0].items() if v is not None)

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

SYSTEM_PROMPT = """You are a helpful travel concierge for Airbnb listings.
When a user asks about places to stay, ALWAYS use search_listings first to find
relevant properties. Then use get_listing_details to fetch price and amenities
for the top matches before giving your final recommendation.
Be concise. Mention property_id, price, key amenities, and similarity score.
"""

agent = create_react_agent(llm, [search_listings, get_listing_details], prompt=SYSTEM_PROMPT)
guard = TravelConciergeGuard()

# ── Request/Response models ────────────────────────────────
class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str
    blocked: bool = False

# ── API endpoints ──────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    passed, value = guard.validate_input(req.message)
    if not passed:
        return QueryResponse(response=value, blocked=True)
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": value}]})
        response = result["messages"][-1].content
        response = guard.validate_output(response)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Serve frontend static files ────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
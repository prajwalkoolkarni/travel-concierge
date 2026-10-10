"""
Phase 3 — Step 4: LangGraph Agent
LLM + direct LangChain tools (no MCP client dependency)
"""
from dotenv import load_dotenv
load_dotenv()
import os
import requests
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from guardrails_layer import TravelConciergeGuard



# ── Config ──────────────────────────────────────────────────────────────
SUPABASE_URL = "https://nbysvursthvgyntklqvc.supabase.co"
SERVICE_KEY  = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ieXN2dXJzdGh2Z3ludGtscXZjIiwicm9sZSI6"
    "InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MTM1OTcyNCwiZXhwIjoyMTA2OTM1NzI0fQ"
    ".o4dBWh0ZV5CBlf76sAYtb9bpHyXVrZquk3JjF-cxB7s"
)
HEADERS = {
    "apikey":        SERVICE_KEY,
    "Authorization": f"Bearer {SERVICE_KEY}",
    "Content-Type":  "application/json",
}

# ── Embedding model ──────────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("✅ Model ready\n")

# ── Tools ────────────────────────────────────────────────────────────────
@tool
def search_listings(query: str, limit: int = 5) -> str:
    """
    Search Airbnb listings by semantic similarity to a natural language query.
    Use this when the user describes what kind of place they are looking for.

    Args:
        query: Natural language description e.g. 'beachfront villa with pool'
        limit: Number of results to return (default 5, max 20)
    """
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
    """
    Fetch full structured details for a specific listing by its property_id.
    Use this after search_listings to get price, amenities, and other details.

    Args:
        property_id: The numeric ID of the listing
    """
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/listings_structured",
        headers=HEADERS,
        params={
            "property_id": f"eq.{property_id}",
            "select":       "*",
            "limit":        "1",
        },
    )

    if resp.status_code != 200:
        return f"Failed to fetch listing: {resp.status_code} {resp.text[:200]}"

    data = resp.json()
    if not data:
        return f"No listing found with property_id {property_id}"

    listing = data[0]
    return "\n".join(f"{k}: {v}" for k, v in listing.items() if v is not None)


# ── LLM ─────────────────────────────────────────────────────────────────
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

# ── Agent ────────────────────────────────────────────────────────────────
agent = create_react_agent(llm, [search_listings, get_listing_details], prompt=SYSTEM_PROMPT)
guard = TravelConciergeGuard()

def main():
    print("🏖️  Travel Concierge ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # ── Input guard ──────────────────────────────────────────────
        passed, value = guard.validate_input(user_input)
        if not passed:
            print(f"\nAgent: {value}\n")
            print("-" * 60)
            continue

        # ── LLM + tools ──────────────────────────────────────────────
        result = agent.invoke({
            "messages": [{"role": "user", "content": value}]
        })
        response = result['messages'][-1].content

        # ── Output guard ─────────────────────────────────────────────
        response = guard.validate_output(response)

        print(f"\nAgent: {response}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
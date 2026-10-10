"""
Phase 3 — Step 3: FastMCP Tool Server
Exposes two tools to the LangGraph agent:
  1. search_listings       — semantic vector search via pgvector
  2. get_listing_details   — fetch structured data for one property
"""

import json
import requests
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP

# ── Config ─────────────────────────────────────────────────────────────
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

# ── Load embedding model ONCE at startup ───────────────────────────────
# This is the key difference from Databricks — the model stays in memory
# for the lifetime of the server. Every tool call reuses it instantly.
import sys
print("Loading embedding model...", file=sys.stderr)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("✅ Model ready — server starting\n", file=sys.stderr)

# ── FastMCP server ─────────────────────────────────────────────────────
mcp = FastMCP("travel-concierge")


@mcp.tool()
def search_listings(query: str, limit: int = 5) -> str:
    """
    Search Airbnb listings by semantic similarity to a natural language query.
    Use this when the user describes what kind of place they are looking for.

    Args:
        query: Natural language description e.g. 'beachfront villa with pool'
        limit: Number of results to return (default 5, max 20)
    """
    # 1. Embed the query — uses search_query mode (asymmetric: matches search_document embeddings)
    vec = model.encode(query, normalize_embeddings=True).tolist()

    # 2. Call the Postgres function we created in Supabase
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/search_listings_vector",
        headers=HEADERS,
        json={
            "query_embedding": vec,
            "match_count":     min(limit, 20),
        },
    )

    if resp.status_code != 200:
        return f"Search failed: {resp.status_code} {resp.text[:200]}"

    results = resp.json()

    if not results:
        return "No listings found matching that description."

    # 3. Format results as readable text for the LLM
    output = []
    for r in results:
        output.append(
            f"property_id: {r['property_id']}\n"
            f"similarity:  {r['similarity']:.3f}\n"
            f"description: {r['vector_document_chunk'][:300]}...\n"
        )

    return "\n---\n".join(output)


@mcp.tool()
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

    # Format as clean key-value text
    lines = [f"{k}: {v}" for k, v in listing.items() if v is not None]
    return "\n".join(lines)


# ── Run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8000)
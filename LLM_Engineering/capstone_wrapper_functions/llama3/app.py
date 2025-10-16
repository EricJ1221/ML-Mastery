from typing import Any, List, Optional, Literal, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from db_utils import make_engine, fetch_rows_as_json
from summarizer import summarize_json, OllamaProvider  # from your previous code

app = FastAPI(title="LLM JSON Summarizer (Llama 3)")
ENGINE = make_engine("postgresql+asyncpg://user:pass@localhost:5432/mydb")

class QuerySummarizeRequest(BaseModel):
    dataset: Literal["orders", "customers"]          # keys in ALLOWED
    columns: List[str] = Field(..., min_items=1)     # must be allowlisted
    where: Optional[Dict[str, Any]] = None           # simple equality filters
    limit: int = 500

    # LLM controls (optional)
    task: Optional[str] = "Executive summary of provided data"
    audience: Optional[Literal["exec","pm","eng","general"]] = "general"
    style: Optional[Literal["bullet","paragraphs"]] = "paragraphs"
    wordLimit: Optional[int] = 200
    model: Optional[str] = "llama3"                  # Ollama model name

@app.post("/api/summarize/query")
async def summarize_from_query(req: QuerySummarizeRequest):
    try:
        data = await fetch_rows_as_json(
            ENGINE,
            dataset=req.dataset,
            columns=req.columns,
            where=req.where,
            limit=req.limit
        )

        # Optional: early bail-out if the query is empty
        if not data:
            return {
                "summary": "No rows matched the query.",
                "bullets": [],
                "insights": [],
                "caveats": ["Empty result set"],
                "stats": {"row_count": 0},
                "model": req.model,
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }

        prov = OllamaProvider(model=req.model or "llama3")
        payload, model_name, usage = summarize_json(
            provider=prov,
            data=data,
            task=req.task or "Executive summary of provided data",
            audience=req.audience or "general",
            style=req.style or "paragraphs",
            word_limit=req.wordLimit or 200,
        )
        return {**payload.model_dump(), "model": model_name, "tokens": usage}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# CALL TO THE FRONT END

# await fetch("/api/summarize/query", {
#   method: "POST",
#   headers: { "Content-Type": "application/json" },
#   body: JSON.stringify({
#     dataset: "orders",
#     columns: ["region", "amount", "status", "created_at"],
#     where: { status: "closed" },
#     limit: 1000,
#     task: "Summarize revenue and anomalies by region",
#     audience: "exec",
#     style: "bullet",
#     wordLimit: 160
#   })
# });

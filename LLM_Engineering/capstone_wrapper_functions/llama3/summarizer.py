from __future__ import annotations
import json, math, textwrap, ujson
from typing import List, Dict, Any, Optional, TypedDict, Protocol
from pydantic import BaseModel, Field, ValidationError

# ---------------------------
# Output schema (strict JSON) ------ Definintely Needs to be modified based on our Database Schema
# ---------------------------
class SummaryPayload(BaseModel):
    summary: str
    bullets: Optional[List[str]] = None
    insights: Optional[List[str]] = None
    caveats: Optional[List[str]] = None
    stats: Optional[Dict[str, float | int | str]] = None

class Usage(TypedDict):
    prompt: int
    completion: int
    total: int

# ---------------------------
# Provider interface
# ---------------------------
class LLMProvider(Protocol):
    name: str
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 768,
        json_mode: bool = True,
    ) -> tuple[str, Usage]: ...

# ---------------------------
# Provider A: Ollama (Llama 3)
# ---------------------------
class OllamaProvider:
    """
    Requires Ollama running locally (default http://localhost:11434).
    `ollama run llama3` to verify the model exists.
    """
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.name = model
        self.model = model
        self.host = host

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 768,
        json_mode: bool = True,
    ) -> tuple[str, Usage]:
        import requests
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            # Llama 3 doesn't have a native "JSON mode" in Ollama;
            # we enforce via instructions + response validation.
            "format": "json" if json_mode else None,
            "stream": False,
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "")
        # token usage is approximate; Ollama returns eval_count/prompt_eval_count
        usage: Usage = {
            "prompt": int(data.get("prompt_eval_count", 0)),
            "completion": int(data.get("eval_count", 0)),
            "total": int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0)),
        }
        return text, usage

# ---------------------------
# Provider B: Transformers (HF)
# ---------------------------
class TransformersProvider:
    """
    HuggingFace Transformers pipeline (instruct-tuned Llama 3).
    Note: You must have access to the model (license/weights).
    """
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", device: Optional[int] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        self.name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device if device is not None else 0 if self.model.device.index is not None else -1,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 768,
        json_mode: bool = True,
    ) -> tuple[str, Usage]:
        # Convert chat messages to a single prompt using the model's chat template if available.
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Simple fallback
            lines = []
            for m in messages:
                lines.append(f"{m['role'].upper()}: {m['content']}")
            prompt = "\n".join(lines) + "\nASSISTANT:"

        out = self.pipe(
            prompt,
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Extract only the assistant part (best-effort)
        text = out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
        usage: Usage = {"prompt": 0, "completion": max_tokens, "total": max_tokens}  # rough; HF doesn’t return usage
        return text, usage

# ---------------------------
# Prompt helpers for Llama 3
# ---------------------------
def system_prompt(audience: str) -> str:
    return textwrap.dedent(f"""
    You are a careful data summarizer.
    RULES:
    - Only use facts present in the provided JSON.
    - Do not infer missing values or fabricate statistics.
    - If something is unclear or missing, add it to "caveats".
    - Tailor tone for the audience: {audience}.
    OUTPUT FORMAT:
    Return STRICT JSON with keys:
      summary (string), bullets? (string[]), insights? (string[]), caveats? (string[]), stats? (object of string|number).
    No markdown, no code fences, no extra keys.
    """).strip()

def user_prompt(task: str, style: str, word_limit: int, json_chunk: str) -> str:
    return textwrap.dedent(f"""
    TASK: {task}
    STYLE: {style}
    WORD_LIMIT: {word_limit}

    DATA_CHUNK_JSON:
    {json_chunk}

    Respond with STRICT JSON only.
    """).strip()

# ---------------------------
# Chunking (char-based; swap to semantic if you like)
# ---------------------------
def chunk_json(data: Any, target_chars: int = 12_000) -> List[str]:
    s = ujson.dumps(data, ensure_ascii=False)
    if len(s) <= target_chars:
        return [s]
    chunks = []
    N = math.ceil(len(s) / target_chars)
    for i in range(N):
        chunks.append(s[i*target_chars:(i+1)*target_chars])
    return chunks

# ---------------------------
# Map-Reduce summarization
# ---------------------------
def summarize_json(
    provider: LLMProvider,
    data: Any,
    task: str = "Executive summary of provided data",
    audience: str = "general",
    style: str = "paragraphs",  # or "bullet"
    word_limit: int = 200,
) -> tuple[SummaryPayload, str, Usage]:
    parts = chunk_json(data)
    usage_total: Usage = {"prompt": 0, "completion": 0, "total": 0}
    partials: List[SummaryPayload] = []

    # MAP
    for part in parts:
        messages = [
            {"role": "system", "content": system_prompt(audience)},
            {"role": "user", "content": user_prompt(task, style, word_limit, part)},
        ]
        text, usage = provider.generate(messages, temperature=0.1, max_tokens=700, json_mode=True)
        for k in ("prompt", "completion", "total"):
            usage_total[k] += int(usage.get(k, 0))

        # Try strict JSON parse; fallback makes the pipeline resilient
        try:
            obj = json.loads(text)
            partials.append(SummaryPayload.model_validate(obj))
        except Exception:
            partials.append(SummaryPayload(summary="Invalid JSON from model for this chunk.",
                                           caveats=["Malformed JSON from model"]))

    # REDUCE
    reducer_input = {
        "partials": [p.model_dump() for p in partials],
        "reduceStrategy": "merge_dedupe_arrays_keep_caveats_limit_summary",
        "word_limit": word_limit
    }
    reduce_messages = [
        {"role": "system", "content": "Merge chunk summaries into one concise, de-duplicated report. Keep it factual."},
        {"role": "user", "content": textwrap.dedent(f"""
        Combine the following JSON summaries into one. Keep "summary" ~{word_limit} words.
        Merge & dedupe bullets/insights/caveats. If stats have same keys, prefer numeric if consistent, else add caveat.
        Return STRICT JSON with keys: summary, bullets?, insights?, caveats?, stats?.
        INPUT:
        {json.dumps(reducer_input, ensure_ascii=False)}
        """).strip()}
    ]
    red_text, red_usage = provider.generate(reduce_messages, temperature=0.1, max_tokens=800, json_mode=True)
    for k in ("prompt", "completion", "total"):
        usage_total[k] += int(red_usage.get(k, 0))

    try:
        final = SummaryPayload.model_validate(json.loads(red_text))
    except (json.JSONDecodeError, ValidationError):
        final = SummaryPayload(summary="Invalid JSON during reduce step.",
                               caveats=["Malformed JSON from model in reduce"])

    return final, getattr(provider, "name", "unknown"), usage_total

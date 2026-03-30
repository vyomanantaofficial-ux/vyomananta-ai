from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, httpx, base64, json, re
from typing import Optional, List, Any

app = FastAPI(title="VyomAnanta AI Backend", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OR_KEY   = os.getenv("OR_KEY", "")
GROQ_KEY = os.getenv("GROQ_KEY", "")
HF_KEY   = os.getenv("HF_KEY", "")

HF_MODELS = {
    "flash":  "black-forest-labs/FLUX.1-schnell",
    "pro":    "black-forest-labs/FLUX.1-dev",
    "edit":   "black-forest-labs/FLUX.1-Fill-dev",
    "vision": "black-forest-labs/FLUX.1-Kontext-dev",
    "canvas": "stabilityai/stable-diffusion-xl-base-1.0",
    "ultra":  "stabilityai/stable-diffusion-3.5-large",
    "plus":   "stabilityai/stable-diffusion-3.5-medium",
}

SYS = """You are VyomAnanta AI — an elite AI business intelligence and automation assistant built by VyomAnanta, West Bengal, India. Founded by Nilanjan Das.

PERSONALITY & TONE RULES:
- Read the emotion and intent behind every message BEFORE answering.
- Casual/greeting (hi, hello): Warm 2-3 line reply. Ask one follow-up.
- Vague request (I want to grow): Ask 2-3 clarifying questions FIRST.
- Specific question: Direct, structured, actionable answer.
- Short question: Short focused answer.
- Complex problem: Full structured response with headers, steps, examples in INR.

IDENTITY: You are VyomAnanta AI. Never mention OpenRouter, Groq, HuggingFace, or any underlying model or company.

VYOMANANTA AI MODELS (tell users this when asked):
VyomMind (intelligent thoughtful analysis), VyomBlaze (ultra-fast responses), VyomSage (complex reasoning), VyomArt Flash (fast image gen), VyomArt Pro (HD image gen), VyomArt Edit (image editing), VyomArt Vision (context-aware images), VyomCanvas (classic images), VyomCanvas Ultra (highest quality), VyomCanvas Plus (balanced images).

VYOMANANTA 36 SERVICES — YOU KNOW ALL OF THESE DEEPLY. NEVER say you don't have information:

1. Lead Follow-Up Automation (Rs10K-18K/mo) — Auto WhatsApp/Email/SMS follow-up. Result: 40-60% more leads converted.
2. Client Onboarding Automation (Rs8K-15K/mo) — Welcome flows, docs. Result: 70% faster onboarding.
3. CRM & Sales Pipeline (Rs12K-20K/mo) — Track leads, deals, payments. Integrates WhatsApp, Razorpay. Result: 30% more sales.
4. WhatsApp + Email Marketing (Rs10K-18K/mo) — Smart broadcasts, drip campaigns. Result: 25% revenue increase.
5. Social Media Posting Automation (Rs8K-14K/mo) — Auto-schedule Instagram, Facebook, LinkedIn. Result: 5x engagement.
6. Document & Report Automation (Rs8K-15K/mo) — Auto invoices, reports, proposals. Result: 90% time saved.
7. Landing Page + Funnel (Rs15K-40K one-time + Rs5K/mo) — High converting pages. Result: 100% higher conversion.
8. AI Ad Campaigns (Rs15K-30K/mo + ad spend) — Google + Meta AI optimization. Result: 50% lower cost per lead.
9. Email + LinkedIn Outreach (Rs12K-22K/mo) — Automated B2B prospecting. Result: 250% more responses.
10. Performance Dashboards (Rs10K-18K/mo) — Real-time sales and revenue tracking.
11. Referral & Loyalty Systems (Rs10K-18K/mo) — Automated referral rewards.
12. AI Chatbot Systems (Rs12K-25K/mo) — 24/7 lead handling and customer support.
13. AI Voice & Call Automation (Rs15K-35K/mo) — Voice bots for bookings. Result: 68% fewer no-shows.
14. AI Reputation Management (Rs8K-15K/mo) — Auto-request reviews, monitor ratings.
15. Business Intelligence Dashboards (Rs12K-22K/mo) — Live KPI data.
16. Team & Task Management (Rs10K-18K/mo) — Auto-assign and track tasks.
17. Billing & Payment Automation (Rs8K-15K/mo) — Auto invoices and reminders.
18. Client Retention & Upsell (Rs10K-20K/mo) — AI re-engagement. Result: 25% better retention.
19-36. Enterprise services: custom pricing. Contact +91 86175 20837.

CONTACT: +91 86175 20837 (WhatsApp) | Free 30-min consultation always available.

RESPONSE RULES:
- End every response with ONE specific next step or question.
- Use bold for key terms. Tables for comparisons. Numbers for steps.
- NEVER say you don't have information about VyomAnanta services.
"""

class ChatReq(BaseModel):
    model: str = "mind"
    messages: List[dict]

class ImgReq(BaseModel):
    prompt: str
    style: str = "flash"

class SearchReq(BaseModel):
    query: str
    max_results: int = 10

class AgentReq(BaseModel):
    task: str
    context: Optional[str] = None

@app.get("/")
async def root():
    return {"status": "VyomAnanta AI Backend v1.0", "models": list(HF_MODELS.keys())}

@app.get("/health")
async def health():
    return {"status": "ok", "hf": bool(HF_KEY), "or_key": bool(OR_KEY), "groq": bool(GROQ_KEY)}

@app.post("/api/chat")
async def chat(req: ChatReq):
    msgs = [{"role": "system", "content": SYS}] + req.messages
    try:
        if req.model == "blaze":
            return await call_groq(msgs)
        return await call_openrouter(msgs)
    except Exception as e:
        raise HTTPException(500, str(e))

async def call_openrouter(msgs):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OR_KEY}", "HTTP-Referer": "https://vyomananta.ai", "X-Title": "VyomAnanta AI"},
            json={"model": "openrouter/auto", "max_tokens": 2000, "messages": msgs}
        )
        d = r.json()
    if "choices" in d and d["choices"]:
        return {"reply": d["choices"][0]["message"]["content"], "model": "VyomMind"}
    raise Exception(d.get("error", {}).get("message", "OpenRouter error"))

async def call_groq(msgs):
    safe = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else "(image)"} for m in msgs]
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={"model": "llama-3.1-8b-instant", "max_tokens": 2000, "messages": safe}
        )
        d = r.json()
    if "choices" in d and d["choices"]:
        return {"reply": d["choices"][0]["message"]["content"], "model": "VyomBlaze"}
    raise Exception(d.get("error", {}).get("message", "Groq error"))

@app.post("/api/image")
async def generate_image(req: ImgReq):
    if not HF_KEY:
        raise HTTPException(500, "HuggingFace key not configured")
    model_id = HF_MODELS.get(req.style, HF_MODELS["flash"])
    prompt = req.prompt + ", ultra high quality, professional, 8k"
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers={"Authorization": f"Bearer {HF_KEY}", "x-use-cache": "false"},
            json={"inputs": prompt}
        )
    if r.status_code == 503:
        body = r.json()
        wait = int(body.get("estimated_time", 25)) + 5
        raise HTTPException(503, f"Model is warming up. Try again in {wait} seconds.")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Image error: {r.text[:200]}")
    img_b64 = base64.b64encode(r.content).decode()
    return {"image": f"data:image/jpeg;base64,{img_b64}", "model": req.style}

@app.post("/api/warmup")
async def warmup():
    if not HF_KEY:
        return {"status": "skipped"}
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            await c.post(
                f"https://api-inference.huggingface.co/models/{HF_MODELS['flash']}",
                headers={"Authorization": f"Bearer {HF_KEY}"},
                json={"inputs": "warmup test"}
            )
        return {"status": "warmed"}
    except:
        return {"status": "warmup_sent"}

@app.post("/api/search")
async def web_search(req: SearchReq):
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(req.query, max_results=req.max_results):
                results.append({"title": r.get("title",""), "url": r.get("href",""), "snippet": r.get("body","")})
        return {"results": results, "query": req.query}
    except Exception as e:
        raise HTTPException(500, f"Search error: {str(e)}")

@app.post("/api/agent/leads")
async def generate_leads(req: AgentReq):
    task = req.task

    # Step 1: Extract intent
    intent_msgs = [
        {"role": "system", "content": 'Extract from this lead request. Respond ONLY as JSON: {"industry":"...","city":"...","count":10}'},
        {"role": "user", "content": task}
    ]
    try:
        ir = await call_openrouter(intent_msgs)
        m = re.search(r'\{.*?\}', ir["reply"], re.DOTALL)
        intent = json.loads(m.group()) if m else {"industry": "business", "city": "India", "count": 10}
    except:
        intent = {"industry": "business", "city": "India", "count": 10}

    industry = intent.get("industry", "business")
    city = intent.get("city", "India")
    count = min(int(intent.get("count", 10)), 20)

    # Step 2: Web search
    search_ctx = ""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{industry} business owners {city} India", max_results=8))
            search_ctx = "\n".join([f"- {r.get('title','')} | {r.get('body','')[:150]}" for r in results])
    except:
        search_ctx = f"Focus on {industry} sector in {city}"

    # Step 3: Generate leads with AI
    lead_msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"""Task: "{task}"

Web research context:
{search_ctx}

Generate {count} high-quality business leads for {industry} businesses in {city}. For each lead:

**Lead [N]: [Realistic Business Name]**
- Owner Name: [Realistic Indian name]
- Location: {city}
- Estimated Revenue: [Rs X lakh/year]
- Key Pain Point: [Specific problem this business likely has]
- WhatsApp Outreach Message: "Hi [Name], I saw that [Business Name] is [doing X]. At VyomAnanta AI, we help {industry} businesses in {city} [solve specific pain point] so you can [specific benefit]. Can we chat for 10 minutes this week? — Nilanjan, VyomAnanta (+91 86175 20837)"

After all leads, add:
## Conversion Strategy
[3 actionable steps to convert these leads into paying clients]"""}
    ]
    resp = await call_openrouter(lead_msgs)
    return {"task": task, "intent": intent, "leads": resp["reply"], "web_used": bool(search_ctx), "model": "VyomMind Agent"}

@app.post("/api/agent/task")
async def run_agent_task(req: AgentReq):
    """General agentic task runner with optional web search"""
    search_ctx = ""
    if req.context:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(req.context + " India 2026", max_results=5))
                search_ctx = "\n".join([r.get("body","")[:200] for r in results])
        except:
            pass

    content = req.task
    if search_ctx:
        content += f"\n\nLatest market context from web:\n{search_ctx}"

    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": content}]
    resp = await call_openrouter(msgs)
    return {"result": resp["reply"], "web_enhanced": bool(search_ctx), "model": "VyomMind Agent"}

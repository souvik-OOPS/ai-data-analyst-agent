# Submission

| Field | Value |
|---|---|
| **Candidate Name** | _<your name>_ |
| **GitHub Repository** | _<repository link>_ |
| **Demo Video** | _<video link>_ |
| **LLM Used** | Google Gemini 3.5 Flash (`gemini-3.5-flash`) / Google, via the official `google-genai` SDK, with automatic failover to other Gemini models. Anthropic Claude Opus 5.5 is also supported. Used for question understanding and tool selection (tool-calling loop), writing grounded answers, and rewriting dashboard insights. |
| **Database Used** | SQLite (explicit schema with constraints, indexes and analytical views; built from CSV by a pandas cleaning pipeline) |
| **Main Technologies** | Python 3.12, FastAPI, Pydantic, pandas, SQLite, google-genai / anthropic SDKs, Plotly.js, pytest, Docker |
| **Known Limitations** | Synthetic static dataset (relative dates anchored to the last order date; no cost data, so no profit). Grounding verifies numbers, not qualitative claims. The offline rule-based engine handles common phrasings only. The Gemini free tier has small daily quotas. No auth or streaming; single-instance SQLite; dashboard loads chart libraries from a CDN. |

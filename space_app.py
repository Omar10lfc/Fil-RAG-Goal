"""
FilGoalBot — Hugging Face Space entry point (Option A: no Docker, no Gradio UI)
==============================================================================
This serves the FastAPI backend on a Hugging Face *Gradio-SDK* Space without
ever rendering a Gradio interface. Gradio is used purely as the process
launcher (its server is FastAPI/Starlette under the hood); we mount our own
FastAPI app onto it, so users only ever hit the JSON API:

    GET  /          → API info        (api/main.py)
    GET  /health    → readiness probe
    POST /ask       → main Q&A endpoint
    GET  /docs      → OpenAPI UI

The real user-facing website is a separate frontend (e.g. Vercel/Netlify) that
calls this API. Point it at the Space URL and add its origin to
FILGOAL_ALLOWED_ORIGINS so CORS lets it through.

Deploy:
  - Space SDK:  gradio      (no Docker)
  - app_file:   space_app.py   ← set this in the Space README front matter
  - Secrets:    GROQ_API_KEY (required)
  - Optional:   FILGOAL_ALLOWED_ORIGINS=https://your-frontend.vercel.app

Why Gradio-as-launcher instead of Docker: HF's no-Docker Python SDKs are
gradio / streamlit only, and gradio is the one whose server can host an
arbitrary FastAPI app via gr.mount_gradio_app. The `gradio` dependency stays
in requirements.txt as plumbing; no Gradio UI is exposed.
"""

import os

import gradio as gr
import uvicorn

from api.main import app as fastapi_app

# A minimal Blocks instance the Gradio runtime can mount. It is intentionally
# empty and parked on an obscure path — there is no UI to show. Everything the
# user interacts with lives on the FastAPI routes at the root.
_placeholder = gr.Blocks()
app = gr.mount_gradio_app(fastapi_app, _placeholder, path="/_gradio_internal")


if __name__ == "__main__":
    # HF Spaces exposes the service on 7860; honour GRADIO_SERVER_PORT / PORT if
    # the platform sets one. Bind 0.0.0.0 so the Space's proxy can reach it.
    port = int(os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT") or "7860")
    uvicorn.run(app, host="0.0.0.0", port=port)

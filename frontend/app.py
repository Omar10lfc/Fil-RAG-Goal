"""
FilGoalBot — frontend/app.py
==============================
Gradio interface for the FilGoalBot Arabic football Q&A system.

Run:
    python -m frontend.app              # connects to local FastAPI on port 8000
    python -m frontend.app --api-url http://your-server:8000

Install:
    pip install gradio requests
"""

import re
import argparse
import requests
import gradio as gr
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_API = "http://127.0.0.1:8000"
TIMEOUT     = 30

INTENT_LABELS = {
    "match_result":     "🏆 نتيجة مباراة",
    "lineup":           "📋 تشكيلة",
    "player_info":      "⚽ معلومات لاعب",
    "team_news":        "📰 أخبار الفريق",
    "transfer_news":    "🔄 ميركاتو",
    "general_football": "🌍 كرة القدم",
}

EXAMPLE_QUESTIONS = [
    "ما نتيجة مباراة بيراميدز والجيش الملكي؟",
    "ما تشكيل الأهلي أمام المقاولون العرب؟",
    "آخر أخبار محمد صلاح في ليفربول",
    "آخر صفقات الزمالك في الميركاتو؟",
    "من سجل في مباراة الأهلي والاتحاد؟",
    "أخبار مران الزمالك قبل مباراة أبطال إفريقيا",
    "نتيجة مباراة برشلونة وريال مدريد؟",
    "هادي رياض في الأهلي",
]

# ── API client ────────────────────────────────────────────────────────────────

def ask_api(query: str, api_url: str) -> dict:
    """Call the /ask endpoint and return the result dict."""
    try:
        resp = requests.post(
            f"{api_url}/ask",
            json={"query": query},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ لا يمكن الاتصال بالخادم. تأكد من تشغيل الـ API."}
    except requests.exceptions.Timeout:
        return {"error": "❌ انتهت مهلة الاتصال. حاول مرة أخرى."}
    except Exception as e:
        return {"error": f"❌ خطأ: {str(e)}"}


def check_health(api_url: str) -> bool:
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = []
    for i, s in enumerate(sources, 1):
        title    = s.get("title", "")[:70]
        pub_date = s.get("pub_date", "")[:10]
        url      = s.get("url", "")
        league   = s.get("league", "")

        meta_parts = []
        if pub_date:
            meta_parts.append(pub_date)
        if league and league != "other":
            meta_parts.append(league)
        meta = " · ".join(meta_parts)

        if url:
            lines.append(f"**{i}.** [{title}]({url})  \n*{meta}*")
        else:
            lines.append(f"**{i}.** {title}  \n*{meta}*")

    return "\n\n".join(lines)


# ── Main chat function ────────────────────────────────────────────────────────

def chat(query, history, api_url, show_sources, show_intent):
    if not query.strip():
        return history, history, "", "", ""

    result = ask_api(query.strip(), api_url)

    if "error" in result:
        answer = result["error"]
        intent_text = ""
        sources_md  = ""
        latency_text = ""
    else:
        answer       = result.get("answer", "لا توجد إجابة")
        intent       = result.get("intent", "")
        sources      = result.get("sources", [])
        latency      = result.get("latency_ms", 0)
        intent_label = INTENT_LABELS.get(intent, intent)
        intent_text  = f"{intent_label}" if show_intent else ""
        sources_md   = format_sources(sources) if show_sources and sources else ""
        latency_text = f"{latency}ms" if latency else ""

    # Gradio 6 format — dicts with role/content instead of tuples
    history.append({"role": "user",      "content": query})
    history.append({"role": "assistant", "content": answer})

    return history, history, intent_text, sources_md, latency_text

def clear_chat():
    return [], [], "", "", ""


# ── Build Gradio UI ───────────────────────────────────────────────────────────

def build_ui(api_url: str):
    # Custom CSS for RTL Arabic support and styling
    css = """
    /* RTL support */
    .rtl-text, .rtl-text * {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', 'Cairo', 'Noto Naskh Arabic', sans-serif;
    }

    /* Chat bubbles */
    .message.user {
        direction: rtl;
        text-align: right;
    }
    .message.bot {
        direction: rtl;
        text-align: right;
    }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Status indicator */
    .status-online  { color: #22c55e; font-weight: bold; }
    .status-offline { color: #ef4444; font-weight: bold; }

    /* Intent badge */
    .intent-badge {
        display: inline-block;
        background: #0f3460;
        color: #e2e8f0;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 600;
        direction: rtl;
    }

    /* Source cards */
    .source-card {
        background: #1e293b;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        direction: rtl;
    }

    /* Latency chip */
    .latency-chip {
        font-family: monospace;
        font-size: 12px;
        color: #94a3b8;
        direction: ltr;
    }

    /* Example buttons */
    .example-btn {
        direction: rtl;
        text-align: right;
    }

    /* Input box */
    #query-input textarea {
        direction: rtl;
        text-align: right;
        font-size: 15px;
        font-family: 'Segoe UI', 'Cairo', sans-serif;
    }

    /* Chatbot messages */
    #chatbot .message-wrap {
        direction: rtl;
    }

    /* Submit button */
    #submit-btn {
        background: #0f3460 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    #submit-btn:hover {
        background: #1a4a8a !important;
    }
    """

    with gr.Blocks(
        css=css,
        title="FilGoalBot — مساعد كرة القدم",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Cairo"),
        ),
    ) as demo:

        # ── State ─────────────────────────────────────────────────────────────
        chat_history = gr.State([])
        api_url_state = gr.State(api_url)

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row():
            gr.HTML("""
            <div class="header-box">
                <h1 style="color:#f1f5f9; margin:0; font-size:28px; font-weight:700;">
                    ⚽ FilGoalBot
                </h1>
                <p style="color:#94a3b8; margin:8px 0 0; font-size:15px; direction:rtl;">
                    مساعد ذكي لأخبار كرة القدم المصرية والعربية والعالمية
                </p>
            </div>
            """)

        # ── API status ────────────────────────────────────────────────────────
        is_online = check_health(api_url)
        status_html = (
            '<span class="status-online">● متصل</span>'
            if is_online else
            '<span class="status-offline">● غير متصل — شغّل الـ API أولاً</span>'
        )
        with gr.Row():
            gr.HTML(f'<div style="text-align:center; padding:6px 0; direction:rtl;">حالة الخادم: {status_html}</div>')

        # ── Main layout ───────────────────────────────────────────────────────
        with gr.Row():
            # Left panel — chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="المحادثة",
                    height=480,
                    show_label=False,
                    rtl=True,
                    type="messages",
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        elem_id="query-input",
                        placeholder="اكتب سؤالك هنا... مثال: من سجل هدف الأهلي؟",
                        show_label=False,
                        scale=5,
                        lines=1,
                        max_lines=3,
                        rtl=True,
                    )
                    submit_btn = gr.Button(
                        "إرسال ➤",
                        elem_id="submit-btn",
                        scale=1,
                        variant="primary",
                    )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ مسح المحادثة", size="sm", variant="secondary")

            # Right panel — metadata
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### معلومات الإجابة", elem_classes=["rtl-text"])

                    intent_display = gr.Markdown(
                        label="نوع السؤال",
                        elem_classes=["rtl-text"],
                    )

                    latency_display = gr.Markdown(
                        label="زمن الاستجابة",
                        elem_classes=["rtl-text"],
                    )

                with gr.Accordion("📰 المصادر", open=True):
                    sources_display = gr.Markdown(
                        elem_classes=["rtl-text"],
                    )

                with gr.Accordion("⚙️ الإعدادات", open=False):
                    show_sources = gr.Checkbox(label="عرض المصادر", value=True)
                    show_intent  = gr.Checkbox(label="عرض نوع السؤال", value=True)
                    api_url_box  = gr.Textbox(
                        label="API URL",
                        value=api_url,
                        placeholder="http://127.0.0.1:8000",
                    )

        # ── Example questions ─────────────────────────────────────────────────
        gr.Markdown("### 💡 أسئلة مقترحة", elem_classes=["rtl-text"])
        with gr.Row():
            for q in EXAMPLE_QUESTIONS[:4]:
                gr.Button(q, size="sm", elem_classes=["example-btn"]).click(
                    fn=lambda x=q: x,
                    outputs=query_input,
                )
        with gr.Row():
            for q in EXAMPLE_QUESTIONS[4:]:
                gr.Button(q, size="sm", elem_classes=["example-btn"]).click(
                    fn=lambda x=q: x,
                    outputs=query_input,
                )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:16px 0 4px; color:#64748b; font-size:12px; direction:rtl;">
            FilGoalBot · مدعوم بـ FilGoal + Groq · البيانات مُحدَّثة يومياً
        </div>
        """)

        # ── Event wiring ──────────────────────────────────────────────────────

        def _chat(query, history, api_url_val, sources, intent):
            return chat(query, history, api_url_val, sources, intent)

        def _clear():
            return clear_chat()

        submit_inputs  = [query_input, chat_history, api_url_box, show_sources, show_intent]
        submit_outputs = [chatbot, chat_history, intent_display, sources_display, latency_display]

        submit_btn.click(
            fn=_chat,
            inputs=submit_inputs,
            outputs=submit_outputs,
        ).then(
            fn=lambda: "",
            outputs=query_input,
        )

        query_input.submit(
            fn=_chat,
            inputs=submit_inputs,
            outputs=submit_outputs,
        ).then(
            fn=lambda: "",
            outputs=query_input,
        )

        clear_btn.click(
            fn=_clear,
            outputs=[chatbot, chat_history, intent_display, sources_display, latency_display],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FilGoalBot Gradio Frontend")
    parser.add_argument("--api-url", default=DEFAULT_API, help="FastAPI backend URL")
    parser.add_argument("--port",    type=int, default=7860, help="Gradio port")
    parser.add_argument("--share",   action="store_true",    help="Create public Gradio link")
    args = parser.parse_args()

    print(f"\n🚀 Starting FilGoalBot frontend...")
    print(f"   API backend : {args.api_url}")
    print(f"   Gradio port : {args.port}")
    print(f"   Public link : {'yes' if args.share else 'no'}\n")

    demo = build_ui(api_url=args.api_url)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=None,
    )
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
    """Render sources as HTML cards. Uses HTML <a> tags (not markdown link
    syntax) because markdown links inside an HTML <div> aren't parsed by
    Gradio's renderer."""
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

        body = f'<a href="{url}" target="_blank" rel="noopener" class="source-link">{title}</a>' if url else title
        lines.append(f'<div class="source-card"><b>{i}.</b> {body}<br><i>{meta}</i></div>')

    return "\n".join(lines)


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
    # Theme inspired by filgoal.com — dark navy background with orange/gold accents.
    # CSS variables drive both light and dark modes; the toggle button at the top
    # flips a [data-theme] attribute on <html>, and the rest cascades from there.
    css = """
    /* ── Theme tokens ─────────────────────────────────────────────────── */
    :root, html[data-theme="dark"] {
        --bg:           #0f1115;
        --bg-elevated:  #181b22;
        --panel:        #1e2129;
        --panel-strong: #252834;
        --border:       #2a2e3a;
        --text:         #f1f3f7;
        --text-muted:   #9aa0aa;
        --accent:       #f5a623;
        --accent-hover: #ffb84d;
        --accent-soft:  rgba(245, 166, 35, 0.12);
        --user-bubble:  #2a2e3a;
        --bot-bubble:   #1e2129;
        --shadow:       0 4px 12px rgba(0,0,0,0.3);
        /* Override Gradio's internal theme tokens too — the Chatbot, gr.Group,
           and gr.Accordion read these, not our --bg-* vars. */
        --body-background-fill:       #0f1115 !important;
        --background-fill-primary:    #181b22 !important;
        --background-fill-secondary:  #1e2129 !important;
        --block-background-fill:      #1e2129 !important;
        --panel-background-fill:      #1e2129 !important;
        --input-background-fill:      #181b22 !important;
        --block-border-color:         #2a2e3a !important;
        --border-color-primary:       #2a2e3a !important;
        --color-text-primary:         #f1f3f7 !important;
        --body-text-color:            #f1f3f7 !important;
    }
    html[data-theme="light"] {
        --bg:           #f5f6fa;
        --bg-elevated:  #ffffff;
        --panel:        #ffffff;
        --panel-strong: #f0f1f5;
        --border:       #e2e4ea;
        --text:         #1a1d24;
        --text-muted:   #5c6370;
        --accent:       #e6951a;
        --accent-hover: #c97f12;
        --accent-soft:  rgba(230, 149, 26, 0.10);
        --user-bubble:  #f0f1f5;
        --bot-bubble:   #ffffff;
        --shadow:       0 2px 8px rgba(0,0,0,0.06);
        --body-background-fill:       #f5f6fa !important;
        --background-fill-primary:    #ffffff !important;
        --background-fill-secondary:  #f0f1f5 !important;
        --block-background-fill:      #ffffff !important;
        --panel-background-fill:      #ffffff !important;
        --input-background-fill:      #ffffff !important;
        --block-border-color:         #e2e4ea !important;
        --border-color-primary:       #e2e4ea !important;
        --color-text-primary:         #1a1d24 !important;
        --body-text-color:            #1a1d24 !important;
    }
    /* Source link styling — accent colour in both themes. */
    .source-link, .source-link:visited {
        color: var(--accent); text-decoration: none; font-weight: 600;
    }
    .source-link:hover { text-decoration: underline; color: var(--accent-hover); }
    /* Force text colour on Gradio-managed nested elements (chatbot bubbles,
       accordion bodies, group panels, inputs, source cards). Without this,
       light theme leaves white-on-white text. */
    .gradio-container #chatbot *:not(a),
    .gradio-container .gr-form *:not(a):not(button),
    .gradio-container .gr-group *:not(a):not(button),
    .gradio-container .gr-accordion *:not(a):not(button),
    .gradio-container .source-card *:not(a),
    .gradio-container .prose *:not(a),
    .gradio-container textarea,
    .gradio-container input {
        color: var(--text) !important;
    }

    /* ── Global ───────────────────────────────────────────────────────── */
    body, gradio-app, .gradio-container {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Cairo', 'Segoe UI', 'Noto Naskh Arabic', sans-serif !important;
    }
    .gradio-container * {
        border-color: var(--border) !important;
    }

    /* ── RTL support ──────────────────────────────────────────────────── */
    .rtl-text, .rtl-text * {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', 'Segoe UI', 'Noto Naskh Arabic', sans-serif;
    }

    /* ── Header ───────────────────────────────────────────────────────── */
    .header-box {
        background: linear-gradient(135deg, var(--bg-elevated) 0%, var(--panel) 100%);
        border-radius: 14px;
        padding: 28px 24px;
        margin-bottom: 18px;
        text-align: center;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    .header-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
    }
    .header-box h1 {
        color: var(--text) !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    .header-box .accent {
        color: var(--accent);
    }
    .header-box p {
        color: var(--text-muted) !important;
    }

    /* ── Theme toggle ─────────────────────────────────────────────────── */
    #theme-toggle {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 6px 14px !important;
        min-width: 0 !important;
    }
    #theme-toggle:hover {
        background: var(--accent-soft) !important;
        border-color: var(--accent) !important;
    }

    /* ── Status pill ──────────────────────────────────────────────────── */
    .status-online  { color: #4ade80; font-weight: 600; }
    .status-offline { color: #f87171; font-weight: 600; }

    /* ── Intent badge ─────────────────────────────────────────────────── */
    .intent-badge {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--accent);
        border-radius: 18px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 600;
        direction: rtl;
    }

    /* ── Source cards ─────────────────────────────────────────────────── */
    .source-card {
        background: var(--panel-strong);
        border-right: 3px solid var(--accent);
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        direction: rtl;
        color: var(--text);
    }

    /* ── Chatbot ──────────────────────────────────────────────────────── */
    #chatbot {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    #chatbot .message-wrap { direction: rtl; }
    #chatbot .message.user, #chatbot .user {
        background: var(--user-bubble) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }
    #chatbot .message.bot, #chatbot .bot {
        background: var(--bot-bubble) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Input box ────────────────────────────────────────────────────── */
    #query-input textarea {
        direction: rtl;
        text-align: right;
        font-size: 15px;
        font-family: 'Cairo', 'Segoe UI', sans-serif;
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }
    #query-input textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-soft) !important;
    }

    /* ── Submit button (orange/gold accent — the "اليوم" look) ─────── */
    #submit-btn {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
        color: #0f1115 !important;
        border: none !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 8px var(--accent-soft);
    }
    #submit-btn:hover {
        filter: brightness(1.1);
        transform: translateY(-1px);
    }

    /* ── Example & secondary buttons ──────────────────────────────────── */
    .example-btn {
        direction: rtl;
        text-align: right;
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }
    .example-btn:hover {
        background: var(--accent-soft) !important;
        border-color: var(--accent) !important;
    }

    /* ── Group/accordion panels ───────────────────────────────────────── */
    /* Gradio 4.x renders gr.Group/gr.Accordion/gr.Form with a mix of .block,
       .form, .gradio-* and svelte-hashed classes — list them all so the
       right sidebar follows the theme. */
    .gradio-container .block,
    .gradio-container .form,
    .gradio-container .gr-form,
    .gradio-container .gr-group,
    .gradio-container .gradio-group,
    .gradio-container .gr-accordion,
    .gradio-container .gradio-accordion,
    .gradio-container .panel,
    .gradio-container .padded {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    label, .gradio-container label {
        color: var(--text-muted) !important;
    }

    /* ── Latency chip ─────────────────────────────────────────────────── */
    .latency-chip {
        font-family: monospace;
        font-size: 12px;
        color: var(--text-muted);
        direction: ltr;
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

        # ── Header + theme toggle ─────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=10):
                gr.HTML("""
                <div class="header-box">
                    <h1 style="margin:0; font-size:30px;">
                        ⚽ <span class="accent">FilGoal</span>Bot
                    </h1>
                    <p style="margin:8px 0 0; font-size:15px; direction:rtl;">
                        مساعد ذكي لأخبار كرة القدم المصرية والعربية والعالمية
                    </p>
                </div>
                """)
            with gr.Column(scale=1, min_width=120):
                # Theme toggle — flips the data-theme attribute on <html>; CSS variables
                # cascade automatically.
                theme_toggle = gr.Button(
                    "☀️ فاتح",
                    elem_id="theme-toggle",
                    size="sm",
                )
                theme_toggle.click(
                    fn=None,
                    inputs=None,
                    outputs=theme_toggle,
                    js="""
                    () => {
                        const cur = document.documentElement.getAttribute('data-theme') || 'dark';
                        const next = cur === 'dark' ? 'light' : 'dark';
                        document.documentElement.setAttribute('data-theme', next);
                        return next === 'dark' ? '☀️ فاتح' : '🌙 مظلم';
                    }
                    """,
                )

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
        <div style="text-align:center; padding:16px 0 4px; color:var(--text-muted); font-size:12px; direction:rtl;">
            <span style="color:var(--accent); font-weight:700;">FilGoal</span>Bot · مدعوم بـ FilGoal + Groq · البيانات مُحدَّثة يومياً
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
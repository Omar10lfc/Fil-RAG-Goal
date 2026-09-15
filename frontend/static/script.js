/**
 * FilGoalBot — Frontend Logic
 * Handles chat interaction, API calls, theme toggle, and UI state.
 */

// ── Intent labels (Arabic) ─────────────────────────────────────────────────
const INTENT_LABELS = {
    match_result:     '🏆 نتيجة مباراة',
    lineup:           '📋 تشكيلة',
    player_info:      '⚽ معلومات لاعب',
    team_news:        '📰 أخبار الفريق',
    transfer_news:    '🔄 ميركاتو',
    general_football: '🌍 كرة القدم',
    out_of_scope:     '🚫 خارج النطاق',
};

// ── DOM refs ────────────────────────────────────────────────────────────────
const $messages   = document.getElementById('messages');
const $emptyState = document.getElementById('empty-state');
const $input      = document.getElementById('query-input');
const $sendBtn    = document.getElementById('send-btn');
const $clearBtn   = document.getElementById('clear-btn');
const $themeBtn   = document.getElementById('theme-toggle');
const $chatArea   = document.querySelector('.chat-area');

// ── State ───────────────────────────────────────────────────────────────────
let isLoading = false;
// Last completed exchange, sent back as conversation_context so follow-up
// questions ("وماذا حدث بعدها؟") resolve against it. Reset on clear.
let lastExchange = null;

function buildConversationContext() {
    if (!lastExchange) return undefined;
    const ctx = `السؤال السابق: ${lastExchange.q}\nالإجابة السابقة: ${lastExchange.a}`;
    return ctx.slice(0, 2000);
}

// ── Theme toggle ────────────────────────────────────────────────────────────
function getTheme() {
    return document.documentElement.getAttribute('data-theme') || 'dark';
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    $themeBtn.querySelector('.theme-icon').textContent =
        theme === 'dark' ? '☀️' : '🌙';
    localStorage.setItem('filgoal-theme', theme);
}

$themeBtn.addEventListener('click', () => {
    setTheme(getTheme() === 'dark' ? 'light' : 'dark');
});

// Restore saved theme
const savedTheme = localStorage.getItem('filgoal-theme');
if (savedTheme) setTheme(savedTheme);

// ── Textarea auto-resize ────────────────────────────────────────────────────
$input.addEventListener('input', () => {
    $input.style.height = 'auto';
    $input.style.height = Math.min($input.scrollHeight, 120) + 'px';
});

// ── Keyboard shortcuts ──────────────────────────────────────────────────────
$input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// ── Send button ─────────────────────────────────────────────────────────────
$sendBtn.addEventListener('click', handleSend);

// ── Clear button ────────────────────────────────────────────────────────────
$clearBtn.addEventListener('click', () => {
    $messages.innerHTML = '';
    lastExchange = null;
    $emptyState.classList.remove('hidden');
    $input.focus();
});

// ── Example chips ───────────────────────────────────────────────────────────
document.querySelectorAll('.example-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
        const query = chip.getAttribute('data-query');
        $input.value = query;
        $input.style.height = 'auto';
        handleSend();
    });
});

// ── Core: send message ──────────────────────────────────────────────────────
async function handleSend() {
    const query = $input.value.trim();
    if (!query || isLoading) return;

    // Hide empty state
    $emptyState.classList.add('hidden');

    // Add user message
    appendMessage('user', query);

    // Clear input
    $input.value = '';
    $input.style.height = 'auto';

    // Show typing indicator
    const loadingEl = appendLoading();
    scrollToBottom();

    // Lock UI
    isLoading = true;
    $sendBtn.disabled = true;

    try {
        const streamed = await tryStream(query, buildConversationContext(), loadingEl);
        if (!streamed) {
            // Streaming unavailable/failed — graceful fallback to /ask.
            loadingEl.remove();
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, conversation_context: buildConversationContext() }),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                const detail = errData.detail || `خطأ ${res.status}`;
                appendBotError(detail);
            } else {
                const data = await res.json();
                appendBotResponse(data);
                lastExchange = { q: query, a: data.answer || '' };
            }
        }
    } catch (err) {
        loadingEl.remove();
        appendBotError('تعذر الاتصال بالخادم. تحقق من اتصالك بالإنترنت.');
        console.error('Fetch error:', err);
    } finally {
        isLoading = false;
        $sendBtn.disabled = false;
        scrollToBottom();
        $input.focus();
    }
}

/**
 * Attempt token-streaming via POST /ask/stream (SSE).
 * Returns true when the stream completed (or rendered an error itself),
 * false when the caller should fall back to non-streaming /ask.
 */
async function tryStream(query, conversationContext, loadingEl) {
    let res;
    try {
        res = await fetch('/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, conversation_context: conversationContext }),
        });
    } catch {
        return false; // network error → fallback
    }
    if (!res.ok || !res.body) {
        if (res.status === 429) {
            loadingEl.remove();
            const errData = await res.json().catch(() => ({}));
            appendBotError(errData.detail || 'تم تجاوز حد الطلبات. حاول بعد قليل.');
            return true; // handled — don't double-report via fallback
        }
        return false; // other HTTP errors → fallback
    }

    // SSE parse: buffer text, split on blank line, handle event:/data: pairs.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let meta = null;
    let fullText = '';
    let msgEl = null;
    let bubbleEl = null;
    let donePayload = null;

    const ensureBubble = () => {
        if (bubbleEl) return;
        loadingEl.remove();
        msgEl = document.createElement('div');
        msgEl.className = 'message bot';
        msgEl.innerHTML = '<div class="message-bubble"></div>';
        $messages.appendChild(msgEl);
        bubbleEl = msgEl.querySelector('.message-bubble');
    };

    try {
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const blocks = buf.split('\n\n');
            buf = blocks.pop(); // keep last (possibly partial) block buffered
            for (const block of blocks) {
                let event = 'message';
                const dataLines = [];
                for (const line of block.split('\n')) {
                    if (line.startsWith('event:')) event = line.slice(6).trim();
                    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
                }
                if (!dataLines.length) continue;
                let payload;
                try {
                    payload = JSON.parse(dataLines.join('\n'));
                } catch {
                    continue;
                }
                if (event === 'meta') {
                    meta = payload;
                } else if (event === 'delta') {
                    ensureBubble();
                    fullText += payload.text || '';
                    bubbleEl.innerHTML = escapeHtml(fullText);
                    scrollToBottom();
                } else if (event === 'done') {
                    donePayload = payload;
                } else if (event === 'error') {
                    ensureBubble();
                    bubbleEl.classList.add('error-text');
                    bubbleEl.textContent = '❌ ' + (payload.detail || 'خطأ أثناء التوليد');
                    return true;
                }
            }
        }
    } catch {
        if (bubbleEl && fullText) {
            // Partial tokens already shown — keep them, note the cut, no fallback dup.
            bubbleEl.innerHTML = escapeHtml(fullText);
            const note = document.createElement('div');
            note.className = 'message-meta';
            note.textContent = '⚠️ انقطع البث — الرد الجزئي أعلاه';
            msgEl.appendChild(note);
            lastExchange = { q: query, a: fullText };
            return true;
        }
        return false; // nothing shown yet → fallback
    }

    if (!meta && !donePayload && !fullText) return false; // empty stream → fallback
    ensureBubble();
    if (!fullText) {
        bubbleEl.classList.add('error-text');
        bubbleEl.textContent = '❌ تعذر توليد الإجابة. حاول مرة أخرى.';
        return true;
    }
    finalizeStreamedMessage(msgEl, bubbleEl, fullText, meta, donePayload);
    lastExchange = { q: query, a: fullText };
    return true;
}

function finalizeStreamedMessage(msgEl, bubbleEl, fullText, meta, done) {
    bubbleEl.innerHTML = escapeHtml(fullText);
    meta = meta || {};
    done = done || {};
    const intent = meta.intent || '';
    const sources = meta.sources || [];
    const latencyMs = done.latency_ms || 0;
    const model = meta.model || done.model || '';
    const cached = done.cached || false;

    let html = '';
    const metaParts = [];
    if (intent && INTENT_LABELS[intent]) {
        metaParts.push(`<span class="intent-badge">${INTENT_LABELS[intent]}</span>`);
    }
    if (latencyMs) {
        metaParts.push(`<span class="latency-badge">⚡ ${latencyMs}ms${cached ? ' (مخبأ)' : ''}</span>`);
    }
    if (model) {
        const shortModel = String(model).split('/').pop();
        metaParts.push(`<span class="model-badge">${escapeHtml(shortModel)}</span>`);
    }
    if (metaParts.length) {
        html += `<div class="message-meta">${metaParts.join('')}</div>`;
    }
    if (sources.length) {
        html += `<div class="sources-container">`;
        html += `<span class="sources-label">📰 المصادر</span>`;
        for (const s of sources) {
            const title = escapeHtml((s.title || '').slice(0, 70));
            const url = s.url || '';
            const date = (s.pub_date || '').slice(0, 10);
            const league = s.league && s.league !== 'other' ? s.league : '';
            const srcMeta = [date, league].filter(Boolean).join(' · ');
            const link = isValidUrl(url)
                ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${title}</a>`
                : `<span>${title}</span>`;
            html += `
                <div class="source-card">
                    ${link}
                    ${srcMeta ? `<span class="source-meta">${escapeHtml(srcMeta)}</span>` : ''}
                </div>`;
        }
        html += `</div>`;
    }
    if (html) {
        const extra = document.createElement('div');
        extra.innerHTML = html;
        msgEl.appendChild(extra);
    }
}

// ── UI Helpers ──────────────────────────────────────────────────────────────

function appendMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
    $messages.appendChild(msg);
    scrollToBottom();
}

function appendBotResponse(data) {
    const msg = document.createElement('div');
    msg.className = 'message bot';

    const answer = data.answer || 'لا توجد إجابة';
    const intent = data.intent || '';
    const sources = data.sources || [];
    const latencyMs = data.latency_ms || 0;
    const model = data.model || '';
    const cached = data.cached || false;

    let html = `<div class="message-bubble">${escapeHtml(answer)}</div>`;

    // Meta row: intent + latency
    const metaParts = [];
    if (intent && INTENT_LABELS[intent]) {
        metaParts.push(`<span class="intent-badge">${INTENT_LABELS[intent]}</span>`);
    }
    if (latencyMs) {
        metaParts.push(`<span class="latency-badge">⚡ ${latencyMs}ms${cached ? ' (مخبأ)' : ''}</span>`);
    }
    if (model) {
        const shortModel = model.split('/').pop();
        metaParts.push(`<span class="model-badge">${shortModel}</span>`);
    }
    if (metaParts.length) {
        html += `<div class="message-meta">${metaParts.join('')}</div>`;
    }

    // Sources
    if (sources.length) {
        html += `<div class="sources-container">`;
        html += `<span class="sources-label">📰 المصادر</span>`;
        for (const s of sources) {
            const title = escapeHtml((s.title || '').slice(0, 70));
            const url = s.url || '';
            const date = (s.pub_date || '').slice(0, 10);
            const league = s.league && s.league !== 'other' ? s.league : '';
            const meta = [date, league].filter(Boolean).join(' · ');

            const link = isValidUrl(url)
                ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${title}</a>`
                : `<span>${title}</span>`;

            html += `
                <div class="source-card">
                    ${link}
                    ${meta ? `<span class="source-meta">${escapeHtml(meta)}</span>` : ''}
                </div>`;
        }
        html += `</div>`;
    }

    msg.innerHTML = html;
    $messages.appendChild(msg);
}

function appendBotError(text) {
    const msg = document.createElement('div');
    msg.className = 'message bot';
    msg.innerHTML = `<div class="message-bubble error-text">❌ ${escapeHtml(text)}</div>`;
    $messages.appendChild(msg);
}

function appendLoading() {
    const msg = document.createElement('div');
    msg.className = 'message bot';
    msg.innerHTML = `
        <div class="message-bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    $messages.appendChild(msg);
    return msg;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        $chatArea.scrollTop = $chatArea.scrollHeight;
    });
}

// ── Utilities ───────────────────────────────────────────────────────────────

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function isValidUrl(str) {
    try {
        const url = new URL(str);
        return url.protocol === 'http:' || url.protocol === 'https:';
    } catch {
        return false;
    }
}

// ── Focus input on load ─────────────────────────────────────────────────────
$input.focus();

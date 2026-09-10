/* AlphaAgent web chat UI — vanilla JS, no build step. */

"use strict";

const chatArea = document.getElementById("chat-area");
const welcome = document.getElementById("welcome");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const backendBadge = document.getElementById("backend-badge");
const sessionBadge = document.getElementById("session-badge");
const newChatBtn = document.getElementById("new-chat");

let sessionId = null;
let busy = false;

/* ---------- helpers ---------- */

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function scrollBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function addMessage(role, text, opts = {}) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = new Date().toLocaleTimeString();
  el.appendChild(meta);

  const body = document.createElement("div");
  body.innerHTML = esc(text);
  el.appendChild(body);

  if (opts.grounded) {
    const badge = document.createElement("div");
    badge.className = "grounded-badge";
    badge.textContent = "grounded in tool data";
    el.appendChild(badge);
  }
  if (opts.blocked) {
    const badge = document.createElement("div");
    badge.className = "empty-badge";
    badge.textContent = "blocked by guardrail";
    el.appendChild(badge);
  }

  chatArea.appendChild(el);
  scrollBottom();
  return el;
}

function addLoading() {
  const el = document.createElement("div");
  el.className = "loading";
  el.textContent = "Thinking…";
  chatArea.appendChild(el);
  scrollBottom();
  return el;
}

/* ---------- backend probing ---------- */

async function refreshHealth() {
  try {
    const resp = await fetch("/health");
    const data = await resp.json();
    backendBadge.textContent = `backend: ${data.backend} · ${data.model}`;
  } catch (_) {
    backendBadge.textContent = "backend: unreachable";
  }
}

/* ---------- session ---------- */

async function newSession() {
  sessionId = null;
  sessionBadge.textContent = "session: single-shot";
  chatArea.querySelectorAll(".msg").forEach((el) => el.remove());
  welcome.style.display = "";
}

/* ---------- ask ---------- */

async function ask(question) {
  busy = true;
  sendBtn.disabled = true;
  input.disabled = true;

  welcome.style.display = "none";
  addMessage("user", question);
  const loading = addLoading();

  const body = sessionId ? { question, session_id: sessionId } : { question };

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await resp.json();

    loading.remove();

    if (!resp.ok) {
      addMessage("assistant", `Error ${resp.status}: ${data.detail || "unknown error"}`);
    } else {
      addMessage("assistant", data.answer, {
        grounded: data.grounded && !data.blocked,
        blocked: data.blocked,
      });
      if (data.session_id) {
        sessionId = data.session_id;
        sessionBadge.textContent = `session: ${data.session_id.slice(0, 8)}…`;
      }
    }
  } catch (err) {
    loading.remove();
    addMessage("assistant", `Network error: ${err.message}`);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

/* ---------- events ---------- */

function onSend() {
  const q = input.value.trim();
  if (!q || busy) return;
  input.value = "";
  ask(q);
}

sendBtn.addEventListener("click", onSend);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") onSend();
});
input.addEventListener("input", () => {
  sendBtn.disabled = !input.value.trim() || busy;
});

document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    ask(btn.textContent.trim());
  });
});

newChatBtn.addEventListener("click", newSession);

/* ---------- init ---------- */
refreshHealth();
input.focus();

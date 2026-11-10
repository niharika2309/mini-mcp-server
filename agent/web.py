"""
agent/web.py
============

A zero-dependency web UI for the agent: chat pane + live tool timeline +
Approve/Deny buttons. It adds NO new packages (stdlib http.server only) and
no new logic - the SAME Agent object from agent/agent.py runs the loop; this
module only translates between HTTP/SSE and the agent's events/callbacks.

    python -m agent --web              # real LLM (needs LLM_* config)
    python -m agent --web --demo       # scripted Week-11 demo, no API key

    -> http://127.0.0.1:8765

Safeguards are unchanged: the publish tool still goes through the approval
gate (now a button click instead of a terminal y/N) and the server still
enforces confirm/overwrite/duplicate checks on its own. If nobody clicks a
decision within AGENT_APPROVAL_TIMEOUT seconds, the call is DENIED
(fail-closed). The server binds to 127.0.0.1 only - it is a local demo tool,
not a public web app.

How the pieces fit:

    browser --EventSource(/api/events)--> queue <--publish()-- Agent on_event
    browser --POST /api/chat-----------> threadpool --run_turn()--> event-loop thread
    browser --POST /api/approval-------> resolves the Future the
                                         approval_callback is awaiting

One asyncio event loop runs on a dedicated thread (the agent + MCP session
live there); HTTP handlers are plain threads that submit coroutines to it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import sys
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from .agent import Agent
from .config import AgentConfig
from .llm import LLMError, create_provider
from .mcp_client import McpClient

logger = logging.getLogger("agent.web")

DEFAULT_PORT = 8765


# ---------------------------------------------------------------------------
# The bridge: agent <-> HTTP
# ---------------------------------------------------------------------------
class WebAgentUI:
    """Owns the event-loop thread, the agent, live event fan-out and the
    pending approvals map. HTTP handlers call its public methods."""

    def __init__(
        self,
        provider_label: str,
        model_label: str,
        demo: bool = False,
        approval_timeout: float = 300.0,
    ):
        self.agent: Optional[Agent] = None
        self.provider_label = provider_label
        self.model_label = model_label
        self.demo = demo
        self.approval_timeout = approval_timeout

        self.loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        self.turn_lock = threading.Lock()          # one active turn at a time
        self.subscribers: List["queue.Queue[dict]"] = []
        self._subscribers_lock = threading.Lock()
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}

    def attach(self, agent: Agent) -> None:
        """Wire the Agent to the UI: events fan out to the browser, approvals
        come back through the browser's Approve/Deny buttons."""
        agent.on_event = self.publish
        agent.approval_callback = self._approval_callback
        self.agent = agent

    # -- lifecycle -----------------------------------------------------------
    def start_loop(self) -> None:
        def run() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.call_soon(self._loop_ready.set)
            self.loop.run_forever()

        self._loop_thread = threading.Thread(target=run, name="agent-loop", daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait(timeout=10)

    def stop_loop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)

    def submit(self, coro, timeout: Optional[float] = None):
        """Run a coroutine on the agent's event loop and wait for it."""
        self._loop_ready.wait(timeout=10)
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout)

    # -- events (agent thread -> browser) ------------------------------------
    def publish(self, event: str, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Accept both call styles: Agent._emit passes a dict positionally,
        internal code may pass keyword fields."""
        payload = data if data is not None else kwargs
        message = json.dumps({"event": event, "data": payload}, ensure_ascii=False, default=str)
        with self._subscribers_lock:
            targets = list(self.subscribers)
        for subscriber in targets:
            try:
                subscriber.put_nowait(message)
            except Exception:  # a slow/closed browser tab must not break the turn
                logger.debug("dropping event for a closed SSE subscriber")

    def subscribe(self) -> "queue.Queue[dict]":
        subscriber: "queue.Queue[dict]" = queue.Queue()
        with self._subscribers_lock:
            self.subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: "queue.Queue[dict]") -> None:
        with self._subscribers_lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

    # -- approvals (browser -> agent) ------------------------------------------
    async def _approval_callback(self, name: str, arguments: Dict[str, Any]) -> bool:
        """Runs on the loop thread inside Agent.run_turn."""
        call_id = uuid.uuid4().hex[:12]
        future: "asyncio.Future[bool]" = self.loop.create_future()
        tool = self.agent.mcp.tool(name)
        can_write = bool(tool and "confirm" in tool.input_schema.get("properties", {}))
        self.pending_approvals[call_id] = {
            "id": call_id, "name": name, "arguments": dict(arguments), "can_write": can_write,
            "future": future,
        }
        self.publish("approval_required", id=call_id, name=name, arguments=arguments, can_write=can_write)
        logger.info("approval requested for %s (%s)", name, call_id)
        try:
            approved = await asyncio.wait_for(future, self.approval_timeout)
            if not approved:
                # carries the card id so the browser can settle THAT card
                self.publish("approval_denied", id=call_id, name=name)
            return approved
        except asyncio.TimeoutError:
            # fail closed: silence is a "no"
            self.publish("approval_timeout", id=call_id, name=name)
            logger.warning("approval for %s timed out - denying", name)
            return False
        finally:
            self.pending_approvals.pop(call_id, None)

    def pending_approvals_snapshot(self) -> List[Dict[str, Any]]:
        """Public data (no futures) for replaying open approval cards to a
        fresh EventSource connection - so a page reload cannot lose a button."""
        return [
            {k: v for k, v in entry.items() if k != "future"}
            for entry in self.pending_approvals.values()
        ]

    def resolve_approval(self, call_id: str, approved: bool) -> bool:
        """Called from an HTTP thread; resolves the pending future."""

        def set_it() -> None:
            entry = self.pending_approvals.get(call_id)
            if entry is not None and not entry["future"].done():
                entry["future"].set_result(approved)

        self.loop.call_soon_threadsafe(set_it)
        return call_id in self.pending_approvals

    # -- turns -----------------------------------------------------------
    def run_turn(self, text: str) -> Dict[str, Any]:
        if not self.turn_lock.acquire(blocking=False):
            return {"ok": False, "busy": True, "error": "The agent is still working on your last message."}
        try:
            answer = self.submit(self.agent.run_turn(text), timeout=600)
            return {"ok": True, "text": answer}
        except LLMError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # keep the UI alive; log the detail
            logger.exception("agent turn failed")
            return {"ok": False, "error": f"Unexpected agent error: {exc}"}
        finally:
            self.turn_lock.release()

    def reset(self) -> Dict[str, Any]:
        if not self.turn_lock.acquire(blocking=False):
            return {"ok": False, "error": "Agent is busy; wait for the current turn to finish."}
        try:
            self.agent.messages = self.agent.messages[:1]
            return {"ok": True}
        finally:
            self.turn_lock.release()

    # -- view helpers ------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        if self.agent is None:  # requested before startup finished (shouldn't happen)
            return {"provider": self.provider_label, "model": self.model_label,
                    "server": "", "demo": self.demo, "tools": []}
        return {
            "provider": self.provider_label,
            "model": self.model_label,
            "server": self.agent.mcp.server_info,
            "demo": self.demo,
            "tools": [
                {
                    "name": t.name,
                    "description": (t.description or "").split("\n")[0],
                    "required": t.input_schema.get("required", []),
                    "approval": t.name in self.agent.config.approval_tools or t.is_destructive(),
                }
                for t in self.agent.mcp.tools
            ],
        }

    def history_blocks(self) -> List[Dict[str, Any]]:
        """Render the conversation context into displayable blocks (used when
        the page (re)loads mid-session)."""
        if self.agent is None:
            return []
        blocks: List[Dict[str, Any]] = []
        for message in self.agent.messages[1:]:
            role = message["role"]
            if role == "user":
                blocks.append({"kind": "user", "text": message["content"]})
            elif role == "assistant":
                if message.get("content"):
                    blocks.append({"kind": "agent", "text": message["content"]})
                for call in message.get("tool_calls") or []:
                    blocks.append({"kind": "tool_call", "name": call["name"], "arguments": call["arguments"]})
            elif role == "tool":
                content = message["content"]
                failed = '"success": false' in content or '"tool_error": true' in content
                blocks.append(
                    {
                        "kind": "tool_result",
                        "name": message.get("name", ""),
                        "ok": not failed,
                        "preview": " ".join(content.split())[:280],
                    }
                )
        return blocks


# ---------------------------------------------------------------------------
# HTTP + SSE layer
# ---------------------------------------------------------------------------
class RequestHandler(BaseHTTPRequestHandler):
    ui: WebAgentUI  # set on the class before serving

    protocol_version = "HTTP/1.0"  # close-delimited; simple and SSE-friendly

    def log_message(self, fmt: str, *args) -> None:  # route to logging, not stderr
        logger.info("%s - " + fmt, self.address_string(), *args)

    # -- helpers --
    def _is_local(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in {"127.0.0.1", "localhost", "::1", ""}

    def _json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser tab went away mid-response; nothing to report to

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    # -- routes --
    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        path = self.path.split("?", 1)[0]
        if not self._is_local():
            return self._json(403, {"error": "This UI only answers requests addressed to localhost."})
        if path in {"/", "/index.html"}:
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/info":
            self._json(200, self.ui.info())
        elif path == "/api/history":
            self._json(200, {"blocks": self.ui.history_blocks()})
        elif path == "/api/events":
            self._sse()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._is_local():
            return self._json(403, {"error": "This UI only answers requests addressed to localhost."})
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_body()
        except ValueError:
            return self._json(400, {"error": "invalid JSON body"})

        if path == "/api/chat":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json(400, {"error": "text is required"})
            self._json(200, self.ui.run_turn(text))
        elif path == "/api/approval":
            call_id = str(body.get("id") or "")
            approved = bool(body.get("approved"))
            found = self.ui.resolve_approval(call_id, approved)
            self._json(200 if found else 404, {"ok": found})
        elif path == "/api/reset":
            self._json(200, self.ui.reset())
        else:
            self._json(404, {"error": "not found"})

    def _sse(self) -> None:
        subscriber = self.ui.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            # Replay open approvals FIRST: a page reload during a waiting
            # approval must re-render the Approve/Deny buttons, never strand
            # the turn until the fail-closed timeout.
            for entry in self.ui.pending_approvals_snapshot():
                message = json.dumps({"event": "approval_required", "data": entry}, ensure_ascii=False)
                self.wfile.write(f"data: {message}\n\n".encode("utf-8"))
                self.wfile.flush()
            while True:
                try:
                    message = subscriber.get(timeout=20)
                    self.wfile.write(f"data: {message}\n\n".encode("utf-8"))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # keep-alive
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.ui.unsubscribe(subscriber)


# ---------------------------------------------------------------------------
# Entry point (called by cli.py on --web)
# ---------------------------------------------------------------------------
def serve(args, config: AgentConfig, mcp: McpClient, provider, demo: bool, open_browser: bool = True) -> int:
    ui = WebAgentUI(
        provider_label=getattr(provider, "name", config.provider),
        model_label=config.model,
        demo=demo,
        approval_timeout=config.approval_timeout,
    )
    ui.start_loop()

    # The MCP session (child process + anyio task groups) is born on the
    # agent's event-loop thread, where every later call also runs.
    try:
        ui.submit(mcp.start(), timeout=30)
    except Exception as exc:
        print(f"Could not start the MCP server: {exc}", file=sys.stderr)
        ui.stop_loop()
        return 1

    ui.attach(Agent(provider, mcp, config))

    port = args.port
    httpd: Optional[ThreadingHTTPServer] = None
    for candidate in range(port, port + 10):  # auto-skip a busy port
        try:
            RequestHandler.ui = ui
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), RequestHandler)
            break
        except OSError:
            logger.warning("port %d busy, trying the next one", candidate)
    if httpd is None:
        print(f"Could not find a free port near {port}", file=sys.stderr)
        ui.stop_loop()
        return 1

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"agent UI: {url}   (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
        return 0
    except KeyboardInterrupt:
        print("\nstopping...")
        return 0
    finally:
        httpd.shutdown()
        try:
            ui.submit(mcp.close(), timeout=10)
        except Exception:
            logger.debug("mcp close failed", exc_info=True)
        ui.stop_loop()


# ---------------------------------------------------------------------------
# The page itself: one self-contained file, vanilla JS, no external assets.
# All dynamic content is rendered with textContent (never innerHTML) so
# article text or model output can never inject markup.
# ---------------------------------------------------------------------------
HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-mcp agent</title>
<style>
  :root { --bg:#0f1216; --panel:#171c22; --border:#2a323c; --text:#dfe6ee;
          --dim:#8a97a6; --accent:#4cc2ff; --ok:#5dd48b; --err:#ff7b72; --warn:#e3b341; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  header { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
           padding:12px 18px; border-bottom:1px solid var(--border); background:var(--panel); }
  header h1 { font-size:15px; margin:0 8px 0 0; }
  .chip { font-size:12px; color:var(--dim); border:1px solid var(--border);
          border-radius:999px; padding:2px 10px; background:var(--bg); }
  .chip b { color:var(--text); font-weight:600; }
  main { display:flex; gap:16px; max-width:1100px; margin:0 auto; padding:16px; }
  #chat { flex:1 1 auto; min-width:0; display:flex; flex-direction:column; gap:8px; }
  aside { flex:0 0 250px; }
  .block { border:1px solid var(--border); border-radius:10px; padding:9px 12px;
           background:var(--panel); white-space:pre-wrap; word-break:break-word; }
  .block.user { border-color:#2d5f8a; }
  .block.user::before { content:"you"; color:var(--accent); font-weight:700; display:block; }
  .block.agent::before { content:"agent"; color:var(--ok); font-weight:700; display:block; }
  .block.tool { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
  .block.tool .t-ok { color:var(--ok); } .block.tool .t-err { color:var(--err); }
  .block.err { border-color:var(--err); color:var(--err); }
  .block.warn { border-color:var(--warn); color:var(--warn); }
  .block.note { color:var(--dim); }
  .approval { border-color:var(--warn); }
  .approval.active { box-shadow:0 0 0 2px var(--warn); }
  .approval .head { color:var(--warn); font-weight:700; }
  .approval pre { background:var(--bg); border:1px solid var(--border); border-radius:8px;
                  padding:8px; overflow:auto; font-size:12px; }
  .approval .btns { display:flex; gap:8px; margin-top:8px; }
  #waiting-banner { display:none; max-width:1100px; margin:0 auto 8px; padding:8px 16px;
                    color:var(--warn); font-weight:600; }
  button { font:inherit; border-radius:8px; padding:6px 16px; cursor:pointer;
           border:1px solid var(--border); background:var(--bg); color:var(--text); }
  button.approve { border-color:var(--ok); color:var(--ok); font-weight:700; }
  button.deny { border-color:var(--err); color:var(--err); }
  button:disabled { opacity:.5; cursor:default; }
  #tools { position:sticky; top:16px; }
  #tools h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
              color:var(--dim); margin:4px 0 8px; }
  .toolitem { border:1px solid var(--border); border-radius:8px; padding:6px 9px;
              margin-bottom:6px; background:var(--panel); font-size:12.5px; }
  .toolitem .name { font-family:ui-monospace,Menlo,monospace; }
  .toolitem .gate { color:var(--warn); font-size:11px; margin-left:6px; }
  #composer { display:flex; gap:8px; max-width:1100px; margin:0 auto 20px; padding:0 16px; width:100%; }
  #input { flex:1; font:inherit; color:var(--text); background:var(--panel);
           border:1px solid var(--border); border-radius:10px; padding:10px 12px; }
  #input:focus { outline:1px solid var(--accent); }
  #send { border-color:var(--accent); color:var(--accent); font-weight:700; }
  .muted { color:var(--dim); }
</style>
</head>
<body>
<header>
  <h1>mini-mcp agent</h1>
  <span class="chip" id="chip-provider">provider <b>-</b></span>
  <span class="chip" id="chip-model">model <b>-</b></span>
  <span class="chip" id="chip-server">mcp server <b>-</b></span>
  <span class="chip muted" id="chip-demo" style="display:none">demo mode (mock LLM)</span>
</header>
<main>
  <div id="chat"></div>
  <aside id="tools"><h2>Discovered tools</h2><div id="toollist"></div></aside>
</main>
<div id="waiting-banner">\u26a0 The agent is waiting for your decision: click Approve or Deny on the highlighted card above. Nothing runs until you choose.</div>
<form id="composer">
  <input id="input" autocomplete="off" placeholder="e.g. Find my latest Week 11 article and preview it">
  <button id="send" type="submit">Send</button>
</form>
<script>
"use strict";
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

function api(path, body) {
  return fetch("/api/" + path, body
    ? { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) }
    : {}).then(r => r.json());
}
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;  // textContent only: no HTML injection
  return node;
}
function add(kind, text, cls) {
  const block = el("div", "block " + (cls || kind), text);
  chat.appendChild(block);
  chat.scrollTop = chat.scrollHeight;
  return block;
}
function fmt(obj) { try { return JSON.stringify(obj); } catch { return String(obj); } }

function addToolCall(name, args) {
  add("tool", "\\u2699 " + name + " " + fmt(args), "tool");
}
function addToolResult(name, ok, preview) {
  const block = add("tool", "", "tool");
  block.appendChild(el("span", ok ? "t-ok" : "t-err", (ok ? "\\u2714 " : "\\u2716 ") + name + "  "));
  block.appendChild(el("span", "muted", preview));
}
const shownApprovals = new Set();        // dedupe live events vs reconnect/reload replays
const liveApprovals = new Map();         // id -> card element, while awaiting a decision
const waitingBanner = document.getElementById("waiting-banner");
function refreshWaitingUi() {
  waitingBanner.style.display = liveApprovals.size ? "block" : "none";
  if (liveApprovals.size) sendBtn.textContent = "Waiting for approval\\u2026";
  else if (!busy) sendBtn.textContent = "Send";
}
function resolveApprovalCard(id, label) {
  const block = liveApprovals.get(id);
  if (!block) return false;
  liveApprovals.delete(id);
  block.classList.remove("active");
  for (const btn of block.querySelectorAll("button")) btn.disabled = true;
  const btns = block.querySelector(".btns");
  if (btns && label) btns.appendChild(el("span", "muted", label));
  refreshWaitingUi();
  return true;
}
function addApproval(data) {
  if (!data || !data.id || shownApprovals.has(data.id)) return;
  shownApprovals.add(data.id);
  const block = add("approval", "", "approval");
  liveApprovals.set(data.id, block);
  block.classList.add("active");
  block.scrollIntoView({behavior:"smooth", block:"center"});
  refreshWaitingUi();
  block.appendChild(el("div", "head",
      "\\u26a0 " + data.name + " needs your approval"));
  if (data.can_write)
    block.appendChild(el("div", "muted",
      "This can WRITE to GitHub. Check the preview output above for the exact file path and content."));
  block.appendChild(el("pre", null, JSON.stringify(data.arguments, null, 2)));
  const btns = el("div", "btns");
  const approve = el("button", "approve", "Approve");
  const deny = el("button", "deny", "Deny");
  const finish = (label, approved) => {
    if (resolveApprovalCard(data.id, label))
      api("approval", { id: data.id, approved: approved });
  };
  approve.onclick = () => finish("Approved by you", true);
  deny.onclick = () => finish("Denied by you", false);
  btns.appendChild(approve); btns.appendChild(deny);
  block.appendChild(btns);
}

const handlers = {
  assistant_text: d => add("agent", d.text),
  tool_call:      d => addToolCall(d.name, d.arguments),
  tool_result:    d => addToolResult(d.name, d.ok,
                       (typeof d.payload === "string" ? d.payload : fmt(d.payload)).slice(0, 220)),
  tool_error:     d => addToolResult(d.name, false, "rejected: " + String(d.error).slice(0, 220)),
  approval_denied:d => { if (resolveApprovalCard(d.id, "Denied by you")) add("tool", "\\u26d4 " + d.name + " DENIED by user \\u2014 not sent to the server", "tool"); },
  approval_confirmed: d => add("note", "human approved \\u2014 setting confirm=true for the server's own check"),
  approval_timeout: d => { if (resolveApprovalCard(d.id, "No decision \\u2014 auto-denied")) add("warn", "\\u23f3 No decision on " + d.name + " \\u2014 auto-denied"); },
  limit:          d => add("warn", "\\u23f9 " + d.notice),
};
new EventSource("/api/events").onmessage = e => {
  const msg = JSON.parse(e.data);
  const h = handlers[msg.event];
  if (h) h(msg.data || {});
};

let busy = false;
function setBusy(b) {
  busy = b;
  input.disabled = b; sendBtn.disabled = b;
  sendBtn.textContent = b ? "Thinking\\u2026" : "Send";
}
document.getElementById("composer").addEventListener("submit", async ev => {
  ev.preventDefault();
  const text = input.value.trim();
  if (!text || busy) return;
  input.value = "";
  add("user", text);
  setBusy(true);
  try {
    const res = await api("chat", { text });
    if (res.busy || (!res.ok && res.error)) add("err", res.error);
    else if (res.ok && res.text) add("agent", res.text);
  } catch (err) {
    add("err", "Connection to the agent backend failed: " + err);
  } finally {
    setBusy(false);
    input.focus();
  }
});

// initial state
api("info").then(info => {
  document.querySelector("#chip-provider b").textContent = info.provider;
  document.querySelector("#chip-model b").textContent = info.model;
  document.querySelector("#chip-server b").textContent = info.server
      + " \\u00b7 " + info.tools.length + " tools";
  if (info.demo) document.getElementById("chip-demo").style.display = "";
  const list = document.getElementById("toollist");
  for (const t of info.tools) {
    const item = el("div", "toolitem");
    item.appendChild(el("span", "name", t.name));
    if (t.approval) item.appendChild(el("span", "gate", "\\u26a0 approval"));
    item.appendChild(el("div", "muted", t.description));
    list.appendChild(item);
  }
});
api("history").then(h => {
  for (const b of h.blocks) {
    if (b.kind === "user") add("user", b.text);
    else if (b.kind === "agent") add("agent", b.text);
    else if (b.kind === "tool_call") addToolCall(b.name, b.arguments);
    else if (b.kind === "tool_result") addToolResult(b.name, b.ok, b.preview);
  }
});
</script>
</body>
</html>
"""

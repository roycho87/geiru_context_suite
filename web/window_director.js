// Geiru Window Director: buttons and a window strip, in the style of Geiru Chain
// (whose buttons come from NikoDemon80's ComfyUI-H3-Motion-Context Chain node, GPL-3.0).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "GeiruWindowDirector";
const EVENT = "geiru_window_director.status";

const CSS = `
.gwd{display:flex;flex-direction:column;gap:6px;color:#ddd;font:12px/1.3 sans-serif;width:100%;box-sizing:border-box;padding:2px 0 4px;}
.gwd-row{display:flex;gap:6px;}
.gwd-row button{flex:1;cursor:pointer;border:1px solid #555;background:#2a2a2a;color:#ddd;border-radius:4px;padding:6px 4px;font:12px sans-serif;}
.gwd-row button:hover{background:#3a3a3a;}
.gwd-row button:disabled{opacity:.4;cursor:default;background:#2a2a2a;}
.gwd-row button.on{border-color:#6f8bbd;background:#1f2a3a;color:#c5d4ee;}
.gwd-meta{opacity:.85;font-size:11px;min-height:14px;}
.gwd-strip{display:flex;flex-wrap:wrap;gap:3px;}
.gwd-sq{min-width:22px;flex:1 0 22px;max-width:40px;height:18px;border-radius:3px;font-size:10px;text-align:center;line-height:18px;cursor:pointer;background:#3a2430;color:#b08898;box-sizing:border-box;}
.gwd-sq.ok{background:#66bb6a;color:#111;}
.gwd-sq.cur{background:#ffca28;color:#111;}
.gwd-sq.run{background:#5c8dd6;color:#fff;animation:gwdpulse 1.2s ease-in-out infinite;}
@keyframes gwdpulse{0%,100%{opacity:1}50%{opacity:.45}}
.gwd-sq.sel{outline:2px solid #fff;outline-offset:1px;}
`;
const TITLE_PINK = "#c2407a";
const BODY_PINK = "#6e2347";

let cssOnce = false;
function injectCss() {
  if (cssOnce) return;
  cssOnce = true;
  const s = document.createElement("style");
  s.textContent = CSS;
  document.head.appendChild(s);
}

function swallow(el) {
  for (const t of ["pointerdown", "mousedown", "pointerup", "mouseup", "click", "dblclick", "contextmenu"]) {
    el.addEventListener(t, (e) => e.stopPropagation());
  }
}

const W = (node, name) => node.widgets?.find((w) => w.name === name);
const runName = (node) => String(W(node, "run_name")?.value ?? "take1");
const safe = (n) => (String(n).trim().replace(/[^A-Za-z0-9\-_.]/g, "_").replace(/^[._]+|[._]+$/g, "") || "run");

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
  w.type = "hidden";
}

let live = null;      // the Director node whose render is running

function paint(node) {
  const g = node._gwd;
  if (!g) return;
  const s = g.status || {};
  const total = s.total | 0, approved = s.approved | 0, pending = s.pending | 0;
  const sel = W(node, "rewind_to_window")?.value | 0;
  const made = Math.max(approved, pending);
  g.strip.innerHTML = "";
  if (!total) {
    const t = document.createElement("div");
    t.className = "gwd-meta";
    t.textContent = "Run once to see the windows.";
    g.strip.appendChild(t);
  } else {
    for (let i = 1; i <= total; i++) {
      const sq = document.createElement("div");
      const run = g.awaiting && i === g.rendering;
      sq.className = "gwd-sq" + (run ? " run" : (i <= approved ? " ok" : (i === pending ? " cur" : ""))) + (i === sel ? " sel" : "");
      sq.textContent = i;
      sq.title = (run ? `window ${i}: rendering` : i <= approved ? `window ${i}: approved` : (i === pending ? `window ${i}: ready to look at` : `window ${i}: not made yet`))
                 + (i <= made ? " · click to select it for Rewind to selected" : "");
      sq.onclick = (e) => {
        e.stopPropagation();
        if (i > made) return;            // can only rewind to a window that exists
        const w = W(node, "rewind_to_window");
        if (w) w.value = i;
        node.setDirtyCanvas?.(true, true);
        paint(node);
      };
      g.strip.appendChild(sq);
    }
  }
  let txt;
  if (!total) txt = "Nothing made yet.";
  else if (s.blended) txt = s.joined ? `All ${total} windows joined.` : `All ${total} windows blended.`;
  else txt = `Window ${pending || Math.min(approved + 1, total)} of ${total} · ` +
             (approved ? `windows 1–${approved} approved` : "none approved yet");
  if (g.directing) txt += " · Directing";
  if (s.msg && !s.blended) txt += ` · ${s.msg}`;
  g.meta.textContent = txt;
  const busy = !!g.awaiting;
  g.approve.disabled = busy || (total && made >= total && approved >= total);
  g.reroll.disabled = busy;
  g.rewind.disabled = busy || !made || !sel || sel > made;
  g.blend.disabled = busy || !total || made < total;           // greyed out until every window is made
  const bypass = !!W(node, "bypass_blend")?.value;
  g.blend.textContent = bypass ? "Join all" : "Blend all";
  g.blend.title = bypass ? "Once every window is made: join them as approved, no blend steps."
                         : "Once every window is made: the blended last steps over the whole video.";
  g.reset.disabled = busy;
  g.direct.textContent = g.directing ? "Stop" : "Direct";
  g.direct.classList.toggle("on", !!g.directing);
  g.direct.disabled = !g.directing && (busy || (total && made >= total));
  fit(node);
}

// the height the buttons, status line and strip need. Measured from the pieces themselves, not from the
// box around them: that box is stretched to the node's size, so measuring it made the node grow on every reload.
// null while the page is still laying it out.
function contentHeight(root) {
  const kids = [...root.children];
  if (!root.isConnected || !root.offsetWidth || !kids.length) return null;
  const h = kids.reduce((a, k) => a + k.offsetHeight, 0);
  return h ? h + 6 * (kids.length - 1) + 6 + 6 : null;   // gaps, padding, a little room
}

// keep the node at least as tall as its buttons and strip, so nothing spills out of it
function fit(node) {
  requestAnimationFrame(() => {
    if (contentHeight(node._gwd?.root ?? document.createElement("div")) == null) return;
    const min = node.computeSize?.();
    if (!min) return;
    if (node.size[1] < min[1] || node.size[0] < min[0]) {
      node.setSize([Math.max(node.size[0], min[0]), Math.max(node.size[1], min[1])]);
      node.setDirtyCanvas?.(true, true);
    }
  });
}

async function refresh(node) {
  try {
    const r = await api.fetchApi(`/geiru_window_director/status?run=${encodeURIComponent(runName(node))}`);
    if (r.ok) node._gwd.status = await r.json();
  } catch (e) { /* server not ready */ }
  paint(node);
}

async function press(node, action) {
  const g = node._gwd;
  if (g.awaiting) return;
  const a = W(node, "action");
  if (a) a.value = action;
  // show it straight away: Approve turns the current window green, and the window being made pulses blue
  const s = g.status ? { ...g.status } : null;
  if (s && s.total) {
    let target;
    if (action === "approve") {
      if (s.pending) { s.approved = Math.max(s.approved | 0, s.pending); s.pending = 0; }
      target = (s.approved | 0) + 1;
    } else if (action === "rewind") {
      target = Math.max(1, Math.min(W(node, "rewind_to_window")?.value | 0, s.total));
      s.approved = Math.min(s.approved | 0, target - 1); s.pending = 0;
    } else if (action === "reroll") {
      target = s.pending || Math.min((s.approved | 0) + 1, s.total);
    }
    g.rendering = target && target <= s.total ? target : 0;
    s.msg = g.rendering ? `rendering window ${g.rendering}` : s.msg;
    g.status = s;
  } else {
    g.rendering = 0;
  }
  g.awaiting = true;
  g.started = false;
  live = node;
  paint(node);
  let queued = true;
  try {
    const res = await app.queuePrompt(0, 1);
    if (res === false) queued = false;
  } catch (err) {
    queued = false;
  } finally {
    if (a) a.value = "reroll";          // a plain Run afterwards only re-rolls the current window
  }
  if (queued) {
    // ComfyUI can turn a run down before it starts (a missing input, a bad value): then no "finished" or "failed"
    // message ever comes, so check that it actually started or is waiting in the queue
    await new Promise((r) => setTimeout(r, 1500));
    if (g.awaiting && !g.started) {
      try {
        const q = await (await api.fetchApi("/queue")).json();
        if (!(q.queue_running?.length || q.queue_pending?.length)) queued = false;
      } catch (err) { /* can't tell: keep waiting */ }
    }
  }
  if (!queued && g.awaiting) rejected(node);
}

// the run was turned down before it started: give the buttons back and say so
function rejected(node) {
  const g = node._gwd;
  g.awaiting = false;
  g.directing = false;
  g.rendering = 0;
  if (live === node) live = null;
  refresh(node).then(() => {
    if (g.status) g.status.msg = "ComfyUI didn't run it - check its error message";
    paint(node);
  });
}

function done(ok) {
  const node = live;
  if (!node?._gwd?.awaiting) return;
  const g = node._gwd;
  g.awaiting = false;
  g.rendering = 0;
  live = null;
  if (!ok) { g.directing = false; refresh(node); }
  const s = g.status || {};
  if (g.directing) {
    const made = Math.max(s.approved | 0, s.pending | 0);
    if (s.total && made >= s.total) g.directing = false;
    else { paint(node); press(node, "approve"); return; }
  }
  paint(node);
}

api.addEventListener(EVENT, (e) => {
  const s = e.detail || {};
  for (const n of app.graph?._nodes || app.graph?.nodes || []) {
    if (n.comfyClass === NODE && n._gwd && safe(runName(n)) === s.run) {
      n._gwd.status = s;
      paint(n);
    }
  }
});
api.addEventListener("execution_start", () => { if (live?._gwd) live._gwd.started = true; });
api.addEventListener("execution_success", () => done(true));
api.addEventListener("execution_error", () => done(false));
api.addEventListener("execution_interrupted", () => done(false));

app.registerExtension({
  name: "geiru.context.suite.window_director",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      injectCss();
      this.color = TITLE_PINK;
      this.bgcolor = BODY_PINK;
      hideWidget(W(this, "action"));
      hideWidget(W(this, "rewind_to_window"));
      const root = document.createElement("div");
      root.className = "gwd";
      const mk = (label, tip) => { const b = document.createElement("button"); b.textContent = label; b.title = tip; return b; };
      const approve = mk("Approve", "Keep this window and make the next one.");
      const reroll = mk("Run/Re-roll", "Make the current window again. Use this instead of ComfyUI's Run button.");
      const rewind = mk("Rewind to selected", "Click a window in the strip, then press this to go back to it and make it again. The windows after it are dropped.");
      const direct = mk("Direct", "Approve on a loop until every window is made, or until Stop.");
      const blend = mk("Blend all", "Once every window is made: the blended last steps over the whole video.");
      const reset = mk("Reset", "Clear this run's saved windows. Doesn't queue anything.");
      const rows = [[approve, reroll], [rewind, direct], [blend, reset]].map((bs) => {
        const row = document.createElement("div");
        row.className = "gwd-row";
        row.append(...bs);
        return row;
      });
      const meta = document.createElement("div");
      meta.className = "gwd-meta";
      const strip = document.createElement("div");
      strip.className = "gwd-strip";
      root.append(...rows, meta, strip);
      swallow(root);
      const dom = this.addDOMWidget("window_director", "GEIRU_WINDOW_DIRECTOR", root, {
        serialize: false,
        getMinHeight: () => contentHeight(root) ?? 150,
      });
      dom.computeSize = (width) => [width, contentHeight(root) ?? 150];
      this._gwd = { root, status: null, awaiting: false, directing: false, meta, strip, approve, reroll, rewind, direct, blend, reset };
      approve.onclick = (e) => { e.stopPropagation(); this._gwd.directing = false; press(this, "approve"); };
      reroll.onclick = (e) => { e.stopPropagation(); this._gwd.directing = false; press(this, "reroll"); };
      rewind.onclick = (e) => { e.stopPropagation(); this._gwd.directing = false; press(this, "rewind"); };
      blend.onclick = (e) => { e.stopPropagation(); this._gwd.directing = false; press(this, "blend"); };
      direct.onclick = (e) => {
        e.stopPropagation();
        if (this._gwd.directing) { this._gwd.directing = false; paint(this); return; }
        this._gwd.directing = true;
        press(this, "approve");
      };
      reset.onclick = async (e) => {
        e.stopPropagation();
        if (this._gwd.awaiting) return;
        if (!confirm(`Clear every saved window of run '${runName(this)}'?`)) return;
        this._gwd.directing = false;
        try {
          const r = await api.fetchApi("/geiru_window_director/reset", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run: runName(this) }),
          });
          if (r.ok) this._gwd.status = await r.json();
        } catch (err) { /* ignore */ }
        paint(this);
      };
      for (const name of ["run_name", "rewind_to_window", "bypass_blend"]) {
        const w = W(this, name);
        if (!w) continue;
        const prev = w.callback;
        w.callback = (...args) => { const out = prev?.apply(w, args); refresh(this); return out; };
      }
      this.setSize?.([360, 300]);
      refresh(this);
      return r;
    };
    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const r = onResize?.apply(this, arguments);
      const min = this.computeSize?.();
      if (min && size) { size[0] = Math.max(size[0], min[0]); size[1] = Math.max(size[1], min[1]); }
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      // a node saved by an older version can have its values one slot off: put back any that don't fit
      for (const w of this.widgets || []) {
        const vals = w.options?.values;
        if (Array.isArray(vals) && !vals.includes(w.value)) w.value = vals.includes("reroll") ? "reroll" : vals[0];
        if (["new_seed_on_reroll", "keep_late_steps", "bypass_blend"].includes(w.name) && typeof w.value !== "boolean") w.value = false;
        if (w.name === "rewind_to_window" && !(Number.isInteger(w.value) && w.value >= 1)) w.value = 1;
        if (w.name === "run_name" && typeof w.value !== "string") w.value = "take1";
      }
      hideWidget(W(this, "action"));
      hideWidget(W(this, "rewind_to_window"));
      if (this._gwd) { this._gwd.awaiting = false; refresh(this); }
      // a node saved while it was growing: bring it back down to what it needs, once it's laid out
      const shrink = (tries) => requestAnimationFrame(() => {
        if (!this._gwd || contentHeight(this._gwd.root) == null) { if (tries > 0) shrink(tries - 1); return; }
        const min = this.computeSize?.();
        if (min && this.size[1] > min[1] + 40) { this.setSize([this.size[0], min[1]]); this.setDirtyCanvas?.(true, true); }
      });
      shrink(60);
      return r;
    };
  },
});

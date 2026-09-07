/* LLMTrace local web UI — vanilla JS（无 React / 无构建流水线） */

"use strict";

function el(tag, attrs, text) {
  const node = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtIso(v) {
  return v ? String(v).replace("T", " ").replace(/\.\d+Z?$/, "") : "—";
}

/* ---------- 通用 API 封装 ---------- */
async function api(url, options) {
  const res = await fetch(url, options);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-json */ }
  if (!res.ok) {
    const msg = (data && data.error && data.error.message) ||
      (data && data.error && data.error.code) ||
      (data && data.detail && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) ||
      `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

/* ---------- New Audit 表单 ---------- */
function bindAuditForm() {
  const form = document.getElementById("audit-form");
  if (!form) return;

  const refSelect = document.getElementById("f-reference");
  api("/api/reference-sets")
    .then((d) => {
      for (const rs of d.reference_sets || []) {
        const label = rs.reference_set_id + (rs.reference_set_version ? " v" + rs.reference_set_version : "");
        refSelect.appendChild(el("option", { value: rs.path }, `${label} (${rs.members ?? "?"} members)`));
      }
    })
    .catch(() => { /* reference sets 可选，加载失败静默 */ });

  const errorBox = document.getElementById("form-error");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorBox.hidden = true;
    const submitBtn = document.getElementById("btn-submit");
    submitBtn.disabled = true;
    submitBtn.textContent = "Creating…";
    const fd = new FormData(form);
    const payload = {
      base_url: fd.get("base_url").trim(),
      model: fd.get("model").trim(),
      api_key: fd.get("api_key"),
      protocol: fd.get("protocol"),
      auth_style: fd.get("auth_style"),
      repeat: parseInt(fd.get("repeat"), 10),
      timeout: parseFloat(fd.get("timeout")),
      check_streaming: fd.get("check_streaming") === "on",
      reference_set_path: fd.get("reference_set_path") || null,
    };
    try {
      const data = await api("/api/runs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      window.location.href = "/run/" + data.run.run_id;
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.hidden = false;
      submitBtn.disabled = false;
      submitBtn.textContent = "Create & Estimate";
    }
  });
}

/* ---------- Run 页：SSE live + Result ---------- */
function statusBadge(status) {
  return el("span", { class: "badge badge-" + String(status || "unknown").toLowerCase() }, status || "—");
}

function bindRunPage() {
  const summary = document.getElementById("run-summary");
  if (!summary) return;
  const runId = summary.dataset.runId;
  const terminal = summary.dataset.terminal === "true";
  const statusEl = document.getElementById("s-status");
  const btnCancel = document.getElementById("btn-cancel");
  const liveBox = document.getElementById("run-live");
  const resultBox = document.getElementById("run-result");

  async function refreshStatus() {
    try {
      const data = await api("/api/runs/" + runId);
      const run = data.run;
      statusEl.textContent = "";
      statusEl.appendChild(statusBadge(run.status));
      if (run.error_summary && !document.getElementById("err-box")) {
        const box = el("div", { class: "card error-box", id: "err-box" }, run.error_summary);
        summary.insertAdjacentElement("afterend", box);
      }
    } catch (_) { /* ignore */ }
  }

  if (btnCancel) {
    btnCancel.addEventListener("click", async () => {
      btnCancel.disabled = true;
      btnCancel.textContent = "Cancelling…";
      try {
        await api("/api/runs/" + runId + "/cancel", { method: "POST" });
      } catch (err) {
        btnCancel.textContent = "Cancel";
        btnCancel.disabled = false;
        alert(err.message);
      }
    });
  }

  if (terminal) {
    if (liveBox) liveBox.hidden = true;
    if (resultBox) resultBox.hidden = false;
    renderResult(runId);
    return;
  }

  // ---- 非终态：SSE live ----
  const logEl = document.getElementById("event-log");
  const stageLine = document.getElementById("stage-line");
  const fill = document.getElementById("progress-fill");
  const es = new EventSource("/api/runs/" + runId + "/events");

  es.onopen = () => {
    if (stageLine) stageLine.innerHTML = "";
    if (stageLine) stageLine.appendChild(el("span", { class: "spinner" }, "◌"));
    if (stageLine) stageLine.appendChild(document.createTextNode("connected — waiting for events"));
  };
  es.onerror = () => {
    // EventSource 自动重连；若后端已终态则通过终态事件关闭。
    es.close();
    if (stageLine) { stageLine.textContent = "connection lost — reload to resume"; }
    window.setTimeout(() => window.location.reload(), 1200);
  };
  es.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch (_) { return; }
    const stageName = ev.stage || ev.type;
    if (stageLine && (ev.type === "stage" || ev.type === "status")) {
      stageLine.textContent = "";
      stageLine.appendChild(el("span", { class: "spinner" }, ev.type === "status" ? "●" : "◌"));
      stageLine.appendChild(document.createTextNode(ev.message || stageName));
    }
    if (fill && typeof ev.completed === "number" && typeof ev.total === "number" && ev.total > 0) {
      fill.style.width = Math.min(100, Math.round((ev.completed / ev.total) * 100)) + "%";
    }
    if (logEl) {
      const line = el("li");
      if (ev.type === "progress") {
        line.appendChild(el("span", { class: "hl" }, `[${ev.stage}]`));
        line.appendChild(document.createTextNode(` ${ev.completed}/${ev.total} · ${ev.message || ""}`.trim()));
      } else {
        const ts = ev.requests || ev.input_tokens || ev.output_tokens;
        line.appendChild(el("span", { class: "hl" }, `[${ev.type}${stageName !== ev.type ? "/" + stageName : ""}]`));
        line.appendChild(document.createTextNode(" " + (ev.message || "")));
      }
      logEl.appendChild(line);
      logEl.scrollTop = logEl.scrollHeight;
    }
    if (ev.type === "status") {
      es.close();
      window.setTimeout(() => window.location.reload(), 500);
    }
  };
}

async function renderResult(runId) {
  const body = document.getElementById("result-body");
  if (!body) return;
  try {
    const data = await api("/api/runs/" + runId + "/result");
    const run = data.run || {};
    body.textContent = "";

    const grid = el("div", { class: "result-grid" });
    grid.appendChild(metric("Capability Score", run.capability_score != null ? run.capability_score : "—", scoreClass(run.capability_score)));
    grid.appendChild(metric("Confidence", run.confidence || "—", "score-mid"));
    grid.appendChild(metric("Claimed Gap", run.claimed_gap != null ? run.claimed_gap : "—", "score-mid"));
    grid.appendChild(metric("Protocol Risk", run.risk_level || "—", "score-mid"));
    grid.appendChild(metric("Requests", run.request_count ?? "—"));
    grid.appendChild(metric("Duration", run.duration_ms ? (run.duration_ms / 1000).toFixed(1) + "s" : "—"));
    body.appendChild(grid);

    const meta = el("p", { class: "muted" });
    meta.textContent = `suite ${run.suite_id || "?"} v${run.suite_version || "?"} · created ${fmtIso(run.created_at)} · ${run.target_id}`;
    body.appendChild(meta);

    if (data.report && typeof data.report === "object") {
      body.appendChild(renderReport(data.report));
    } else {
      body.appendChild(el("p", { class: "muted" }, "（无 report 工件 —— 该 run 未产生正式报告）"));
    }
  } catch (err) {
    body.textContent = "";
    body.appendChild(el("div", { class: "error-box" }, err.message));
  }
}

function metric(label, value, cls) {
  const cell = el("div", { class: "metric" });
  cell.appendChild(el("div", { class: "num " + (cls || "") }, String(value)));
  cell.appendChild(el("div", { class: "lbl" }, label));
  return cell;
}

function scoreClass(v) {
  if (v == null) return "";
  if (v >= 70) return "score-good";
  if (v >= 40) return "score-mid";
  return "score-bad";
}

function renderReport(report) {
  const box = el("div");
  const h = el("h2", null, "Report 摘要");
  box.appendChild(h);
  const dims = report.capability_profile || report.summary || report;
  const dimKeys = [
    ["capability_score", "Capability"],
    ["risk_level", "Risk"],
    ["protocol_findings", "Protocol"],
  ];
  for (const [k, label] of dimKeys) {
    if (dims[k] != null) {
      const p = el("p");
      p.appendChild(el("strong", null, label + ": "));
      p.appendChild(document.createTextNode(typeof dims[k] === "object" ? JSON.stringify(dims[k]) : String(dims[k])));
      box.appendChild(p);
    }
  }
  return box;
}

/* ---------- boot ---------- */
document.addEventListener("DOMContentLoaded", () => {
  bindAuditForm();
  bindRunPage();
});

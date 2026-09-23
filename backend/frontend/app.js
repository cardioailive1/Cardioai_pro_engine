/* CardioAI Pro — dashboard client
 * Talks to the FastAPI backend served from the same origin:
 *   REST  -> /api/*
 *   WS    -> /ws/stream
 * If no backend is reachable (e.g. this file opened directly in a browser
 * before deployment), falls back to an in-browser simulator so the UI can
 * still be previewed — clearly labeled as local demo mode.
 */
(() => {
  const API = "/api";
  const els = {
    connDot: document.getElementById("connDot"),
    connText: document.getElementById("connText"),
    streamModeTag: document.getElementById("streamModeTag"),
    streamLog: document.getElementById("streamLog"),
    vHr: document.getElementById("v-hr"),
    v2: document.getElementById("v-2"), v2u: document.getElementById("v-2-unit"),
    v3: document.getElementById("v-3"), v3u: document.getElementById("v-3-unit"),
    qualityIssues: document.getElementById("qualityIssues"),
    fhirBlock: document.getElementById("fhirBlock"),
    dicomBlock: document.getElementById("dicomBlock"),
    sidebarMode: document.getElementById("sidebarMode"),
  };
  function setConnText(text) {
    els.connText.textContent = text;
    if (els.sidebarMode) els.sidebarMode.textContent = text;
  }
  const canvas = document.getElementById("ecgCanvas");
  const ctx = canvas ? canvas.getContext("2d") : null;
  let activeStreamTab = "ecg";
  let ecgPhase = 0;
  let currentHR = 75;
  const agentNames = ["ingestion", "quality", "fhir", "dicom", "inference", "intake", "diagnostic", "automation", "report", "alert", "continuum", "registry", "claims_aggregator", "longitudinal", "fusion"];
  const agentStats = Object.fromEntries(agentNames.map(a => [a, { runs: 0, errors: 0, flagged: 0 }]));

  // ---------------- Gauges (small SVG donuts) ----------------
  function drawGauge(container, value) {
    const svg = container.querySelector("svg");
    const pct = Math.max(0, Math.min(100, value));
    const r = 34, c = 2 * Math.PI * r;
    const color = pct >= 85 ? "#1ea672" : pct >= 70 ? "#e0a213" : "#d63b30";
    svg.innerHTML = `
      <circle cx="50" cy="50" r="${r}" fill="none" stroke="#dde8f2" stroke-width="10"/>
      <circle cx="50" cy="50" r="${r}" fill="none" stroke="${color}" stroke-width="10"
        stroke-dasharray="${c}" stroke-dashoffset="${c - (pct / 100) * c}"
        stroke-linecap="round" transform="rotate(-90 50 50)"/>
      <text x="50" y="55" text-anchor="middle" class="g-value" font-size="20">${Math.round(pct)}</text>`;
  }
  document.querySelectorAll(".gauge").forEach(g => drawGauge(g, 0));

  function updateQuality(report) {
    if (!report) return;
    const b = report.breakdown;
    document.querySelectorAll(".gauge").forEach(g => drawGauge(g, b[g.dataset.g] ?? 0));
    if (!els.qualityIssues) return;
    els.qualityIssues.textContent = report.issues && report.issues.length
      ? `Score ${report.score} — ${report.issues.join("; ")}`
      : `Score ${report.score} — no issues on latest record.`;
  }

  // ---------------- ECG waveform ----------------
  function drawECGFrame() {
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#071528";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#4fc3e8";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    const speed = currentHR / 60;
    for (let x = 0; x < w; x++) {
      const t = (x / w) * 6 + ecgPhase;
      const beat = t % (2.2 / speed);
      let y = 0;
      if (beat < 0.06) y = -6;
      else if (beat < 0.10) y = 30 * Math.sin((beat - 0.06) / 0.04 * Math.PI);
      else if (beat < 0.16) y = -10 * Math.sin((beat - 0.10) / 0.06 * Math.PI);
      else y = 3 * Math.sin(t * 6);
      const py = h / 2 - y;
      x === 0 ? ctx.moveTo(x, py) : ctx.lineTo(x, py);
    }
    ctx.stroke();
    ecgPhase += 0.045;
    requestAnimationFrame(drawECGFrame);
  }
  if (canvas) requestAnimationFrame(drawECGFrame);

  // ---------------- Log ----------------
  function logLine(text, cls = "muted") {
    if (!els.streamLog) return;
    const row = document.createElement("div");
    row.className = `row ${cls}`;
    row.textContent = text;
    els.streamLog.prepend(row);
    while (els.streamLog.children.length > 60) els.streamLog.removeChild(els.streamLog.lastChild);
  }

  // ---------------- Agent orchestrator pulses ----------------
  function pulseAgent(name, status) {
    const node = document.querySelector(`.agent-node[data-agent="${name}"]`);
    const card = document.querySelector(`.agent-card[data-agent-card="${name}"]`);
    const led = document.getElementById(`led-${name}`);
    const ledCard = document.getElementById(`ledc-${name}`);
    const ledClass = "led " + (status === "error" ? "error" : status === "flagged" ? "flagged" : "ok");

    if (node) {
      node.classList.add("pulse");
      setTimeout(() => node.classList.remove("pulse"), 400);
    }
    if (card) {
      card.classList.add("pulse");
      setTimeout(() => card.classList.remove("pulse"), 400);
    }
    if (led) led.className = ledClass;
    if (ledCard) ledCard.className = ledClass;

    if (!agentStats[name]) return;
    const s = agentStats[name];
    s.runs++;
    if (status === "error") s.errors++;
    if (status === "flagged") s.flagged++;

    const statEl = document.getElementById(`stat-${name}`);
    if (statEl) statEl.textContent = `${s.runs} runs · ${s.flagged} flagged`;

    const acStatEl = document.getElementById(`acstat-${name}`);
    if (acStatEl) acStatEl.textContent = `${s.runs} runs · ${s.flagged} flagged · ${s.errors} errors`;

    logAgentEvent(`${name} → ${status}`, status);
  }

  function logAgentEvent(text, status) {
    const el = document.getElementById("agentEventLog");
    if (!el) return;
    const cls = status === "error" ? "error" : status === "flagged" ? "flagged" : "ok";
    const row = document.createElement("div");
    row.className = `row ${cls}`;
    row.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    el.prepend(row);
    while (el.children.length > 80) el.removeChild(el.lastChild);
  }

  // ---------------- Sidebar navigation ----------------
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.page;
      document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.querySelector(`.page[data-page="${target}"]`).classList.add("active");
    });
  });

  // ---------------- Inference result panels ----------------
  function setResultCard(prefix, prediction) {
    if (!prediction) return;
    const tierEl = document.getElementById(`${prefix}-tier`);
    const barEl = document.getElementById(`${prefix}-bar`);
    const detailEl = document.getElementById(`${prefix}-detail`);
    if (!tierEl || !barEl || !detailEl) return;
    const tier = prediction.risk_tier || "low";
    tierEl.textContent = tier.toUpperCase();
    tierEl.className = `risk-pill ${tier}`;
    barEl.style.width = `${Math.round((prediction.score || 0) * 100)}%`;
    detailEl.textContent = `score ${prediction.score ?? "—"} · model ${prediction.model || "n/a"} (${prediction.model_version || "placeholder"})`;
  }

  // ---------------- Vitals + result routing per modality ----------------
  function handleIngestResult(msg) {
    const modality = msg.modality;
    const trace = msg.trace || [];
    trace.forEach(t => pulseAgent(t.agent, t.status));

    if (!msg.ok || !msg.result) {
      logLine(`[${modality}] task ${msg.task_id} failed at ${msg.failed_agent}: ${msg.error}`, "error");
      return;
    }

    const record = msg.result.normalized?.record || {};
    const quality = msg.result.quality;
    const prediction = msg.result.prediction;
    const alert = msg.result.alert;

    if (quality) updateQuality(quality);

    if (modality === activeStreamTab && els.vHr) {
      if (record.heart_rate_bpm != null) {
        els.vHr.textContent = record.heart_rate_bpm;
        currentHR = record.heart_rate_bpm;
      }
      if (modality === "ecg") {
        els.v2.textContent = record.qt_interval_ms ?? "—"; els.v2u.textContent = "ms QTc";
        els.v3.textContent = record.hrv_sdnn_ms ?? "—"; els.v3u.textContent = "ms HRV-SDNN";
      } else if (modality === "wearable") {
        els.v2.textContent = record.spo2_pct ?? "—"; els.v2u.textContent = "% SpO2";
        els.v3.textContent = record.steps ?? "—"; els.v3u.textContent = "steps";
      } else if (modality === "claims") {
        els.v2.textContent = record.member_age ?? "—"; els.v2u.textContent = "yrs age";
        els.v3.textContent = record.risk_flags_count ?? "—"; els.v3u.textContent = "risk flags";
      }
    }

    const cls = quality && !quality.passed ? "flagged" : "ok";
    logLine(`[${modality}] task ${msg.task_id} · quality ${quality ? quality.score : "n/a"} · ${prediction ? "risk " + prediction.score : "no model"}`, cls);

    if (modality === "ecg") setResultCard("ecg", prediction);
    if (modality === "claims") setResultCard("claims", prediction);
    if (modality === "wearable") setResultCard("wearable", prediction);

    if (alert) logLine(`ALERT · ${alert.message}`, "flagged");

    if (modality === "ecg" && alert) {
      const patientId = record.patient_id || `P-${msg.task_id}`;
      upsertWorklist(patientId, record, prediction, alert);
    }
  }

  // ---------------- Clinician worklist ----------------
  const worklist = new Map(); // patientId -> { record, prediction, alert, flaggedAt, status }

  function upsertWorklist(patientId, record, prediction, alert) {
    const existing = worklist.get(patientId);
    worklist.set(patientId, {
      record, prediction, alert,
      flaggedAt: existing ? existing.flaggedAt : new Date(),
      status: existing ? existing.status : "flagged",
    });
    renderWorklist();
  }

  function renderWorklist() {
    const body = document.getElementById("clinicianTableBody");
    const countEl = document.getElementById("worklistCount");
    if (!body) return;

    const entries = Array.from(worklist.entries())
      .sort((a, b) => (b[1].prediction?.score || 0) - (a[1].prediction?.score || 0));

    countEl.textContent = `${entries.length} patient${entries.length === 1 ? "" : "s"} flagged`;

    if (!entries.length) {
      body.innerHTML = `<tr class="empty-row"><td colspan="9">No high-risk patients flagged yet — watching the hospital ECG stream…</td></tr>`;
      return;
    }

    body.innerHTML = entries.map(([patientId, e]) => {
      const tier = e.prediction?.risk_tier || "moderate";
      const rowCls = `row-${tier}${e.status === "reviewed" ? " reviewed" : ""}`;
      const time = e.flaggedAt.toLocaleTimeString();
      return `
        <tr class="${rowCls}" data-patient="${patientId}">
          <td>${patientId}</td>
          <td>${(e.prediction?.score ?? 0).toFixed(3)}</td>
          <td><span class="risk-pill ${tier}">${tier.toUpperCase()}</span></td>
          <td>${e.record.heart_rate_bpm ?? "—"}</td>
          <td>${e.record.qt_interval_ms ?? "—"}</td>
          <td>${e.record.hrv_sdnn_ms ?? "—"}</td>
          <td>${e.alert?.window_days ?? "—"}d</td>
          <td>${time}</td>
          <td><button class="status-btn ${e.status === "reviewed" ? "done" : ""}" data-review="${patientId}">${e.status === "reviewed" ? "Reviewed" : "Mark reviewed"}</button></td>
        </tr>`;
    }).join("");

    body.querySelectorAll("[data-review]").forEach(btn => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.review;
        const entry = worklist.get(id);
        if (entry) {
          entry.status = entry.status === "reviewed" ? "flagged" : "reviewed";
          renderWorklist();
        }
      });
    });
  }

  document.getElementById("clearReviewedBtn")?.addEventListener("click", () => {
    for (const [id, e] of worklist.entries()) {
      if (e.status === "reviewed") worklist.delete(id);
    }
    renderWorklist();
  });

  // ---------------- Worklist CSV export ----------------
  function csvEscape(val) {
    const s = String(val ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function buildWorklistCSV() {
    const header = ["patient_id", "mace_risk_score", "risk_tier", "heart_rate_bpm", "qtc_ms", "hrv_sdnn_ms", "alert_window_days", "flagged_at", "status"];
    const rows = Array.from(worklist.entries())
      .sort((a, b) => (b[1].prediction?.score || 0) - (a[1].prediction?.score || 0))
      .map(([patientId, e]) => [
        patientId,
        e.prediction?.score ?? "",
        e.prediction?.risk_tier ?? "",
        e.record.heart_rate_bpm ?? "",
        e.record.qt_interval_ms ?? "",
        e.record.hrv_sdnn_ms ?? "",
        e.alert?.window_days ?? "",
        e.flaggedAt.toISOString(),
        e.status,
      ].map(csvEscape).join(","));
    return [header.join(","), ...rows].join("\n");
  }

  document.getElementById("downloadWorklistBtn")?.addEventListener("click", async (ev) => {
    const btn = ev.currentTarget;
    if (!worklist.size) { logLine("No flagged patients to export yet.", "muted"); return; }

    const csv = buildWorklistCSV();
    const filename = `cardioai_worklist_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
    const originalLabel = btn.textContent;

    // Prefer the platform download capability when this page is running as a
    // published Claude artifact (plain browser downloads are inert there).
    if (window.claude && typeof window.claude.use === "function") {
      try {
        const downloads = await window.claude.use("downloads");
        if (downloads) {
          await downloads.save({ filename, data: csv });
          return;
        }
      } catch (err) {
        if (err && err.code === "declined") return; // viewer said no — do nothing further
        // any other capability error: fall through to the plain browser download below
      }
    }

    // Plain browser download — this is the path used once deployed normally (e.g. on Render).
    try {
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      btn.textContent = "Export failed";
      setTimeout(() => { btn.textContent = originalLabel; }, 1500);
    }
  });

  // ---------------- Tabs ----------------
  document.querySelectorAll("[data-stream]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-stream]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeStreamTab = btn.dataset.stream;
      els.vHr.textContent = "—"; els.v2.textContent = "—"; els.v3.textContent = "—";
    });
  });

  const fhirSamples = {
    ecg: () => fetch(`${API}/fhir/sample/ecg`).then(r => r.json()),
    claims: () => fetch(`${API}/fhir/sample/claims`).then(r => r.json()),
    hl7adt: () => fetch(`${API}/hl7/sample/adt`).then(r => r.json()),
    hl7oru: () => fetch(`${API}/hl7/sample/oru`).then(r => r.json()),
  };
  document.querySelectorAll("[data-fhir]").forEach(btn => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll("[data-fhir]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const key = btn.dataset.fhir;
      try {
        const data = await fhirSamples[key]();
        els.fhirBlock.textContent = JSON.stringify(data, null, 2);
      } catch {
        els.fhirBlock.textContent = "Backend not reachable — deploy the service to see live FHIR/HL7 samples.";
      }
    });
  });
  document.querySelector('[data-fhir="ecg"]')?.click();

  document.getElementById("dicomUploadBtn")?.addEventListener("click", async () => {
    const input = document.getElementById("dicomFile");
    if (!input.files.length) { els.dicomBlock.textContent = "Choose a .dcm file first."; return; }
    const fd = new FormData();
    fd.append("file", input.files[0]);
    els.dicomBlock.textContent = "Extracting…";
    try {
      const res = await fetch(`${API}/dicom/metadata`, { method: "POST", body: fd });
      const data = await res.json();
      els.dicomBlock.textContent = JSON.stringify(data, null, 2);
    } catch {
      els.dicomBlock.textContent = "Backend not reachable — deploy the service to enable DICOM parsing.";
    }
  });

  // ---------------- Automation Tier Engine (client-side port for demo mode) ----------------
  const AUTO_URGENT_CODES = ["I21.19", "I21.09"];
  const AUTO_CONF_URGENT_MIN = 95.0;
  const AUTO_CONF_REC_MIN = 80.0;
  const AUTO_ACTION_LABEL = {
    urgent: "Auto-escalated — on-call cardiologist paged",
    recommendation: "Queued as recommendation — awaiting clinician sign-off",
    informative: "Logged for reference only — no action taken",
  };
  const AUTO_DIAGNOSIS_ACTIONS = {
    "STEMI": {
      urgent: ["Activate STEMI protocol", "Page on-call interventional cardiologist immediately", "Prepare cath lab — door-to-balloon clock started", "Notify ED attending and charge nurse"],
      recommendation: ["Recommend urgent cardiology consult", "Consider serial troponins and repeat ECG in 15-30 min", "Flag for physician review within the hour"],
      informative: ["Findings logged for physician review", "Confidence or coding did not meet the urgent-tier bar — no automated escalation"],
    },
    "Atrial fibrillation": {
      recommendation: ["Recommend rate/rhythm control evaluation", "Consider anticoagulation risk assessment (e.g. CHA2DS2-VASc)", "Flag for cardiology follow-up within 24-48h"],
      informative: ["Findings logged for physician review at next visit", "No immediate action indicated by automation — clinical correlation advised"],
    },
    "Heart failure, systolic": {
      recommendation: ["Recommend BNP/NT-proBNP and volume status assessment", "Consider diuretic titration per guideline-directed therapy", "Flag for cardiology follow-up within 1 week"],
      informative: ["Findings logged for physician review", "Trend against prior echo/labs at next visit"],
    },
  };
  const AUTO_GENERIC_ACTIONS = {
    urgent: ["Auto-escalated to on-call physician — high-confidence urgent finding", "Immediate clinical correlation required"],
    recommendation: ["Recommend clinician review within 24 hours", "Consider correlating with additional diagnostics before acting"],
    informative: ["Findings logged for physician review", "No automated action taken — confidence below recommendation threshold"],
  };

  function localAutomationEvaluate(diagnosis, icd10Codes, ecgConf, echoConf, labConf) {
    const confidences = [ecgConf ?? 0];
    if (echoConf !== null && echoConf !== undefined && echoConf !== "") confidences.push(Number(echoConf));
    if (labConf !== null && labConf !== undefined && labConf !== "") confidences.push(Number(labConf));
    const confidence = Math.round((confidences.reduce((a,b) => a+b, 0) / confidences.length) * 100) / 100;

    let tier = "informative";
    if (icd10Codes.some(c => AUTO_URGENT_CODES.includes(c)) && confidence >= AUTO_CONF_URGENT_MIN) tier = "urgent";
    else if (confidence >= AUTO_CONF_REC_MIN) tier = "recommendation";

    const table = AUTO_DIAGNOSIS_ACTIONS[diagnosis];
    const recommendations = (table && table[tier]) ? table[tier] : AUTO_GENERIC_ACTIONS[tier];

    return {
      tier, confidence, icd10_codes: icd10Codes, diagnosis, recommendations,
      requires_human_signoff: tier !== "urgent", auto_escalate: tier === "urgent",
      action_label: AUTO_ACTION_LABEL[tier],
    };
  }

  function renderAutomationScenario(s) {
    return `
      <div class="scenario-card">
        <div class="sc-head">
          <span class="sc-label">${s.label || s.diagnosis}</span>
          <span class="tier-badge ${s.tier}">${s.tier.toUpperCase()}</span>
        </div>
        <div class="sc-meta">${s.diagnosis} · ICD-10: ${s.icd10_codes.length ? s.icd10_codes.join(", ") : "none"} · confidence ${s.confidence}% · ${s.action_label}</div>
        <ul>${s.recommendations.map(r => `<li>${r}</li>`).join("")}</ul>
      </div>`;
  }

  document.getElementById("runAutomationSamplesBtn")?.addEventListener("click", async (ev) => {
    const btn = ev.currentTarget;
    btn.disabled = true;
    try {
      const res = await fetch(`${API}/automation-tier/samples`, { signal: AbortSignal.timeout ? AbortSignal.timeout(4000) : undefined });
      if (!res.ok) throw new Error("no backend");
      const data = await res.json();
      document.getElementById("automationSamplesBody").innerHTML = data.scenarios.map(renderAutomationScenario).join("");
      document.getElementById("automationSampleSource").textContent = "live backend · engine's actual policy engine";
    } catch {
      const scenarios = [
        localAutomationEvaluate("STEMI", ["I21.19"], 97, 96, null),
        { ...localAutomationEvaluate("STEMI", ["I21.19"], 68, null, null), label: "STEMI code, but low confidence — should NOT auto-escalate" },
        { ...localAutomationEvaluate("Atrial fibrillation", ["I48.91"], 92, null, null), label: "Atrial fibrillation, high confidence, non-urgent code" },
        { ...localAutomationEvaluate("Nonspecific ST-T changes", [], 55, null, null), label: "Low-confidence, unlisted diagnosis" },
      ];
      scenarios[0].label = "STEMI, high confidence (both modalities agree)";
      document.getElementById("automationSamplesBody").innerHTML = scenarios.map(renderAutomationScenario).join("");
      document.getElementById("automationSampleSource").textContent = "local demo mode · client-side policy port";
    } finally {
      btn.disabled = false;
    }
  });

  document.getElementById("evaluateAutomationBtn")?.addEventListener("click", async (ev) => {
    const btn = ev.currentTarget;
    btn.disabled = true;
    const diagnosis = document.getElementById("autoDxInput").value.trim() || "Unspecified";
    const codes = document.getElementById("autoCodesInput").value.split(",").map(s => s.trim()).filter(Boolean);
    const ecgConf = parseFloat(document.getElementById("autoEcgConfInput").value) || 0;
    const echoRaw = document.getElementById("autoEchoConfInput").value.trim();
    const labRaw = document.getElementById("autoLabConfInput").value.trim();

    try {
      const res = await fetch(`${API}/automation-tier/evaluate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          diagnosis, icd10_codes: codes,
          ecg: { confidence: ecgConf },
          echo: echoRaw ? { confidence: parseFloat(echoRaw) } : null,
          lab: labRaw ? { confidence: parseFloat(labRaw) } : null,
        }),
        signal: AbortSignal.timeout ? AbortSignal.timeout(4000) : undefined,
      });
      if (!res.ok) throw new Error("no backend");
      const data = await res.json();
      document.getElementById("automationEvalResult").innerHTML = renderAutomationScenario({ ...data, label: "Custom evaluation · live backend" });
    } catch {
      const data = localAutomationEvaluate(diagnosis, codes, ecgConf, echoRaw || null, labRaw || null);
      document.getElementById("automationEvalResult").innerHTML = renderAutomationScenario({ ...data, label: "Custom evaluation · local demo mode" });
    } finally {
      btn.disabled = false;
    }
  });

  // ---------------- Business Outputs (Monitoring, Population, Consumer, Compliance) ----------------
  function renderNeedsBackend(elId) {
    const el = document.getElementById(elId);
    if (el) el.innerHTML = `<div class="disclaimer">Needs the live engine — this preview has no backend to read from. Deploy <code>cardioai-pro.zip</code> and open this page from that server.</div>`;
  }

  async function refreshMonitoring() {
    try {
      const data = await fetch(`${API}/monitoring/status`, { signal: AbortSignal.timeout ? AbortSignal.timeout(4000) : undefined }).then(r => { if (!r.ok) throw new Error(); return r.json(); });
      const latencies = Object.entries(data.inference_latency_ms).map(([k, v]) => `${k}: ${v}ms`).join(" · ");
      document.getElementById("monitoringKpis").innerHTML = `
        <div class="kpi ${data.model_accuracy === null ? "" : "ok"}"><div class="kpi-label">Model accuracy</div><div class="kpi-value">${data.model_accuracy !== null ? (data.model_accuracy * 100).toFixed(1) + "%" : "—"}</div><div class="kpi-sub">${data.model_accuracy_source}</div></div>
        <div class="kpi ${data.bias_gap_pp !== null && data.bias_gap_pp <= 2 ? "ok" : data.bias_gap_pp !== null ? "warn" : ""}"><div class="kpi-label">Bias gap (sensitivity)</div><div class="kpi-value">${data.bias_gap_pp !== null ? data.bias_gap_pp + "pp" : "—"}</div><div class="kpi-sub">worst dimension, post-remediation</div></div>
        <div class="kpi"><div class="kpi-label">Inference latency</div><div class="kpi-value" style="font-size:0.95rem;">${Object.keys(data.inference_latency_ms).length} agents</div><div class="kpi-sub" style="font-size:0.65rem;">${latencies}</div></div>
        <div class="kpi ${data.ehr_integration_health.fhir_agent_success_rate === 1 ? "ok" : "warn"}"><div class="kpi-label">EHR integration health</div><div class="kpi-value">${data.ehr_integration_health.fhir_agent_success_rate !== null ? (data.ehr_integration_health.fhir_agent_success_rate * 100).toFixed(0) + "%" : "—"}</div><div class="kpi-sub">${data.ehr_integration_health.note}</div></div>`;
    } catch { renderNeedsBackend("monitoringKpis"); }
  }

  async function refreshPopulationReport() {
    try {
      const data = await fetch(`${API}/payer/population-report`, { signal: AbortSignal.timeout ? AbortSignal.timeout(4000) : undefined }).then(r => { if (!r.ok) throw new Error(); return r.json(); });
      document.getElementById("populationReportBody").innerHTML = `
        <div class="grid-3" style="margin-bottom:12px;">
          <div class="kpi"><div class="kpi-label">Total members</div><div class="kpi-value">${data.total_members}</div></div>
          <div class="kpi ${data.tier_breakdown.high ? "err" : "ok"}"><div class="kpi-label">High-risk</div><div class="kpi-value">${data.tier_breakdown.high}</div></div>
          <div class="kpi"><div class="kpi-label">Avg risk score</div><div class="kpi-value">${data.avg_risk_score}</div></div>
        </div>
        <p style="font-size:0.8rem;">${data.narrative.mlr_impact}</p>
        <p style="font-size:0.76rem; color:var(--ink-soft);">${data.narrative.star_rating_note}</p>
        ${data.high_risk_cohort.length ? `<div class="table-wrap"><table class="clinician-table"><thead><tr><th>Member</th><th>Age</th><th>Risk flags</th><th>Score</th></tr></thead><tbody>${data.high_risk_cohort.map(m => `<tr><td>${m.member_id}</td><td>${m.age}</td><td>${m.risk_flags_count}</td><td>${m.score}</td></tr>`).join("")}</tbody></table></div>` : ""}`;
    } catch { renderNeedsBackend("populationReportBody"); }
  }

  async function lookupConsumerScore() {
    const id = document.getElementById("consumerLookupId").value.trim();
    const el = document.getElementById("consumerScoreBody");
    if (!id) { el.innerHTML = `<div class="form-result error">Enter a subscriber ID.</div>`; return; }
    try {
      const res = await fetch(`${API}/consumer/${encodeURIComponent(id)}/score`, { signal: AbortSignal.timeout ? AbortSignal.timeout(4000) : undefined });
      if (res.status === 404) { el.innerHTML = `<div class="form-result info">No score yet for "${id}" — it needs at least one wearable reading first.</div>`; return; }
      if (!res.ok) throw new Error();
      const data = await res.json();
      el.innerHTML = `
        <div class="form-result ${data.risk_trend.includes("Elevated") ? "error" : data.risk_trend.includes("Worth") ? "info" : "success"}">
          <strong>${data.subscriber_id}</strong> — score ${data.score}, trend: <strong>${data.risk_trend}</strong><br/>
          ${data.personalized_guidance}<br/>
          ${data.share_with_cardiologist_recommended ? "📤 Recommended: share with your cardiologist." : ""}
          <div style="margin-top:6px; font-size:0.7rem; opacity:0.75;">${data.disclaimer}</div>
        </div>`;
    } catch { renderNeedsBackend("consumerScoreBody"); }
  }
  document.getElementById("consumerLookupBtn")?.addEventListener("click", lookupConsumerScore);

  async function refreshCompliance() {
    try {
      const data = await fetch(`${API}/compliance/report`, { signal: AbortSignal.timeout ? AbortSignal.timeout(4000) : undefined }).then(r => { if (!r.ok) throw new Error(); return r.json(); });
      const eq = data.fda_equity_guidance;
      document.getElementById("complianceReportBody").innerHTML = `
        <div class="staff-card"><div class="who">${data.report_id}</div><span class="pill ${eq.status === "pass" ? "approved" : eq.status === "fail" ? "denied" : "pending"}">${eq.status.replace(/_/g, " ")}</span></div>
        ${eq.dimensions ? Object.entries(eq.dimensions).map(([dim, d]) => `
          <div class="staff-card"><div><div class="who">${dim.replace("_", " / ")}</div><div class="meta">${d.baseline_sensitivity_gap_pp}pp → ${d.remediated_sensitivity_gap_pp}pp</div></div><span class="pill ${d.passes ? "approved" : "denied"}">${d.passes ? "PASS" : "FAIL"}</span></div>`).join("") : `<p style="font-size:0.8rem;">${eq.note}</p>`}
        <h4 style="font-size:0.8rem; color:var(--navy); margin:14px 0 8px;">Agent health</h4>
        ${data.agent_health.map(a => `<div class="staff-card"><div class="who">${a.agent}</div><span class="mono" style="font-size:0.74rem;">${a.runs} runs · ${(a.error_rate*100).toFixed(1)}% err · ${a.avg_latency_ms}ms</span></div>`).join("")}`;
    } catch { renderNeedsBackend("complianceReportBody"); }
  }

  document.getElementById("refreshMonitoringBtn")?.addEventListener("click", refreshMonitoring);
  document.getElementById("refreshPopulationBtn")?.addEventListener("click", refreshPopulationReport);
  document.getElementById("refreshComplianceBtn")?.addEventListener("click", refreshCompliance);
  if (document.getElementById("monitoringKpis")) {
    refreshMonitoring(); refreshPopulationReport(); refreshCompliance();
  }

  // ---------------- Diagnostic Agent (client-side port for demo mode) ----------------
  function fakeDiagnosticFinding(prediction, qualityScore) {
    if (prediction.risk_tier === "high") {
      return { diagnosis: "Abnormal ECG pattern, high MACE risk — specialist correlation required", icd10_codes: ["R94.31"], confidence: qualityScore, specialist_review_required: true };
    }
    if (prediction.risk_tier === "moderate") {
      return { diagnosis: "Borderline ECG findings", icd10_codes: ["R94.31"], confidence: Math.round(qualityScore * 0.85 * 100) / 100, specialist_review_required: true };
    }
    return { diagnosis: "No significant finding", icd10_codes: [], confidence: qualityScore, specialist_review_required: false };
  }

  function fakeAutomationFromFinding(finding) {
    // Mirrors the backend fix: no automation decision when there's nothing to act on.
    if (!finding.specialist_review_required) return null;
    const confidence = finding.confidence;
    // R94.31 is never in AUTO_URGENT_CODES, so this path can never reach "urgent" —
    // matches the real engine's structural safety guarantee.
    const tier = confidence >= AUTO_CONF_REC_MIN ? "recommendation" : "informative";
    const recommendations = (AUTO_DIAGNOSIS_ACTIONS[finding.diagnosis] && AUTO_DIAGNOSIS_ACTIONS[finding.diagnosis][tier]) || AUTO_GENERIC_ACTIONS[tier];
    return { tier, confidence, recommendations, requires_human_signoff: true, auto_escalate: false, action_label: AUTO_ACTION_LABEL[tier] };
  }

  // ---------------- Care Continuum (client-side port for demo mode) ----------------
  const CONTINUUM_STAGES = ["screening", "diagnostic_review", "care_decision", "treatment", "monitoring", "resolved"];
  const localContinuum = new Map(); // patient_id -> { stage, history: [], lastUpdated }

  function continuumEnsure(patientId) {
    if (!localContinuum.has(patientId)) localContinuum.set(patientId, { stage: "screening", history: [], lastUpdated: Date.now() / 1000 });
    return localContinuum.get(patientId);
  }
  function continuumAdvanceLocal(patientId, stage, reason) {
    const p = continuumEnsure(patientId);
    if (CONTINUUM_STAGES.indexOf(stage) <= CONTINUUM_STAGES.indexOf(p.stage)) return false;
    p.history.push({ from: p.stage, to: stage, reason, timestamp: Date.now() / 1000 });
    p.stage = stage;
    p.lastUpdated = Date.now() / 1000;
    return true;
  }
  function continuumRecordDiagnostic(patientId, diagnosis) {
    if (diagnosis && diagnosis !== "No significant finding") continuumAdvanceLocal(patientId, "diagnostic_review", `finding: ${diagnosis}`);
  }
  function continuumRecordCareDecision(patientId, tier) {
    if (tier === "urgent" || tier === "recommendation") continuumAdvanceLocal(patientId, "care_decision", `automation tier: ${tier}`);
  }
  function continuumFunnelCountsLocal() {
    const counts = Object.fromEntries(CONTINUUM_STAGES.map(s => [s, 0]));
    localContinuum.forEach(p => { counts[p.stage]++; });
    return counts;
  }

  const STAGE_DISPLAY = { screening: "Screening", diagnostic_review: "Diagnostic review", care_decision: "Care decision", treatment: "Treatment", monitoring: "Monitoring", resolved: "Resolved" };
  const ADVANCE_NEXT = { care_decision: ["treatment", "Mark treated"], treatment: ["monitoring", "Move to monitoring"], monitoring: ["resolved", "Resolve"] };

  function renderContinuumFunnel(counts) {
    document.getElementById("continuumFunnel").innerHTML = CONTINUUM_STAGES.map(s => `
      <div class="funnel-stage ${s}">
        <div class="fs-count">${counts[s] ?? 0}</div>
        <div class="fs-label">${STAGE_DISPLAY[s]}</div>
      </div>`).join("");
  }

  function renderContinuumTable(patients) {
    const tbody = patients.length ? patients.map(p => {
      const next = ADVANCE_NEXT[p.stage];
      return `
        <tr>
          <td>${p.patient_id}</td>
          <td><span class="stage-pill ${p.stage}">${STAGE_DISPLAY[p.stage]}</span></td>
          <td class="mono" style="font-family:var(--mono); font-size:0.72rem;">${p.history_len ?? (p.history ? p.history.length : 0)} transition(s)</td>
          <td>${next ? `<button class="status-btn" data-advance-continuum="${p.patient_id}" data-advance-action="${next[0]}">${next[1]}</button>` : "—"}</td>
        </tr>`;
    }).join("") : `<tr class="empty-row"><td colspan="4">No patients in the continuum yet — watching the hospital ECG stream…</td></tr>`;

    document.getElementById("continuumTable").innerHTML = `
      <thead><tr><th>Patient</th><th>Stage</th><th>History</th><th></th></tr></thead>
      <tbody>${tbody}</tbody>`;

    document.querySelectorAll("[data-advance-continuum]").forEach(btn => {
      btn.addEventListener("click", async () => {
        const patientId = btn.dataset.advanceContinuum;
        const action = btn.dataset.advanceAction;
        try {
          const res = await fetch(`${API}/care-continuum/patient/${encodeURIComponent(patientId)}/advance`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action }),
            signal: AbortSignal.timeout ? AbortSignal.timeout(3000) : undefined,
          });
          if (!res.ok) throw new Error("no backend");
        } catch {
          continuumAdvanceLocal(patientId, action === "treatment" ? "treatment" : action === "monitoring" ? "monitoring" : "resolved", "manual advance (local demo mode)");
        }
        refreshContinuum();
      });
    });
  }

  async function refreshContinuum() {
    try {
      const res = await fetch(`${API}/care-continuum/status`, { signal: AbortSignal.timeout ? AbortSignal.timeout(3000) : undefined });
      if (!res.ok) throw new Error("no backend");
      const data = await res.json();
      renderContinuumFunnel(data.funnel);
      renderContinuumTable(data.patients);
      document.getElementById("continuumSource").textContent = `live backend · ${data.total_patients} patient(s) tracked`;
    } catch {
      const counts = continuumFunnelCountsLocal();
      const patients = Array.from(localContinuum.entries())
        .map(([patient_id, p]) => ({ patient_id, stage: p.stage, history: p.history, lastUpdated: p.lastUpdated }))
        .sort((a, b) => b.lastUpdated - a.lastUpdated)
        .slice(0, 50);
      renderContinuumFunnel(counts);
      renderContinuumTable(patients);
      document.getElementById("continuumSource").textContent = `local demo mode · ${localContinuum.size} patient(s) tracked`;
    }
  }
  document.getElementById("refreshContinuumBtn")?.addEventListener("click", refreshContinuum);
  if (document.getElementById("continuumFunnel")) {
    refreshContinuum();
    setInterval(refreshContinuum, 5000);
  }

  // ---------------- Bias Audit ----------------
  const BIAS_DIMENSIONS = {
    race_ethnicity: ["White", "Black", "Hispanic", "Asian", "Other/Unspecified"],
    gender: ["Female", "Male"],
    age_band: ["<50", "50-64", "65+"],
  };
  const BIAS_REP_WEIGHT = {
    White: 1.00, Black: 0.55, Hispanic: 0.62, Asian: 0.78, "Other/Unspecified": 0.65,
    Male: 1.00, Female: 0.80,
    "50-64": 1.00, "<50": 0.88, "65+": 0.90,
  };
  const BIAS_EQUITY_TARGET_PP = 2.0;
  const BIAS_BASELINE_THRESHOLD = 0.6;

  function biasEcgScore(f) {
    let s = 0;
    s += Math.max(0, f.heart_rate_bpm - 100) * 0.01;
    s += Math.max(0, f.qt_interval_ms - 460) * 0.004;
    s += Math.max(0, 30 - f.hrv_sdnn_ms) * 0.01;
    return Math.min(s, 1.0);
  }

  function runLocalBiasAudit(nPerCell = 50) {
    let seed = 7;
    const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };

    const cohort = [];
    for (const race of BIAS_DIMENSIONS.race_ethnicity) {
      for (const gender of BIAS_DIMENSIONS.gender) {
        for (const age of BIAS_DIMENSIONS.age_band) {
          for (let i = 0; i < nPerCell; i++) {
            const features = {
              heart_rate_bpm: 60 + rnd() * 90,
              qt_interval_ms: 370 + rnd() * 170,
              hrv_sdnn_ms: 8 + rnd() * 70,
            };
            const score = biasEcgScore(features);
            let w = 1.0;
            [race, gender, age].forEach(v => { w *= (BIAS_REP_WEIGHT[v] ?? 1.0); });
            w = Math.max(0.15, w);
            const signal = w * score + (1 - w) * rnd();
            const groundTruth = rnd() < signal ? 1 : 0;
            cohort.push({ subgroups: { race_ethnicity: race, gender, age_band: age }, score, groundTruth, pred: false });
          }
        }
      }
    }

    function scoreAt(cases, t) { cases.forEach(c => { c.pred = c.score >= t; }); }
    function confusion(cases) {
      let tp=0,fp=0,tn=0,fn=0;
      cases.forEach(c => {
        if (c.pred && c.groundTruth === 1) tp++;
        else if (c.pred && c.groundTruth === 0) fp++;
        else if (!c.pred && c.groundTruth === 0) tn++;
        else fn++;
      });
      return { tp, fp, tn, fn };
    }
    function metricsFrom(conf) {
      const { tp, fp, tn, fn } = conf;
      const n = tp + fp + tn + fn;
      const sensitivity = (tp + fn) ? tp / (tp + fn) : 0;
      const specificity = (tn + fp) ? tn / (tn + fp) : 0;
      const accuracy = n ? (tp + tn) / n : 0;
      return { n, tp, fp, tn, fn, sensitivity, specificity, accuracy };
    }
    function subgroupMetrics(dimension) {
      const out = {};
      BIAS_DIMENSIONS[dimension].forEach(v => {
        const cases = cohort.filter(c => c.subgroups[dimension] === v);
        out[v] = metricsFrom(confusion(cases));
      });
      return out;
    }
    function gapPP(metrics, key) {
      const vals = Object.values(metrics).filter(m => m.n > 0).map(m => m[key]);
      if (!vals.length) return 0;
      return Math.round((Math.max(...vals) - Math.min(...vals)) * 1000) / 10;
    }
    function remediateThresholds(dimension, targetSens) {
      const grid = [];
      for (let t = 0.20; t <= 0.80; t += 0.01) grid.push(Math.round(t * 100) / 100);
      const out = {};
      BIAS_DIMENSIONS[dimension].forEach(v => {
        const cases = cohort.filter(c => c.subgroups[dimension] === v);
        let bestT = BIAS_BASELINE_THRESHOLD, bestGap = Infinity;
        grid.forEach(t => {
          scoreAt(cases, t);
          const sens = metricsFrom(confusion(cases)).sensitivity;
          const gap = Math.abs(sens - targetSens);
          if (gap < bestGap) { bestGap = gap; bestT = t; }
        });
        out[v] = bestT;
      });
      return out;
    }

    scoreAt(cohort, BIAS_BASELINE_THRESHOLD);
    const overallSens = metricsFrom(confusion(cohort)).sensitivity;

    const report = { cohort_size: cohort.length, baseline_threshold: BIAS_BASELINE_THRESHOLD, equity_target_pp: BIAS_EQUITY_TARGET_PP, dimensions: {} };
    for (const dimension of Object.keys(BIAS_DIMENSIONS)) {
      scoreAt(cohort, BIAS_BASELINE_THRESHOLD);
      const baseline = subgroupMetrics(dimension);
      const baselineSensGap = gapPP(baseline, "sensitivity");
      const baselineAccGap = gapPP(baseline, "accuracy");

      const thresholds = remediateThresholds(dimension, overallSens);
      BIAS_DIMENSIONS[dimension].forEach(v => {
        cohort.filter(c => c.subgroups[dimension] === v).forEach(c => { c.pred = c.score >= thresholds[v]; });
      });
      const remediated = subgroupMetrics(dimension);
      const remediatedSensGap = gapPP(remediated, "sensitivity");
      const remediatedAccGap = gapPP(remediated, "accuracy");

      report.dimensions[dimension] = {
        baseline: { metrics: baseline, sensitivity_gap_pp: baselineSensGap, accuracy_gap_pp: baselineAccGap },
        remediated: { thresholds, metrics: remediated, sensitivity_gap_pp: remediatedSensGap, accuracy_gap_pp: remediatedAccGap },
        passes_equity_target: remediatedSensGap <= BIAS_EQUITY_TARGET_PP,
        improvement_pp: Math.round((baselineSensGap - remediatedSensGap) * 100) / 100,
      };
    }
    report.overall_pass = Object.values(report.dimensions).every(d => d.passes_equity_target);
    return report;
  }

  function renderBiasAudit(report, source) {
    document.getElementById("biasEquityTarget").textContent = report.equity_target_pp.toFixed(1);
    const badge = document.getElementById("biasOverallBadge");
    badge.innerHTML = `<span class="pass-badge ${report.overall_pass ? "pass" : "fail"}">${report.overall_pass ? "PASSES" : "FAILS"} equity target</span> <span style="font-size:0.7rem;color:var(--ink-soft);margin-left:6px;">${source} · cohort n=${report.cohort_size}</span>`;

    const body = document.getElementById("biasAuditBody");
    body.innerHTML = Object.entries(report.dimensions).map(([dim, d]) => {
      const rows = Object.keys(d.baseline.metrics).map(sg => {
        const before = d.baseline.metrics[sg];
        const after = d.remediated.metrics[sg];
        const thr = d.remediated.thresholds[sg];
        return `
          <div class="sens-bar-row">
            <span class="sg-label">${sg}</span>
            <div class="sens-bar-track"><div class="sens-bar-fill before" style="width:${Math.round(before.sensitivity*100)}%"></div></div>
            <span class="sg-val">${(before.sensitivity*100).toFixed(1)}%</span>
          </div>
          <div class="sens-bar-row" style="margin-bottom:8px;">
            <span class="sg-label" style="color:var(--ok);">→ remediated (thr ${thr})</span>
            <div class="sens-bar-track"><div class="sens-bar-fill after" style="width:${Math.round(after.sensitivity*100)}%"></div></div>
            <span class="sg-val">${(after.sensitivity*100).toFixed(1)}%</span>
          </div>`;
      }).join("");

      return `
        <div class="bias-dim-block">
          <div class="bias-dim-head">
            <h4>${dim.replace("_"," / ")}</h4>
            <span class="pass-badge ${d.passes_equity_target ? "pass" : "fail"}">${d.passes_equity_target ? "PASS" : "FAIL"}</span>
          </div>
          <div class="bias-gap-compare">
            <span>Sensitivity gap: <span class="gv before">${d.baseline.sensitivity_gap_pp}pp</span> → <span class="gv after">${d.remediated.sensitivity_gap_pp}pp</span></span>
            <span>Improvement: <span class="gv after">${d.improvement_pp}pp</span></span>
          </div>
          ${rows}
        </div>`;
    }).join("");
  }

  document.getElementById("runBiasAuditBtn")?.addEventListener("click", async (ev) => {
    const btn = ev.currentTarget;
    btn.disabled = true;
    const original = btn.textContent;
    btn.textContent = "Running audit…";
    try {
      const res = await fetch(`${API}/bias-audit/run`, { signal: AbortSignal.timeout ? AbortSignal.timeout(4000) : undefined });
      if (!res.ok) throw new Error("backend audit failed");
      const report = await res.json();
      renderBiasAudit(report, "live backend · engine's actual inference model");
    } catch {
      const report = runLocalBiasAudit(50);
      renderBiasAudit(report, "local demo mode · client-side audit port");
    } finally {
      btn.disabled = false;
      btn.textContent = original;
    }
  });

  // ---------------- Agent status polling ----------------
  async function pollStatus() {
    try {
      const res = await fetch(`${API}/agents/status`);
      const data = await res.json();
      data.agents.forEach(a => {
        const statEl = document.getElementById(`stat-${a.name}`);
        if (statEl) statEl.textContent = `${a.runs} runs · ${a.flagged} flagged · ${a.avg_ms}ms avg`;
      });
    } catch { /* backend not up yet */ }
  }

  // ---------------- WebSocket connection ----------------
  let usingFallback = false;
  function connectWS() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    let ws;
    try {
      ws = new WebSocket(`${proto}//${location.host}/ws/stream`);
    } catch {
      startFallback();
      return;
    }
    let opened = false;
    const timeout = setTimeout(() => { if (!opened) { ws.close(); startFallback(); } }, 3500);

    ws.onopen = () => {
      opened = true;
      clearTimeout(timeout);
      els.connDot.classList.add("live");
      setConnText("Live — connected to backend");
      if (els.streamModeTag) els.streamModeTag.textContent = "live";
      logLine("WebSocket connected to /ws/stream", "ok");
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "ingest_result") handleIngestResult(msg);
      if (msg.type === "orchestrator_event" && msg.event.type === "agent_hop") {
        pulseAgent(msg.event.agent, msg.event.status);
      }
    };
    ws.onclose = () => {
      els.connDot.classList.remove("live");
      setConnText("Disconnected — retrying…");
      if (!usingFallback) setTimeout(connectWS, 2500);
    };
    ws.onerror = () => ws.close();
  }

  // ---------------- Local fallback simulator (no backend deployed yet) ----------------
  function startFallback() {
    if (usingFallback) return;
    usingFallback = true;
    els.connDot.classList.remove("live");
    setConnText("Local demo mode — backend not deployed");
    if (els.streamModeTag) els.streamModeTag.textContent = "local demo";
    logLine("No backend reachable — running local demo simulation. Deploy to Render for live data.", "muted");

    let taskCounter = 0;
    function fakeQuality() {
      const score = 78 + Math.random() * 20;
      return {
        score: Math.round(score * 10) / 10, passed: score >= 70,
        breakdown: {
          schema: 100, range: Math.round(85 + Math.random() * 15),
          completeness: 100, drift: Math.round(70 + Math.random() * 30),
        },
        issues: score < 85 ? ["statistical outlier vs. recent baseline: heart_rate_bpm"] : [],
      };
    }
    function fakePrediction(modality, record) {
      let score = 0.1;
      if (modality === "ecg") score = Math.min(1, Math.max(0, (record.heart_rate_bpm - 90) * 0.012 + Math.random() * 0.05));
      if (modality === "wearable") score = Math.min(1, Math.max(0, (record.heart_rate_bpm - 95) * 0.01 + Math.random() * 0.05));
      if (modality === "claims") score = Math.min(1, Math.max(0, (record.member_age - 50) * 0.008 + record.risk_flags_count * 0.06));
      score = Math.round(score * 1000) / 1000;
      const tier = score >= 0.6 ? "high" : score >= 0.3 ? "moderate" : "low";
      return { score, risk_tier: tier, model: modality + "_placeholder", model_version: "0.0.0-local-demo", window_days: modality === "ecg" ? 60 : 90 };
    }

    const gens = {
      ecg: () => {
        // ~12% of readings simulate a genuine high-risk patient, so the
        // clinician worklist has something real to demonstrate in demo mode.
        const spike = Math.random() < 0.12;
        return {
          patient_id: `P-${String(Math.floor(1 + Math.random() * 40)).padStart(4, "0")}`,
          heart_rate_bpm: spike ? Math.round(140 + Math.random() * 45) : Math.round(70 + Math.random() * 45),
          qt_interval_ms: spike ? Math.round(480 + Math.random() * 60) : Math.round(380 + Math.random() * 80),
          hrv_sdnn_ms: spike ? Math.round(6 + Math.random() * 18) : Math.round(25 + Math.random() * 50),
        };
      },
      wearable: () => ({ heart_rate_bpm: Math.round(65 + Math.random() * 45), spo2_pct: Math.round(93 + Math.random() * 6), steps: Math.round(Math.random() * 400) }),
      claims: () => ({ member_age: Math.round(30 + Math.random() * 55), risk_flags_count: Math.round(Math.random() * 6) }),
    };

    function tick(modality) {
      const record = gens[modality]();
      const quality = fakeQuality();
      const prediction = fakePrediction(modality, record);
      taskCounter++;

      let finding = null, automation = null;
      if (modality === "ecg") {
        finding = fakeDiagnosticFinding(prediction, quality.score);
        automation = fakeAutomationFromFinding(finding);
        if (record.patient_id) {
          continuumEnsure(record.patient_id);
          continuumRecordDiagnostic(record.patient_id, finding.diagnosis);
          if (automation) continuumRecordCareDecision(record.patient_id, automation.tier);
        }
      }

      const msg = {
        type: "ingest_result", modality, ok: true, task_id: `demo-${taskCounter}`,
        trace: ["ingestion", "quality", "fhir", "inference", "intake", "longitudinal", "fusion", "diagnostic", "automation", "report", "alert", "continuum", "registry"].map(a => ({
          agent: a,
          status: !quality.passed && a === "quality" ? "flagged"
            : (prediction.risk_tier === "high" && a === "inference") ? "flagged"
            : (finding && finding.specialist_review_required && a === "diagnostic") ? "flagged"
            : "ok",
        })),
        result: {
          normalized: { record }, quality, prediction, diagnostic_finding: finding, automation,
          alert: prediction.risk_tier === "high" ? { message: `MACE risk flagged (${prediction.score}) — route to cardiologist worklist`, window_days: prediction.window_days } : null,
        },
      };
      handleIngestResult(msg);
    }

    setInterval(() => tick("ecg"), 1500);
    setInterval(() => tick("wearable"), 2200);
    setInterval(() => tick("claims"), 3200);
  }

  connectWS();
  pollStatus();
  setInterval(pollStatus, 6000);
})();

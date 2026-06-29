const $ = (id) => document.getElementById(id);
const API = "";
let selectedModelId = null;
let datasetPath = null;
let recommendedProfile = null;
const jobTimers = {};

function toast(msg, level = "info") {
  const root = $("toast-root");
  const el = document.createElement("div");
  el.className = `toast toast-${level}`;
  el.textContent = msg;
  root.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => { el.classList.remove("show"); setTimeout(() => el.remove(), 250); }, 4000);
}
async function jget(url) { const r = await fetch(API + url); if (!r.ok) throw new Error((await r.json()).detail || r.statusText); return r.json(); }
async function jpost(url, body) {
  const r = await fetch(API + url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  if (!r.ok) { let d; try { d = (await r.json()).detail; } catch (e) {} throw new Error(d || r.statusText); }
  return r.json();
}

/* ===================== سخت‌افزار ===================== */
async function loadHardware() {
  try {
    const data = await jget("/api/hardware");
    recommendedProfile = data.recommended_profile;
    const hw = data.hardware;
    $("hardware-body").innerHTML = `
      <div class="kv">سیستم‌عامل: ${hw.os} &nbsp;|&nbsp; هسته‌های CPU: ${hw.cpu_cores_logical}</div>
      <div class="kv">RAM کل: ${hw.ram_total_gb ?? "?"} GB &nbsp;|&nbsp; RAM آزاد: ${hw.ram_available_gb ?? "?"} GB</div>
      <div class="kv">GPU: ${hw.gpu_available ? (hw.gpu_name + " (" + hw.gpu_vram_total_gb + " GB VRAM)") : "ندارد / شناسایی نشد"}</div>
      <div class="note-box" style="margin-top:8px;background:var(--accent-soft);border:1px dashed var(--accent-dim);padding:8px 10px;border-radius:6px;font-size:11.5px;">
        پروفایل پیشنهادی: <b>${recommendedProfile.label}</b><br>${recommendedProfile.warning}
      </div>`;
    $("hw-pill").textContent = recommendedProfile.label;
    $("hw-pill").dataset.status = hw.gpu_available ? "ok" : "warn";
    applyProfileDefaults(recommendedProfile);
    loadModels();
  } catch (e) {
    $("hardware-body").innerHTML = `<span style="color:var(--danger)">خطا در تشخیص سخت‌افزار: ${e.message}</span>`;
  }
}
function applyProfileDefaults(p) {
  $("p-lora_r").value = p.lora_r;
  $("p-lora_alpha").value = p.lora_alpha;
  $("p-per_device_batch_size").value = p.per_device_batch_size;
  $("p-grad_accum_steps").value = p.grad_accum_steps;
  $("p-max_seq_len").value = p.max_seq_len;
  $("p-use_4bit").checked = p.use_4bit;
  $("p-gradient_checkpointing").checked = p.gradient_checkpointing;
  $("p-precision").value = p.precision;
}

/* ===================== مدل‌ها ===================== */
async function loadModels() {
  try {
    const data = await jget("/api/models");
    $("model-list").innerHTML = data.models.map(m => `
      <div class="model-row ${m.fits_current_hardware ? "" : "unfit"}" data-id="${m.id}">
        <div style="flex:1">
          <div class="mname">${m.label} ${m.gated ? '<span class="tag">Gated</span>' : ""}</div>
          <div class="mmeta">${m.params_b}B · حداقل RAM (CPU): ${m.min_ram_gb_cpu}GB · حداقل VRAM (QLoRA): ${m.min_vram_gb_qlora}GB</div>
          <div class="mmeta">${m.notes}</div>
        </div>
      </div>`).join("");
    document.querySelectorAll(".model-row").forEach(row => row.addEventListener("click", () => {
      document.querySelectorAll(".model-row").forEach(r => r.classList.remove("selected"));
      row.classList.add("selected");
      selectedModelId = row.dataset.id;
      $("exp-base-model").value = selectedModelId;
    }));
  } catch (e) { toast("خطا در بارگذاری مدل‌ها: " + e.message, "error"); }
}

/* ===================== دیتاست ===================== */
$("dataset-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData(); fd.append("file", file);
  try {
    const r = await fetch("/api/dataset/upload", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "خطا");
    datasetPath = data.path;
    $("dataset-info").textContent = `فرمت: ${data.format_detected} | تعداد نمونه قابل‌استفاده: ${data.total_examples}`;
    toast("دیتاست با موفقیت بارگذاری شد.", "success");
  } catch (err) { toast("خطا در بارگذاری دیتاست: " + err.message, "error"); }
});

/* ===================== شروع آموزش ===================== */
$("btn-start-train").addEventListener("click", async () => {
  if (!datasetPath) return toast("ابتدا فایل دیتاست را بارگذاری کنید.", "error");
  if (!selectedModelId) return toast("یک مدل پایه انتخاب کنید.", "error");
  const payload = {
    dataset_path: datasetPath,
    base_model: selectedModelId,
    hf_token: $("hf-token").value || null,
    lora_r: parseInt($("p-lora_r").value),
    lora_alpha: parseInt($("p-lora_alpha").value),
    lora_dropout: parseFloat($("p-lora_dropout").value),
    learning_rate: parseFloat($("p-learning_rate").value),
    num_train_epochs: parseFloat($("p-num_train_epochs").value),
    per_device_batch_size: parseInt($("p-per_device_batch_size").value),
    grad_accum_steps: parseInt($("p-grad_accum_steps").value),
    max_seq_len: parseInt($("p-max_seq_len").value),
    use_4bit: $("p-use_4bit").checked,
    gradient_checkpointing: $("p-gradient_checkpointing").checked,
    precision: $("p-precision").value,
  };
  try {
    const res = await jpost("/api/train/start", payload);
    toast(`آموزش شروع شد (Job: ${res.job_id}) — نمونه‌های train: ${res.dataset_stats.n_train}`, "success");
    refreshJobs();
  } catch (e) { toast("خطا در شروع آموزش: " + e.message, "error"); }
});

/* ===================== فهرست Jobs و پایش زنده ===================== */
function drawLossChart(canvas, history) {
  if (!canvas || !history || !history.length) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.clientWidth * 2;
  const h = canvas.height = canvas.clientHeight * 2;
  ctx.clearRect(0, 0, w, h);
  const losses = history.map(p => p.loss);
  const min = Math.min(...losses), max = Math.max(...losses);
  const range = (max - min) || 1;
  ctx.beginPath();
  ctx.strokeStyle = "#4FD1C5";
  ctx.lineWidth = 3;
  history.forEach((p, i) => {
    const x = (i / Math.max(1, history.length - 1)) * w;
    const y = h - ((p.loss - min) / range) * (h - 10) - 5;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
function jobActionsHtml(job) {
  const acts = [];
  if (job.status === "running") acts.push(`<button class="btn btn-sm btn-danger act-stop" data-id="${job.job_id}">توقف</button>`);
  if (job.status === "stopped" || job.status === "crashed_or_stopped") acts.push(`<button class="btn btn-sm act-resume" data-id="${job.job_id}">ادامه از چک‌پوینت</button>`);
  return acts.join("");
}
async function refreshJobs() {
  try {
    const jobs = await jget("/api/jobs");
    if (!jobs.length) { $("jobs-list").innerHTML = '<p class="muted">هنوز هیچ آموزشی شروع نشده است.</p>'; return; }
    $("jobs-list").innerHTML = jobs.map(j => {
      const pct = j.total_steps ? Math.min(100, Math.round((j.step / j.total_steps) * 100)) : 0;
      return `
      <div class="job-card" data-job="${j.job_id}">
        <div class="job-head">
          <span class="jid">#${j.job_id}</span>
          <span class="badge ${j.status}">${j.status}</span>
          <span class="muted">${j.phase || ""} — مرحله ${j.step || 0}/${j.total_steps || "?"} — loss: ${j.last_loss ?? "—"}</span>
          <span class="job-acts">${jobActionsHtml(j)}</span>
        </div>
        <div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>
        <canvas class="loss-chart" id="chart-${j.job_id}"></canvas>
      </div>`;
    }).join("");
    jobs.forEach(j => drawLossChart($("chart-" + j.job_id), j.loss_history));
    document.querySelectorAll(".act-stop").forEach(b => b.addEventListener("click", async () => { await jpost(`/api/train/${b.dataset.id}/stop`); refreshJobs(); }));
    document.querySelectorAll(".act-resume").forEach(b => b.addEventListener("click", async () => { const r = await jpost(`/api/train/${b.dataset.id}/resume`); toast("ادامه آموزش با job جدید: " + r.job_id, "success"); refreshJobs(); }));
  } catch (e) { /* silent */ }
}

/* ===================== ادغام و خروجی‌گیری ===================== */
$("btn-merge").addEventListener("click", async () => {
  try {
    const res = await jpost("/api/export/merge", {
      base_model: $("exp-base-model").value,
      adapter_dir: $("exp-adapter-dir").value,
      output_dir: $("exp-output-dir").value,
      hf_token: $("hf-token").value || null,
    });
    $("export-result").textContent = "مدل ادغام‌شده ذخیره شد در: " + res.output_dir;
    toast("ادغام با موفقیت انجام شد.", "success");
  } catch (e) { toast("خطا در ادغام: " + e.message, "error"); }
});
$("btn-gguf-info").addEventListener("click", async () => {
  const data = await jget("/api/export/gguf-instructions");
  $("info-modal-title").textContent = "راهنمای خروجی GGUF";
  $("info-modal-body").textContent = data.instructions;
  $("info-modal").classList.add("show");
});
$("info-modal-close").addEventListener("click", () => $("info-modal").classList.remove("show"));

/* ===================== سرو مدل ===================== */
$("btn-serve-start").addEventListener("click", async () => {
  try { await jpost("/api/serve/start", { model_dir: $("serve-model-dir").value }); toast("مدل با موفقیت سوار شد.", "success"); }
  catch (e) { toast("خطا: " + e.message, "error"); }
});
$("btn-serve-stop").addEventListener("click", async () => { await jpost("/api/serve/stop", {}); toast("مدل پیاده شد.", "warn"); });
$("btn-serve-generate").addEventListener("click", async () => {
  try {
    const res = await jpost("/api/serve/generate", { prompt: $("serve-prompt").value, max_new_tokens: 256, temperature: 0.7 });
    $("serve-output").textContent = res.text;
  } catch (e) { toast("خطا در تولید پاسخ: " + e.message, "error"); }
});

/* ===================== تم ===================== */
$("theme-toggle").addEventListener("click", () => {
  const html = document.documentElement;
  const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
  $("theme-toggle").textContent = next === "dark" ? "🌙" : "☀️";
});

/* ===================== شروع ===================== */
loadHardware();
refreshJobs();
setInterval(refreshJobs, 4000);

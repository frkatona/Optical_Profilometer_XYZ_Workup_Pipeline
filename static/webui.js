const state = {
  jobs: [],
  selectedJobId: null,
  detailJobId: null,
  pollHandle: null,
  ui: {
    sidebarCollapsed: false,
    jobsCollapsed: false,
    detailSections: {
      stats: false,
      artifacts: false,
      logs: false,
    },
  },
  logView: {
    jobId: null,
    scrollTop: 0,
    stickToBottom: true,
  },
};

const renderOptionFieldMap = {
  render_source: "render_source",
  render_output_format: "output_format",
  render_engine: "engine",
  render_samples: "samples",
  render_resolution_x: "resolution_x",
  render_resolution_y: "resolution_y",
  camera_distance_scale: "camera_distance_scale",
  camera_azimuth: "camera_azimuth",
  camera_elevation: "camera_elevation",
  camera_lens: "camera_lens",
  height_scale: "height_scale",
  auto_height_ratio: "auto_height_ratio",
  rotation_x: "rotation_x",
  rotation_y: "rotation_y",
  rotation_z: "rotation_z",
  world_strength: "world_strength",
  key_light_energy: "key_light_energy",
  fill_light_energy: "fill_light_energy",
  shading: "shading",
  material_preset: "material_preset",
  material_color: "material_color",
  material_roughness: "material_roughness",
  material_metallic: "material_metallic",
  material_specular: "material_specular",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatDate(timestamp) {
  if (!timestamp) return "n/a";
  return new Date(timestamp * 1000).toLocaleString();
}

function formatPercent(value) {
  if (value == null) return "0.0%";
  return `${Number(value).toFixed(1)}%`;
}

function formatBytes(bytes) {
  if (bytes == null) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function titleCase(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function statusTone(status) {
  switch (status) {
    case "completed":
      return "good";
    case "failed":
      return "error";
    case "running":
      return "running";
    case "queued":
    default:
      return "queued";
  }
}

function classifyLogLine(line) {
  const text = String(line || "");
  const normalized = text.toLowerCase();
  if (
    normalized.includes("traceback") ||
    normalized.includes("runtimeerror") ||
    normalized.includes(" error") ||
    normalized.startsWith("error") ||
    normalized.includes("failed")
  ) {
    return { tone: "tone-error", label: "error" };
  }
  if (normalized.includes("warning") || normalized.includes("warn")) {
    return { tone: "tone-warn", label: "warn" };
  }
  if (normalized.includes("render")) {
    return { tone: "tone-render", label: "render" };
  }
  if (
    normalized.includes("analysis") ||
    normalized.includes("statistics") ||
    normalized.includes("roughness") ||
    normalized.includes("decomposing")
  ) {
    return { tone: "tone-analysis", label: "analysis" };
  }
  return { tone: "", label: "info" };
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function isNearBottom(element) {
  return element.scrollHeight - element.clientHeight - element.scrollTop < 24;
}

function captureLogScrollState() {
  const container = document.querySelector("[data-log-scroll]");
  if (!container) return;
  state.logView = {
    jobId: state.detailJobId,
    scrollTop: container.scrollTop,
    stickToBottom: isNearBottom(container),
  };
}

function bindLogScrollTracking(container) {
  if (!container) return;
  container.addEventListener(
    "scroll",
    () => {
      state.logView = {
        jobId: state.detailJobId,
        scrollTop: container.scrollTop,
        stickToBottom: isNearBottom(container),
      };
    },
    { passive: true }
  );
}

function restoreLogScrollState(container, sameJob) {
  if (!container) return;
  if (!sameJob || state.logView.stickToBottom) {
    container.scrollTop = container.scrollHeight;
  } else {
    const maxScrollTop = Math.max(container.scrollHeight - container.clientHeight, 0);
    container.scrollTop = Math.min(state.logView.scrollTop, maxScrollTop);
  }
  bindLogScrollTracking(container);
}

function scrollPanelIntoView(id) {
  const element = document.getElementById(id);
  if (element) {
    element.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function getSelectedJob() {
  return state.jobs.find((job) => job.id === state.selectedJobId) || null;
}

function jobSupportsRerender(job) {
  return Boolean(job && job.rerender_ready);
}

function setStatusMessage(message) {
  const status = document.getElementById("submit-status");
  if (status) {
    status.textContent = message || "";
  }
}

function renderHintText(job) {
  const renderToggle = document.querySelector('input[name="enable_render"]');
  if (!window.WEBUI_BOOTSTRAP?.blenderAvailable) {
    return "Blender rendering is unavailable in this runtime.";
  }
  if (!job) {
    return "Select a completed job with an OBJ export to reuse its existing mesh.";
  }
  if (job.status === "queued" || job.status === "running") {
    return "Wait for the selected job to finish before launching a render-only rerun.";
  }
  if (!job.available_obj_sources?.length) {
    return "The selected job has no reusable OBJ exports yet.";
  }
  if (!renderToggle?.checked) {
    return "Enable the Blender render section to rerun the selected job with the current settings.";
  }
  return `Ready to rerun ${job.source_name} from existing OBJ sources: ${job.available_obj_sources.join(", ")}.`;
}

function applyRenderActionState() {
  const job = getSelectedJob();
  const rerenderButton = document.getElementById("rerender-button");
  const loadButton = document.getElementById("load-render-settings");
  const renderToggle = document.querySelector('input[name="enable_render"]');
  const hint = document.getElementById("rerender-hint");
  if (!rerenderButton || !loadButton || !hint) return;

  const renderEnabled = Boolean(renderToggle?.checked);
  const canLoad = Boolean(job);
  const canRerender = jobSupportsRerender(job) && renderEnabled;

  loadButton.disabled = !canLoad;
  rerenderButton.disabled = !canRerender;
  hint.textContent = renderHintText(job);
}

function applyLayoutState() {
  const appShell = document.querySelector(".app-shell");
  const sidebar = document.getElementById("left-sidebar");
  const sidebarToggle = document.getElementById("toggle-sidebar");
  const grid = document.querySelector(".workspace-grid");
  const jobsPanel = document.getElementById("jobs-panel");
  const toggleButton = document.getElementById("toggle-jobs-panel");

  if (appShell && sidebar && sidebarToggle) {
    appShell.classList.toggle("sidebar-collapsed", state.ui.sidebarCollapsed);
    sidebar.classList.toggle("is-collapsed", state.ui.sidebarCollapsed);
    sidebarToggle.textContent = state.ui.sidebarCollapsed ? "Open" : "Fold";
    sidebarToggle.setAttribute("aria-expanded", String(!state.ui.sidebarCollapsed));
    sidebarToggle.setAttribute(
      "aria-label",
      state.ui.sidebarCollapsed ? "Expand navigation rail" : "Collapse navigation rail"
    );
  }

  if (grid && jobsPanel && toggleButton) {
    grid.classList.toggle("jobs-collapsed", state.ui.jobsCollapsed);
    jobsPanel.classList.toggle("is-collapsed", state.ui.jobsCollapsed);
    toggleButton.textContent = state.ui.jobsCollapsed ? "Open" : "Fold";
    toggleButton.setAttribute("aria-expanded", String(!state.ui.jobsCollapsed));
  }
}

function toggleSidebar() {
  state.ui.sidebarCollapsed = !state.ui.sidebarCollapsed;
  applyLayoutState();
}

function toggleJobsPanel() {
  state.ui.jobsCollapsed = !state.ui.jobsCollapsed;
  applyLayoutState();
}

function toggleDetailSection(sectionKey) {
  state.ui.detailSections[sectionKey] = !state.ui.detailSections[sectionKey];
  const selectedJob = state.jobs.find((job) => job.id === state.selectedJobId);
  renderJobDetail(selectedJob || null);
}

function renderJobs() {
  const list = document.getElementById("jobs-list");
  if (!state.jobs.length) {
    list.innerHTML = `<div class="empty-state">No jobs yet.</div>`;
    return;
  }

  list.innerHTML = state.jobs
    .map((job) => {
      const tone = statusTone(job.status);
      const resolution = job.options?.resolution_factor || 1;
      const renderMode = job.render_options?.enabled ? job.render_options.render_source : "off";
      return `
        <article class="job-card ${job.id === state.selectedJobId ? "active" : ""}" data-job-id="${escapeHtml(job.id)}">
          <div class="job-card-header">
            <div>
              <h3>${escapeHtml(job.source_name)}</h3>
              <span class="panel-kicker">${escapeHtml(titleCase(job.stage || "queued"))}</span>
            </div>
            <span class="status-badge ${tone}">${escapeHtml(titleCase(job.status))}</span>
          </div>

          <div class="progress-shell">
            <div class="progress-bar" style="width:${job.progress || 0}%"></div>
          </div>

          <div class="job-meta">
            <span>${escapeHtml(job.message || "Waiting to start")}</span>
            <span>${escapeHtml(formatPercent(job.progress))}</span>
          </div>

          <div class="job-meta-grid">
            <span><strong>Resolution</strong> ${escapeHtml(`${resolution}x`)}</span>
            <span><strong>Render</strong> ${escapeHtml(renderMode)}</span>
            <span><strong>Created</strong> ${escapeHtml(formatDate(job.created_at))}</span>
          </div>
        </article>
      `;
    })
    .join("");

  list.querySelectorAll(".job-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedJobId = card.dataset.jobId;
      renderJobs();
      applyRenderActionState();
      fetchJobDetail();
    });
  });
}

function renderStats(job) {
  const stats = job.stats_summary || {};
  const entries = Object.entries(stats);
  if (!entries.length) {
    return `<div class="empty-state compact">Stats not available yet.</div>`;
  }

  return `
    <div class="stats-grid">
      ${entries
        .map(([key, value]) => `
          <div class="stat-card">
            <span>${escapeHtml(titleCase(key))}</span>
            <strong>${escapeHtml(typeof value === "number" ? value.toFixed(4) : value)}</strong>
          </div>
        `)
        .join("")}
    </div>
  `;
}

function renderArtifacts(job) {
  if (!job.artifacts || !job.artifacts.length) {
    return `<div class="empty-state compact">Artifacts will appear here when the job produces them.</div>`;
  }

  return `
    <div class="artifact-list">
      ${job.artifacts
        .map((artifact) => `
          <div class="artifact-item">
            <div>
              <strong>${escapeHtml(artifact.label)}</strong>
              <div class="muted">${escapeHtml(artifact.name)} | ${escapeHtml(formatBytes(artifact.size_bytes))}</div>
            </div>
            <a class="button-link secondary-button" href="${artifact.download_url}" download>Download</a>
          </div>
        `)
        .join("")}
    </div>
  `;
}

function renderPreviews(job) {
  const previews = (job.artifacts || []).filter((artifact) => artifact.preview_url);
  if (!previews.length) return "";

  return `
    <section class="preview-stage">
      <div class="preview-stage-head">
        <h3>Preview Stage</h3>
        <span class="status-badge neutral">${escapeHtml(String(previews.length))} images</span>
      </div>
      <div class="preview-grid">
        ${previews
          .map((artifact) => `
            <figure>
              <button
                type="button"
                class="preview-image-button"
                data-preview-open="${escapeHtml(artifact.preview_url)}"
                data-preview-title="${escapeHtml(artifact.label)}"
                data-preview-download="${escapeHtml(artifact.download_url)}"
              >
                <img src="${artifact.preview_url}" alt="${escapeHtml(artifact.label)}">
              </button>
              <figcaption>${escapeHtml(artifact.label)}</figcaption>
              <div class="preview-actions">
                <button
                  type="button"
                  class="panel-collapse"
                  data-preview-open="${escapeHtml(artifact.preview_url)}"
                  data-preview-title="${escapeHtml(artifact.label)}"
                  data-preview-download="${escapeHtml(artifact.download_url)}"
                >
                  Expand
                </button>
                <a class="button-link secondary-button" href="${artifact.download_url}" download>Download</a>
              </div>
            </figure>
          `)
          .join("")}
      </div>
    </section>
  `;
}

function renderLogsBody(job) {
  const lines = job.logs || [];
  return `
    <div class="log-toolbar">
      <span class="log-filter">All</span>
      <span class="log-filter">Info</span>
      <span class="log-filter">Error</span>
      <div class="log-search">Event stream updates live during analysis and render steps.</div>
    </div>
    <div class="log-scroll" data-log-scroll>
      ${
        lines.length
          ? lines
              .map((line, index) => {
                const meta = classifyLogLine(line);
                return `
                  <div class="log-row ${meta.tone}">
                    <div class="log-index">${escapeHtml(String(index + 1).padStart(3, "0"))}</div>
                    <div class="log-level">${escapeHtml(meta.label)}</div>
                    <div class="log-message">${escapeHtml(line)}</div>
                  </div>
                `;
              })
              .join("")
          : `<div class="empty-state compact">No logs yet.</div>`
      }
    </div>
  `;
}

function renderCollapsibleSection(sectionKey, title, badge, body, extraClass = "") {
  const collapsed = Boolean(state.ui.detailSections[sectionKey]);
  return `
    <section class="detail-section ${extraClass} ${collapsed ? "is-collapsed" : ""}" data-detail-section="${escapeHtml(sectionKey)}">
      <div class="detail-section-head">
        <h3>${escapeHtml(title)}</h3>
        <div class="detail-section-tools">
          ${badge ? `<span class="status-badge neutral">${escapeHtml(badge)}</span>` : ""}
          <button
            type="button"
            class="panel-collapse"
            data-section-toggle="${escapeHtml(sectionKey)}"
            aria-expanded="${String(!collapsed)}"
          >
            ${collapsed ? "Expand" : "Collapse"}
          </button>
        </div>
      </div>
      <div class="detail-section-body">
        ${body}
      </div>
    </section>
  `;
}

function bindDetailInteractions(detail) {
  detail.querySelectorAll("[data-section-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      toggleDetailSection(button.dataset.sectionToggle);
    });
  });

  detail.querySelectorAll("[data-preview-open]").forEach((element) => {
    element.addEventListener("click", () => {
      openImageModal({
        src: element.dataset.previewOpen,
        title: element.dataset.previewTitle,
        downloadUrl: element.dataset.previewDownload,
      });
    });
  });
}

function renderJobDetail(job) {
  const detail = document.getElementById("job-detail");
  const previousJobId = state.detailJobId;
  const sameJob = Boolean(job && previousJobId && previousJobId === job.id);
  captureLogScrollState();

  if (!job) {
    detail.innerHTML = `<div class="empty-state">Select a job to see progress, stats, previews, and downloads.</div>`;
    state.detailJobId = null;
    return;
  }

  const tone = statusTone(job.status);
  detail.innerHTML = `
    <div class="detail-shell">
      ${job.error ? `<div class="error-box">${escapeHtml(job.error)}</div>` : ""}

      <section class="detail-summary">
        <div class="job-card-header">
          <div>
            <h3>${escapeHtml(job.source_name)}</h3>
            <div class="detail-summary-meta">
              <span>Stage: ${escapeHtml(titleCase(job.stage || "queued"))}</span>
              <span>Created ${escapeHtml(formatDate(job.created_at))}</span>
              <span>Started ${escapeHtml(formatDate(job.started_at))}</span>
              <span>Finished ${escapeHtml(formatDate(job.finished_at))}</span>
            </div>
          </div>
          <span class="status-badge ${tone}">${escapeHtml(titleCase(job.status))}</span>
        </div>

        <div class="progress-shell">
          <div class="progress-bar" style="width:${job.progress || 0}%"></div>
        </div>

        <div class="job-meta">
          <span>${escapeHtml(job.message || "")}</span>
          <span>${escapeHtml(formatPercent(job.progress))}</span>
        </div>
      </section>

      <div class="detail-grid">
        ${renderCollapsibleSection("stats", "Stats Summary", "Surface metrics", renderStats(job))}
        ${renderCollapsibleSection("artifacts", "Artifacts", `${job.artifacts?.length || 0} files`, renderArtifacts(job))}
      </div>

      ${renderPreviews(job)}

      ${renderCollapsibleSection(
        "logs",
        "Event Stream Updates",
        `${job.logs?.length || 0} lines`,
        renderLogsBody(job),
        "log-panel"
      )}
    </div>
  `;

  state.detailJobId = job.id;
  bindDetailInteractions(detail);
  restoreLogScrollState(detail.querySelector("[data-log-scroll]"), sameJob);
  applyRenderActionState();
}

function openImageModal({ src, title, downloadUrl }) {
  const modal = document.getElementById("image-modal");
  const image = document.getElementById("image-modal-image");
  const titleNode = document.getElementById("image-modal-title");
  const download = document.getElementById("image-modal-download");
  if (!modal || !image || !titleNode || !download) return;

  image.src = src;
  image.alt = title || "Expanded preview";
  titleNode.textContent = title || "Preview";
  download.href = downloadUrl || src;
  download.setAttribute("download", "");

  modal.classList.remove("is-hidden");
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeImageModal() {
  const modal = document.getElementById("image-modal");
  const image = document.getElementById("image-modal-image");
  if (!modal || !image) return;

  modal.classList.add("is-hidden");
  modal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
  image.src = "";
  image.alt = "";
}

function bindImageModal() {
  const modal = document.getElementById("image-modal");
  if (!modal) return;

  modal.querySelectorAll("[data-modal-close], #image-modal-close").forEach((element) => {
    element.addEventListener("click", closeImageModal);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.classList.contains("is-hidden")) {
      closeImageModal();
    }
  });
}

function syncRenderControls() {
  const fieldset = document.getElementById("render-fieldset");
  if (!fieldset) return;

  const renderToggle = fieldset.querySelector('input[name="enable_render"]');
  const allControls = fieldset.querySelectorAll("select, input");
  const available = !fieldset.dataset.disabled;
  const enabled = available && Boolean(renderToggle?.checked);
  const presetControl = fieldset.querySelector('select[name="material_preset"]');
  const autoHeightToggle = fieldset.querySelector('input[name="auto_height_scale"]');

  fieldset.classList.toggle("is-disabled", !enabled);

  allControls.forEach((control) => {
    if (control === renderToggle) return;
    control.disabled = !enabled;
  });

  if (!enabled) {
    return;
  }

  const manualMaterialControls = [
    'input[name="material_color"]',
    'input[name="material_roughness"]',
    'input[name="material_metallic"]',
    'input[name="material_specular"]',
  ];
  const usingCustomMaterial = !presetControl || presetControl.value === "custom";
  manualMaterialControls.forEach((selector) => {
    const control = fieldset.querySelector(selector);
    if (control) {
      control.disabled = !usingCustomMaterial;
    }
  });

  const autoHeightRatio = fieldset.querySelector('input[name="auto_height_ratio"]');
  if (autoHeightRatio && autoHeightToggle) {
    autoHeightRatio.disabled = !autoHeightToggle.checked;
  }

  applyRenderActionState();
}

function applyRenderOptions(job) {
  if (!job?.render_options) return;
  const form = document.getElementById("job-form");
  if (!form) return;

  const renderToggle = form.querySelector('input[name="enable_render"]');
  if (renderToggle) {
    renderToggle.checked = true;
  }

  Object.entries(renderOptionFieldMap).forEach(([fieldName, optionKey]) => {
    const control = form.querySelector(`[name="${fieldName}"]`);
    const value = job.render_options?.[optionKey];
    if (!control || value == null) return;
    control.value = String(value);
  });

  const autoHeightToggle = form.querySelector('input[name="auto_height_scale"]');
  if (autoHeightToggle) {
    autoHeightToggle.checked = Boolean(job.render_options.auto_height_scale);
  }

  const transparentBackground = form.querySelector('input[name="transparent_background"]');
  if (transparentBackground) {
    transparentBackground.checked = Boolean(job.render_options.transparent_background);
  }

  const renderSource = form.querySelector('select[name="render_source"]');
  if (renderSource && job.available_obj_sources?.length && !job.available_obj_sources.includes(renderSource.value)) {
    renderSource.value = job.available_obj_sources[0];
  }

  syncRenderControls();
  setStatusMessage(`Loaded render settings from job ${job.id}`);
}

async function rerenderSelectedJob() {
  const job = getSelectedJob();
  if (!job) {
    setStatusMessage("Select a job before rerendering.");
    return;
  }
  if (!jobSupportsRerender(job)) {
    setStatusMessage("The selected job is not ready for render-only reruns yet.");
    return;
  }

  const form = document.getElementById("job-form");
  const rerenderButton = document.getElementById("rerender-button");
  const renderToggle = form?.querySelector('input[name="enable_render"]');
  if (!form || !renderToggle?.checked) {
    setStatusMessage("Enable the Blender render section, adjust the settings, then rerun.");
    return;
  }

  setStatusMessage(`Queueing render-only rerun for job ${job.id}...`);
  rerenderButton.disabled = true;

  try {
    const data = new FormData(form);
    data.set("enable_render", "on");
    const response = await api(`/api/jobs/${job.id}/rerender`, {
      method: "POST",
      body: data,
    });
    state.selectedJobId = response.job.id;
    setStatusMessage(`Queued render-only rerun for job ${response.job.id}`);
    await poll();
  } catch (error) {
    setStatusMessage(error.message);
  } finally {
    applyRenderActionState();
  }
}

async function fetchJobs() {
  const data = await api("/api/jobs");
  state.jobs = data.jobs || [];
  if (!state.selectedJobId && state.jobs.length) {
    state.selectedJobId = state.jobs[0].id;
  }
  if (
    state.selectedJobId &&
    state.jobs.length &&
    !state.jobs.some((job) => job.id === state.selectedJobId)
  ) {
    state.selectedJobId = state.jobs[0].id;
  }
  renderJobs();
  applyRenderActionState();
}

async function fetchJobDetail() {
  if (!state.selectedJobId) {
    renderJobDetail(null);
    return;
  }
  const data = await api(`/api/jobs/${state.selectedJobId}`);
  state.jobs = state.jobs.map((job) => (job.id === data.job.id ? data.job : job));
  renderJobs();
  renderJobDetail(data.job);
}

async function poll() {
  try {
    await fetchJobs();
    await fetchJobDetail();
  } catch (error) {
    console.error(error);
  }
}

async function submitJob(event) {
  event.preventDefault();
  const form = document.getElementById("job-form");
  const status = document.getElementById("submit-status");
  const button = document.getElementById("submit-button");
  status.textContent = "Submitting job...";
  button.disabled = true;

  try {
    const data = new FormData(form);
    const response = await api("/api/jobs", {
      method: "POST",
      body: data,
    });
    state.selectedJobId = response.job.id;
    form.reset();
    syncRenderControls();
    status.textContent = `Queued job ${response.job.id}`;
    await poll();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function bootstrap() {
  document.getElementById("job-form").addEventListener("submit", submitJob);
  document.getElementById("refresh-jobs").addEventListener("click", poll);
  document.getElementById("focus-form").addEventListener("click", () => scrollPanelIntoView("form-panel"));
  document.getElementById("focus-jobs").addEventListener("click", () => scrollPanelIntoView("jobs-panel"));
  document.getElementById("focus-detail").addEventListener("click", () => scrollPanelIntoView("detail-panel"));
  document.getElementById("toggle-sidebar").addEventListener("click", toggleSidebar);
  document.getElementById("toggle-jobs-panel").addEventListener("click", toggleJobsPanel);
  document.getElementById("rerender-button").addEventListener("click", rerenderSelectedJob);
  document.getElementById("load-render-settings").addEventListener("click", () => {
    const job = getSelectedJob();
    if (!job) {
      setStatusMessage("Select a job before loading saved render settings.");
      return;
    }
    applyRenderOptions(job);
  });

  const renderToggle = document.querySelector('input[name="enable_render"]');
  const presetControl = document.querySelector('select[name="material_preset"]');
  const autoHeightToggle = document.querySelector('input[name="auto_height_scale"]');

  if (renderToggle) {
    renderToggle.addEventListener("change", syncRenderControls);
  }
  if (presetControl) {
    presetControl.addEventListener("change", syncRenderControls);
  }
  if (autoHeightToggle) {
    autoHeightToggle.addEventListener("change", syncRenderControls);
  }

  bindImageModal();
  syncRenderControls();
  applyLayoutState();
  applyRenderActionState();

  poll();
  state.pollHandle = setInterval(poll, 1500);
}

document.addEventListener("DOMContentLoaded", bootstrap);

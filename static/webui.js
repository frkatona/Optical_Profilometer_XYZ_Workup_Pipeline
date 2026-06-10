const state = {
  jobs: [],
  selectedJobId: null,
  detailJobId: null,
  pollHandle: null,
  ui: {
    sidebarCollapsed: false,
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
  filePreviewToken: 0,
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

function setFilePreviewState({ mode = "empty", title, summary, imageUrl, placeholder }) {
  const card = document.getElementById("file-preview-card");
  const frame = document.getElementById("file-preview-frame");
  const image = document.getElementById("file-preview-image");
  const placeholderNode = document.getElementById("file-preview-placeholder");
  const titleNode = document.getElementById("file-preview-title");
  const summaryNode = document.getElementById("file-preview-summary");
  const dropzone = document.getElementById("file-dropzone");
  const dropzoneTitle = document.getElementById("file-dropzone-title");
  const dropzoneSummary = document.getElementById("file-dropzone-summary");

  if (!card || !frame || !image || !placeholderNode || !titleNode || !summaryNode) return;

  card.classList.toggle("is-error", mode === "error");
  card.classList.toggle("is-ready", mode === "ready");
  card.classList.toggle("is-visible", mode !== "empty");
  frame.classList.toggle("is-loading", mode === "loading");
  frame.classList.toggle("is-empty", !imageUrl);
  if (dropzone) {
    dropzone.classList.toggle("has-file", mode === "ready" || mode === "loading");
    dropzone.classList.toggle("is-error", mode === "error");
  }

  titleNode.textContent = title || "";
  summaryNode.textContent = summary || "";
  placeholderNode.textContent = placeholder || (mode === "loading" ? "Rendering preview" : "");
  placeholderNode.hidden = Boolean(imageUrl) || mode === "loading";

  if (dropzoneTitle && dropzoneSummary) {
    if (mode === "ready") {
      dropzoneTitle.textContent = title || "Data loaded";
      dropzoneSummary.textContent = "Ready to start a job, or drop different .xyz data to replace it.";
    } else if (mode === "loading") {
      dropzoneTitle.textContent = title || "Data selected";
      dropzoneSummary.textContent = "Rendering a quick preview...";
    } else if (mode === "error") {
      dropzoneTitle.textContent = title || "File could not be loaded";
      dropzoneSummary.textContent = summary || "Drop valid .xyz data or click to choose another source.";
    } else {
      dropzoneTitle.textContent = "Drop data here";
      dropzoneSummary.textContent = "or click to choose .xyz data from your computer";
    }
  }

  if (imageUrl) {
    image.src = imageUrl;
    image.hidden = false;
  } else {
    image.removeAttribute("src");
    image.hidden = true;
  }
}

function resetFilePreview() {
  state.filePreviewToken += 1;
  setFilePreviewState({
    mode: "empty",
    title: "",
    summary: "",
    placeholder: "",
  });
}

function describeFilePreview(preview) {
  const sourceSize = `${preview.width} x ${preview.height}`;
  const previewSize = `${preview.preview_width} x ${preview.preview_height}`;
  const warning = preview.warnings?.length ? ` Warning: ${preview.warnings[0]}` : "";
  return `${sourceSize} source, ${previewSize} preview, ${formatPercent(preview.coverage_percent)} coverage, ${preview.resolution_factor}x downsample.${warning}`;
}

async function previewSelectedFile() {
  const input = document.querySelector('input[name="input_file"]');
  const file = input?.files?.[0];
  const token = state.filePreviewToken + 1;
  state.filePreviewToken = token;

  if (!file) {
    resetFilePreview();
    return;
  }

  setFilePreviewState({
    mode: "loading",
    title: file.name,
    summary: `Rendering a tiny preview from ${formatBytes(file.size)}...`,
    placeholder: "Rendering preview",
  });

  try {
    const body = new FormData();
    body.append("input_file", file);
    const response = await api("/api/preview", {
      method: "POST",
      body,
    });
    if (token !== state.filePreviewToken) return;

    const preview = response.preview;
    setFilePreviewState({
      mode: "ready",
      title: preview.filename || file.name,
      summary: describeFilePreview(preview),
      imageUrl: preview.image_url,
    });
  } catch (error) {
    if (token !== state.filePreviewToken) return;
    setFilePreviewState({
      mode: "error",
      title: file.name,
      summary: error.message,
      placeholder: "Preview unavailable",
    });
  }
}

function assignDroppedFile(input, file) {
  if (!input || !file) return false;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  return true;
}

function bindFileDropzone() {
  const dropzone = document.getElementById("file-dropzone");
  const input = document.querySelector('input[name="input_file"]');
  if (!dropzone || !input) return;

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.add("is-dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.remove("is-dragging");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const files = Array.from(event.dataTransfer?.files || []);
    const xyzFile = files.find((file) => file.name.toLowerCase().endsWith(".xyz"));
    if (!xyzFile) {
      setFilePreviewState({
        mode: "error",
        title: "No compatible data found",
        summary: "Drop data ending in .xyz.",
        placeholder: "Drop .xyz data",
      });
      return;
    }
    if (assignDroppedFile(input, xyzFile)) {
      previewSelectedFile();
    }
  });
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

  if (appShell && sidebar && sidebarToggle) {
    appShell.classList.toggle("sidebar-collapsed", state.ui.sidebarCollapsed);
    sidebar.classList.toggle("is-collapsed", state.ui.sidebarCollapsed);
    sidebarToggle.textContent = state.ui.sidebarCollapsed ? "\u2192" : "\u2190";
    sidebarToggle.setAttribute("aria-expanded", String(!state.ui.sidebarCollapsed));
    sidebarToggle.setAttribute(
      "aria-label",
      state.ui.sidebarCollapsed ? "Expand navigation rail" : "Collapse navigation rail"
    );
  }
}

function toggleSidebar() {
  state.ui.sidebarCollapsed = !state.ui.sidebarCollapsed;
  applyLayoutState();
}

function toggleDetailSection(sectionKey) {
  state.ui.detailSections[sectionKey] = !state.ui.detailSections[sectionKey];
  const selectedJob = state.jobs.find((job) => job.id === state.selectedJobId);
  renderJobDetail(selectedJob || null);
}

function renderJobs() {
  const list = document.getElementById("jobs-list");
  if (!list) return;

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
      renderFixedProgress();
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

function primaryAnalysisPreview(job) {
  const previews = (job.artifacts || []).filter((artifact) => artifact.preview_url);
  if (job.status !== "completed" || !previews.length) return null;
  return (
    previews.find((artifact) => /analysis/i.test(artifact.label) || /analysis/i.test(artifact.name)) ||
    previews[0]
  );
}

function renderPrimaryAnalysisImage(job) {
  const artifact = primaryAnalysisPreview(job);
  if (!artifact) return "";

  return `
    <section class="primary-analysis-preview">
      <button
        type="button"
        class="preview-image-button"
        data-preview-open="${escapeHtml(artifact.preview_url)}"
        data-preview-title="${escapeHtml(artifact.label)}"
        data-preview-download="${escapeHtml(artifact.download_url)}"
      >
        <img src="${artifact.preview_url}" alt="${escapeHtml(artifact.label)}">
      </button>
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
    </section>
  `;
}

function renderPreviews(job) {
  const primary = primaryAnalysisPreview(job);
  const previews = (job.artifacts || []).filter(
    (artifact) => artifact.preview_url && artifact !== primary
  );
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

function renderFixedProgress() {
  const progress = document.getElementById("fixed-progress");
  if (!progress) return;

  const job = getSelectedJob();
  if (!job) {
    progress.className = "fixed-progress is-empty";
    progress.innerHTML = "";
    return;
  }

  const tone = statusTone(job.status);
  progress.className = "fixed-progress";
  progress.innerHTML = `
    <div class="fixed-progress-head">
      <span>${escapeHtml(job.message || titleCase(job.stage || job.status || "queued"))}</span>
      <span class="status-badge ${tone}">${escapeHtml(titleCase(job.status))}</span>
    </div>
    <div class="progress-shell">
      <div class="progress-bar" style="width:${job.progress || 0}%"></div>
    </div>
    <div class="job-meta">
      <span>${escapeHtml(job.source_name || "")}</span>
      <span>${escapeHtml(formatPercent(job.progress))}</span>
    </div>
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
    renderFixedProgress();
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

        <div class="job-meta">
          <span>${escapeHtml(job.message || "")}</span>
          <span>${escapeHtml(formatPercent(job.progress))}</span>
        </div>
      </section>

      ${renderPrimaryAnalysisImage(job)}

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
  refreshOpenImageModal(job);
  restoreLogScrollState(detail.querySelector("[data-log-scroll]"), sameJob);
  applyRenderActionState();
  renderFixedProgress();
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

function refreshOpenImageModal(job) {
  const modal = document.getElementById("image-modal");
  const titleNode = document.getElementById("image-modal-title");
  if (!modal || modal.classList.contains("is-hidden") || !titleNode || !job) {
    return;
  }

  const activeTitle = titleNode.textContent;
  const artifact = (job.artifacts || []).find((item) => item.preview_url && item.label === activeTitle);
  if (!artifact) {
    return;
  }

  openImageModal({
    src: artifact.preview_url,
    title: artifact.label,
    downloadUrl: artifact.download_url,
  });
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

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function radians(value) {
  return (Number(value || 0) * Math.PI) / 180;
}

function readRenderNumber(form, name, fallback) {
  const control = form?.querySelector(`[name="${name}"]`);
  const value = Number.parseFloat(control?.value ?? "");
  return Number.isFinite(value) ? value : fallback;
}

function vectorAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vectorSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vectorScale(vector, scale) {
  return [vector[0] * scale, vector[1] * scale, vector[2] * scale];
}

function vectorDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vectorCross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function vectorLength(vector) {
  return Math.sqrt(vectorDot(vector, vector));
}

function vectorNormalize(vector) {
  const length = vectorLength(vector);
  if (length < 1e-9) {
    return [0, 0, 0];
  }
  return vectorScale(vector, 1 / length);
}

function rotatePointXYZ(point, rotation) {
  const [rx, ry, rz] = rotation;
  let [x, y, z] = point;

  const cosX = Math.cos(rx);
  const sinX = Math.sin(rx);
  let nextY = y * cosX - z * sinX;
  let nextZ = y * sinX + z * cosX;
  y = nextY;
  z = nextZ;

  const cosY = Math.cos(ry);
  const sinY = Math.sin(ry);
  let nextX = x * cosY + z * sinY;
  nextZ = -x * sinY + z * cosY;
  x = nextX;
  z = nextZ;

  const cosZ = Math.cos(rz);
  const sinZ = Math.sin(rz);
  nextX = x * cosZ - y * sinZ;
  nextY = x * sinZ + y * cosZ;
  x = nextX;
  y = nextY;

  return [x, y, z];
}

function fitProjectedPoints(points, width, height, padding = 18) {
  if (!points.length) {
    return [];
  }

  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const rangeX = Math.max(maxX - minX, 1e-6);
  const rangeY = Math.max(maxY - minY, 1e-6);
  const scale = Math.min((width - padding * 2) / rangeX, (height - padding * 2) / rangeY);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  return points.map((point) => [
    width / 2 + (point[0] - centerX) * scale,
    height / 2 - (point[1] - centerY) * scale,
  ]);
}

function formatPointList(points) {
  return points.map((point) => `${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(" ");
}

function lineMarkup(points, edges, color, width, dash = "") {
  return edges
    .map(([start, end]) => {
      const attributes = dash ? ` stroke-dasharray="${dash}"` : "";
      return `<line x1="${points[start][0].toFixed(1)}" y1="${points[start][1].toFixed(1)}" x2="${points[end][0].toFixed(1)}" y2="${points[end][1].toFixed(1)}" stroke="${color}" stroke-width="${width}" stroke-linecap="round"${attributes}></line>`;
    })
    .join("");
}

function renderPreviewChip(label, value) {
  return `<span class="render-preview-chip"><strong>${escapeHtml(label)}</strong>${escapeHtml(value)}</span>`;
}

function getSvgViewport(svg, fallbackWidth, fallbackHeight) {
  const viewBox = svg?.getAttribute("viewBox")?.split(/\s+/).map(Number);
  if (viewBox && viewBox.length === 4 && viewBox.every(Number.isFinite)) {
    return { width: viewBox[2], height: viewBox[3] };
  }
  return { width: fallbackWidth, height: fallbackHeight };
}

function buildRenderPreviewState() {
  const form = document.getElementById("job-form");
  const fieldset = document.getElementById("render-fieldset");
  if (!form || !fieldset) {
    return null;
  }

  const renderToggle = form.querySelector('input[name="enable_render"]');
  const autoHeightToggle = form.querySelector('input[name="auto_height_scale"]');

  const heightScale = Math.max(readRenderNumber(form, "height_scale", 150), 0.01);
  const autoHeightRatio = Math.max(readRenderNumber(form, "auto_height_ratio", 0.12), 0.01);
  const autoHeightScale = Boolean(autoHeightToggle?.checked);

  const baseSpan = 2.2;
  const baseHeight = 0.22;
  const slabHeight = clamp(
    autoHeightScale ? baseSpan * autoHeightRatio * heightScale : baseHeight * heightScale,
    0.08,
    1.6
  );

  return {
    enabled: !fieldset.dataset.disabled && Boolean(renderToggle?.checked),
    distanceScale: clamp(readRenderNumber(form, "camera_distance_scale", 2.1), 0.9, 5.5),
    azimuth: readRenderNumber(form, "camera_azimuth", 35),
    elevation: clamp(readRenderNumber(form, "camera_elevation", 55), 5, 89),
    lens: clamp(readRenderNumber(form, "camera_lens", 75), 12, 200),
    rotationX: readRenderNumber(form, "rotation_x", -10),
    rotationY: readRenderNumber(form, "rotation_y", -80),
    rotationZ: readRenderNumber(form, "rotation_z", -90),
    heightScale,
    autoHeightScale,
    autoHeightRatio,
    slabWidth: baseSpan,
    slabDepth: baseSpan,
    slabHeight,
  };
}

function getRenderFieldControl(fieldName) {
  return document.querySelector(`#job-form [name="${fieldName}"]`);
}

function formatPreviewProxyValue(fieldName, value, checked = false) {
  if (checked && fieldName === "auto_height_scale") {
    return value ? "On" : "Off";
  }

  const number = Number(value);
  switch (fieldName) {
    case "camera_distance_scale":
    case "height_scale":
      return `${number.toFixed(2)}x`;
    case "camera_lens":
      return `${number.toFixed(0)} mm`;
    case "camera_azimuth":
    case "camera_elevation":
    case "rotation_x":
    case "rotation_y":
    case "rotation_z":
      return `${number.toFixed(0)} deg`;
    case "auto_height_ratio":
      return `${number.toFixed(2)} span`;
    default:
      return String(value ?? "");
  }
}

function syncPreviewModalControlsFromForm() {
  const modal = document.getElementById("render-preview-modal");
  if (!modal) return;

  const autoHeightSource = getRenderFieldControl("auto_height_scale");
  modal.querySelectorAll("[data-preview-proxy]").forEach((control) => {
    const fieldName = control.dataset.previewProxy;
    const source = getRenderFieldControl(fieldName);
    if (!source) return;

    if (control.type === "checkbox") {
      control.checked = source.checked;
    } else {
      control.value = source.value;
    }

    if (fieldName === "auto_height_ratio") {
      control.disabled = source.disabled || !autoHeightSource?.checked;
    } else {
      control.disabled = source.disabled;
    }

    const output = modal.querySelector(`[data-preview-output-for="${fieldName}"]`);
    if (output) {
      const proxyValue = control.type === "checkbox" ? control.checked : control.value;
      output.textContent = formatPreviewProxyValue(fieldName, proxyValue, control.type === "checkbox");
    }
  });
}

function buildRenderPreviewGeometry(previewState) {
  const halfWidth = previewState.slabWidth / 2;
  const halfDepth = previewState.slabDepth / 2;
  const halfHeight = previewState.slabHeight / 2;
  const rotation = [
    radians(previewState.rotationX),
    radians(previewState.rotationY),
    radians(previewState.rotationZ),
  ];

  const baseCorners = [
    [-halfWidth, -halfDepth, -halfHeight],
    [halfWidth, -halfDepth, -halfHeight],
    [-halfWidth, halfDepth, -halfHeight],
    [halfWidth, halfDepth, -halfHeight],
    [-halfWidth, -halfDepth, halfHeight],
    [halfWidth, -halfDepth, halfHeight],
    [-halfWidth, halfDepth, halfHeight],
    [halfWidth, halfDepth, halfHeight],
  ];
  const corners = baseCorners.map((point) => rotatePointXYZ(point, rotation));

  const span = Math.max(previewState.slabWidth, previewState.slabDepth, previewState.slabHeight, 1);
  const radius = span * previewState.distanceScale;
  const azimuth = radians(previewState.azimuth);
  const elevation = radians(previewState.elevation);
  const camera = [
    radius * Math.cos(elevation) * Math.cos(azimuth),
    radius * Math.cos(elevation) * Math.sin(azimuth),
    radius * Math.sin(elevation),
  ];

  const axisLength = Math.max(halfWidth, halfDepth, halfHeight) * 1.18;
  const axisVectors = {
    x: rotatePointXYZ([axisLength, 0, 0], rotation),
    y: rotatePointXYZ([0, axisLength, 0], rotation),
    z: rotatePointXYZ([0, 0, Math.max(axisLength, previewState.slabHeight * 0.95)], rotation),
  };

  return { corners, camera, axisVectors };
}

function projectPerspective(points, camera, width, height) {
  const target = [0, 0, 0];
  const forward = vectorNormalize(vectorSub(target, camera));
  let right = vectorNormalize(vectorCross(forward, [0, 0, 1]));
  if (vectorLength(right) < 1e-6) {
    right = vectorNormalize(vectorCross(forward, [0, 1, 0]));
  }
  const up = vectorNormalize(vectorCross(right, forward));

  const projected = points.map((point) => {
    const relative = vectorSub(point, camera);
    const depth = Math.max(vectorDot(relative, forward), 0.12);
    return [vectorDot(relative, right) / depth, vectorDot(relative, up) / depth];
  });
  return fitProjectedPoints(projected, width, height, 24);
}

function drawMainRenderPreview(svg, previewState, geometry) {
  if (!svg) return;

  const { width, height } = getSvgViewport(svg, 420, 260);
  const corners = projectPerspective(geometry.corners, geometry.camera, width, height);
  const axisPoints = projectPerspective(
    [
      [0, 0, 0],
      geometry.axisVectors.x,
      geometry.axisVectors.y,
      geometry.axisVectors.z,
    ],
    geometry.camera,
    width,
    height
  );
  const edges = [
    [0, 1], [0, 2], [1, 3], [2, 3],
    [4, 5], [4, 6], [5, 7], [6, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const topFace = [4, 5, 7, 6].map((index) => corners[index]);
  const center = axisPoints[0];

  svg.innerHTML = `
    <defs>
      <linearGradient id="previewTopFace" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#dcecff"></stop>
        <stop offset="100%" stop-color="#b9d9ff"></stop>
      </linearGradient>
      <marker id="previewArrowBlue" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#2588f4"></path>
      </marker>
      <marker id="previewArrowRed" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#f45b5b"></path>
      </marker>
      <marker id="previewArrowGreen" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#2d9b4d"></path>
      </marker>
    </defs>
    <rect x="0" y="0" width="${width}" height="${height}" rx="14" fill="transparent"></rect>
    <polygon points="${formatPointList(topFace)}" fill="url(#previewTopFace)" opacity="0.88"></polygon>
    ${lineMarkup(corners, edges, "#536a82", 1.8)}
    <line x1="${center[0].toFixed(1)}" y1="${center[1].toFixed(1)}" x2="${axisPoints[1][0].toFixed(1)}" y2="${axisPoints[1][1].toFixed(1)}" stroke="#f45b5b" stroke-width="2.2" marker-end="url(#previewArrowRed)"></line>
    <line x1="${center[0].toFixed(1)}" y1="${center[1].toFixed(1)}" x2="${axisPoints[2][0].toFixed(1)}" y2="${axisPoints[2][1].toFixed(1)}" stroke="#2d9b4d" stroke-width="2.2" marker-end="url(#previewArrowGreen)"></line>
    <line x1="${center[0].toFixed(1)}" y1="${center[1].toFixed(1)}" x2="${axisPoints[3][0].toFixed(1)}" y2="${axisPoints[3][1].toFixed(1)}" stroke="#2588f4" stroke-width="2.4" marker-end="url(#previewArrowBlue)"></line>
    <text x="${(axisPoints[1][0] + 6).toFixed(1)}" y="${(axisPoints[1][1] - 4).toFixed(1)}" fill="#c74545" font-size="12" font-weight="700">X</text>
    <text x="${(axisPoints[2][0] + 6).toFixed(1)}" y="${(axisPoints[2][1] - 4).toFixed(1)}" fill="#2d7a42" font-size="12" font-weight="700">Y</text>
    <text x="${(axisPoints[3][0] + 6).toFixed(1)}" y="${(axisPoints[3][1] - 4).toFixed(1)}" fill="#1f6fc0" font-size="12" font-weight="700">Z</text>
    <text x="18" y="26" fill="#607286" font-size="12">Approximate camera view</text>
    <text x="18" y="44" fill="#607286" font-size="12">Top surface is shaded blue. X/Y/Z arrows follow the current rotations.</text>
  `;
}

function drawPlanPreview(svg, previewState, geometry) {
  if (!svg) return;

  const { width, height } = getSvgViewport(svg, 180, 160);
  const points = geometry.corners.map((point) => [point[0], point[1]]);
  const cameraPoint = [geometry.camera[0], geometry.camera[1]];
  const projected = fitProjectedPoints([...points, cameraPoint, [0, 0]], width, height, 16);
  const corners = projected.slice(0, points.length);
  const camera = projected[points.length];
  const center = projected[points.length + 1];
  const edges = [
    [0, 1], [0, 2], [1, 3], [2, 3],
    [4, 5], [4, 6], [5, 7], [6, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="12" fill="transparent"></rect>
    ${lineMarkup(corners, edges, "#5f7488", 1.4)}
    <line x1="${camera[0].toFixed(1)}" y1="${camera[1].toFixed(1)}" x2="${center[0].toFixed(1)}" y2="${center[1].toFixed(1)}" stroke="#2588f4" stroke-width="1.8" stroke-dasharray="5 4"></line>
    <circle cx="${camera[0].toFixed(1)}" cy="${camera[1].toFixed(1)}" r="6" fill="#2588f4"></circle>
    <circle cx="${center[0].toFixed(1)}" cy="${center[1].toFixed(1)}" r="3.5" fill="#132f39"></circle>
    <text x="${(camera[0] + 9).toFixed(1)}" y="${(camera[1] - 7).toFixed(1)}" fill="#1f6fc0" font-size="11" font-weight="700">Cam</text>
    <text x="12" y="20" fill="#607286" font-size="11">Top-down XY</text>
  `;
}

function drawSidePreview(svg, previewState, geometry) {
  if (!svg) return;

  const { width, height } = getSvgViewport(svg, 180, 160);
  const lateralDirection = vectorNormalize([geometry.camera[0], geometry.camera[1], 0]);
  const axis = vectorLength(lateralDirection) < 1e-6 ? [1, 0, 0] : lateralDirection;
  const points = geometry.corners.map((point) => [vectorDot(point, axis), point[2]]);
  const cameraPoint = [Math.sqrt(geometry.camera[0] ** 2 + geometry.camera[1] ** 2), geometry.camera[2]];
  const projected = fitProjectedPoints([...points, cameraPoint, [0, 0]], width, height, 16);
  const corners = projected.slice(0, points.length);
  const camera = projected[points.length];
  const center = projected[points.length + 1];
  const edges = [
    [0, 1], [0, 2], [1, 3], [2, 3],
    [4, 5], [4, 6], [5, 7], [6, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="12" fill="transparent"></rect>
    ${lineMarkup(corners, edges, "#5f7488", 1.4)}
    <line x1="${camera[0].toFixed(1)}" y1="${camera[1].toFixed(1)}" x2="${center[0].toFixed(1)}" y2="${center[1].toFixed(1)}" stroke="#2588f4" stroke-width="1.8" stroke-dasharray="5 4"></line>
    <circle cx="${camera[0].toFixed(1)}" cy="${camera[1].toFixed(1)}" r="6" fill="#2588f4"></circle>
    <circle cx="${center[0].toFixed(1)}" cy="${center[1].toFixed(1)}" r="3.5" fill="#132f39"></circle>
    <text x="${(camera[0] + 9).toFixed(1)}" y="${(camera[1] - 7).toFixed(1)}" fill="#1f6fc0" font-size="11" font-weight="700">Cam</text>
    <text x="12" y="20" fill="#607286" font-size="11">Side elevation</text>
  `;
}

function openRenderPreviewModal() {
  const modal = document.getElementById("render-preview-modal");
  if (!modal) return;
  syncPreviewModalControlsFromForm();
  updateRenderPreview();
  modal.classList.remove("is-hidden");
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeRenderPreviewModal() {
  const modal = document.getElementById("render-preview-modal");
  if (!modal) return;
  modal.classList.add("is-hidden");
  modal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
}

function bindRenderPreviewModal() {
  const modal = document.getElementById("render-preview-modal");
  const trigger = document.getElementById("open-render-preview-modal");
  if (!modal || !trigger) return;

  trigger.addEventListener("click", openRenderPreviewModal);
  modal.querySelectorAll("[data-render-preview-close], #render-preview-modal-close").forEach((element) => {
    element.addEventListener("click", closeRenderPreviewModal);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.classList.contains("is-hidden")) {
      closeRenderPreviewModal();
    }
  });
}

function bindRenderPreviewModalControls() {
  const modal = document.getElementById("render-preview-modal");
  if (!modal) return;

  modal.querySelectorAll("[data-preview-proxy]").forEach((control) => {
    const forwardToForm = () => {
      const fieldName = control.dataset.previewProxy;
      const source = getRenderFieldControl(fieldName);
      if (!source) return;

      if (control.type === "checkbox") {
        source.checked = control.checked;
      } else {
        source.value = control.value;
      }

      source.dispatchEvent(new Event("input", { bubbles: true }));
      source.dispatchEvent(new Event("change", { bubbles: true }));
      syncPreviewModalControlsFromForm();
    };

    control.addEventListener("input", forwardToForm);
    control.addEventListener("change", forwardToForm);
  });
}

function legacyInlineRenderPreview() {
  const previewState = buildRenderPreviewState();
  const mainSvg = document.getElementById("render-preview-main");
  const planSvg = document.getElementById("render-preview-plan");
  const sideSvg = document.getElementById("render-preview-side");
  const caption = document.getElementById("render-preview-caption");
  const note = document.getElementById("render-preview-note");
  const readout = document.getElementById("render-preview-readout");
  if (!previewState || !mainSvg || !planSvg || !sideSvg || !caption || !note || !readout) {
    return;
  }

  const geometry = buildRenderPreviewGeometry(previewState);
  drawMainRenderPreview(mainSvg, previewState, geometry);
  drawPlanPreview(planSvg, previewState, geometry);
  drawSidePreview(sideSvg, previewState, geometry);

  const autoText = previewState.autoHeightScale
    ? `auto relief to ${previewState.autoHeightRatio.toFixed(2)} lateral ratio`
    : `manual relief ${previewState.heightScale.toFixed(2)}x`;
  caption.textContent = `${previewState.enabled ? "Render-ready preview" : "Preview of current defaults"} · az ${previewState.azimuth.toFixed(0)}° · el ${previewState.elevation.toFixed(0)}° · ${autoText}`;
  note.textContent = previewState.autoHeightScale
    ? "Auto-fit uses a representative slab thickness here; the real OBJ may scale slightly differently."
    : "Top and side diagrams show the current camera position around the object.";
  readout.innerHTML = [
    renderPreviewChip("Distance", `${previewState.distanceScale.toFixed(2)}x`),
    renderPreviewChip("Lens", `${previewState.lens.toFixed(0)} mm`),
    renderPreviewChip("Relief", `${previewState.slabHeight.toFixed(2)} span`),
    renderPreviewChip("Rotate X", `${previewState.rotationX.toFixed(0)}°`),
    renderPreviewChip("Rotate Y", `${previewState.rotationY.toFixed(0)}°`),
    renderPreviewChip("Rotate Z", `${previewState.rotationZ.toFixed(0)}°`),
  ].join("");
}

function updateRenderPreview() {
  const previewState = buildRenderPreviewState();
  const caption = document.getElementById("render-preview-caption");
  const note = document.getElementById("render-preview-note");
  const readout = document.getElementById("render-preview-readout");
  const modalCaption = document.getElementById("render-preview-modal-caption");
  const modalNote = document.getElementById("render-preview-modal-note");
  const modalReadout = document.getElementById("render-preview-modal-readout");
  if (!previewState || !caption || !note || !readout) {
    return;
  }

  const geometry = buildRenderPreviewGeometry(previewState);
  [
    document.getElementById("render-preview-main"),
    document.getElementById("render-preview-modal-main"),
  ].forEach((svg) => drawMainRenderPreview(svg, previewState, geometry));
  [
    document.getElementById("render-preview-plan"),
    document.getElementById("render-preview-modal-plan"),
  ].forEach((svg) => drawPlanPreview(svg, previewState, geometry));
  [
    document.getElementById("render-preview-side"),
    document.getElementById("render-preview-modal-side"),
  ].forEach((svg) => drawSidePreview(svg, previewState, geometry));

  const autoText = previewState.autoHeightScale
    ? `auto relief to ${previewState.autoHeightRatio.toFixed(2)} lateral ratio`
    : `manual relief ${previewState.heightScale.toFixed(2)}x`;
  const captionText = `${previewState.enabled ? "Render-ready preview" : "Preview of current defaults"} | az ${previewState.azimuth.toFixed(0)} deg | el ${previewState.elevation.toFixed(0)} deg | ${autoText}`;
  const noteText = previewState.autoHeightScale
    ? "Auto-fit uses a representative slab thickness here; the real OBJ may scale slightly differently."
    : "Top and side diagrams show the current camera position around the object.";
  const readoutHtml = [
    renderPreviewChip("Distance", `${previewState.distanceScale.toFixed(2)}x`),
    renderPreviewChip("Lens", `${previewState.lens.toFixed(0)} mm`),
    renderPreviewChip("Relief", `${previewState.slabHeight.toFixed(2)} span`),
    renderPreviewChip("Rotate X", `${previewState.rotationX.toFixed(0)} deg`),
    renderPreviewChip("Rotate Y", `${previewState.rotationY.toFixed(0)} deg`),
    renderPreviewChip("Rotate Z", `${previewState.rotationZ.toFixed(0)} deg`),
  ].join("");

  caption.textContent = captionText;
  note.textContent = noteText;
  readout.innerHTML = readoutHtml;
  if (modalCaption) modalCaption.textContent = captionText;
  if (modalNote) modalNote.textContent = noteText;
  if (modalReadout) modalReadout.innerHTML = readoutHtml;
  syncPreviewModalControlsFromForm();
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

  fieldset.classList.toggle("is-disabled", !available);
  fieldset.classList.toggle("is-folded", !enabled);

  allControls.forEach((control) => {
    if (control === renderToggle) return;
    control.disabled = !enabled;
  });

  if (!enabled) {
    updateRenderPreview();
    applyRenderActionState();
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

  updateRenderPreview();
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
  renderFixedProgress();
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
    resetFilePreview();
    syncRenderControls();
    status.textContent = `Queued job ${response.job.id}`;
    renderFixedProgress();
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
  document.getElementById("toggle-sidebar").addEventListener("click", toggleSidebar);
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
  const fileInput = document.querySelector('input[name="input_file"]');
  const previewControls = document.querySelectorAll("#render-fieldset input, #render-fieldset select");

  if (fileInput) {
    fileInput.addEventListener("change", previewSelectedFile);
  }
  if (renderToggle) {
    renderToggle.addEventListener("change", syncRenderControls);
  }
  if (presetControl) {
    presetControl.addEventListener("change", syncRenderControls);
  }
  if (autoHeightToggle) {
    autoHeightToggle.addEventListener("change", syncRenderControls);
  }
  previewControls.forEach((control) => {
    control.addEventListener("input", updateRenderPreview);
    control.addEventListener("change", updateRenderPreview);
  });

  bindImageModal();
  bindRenderPreviewModal();
  bindRenderPreviewModalControls();
  bindFileDropzone();
  resetFilePreview();
  syncRenderControls();
  applyLayoutState();
  applyRenderActionState();
  updateRenderPreview();

  poll();
  state.pollHandle = setInterval(poll, 1500);
}

document.addEventListener("DOMContentLoaded", bootstrap);

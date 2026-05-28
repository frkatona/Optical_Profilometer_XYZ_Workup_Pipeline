const state = {
  jobs: [],
  selectedJobId: null,
  pollHandle: null,
};

function formatDate(timestamp) {
  if (!timestamp) return "n/a";
  return new Date(timestamp * 1000).toLocaleString();
}

function formatPercent(value) {
  if (value == null) return "0%";
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

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function renderJobs() {
  const list = document.getElementById("jobs-list");
  if (!state.jobs.length) {
    list.innerHTML = `<div class="empty-state">No jobs yet.</div>`;
    return;
  }

  list.innerHTML = state.jobs.map((job) => `
    <article class="job-card ${job.id === state.selectedJobId ? "active" : ""}" data-job-id="${job.id}">
      <h3>${job.source_name}</h3>
      <div class="job-meta">
        <span>${job.status}</span>
        <span>${formatPercent(job.progress)}</span>
        <span>${job.stage || "queued"}</span>
      </div>
      <div class="progress-shell">
        <div class="progress-bar" style="width:${job.progress || 0}%"></div>
      </div>
      <div class="job-meta">
        <span>Created ${formatDate(job.created_at)}</span>
      </div>
    </article>
  `).join("");

  list.querySelectorAll(".job-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedJobId = card.dataset.jobId;
      renderJobs();
      fetchJobDetail();
    });
  });
}

function renderStats(job) {
  const stats = job.stats_summary || {};
  const entries = Object.entries(stats);
  if (!entries.length) return `<div class="empty-state">Stats not available yet.</div>`;
  return `
    <table class="stats-table">
      <tbody>
        ${entries.map(([key, value]) => `
          <tr>
            <td>${key}</td>
            <td>${typeof value === "number" ? value.toFixed(4) : value}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderArtifacts(job) {
  if (!job.artifacts || !job.artifacts.length) {
    return `<div class="empty-state">Artifacts will appear here when the job produces them.</div>`;
  }
  return `
    <div class="artifact-list">
      ${job.artifacts.map((artifact) => `
        <div class="artifact-item">
          <div>
            <strong>${artifact.label}</strong>
            <div class="muted">${artifact.name} · ${formatBytes(artifact.size_bytes)}</div>
          </div>
          <a href="${artifact.download_url}" download>Download</a>
        </div>
      `).join("")}
    </div>
  `;
}

function renderPreviews(job) {
  const previews = (job.artifacts || []).filter((artifact) => artifact.preview_url);
  if (!previews.length) return "";
  return `
    <section>
      <h3>Previews</h3>
      <div class="preview-grid">
        ${previews.map((artifact) => `
          <figure>
            <img src="${artifact.preview_url}" alt="${artifact.label}">
            <figcaption>${artifact.label}</figcaption>
          </figure>
        `).join("")}
      </div>
    </section>
  `;
}

function renderJobDetail(job) {
  const detail = document.getElementById("job-detail");
  if (!job) {
    detail.innerHTML = `<div class="empty-state">Select a job to see progress, stats, previews, and downloads.</div>`;
    return;
  }

  detail.innerHTML = `
    ${job.error ? `<div class="error-box">${job.error}</div>` : ""}
    <div class="status-line">
      <strong>${job.source_name}</strong>
      <span>Status: ${job.status}</span>
      <span>Stage: ${job.stage || "queued"}</span>
      <span>${job.message || ""}</span>
    </div>
    <div class="progress-shell">
      <div class="progress-bar" style="width:${job.progress || 0}%"></div>
    </div>
    <div class="muted">Created ${formatDate(job.created_at)} · Started ${formatDate(job.started_at)} · Finished ${formatDate(job.finished_at)}</div>

    <div class="detail-grid">
      <section>
        <h3>Stats Summary</h3>
        ${renderStats(job)}
      </section>
      <section>
        <h3>Artifacts</h3>
        ${renderArtifacts(job)}
      </section>
    </div>

    ${renderPreviews(job)}

    <section>
      <h3>Logs</h3>
      <pre class="log-box">${(job.logs || []).join("\n") || "No logs yet."}</pre>
    </section>
  `;
}

async function fetchJobs() {
  const data = await api("/api/jobs");
  state.jobs = data.jobs || [];
  if (!state.selectedJobId && state.jobs.length) {
    state.selectedJobId = state.jobs[0].id;
  }
  renderJobs();
}

async function fetchJobDetail() {
  if (!state.selectedJobId) {
    renderJobDetail(null);
    return;
  }
  const data = await api(`/api/jobs/${state.selectedJobId}`);
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
  poll();
  state.pollHandle = setInterval(poll, 1500);
}

document.addEventListener("DOMContentLoaded", bootstrap);

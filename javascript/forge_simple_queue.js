(() => {
  if (window.forgeSimpleQueueExternalInstalled) return;
  window.forgeSimpleQueueExternalInstalled = true;

  const api = async (path, body) => {
    const options = body === undefined ? {} : {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)
    };
    const res = await fetch(path, options);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  };

  const escapeHtml = (text) => String(text ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[ch]);

  const truncate = (text, fallback) => {
    text = String(text || "").replace(/\s+/g, " ").trim();
    return text || fallback || "(empty prompt)";
  };

  const jobLabel = (count) => `${count} job${Number(count) === 1 ? "" : "s"}`;

  const formatDuration = (seconds) => {
    if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds)) || Number(seconds) < 0) return "";
    const total = Number(seconds);
    const minutes = Math.floor(total / 60);
    const remainingSeconds = total - (minutes * 60);
    const secondsLabel = `${remainingSeconds.toFixed(1)} sec`;
    return minutes ? `${minutes} min ${secondsLabel}` : secondsLabel;
  };

  const ensureModal = () => {
    let modal = document.getElementById("forge-simple-queue-modal");
    if (modal) return modal;

    modal = document.createElement("div");
    modal.id = "forge-simple-queue-modal";
    modal.className = "fsq-backdrop";
    modal.innerHTML = `
      <div class="fsq-panel" role="dialog" aria-modal="true">
        <div class="fsq-head">
          <div class="fsq-head-main">
            <div class="fsq-title">Queue</div>
            <div class="fsq-tabs">
              <button class="fsq-tab fsq-active" data-tab-view="queue">Queue</button>
              <button class="fsq-tab" data-tab-view="history">History</button>
            </div>
            <label class="fsq-repeat-toggle" title="Repeat selected jobs after they finish">
              <input id="forge-simple-queue-repeat-toggle" type="checkbox">
              <span class="fsq-switch"></span>
              <span>Auto repeat</span>
              <span id="forge-simple-queue-repeat-count"></span>
            </label>
            <label class="fsq-repeat-toggle" title="Keep queued jobs after a restart. An interrupted job returns to the top and recovery stays paused.">
              <input id="forge-simple-queue-recovery-toggle" type="checkbox">
              <span class="fsq-switch"></span>
              <span>Recover</span>
            </label>
          </div>
          <div class="fsq-head-actions">
            <button class="fsq-select-all" data-select-all-jobs="true" title="Select all visible jobs" aria-label="Select all visible jobs">
              <svg viewBox="0 0 24 24" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"></rect><path d="m8 12 2.5 2.5L16 9"></path></g></svg>
            </button>
            <button class="fsq-bulk-delete" data-bulk-delete-selected="true" hidden disabled title="Delete selected queued jobs">Delete</button>
            <div class="fsq-saved-controls" role="group" aria-label="Saved queue">
              <button class="fsq-saved-control" data-saved-queue-action="save" title="Save queue" aria-label="Save queue">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h12l2 2v14H5z"></path><path d="M8 4v6h8V4M8 20v-6h8v6"></path></svg>
              </button>
              <button class="fsq-saved-control" data-saved-queue-action="restore" title="Restore queue" aria-label="Restore queue">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7v5h5"></path><path d="M5.5 12a7 7 0 1 0 2-5"></path><path d="M12 8v5l3 2"></path></svg>
              </button>
              <button class="fsq-saved-control" data-saved-queue-action="clear" title="Clear saved queue" aria-label="Clear saved queue">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"></path><path d="M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"></path></svg>
              </button>
            </div>
            <div class="fsq-queue-controls" role="group" aria-label="Queue controls">
              <button class="fsq-control" data-queue-control="play" title="Play queue" aria-label="Play queue">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"></path></svg>
              </button>
              <button class="fsq-control" data-queue-control="pause" title="Pause after current job" aria-label="Pause after current job">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z"></path></svg>
              </button>
              <button class="fsq-control" data-queue-control="stop" title="Stop current job and pause queue" aria-label="Stop current job and pause queue">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7h10v10H7z"></path></svg>
              </button>
            </div>
            <button class="fsq-close" title="Close" aria-label="Close queue modal">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"></path>
              </svg>
            </button>
          </div>
        </div>
        <div class="fsq-body" id="forge-simple-queue-body"></div>
      </div>`;
    document.body.appendChild(modal);
    modal.addEventListener("click", (event) => {
      if (event.target === modal || event.target.closest?.(".fsq-close")) {
        closeSettingsEditor?.();
        modal.classList.remove("fsq-open");
      }
    });
    return modal;
  };

  const openEditors = new Set();
  const openDetails = new Set();

  const activeModalTab = () => document.querySelector(".fsq-tab.fsq-active")?.dataset.tabView || "queue";
  const fullEditLaunches = new Set();
  let fullEditReturnModalTab = null;
  let fullEditReturnJobId = null;
  let fullEditJobName = "";

  const setGradioTextValue = (container, value) => {
    const field = container?.querySelector("textarea, input");
    if (!field) return false;
    const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(field), "value")?.set;
    if (setter) setter.call(field, value);
    else field.value = value;
    field.dispatchEvent(new Event("input", {bubbles: true}));
    field.dispatchEvent(new Event("change", {bubbles: true}));
    return true;
  };

  const switchToGenerationTab = (tab) => {
    if (tab === "img2img" && typeof switch_to_img2img === "function") {
      switch_to_img2img();
    } else if (tab === "txt2img" && typeof switch_to_txt2img === "function") {
      switch_to_txt2img();
    }
  };

  const startFullEdit = async (id, tab, name) => {
    if (fullEditLaunches.has(id)) return;
    const app = typeof gradioApp === "function" ? gradioApp() : document;
    const field = app.getElementById(`${tab}_simple_queue_full_edit_job_id`);
    const loader = app.getElementById(`${tab}_simple_queue_full_edit_load`);
    if (!setGradioTextValue(field, id) || !loader) {
      throw new Error("Full Edit controls are not available yet.");
    }
    const modal = document.getElementById("forge-simple-queue-modal");
    if (modal?.classList.contains("fsq-open")) {
      fullEditReturnModalTab = activeModalTab();
      fullEditReturnJobId = id;
      fullEditJobName = name;
      modal.classList.remove("fsq-open");
    }
    fullEditLaunches.add(id);
    loader.click();
    for (let attempt = 0; attempt < 25; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 160));
      let state = await api("/forge-simple-queue/status-lite");
      let job = [state.active, ...(state.pending || [])].find((item) => item?.id === id);
      if (!job && Number(state.pending_count || 0) > 0) {
        state = await api("/forge-simple-queue/status");
        job = [state.active, ...(state.pending || [])].find((item) => item?.id === id);
      }
      if (job?.editing) {
        setTimeout(() => switchToGenerationTab(tab), 120);
        setTimeout(() => fullEditLaunches.delete(id), 500);
        return;
      }
    }
    fullEditLaunches.delete(id);
    if (fullEditReturnModalTab) {
      await restoreQueueModalAfterFullEdit();
    }
    throw new Error("The queued job could not enter Full Edit.");
  };

  const startReuse = async (id, tab) => {
    const result = await api("/forge-simple-queue/reuse", {id});
    if (!result.ok) throw new Error(result.message || "The source job settings are unavailable.");
    const app = typeof gradioApp === "function" ? gradioApp() : document;
    const field = app.getElementById(`${tab}_simple_queue_reuse_job_id`);
    const loader = app.getElementById(`${tab}_simple_queue_reuse_load`);
    if (!setGradioTextValue(field, id) || !loader) {
      throw new Error("Reuse controls are not available yet.");
    }
    loader.click();
    setTimeout(() => switchToGenerationTab(tab), 180);
    document.getElementById("forge-simple-queue-modal")?.classList.remove("fsq-open");
  };

  const captureJobPositions = () => new Map(
    [...document.querySelectorAll(".fsq-job[draggable='true']")].map((row) => [row.dataset.id, row.getBoundingClientRect()])
  );

  const animateReorder = (before, movedId) => {
    for (const row of document.querySelectorAll(".fsq-job[draggable='true']")) {
      const prior = before.get(row.dataset.id);
      if (!prior) continue;
      const delta = prior.top - row.getBoundingClientRect().top;
      if (Math.abs(delta) > 1) {
        row.style.transition = "none";
        row.style.transform = `translateY(${delta}px)`;
        requestAnimationFrame(() => {
          row.style.transition = "transform 220ms ease";
          row.style.transform = "";
          row.addEventListener("transitionend", () => { row.style.transition = ""; }, {once: true});
        });
      }
      if (row.dataset.id === movedId) {
        row.classList.add("fsq-reordered");
        setTimeout(() => row.classList.remove("fsq-reordered"), 900);
      }
    }
  };

  const queueJobsFromData = (data) => [
    ...(data?.active ? [data.active] : []),
    ...(data?.pending || [])
  ];

  const selectedDeletableJobs = (data) => queueJobsFromData(data).filter((job) => (
    job?.repeat_selected
    && job.status !== "running"
    && job.progress_active !== true
  ));

  const jobActionButton = (action, id, title, icon, tab = "") => `
    <button class="fsq-icon-action fsq-${action}" data-action="${action}" data-id="${id}" ${tab ? `data-tab="${tab}"` : ""} title="${title}" aria-label="${title}">${icon}</button>`;
  const icon = {
    pause: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5h3v14H8zM13 5h3v14h-3z" fill="currentColor"></path></svg>`,
    play: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor"></path></svg>`,
    edit: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14.7 6.3 3 3M4 20l3.6-.8L18.4 8.4a2.1 2.1 0 0 0-3-3L4.8 16.2 4 20Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>`,
    fullEdit: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 3h10l4 4v5M14 3v5h5M4 3v18h8M13.5 15.5l2.5 2.5m-5.5.5 1.8-.4 5.4-5.4a1.5 1.5 0 0 0-2.1-2.1l-5.4 5.4-.4 1.8Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path></svg>`,
    details: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h9l3 3v15H6zM15 3v4h4M9 11h6M9 15h6M9 19h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>`,
    delete: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>`,
    stop: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7h10v10H7z" fill="currentColor"></path></svg>`,
    skip: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14l9-7zM17 5h2v14h-2z" fill="currentColor"></path></svg>`
  };

  const showQueueToast = (message, ok) => {
    const toast = document.createElement("div");
    toast.className = ok ? "fsq-toast" : "fsq-toast fsq-toast-error";
    toast.setAttribute("role", ok ? "status" : "alert");
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  };

  const renderJob = (job, pending, active = false, historyIndex = null) => {
    const prompt = truncate(job.prompt);
    const negative = truncate(job.negative_prompt, "");
    const summary = job.summary || {};
    const forge = job.forge || {};
    const meta = [job.tab, summary.size, summary.steps ? `${summary.steps} steps` : "", summary.sampler, summary.schedule, summary.batch ? `batch ${summary.batch}` : "", job.error || ""].filter(Boolean).join(" | ");
    const forgeMeta = [forge.preset ? `preset ${forge.preset}` : "", forge.dtype, ...(forge.modules || [])].filter(Boolean).join(" | ");
    const field = (key, label, value, text = value) => {
      if (value == null) return "";
      if (!pending || job.editing) return escapeHtml(text);
      return `<button type="button" class="fsq-job-name" data-action="settings" data-id="${job.id}" data-field="${key}" data-value="${escapeHtml(value)}" data-label="${escapeHtml(label)}" title="Edit ${escapeHtml(label)}">${escapeHtml(text)}</button>`;
    };
    const settingsMeta = [
      summary.width != null && summary.height != null
        ? `${field("width", "Width", summary.width)}x${field("height", "Height", summary.height)}`
        : escapeHtml(summary.size || ""),
      field("steps", "Steps", summary.steps, `${summary.steps} steps`),
      escapeHtml(summary.sampler || ""), escapeHtml(summary.schedule || ""),
      field("cfg", "CFG", summary.cfg, `CFG ${summary.cfg}`),
      field("distilled_cfg", "Distilled CFG", summary.distilled_cfg, `Distilled CFG ${summary.distilled_cfg}`),
      field("denoising", "Denoising", summary.denoising, `Denoising ${summary.denoising}`),
      summary.batch_count != null && summary.batch_size != null
        ? `batch ${field("batch_count", "Batch count", summary.batch_count)}x${field("batch_size", "Batch size", summary.batch_size)}`
        : escapeHtml(summary.batch ? `batch ${summary.batch}` : "")
    ].filter(Boolean).join(" | ");
    const refinerMeta = [
      summary.refiner || (pending && summary.has_refiner)
        ? field("refiner", "Refiner", summary.refiner || "", summary.refiner || "Off")
        : "",
      field("refiner_switch", "Refiner switch (%)",
        summary.refiner_switch == null ? null : Math.round(summary.refiner_switch * 1000) / 10,
        `Switch at ${Math.round(summary.refiner_switch * 1000) / 10}%`),
      field("refiner_cfg", "Refiner CFG", summary.refiner_cfg, summary.refiner_cfg < 1 ? "Base CFG" : `CFG ${summary.refiner_cfg}`)
    ].filter(Boolean).join(" | ");
    const editable = Boolean(job.editable);
    const canRepeat = active || pending;
    const hasThumbnail = historyIndex !== null && Boolean(job.thumbnail);
    const editorOpen = openEditors.has(job.id) ? " fsq-open" : "";
    const detailsOpen = openDetails.has(job.id) ? " fsq-open" : "";
    const status = job.editing ? "editing" : (job.progress_queued && !job.progress_active ? "waiting" : job.status);
    const queuePosition = pending && Number.isInteger(job.index) ? `#${String(job.index + 1).padStart(2, "0")}` : "";
    const historyPosition = Number.isInteger(historyIndex) ? `#${String(historyIndex + 1).padStart(2, "0")}` : "";
    const name = job.name || "Unnamed job";
    const runs = Number(job.runs ?? 1);
    const failures = Number(job.failures || 0);
    const deleted = Number(job.deleted || 0);
    const skipped = Number(job.skipped || 0);
    const interrupted = Number(job.interrupted || 0);
    const duration = formatDuration(job.duration_seconds);
    const durationMeta = duration ? `${runs > 1 ? "Avg time" : "Time"}: ${duration}` : "";
    const runMeta = [runs > 1 ? `x${runs}` : "", failures ? `${failures} failed` : "", deleted ? `${deleted} deleted` : "", skipped ? `${skipped} skipped` : "", interrupted ? `${interrupted} stopped` : "", durationMeta].filter(Boolean).join(" | ");
    const utilityControls = job.source === "generate" ? "" : `
      <button class="fsq-icon-action" data-action="reuse" data-id="${job.id}" data-tab="${job.tab}" title="Reuse settings in ${job.tab}" aria-label="Reuse settings in ${job.tab}">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a7 7 0 0 0-12-2L3 9m0-5v5h5M6 16a7 7 0 0 0 12 2l3-3m0 5v-5h-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
      </button>
      <button class="fsq-icon-action" data-action="copy" data-id="${job.id}" title="Copy job to end of queue" aria-label="Copy job to end of queue">
        <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="2"></rect><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>
      </button>`;
    const deleteControl = !job.editing && (pending || (active && editable && !job.progress_active))
      ? jobActionButton("delete", job.id, "Delete job", icon.delete)
      : "";
    const controls = job.editing ? jobActionButton("details", job.id, "Details", icon.details) : active && job.progress_active ? `
      ${jobActionButton("interrupt", job.id, "Stop generation", icon.stop, job.tab)}
      ${jobActionButton("skip", job.id, "Skip generation", icon.skip, job.tab)}
      ${jobActionButton("details", job.id, "Details", icon.details)}` : active && editable ? `
      ${jobActionButton(job.paused ? "resume" : "pause", job.id, job.paused ? "Run job" : "Pause job", job.paused ? icon.play : icon.pause)}
      ${jobActionButton("edit", job.id, "Edit prompt", icon.edit)}
      ${jobActionButton("full-edit", job.id, "Full Edit", icon.fullEdit, job.tab)}
      ${jobActionButton("details", job.id, "Details", icon.details)}` : active ? `
      ${jobActionButton("details", job.id, "Details", icon.details)}` : pending ? `
      ${jobActionButton(job.paused ? "resume" : "pause", job.id, job.paused ? "Run job" : "Pause job", job.paused ? icon.play : icon.pause)}
      ${jobActionButton("edit", job.id, "Edit prompt", icon.edit)}
      ${jobActionButton("full-edit", job.id, "Full Edit", icon.fullEdit, job.tab)}
      ${jobActionButton("details", job.id, "Details", icon.details)}` : jobActionButton("details", job.id, "Details", icon.details);
    return `
      <div class="fsq-job" data-id="${job.id}" data-tab="${job.tab}" ${pending ? 'draggable="true"' : ""}>
        ${hasThumbnail ? "" : canRepeat ? `
          <label class="fsq-repeat-check" title="Select job for repeat or bulk delete">
            <input type="checkbox" data-action="repeat-job" data-id="${job.id}" ${job.repeat_selected ? "checked" : ""}>
        </label>` : '<span class="fsq-repeat-spacer"></span>'}
        ${hasThumbnail
          ? `<img class="fsq-thumbnail" src="${escapeHtml(job.thumbnail)}" alt="" loading="lazy" decoding="async">`
          : `<span class="fsq-handle" ${pending ? 'title="Drag to reorder"' : ""} aria-hidden="true">${pending ? "::" : ""}</span>`}
        <div>
          <div class="fsq-identity">
            ${queuePosition || historyPosition ? `<span>${queuePosition || historyPosition}</span>` : ""}
            <button type="button" class="fsq-job-name" data-action="rename" data-id="${job.id}" data-name="${escapeHtml(job.name || "")}" title="Rename job: ${escapeHtml(name)}">${escapeHtml(name)}</button>
            ${historyIndex === 0 ? '<span class="fsq-history-latest" title="Newest history item">Latest</span>' : ""}
          </div>
          <div class="fsq-prompt" title="${escapeHtml(prompt)}">${escapeHtml(prompt)}</div>
          <div class="fsq-meta" title="${escapeHtml(meta)}">${escapeHtml(meta)}</div>
          ${runMeta ? `<div class="fsq-meta">${escapeHtml(runMeta)}</div>` : ""}
          <div class="fsq-meta" title="${escapeHtml(negative)}">${negative ? "Negative: " + escapeHtml(negative) : ""}</div>
        </div>
        <div class="fsq-actions">
          <span class="fsq-status ${escapeHtml(status)}">${escapeHtml(status)}</span>
          ${controls}
          ${utilityControls}
          ${deleteControl}
        </div>
        ${editable ? `
          <div class="fsq-editor${editorOpen}" data-editor="${job.id}">
            <textarea data-field="prompt">${escapeHtml(job.prompt || "")}</textarea>
            <textarea data-field="negative_prompt">${escapeHtml(job.negative_prompt || "")}</textarea>
            <div class="fsq-editor-actions">
              <button class="fsq-save" data-action="save" data-id="${job.id}">Save</button>
              <button class="fsq-save fsq-cancel" data-action="cancel-edit" data-id="${job.id}">Cancel</button>
            </div>
          </div>` : ""}
        <div class="fsq-details-panel${detailsOpen}" data-details="${job.id}">
          <div><b>ID:</b> ${escapeHtml(job.id)} / ${escapeHtml(job.task_id || "")}</div>
          <div><b>Runs:</b> ${escapeHtml(runs)}${failures ? `, failed ${escapeHtml(failures)}` : ""}${deleted ? `, deleted ${escapeHtml(deleted)}` : ""}</div>
          ${forge.checkpoint ? `<div><b>Checkpoint:</b> ${field("checkpoint", "Checkpoint", forge.checkpoint)}</div>` : ""}
          ${refinerMeta ? `<div><b>Refiner:</b> ${refinerMeta}</div>` : ""}
          ${settingsMeta ? `<div><b>Generation:</b> ${settingsMeta}</div>` : ""}
          ${forgeMeta ? `<div><b>Forge:</b> ${escapeHtml(forgeMeta)}</div>` : ""}
          <div><b>Prompt:</b> ${escapeHtml(job.prompt || "")}</div>
          <div><b>Negative:</b> ${escapeHtml(job.negative_prompt || "")}</div>
        </div>
      </div>`;
  };

  let dragId = null;
  let editingName = false;
  let closeSettingsEditor = null;
  let historyMarkup = null;

  const editQueueSetting = async (button) => {
    closeSettingsEditor?.();
    const {id, field, label, value} = button.dataset;
    const modelField = field === "checkpoint" || field === "refiner";
    const editor = document.createElement("span");
    editor.className = "fsq-settings-editor";
    const bounds = button.getBoundingClientRect();
    editor.style.width = `${bounds.width}px`;
    editor.style.height = `${bounds.height}px`;
    const input = document.createElement("input");
    input.type = modelField ? "search" : "number";
    input.setAttribute("aria-label", label);
    input.step = "any";
    input.value = value;
    input.placeholder = modelField ? `Search ${label.toLowerCase()}` : "";
    editor.append(input);

    let saving = false;
    const close = () => {
      if (closeSettingsEditor !== close) return;
      editor.replaceWith(button);
      closeSettingsEditor = null;
    };
    closeSettingsEditor = close;
    button.replaceWith(editor);

    const save = async (nextValue) => {
      if (saving || closeSettingsEditor !== close) return;
      if (String(nextValue) === value) {
        close();
        return;
      }
      saving = true;
      const next = field === "refiner_switch" ? nextValue / 100 : nextValue;
      try {
        const result = await api("/forge-simple-queue/settings", {id, [field]: next});
        if (!result.ok) throw new Error(result.message);
        showQueueToast(result.message, true);
      } catch (err) {
        showQueueToast(err.message || "Could not update queued settings.", false);
      } finally {
        close();
        refreshQueueState({forceRender: true}).catch(console.error);
      }
    };

    editor.addEventListener("keydown", (event) => {
      if (event.isComposing) return;
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        close();
        button.focus();
      } else if (event.key === "Enter" && event.target === input) {
        event.preventDefault();
        if (modelField) editor.querySelector(".fsq-model-choice")?.click();
        else input.blur();
      } else if (event.key === "ArrowDown" && modelField && event.target === input) {
        event.preventDefault();
        editor.querySelector(".fsq-model-choice")?.focus();
      }
    });
    editor.addEventListener("focusout", (event) => {
      if (editor.contains(event.relatedTarget) || saving || closeSettingsEditor !== close) return;
      if (modelField || !input.value) close();
      else save(input.valueAsNumber);
    });
    input.focus({preventScroll: true});
    input.select();
    if (!modelField) return;

    try {
      const models = await api("/forge-simple-queue/checkpoints");
      if (closeSettingsEditor !== close) return;
      const list = document.createElement("span");
      list.className = "fsq-model-choices";
      editor.append(list);
      const bodyBounds = editor.closest(".fsq-body").getBoundingClientRect();
      const inputBounds = input.getBoundingClientRect();
      const above = inputBounds.top - bodyBounds.top;
      const below = bodyBounds.bottom - inputBounds.bottom;
      const openAbove = below < 132 && above > below;
      list.classList.toggle("fsq-model-choices-above", openAbove);
      list.style.maxHeight = `${Math.max(24, Math.min(128, (openAbove ? above : below) - 4))}px`;

      const choices = [...new Set([value, ...(field === "refiner" ? [""] : []), ...models])];
      const filter = (text = "") => {
        const query = text.trim().toLowerCase();
        list.replaceChildren();
        for (const model of choices) {
          const text = model || "Off";
          if (!text.toLowerCase().includes(query)) continue;
          const option = document.createElement("button");
          option.type = "button";
          option.className = "fsq-job-name fsq-model-choice";
          option.textContent = text;
          option.title = text;
          option.addEventListener("click", () => save(model));
          list.append(option);
        }
        if (!list.childElementCount) list.textContent = "No matching models";
      };
      input.addEventListener("input", () => filter(input.value));
      filter();
    } catch (err) {
      close();
      showQueueToast(err.message || "Could not load models.", false);
    }
  };

  const renderModal = (data) => {
    if (dragId || editingName || closeSettingsEditor) return;
    const body = document.getElementById("forge-simple-queue-body");
    if (!body) return;
    const view = activeModalTab();
    if (view === "history") {
      const history = (data.history || []).slice(0, 30).map((job, index) => renderJob(job, false, false, index)).join("");
      const markup = history || '<div class="fsq-empty">No history.</div>';
      if (markup !== historyMarkup) {
        body.innerHTML = markup;
        historyMarkup = markup;
      }
      return;
    }

    historyMarkup = null;
    const active = data.active ? renderJob(data.active, false, true) : "";
    const pending = (data.pending || []).map(job => renderJob(job, true)).join("");
    const etaSeconds = data?.eta?.seconds ?? data?.eta?.known_seconds;
    const unknownJobs = Number(data?.eta?.unknown_jobs || 0);
    const etaText = Number.isFinite(etaSeconds)
      ? `ETA ~ ${formatDuration(etaSeconds)}${unknownJobs ? "+" : ""}`
      : unknownJobs ? "ETA -" : "";
    const etaHint = data?.eta?.overdue ? "The running job has exceeded its estimate."
      : data?.eta?.stage === "preparing" ? "Estimate excludes time remaining in model preparation."
      : unknownJobs ? "Some jobs have no estimate."
      : data?.eta?.live ? "Estimate uses the current sampling speed."
      : data?.eta?.rough ? "Estimate based on less similar jobs." : "";
    const eta = etaText ? `<span class="fsq-eta"${etaHint ? ` title="${escapeHtml(etaHint)}"` : ""}>${escapeHtml(etaText)}</span>` : "";
    body.innerHTML = `
      ${data.active ? `<div class="fsq-section-title">Running${eta}</div>` + active : ''}
      <div class="fsq-section-title">Pending (${data.pending_count || 0})${data.active ? "" : eta}</div>
      ${pending || '<div class="fsq-empty">No pending jobs.</div>'}`;
  };

  const refreshModal = async () => {
    const data = await api("/forge-simple-queue/status");
    updateRepeatControls(data);
    updateRecoveryControls(data);
    updateQueueControls(data);
    updateSelectAllControl(data);
    updateBulkDeleteControl(data);
    updateSavedQueueControls(data);
    renderModal(data);
  };

  const queuePausedIcon = `<span class="fsq-queue-status-icon" aria-hidden="true"><svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M7 5h4v14H7zM13 5h4v14h-4z"></path></svg></span>`;

  const updateQueueButtons = (data) => {
    const queueCount = Number(data?.queue_count ?? data?.pending_count ?? 0);
    const active = data?.active;
    const running = Boolean(data?.generation_active || (active && (active.progress_active === true || active.status === "running")));
    const paused = Boolean(data?.control?.paused);
    const label = queueCount > 0 ? `Queue (${queueCount})` : (running ? "Queue (Running)" : "Queue");
    for (const id of ["txt2img_simple_queue_button", "img2img_simple_queue_button"]) {
      const button = document.getElementById(id);
      if (!button) continue;
      const wasPaused = button.classList.contains("fsq-queue-paused");
      if (paused && !wasPaused) {
        button.innerHTML = `Queue (${queuePausedIcon})`;
        button.classList.add("fsq-queue-paused");
        button.title = "Resume queue";
      } else if (!paused && wasPaused) {
        button.textContent = label;
        button.classList.remove("fsq-queue-paused");
        button.removeAttribute("title");
      } else if (!paused && button.textContent !== label) {
        button.textContent = label;
      }
    }
  };

  const updateRepeatControls = (data) => {
    const repeat = data?.repeat || {};
    const toggle = document.getElementById("forge-simple-queue-repeat-toggle");
    const count = document.getElementById("forge-simple-queue-repeat-count");
    if (toggle) toggle.checked = Boolean(repeat.enabled);
    if (count) {
      const selected = Number(repeat.count || 0);
      count.textContent = selected ? `${selected} task${selected === 1 ? "" : "s"}` : "none";
    }
  };

  const updateRecoveryControls = (data) => {
    const toggle = document.getElementById("forge-simple-queue-recovery-toggle");
    if (toggle) toggle.checked = data?.recovery?.enabled !== false;
  };

  const updateQueueControls = (data) => {
    const control = data?.control || {};
    const mode = control.paused ? "pause" : (control.mode || "play");
    for (const button of document.querySelectorAll("[data-queue-control]")) {
      const active = button.dataset.queueControl === mode;
      button.classList.toggle("fsq-control-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    }
  };

  const updateSelectAllControl = (data) => {
    const button = document.querySelector("[data-select-all-jobs]");
    if (!button) return;
    const queueJobs = queueJobsFromData(data);
    const allSelected = queueJobs.length > 0 && queueJobs.every((job) => job.repeat_selected);
    const queueView = activeModalTab() === "queue";
    button.disabled = !queueView || queueJobs.length === 0;
    button.classList.toggle("fsq-active", queueView && allSelected);
    button.setAttribute("aria-pressed", queueView && allSelected ? "true" : "false");
    button.title = !queueView ? "Select all visible jobs" : (allSelected ? "Clear all selected jobs" : "Select all visible jobs");
  };

  const updateBulkDeleteControl = (data) => {
    const button = document.querySelector("[data-bulk-delete-selected]");
    if (!button) return;
    const queueView = activeModalTab() === "queue";
    const count = queueView ? selectedDeletableJobs(data).length : 0;
    const visible = count > 1;
    button.hidden = !visible;
    button.disabled = !visible;
    button.title = visible ? `Delete ${jobLabel(count)} from the queue` : "Select multiple queued jobs to delete";
  };

  const updateSavedQueueControls = (data) => {
    const queueCount = Number(data?.queue_count ?? data?.pending_count ?? 0);
    const savedCount = Number(data?.saved_queue?.count || 0);
    const controls = {
      save: {
        disabled: queueCount <= 0,
        title: queueCount > 0 ? `Save queue (${queueCount} job${queueCount === 1 ? "" : "s"})` : "No queued jobs to save"
      },
      restore: {
        disabled: savedCount <= 0,
        title: savedCount > 0 ? `Restore queue (${savedCount} job${savedCount === 1 ? "" : "s"})` : "No saved queue"
      },
      clear: {
        disabled: savedCount <= 0,
        title: savedCount > 0 ? `Clear saved queue (${savedCount} job${savedCount === 1 ? "" : "s"})` : "No saved queue"
      }
    };
    for (const button of document.querySelectorAll("[data-saved-queue-action]")) {
      const action = button.dataset.savedQueueAction;
      const state = controls[action];
      if (!state) continue;
      button.disabled = state.disabled;
      button.title = state.title;
      button.classList.toggle("fsq-ready", !state.disabled);
    }
  };

  const followedTasks = new Set();
  const activeTaskByTab = new Map();
  let refreshQueueStateSerial = 0;

  const setTaskStorage = (tab, taskId) => {
    const key = `${tab}_task_id`;
    if (typeof localSet === "function") {
      localSet(key, taskId);
    } else {
      localStorage.setItem(key, taskId);
    }
  };

  const setCrossTabGenerationState = (data) => {
    const app = typeof gradioApp === "function" ? gradioApp() : document;
    const activeTab = data?.generation_active && ["txt2img", "img2img"].includes(data?.active?.tab)
      ? data.active.tab
      : null;
    for (const tab of ["txt2img", "img2img"]) {
      const generate = app.getElementById(`${tab}_generate`);
      let notice = app.getElementById(`${tab}_simple_queue_cross_tab_status`);
      const blocked = Boolean(activeTab && tab !== activeTab);
      if (blocked) {
        for (const button of [generate]) {
          if (!button || button.dataset.fsqCrossTabDisabled) continue;
          button.dataset.fsqCrossTabDisabled = "true";
          button.disabled = true;
        }
        if (!notice) {
          notice = document.createElement("div");
          notice.id = `${tab}_simple_queue_cross_tab_status`;
          notice.className = "fsq-cross-tab-status";
          app.getElementById(`${tab}_simple_queue_box`)?.appendChild(notice);
        }
        const message = `Generating in ${activeTab}…`;
        if (notice && notice.textContent !== message) notice.textContent = message;
      } else {
        for (const button of [generate]) {
          if (!button?.dataset.fsqCrossTabDisabled) continue;
          delete button.dataset.fsqCrossTabDisabled;
          button.disabled = false;
        }
        notice?.remove();
      }
    }
  };

  const syncActiveGeneration = (data) => {
    const active = data?.active;
    if (active?.tab && active?.task_id) {
      activeTaskByTab.set(active.tab, active.task_id);
      setTaskStorage(active.tab, active.task_id);
      if (typeof showSubmitButtons === "function") showSubmitButtons(active.tab, false);
      followTaskProgress(active.tab, active.task_id);
      setCrossTabGenerationState(data);
      return;
    }
    setCrossTabGenerationState(data);
  };

  const showSubmitButtonsIfIdle = (tab, data = null) => {
    if (typeof showSubmitButtons !== "function") return;
    const apply = (state) => {
      if (!state?.generation_active) showSubmitButtons(tab, true);
    };
    if (data) {
      apply(data);
    } else {
      api("/forge-simple-queue/status-lite").then(apply).catch(() => showSubmitButtons(tab, true));
    }
  };

  const refreshQueueState = async (options = {}) => {
    const serial = ++refreshQueueStateSerial;
    const modal = document.getElementById("forge-simple-queue-modal");
    const modalOpen = Boolean(modal?.classList.contains("fsq-open"));
    const data = await api(modalOpen || options.forceRender ? "/forge-simple-queue/status" : "/forge-simple-queue/status-lite");
    if (serial !== refreshQueueStateSerial) return data;
    updateQueueButtons(data);
    updateRepeatControls(data);
    updateRecoveryControls(data);
    updateQueueControls(data);
    updateSelectAllControl(data);
    updateBulkDeleteControl(data);
    updateSavedQueueControls(data);
    syncActiveGeneration(data);
    const focusedEditor = modal?.contains(document.activeElement) && document.activeElement.closest(".fsq-editor");
    const editing = openEditors.size > 0 || focusedEditor;
    if (modalOpen && (options.forceRender || !editing)) {
      renderModal(data);
    }
    return data;
  };

  const openModal = async () => {
    const modal = ensureModal();
    modal.classList.add("fsq-open");
    await refreshQueueState();
  };

  const restoreQueueModalAfterFullEdit = async () => {
    const returnTab = fullEditReturnModalTab;
    if (!returnTab) return;
    fullEditReturnModalTab = null;
    fullEditReturnJobId = null;
    fullEditJobName = "";
    await openModal();
    if (activeModalTab() === returnTab) return;
    document.querySelectorAll(".fsq-tab").forEach((button) => {
      button.classList.toggle("fsq-active", button.dataset.tabView === returnTab);
    });
    await refreshQueueState();
  };

  function followTaskProgress(tab, taskId) {
    if (!tab || !taskId || followedTasks.has(taskId)) return;
    if (typeof requestProgress !== "function") {
      console.warn("[Forge Simple Queue] requestProgress is not available.");
      return;
    }

    const app = typeof gradioApp === "function" ? gradioApp() : document;
    const galleryContainer = app.getElementById(`${tab}_gallery_container`);
    const gallery = app.getElementById(`${tab}_gallery`);

    if (!galleryContainer) {
      console.warn("[Forge Simple Queue] Gallery container was not found for", tab);
      return;
    }

    followedTasks.add(taskId);
    setTaskStorage(tab, taskId);

    const startProgress = () => {
      if (typeof showSubmitButtons === "function") showSubmitButtons(tab, false);
      requestProgress(
        taskId,
        galleryContainer,
        gallery,
        () => {
          showSubmitButtonsIfIdle(tab);
          if (typeof showRestoreProgressButton === "function") showRestoreProgressButton(tab, true);
          followedTasks.delete(taskId);
          activeTaskByTab.delete(tab);
          setTaskStorage(tab, taskId);
          setTimeout(() => {
            app.getElementById(`${tab}_restore_progress`)?.click();
            refreshQueueState().catch((err) => console.error("[Forge Simple Queue]", err));
          }, 120);
        },
        null,
        3600
      );
    };

    const waitForActive = async () => {
      const deadline = Date.now() + 60 * 60 * 1000;
      while (Date.now() < deadline) {
        const state = await api("/forge-simple-queue/status-lite");
        updateQueueButtons(state);
        if (state.active?.task_id === taskId && state.active.progress_active !== false) {
          startProgress();
          return;
        }
        if ((state.recent_tasks || []).includes(taskId) || (state.history || []).some((job) => job.task_id === taskId)) {
          showSubmitButtonsIfIdle(tab, state);
          followedTasks.delete(taskId);
          activeTaskByTab.delete(tab);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 650));
      }
      followedTasks.delete(taskId);
      showSubmitButtonsIfIdle(tab);
    };

    waitForActive().catch((err) => {
      console.error("[Forge Simple Queue]", err);
      followedTasks.delete(taskId);
      showSubmitButtonsIfIdle(tab);
    });
  }

  let queuedStatusRefreshTimer = null;
  const transientStatusTimers = new WeakMap();
  const clearTransientStatus = (node) => {
    const marker = node?.querySelector?.("[data-fsq-full-edit-job]");
    if (!marker || transientStatusTimers.has(marker)) return;
    const name = marker.dataset.fsqFullEditJob === fullEditReturnJobId ? fullEditJobName : "";
    showQueueToast(name ? `${name}: ${marker.textContent}` : marker.textContent, marker.dataset.fsqOk === "1");
    if (marker.dataset.fsqClose === "1" && marker.dataset.fsqFullEditJob === fullEditReturnJobId) {
      restoreQueueModalAfterFullEdit().catch((err) => console.error("[Forge Simple Queue]", err));
    }
    transientStatusTimers.set(marker, setTimeout(() => marker.remove(), 2600));
  };
  const scheduleQueuedStatusRefresh = () => {
    if (queuedStatusRefreshTimer) return;
    queuedStatusRefreshTimer = setTimeout(() => {
      queuedStatusRefreshTimer = null;
      refreshQueueState().catch((err) => console.error("[Forge Simple Queue]", err));
    }, 180);
  };

  const scanQueuedStatus = () => {
    const app = typeof gradioApp === "function" ? gradioApp() : document;
    for (const id of ["txt2img_simple_queue_status", "img2img_simple_queue_status"]) {
      clearTransientStatus(app.getElementById(id));
    }
    scheduleQueuedStatusRefresh();
  };

  const statusObserver = new MutationObserver(scanQueuedStatus);
  const startStatusObserver = () => {
    const app = typeof gradioApp === "function" ? gradioApp() : document;
    let attached = false;
    for (const id of ["txt2img_simple_queue_status", "img2img_simple_queue_status"]) {
      const node = app.getElementById(id);
      clearTransientStatus(node);
      if (node && !node.dataset.fsqObserved) {
        node.dataset.fsqObserved = "true";
        statusObserver.observe(node, {childList: true, subtree: true});
        attached = true;
      }
    }
    if (attached) scheduleQueuedStatusRefresh();
  };

  document.addEventListener("wheel", (event) => {
    if (event.target.closest(".fsq-editor textarea")) {
      event.stopPropagation();
    }
  }, {capture: true, passive: true});

  document.addEventListener("click", async (event) => {
    const tabButton = event.target.closest(".fsq-tab[data-tab-view]");
    if (tabButton) {
      event.preventDefault();
      event.stopPropagation();
      closeSettingsEditor?.();
      document.querySelectorAll(".fsq-tab").forEach((button) => button.classList.toggle("fsq-active", button === tabButton));
      await refreshQueueState();
      return;
    }

    const queueButton = event.target.closest("#txt2img_simple_queue_button, #img2img_simple_queue_button");
    if (queueButton && queueButton.classList.contains("fsq-queue-paused")) {
      event.preventDefault();
      event.stopPropagation();
      queueButton.disabled = true;
      try {
        await api("/forge-simple-queue/control", {action: "play"});
        await refreshQueueState();
      } catch (err) {
        console.error("[Forge Simple Queue]", err);
      } finally {
        queueButton.disabled = false;
      }
      return;
    }

    const queueControlButton = event.target.closest("[data-queue-control]");
    if (queueControlButton) {
      event.preventDefault();
      event.stopPropagation();
      const action = queueControlButton.dataset.queueControl;
      queueControlButton.classList.add("fsq-control-working");
      try {
        const before = action === "stop" ? await api("/forge-simple-queue/status") : null;
        await api("/forge-simple-queue/control", {action});
        if (action === "stop" && before?.active?.tab && before.active.progress_active !== false) {
          const tab = before.active.tab;
          document.getElementById(`${tab}_interrupt`)?.click();
        }
        await refreshQueueState();
      } catch (err) {
        console.error("[Forge Simple Queue]", err);
      } finally {
        queueControlButton.classList.remove("fsq-control-working");
      }
      return;
    }

    const savedQueueButton = event.target.closest("[data-saved-queue-action]");
    if (savedQueueButton) {
      event.preventDefault();
      event.stopPropagation();
      if (savedQueueButton.disabled) return;
      const action = savedQueueButton.dataset.savedQueueAction;
      try {
        const data = await api("/forge-simple-queue/status");
        const queueCount = Number(data?.queue_count ?? data?.pending_count ?? 0);
        const savedCount = Number(data?.saved_queue?.count || 0);
        let message = "";
        if (action === "save") {
          if (queueCount <= 0) {
            alert("No queued jobs to save.");
            return;
          }
          message = `Save ${jobLabel(queueCount)} to the saved queue?`;
        } else if (action === "restore") {
          if (savedCount <= 0) {
            alert("No saved queue to restore.");
            return;
          }
          message = `Restore ${jobLabel(savedCount)} from the saved queue?`;
          if (queueCount > 0) {
            message += `\n\nThis will replace the current ${jobLabel(queueCount)}.`;
          }
        } else if (action === "clear") {
          if (savedCount <= 0) {
            alert("No saved queue to clear.");
            return;
          }
          message = `Clear ${jobLabel(savedCount)} from the saved queue?`;
        } else {
          return;
        }
        if (!confirm(message)) return;
        savedQueueButton.classList.add("fsq-control-working");
        const result = await api(`/forge-simple-queue/saved-queue/${action}`, {});
        if (result?.ok === false && result?.message) alert(result.message);
        await refreshQueueState({forceRender: true});
      } catch (err) {
        console.error("[Forge Simple Queue]", err);
      } finally {
        savedQueueButton.classList.remove("fsq-control-working");
      }
      return;
    }

    const selectAllButton = event.target.closest("[data-select-all-jobs]");
    if (selectAllButton) {
      event.preventDefault();
      event.stopPropagation();
      if (activeModalTab() !== "queue") return;
      try {
        const data = await api("/forge-simple-queue/status");
        const queueJobs = [
          ...(data?.active ? [data.active] : []),
          ...(data?.pending || [])
        ];
        const allSelected = queueJobs.length > 0 && queueJobs.every((job) => job.repeat_selected);
        await api("/forge-simple-queue/repeat-all", {selected: !allSelected});
        await refreshQueueState({forceRender: true});
      } catch (err) {
        console.error("[Forge Simple Queue]", err);
      }
      return;
    }

    const bulkDeleteButton = event.target.closest("[data-bulk-delete-selected]");
    if (bulkDeleteButton) {
      event.preventDefault();
      event.stopPropagation();
      if (bulkDeleteButton.disabled || activeModalTab() !== "queue") return;
      try {
        const data = await api("/forge-simple-queue/status");
        const jobs = selectedDeletableJobs(data);
        if (jobs.length <= 1) {
          await refreshQueueState({forceRender: true});
          return;
        }
        if (!confirm(`Delete ${jobLabel(jobs.length)} from the queue?`)) return;
        bulkDeleteButton.classList.add("fsq-control-working");
        const ids = jobs.map((job) => job.id);
        const result = await api("/forge-simple-queue/delete-many", {ids});
        for (const id of ids) {
          openEditors.delete(id);
          openDetails.delete(id);
        }
        if (result?.skipped_count) {
          console.warn("[Forge Simple Queue] Some selected jobs were not deleted.", result);
        }
        await refreshQueueState({forceRender: true});
      } catch (err) {
        console.error("[Forge Simple Queue]", err);
      } finally {
        bulkDeleteButton.classList.remove("fsq-control-working");
      }
      return;
    }

    const viewButton = event.target.closest("#txt2img_simple_queue_view, #img2img_simple_queue_view");
    if (viewButton) {
      event.preventDefault();
      event.stopPropagation();
      await openModal();
      return;
    }

    const autoRepeatLabel = event.target.closest(".fsq-repeat-toggle");
    if (autoRepeatLabel) {
      event.preventDefault();
      event.stopPropagation();
      const input = autoRepeatLabel.querySelector("input");
      const enabled = !input.checked;
      input.checked = enabled;
      await api(input.id === "forge-simple-queue-recovery-toggle" ? "/forge-simple-queue/recovery" : "/forge-simple-queue/repeat", {enabled});
      await refreshQueueState();
      return;
    }

    const nativeInterrupt = event.target.closest("#txt2img_interrupt, #img2img_interrupt");
    if (nativeInterrupt) {
      const tab = nativeInterrupt.id.startsWith("img2img") ? "img2img" : "txt2img";
      if (activeTaskByTab.has(tab)) {
        api("/forge-simple-queue/control", {action: "stop"}).catch((err) => console.error("[Forge Simple Queue]", err));
      }
      return;
    }

    const nativeSkip = event.target.closest("#txt2img_skip, #img2img_skip");
    if (nativeSkip) {
      api("/forge-simple-queue/skip", {}).catch((err) => console.error("[Forge Simple Queue]", err));
      return;
    }

    const actionButton = event.target.closest("[data-action]");
    if (!actionButton) return;
    event.preventDefault();
    event.stopPropagation();
    const action = actionButton.dataset.action;
    const id = actionButton.dataset.id;
    try {
      if (action === "repeat-job") {
        await api("/forge-simple-queue/repeat", {id, selected: actionButton.checked});
        await refreshQueueState();
      } else if (action === "rename" && !editingName) {
        editingName = true;
        const input = document.createElement("input");
        input.type = "text";
        input.className = "fsq-name-input";
        input.setAttribute("aria-label", "Job name");
        input.maxLength = 60;
        input.value = actionButton.dataset.name;
        input.addEventListener("keydown", (event) => {
          if (event.isComposing || !["Enter", "Escape"].includes(event.key)) return;
          event.preventDefault();
          event.stopPropagation();
          if (event.key === "Escape") input.value = actionButton.dataset.name;
          input.blur();
        });
        input.addEventListener("blur", async () => {
          const name = input.value.trim();
          try {
            if (name && name !== actionButton.dataset.name) {
              const result = await api("/forge-simple-queue/rename", {id, name});
              if (!result.ok) throw new Error(result.message);
              actionButton.textContent = result.name;
              actionButton.dataset.name = result.name;
              actionButton.title = `Rename job: ${result.name}`;
            }
          } catch (err) {
            console.error("[Forge Simple Queue]", err);
          } finally {
            input.replaceWith(actionButton);
            editingName = false;
          }
        }, {once: true});
        actionButton.replaceWith(input);
        input.focus();
        input.select();
      } else if (action === "settings") {
        await editQueueSetting(actionButton);
      } else if (action === "edit") {
        if (openEditors.has(id)) {
          openEditors.delete(id);
        } else {
          openEditors.add(id);
        }
        document.querySelector(`[data-editor="${id}"]`)?.classList.toggle("fsq-open", openEditors.has(id));
      } else if (action === "full-edit") {
        openEditors.delete(id);
        const row = actionButton.closest(".fsq-job");
        await startFullEdit(id, actionButton.dataset.tab || row?.dataset.tab || "txt2img", row?.querySelector('[data-action="rename"]')?.textContent?.trim());
      } else if (action === "reuse") {
        await startReuse(id, actionButton.dataset.tab || actionButton.closest(".fsq-job")?.dataset.tab || "txt2img");
      } else if (action === "cancel-edit") {
        openEditors.delete(id);
        await refreshQueueState({forceRender: true});
      } else if (action === "details") {
        if (openDetails.has(id)) {
          openDetails.delete(id);
        } else {
          openDetails.add(id);
        }
        document.querySelector(`[data-details="${id}"]`)?.classList.toggle("fsq-open", openDetails.has(id));
      } else if (action === "interrupt" || action === "skip") {
        const tab = actionButton.dataset.tab || actionButton.closest(".fsq-job")?.dataset.tab || "txt2img";
        await api(`/forge-simple-queue/${action}`, {});
        const target = document.getElementById(`${tab}_${action === "interrupt" ? "interrupt" : "skip"}`);
        target?.click();
        await refreshQueueState();
      } else if (action === "save") {
        const editor = document.querySelector(`[data-editor="${id}"]`);
        await api("/forge-simple-queue/update", {
          id,
          prompt: editor?.querySelector('[data-field="prompt"]')?.value,
          negative_prompt: editor?.querySelector('[data-field="negative_prompt"]')?.value
        });
        openEditors.delete(id);
        await refreshQueueState({forceRender: true});
      } else {
        await api(`/forge-simple-queue/${action}`, {id});
        if (action === "delete") {
          openEditors.delete(id);
          openDetails.delete(id);
        }
        await refreshQueueState();
      }
    } catch (err) {
      console.error("[Forge Simple Queue]", err);
    }
  }, true);

  let dragTarget = null;
  const clearDragTarget = () => {
    dragTarget?.classList.remove("fsq-drop-before", "fsq-drop-after");
    dragTarget = null;
  };
  document.addEventListener("dragstart", (event) => {
    const row = event.target.closest(".fsq-job[draggable='true']");
    if (!row || editingName || closeSettingsEditor) return;
    dragId = row.dataset.id;
    row.classList.add("fsq-dragging");
    event.dataTransfer.effectAllowed = "move";
  });
  document.addEventListener("dragend", (event) => {
    event.target.closest(".fsq-job")?.classList.remove("fsq-dragging");
    clearDragTarget();
    dragId = null;
  });
  document.addEventListener("dragover", (event) => {
    if (!dragId) return;
    const row = event.target.closest(".fsq-job[draggable='true']");
    if (!row || row.dataset.id === dragId) return;
    event.preventDefault();
    clearDragTarget();
    dragTarget = row;
    const rect = row.getBoundingClientRect();
    row.classList.add(event.clientY > rect.top + rect.height / 2 ? "fsq-drop-after" : "fsq-drop-before");
  });
  document.addEventListener("drop", async (event) => {
    const row = event.target.closest(".fsq-job[draggable='true']");
    if (!row || !dragId || row.dataset.id === dragId) return;
    event.preventDefault();
    const movedId = dragId;
    const rect = row.getBoundingClientRect();
    const after = event.clientY > rect.top + rect.height / 2;
    const positions = captureJobPositions();
    clearDragTarget();
    try {
      await api("/forge-simple-queue/move", {id: movedId, target_id: row.dataset.id, after});
    } catch (err) {
      console.error("[Forge Simple Queue]", err);
    } finally {
      await refreshQueueState({forceRender: true});
    }
    animateReorder(positions, movedId);
  });

  const keepQueueButtonsAlive = () => {
    for (const selector of ["#txt2img_simple_queue_button", "#img2img_simple_queue_button", "#txt2img_simple_queue_view", "#img2img_simple_queue_view"]) {
      const button = document.querySelector(selector);
      if (!button) continue;
      if (button.dataset.fsqCrossTabDisabled) continue;
      button.disabled = false;
      button.removeAttribute("disabled");
      button.classList.remove("disabled");
      button.style.pointerEvents = "auto";
      if (selector.endsWith("_view")) button.title = "View queue";
    }
    for (const [selector, title] of [
      ["#txt2img_simple_queue_update", "Update queued job"],
      ["#img2img_simple_queue_update", "Update queued job"],
      ["#txt2img_simple_queue_cancel", "Cancel Full Edit"],
      ["#img2img_simple_queue_cancel", "Cancel Full Edit"]
    ]) {
      const button = document.querySelector(selector);
      if (button) button.title = title;
    }
  };

  let queueStatePollTimer = null;
  let queueStatePolling = false;

  const queueStatePollDelay = (data) => {
    const modal = document.getElementById("forge-simple-queue-modal");
    const modalOpen = Boolean(modal?.classList.contains("fsq-open"));
    const active = Boolean(data?.generation_active || data?.active);
    const queueCount = Number(data?.queue_count ?? data?.pending_count ?? 0);
    if (modalOpen) return data?.eta?.live_enabled ? 1000 : active ? 3000 : 1500;
    if (document.hidden) return active || queueCount > 0 ? 5000 : 12000;
    if (active || queueCount > 0) return 2500;
    return 8000;
  };

  const scheduleQueueStatePoll = (delay) => {
    clearTimeout(queueStatePollTimer);
    queueStatePollTimer = setTimeout(pollQueueState, delay);
  };

  const pollQueueState = async () => {
    if (queueStatePolling) {
      scheduleQueueStatePoll(1000);
      return;
    }
    queueStatePolling = true;
    let data = null;
    try {
      data = await refreshQueueState();
    } catch (err) {
      console.error("[Forge Simple Queue]", err);
    } finally {
      queueStatePolling = false;
      scheduleQueueStatePoll(queueStatePollDelay(data));
    }
  };

  setInterval(keepQueueButtonsAlive, 1500);
  setInterval(startStatusObserver, 2000);
  scheduleQueueStatePoll(1000);
  document.addEventListener("visibilitychange", () => scheduleQueueStatePoll(document.hidden ? 5000 : 250));
  document.addEventListener("DOMContentLoaded", keepQueueButtonsAlive);
  document.addEventListener("DOMContentLoaded", startStatusObserver);
  keepQueueButtonsAlive();
  startStatusObserver();
  refreshQueueState().catch(() => {});
})();

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
          </div>
          <div class="fsq-head-actions">
            <button class="fsq-select-all" data-select-all-jobs="true" title="Select all visible jobs" aria-label="Select all visible jobs">
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16v14H4zM7 9l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>
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
      if (event.target === modal || event.target.classList.contains("fsq-close")) {
        modal.classList.remove("fsq-open");
      }
    });
    return modal;
  };

  const openEditors = new Set();
  const openDetails = new Set();

  const activeModalTab = () => document.querySelector(".fsq-tab.fsq-active")?.dataset.tabView || "queue";

  const queueJobsFromData = (data) => [
    ...(data?.active ? [data.active] : []),
    ...(data?.pending || [])
  ];

  const selectedDeletableJobs = (data) => queueJobsFromData(data).filter((job) => (
    job?.repeat_selected
    && job.status !== "running"
    && job.progress_active !== true
  ));

  const renderJob = (job, pending, active = false) => {
    const prompt = truncate(job.prompt);
    const negative = truncate(job.negative_prompt, "");
    const summary = job.summary || {};
    const forge = job.forge || {};
    const meta = [job.tab, summary.size, summary.steps ? `${summary.steps} steps` : "", summary.sampler, summary.schedule, summary.batch ? `batch ${summary.batch}` : "", job.error || ""].filter(Boolean).join(" | ");
    const forgeMeta = [forge.preset ? `preset ${forge.preset}` : "", forge.checkpoint, forge.dtype, ...(forge.modules || [])].filter(Boolean).join(" | ");
    const editable = Boolean(job.editable);
    const canRepeat = active || pending;
    const editorOpen = openEditors.has(job.id) ? " fsq-open" : "";
    const detailsOpen = openDetails.has(job.id) ? " fsq-open" : "";
    const status = job.progress_queued && !job.progress_active ? "waiting" : job.status;
    const runs = Number(job.runs || 1);
    const failures = Number(job.failures || 0);
    const deleted = Number(job.deleted || 0);
    const runMeta = [runs > 1 ? `x${runs}` : "", failures ? `${failures} failed` : "", deleted ? `${deleted} deleted` : ""].filter(Boolean).join(" | ");
    const controls = active && job.progress_active ? `
      <button data-action="interrupt" data-tab="${job.tab}">Stop</button>
      <button data-action="skip" data-tab="${job.tab}">Skip</button>
      <button data-action="details" data-id="${job.id}">Details</button>` : active && editable ? `
      <button data-action="${job.paused ? "resume" : "pause"}" data-id="${job.id}">${job.paused ? "Run" : "Pause"}</button>
      <button data-action="edit" data-id="${job.id}">Edit</button>
      <button data-action="details" data-id="${job.id}">Details</button>
      <button data-action="delete" data-id="${job.id}">Delete</button>` : active ? `
      <button data-action="details" data-id="${job.id}">Details</button>` : pending ? `
      <button data-action="${job.paused ? "resume" : "pause"}" data-id="${job.id}">${job.paused ? "Run" : "Pause"}</button>
      <button data-action="edit" data-id="${job.id}">Edit</button>
      <button data-action="details" data-id="${job.id}">Details</button>
      <button data-action="delete" data-id="${job.id}">Delete</button>` : `
      <button data-action="details" data-id="${job.id}">Details</button>`;
    return `
      <div class="fsq-job" data-id="${job.id}" data-tab="${job.tab}" ${pending ? 'draggable="true"' : ""}>
        ${canRepeat ? `
          <label class="fsq-repeat-check" title="Select job for repeat or bulk delete">
            <input type="checkbox" data-action="repeat-job" data-id="${job.id}" ${job.repeat_selected ? "checked" : ""}>
          </label>` : '<span class="fsq-repeat-spacer"></span>'}
        <button class="fsq-handle" title="Drag to reorder">::</button>
        <div>
          <div class="fsq-prompt" title="${escapeHtml(prompt)}">${escapeHtml(prompt)}</div>
          <div class="fsq-meta" title="${escapeHtml(meta)}">${escapeHtml(meta)}</div>
          ${runMeta ? `<div class="fsq-meta">${escapeHtml(runMeta)}</div>` : ""}
          <div class="fsq-meta" title="${escapeHtml(negative)}">${negative ? "Negative: " + escapeHtml(negative) : ""}</div>
        </div>
        <div class="fsq-actions">
          <span class="fsq-status ${escapeHtml(status)}">${escapeHtml(status)}</span>
          ${controls}
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
        <div class="fsq-details${detailsOpen}" data-details="${job.id}">
          <div><b>ID:</b> ${escapeHtml(job.id)} / ${escapeHtml(job.task_id || "")}</div>
          <div><b>Runs:</b> ${escapeHtml(runs)}${failures ? `, failed ${escapeHtml(failures)}` : ""}${deleted ? `, deleted ${escapeHtml(deleted)}` : ""}</div>
          <div><b>Forge:</b> ${escapeHtml(forgeMeta)}</div>
          <div><b>Prompt:</b> ${escapeHtml(job.prompt || "")}</div>
          <div><b>Negative:</b> ${escapeHtml(job.negative_prompt || "")}</div>
        </div>
      </div>`;
  };

  const renderModal = (data) => {
    const body = document.getElementById("forge-simple-queue-body");
    if (!body) return;
    const view = activeModalTab();
    if (view === "history") {
      const history = (data.history || []).slice(0, 30).map(job => renderJob(job, false)).join("");
      body.innerHTML = `${history || '<div class="fsq-empty">No history.</div>'}`;
      return;
    }

    const active = data.active ? renderJob(data.active, false, true) : "";
    const pending = (data.pending || []).map(job => renderJob(job, true)).join("");
    body.innerHTML = `
      ${data.active ? '<div class="fsq-section-title">Running</div>' + active : ''}
      <div class="fsq-section-title">Pending (${data.pending_count || 0})</div>
      ${pending || '<div class="fsq-empty">No pending jobs.</div>'}`;
  };

  const refreshModal = async () => {
    const data = await api("/forge-simple-queue/status");
    updateRepeatControls(data);
    updateQueueControls(data);
    updateSelectAllControl(data);
    updateBulkDeleteControl(data);
    updateSavedQueueControls(data);
    renderModal(data);
  };

  const updateQueueButtons = (data) => {
    const queueCount = Number(data?.queue_count ?? data?.pending_count ?? 0);
    const active = data?.active;
    const running = Boolean(data?.generation_active || (active && (active.progress_active === true || active.status === "running")));
    const label = queueCount > 0 ? `Queue (${queueCount})` : (running ? "Queue (Running)" : "Queue");
    for (const id of ["txt2img_simple_queue_button", "img2img_simple_queue_button"]) {
      const button = document.getElementById(id);
      if (button && button.textContent !== label) button.textContent = label;
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

  const syncActiveGeneration = (data) => {
    const active = data?.active;
    if (active?.tab && active?.task_id) {
      activeTaskByTab.set(active.tab, active.task_id);
      setTaskStorage(active.tab, active.task_id);
      if (typeof showSubmitButtons === "function") showSubmitButtons(active.tab, false);
      followTaskProgress(active.tab, active.task_id);
      return;
    }
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
  const scheduleQueuedStatusRefresh = () => {
    if (queuedStatusRefreshTimer) return;
    queuedStatusRefreshTimer = setTimeout(() => {
      queuedStatusRefreshTimer = null;
      refreshQueueState().catch((err) => console.error("[Forge Simple Queue]", err));
    }, 180);
  };

  const scanQueuedStatus = () => {
    scheduleQueuedStatusRefresh();
  };

  const statusObserver = new MutationObserver(scanQueuedStatus);
  const startStatusObserver = () => {
    let attached = false;
    for (const id of ["txt2img_simple_queue_status", "img2img_simple_queue_status"]) {
      const node = document.getElementById(id);
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
      document.querySelectorAll(".fsq-tab").forEach((button) => button.classList.toggle("fsq-active", button === tabButton));
      await refreshQueueState();
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
      const input = autoRepeatLabel.querySelector("#forge-simple-queue-repeat-toggle");
      const enabled = !input.checked;
      input.checked = enabled;
      await api("/forge-simple-queue/repeat", {enabled});
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
      } else if (action === "edit") {
        if (openEditors.has(id)) {
          openEditors.delete(id);
        } else {
          openEditors.add(id);
        }
        document.querySelector(`[data-editor="${id}"]`)?.classList.toggle("fsq-open", openEditors.has(id));
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

  let dragId = null;
  document.addEventListener("dragstart", (event) => {
    const row = event.target.closest(".fsq-job[draggable='true']");
    if (!row) return;
    dragId = row.dataset.id;
    row.classList.add("fsq-dragging");
    event.dataTransfer.effectAllowed = "move";
  });
  document.addEventListener("dragend", (event) => {
    event.target.closest(".fsq-job")?.classList.remove("fsq-dragging");
    dragId = null;
  });
  document.addEventListener("dragover", (event) => {
    if (!dragId) return;
    const row = event.target.closest(".fsq-job[draggable='true']");
    if (row) event.preventDefault();
  });
  document.addEventListener("drop", async (event) => {
    const row = event.target.closest(".fsq-job[draggable='true']");
    if (!row || !dragId || row.dataset.id === dragId) return;
    event.preventDefault();
    const rows = [...document.querySelectorAll(".fsq-job[draggable='true']")];
    const rect = row.getBoundingClientRect();
    let index = rows.indexOf(row);
    if (event.clientY > rect.top + rect.height / 2) index += 1;
    await api("/forge-simple-queue/move", {id: dragId, index});
    await refreshQueueState();
  });

  const keepQueueButtonsAlive = () => {
    for (const selector of ["#txt2img_simple_queue_button", "#img2img_simple_queue_button", "#txt2img_simple_queue_view", "#img2img_simple_queue_view"]) {
      const button = document.querySelector(selector);
      if (!button) continue;
      button.disabled = false;
      button.removeAttribute("disabled");
      button.classList.remove("disabled");
      button.style.pointerEvents = "auto";
      if (selector.endsWith("_view")) button.title = "View queue";
    }
  };

  let queueStatePollTimer = null;
  let queueStatePolling = false;

  const queueStatePollDelay = (data) => {
    const modal = document.getElementById("forge-simple-queue-modal");
    const modalOpen = Boolean(modal?.classList.contains("fsq-open"));
    const active = Boolean(data?.generation_active || data?.active);
    const queueCount = Number(data?.queue_count ?? data?.pending_count ?? 0);
    if (modalOpen) return 1500;
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

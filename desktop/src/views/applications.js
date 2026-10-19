import { call, setStatus } from "../api.js";

let applicationsLoaded = false;

const TERMINAL_STATUSES = new Set(["rejected", "skipped"]);
const STATUS_OPTIONS = ["saved", "applied", "screening", "interview", "rejected", "skipped"];

function isOverdue(app) {
  if (!app.follow_up_date || TERMINAL_STATUSES.has(app.status)) return false;
  return app.follow_up_date < new Date().toISOString().slice(0, 10);
}

function applicationCard(app) {
  const card = document.createElement("div");
  card.className = "card";

  const main = document.createElement("div");
  main.className = "posting-main";
  const h3 = document.createElement("h3");
  h3.textContent = app.posting_title;
  const subtitle = document.createElement("div");
  subtitle.className = "subtitle";
  subtitle.textContent = app.posting_company_name;
  main.append(h3, subtitle);

  if (app.posting_freshness === "closed") {
    const warn = document.createElement("span");
    warn.className = "badge tone-weak";
    warn.textContent = "posting closed";
    main.appendChild(warn);
  }
  if (isOverdue(app)) {
    const overdue = document.createElement("span");
    overdue.className = "badge tone-weak";
    overdue.textContent = `follow-up overdue (${app.follow_up_date})`;
    main.appendChild(overdue);
  }

  const form = document.createElement("div");
  form.className = "application-form";

  const statusSelect = document.createElement("select");
  statusSelect.innerHTML = STATUS_OPTIONS.map((s) => `<option value="${s}">${s}</option>`).join("");
  statusSelect.value = app.status ?? "saved";

  const resumeInput = document.createElement("input");
  resumeInput.placeholder = "Resume used";
  resumeInput.value = app.resume_used ?? "";

  const dateAppliedInput = document.createElement("input");
  dateAppliedInput.type = "date";
  dateAppliedInput.title = "Date applied";
  dateAppliedInput.value = app.date_applied ?? "";

  const followUpInput = document.createElement("input");
  followUpInput.type = "date";
  followUpInput.title = "Follow-up date";
  followUpInput.value = app.follow_up_date ?? "";

  const row1 = document.createElement("div");
  row1.className = "row application-form-row";
  row1.append(statusSelect, resumeInput, dateAppliedInput, followUpInput);

  const notesInput = document.createElement("textarea");
  notesInput.className = "note-box";
  notesInput.placeholder = "Notes...";
  notesInput.value = app.notes ?? "";

  const saveBtn = document.createElement("button");
  saveBtn.className = "btn-secondary application-save-btn";
  saveBtn.textContent = "Save";
  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    const updated = await call("set_application_status", {
      postingId: app.posting_id,
      status: statusSelect.value,
      resumeUsed: resumeInput.value || null,
      dateApplied: dateAppliedInput.value || null,
      followUpDate: followUpInput.value || null,
      notes: notesInput.value || null,
    });
    saveBtn.disabled = false;
    setStatus(`Saved "${app.posting_title}".`, "success");
    // Reflect anything the backend auto-set (e.g. date_applied) without a
    // full list reload.
    dateAppliedInput.value = updated.date_applied ?? "";
  });

  form.append(row1, notesInput, saveBtn);
  card.append(main, form);
  return card;
}

export async function loadApplications() {
  applicationsLoaded = true;
  const list = document.getElementById("applicationsList");
  list.innerHTML = "";
  setStatus("Loading applications...");
  const status = document.getElementById("applicationStatusFilter").value || null;
  const sort = document.getElementById("applicationSort").value || "updated";
  const apps = await call("list_applications", { status, sort });
  setStatus("");
  if (apps.length === 0) {
    list.innerHTML = '<div class="empty-state">No applications tracked yet.</div>';
    return;
  }
  apps.forEach((a) => list.appendChild(applicationCard(a)));
}

export function initApplicationsView() {
  document.getElementById("applyApplicationFilter").addEventListener("click", () => {
    applicationsLoaded = false;
    loadApplications();
  });
  document.getElementById("exportApplicationsCsv").addEventListener("click", async () => {
    const { save } = window.__TAURI__.dialog;
    const path = await save({ defaultPath: "applications.csv", filters: [{ name: "CSV", extensions: ["csv"] }] });
    if (!path) return; // user cancelled
    setStatus("Exporting...");
    await call("export_applications_csv", { destPath: path });
    setStatus(`Exported to ${path}.`, "success");
  });
}

export function applicationsTabActivated() {
  if (!applicationsLoaded) loadApplications();
}

// The Profile tab (3.9 display + 4.4-follow-up editing). Read-only view
// plus a full "Edit profile" form -- POST /profile (4.4) already accepts
// a full replacement snapshot + changelog; this is the frontend for it.
import { call, setStatus } from "../api.js";
import { mountRepeatableList, mountTagList } from "../components.js";
import { esc } from "../escape.js";

let profileLoaded = false;
let currentSnapshot = null;

const EXPERIENCE_FIELDS = [
  { key: "company", label: "Company", type: "text" },
  { key: "title", label: "Title", type: "text" },
  { key: "date_range", label: "Date range", type: "text" },
  { key: "signals", label: "Signals", type: "taglist" },
  { key: "weight", label: "Weight", type: "select", options: ["high", "medium", "low"], default: "medium" },
];

const PROJECT_FIELDS = [
  { key: "name", label: "Name", type: "text" },
  { key: "description", label: "Description", type: "textarea" },
  { key: "stack", label: "Stack", type: "taglist" },
  { key: "key_metrics", label: "Key metrics", type: "taglist" },
  { key: "visibility", label: "Visibility", type: "select", options: ["private", "public", "deployed"], default: "private" },
  { key: "weight", label: "Weight", type: "select", options: ["high", "medium", "low"], default: "medium" },
];

const OSS_FIELDS = [
  { key: "repo", label: "Repo", type: "text" },
  { key: "type", label: "Type", type: "select", options: ["docs", "bug_fix", "feature"], default: "feature" },
  { key: "merged", label: "Merged", type: "checkbox" },
  { key: "description", label: "Description", type: "textarea" },
  { key: "relevance", label: "Relevance", type: "textarea" },
];

function section(label, bodyHtml) {
  return `<div class="detail-section"><div class="card-label">${esc(label)}</div>${bodyHtml}</div>`;
}

function experienceBlock(snapshot) {
  if (!snapshot.experience?.length) return "";
  const rows = snapshot.experience
    .map(
      (e) =>
        `<div class="verdict-line"><span class="badge">${esc(e.weight)}</span> ${esc(e.title)} @ ${esc(e.company)} (${esc(e.date_range)})${e.signals?.length ? " -- " + esc(e.signals.join(", ")) : ""}</div>`
    )
    .join("");
  return section("Experience", rows);
}

function projectsBlock(snapshot) {
  if (!snapshot.projects?.length) return "";
  const rows = snapshot.projects
    .map(
      (p) =>
        `<div class="verdict-line"><span class="badge">${esc(p.weight)}</span> ${esc(p.name)} (${esc(p.visibility)}): ${esc(p.description)}${p.stack?.length ? " -- stack: " + esc(p.stack.join(", ")) : ""}</div>`
    )
    .join("");
  return section("Projects", rows);
}

function skillsBlock(snapshot) {
  const skills = snapshot.skills ?? {};
  const tiers = [
    ["Core (full weight)", skills.core],
    ["Developing (partial weight)", skills.developing],
    ["Coursework (low weight)", skills.coursework],
  ];
  const rows = tiers
    .filter(([, items]) => items?.length)
    .map(([label, items]) => `<div class="verdict-line"><span class="stage">${label}</span>${esc(items.join(", "))}</div>`)
    .join("");
  return rows ? section("Skills", rows) : "";
}

function credentialsBlock(snapshot) {
  const creds = snapshot.credentials ?? {};
  const parts = [];
  if (creds.education?.length) {
    parts.push(`<div class="verdict-line"><span class="stage">Education</span>${esc(creds.education.join("; "))}</div>`);
  }
  if (creds.oss_contributions?.length) {
    parts.push(
      `<div class="verdict-line"><span class="stage">OSS</span>${esc(creds.oss_contributions.map((c) => c.repo).join(", "))}</div>`
    );
  }
  if (creds.publications?.length) {
    parts.push(`<div class="verdict-line"><span class="stage">Publications</span>${esc(creds.publications.join("; "))}</div>`);
  }
  return parts.length ? section("Credentials", parts.join("")) : "";
}

function criteriaBlock(snapshot) {
  const criteria = snapshot.criteria ?? {};
  const loc = criteria.location_prefs ?? {};
  const parts = [
    `<div class="verdict-line"><span class="stage">Seniority</span>${esc(criteria.seniority_floor ?? "-")} to ${esc(criteria.seniority_ceiling ?? "-")}</div>`,
  ];
  if (criteria.target_signals?.length) {
    parts.push(`<div class="verdict-line"><span class="stage">Target signals</span>${esc(criteria.target_signals.join(", "))}</div>`);
  }
  if (criteria.skip_patterns?.length) {
    parts.push(`<div class="verdict-line"><span class="stage">Skip patterns</span>${esc(criteria.skip_patterns.join(", "))}</div>`);
  }
  if (criteria.title_fast_track?.length) {
    parts.push(`<div class="verdict-line"><span class="stage">Fast-track titles</span>${esc(criteria.title_fast_track.join(", "))}</div>`);
  }
  if (criteria.title_skip?.length) {
    parts.push(`<div class="verdict-line"><span class="stage">Skip titles</span>${esc(criteria.title_skip.join(", "))}</div>`);
  }
  const locParts = [];
  if (loc.cities?.length) locParts.push(`cities: ${loc.cities.join(", ")}`);
  if (loc.countries?.length) locParts.push(`countries: ${loc.countries.join(", ")}`);
  if (loc.exclude_cities?.length) locParts.push(`exclude cities: ${loc.exclude_cities.join(", ")}`);
  if (loc.exclude_countries?.length) locParts.push(`exclude countries: ${loc.exclude_countries.join(", ")}`);
  locParts.push(`remote_ok: ${loc.remote_ok ?? true}`, `hybrid_ok: ${loc.hybrid_ok ?? true}`, `onsite_ok: ${loc.onsite_ok ?? true}`);
  parts.push(`<div class="verdict-line"><span class="stage">Location prefs</span>${esc(locParts.join(" -- "))}</div>`);
  return section("Fit criteria", parts.join(""));
}

async function loadCurrentProfile() {
  const el = document.getElementById("profileCurrent");
  el.innerHTML = "Loading...";
  const profile = await call("get_current_profile", {});
  currentSnapshot = profile.snapshot;
  el.innerHTML =
    `<div class="card-label">Current profile (version #${esc(profile.id)})</div>` +
    experienceBlock(currentSnapshot) +
    projectsBlock(currentSnapshot) +
    skillsBlock(currentSnapshot) +
    credentialsBlock(currentSnapshot) +
    criteriaBlock(currentSnapshot);
}

async function loadProfileVersions() {
  const list = document.getElementById("profileVersions");
  list.innerHTML = "";
  const versions = await call("list_profile_versions", {});
  if (versions.length === 0) {
    list.innerHTML = '<div class="empty-state">No profile versions yet.</div>';
    return;
  }
  versions.forEach((v) => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="subtitle">#${esc(v.id)} -- ${esc(v.created_at.slice(0, 10))}</div>
      ${v.changelog ? `<div class="verdict-line">${esc(v.changelog)}</div>` : ""}`;
    list.appendChild(card);
  });
}

export async function loadProfile() {
  profileLoaded = true;
  setStatus("Loading profile...");
  await Promise.all([loadCurrentProfile(), loadProfileVersions()]);
  setStatus("");
}

export function profileTabActivated() {
  if (!profileLoaded) loadProfile();
}

// ---- Editing ----

function mountEditForm(snapshot) {
  const skills = snapshot.skills ?? {};
  const creds = snapshot.credentials ?? {};
  const criteria = snapshot.criteria ?? {};
  const loc = criteria.location_prefs ?? {};

  mountRepeatableList("editExperience", EXPERIENCE_FIELDS, snapshot.experience);
  mountRepeatableList("editProjects", PROJECT_FIELDS, snapshot.projects);
  mountTagList("editSkillsCore", skills.core);
  mountTagList("editSkillsDeveloping", skills.developing);
  mountTagList("editSkillsCoursework", skills.coursework);
  mountTagList("editEducation", creds.education);
  mountRepeatableList("editOssContributions", OSS_FIELDS, creds.oss_contributions);
  mountTagList("editPublications", creds.publications);
  mountTagList("editSkipPatterns", criteria.skip_patterns);
  mountTagList("editTargetSignals", criteria.target_signals);
  mountTagList("editTitleFastTrack", criteria.title_fast_track);
  mountTagList("editTitleSkip", criteria.title_skip);
  document.getElementById("editSeniorityFloor").value = criteria.seniority_floor ?? "";
  document.getElementById("editSeniorityCeiling").value = criteria.seniority_ceiling ?? "";
  mountTagList("editLocCities", loc.cities);
  mountTagList("editLocCountries", loc.countries);
  mountTagList("editLocExcludeCities", loc.exclude_cities);
  mountTagList("editLocExcludeCountries", loc.exclude_countries);
  document.getElementById("editLocRemoteOk").checked = loc.remote_ok ?? true;
  document.getElementById("editLocHybridOk").checked = loc.hybrid_ok ?? true;
  document.getElementById("editLocOnsiteOk").checked = loc.onsite_ok ?? true;
  document.getElementById("editChangelog").value = "";
}

function readEditFormSnapshot() {
  return {
    experience: document.getElementById("editExperience").getValues(),
    projects: document.getElementById("editProjects").getValues(),
    skills: {
      core: document.getElementById("editSkillsCore").getValues(),
      developing: document.getElementById("editSkillsDeveloping").getValues(),
      coursework: document.getElementById("editSkillsCoursework").getValues(),
    },
    credentials: {
      education: document.getElementById("editEducation").getValues(),
      oss_contributions: document.getElementById("editOssContributions").getValues(),
      publications: document.getElementById("editPublications").getValues(),
    },
    criteria: {
      skip_patterns: document.getElementById("editSkipPatterns").getValues(),
      target_signals: document.getElementById("editTargetSignals").getValues(),
      title_fast_track: document.getElementById("editTitleFastTrack").getValues(),
      title_skip: document.getElementById("editTitleSkip").getValues(),
      seniority_floor: document.getElementById("editSeniorityFloor").value.trim() || null,
      seniority_ceiling: document.getElementById("editSeniorityCeiling").value.trim() || null,
      location_prefs: {
        cities: document.getElementById("editLocCities").getValues(),
        countries: document.getElementById("editLocCountries").getValues(),
        exclude_cities: document.getElementById("editLocExcludeCities").getValues(),
        exclude_countries: document.getElementById("editLocExcludeCountries").getValues(),
        remote_ok: document.getElementById("editLocRemoteOk").checked,
        hybrid_ok: document.getElementById("editLocHybridOk").checked,
        onsite_ok: document.getElementById("editLocOnsiteOk").checked,
      },
    },
  };
}

function enterEditMode(snapshot = currentSnapshot, changelog = "") {
  if (!snapshot) return;
  mountEditForm(snapshot);
  document.getElementById("editChangelog").value = changelog;
  document.getElementById("profileCurrent").hidden = true;
  document.getElementById("profileEditForm").hidden = false;
  document.getElementById("profileEditBtn").hidden = true;
  document.getElementById("profileChatPanel").hidden = true;
}

function exitEditMode() {
  document.getElementById("profileCurrent").hidden = false;
  document.getElementById("profileEditForm").hidden = true;
  document.getElementById("profileEditBtn").hidden = false;
}

export function initProfileView() {
  document.getElementById("profileEditBtn").addEventListener("click", () => enterEditMode());
  document.getElementById("profileCancelBtn").addEventListener("click", exitEditMode);
  document.getElementById("profileSaveBtn").addEventListener("click", async () => {
    const changelog = document.getElementById("editChangelog").value.trim();
    if (!changelog) {
      setStatus("A changelog is required.", "error");
      return;
    }
    const snapshot = readEditFormSnapshot();
    const saveBtn = document.getElementById("profileSaveBtn");
    saveBtn.disabled = true;
    // try/finally, not try/catch -- call() already surfaces a failure via
    // setStatus and re-throws; on error this just re-enables Save and
    // stays in edit mode (exitEditMode/loadProfile below never run) so
    // the user's in-progress edits aren't lost.
    try {
      await call("create_profile_version", { snapshot, changelog });
      setStatus("Profile saved.", "success");
      exitEditMode();
      profileLoaded = false;
      await loadProfile();
    } finally {
      saveBtn.disabled = false;
    }
  });
  initProfileChat();
}

// The conversation itself is persisted server-side (ProfileIntakeMessage)
// -- no client-side history array to maintain. A proposal turn's
// persisted form is a short synthesized summary text ("Proposed update:
// ..."), not the structured snapshot, so re-opening a past conversation
// shows what was proposed but not a "Review" button for it -- only the
// live turn that just returned a proposal gets one.
function appendChatBubble(role, text) {
  const container = document.getElementById("profileChatMessages");
  const bubble = document.createElement("div");
  bubble.className = `card chat-bubble chat-bubble-${role}`;
  bubble.textContent = text;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

function appendProposalBubble(response) {
  const container = document.getElementById("profileChatMessages");
  const card = document.createElement("div");
  card.className = "card chat-bubble chat-bubble-assistant";
  const questionsHtml = response.open_questions?.length
    ? `<div class="verdict-line"><span class="stage">Worth confirming</span>${esc(response.open_questions.join("; "))}</div>`
    : "";
  card.innerHTML = `<div class="card-label">Proposed update</div>
    <div class="verdict-line">${esc(response.changelog)}</div>
    ${questionsHtml}`;
  const reviewBtn = document.createElement("button");
  reviewBtn.className = "btn-secondary";
  reviewBtn.textContent = "Review this update";
  reviewBtn.addEventListener("click", () => enterEditMode(response.proposed_snapshot, response.changelog));
  card.appendChild(reviewBtn);
  container.appendChild(card);
  container.scrollTop = container.scrollHeight;
}

async function loadChatHistory() {
  const container = document.getElementById("profileChatMessages");
  container.innerHTML = "";
  const messages = await call("get_intake_chat_history", {});
  messages.forEach((m) => appendChatBubble(m.role, m.content));
}

function initProfileChat() {
  const toggleBtn = document.getElementById("profileChatBtn");
  const panel = document.getElementById("profileChatPanel");
  const input = document.getElementById("profileChatInput");
  const sendBtn = document.getElementById("profileChatSendBtn");
  const attachBtn = document.getElementById("profileChatAttachBtn");

  attachBtn.addEventListener("click", async () => {
    const { open } = window.__TAURI__.dialog;
    const srcPath = await open({
      multiple: false,
      filters: [{ name: "Resume", extensions: ["pdf", "docx"] }],
    });
    if (!srcPath) return; // user cancelled
    attachBtn.disabled = true;
    setStatus("Extracting text...");
    try {
      const { text } = await call("parse_resume_file", { srcPath });
      // Loads into the input for review/editing, never auto-sent --
      // PDF/DOCX extraction can come out messy (spacing, headers/footers,
      // column artifacts), so the user should look at it before it goes
      // to the model.
      input.value = input.value ? `${input.value}\n\n${text}` : text;
      setStatus("Resume text loaded -- review before sending.", "success");
    } finally {
      attachBtn.disabled = false;
    }
  });

  toggleBtn.addEventListener("click", async () => {
    panel.hidden = false;
    document.getElementById("profileCurrent").hidden = true;
    document.getElementById("profileEditForm").hidden = true;
    document.getElementById("profileEditBtn").hidden = true;
    await loadChatHistory();
  });
  document.getElementById("profileChatCloseBtn").addEventListener("click", () => {
    panel.hidden = true;
    document.getElementById("profileCurrent").hidden = false;
    document.getElementById("profileEditBtn").hidden = false;
  });
  document.getElementById("profileChatStartOverBtn").addEventListener("click", async () => {
    await call("clear_intake_chat_history", {});
    document.getElementById("profileChatMessages").innerHTML = "";
  });

  sendBtn.addEventListener("click", async () => {
    const userMessage = input.value.trim();
    if (!userMessage) return;
    appendChatBubble("user", userMessage);
    input.value = "";
    sendBtn.disabled = true;
    try {
      const response = await call("profile_intake_chat", { userMessage });
      if (response.proposed_snapshot) {
        appendProposalBubble(response);
      } else {
        appendChatBubble("assistant", response.assistant_message);
      }
    } finally {
      sendBtn.disabled = false;
    }
  });
}

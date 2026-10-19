const loadingView = document.getElementById("loadingView");
const captureView = document.getElementById("captureView");
const existingView = document.getElementById("existingView");

const companyInput = document.getElementById("company");
const titleInput = document.getElementById("title");
const locationInput = document.getElementById("location");
const captureBtn = document.getElementById("captureBtn");

const postingTitleEl = document.getElementById("postingTitle");
const postingCompanyEl = document.getElementById("postingCompany");
const postingMetaEl = document.getElementById("postingMeta");
const verdictInfoEl = document.getElementById("verdictInfo");
const starBtn = document.getElementById("starBtn");
const statusSelect = document.getElementById("statusSelect");
const updateStatusBtn = document.getElementById("updateStatusBtn");

const statusEl = document.getElementById("status");

let extracted = null; // { url, text } -- raw_jd_text isn't shown in the UI, just held here
let currentPosting = null; // PostingDetail once captured or found via LOOKUP

function setStatus(text, tone = "") {
  statusEl.textContent = text;
  statusEl.className = tone ? `tone-${tone}` : "";
}

function showView(view) {
  loadingView.hidden = view !== loadingView;
  captureView.hidden = view !== captureView;
  existingView.hidden = view !== existingView;
}

// Stage 3's own verdict string -> the tone class that colors its line.
const STAGE3_TONE = {
  "Strong Fit": "strong",
  "Moderate Fit": "moderate",
  "Weak Fit": "weak",
  Skip: "weak",
};

// Stage 1/2 share the same result vocabulary (fast_track/analyze/skip).
const STAGE12_TONE = {
  fast_track: "strong",
  skip: "weak",
};

function formatStageValue(result, reason) {
  return reason ? `${result} -- ${reason}` : result;
}

function addVerdictLine(stage, value, tone) {
  const line = document.createElement("div");
  line.className = tone ? `verdict-line tone-${tone}` : "verdict-line";
  const stageEl = document.createElement("div");
  stageEl.className = "stage";
  stageEl.textContent = stage;
  const valueEl = document.createElement("div");
  valueEl.className = "value";
  valueEl.textContent = value;
  line.append(stageEl, valueEl);
  verdictInfoEl.appendChild(line);
}

function renderVerdict(v) {
  verdictInfoEl.textContent = "";
  if (!v) {
    const empty = document.createElement("div");
    empty.className = "verdict-empty";
    empty.textContent = "No verdict yet.";
    verdictInfoEl.appendChild(empty);
    return;
  }
  addVerdictLine(
    "Stage 1",
    formatStageValue(v.stage1_result ?? "-", v.stage1_reason),
    STAGE12_TONE[v.stage1_result]
  );
  if (v.stage2_result) {
    addVerdictLine(
      "Stage 2",
      formatStageValue(v.stage2_result, v.stage2_reason),
      STAGE12_TONE[v.stage2_result]
    );
  }
  if (v.stage3_result) {
    addVerdictLine(
      "Stage 3",
      `${v.stage3_result.verdict} -- ${v.stage3_result.one_liner}`,
      STAGE3_TONE[v.stage3_result.verdict]
    );
  }
}

function formatComp(posting) {
  if (posting.comp_min == null && posting.comp_max == null) return null;
  const parts = [posting.comp_min, posting.comp_max].filter((v) => v != null);
  const range =
    parts.length === 2
      ? `${parts[0].toLocaleString()}-${parts[1].toLocaleString()}`
      : parts[0].toLocaleString();
  return `${posting.comp_currency ?? "$"} ${range}${posting.comp_period ? "/" + posting.comp_period : ""}`;
}

function renderPostingMeta(posting) {
  postingMetaEl.innerHTML = "";
  const badges = [];
  if (posting.locations?.length) badges.push(posting.locations[0]);
  if (posting.workplace_type) badges.push(posting.workplace_type);
  const comp = formatComp(posting);
  if (comp) badges.push(comp);
  badges.forEach((text) => {
    const span = document.createElement("span");
    span.className = "badge";
    span.textContent = text;
    postingMetaEl.appendChild(span);
  });
}

function renderExisting(posting, application) {
  currentPosting = posting;
  postingTitleEl.textContent = posting.title;
  postingCompanyEl.textContent = posting.company_name;
  renderPostingMeta(posting);
  renderVerdict(posting.latest_verdict);

  starBtn.textContent = posting.starred ? "★ Starred" : "☆ Star this role";
  starBtn.disabled = posting.starred;
  starBtn.classList.toggle("is-starred", posting.starred);

  statusSelect.value = application?.status ?? "";

  showView(existingView);
  setStatus("");
}

document.addEventListener("DOMContentLoaded", async () => {
  setStatus("Reading page...");
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const resp = await chrome.runtime.sendMessage({ type: "EXTRACT", tabId: tab.id });
  if (!resp.ok) {
    setStatus(`Error: ${resp.error}`, "error");
    return;
  }
  extracted = { url: resp.data.url, text: resp.data.text };

  const lookup = await chrome.runtime.sendMessage({ type: "LOOKUP", url: extracted.url });
  if (!lookup.ok) {
    setStatus(`Error: ${lookup.error}`, "error");
    return;
  }

  if (lookup.data) {
    const [detail, application] = await Promise.all([
      chrome.runtime.sendMessage({ type: "GET_DETAIL", postingId: lookup.data.id }),
      chrome.runtime.sendMessage({ type: "GET_APPLICATION", postingId: lookup.data.id }),
    ]);
    if (!detail.ok) {
      setStatus(`Error: ${detail.error}`, "error");
      return;
    }
    if (!application.ok) {
      setStatus(`Error: ${application.error}`, "error");
      return;
    }
    renderExisting(detail.data, application.data);
    return;
  }

  companyInput.value = resp.data.companyGuess;
  titleInput.value = resp.data.titleGuess;
  captureBtn.disabled = false;
  showView(captureView);
  setStatus("Review the fields, then capture.");
});

captureBtn.addEventListener("click", async () => {
  captureBtn.disabled = true;
  setStatus("Submitting...");
  const resp = await chrome.runtime.sendMessage({
    type: "SUBMIT",
    payload: {
      company_name: companyInput.value,
      title: titleInput.value,
      raw_jd_text: extracted.text,
      url: extracted.url,
      location: locationInput.value || null,
    },
  });
  if (!resp.ok) {
    setStatus(`Error: ${resp.error}`, "error");
    captureBtn.disabled = false;
    return;
  }
  renderExisting(resp.data.posting, null);
});

starBtn.addEventListener("click", async () => {
  starBtn.disabled = true;
  setStatus("Starring...");
  const resp = await chrome.runtime.sendMessage({ type: "STAR", postingId: currentPosting.id });
  if (!resp.ok) {
    setStatus(`Error: ${resp.error}`, "error");
    starBtn.disabled = false;
    return;
  }
  currentPosting.starred = resp.data.starred;
  starBtn.textContent = "★ Starred";
  starBtn.classList.add("is-starred");
  setStatus("");
});

updateStatusBtn.addEventListener("click", async () => {
  if (!statusSelect.value) {
    setStatus("Pick a status first.", "error");
    return;
  }
  updateStatusBtn.disabled = true;
  setStatus("Updating...");
  const resp = await chrome.runtime.sendMessage({
    type: "SET_STATUS",
    postingId: currentPosting.id,
    status: statusSelect.value,
  });
  updateStatusBtn.disabled = false;
  if (!resp.ok) {
    setStatus(`Error: ${resp.error}`, "error");
    return;
  }
  setStatus(`Status set to "${resp.data.status}".`, "success");
});

// The Postings tab: filter bar, master list, and pending-batches strip.
// Selection (which posting is shown in the detail pane) lives here since
// it's this view's own list that owns "which row is active."
import { call, setStatus } from "../api.js";
import { badgeRow, postingIdChipsRow, mountMultiSelectDropdown } from "../components.js";
import { renderDetail, clearDetail } from "./detail.js";

const FILTER_STORAGE_KEY = "postingsFilters";

let postingsLoaded = false;
let currentPostingIds = [];
let selectedPostingId = null;

function readFiltersFromForm() {
  const multiValues = (id) => document.getElementById(id).getValues();
  return {
    company: document.getElementById("filterCompany").value.trim() || null,
    keyword: document.getElementById("filterKeyword").value.trim() || null,
    keywordScope: document.getElementById("filterKeywordScope").value || "title",
    starred: document.getElementById("filterStarred").checked ? true : null,
    postedAfter: document.getElementById("filterPostedAfter").value || null,
    firstSeenAfter: document.getElementById("filterFirstSeenAfter").value || null,
    location: document.getElementById("filterLocation").value.trim() || null,
    workplaceType: document.getElementById("filterWorkplaceType").value || null,
    locationMatch: document.getElementById("filterLocationMatch").checked ? true : null,
    compMin: document.getElementById("filterCompMin").value ? Number(document.getElementById("filterCompMin").value) : null,
    hasComp: document.getElementById("filterHasComp").checked ? true : null,
    openOnly: document.getElementById("filterOpenOnly").checked,
    includeDismissed: document.getElementById("filterIncludeDismissed").checked,
    stage3Verdict: multiValues("filterStage3Verdict"),
    stage2Result: multiValues("filterStage2Result"),
    // mineOnly overrides the source dropdown's own selection when checked
    // (see the "Mine only" toggle's change listener below) -- kept as a
    // separate field, not folded into `source` here, so restoring saved
    // filters can reproduce both the checkbox and the dropdown's actual
    // selection independently.
    source: document.getElementById("filterSource").value || null,
    mineOnly: document.getElementById("filterMineOnly").checked,
    department: document.getElementById("filterDepartment").value.trim() || null,
    sort: document.getElementById("filterSort").value || "recent",
  };
}

function applyFiltersToForm(saved) {
  if (!saved) return;
  const set = (id, value) => {
    const el = document.getElementById(id);
    if (el && value != null) el.value = value;
  };
  set("filterCompany", saved.company);
  set("filterKeyword", saved.keyword);
  set("filterKeywordScope", saved.keywordScope);
  document.getElementById("filterStarred").checked = !!saved.starred;
  set("filterPostedAfter", saved.postedAfter);
  set("filterFirstSeenAfter", saved.firstSeenAfter);
  set("filterLocation", saved.location);
  set("filterWorkplaceType", saved.workplaceType);
  document.getElementById("filterLocationMatch").checked = !!saved.locationMatch;
  set("filterCompMin", saved.compMin);
  document.getElementById("filterHasComp").checked = !!saved.hasComp;
  document.getElementById("filterOpenOnly").checked = saved.openOnly !== false;
  document.getElementById("filterIncludeDismissed").checked = !!saved.includeDismissed;
  set("filterSource", saved.source);
  document.getElementById("filterMineOnly").checked = !!saved.mineOnly;
  applyMineOnlyState(!!saved.mineOnly);
  set("filterDepartment", saved.department);
  set("filterSort", saved.sort);
  document.getElementById("filterStage3Verdict").setValues(saved.stage3Verdict);
  document.getElementById("filterStage2Result").setValues(saved.stage2Result);
}

// "Mine only" is a quick shortcut for source=manual_capture (self-imported
// via the extension), distinct from starred (a priority signal that
// applies regardless of source) -- disables the source dropdown while
// checked so the two controls can't disagree silently.
function applyMineOnlyState(checked) {
  const sourceSelect = document.getElementById("filterSource");
  sourceSelect.disabled = checked;
  if (checked) sourceSelect.value = "manual_capture";
}

function loadSavedFilters() {
  try {
    const raw = localStorage.getItem(FILTER_STORAGE_KEY);
    if (raw) applyFiltersToForm(JSON.parse(raw));
  } catch {
    // Private window, blocked storage, or corrupt JSON -- filters just
    // start blank, which is a safe fallback.
  }
}

function saveFilters(filters) {
  try {
    localStorage.setItem(FILTER_STORAGE_KEY, JSON.stringify(filters));
  } catch {
    // Nothing to do -- per-viewer convenience only, not load-bearing.
  }
}

function postingCard(posting) {
  const card = document.createElement("div");
  card.className = posting.id === selectedPostingId ? "card posting-row is-selected" : "card posting-row";

  const main = document.createElement("div");
  main.className = "posting-main";
  const h3 = document.createElement("h3");
  h3.textContent = posting.title;
  const subtitle = document.createElement("div");
  subtitle.className = "subtitle";
  const repostNote = posting.repost_count > 0 ? ` -- reopened ${posting.repost_count}x` : "";
  const locationNote = posting.locations?.length ? ` -- ${posting.locations[0]}` : "";
  subtitle.textContent = `${posting.company_name}${locationNote}${repostNote}`;
  main.append(h3, subtitle, badgeRow(posting));
  if (posting.starred) {
    const star = document.createElement("span");
    star.className = "list-star-marker";
    star.textContent = "★";
    main.appendChild(star);
  }
  card.appendChild(main);
  card.addEventListener("click", () => selectPosting(posting.id));
  return card;
}

let lastItems = [];

function rerenderList() {
  const list = document.getElementById("postingsList");
  list.innerHTML = "";
  lastItems.forEach((p) => list.appendChild(postingCard(p)));
}

// Keeps the list card in sync with whatever the detail pane just showed --
// star/dismiss/hydrate all flow through here since renderDetail calls this
// on every render, not just after a mutation. No-ops if the posting isn't
// in the currently-loaded list (e.g. a similar-role result from outside
// the current filtered set).
function patchPosting(updated) {
  const idx = lastItems.findIndex((p) => p.id === updated.id);
  if (idx === -1) return;
  lastItems[idx] = { ...lastItems[idx], ...updated };
  rerenderList();
}

async function selectPosting(id) {
  selectedPostingId = id;
  rerenderList();
  await renderDetail(id, { onSelectPosting: selectPosting, onPostingUpdated: patchPosting });
}

function resetFunnelPreviews() {
  // A preview's count/cost is only valid for the filter set it was
  // computed against -- hide stale ones rather than let a re-filter leave
  // an outdated "would submit N" and enabled Submit button on screen.
  ["stage2PreviewResult", "stage3PreviewResult"].forEach((id) => {
    document.getElementById(id).hidden = true;
  });
  ["submitStage2Btn", "submitStage3Btn"].forEach((id) => {
    document.getElementById(id).hidden = true;
  });
}

export async function loadPostings() {
  postingsLoaded = true;
  const list = document.getElementById("postingsList");
  const countEl = document.getElementById("postingsCount");
  list.innerHTML = "";
  countEl.textContent = "";
  setStatus("Loading postings...");
  resetFunnelPreviews();

  const filters = readFiltersFromForm();
  saveFilters(filters);

  const { items, total } = await call("list_postings", {
    company: filters.company,
    starred: filters.starred,
    urlFilter: null,
    postedAfter: filters.postedAfter,
    firstSeenAfter: filters.firstSeenAfter,
    keyword: filters.keyword,
    keywordScope: filters.keywordScope,
    location: filters.location,
    workplaceType: filters.workplaceType,
    locationMatch: filters.locationMatch,
    compMin: filters.compMin,
    hasComp: filters.hasComp,
    openOnly: filters.openOnly,
    includeDismissed: filters.includeDismissed,
    stage3Verdict: filters.stage3Verdict.length ? filters.stage3Verdict : null,
    stage2Result: filters.stage2Result.length ? filters.stage2Result : null,
    source: filters.source,
    department: filters.department,
    sort: filters.sort,
    limit: 200,
  });
  setStatus("");
  currentPostingIds = items.map((p) => p.id);
  lastItems = items;

  countEl.textContent =
    total > items.length ? `Showing ${items.length} of ${total} (narrow the filters to see more)` : `${total} posting(s)`;

  if (items.length === 0) {
    list.innerHTML = '<div class="empty-state">No postings match these filters.</div>';
    if (selectedPostingId !== null) {
      selectedPostingId = null;
      clearDetail();
    }
    return;
  }

  // Selection persists across a re-filter only when the selected posting
  // is still in the new result set -- otherwise the detail pane clears
  // rather than silently showing a stale posting that's now filtered out.
  if (selectedPostingId !== null && !currentPostingIds.includes(selectedPostingId)) {
    selectedPostingId = null;
    clearDetail();
  }

  rerenderList();
}

// ---- Batches ----
// Tracked entirely in the backend (batch_submissions table) so a batch
// submitted via curl/MCP, or before an app restart, still shows up here.

const KIND_LABEL = {
  stage2_triage: "Stage 2 triage",
  stage3_analysis: "Stage 3 analysis",
  key_skills_backfill: "Key-skills backfill",
  key_skills_refresh: "Key-skills refresh",
};

// The backend routes a key_skills_backfill/refresh batch by its recorded kind
// through the same endpoint as Stage 2 triage.
function applyBatch(batch) {
  const command = batch.kind === "stage3_analysis" ? "apply_stage3_batch" : "apply_stage2_batch";
  return call(command, { batchId: batch.batch_id });
}

// Once per app session, check every batch whose results haven't been applied
// yet. Applying is idempotent and a still-running batch just reports its
// status, so there's no reason to make the user click for each one.
let autoCheckedBatches = false;

async function autoCheckPendingBatches(batchList) {
  autoCheckedBatches = true;
  const unapplied = batchList.filter((b) => !b.applied_at);
  let appliedTotal = 0;
  for (const batch of unapplied) {
    const result = await applyBatch(batch);
    if (result.status === "ended") {
      appliedTotal += result.applied;
      await offerStage3ForSurvivors(batch, result);
    }
  }
  if (appliedTotal > 0) setStatus(`Applied ${appliedTotal} result(s) from finished batches.`, "success");
  if (unapplied.length > 0) loadPendingBatches();
}

// After a Stage 2 batch is applied: of the postings it just triaged, how many
// are now ready for Stage 3 (Stage 3 only takes Stage 1 fast-tracks and Stage 2
// passes), and what would that cost? One click submits exactly those.
async function offerStage3ForSurvivors(batch, result) {
  const el = document.getElementById("survivorsPrompt");
  if (batch.kind !== "stage2_triage" || result.applied === 0 || result.posting_ids.length === 0) return;
  const preview = await call("analyze_pending_preview", {
    postingIds: result.posting_ids,
    limit: 200,
    includeStale: false,
  });
  el.innerHTML = "";
  const card = document.createElement("div");
  card.className = "card";
  if (preview.eligible === 0) {
    card.textContent = `Stage 2 passed none of these ${result.applied} posting(s) on to Stage 3 -- nothing to analyze.`;
    el.appendChild(card);
    return;
  }
  const cost = preview.available ? ` for ${formatCost(preview.estimated_cost_usd)} (estimate)` : "";
  const text = document.createElement("div");
  text.textContent = `Stage 2 passed ${preview.eligible} of these ${result.applied} posting(s). Analyze them with Stage 3${cost}?`;
  const btn = document.createElement("button");
  btn.className = "btn-primary";
  btn.textContent = `Submit Stage 3 for ${preview.eligible}`;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    const submitted = await call("submit_stage3_batch", {
      postingIds: result.posting_ids,
      limit: 200,
      includeStale: false,
    });
    el.innerHTML = "";
    setStatus(`Submitted Stage 3 batch ${submitted.batch_id} for ${submitted.submitted} posting(s).`, "success");
    loadPendingBatches();
    loadSpendSummary();
  });
  card.append(text, btn);
  el.appendChild(card);
}

export async function loadPendingBatches() {
  const el = document.getElementById("pendingBatches");
  const batchList = await call("list_batches", { kind: null });
  el.innerHTML = "";
  if (batchList.length === 0) return;
  if (!autoCheckedBatches) autoCheckPendingBatches(batchList);

  batchList.forEach((batch) => {
    const card = document.createElement("div");
    card.className = "card";
    const info = document.createElement("div");
    const label = KIND_LABEL[batch.kind] ?? batch.kind;
    const title = document.createElement("h3");
    title.className = "batch-title";
    title.textContent = `${label} batch ${batch.batch_id}`;
    const subtitle = document.createElement("div");
    subtitle.className = "subtitle";
    subtitle.textContent = `submitted ${batch.submitted_count} posting(s)${batch.capped ? " (capped)" : ""} -- last status: ${batch.last_status ?? "not checked yet"}`;
    info.append(title, subtitle);
    const applyBtn = document.createElement("button");
    applyBtn.className = "btn-secondary batch-apply-btn";
    applyBtn.textContent = batch.applied_at ? "Results applied -- re-check" : "Check / apply results";
    applyBtn.addEventListener("click", async () => {
      applyBtn.disabled = true;
      const result = await applyBatch(batch);
      applyBtn.disabled = false;
      if (result.status !== "ended") {
        setStatus(`Batch ${batch.batch_id} still ${result.status} -- check back later.`);
        document.getElementById("lastAppliedResults").innerHTML = "";
      } else {
        setStatus(
          `Batch ${batch.batch_id} done: ${result.applied} applied, ${result.errors} errors, ${result.skipped_stale} skipped (stale).`,
          "success"
        );
        renderLastAppliedResults(batch.batch_id, result.posting_ids);
        await offerStage3ForSurvivors(batch, result);
      }
      loadPendingBatches();
    });
    info.appendChild(applyBtn);
    card.appendChild(info);
    el.appendChild(card);
  });
}

function renderLastAppliedResults(batchId, postingIds) {
  const el = document.getElementById("lastAppliedResults");
  el.innerHTML = "";
  if (!postingIds || postingIds.length === 0) return;
  const card = document.createElement("div");
  card.className = "card";
  const label = document.createElement("div");
  label.className = "card-label";
  label.textContent = `Results for batch ${batchId}`;
  card.appendChild(label);
  card.appendChild(postingIdChipsRow(postingIds));
  el.appendChild(card);
}

// ---- Spend summary + funnel (3.7) ----
// Every dollar figure here is explicitly an estimate (see
// app.services.pricing) -- never presented as a real bill line.

function formatCost(usd) {
  return `~$${usd.toFixed(4)}`;
}

async function loadSpendSummary() {
  const el = document.getElementById("spendSummary");
  const spend = await call("get_spend_summary", {});
  el.innerHTML = `<div class="card-label">Spend so far (estimated)</div>
    <div class="subtitle">Stage 2 triage: ${formatCost(spend.stage2.estimated_cost_usd_last_30d)} last 30d / ${formatCost(spend.stage2.estimated_cost_usd_all_time)} all time (${spend.stage2.tokens_last_30d.toLocaleString()} tokens / 30d)</div>
    <div class="subtitle">Stage 3 analysis: ${formatCost(spend.stage3.estimated_cost_usd_last_30d)} last 30d / ${formatCost(spend.stage3.estimated_cost_usd_all_time)} all time (${spend.stage3.tokens_last_30d.toLocaleString()} tokens / 30d)</div>`;
}

function renderPreview(el, preview) {
  el.hidden = false;
  const parts = [`${preview.eligible} eligible`, `would submit ${preview.would_submit}${preview.capped ? " (capped)" : ""}`];
  if (preview.excluded_unhydrated > 0) {
    parts.push(`${preview.excluded_unhydrated} excluded (unhydrated)`);
  }
  if (preview.in_pending_batches > 0) {
    parts.push(`${preview.in_pending_batches} already in a pending batch (apply it first)`);
  }
  if (preview.available) {
    parts.push(`estimated cost ${formatCost(preview.estimated_cost_usd)} (avg ${preview.avg_tokens_per_posting} tokens/posting)`);
  } else {
    parts.push("no cost history yet for this stage");
  }
  el.textContent = parts.join(" -- ");
}

function setupFunnelStep({
  previewBtnId,
  resultId,
  submitBtnId,
  previewCommand,
  submitCommand,
  kindLabel,
  extraArgs = {},
}) {
  const previewBtn = document.getElementById(previewBtnId);
  const resultEl = document.getElementById(resultId);
  const submitBtn = document.getElementById(submitBtnId);

  previewBtn.addEventListener("click", async () => {
    if (currentPostingIds.length === 0) {
      setStatus("No postings currently loaded to batch.", "error");
      return;
    }
    previewBtn.disabled = true;
    const preview = await call(previewCommand, { postingIds: currentPostingIds, limit: 30, ...extraArgs });
    previewBtn.disabled = false;
    renderPreview(resultEl, preview);
    submitBtn.hidden = preview.would_submit === 0;
  });

  submitBtn.addEventListener("click", async () => {
    submitBtn.disabled = true;
    setStatus(`Submitting ${kindLabel} batch for ${currentPostingIds.length} posting(s)...`);
    const result = await call(submitCommand, { postingIds: currentPostingIds, limit: 30, ...extraArgs });
    submitBtn.disabled = false;
    submitBtn.hidden = true;
    resultEl.hidden = true;
    if (!result.batch_id) {
      setStatus(`Nothing pending ${kindLabel} in this filtered set.`, "success");
      return;
    }
    setStatus(
      `Submitted batch ${result.batch_id}: ${result.submitted} posting(s)${result.capped ? " (capped at 30, relevance-ranked)" : ""}. Batches can take up to ~an hour -- check back with "Check / apply results".`,
      "success"
    );
    loadPendingBatches();
    loadSpendSummary();
  });
}

export function initPostingsView() {
  mountMultiSelectDropdown("filterStage3Verdict", "Stage 3", ["Strong Fit", "Moderate Fit", "Weak Fit", "Skip"]);
  mountMultiSelectDropdown("filterStage2Result", "Stage 2", ["fast_track", "analyze", "skip"]);
  loadSavedFilters();
  clearDetail();
  document.getElementById("filterMineOnly").addEventListener("change", (e) => {
    applyMineOnlyState(e.target.checked);
  });
  document.getElementById("applyFilters").addEventListener("click", () => {
    postingsLoaded = false;
    loadPostings();
  });
  setupFunnelStep({
    previewBtnId: "previewStage2Btn",
    resultId: "stage2PreviewResult",
    submitBtnId: "submitStage2Btn",
    previewCommand: "triage_pending_preview",
    submitCommand: "submit_stage2_batch",
    kindLabel: "Stage 2 triage",
  });
  setupFunnelStep({
    previewBtnId: "previewStage3Btn",
    resultId: "stage3PreviewResult",
    submitBtnId: "submitStage3Btn",
    previewCommand: "analyze_pending_preview",
    submitCommand: "submit_stage3_batch",
    kindLabel: "Stage 3 analysis",
    extraArgs: { includeStale: false },
  });
  loadSpendSummary();
}

export function postingsTabActivated() {
  if (!postingsLoaded) loadPostings();
  loadPendingBatches();
  loadSpendSummary();
}

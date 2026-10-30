import { call, setStatus } from "../api.js";
import { postingIdChipsRow } from "../components.js";
import { html, render, nothing } from "../vendor/lit-html.js";

let readinessLoaded = false;

function trendCard(trend, evidenceLabel = "starred, actively applied to, or a Strong/Moderate fit") {
  const card = document.createElement("div");
  card.className = "card";
  const main = document.createElement("div");
  render(
    html`<h3 class="trend-title">${trend.skill_or_language}</h3>
      <div class="subtitle">seen in ${trend.occurrence_count} posting(s) (${evidenceLabel}) -- click a posting to see its details:</div>`,
    main
  );
  main.appendChild(postingIdChipsRow(trend.posting_ids));
  const briefBtn = document.createElement("button");
  briefBtn.className = "btn-secondary trend-brief-btn";
  briefBtn.textContent = "Generate brief (costs a Sonnet call)";
  briefBtn.addEventListener("click", async () => {
    briefBtn.disabled = true;
    setStatus(`Generating brief for ${trend.skill_or_language}...`);
    await call("generate_roadmap_brief", {
      skillOrLanguage: trend.skill_or_language,
      triggerType: "job_market_gap",
      trendEvidence: trend.posting_ids,
    });
    setStatus(`Brief generated for ${trend.skill_or_language}.`, "success");
    loadRoadmap();
  });
  main.appendChild(briefBtn);
  card.appendChild(main);
  return card;
}

// 4.4: after adding the completed item's skill to the profile, offer to
// re-analyze exactly the postings that surfaced it -- a Stage 3 batch
// scoped to trend_evidence with include_stale=true (they're stale the
// moment the new profile version exists), same preview -> submit shape
// the Postings tab's funnel already uses, just built inline
// here rather than reusing that view's DOM directly.
async function offerReanalysis(item, container) {
  const evidencePostingIds = item.trend_evidence || [];
  if (evidencePostingIds.length === 0) return;

  const box = document.createElement("div");
  box.className = "verdict-line roadmap-reanalyze-box";
  const previewBtn = document.createElement("button");
  previewBtn.className = "btn-secondary";
  previewBtn.textContent = `Preview re-analysis of ${evidencePostingIds.length} evidence posting(s)`;
  const resultEl = document.createElement("div");
  resultEl.className = "verdict-line";
  resultEl.hidden = true;
  const submitBtn = document.createElement("button");
  submitBtn.className = "btn-primary roadmap-reanalyze-submit-btn";
  submitBtn.textContent = "Submit Stage 3 re-analysis";
  submitBtn.hidden = true;

  previewBtn.addEventListener("click", async () => {
    previewBtn.disabled = true;
    const preview = await call("analyze_pending_preview", {
      postingIds: evidencePostingIds,
      limit: null,
      includeStale: true,
    });
    previewBtn.disabled = false;
    resultEl.hidden = false;
    const cost = preview.available
      ? `estimated cost ~$${preview.estimated_cost_usd.toFixed(4)}`
      : "no cost history yet";
    resultEl.textContent = `${preview.eligible} eligible, would submit ${preview.would_submit} -- ${cost}`;
    submitBtn.hidden = preview.would_submit === 0;
  });
  submitBtn.addEventListener("click", async () => {
    submitBtn.disabled = true;
    setStatus(`Submitting Stage 3 re-analysis for ${evidencePostingIds.length} posting(s)...`);
    const result = await call("submit_stage3_batch", {
      postingIds: evidencePostingIds,
      limit: null,
      includeStale: true,
    });
    submitBtn.disabled = false;
    submitBtn.hidden = true;
    setStatus(
      result.batch_id
        ? `Submitted re-analysis batch ${result.batch_id}.`
        : "Nothing eligible to re-analyze.",
      "success"
    );
  });

  box.append(previewBtn, resultEl, submitBtn);
  container.appendChild(box);
}

// Adds skill_or_language to the current profile's skills.<tier> (if not
// already present) as a new, append-only profile version -- this is what
// makes the evidence postings' Stage 3 results stale (0.1's computed
// staleness), which offerReanalysis's include_stale=true then targets.
async function addSkillToProfile(item, tier) {
  const profile = await call("get_current_profile", {});
  const snapshot = profile.snapshot;
  snapshot.skills = snapshot.skills || { core: [], developing: [], coursework: [] };
  const list = snapshot.skills[tier] || [];
  if (!list.includes(item.skill_or_language)) {
    list.push(item.skill_or_language);
  }
  snapshot.skills[tier] = list;
  await call("create_profile_version", {
    snapshot,
    changelog: `Completed roadmap item: ${item.skill_or_language} -- added to skills.${tier}`,
  });
}

function showCompletionPrompt(item, container, onDone) {
  const prompt = document.createElement("div");
  prompt.className = "verdict-line roadmap-completion-prompt";
  const label = document.createElement("div");
  label.textContent = `Add "${item.skill_or_language}" to your profile as developing or core?`;
  const actions = document.createElement("div");
  actions.className = "row";

  function makeChoiceBtn(text, tier) {
    const btn = document.createElement("button");
    btn.className = "btn-secondary";
    btn.textContent = text;
    btn.addEventListener("click", async () => {
      actions.querySelectorAll("button").forEach((b) => (b.disabled = true));
      if (tier) {
        await addSkillToProfile(item, tier);
        setStatus(`Added "${item.skill_or_language}" to skills.${tier}.`, "success");
      }
      prompt.remove();
      await onDone();
      if (tier) await offerReanalysis(item, container);
    });
    return btn;
  }

  actions.append(
    makeChoiceBtn("As developing", "developing"),
    makeChoiceBtn("As core", "core"),
    makeChoiceBtn("Skip", null)
  );
  prompt.append(label, actions);
  container.appendChild(prompt);
}

function roadmapItemCard(item) {
  const card = document.createElement("div");
  card.className = "card";
  const main = document.createElement("div");
  render(
    html`<h3 class="trend-title">${item.skill_or_language}</h3>
      <div class="subtitle">status: ${item.status}</div>
      ${item.what_it_is ? html`<div class="verdict-line">${item.what_it_is}</div>` : nothing}
      ${item.why_it_matters ? html`<div class="verdict-line">${item.why_it_matters}</div>` : nothing}`,
    main
  );
  const row = document.createElement("div");
  row.className = "row roadmap-status-row";
  const select = document.createElement("select");
  select.innerHTML = ["flagged", "researching", "in_progress", "complete"]
    .map((s) => `<option value="${s}">${s}</option>`)
    .join("");
  select.value = item.status;
  const updateBtn = document.createElement("button");
  updateBtn.className = "btn-secondary";
  updateBtn.textContent = "Update status";

  const saveStatus = async () => {
    updateBtn.disabled = true;
    await call("update_roadmap_status", { roadmapId: item.id, status: select.value });
    updateBtn.disabled = false;
    setStatus(`"${item.skill_or_language}" set to ${select.value}.`, "success");
  };

  updateBtn.addEventListener("click", async () => {
    if (select.value === "complete" && item.status !== "complete") {
      showCompletionPrompt(item, main, saveStatus);
      return;
    }
    await saveStatus();
  });
  row.append(select, updateBtn);
  main.appendChild(row);
  card.appendChild(main);
  return card;
}

function readinessEntryCard(entry) {
  const card = document.createElement("details");
  card.className = "card readiness-entry";
  const gaps = entry.hard_gaps.length
    ? html`<div class="verdict-line"><span class="stage">Hard gaps</span>${entry.hard_gaps.join(", ")}</div>`
    : nothing;
  const match = entry.roadmap_item_id
    ? html`<div class="verdict-line"><span class="stage">Maps to</span>roadmap item #${entry.roadmap_item_id} -- "${entry.matched_gap}"</div>`
    : nothing;
  render(
    html`<summary>
        <span class="trend-title">${entry.title}</span>
        <span class="subtitle">${entry.company_name} -- ${entry.verdict}</span>
      </summary>
      <div class="verdict-line">${entry.one_liner}</div>
      ${gaps}${match}`,
    card
  );
  return card;
}

function renderReadinessSection(elId, entries) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  const label = el.previousElementSibling;
  label.dataset.base ??= label.textContent;
  label.textContent = `${label.dataset.base} (${entries.length})`;
  if (entries.length === 0) {
    el.innerHTML = '<div class="empty-state">Nothing here yet.</div>';
    return;
  }
  entries.forEach((e) => el.appendChild(readinessEntryCard(e)));
}

// Interest-sorted, so the first screenful is the skills that matter most.
const DEMAND_VISIBLE = 12;
const DEMAND_TONE = { have: "strong", developing: "moderate", gap: "weak" };

function demandClusterCard(cluster) {
  const card = document.createElement("div");
  card.className = "card";
  const tone = DEMAND_TONE[cluster.status] ?? "";
  render(
    html`<h3 class="trend-title">${cluster.skill_or_language}</h3>
      <div class="subtitle">${cluster.interest_posting_count} of your picked/starred role(s), ${cluster.posting_count} overall
        <span class="badge${tone ? " tone-" + tone : ""}" title="${cluster.judged ? cluster.judgment_reason : "Similarity guess -- not checked against your profile yet"}">${cluster.status}${cluster.judged ? " \u2713" : ""}</span>
        <span class="subtitle">interest score ${cluster.interest_weight}</span></div>`,
    card
  );
  if (cluster.status === "gap") {
    const flagBtn = document.createElement("button");
    flagBtn.className = "btn-secondary trend-brief-btn";
    flagBtn.textContent = "Flag for roadmap (costs a Sonnet call)";
    flagBtn.addEventListener("click", async () => {
      flagBtn.disabled = true;
      setStatus(`Flagging ${cluster.skill_or_language} for the roadmap...`);
      await call("generate_roadmap_brief", {
        skillOrLanguage: cluster.skill_or_language,
        triggerType: "job_market_gap",
        trendEvidence: cluster.posting_ids,
      });
      setStatus(`Flagged ${cluster.skill_or_language}.`, "success");
      loadRoadmap();
    });
    card.appendChild(flagBtn);
  }
  return card;
}

export async function loadMarketDemand() {
  const el = document.getElementById("marketDemandList");
  el.innerHTML = "";
  const clusters = await call("get_market_demand", { postingIds: null, sort: "interest" });
  if (clusters.length === 0) {
    el.innerHTML =
      '<div class="empty-state">No key-skills data yet -- run Stage 2 triage batches from the Postings tab to start filling this in.</div>';
    return;
  }
  const visible = clusters.slice(0, DEMAND_VISIBLE);
  visible.forEach((c) => el.appendChild(demandClusterCard(c)));
  if (clusters.length > DEMAND_VISIBLE) {
    const row = document.createElement("div");
    row.className = "show-more-row";
    const btn = document.createElement("button");
    btn.className = "btn-secondary";
    btn.textContent = `Show all ${clusters.length} skills`;
    btn.addEventListener("click", () => {
      clusters.slice(DEMAND_VISIBLE).forEach((c) => el.insertBefore(demandClusterCard(c), row));
      row.remove();
    });
    row.appendChild(btn);
    el.appendChild(row);
  }
}

export async function loadReadiness() {
  const view = await call("get_readiness_view", {});
  renderReadinessSection("readinessApplyNow", view.apply_now);
  renderReadinessSection("readinessSolidMatch", view.solid_match);
  renderReadinessSection("readinessStretch", view.stretch);
  renderReadinessSection("readinessBuildToward", view.build_toward);
  await loadNearMissTrends(view);
}

// What's actually blocking Apply now -- same detect_skill_trends clustering
// the main Trends section uses, just scoped to readiness_view's Solid
// match + Stretch postings (the ones already close) instead of its
// default starred/applied/Strong-Moderate-fit pool. Free (no LLM call),
// so it loads automatically alongside the rest of Readiness rather than
// needing its own button. threshold=2 (lower than the main section's
// default 3) since this pool is much smaller.
async function loadNearMissTrends(view) {
  const container = document.getElementById("nearMissTrendsList");
  const postingIds = [...view.solid_match, ...view.stretch].map((e) => e.posting_id);
  if (postingIds.length === 0) {
    container.innerHTML = '<div class="empty-state">No Solid match/Stretch postings yet to find common gaps in.</div>';
    return;
  }
  const trends = await call("detect_skill_trends", { threshold: 2, postingIds });
  container.innerHTML = "";
  if (trends.length === 0) {
    container.innerHTML = '<div class="empty-state">No recurring gaps found across your Solid match/Stretch postings yet.</div>';
    return;
  }
  trends.forEach((t) => container.appendChild(trendCard(t, "among your Solid match/Stretch postings")));
}

export async function loadRoadmap() {
  const list = document.getElementById("roadmapList");
  list.innerHTML = "";
  const items = await call("list_roadmap", { status: null });
  if (items.length === 0) {
    list.innerHTML = '<div class="empty-state">No roadmap items yet.</div>';
    return;
  }
  items.forEach((i) => list.appendChild(roadmapItemCard(i)));
}

function initSkillJudgment() {
  const previewBtn = document.getElementById("judgeSkillsPreview");
  const runBtn = document.getElementById("judgeSkillsRun");
  const statusEl = document.getElementById("judgeSkillsStatus");
  previewBtn.addEventListener("click", async () => {
    statusEl.textContent = "Estimating...";
    const p = await call("preview_skill_judgments", {});
    if (p.to_judge === 0) {
      statusEl.textContent = `Nothing to check -- ${p.already_judged} skill(s) already judged against your current profile.`;
      runBtn.hidden = true;
      return;
    }
    statusEl.textContent = `${p.to_judge} skill(s) to check (${p.already_judged} already done) -- about ${p.estimated_input_tokens.toLocaleString()} input tokens, roughly $${p.estimated_cost_usd.toFixed(2)} (estimate).`;
    runBtn.hidden = false;
  });
  document.getElementById("reloadSkills").addEventListener("click", () => loadMarketDemand());
  document.getElementById("refreshSkillNames").addEventListener("click", async () => {
    if (!confirm("Re-run skill extraction for your picked, starred and applied-to roles with the current prompt? One Haiku batch, about 5 cents. Results are applied from the Postings tab once it finishes.")) return;
    const r = await call("submit_key_skills_refresh", {});
    setStatus(
      r.batch_id
        ? `Submitted batch ${r.batch_id} for ${r.submitted} role(s). Check it from the Postings tab ("Check / apply results"), then reload this list.`
        : "No roles to refresh.",
      "success",
    );
  });
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    previewBtn.disabled = true;
    setStatus("Checking skills against your profile (one Sonnet call per skill)...");
    try {
      const r = await call("judge_skills", {});
      setStatus(`Checked ${r.judged} skill(s); ${r.remaining} left.`, "success");
    } finally {
      runBtn.disabled = false;
      previewBtn.disabled = false;
      runBtn.hidden = true;
      statusEl.textContent = "";
    }
    loadMarketDemand();
  });
}

function initSubtabs() {
  const buttons = document.querySelectorAll(".subtab-btn");
  const panels = {
    readiness: document.getElementById("roadmapSubReadiness"),
    skills: document.getElementById("roadmapSubSkills"),
    plan: document.getElementById("roadmapSubPlan"),
  };
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.toggle("active", b === btn));
      Object.entries(panels).forEach(([name, el]) => (el.hidden = name !== btn.dataset.subtab));
    });
  });
}

export function initRoadmapView() {
  initSubtabs();
  initSkillJudgment();
  document.getElementById("detectTrends").addEventListener("click", async () => {
    const threshold = parseInt(document.getElementById("trendThreshold").value, 10) || 3;
    const trendsList = document.getElementById("trendsList");
    trendsList.innerHTML = "";
    setStatus("Detecting trends...");
    const trends = await call("detect_skill_trends", { threshold });
    setStatus("");
    if (trends.length === 0) {
      trendsList.innerHTML = '<div class="empty-state">No trends at this threshold.</div>';
    } else {
      trends.forEach((t) => trendsList.appendChild(trendCard(t)));
    }
    loadRoadmap();
  });
}

export function roadmapTabActivated() {
  if (!readinessLoaded) {
    readinessLoaded = true;
    loadReadiness();
    loadMarketDemand();
  }
}

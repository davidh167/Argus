// The right-hand pane of the Postings tab's master-detail layout (3.2).
import { call, setStatus, openExternal } from "../api.js";
import { renderVerdictLines, badgeRow } from "../components.js";
import { html, render } from "../vendor/lit-html.js";

const paneEl = document.getElementById("postingDetail");

export function clearDetail() {
  paneEl.innerHTML = '<div class="empty-state">Select a posting to see its details.</div>';
}

function section(label) {
  const wrap = document.createElement("div");
  wrap.className = "detail-section";
  const h4 = document.createElement("div");
  h4.className = "card-label";
  h4.textContent = label;
  wrap.appendChild(h4);
  return wrap;
}

function jdSectionBlock(label, text) {
  if (!text) return null;
  const wrap = document.createElement("div");
  wrap.className = "jd-section";
  const h5 = document.createElement("div");
  h5.className = "jd-section-label";
  h5.textContent = label;
  const body = document.createElement("div");
  body.className = "jd-section-body";
  body.textContent = text;
  wrap.append(h5, body);
  return wrap;
}

async function renderSimilar(id, onSelectPosting) {
  const wrap = section("Similar roles");
  const list = document.createElement("div");
  list.className = "list";
  list.textContent = "Loading...";
  wrap.appendChild(list);
  call("similar_postings", { id, limit: 10 }).then((similar) => {
    list.innerHTML = "";
    if (similar.length === 0) {
      list.innerHTML = '<div class="empty-state">No similar postings found.</div>';
      return;
    }
    similar.forEach(({ posting, similarity, possibly_same_role }) => {
      const row = document.createElement("div");
      row.className = "card similar-row";
      const pct = Math.round(similarity * 100);
      render(
        html`<div class="posting-main">
          <h3>${posting.title}</h3>
          <div class="subtitle">${posting.company_name} -- ${pct}% similar${possibly_same_role ? " -- possibly the same role" : ""}${posting.freshness === "closed" ? " -- closed" : ""}</div>
        </div>`,
        row
      );
      row.addEventListener("click", () => onSelectPosting(posting.id));
      list.appendChild(row);
    });
  });
  return wrap;
}

export async function renderDetail(id, { onSelectPosting, onPostingUpdated }) {
  paneEl.innerHTML = '<div class="empty-state">Loading...</div>';
  const [detail, application] = await Promise.all([
    call("get_posting", { id }),
    call("get_application", { postingId: id }),
  ]);
  paneEl.innerHTML = "";
  // Keeps the list card's badges/star marker in sync with whatever's
  // actually showing here -- every render re-syncs, not just star/dismiss,
  // since this is the one place that always has fresh data for this id.
  onPostingUpdated?.(detail);

  // -- Header --
  const header = document.createElement("div");
  const h2 = document.createElement("h2");
  h2.className = "detail-title";
  h2.textContent = detail.title;
  const sub = document.createElement("div");
  sub.className = "subtitle";
  sub.textContent = `${detail.company_name}${detail.department ? " -- " + detail.department : ""}`;
  header.append(h2, sub, badgeRow(detail));
  if (detail.locations?.length) {
    const loc = document.createElement("div");
    loc.className = "subtitle detail-locations";
    loc.textContent = detail.locations.join(", ");
    header.appendChild(loc);
  }
  const dateLine = document.createElement("div");
  dateLine.className = "subtitle";
  dateLine.textContent = `Posted ${detail.posted_date ?? "unknown"} -- first seen ${detail.first_seen_at.slice(0, 10)} -- last seen ${detail.last_seen_at.slice(0, 10)}${detail.closed_at ? " -- closed " + detail.closed_at.slice(0, 10) : ""}`;
  header.appendChild(dateLine);
  if (detail.url) {
    const applyBtn = document.createElement("button");
    applyBtn.className = "btn-primary detail-apply-btn";
    applyBtn.textContent = "Apply ↗";
    applyBtn.addEventListener("click", () => openExternal(detail.url));
    header.appendChild(applyBtn);
  }
  paneEl.appendChild(header);

  // -- Actions --
  const actions = document.createElement("div");
  actions.className = "row detail-actions";
  const starBtn = document.createElement("button");
  starBtn.className = detail.starred ? "btn-star is-starred" : "btn-star";
  starBtn.textContent = detail.starred ? "★ Starred" : "☆ Star";
  starBtn.addEventListener("click", async () => {
    starBtn.disabled = true;
    await call(detail.starred ? "unstar_posting" : "star_posting", { id });
    starBtn.disabled = false;
    renderDetail(id, { onSelectPosting, onPostingUpdated });
  });
  const dismissBtn = document.createElement("button");
  dismissBtn.className = "btn-secondary";
  dismissBtn.textContent = detail.dismissed ? "Undismiss" : "Dismiss";
  dismissBtn.addEventListener("click", async () => {
    dismissBtn.disabled = true;
    await call(detail.dismissed ? "undismiss_posting" : "dismiss_posting", { id });
    dismissBtn.disabled = false;
    renderDetail(id, { onSelectPosting, onPostingUpdated });
  });
  actions.append(starBtn, dismissBtn);
  paneEl.appendChild(actions);

  // -- Fit panel --
  const fitSection = section("Fit");
  const verdictBox = document.createElement("div");
  renderVerdictLines(verdictBox, detail.latest_verdict, { includeStage1: false });
  fitSection.appendChild(verdictBox);
  const analyzeBtn = document.createElement("button");
  analyzeBtn.className = "btn-secondary detail-analyze-btn";
  analyzeBtn.textContent = detail.latest_verdict?.stage3_result
    ? "Re-analyze for fit (Stage 3, costs a Sonnet call)"
    : "Analyze for fit (Stage 3, costs a Sonnet call)";
  analyzeBtn.addEventListener("click", async () => {
    analyzeBtn.disabled = true;
    setStatus(`Analyzing posting #${id}...`);
    const verdict = await call("analyze_posting", { id });
    analyzeBtn.disabled = false;
    renderVerdictLines(verdictBox, verdict, { includeStage1: false });
    setStatus(`Analysis complete for posting #${id}.`, "success");
  });
  fitSection.appendChild(analyzeBtn);
  paneEl.appendChild(fitSection);

  // -- JD by section --
  const jdSection = section("Job description");
  const sections = detail.jd_sections;
  let anySection = false;
  if (sections) {
    [
      ["Summary", sections.summary],
      ["Responsibilities", sections.responsibilities],
      ["Required", sections.required_qualifications],
      ["Preferred", sections.preferred_qualifications],
    ].forEach(([label, text]) => {
      const block = jdSectionBlock(label, text);
      if (block) {
        anySection = true;
        jdSection.appendChild(block);
      }
    });
  }
  if (!anySection && !detail.detail_fetched_at && detail.source === "scraped") {
    const fetchBtn = document.createElement("button");
    fetchBtn.className = "btn-secondary";
    fetchBtn.textContent = "Fetch full description";
    fetchBtn.addEventListener("click", async () => {
      fetchBtn.disabled = true;
      setStatus(`Fetching full description for posting #${id}...`);
      await call("hydrate_postings", { companyId: null, postingIds: [id], limit: 1 });
      setStatus("");
      renderDetail(id, { onSelectPosting, onPostingUpdated });
    });
    jdSection.appendChild(fetchBtn);
  } else if (!anySection) {
    jdSection.innerHTML += '<div class="empty-state">No structured sections found.</div>';
  }
  if (detail.raw_jd_text) {
    const jdToggle = document.createElement("button");
    jdToggle.className = "btn-secondary detail-raw-toggle";
    jdToggle.textContent = "Show raw JD text";
    const jdBox = document.createElement("div");
    jdBox.className = "verdict-line detail-raw-jd";
    jdBox.hidden = true;
    jdBox.textContent = detail.raw_jd_text;
    jdToggle.addEventListener("click", () => {
      jdBox.hidden = !jdBox.hidden;
      jdToggle.textContent = jdBox.hidden ? "Show raw JD text" : "Hide raw JD text";
    });
    jdSection.append(jdToggle, jdBox);
  }
  paneEl.appendChild(jdSection);

  // -- Application status + note --
  const appSection = section("Application");
  const row = document.createElement("div");
  row.className = "row";
  const select = document.createElement("select");
  select.innerHTML =
    `<option value="">-- not tracked --</option>` +
    ["saved", "applied", "screening", "interview", "rejected", "skipped"]
      .map((s) => `<option value="${s}">${s}</option>`)
      .join("");
  select.value = application?.status ?? "";
  const updateBtn = document.createElement("button");
  updateBtn.className = "btn-secondary";
  updateBtn.textContent = "Update status";
  updateBtn.addEventListener("click", async () => {
    if (!select.value) return;
    updateBtn.disabled = true;
    await call("set_application_status", {
      postingId: id,
      status: select.value,
      resumeUsed: null,
      dateApplied: null,
      followUpDate: null,
      notes: null,
    });
    updateBtn.disabled = false;
    setStatus(`Status set to "${select.value}".`, "success");
  });
  row.append(select, updateBtn);
  appSection.appendChild(row);

  const noteBox = document.createElement("textarea");
  noteBox.className = "note-box";
  noteBox.placeholder = "Note...";
  noteBox.value = detail.note ?? "";
  const noteBtn = document.createElement("button");
  noteBtn.className = "btn-secondary";
  noteBtn.textContent = "Save note";
  noteBtn.addEventListener("click", async () => {
    noteBtn.disabled = true;
    await call("set_posting_note", { id, note: noteBox.value || null });
    noteBtn.disabled = false;
    setStatus("Note saved.", "success");
  });
  appSection.append(noteBox, noteBtn);
  paneEl.appendChild(appSection);

  // -- Similar roles --
  paneEl.appendChild(await renderSimilar(id, onSelectPosting));
}

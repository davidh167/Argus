// Shared render helpers used across views/*.js -- badges, verdict lines,
// and the posting-id chip row (roadmap trends + batch results both need
// "click an id, see its detail inline").
import { call, openExternal } from "./api.js";
import { esc } from "./escape.js";

// Mounts a checklist-in-a-popover into an empty container element -- used
// for the filter bar's Stage 2/3 result filters, replacing a native
// <select multiple> (a fixed-height listbox, not actually a dropdown, and
// awkward to scan/toggle). The container itself becomes the widget's
// handle: postings.js reads/writes it via containerEl.getValues()/
// setValues(), the same getElementById-based access pattern every other
// filter control in this app already uses.
export function mountMultiSelectDropdown(containerId, label, options) {
  const container = document.getElementById(containerId);
  container.className = "dropdown-multiselect";
  container.innerHTML = "";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "btn-secondary dropdown-toggle";

  const panel = document.createElement("div");
  panel.className = "dropdown-panel";
  panel.hidden = true;

  const updateLabel = () => {
    const n = checkboxes.filter((c) => c.checked).length;
    toggle.textContent = n > 0 ? `${label} (${n})` : label;
  };

  const checkboxes = options.map((opt) => {
    const row = document.createElement("label");
    row.className = "dropdown-option";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = opt;
    box.addEventListener("change", updateLabel);
    row.append(box, document.createTextNode(opt));
    panel.appendChild(row);
    return box;
  });
  updateLabel();

  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    panel.hidden = !panel.hidden;
  });
  document.addEventListener("click", (e) => {
    if (!container.contains(e.target)) panel.hidden = true;
  });

  container.append(toggle, panel);
  container.getValues = () => checkboxes.filter((c) => c.checked).map((c) => c.value);
  container.setValues = (values) => {
    checkboxes.forEach((c) => (c.checked = (values ?? []).includes(c.value)));
    updateLabel();
  };
}

export const STAGE3_TONE = {
  "Strong Fit": "strong",
  "Moderate Fit": "moderate",
  "Weak Fit": "weak",
  Skip: "weak",
};

const FRESHNESS_LABEL = { fresh: "Fresh", aging: "Aging", stale: "Stale", closed: "Closed" };

export function freshnessBadge(freshness) {
  const span = document.createElement("span");
  const tone = freshness === "closed" ? "weak" : freshness === "fresh" ? "strong" : "";
  span.className = tone ? `badge tone-${tone}` : "badge";
  span.textContent = FRESHNESS_LABEL[freshness] ?? freshness;
  return span;
}

export function workplaceBadge(workplaceType) {
  if (!workplaceType) return null;
  const span = document.createElement("span");
  span.className = "badge";
  span.textContent = workplaceType;
  return span;
}

export function compBadge(posting) {
  if (posting.comp_min == null && posting.comp_max == null) return null;
  const span = document.createElement("span");
  span.className = "badge";
  const parts = [posting.comp_min, posting.comp_max].filter((v) => v != null);
  const range = parts.length === 2 ? `${parts[0].toLocaleString()}-${parts[1].toLocaleString()}` : parts[0].toLocaleString();
  span.textContent = `${posting.comp_currency ?? "$"} ${range}${posting.comp_period ? "/" + posting.comp_period : ""}`;
  if (posting.comp_raw) span.title = posting.comp_raw;
  return span;
}

// Mirrors app/services/postings.py's fit_tier_case() exactly (same order,
// same first-match-wins semantics) so the list view's badge agrees with
// sort=fit's server-side ordering. Duplicated logic rather than a new
// backend field for the tier number itself, since PostingSummary already
// carries the three raw fields (stage1_result/stage2_result/
// stage3_verdict) this derives from -- see that commit's note on the
// tradeoff.
// Keyed by the tier number itself (0-6, matching fit_tier_case()'s CASE
// values), not array position -- a plain array here previously
// mismapped every tier past index 1.
const FIT_TIERS = {
  6: { tone: "strong", label: "Strong Fit" },
  5: { tone: "moderate", label: "Moderate Fit" },
  4: { tone: "", label: "Fast-track" },
  3: { tone: "", label: "Analyze" },
  2: { tone: "", label: "Unscored" },
  1: { tone: "weak", label: "Weak Fit" },
  0: { tone: "weak", label: "Skip" },
};

export function fitTier(posting) {
  if (posting.stage3_verdict === "Strong Fit") return 6;
  if (posting.stage3_verdict === "Moderate Fit") return 5;
  if (posting.stage3_verdict === "Weak Fit") return 1;
  if (posting.stage3_verdict === "Skip") return 0;
  if (posting.stage2_result === "fast_track") return 4;
  if (posting.stage2_result === "analyze") return 3;
  if (posting.stage2_result === "skip") return 0;
  if (posting.stage1_result === "skip") return 0;
  return 2;
}

export function fitBadge(posting) {
  const tier = fitTier(posting);
  const info = FIT_TIERS[tier] ?? { tone: "", label: "Unscored" };
  const span = document.createElement("span");
  span.className = info.tone ? `badge tone-${info.tone}` : "badge";
  span.textContent = info.label;
  return span;
}

export function badgeRow(posting) {
  const row = document.createElement("div");
  row.className = "badge-row";
  [
    fitBadge(posting),
    freshnessBadge(posting.freshness),
    workplaceBadge(posting.workplace_type),
    compBadge(posting),
  ]
    .filter(Boolean)
    .forEach((b) => row.appendChild(b));
  return row;
}

export function renderVerdictLines(container, v, { includeStage1 = true } = {}) {
  container.innerHTML = "";
  if (!v) {
    container.innerHTML = '<div class="verdict-line">No verdict yet.</div>';
    return;
  }
  if (includeStage1 && v.stage1_result) {
    const line = document.createElement("div");
    line.className = "verdict-line";
    line.innerHTML = `<span class="stage">Stage 1</span>${esc(v.stage1_result)}${v.stage1_reason ? " -- " + esc(v.stage1_reason) : ""}`;
    container.appendChild(line);
  }
  if (v.stage2_result) {
    const line = document.createElement("div");
    line.className = "verdict-line";
    line.innerHTML = `<span class="stage">Stage 2</span>${esc(v.stage2_result)}${v.stage2_reason ? " -- " + esc(v.stage2_reason) : ""}${v.stage2_stale ? ' <span class="badge tone-weak">stale</span>' : ""}`;
    container.appendChild(line);
  }
  if (v.stage3_result) {
    const tone = STAGE3_TONE[v.stage3_result.verdict] ?? "";
    const line = document.createElement("div");
    line.className = tone ? `verdict-line tone-${tone}` : "verdict-line";
    line.innerHTML = `<span class="stage">Stage 3</span>${esc(v.stage3_result.verdict)} -- ${esc(v.stage3_result.one_liner)}${v.stage3_stale ? ' <span class="badge tone-weak">stale</span>' : ""}`;
    container.appendChild(line);

    if (v.stage3_result.stack_overlap?.length) {
      container.appendChild(chipList("Stack overlap", v.stage3_result.stack_overlap));
    }
    if (v.stage3_result.hard_gaps?.length) {
      container.appendChild(chipList("Hard gaps", v.stage3_result.hard_gaps, "weak"));
    }
    if (v.stage3_result.soft_gaps?.length) {
      container.appendChild(chipList("Soft gaps", v.stage3_result.soft_gaps));
    }
    if (v.stage3_result.developing_gaps?.length) {
      container.appendChild(chipList("Developing", v.stage3_result.developing_gaps));
    }
    if (v.stage3_result.roadmap_gaps?.length) {
      container.appendChild(chipList("Roadmap gaps", v.stage3_result.roadmap_gaps));
      if (v.stage3_result.roadmap_gap_caveat) {
        const caveat = document.createElement("div");
        caveat.className = "verdict-line";
        caveat.textContent = v.stage3_result.roadmap_gap_caveat;
        container.appendChild(caveat);
      }
    }
  } else {
    const line = document.createElement("div");
    line.className = "verdict-line";
    line.textContent = "Stage 3: not analyzed yet.";
    container.appendChild(line);
  }
}

function chipList(label, items, tone = "") {
  const wrap = document.createElement("div");
  wrap.className = "verdict-line";
  const labelEl = document.createElement("span");
  labelEl.className = "stage";
  labelEl.textContent = label;
  wrap.appendChild(labelEl);
  items.forEach((text) => {
    const chip = document.createElement("span");
    chip.className = tone ? `badge tone-${tone}` : "badge";
    chip.textContent = text;
    wrap.appendChild(chip);
  });
  return wrap;
}

// Used by roadmap trend cards and batch-result cards: a row of "#id"
// buttons, each expanding to a small inline detail card on click.
export function postingIdChipsRow(postingIds) {
  const wrap = document.createElement("div");
  wrap.className = "chips-wrap";

  const chipsRow = document.createElement("div");
  chipsRow.className = "row chips-row";

  const detailArea = document.createElement("div");
  detailArea.className = "chip-detail-area";

  postingIds.forEach((id) => {
    const chip = document.createElement("button");
    chip.className = "btn-secondary";
    chip.textContent = `#${id}`;
    chip.addEventListener("click", async () => {
      if (chip.dataset.open === "true") {
        detailArea.innerHTML = "";
        chip.dataset.open = "false";
        return;
      }
      chip.disabled = true;
      const detail = await call("get_posting", { id });
      chip.disabled = false;
      chip.dataset.open = "true";
      detailArea.innerHTML = "";

      const box = document.createElement("div");
      box.className = "card";
      const h4 = document.createElement("h4");
      h4.className = "chip-detail-title";
      h4.textContent = `${detail.title} -- ${detail.company_name}`;
      box.appendChild(h4);
      if (detail.url) {
        const link = document.createElement("button");
        link.className = "btn-secondary chip-apply-link";
        link.textContent = "Open posting / apply ↗";
        link.addEventListener("click", () => openExternal(detail.url));
        box.appendChild(link);
      }
      const verdictBox = document.createElement("div");
      verdictBox.className = "chip-verdict-box";
      renderVerdictLines(verdictBox, detail.latest_verdict);
      box.appendChild(verdictBox);
      detailArea.appendChild(box);
    });
    chipsRow.appendChild(chip);
  });

  wrap.append(chipsRow, detailArea);
  return wrap;
}

// Chip-list editor for a flat string[] field -- existing items as
// removable chips, a text input + Add button to append. Accepts either a
// container element id (getElementById-based, matching every other
// mounted widget in this app) or a raw DOM element directly, since
// mountRepeatableList needs to mount a taglist inside each dynamically
// created entry card, where there's no stable id to look up.
export function mountTagList(containerOrId, initialItems) {
  const container = typeof containerOrId === "string" ? document.getElementById(containerOrId) : containerOrId;
  container.className = "tag-list";
  container.innerHTML = "";

  const chipsRow = document.createElement("div");
  chipsRow.className = "tag-list-chips";

  let items = [...(initialItems ?? [])];

  function renderChips() {
    chipsRow.innerHTML = "";
    items.forEach((item, i) => {
      const chip = document.createElement("span");
      chip.className = "tag-chip";
      const text = document.createElement("span");
      text.textContent = item;
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "tag-chip-remove";
      removeBtn.textContent = "×";
      removeBtn.addEventListener("click", () => {
        items.splice(i, 1);
        renderChips();
      });
      chip.append(text, removeBtn);
      chipsRow.appendChild(chip);
    });
  }
  renderChips();

  const inputRow = document.createElement("div");
  inputRow.className = "row tag-list-input-row";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Add...";
  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "btn-secondary";
  addBtn.textContent = "Add";

  function addFromInput() {
    const value = input.value.trim();
    if (value && !items.includes(value)) {
      items.push(value);
      input.value = "";
      renderChips();
    }
  }
  addBtn.addEventListener("click", addFromInput);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addFromInput();
    }
  });
  inputRow.append(input, addBtn);

  container.append(chipsRow, inputRow);
  container.getValues = () => [...items];
}

// Generic add/remove/edit form for a list of structured objects -- one
// implementation reused for every repeatable nested list in the profile
// schema (experience, projects, credentials.oss_contributions) via
// different fieldSpecs, rather than three bespoke builders. fieldSpecs:
// [{key, label, type: "text"|"textarea"|"select"|"checkbox"|"taglist",
// options?, default?}]. container.getValues() reconstructs object[] by
// walking every entry card's own field widgets.
export function mountRepeatableList(containerId, fieldSpecs, initialEntries) {
  const container = document.getElementById(containerId);
  container.className = "repeatable-list";
  container.innerHTML = "";

  const entriesEl = document.createElement("div");
  const entryWidgets = [];

  function makeEntryCard(initialData) {
    const card = document.createElement("div");
    card.className = "card repeatable-list-entry";
    const fieldGetters = {};

    fieldSpecs.forEach((spec) => {
      const fieldWrap = document.createElement("div");
      fieldWrap.className = "repeatable-list-field";
      const label = document.createElement("label");
      label.textContent = spec.label;
      fieldWrap.appendChild(label);

      if (spec.type === "text") {
        const input = document.createElement("input");
        input.type = "text";
        input.value = initialData?.[spec.key] ?? "";
        fieldWrap.appendChild(input);
        fieldGetters[spec.key] = () => input.value.trim();
      } else if (spec.type === "textarea") {
        const textarea = document.createElement("textarea");
        textarea.className = "note-box";
        textarea.value = initialData?.[spec.key] ?? "";
        fieldWrap.appendChild(textarea);
        fieldGetters[spec.key] = () => textarea.value.trim();
      } else if (spec.type === "select") {
        const select = document.createElement("select");
        select.innerHTML = spec.options.map((o) => `<option value="${esc(o)}">${esc(o)}</option>`).join("");
        select.value = initialData?.[spec.key] ?? spec.default ?? spec.options[0];
        fieldWrap.appendChild(select);
        fieldGetters[spec.key] = () => select.value;
      } else if (spec.type === "checkbox") {
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = !!initialData?.[spec.key];
        fieldWrap.appendChild(checkbox);
        fieldGetters[spec.key] = () => checkbox.checked;
      } else if (spec.type === "taglist") {
        const taglistDiv = document.createElement("div");
        mountTagList(taglistDiv, initialData?.[spec.key] ?? []);
        fieldWrap.appendChild(taglistDiv);
        fieldGetters[spec.key] = () => taglistDiv.getValues();
      }
      card.appendChild(fieldWrap);
    });

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-secondary repeatable-list-remove-btn";
    removeBtn.textContent = "Remove entry";
    removeBtn.addEventListener("click", () => {
      card.remove();
      const idx = entryWidgets.findIndex((w) => w.root === card);
      if (idx !== -1) entryWidgets.splice(idx, 1);
    });
    card.appendChild(removeBtn);

    entryWidgets.push({
      root: card,
      getValues: () => {
        const obj = {};
        fieldSpecs.forEach((spec) => {
          obj[spec.key] = fieldGetters[spec.key]();
        });
        return obj;
      },
    });
    entriesEl.appendChild(card);
  }

  (initialEntries ?? []).forEach((entry) => makeEntryCard(entry));

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "btn-secondary repeatable-list-add-btn";
  addBtn.textContent = "Add entry";
  addBtn.addEventListener("click", () => makeEntryCard({}));

  container.append(entriesEl, addBtn);
  container.getValues = () => entryWidgets.map((w) => w.getValues());
}

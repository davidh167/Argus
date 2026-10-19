// The Companies tab (3.6): add a company, crawl/scrape it, hydrate its
// scraped postings' full descriptions. "Refresh all" (5.2) loops the same
// per-company crawl/scrape action across every enabled, configured
// company.
import { call, setStatus } from "../api.js";

let companiesLoaded = false;

// boards.greenhouse.io / job-boards.greenhouse.io -> greenhouse,
// jobs.lever.co -> lever, jobs.ashbyhq.com -> ashby, else left null
// (scraper-only).
function detectAtsType(url) {
  if (!url) return null;
  if (/(^|\/\/)(boards|job-boards)\.greenhouse\.io/.test(url)) return "greenhouse";
  if (/(^|\/\/)jobs\.lever\.co/.test(url)) return "lever";
  if (/(^|\/\/)jobs\.ashbyhq\.com/.test(url)) return "ashby";
  return null;
}

function resultLine(result, kind) {
  const parts = [`${result.jobs_found} found`, `${result.created} created`, `${result.updated} updated`];
  if (result.closure_evaluated) {
    parts.push(`${result.closed} closed`, `${result.reopened} reopened`);
  } else {
    parts.push("closure not evaluated this run");
  }
  if (kind === "scrape") {
    if (result.fallback_used) parts.push("fallback strategy used");
    parts.push(`${result.hydrated_inline} hydrated inline`);
  }
  return parts.join(", ");
}

function companyCard(company) {
  const card = document.createElement("div");
  card.className = "card";

  const main = document.createElement("div");
  main.className = "posting-main";
  const h3 = document.createElement("h3");
  h3.textContent = company.name;
  const subtitle = document.createElement("div");
  subtitle.className = "subtitle";
  const atsLabel = company.ats_type ?? "scraper";
  const lastCrawled = company.last_crawled_at ? company.last_crawled_at.slice(0, 10) : "never";
  subtitle.textContent = `${atsLabel} -- ${company.open_postings}/${company.total_postings} open -- last run ${lastCrawled}`;
  main.append(h3, subtitle);
  if (company.careers_url) {
    const urlLine = document.createElement("div");
    urlLine.className = "subtitle";
    urlLine.textContent = company.careers_url;
    main.appendChild(urlLine);
  }

  const resultBox = document.createElement("div");
  resultBox.className = "verdict-line company-result-box";
  resultBox.hidden = true;
  main.appendChild(resultBox);

  const actions = document.createElement("div");
  actions.className = "row company-actions";

  const runBtn = document.createElement("button");
  runBtn.className = "btn-secondary";
  const isCrawl = !!company.ats_type;
  runBtn.textContent = isCrawl ? "Crawl" : "Scrape";
  runBtn.disabled = !company.careers_url;
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    setStatus(`${isCrawl ? "Crawling" : "Scraping"} ${company.name}...`);
    try {
      const result = await call(isCrawl ? "crawl_company" : "scrape_company", { id: company.id });
      resultBox.hidden = false;
      resultBox.textContent = resultLine(result, isCrawl ? "crawl" : "scrape");
      setStatus(`${company.name} done.`, "success");
      loadCompanies();
    } finally {
      runBtn.disabled = false;
    }
  });
  actions.appendChild(runBtn);

  const hydrateBtn = document.createElement("button");
  hydrateBtn.className = "btn-secondary";
  hydrateBtn.textContent = "Fetch full descriptions";
  const progressLabel = document.createElement("span");
  progressLabel.className = "subtitle company-hydrate-progress";
  hydrateBtn.addEventListener("click", async () => {
    hydrateBtn.disabled = true;
    let round = await call("hydrate_postings", { companyId: company.id, postingIds: null, limit: 25 });
    let totalHydrated = round.hydrated;
    let totalFailed = round.failed;
    progressLabel.textContent = `${totalHydrated} hydrated, ${totalFailed} failed, ${round.remaining} remaining...`;
    // Stop when nothing's left, or this round made no progress at all
    // (a persistent per-posting failure would otherwise spin forever).
    while (round.remaining > 0 && round.hydrated + round.failed > 0) {
      round = await call("hydrate_postings", { companyId: company.id, postingIds: null, limit: 25 });
      totalHydrated += round.hydrated;
      totalFailed += round.failed;
      progressLabel.textContent = `${totalHydrated} hydrated, ${totalFailed} failed, ${round.remaining} remaining...`;
    }
    hydrateBtn.disabled = false;
    setStatus(`${company.name}: ${totalHydrated} hydrated, ${totalFailed} failed.`, "success");
    loadCompanies();
  });
  actions.append(hydrateBtn, progressLabel);

  card.append(main, actions);
  return card;
}

function refreshAllEntryLine(entry) {
  if (!entry.ok) return `${entry.company_name}: error -- ${entry.error}`;
  const kind = "fallback_used" in entry.result ? "scrape" : "crawl";
  return `${entry.company_name}: ${resultLine(entry.result, kind)}`;
}

export async function loadCompanies() {
  companiesLoaded = true;
  const list = document.getElementById("companiesList");
  list.innerHTML = "";
  setStatus("Loading companies...");
  const companies = await call("list_companies", {});
  setStatus("");
  if (companies.length === 0) {
    list.innerHTML = '<div class="empty-state">No companies tracked yet.</div>';
    return;
  }
  companies.forEach((c) => list.appendChild(companyCard(c)));
}

export function initCompaniesView() {
  const urlInput = document.getElementById("companyCareersUrl");
  const atsSelect = document.getElementById("companyAtsType");
  urlInput.addEventListener("input", () => {
    atsSelect.value = detectAtsType(urlInput.value.trim()) ?? "";
  });

  document.getElementById("addCompanyBtn").addEventListener("click", async () => {
    const name = document.getElementById("companyName").value.trim();
    if (!name) {
      setStatus("Company name is required.", "error");
      return;
    }
    const careersUrl = urlInput.value.trim() || null;
    const atsType = atsSelect.value || null;
    const btn = document.getElementById("addCompanyBtn");
    btn.disabled = true;
    await call("create_company", { name, careersUrl, atsType, notes: null });
    btn.disabled = false;
    document.getElementById("companyName").value = "";
    urlInput.value = "";
    atsSelect.value = "";
    setStatus(`Added ${name}.`, "success");
    loadCompanies();
  });

  document.getElementById("refreshAllCompaniesBtn").addEventListener("click", async () => {
    const btn = document.getElementById("refreshAllCompaniesBtn");
    const resultsEl = document.getElementById("refreshAllResults");
    btn.disabled = true;
    setStatus("Refreshing all enabled companies...");
    const { results } = await call("refresh_all_companies", {});
    btn.disabled = false;
    resultsEl.innerHTML = "";
    if (results.length === 0) {
      resultsEl.innerHTML = '<div class="empty-state">No enabled companies with a careers URL to refresh.</div>';
    } else {
      const card = document.createElement("div");
      card.className = "card";
      results.forEach((entry) => {
        const line = document.createElement("div");
        line.className = "verdict-line";
        line.textContent = refreshAllEntryLine(entry);
        card.appendChild(line);
      });
      resultsEl.appendChild(card);
    }
    setStatus(`Refreshed ${results.length} company(s).`, "success");
    loadCompanies();
  });
}

export function companiesTabActivated() {
  if (!companiesLoaded) loadCompanies();
}

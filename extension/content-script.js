// Injected on demand (via chrome.scripting.executeScript, triggered only by
// the toolbar-icon click -- see manifest.json, no persistent content_scripts
// entry). Runs in the page's own DOM, which a hostile page can manipulate,
// so everything returned here is treated as untrusted: it only ever
// pre-fills editable fields in the popup, never anything trusted directly.
//
// This must be a single expression statement (IIFE call) -- when injected
// via `files:` rather than `func:`, chrome.scripting.executeScript's result
// is the completion value of the last-evaluated statement.
(function extractJobPosting() {
  function metaContent(name) {
    const el = document.querySelector(`meta[property="${name}"], meta[name="${name}"]`);
    return el ? el.content.trim() : null;
  }

  // <title> separates job title from company with a variety of punctuation
  // across ATS platforms -- hyphen, pipe, em/en dash, "@", or the bare word
  // "at" -- and splitting on just [-|] misses the rest entirely. Confirmed
  // live against Greenhouse ("X at Company"), Ashby ("X @ Company"), and
  // Google Careers ("X — Google Careers"): the old split left the company
  // guess as the ENTIRE unsplit title on Greenhouse/Ashby.
  const TITLE_SEPARATOR = /\s+[-|–—@]\s+|\s+\bat\b\s+/i;

  // LinkedIn's title/og:title has its own multi-segment shape -- "X at
  // Company — Location | LinkedIn Jobs" -- confirmed live against a real
  // posting. Splitting that on TITLE_SEPARATOR yields 4 parts, and taking
  // the LAST one (splitTitleTag's rule for the 2-part ATS case) grabs
  // "LinkedIn Jobs" as the company guess, not "Stripe". The word "at" is
  // an unambiguous marker -- when present, the company is specifically the
  // segment right after it, not whatever segment happens to be last.
  const AT_SEPARATOR = /\s+\bat\b\s+/i;

  // Some ATS platforms (eightfold.ai, Google Careers) render a static
  // page-shell <h1> ("Jobs", "Job Details") instead of the specific
  // posting's title when linked to directly -- treat these as
  // untrustworthy rather than confidently wrong.
  const GENERIC_H1 = /^(jobs?|job details|careers?|open positions?|apply( now)?)$/i;

  function splitTitleTag(rawTitle) {
    if (AT_SEPARATOR.test(rawTitle)) {
      const [beforeAt, afterAt] = rawTitle.split(AT_SEPARATOR);
      // Whatever comes after "at" may itself carry more segments (LinkedIn's
      // "Company — Location | LinkedIn Jobs") -- the company is only the
      // first of those, not the rest.
      const company = afterAt.split(TITLE_SEPARATOR)[0]?.trim();
      return {
        title: beforeAt.replace(/^job application for\s+/i, "").trim(),
        company: company || null,
      };
    }
    const parts = rawTitle.split(TITLE_SEPARATOR).map((s) => s.trim()).filter(Boolean);
    if (parts.length < 2) return { title: rawTitle, company: null };
    return {
      title: parts[0].replace(/^job application for\s+/i, "").trim(),
      company: parts[parts.length - 1],
    };
  }

  // LinkedIn-specific: a direct DOM lookup beats title-parsing when it
  // matches. Confirmed live on LinkedIn's logged-out job view
  // (`topcard__org-name-link`). LinkedIn's logged-in view renders
  // different markup entirely -- `job-details-jobs-unified-top-card__
  // company-name` is that view's commonly-documented class, included as a
  // best-effort fallback since it couldn't be verified live without
  // signing in. Both are harmless no-ops (null) on every other site.
  function linkedInCompanyGuess() {
    if (!/\blinkedin\.com$/.test(location.hostname)) return null;
    const el =
      document.querySelector("a.topcard__org-name-link") ||
      document.querySelector(".job-details-jobs-unified-top-card__company-name");
    return el?.innerText?.trim() || null;
  }

  const h1 = document.querySelector("h1")?.innerText?.trim();
  const fromTitleTag = splitTitleTag(document.title.trim());
  const h1IsTrustworthy = h1 && !GENERIC_H1.test(h1);

  const titleGuess = h1IsTrustworthy ? h1 : fromTitleTag.title;
  const companyGuess =
    linkedInCompanyGuess() ||
    metaContent("og:site_name") ||
    fromTitleTag.company ||
    location.hostname.replace(/^www\./, "");

  return {
    url: location.href,
    titleGuess,
    companyGuess,
    text: document.body.innerText,
  };
})();

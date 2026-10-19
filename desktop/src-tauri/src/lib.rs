use serde::{Deserialize, Serialize};

const BASE_URL: &str = "http://127.0.0.1:8000";

// ---- Response types, mirroring the FastAPI backend's Pydantic schemas ----
// (app/schemas/posting.py, verdict.py, application.py, roadmap.py). Dates
// stay as plain Strings -- nothing in this app needs to compute on them,
// only display them, so pulling in chrono would be unused weight.

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PostingSummary {
    pub id: i64,
    pub company_id: i64,
    pub company_name: String,
    pub title: String,
    pub url: Option<String>,
    pub source: Option<String>,
    pub starred: bool,
    pub starred_at: Option<String>,
    #[serde(default)]
    pub dismissed: bool,
    #[serde(default)]
    pub dismissed_at: Option<String>,
    pub note: Option<String>,
    pub freshness: String,
    pub repost_count: i64,
    pub stage1_result: Option<String>,
    #[serde(default)]
    pub stage2_result: Option<String>,
    #[serde(default)]
    pub stage3_verdict: Option<String>,
    #[serde(default)]
    pub locations: Option<Vec<String>>,
    #[serde(default)]
    pub workplace_type: Option<String>,
    #[serde(default)]
    pub comp_min: Option<i64>,
    #[serde(default)]
    pub comp_max: Option<i64>,
    #[serde(default)]
    pub comp_currency: Option<String>,
    #[serde(default)]
    pub comp_period: Option<String>,
    #[serde(default)]
    pub relevance_score: Option<f64>,
}

/// GET /postings' response, paired with its X-Total-Count header -- lets
/// the frontend show "X of N" when the list is capped by `limit`.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PostingsPage {
    pub items: Vec<PostingSummary>,
    pub total: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VerdictOut {
    pub id: i64,
    pub posting_id: i64,
    pub profile_version_id: i64,
    pub created_at: String,
    pub updated_at: String,
    pub stage1_result: Option<String>,
    pub stage1_reason: Option<String>,
    pub stage2_result: Option<String>,
    pub stage2_reason: Option<String>,
    pub stage2_tokens_used: Option<i64>,
    // Computed at read time by the backend (see build_verdict_out), not
    // stored -- true when this stage's result was computed against a
    // profile version that isn't current anymore. Replaces the old
    // never-written is_stale field. #[serde(default)] so an older cached
    // response shape (or a future field rename) fails soft, not with a
    // hard deserialization error.
    #[serde(default)]
    pub stage2_stale: bool,
    pub stage3_triggered_at: Option<String>,
    pub stage3_result: Option<serde_json::Value>,
    pub stage3_tokens_used: Option<i64>,
    #[serde(default)]
    pub stage3_stale: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct JDSections {
    pub summary: String,
    pub responsibilities: String,
    pub required_qualifications: String,
    pub preferred_qualifications: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PostingDetail {
    #[serde(flatten)]
    pub summary: PostingSummary,
    pub raw_jd_text: Option<String>,
    pub posted_date: Option<String>,
    pub first_seen_at: String,
    pub last_seen_at: String,
    #[serde(default)]
    pub closed_at: Option<String>,
    #[serde(default)]
    pub department: Option<String>,
    #[serde(default)]
    pub comp_source: Option<String>,
    #[serde(default)]
    pub comp_raw: Option<String>,
    #[serde(default)]
    pub detail_fetched_at: Option<String>,
    #[serde(default)]
    pub relevance_matched_terms: Option<Vec<String>>,
    #[serde(default)]
    pub jd_sections: Option<JDSections>,
    pub latest_verdict: Option<VerdictOut>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SimilarPosting {
    pub posting: PostingSummary,
    pub similarity: f64,
    pub possibly_same_role: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HydrateResult {
    pub hydrated: i64,
    pub failed: i64,
    pub remaining: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApplicationOut {
    pub id: i64,
    pub posting_id: i64,
    pub posting_title: String,
    pub posting_company_name: String,
    #[serde(default)]
    pub posting_freshness: String,
    pub status: Option<String>,
    pub resume_used: Option<String>,
    pub date_applied: Option<String>,
    pub follow_up_date: Option<String>,
    pub notes: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnalyzePendingSubmitted {
    pub batch_id: Option<String>,
    pub submitted: i64,
    pub capped: bool,
    #[serde(default)]
    pub excluded_unhydrated: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TriagePendingSubmitted {
    pub batch_id: Option<String>,
    pub submitted: i64,
    pub capped: bool,
    #[serde(default)]
    pub excluded_unhydrated: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnalyzePendingApplied {
    pub status: String,
    pub applied: i64,
    pub errors: i64,
    pub skipped_stale: i64,
    #[serde(default)]
    pub posting_ids: Vec<i64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TriagePendingApplied {
    pub status: String,
    pub applied: i64,
    pub errors: i64,
    pub skipped_stale: i64,
    #[serde(default)]
    pub posting_ids: Vec<i64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BatchSubmissionOut {
    pub id: i64,
    pub batch_id: String,
    pub kind: String,
    pub submitted_count: i64,
    pub capped: bool,
    pub company_id: Option<i64>,
    pub created_at: String,
    pub last_status: Option<String>,
    pub last_checked_at: Option<String>,
    #[serde(default)]
    pub applied_at: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SkillRoadmapOut {
    pub id: i64,
    pub skill_or_language: String,
    pub trigger_type: Option<String>,
    pub is_new_territory: Option<bool>,
    pub trend_evidence: Option<Vec<i64>>,
    pub status: String,
    pub what_it_is: Option<String>,
    pub why_it_matters: Option<String>,
    pub suggested_scope: Option<String>,
    pub suggested_stack: Option<String>,
    pub strengthens_project: Option<String>,
    pub project_note: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SkillTrend {
    pub skill_or_language: String,
    pub posting_ids: Vec<i64>,
    pub occurrence_count: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReadinessEntry {
    pub posting_id: i64,
    pub title: String,
    pub company_name: String,
    pub verdict: String,
    pub one_liner: String,
    pub hard_gaps: Vec<String>,
    #[serde(default)]
    pub roadmap_item_id: Option<i64>,
    #[serde(default)]
    pub matched_gap: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ReadinessView {
    pub apply_now: Vec<ReadinessEntry>,
    pub solid_match: Vec<ReadinessEntry>,
    pub stretch: Vec<ReadinessEntry>,
    pub build_toward: Vec<ReadinessEntry>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DemandCluster {
    pub skill_or_language: String,
    pub posting_count: i64,
    pub posting_ids: Vec<i64>,
    pub status: String,
    #[serde(default)]
    pub interest_weight: f64,
    #[serde(default)]
    pub interest_posting_count: i64,
    #[serde(default)]
    pub phrasings: Vec<String>,
    #[serde(default)]
    pub judged: bool,
    #[serde(default)]
    pub judgment_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SkillJudgmentPreview {
    pub to_judge: i64,
    pub already_judged: i64,
    pub estimated_input_tokens: i64,
    pub estimated_cost_usd: f64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SkillJudgmentRun {
    pub judged: i64,
    pub remaining: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompanyOut {
    pub id: i64,
    pub name: String,
    pub domain: Option<String>,
    pub ats_type: Option<String>,
    pub careers_url: Option<String>,
    pub last_crawled_at: Option<String>,
    pub crawl_enabled: bool,
    pub notes: Option<String>,
    #[serde(default)]
    pub total_postings: i64,
    #[serde(default)]
    pub open_postings: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CrawlResult {
    pub jobs_found: i64,
    pub created: i64,
    pub updated: i64,
    pub closed: i64,
    pub reopened: i64,
    pub closure_evaluated: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ScrapeResult {
    pub jobs_found: i64,
    pub created: i64,
    pub updated: i64,
    pub closed: i64,
    pub reopened: i64,
    pub fallback_used: bool,
    pub closure_evaluated: bool,
    pub hydrated_inline: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RefreshAllEntry {
    pub company_id: i64,
    pub company_name: String,
    pub ok: bool,
    // A loose JSON value rather than CrawlResult|ScrapeResult -- a
    // crawl-type and scrape-type company can both appear in the same run,
    // mirroring RefreshAllEntry's own shape on the backend.
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RefreshAllResult {
    pub results: Vec<RefreshAllEntry>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BatchPreview {
    pub eligible: i64,
    pub would_submit: i64,
    pub capped: bool,
    pub excluded_unhydrated: i64,
    #[serde(default)]
    pub in_pending_batches: i64,
    pub available: bool,
    #[serde(default)]
    pub avg_tokens_per_posting: Option<i64>,
    #[serde(default)]
    pub estimated_cost_usd: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StageSpend {
    pub tokens_all_time: i64,
    pub estimated_cost_usd_all_time: f64,
    pub tokens_last_30d: i64,
    pub estimated_cost_usd_last_30d: f64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SpendSummary {
    pub stage2: StageSpend,
    pub stage3: StageSpend,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProfileSnapshotOut {
    pub id: i64,
    pub created_at: String,
    pub changelog: Option<String>,
    // Stays raw JSON here rather than hand-written structs for every
    // nested experience/project/skills/credentials/criteria shape --
    // mirrors how VerdictOut.stage3_result already does this for a
    // comparably deep, read-only-display structure. views/profile.js
    // reads fields directly off the object.
    pub snapshot: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ParsedFileResponse {
    pub text: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProfileIntakeMessageOut {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProfileIntakeChatResponse {
    #[serde(default)]
    pub assistant_message: Option<String>,
    // Raw JSON, same reasoning as ProfileSnapshotOut.snapshot above --
    // views/profile.js's existing mountEditForm reads fields directly.
    #[serde(default)]
    pub proposed_snapshot: Option<serde_json::Value>,
    #[serde(default)]
    pub changelog: Option<String>,
    #[serde(default)]
    pub open_questions: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProfileVersionSummary {
    pub id: i64,
    pub created_at: String,
    pub changelog: Option<String>,
}

// ---- HTTP plumbing ----

struct AppState {
    client: reqwest::Client,
}

fn build_client() -> reqwest::Client {
    // Reads API_TOKEN from the project-root .env -- this app runs on the
    // same machine as the backend, same token the extension already uses,
    // no separate settings screen needed.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let env_path = std::path::Path::new(manifest_dir).join("../../.env");
    dotenvy::from_path(&env_path).ok();
    let token = std::env::var("API_TOKEN").unwrap_or_default();

    let mut headers = reqwest::header::HeaderMap::new();
    if !token.is_empty() {
        if let Ok(value) = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")) {
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .expect("failed to build HTTP client")
}

async fn request_json<B: Serialize + ?Sized, T: for<'de> Deserialize<'de>>(
    client: &reqwest::Client,
    method: reqwest::Method,
    path: &str,
    query: &[(&str, Option<String>)],
    body: Option<&B>,
) -> Result<T, String> {
    let mut url = reqwest::Url::parse(&format!("{BASE_URL}{path}")).map_err(|e| e.to_string())?;
    {
        let mut pairs = url.query_pairs_mut();
        for (key, value) in query {
            if let Some(v) = value {
                pairs.append_pair(key, v);
            }
        }
    }
    let mut req = client.request(method, url);
    if let Some(b) = body {
        req = req.json(b);
    }
    let resp = req.send().await.map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    resp.json::<T>().await.map_err(|e| e.to_string())
}

// ---- Commands, each a thin wrapper over an existing, already-tested route ----

// Mirrors app/routers/postings.py's list_postings signature exactly (same
// param names) so views/postings.js can pass its filter-bar state straight
// through without any renaming in between.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn list_postings(
    state: tauri::State<'_, AppState>,
    company: Option<String>,
    starred: Option<bool>,
    url_filter: Option<String>,
    posted_after: Option<String>,
    first_seen_after: Option<String>,
    keyword: Option<String>,
    keyword_scope: Option<String>,
    location: Option<String>,
    workplace_type: Option<String>,
    location_match: Option<bool>,
    comp_min: Option<i64>,
    has_comp: Option<bool>,
    open_only: Option<bool>,
    include_dismissed: Option<bool>,
    stage3_verdict: Option<Vec<String>>,
    stage2_result: Option<Vec<String>>,
    source: Option<String>,
    department: Option<String>,
    sort: Option<String>,
    limit: Option<i64>,
) -> Result<PostingsPage, String> {
    // Needs the X-Total-Count response header alongside the JSON body, so
    // this can't go through the generic request_json helper (which only
    // returns the deserialized body). stage3_verdict/stage2_result are
    // repeated query params (?stage3_verdict=A&stage3_verdict=B), which the
    // [(&str, Option<String>)]-shaped scalar params below can't express.
    let mut req_url = reqwest::Url::parse(&format!("{BASE_URL}/postings")).map_err(|e| e.to_string())?;
    {
        let mut pairs = req_url.query_pairs_mut();
        for (key, value) in [
            ("company", company),
            ("starred", starred.map(|b| b.to_string())),
            ("url", url_filter),
            ("posted_after", posted_after),
            ("first_seen_after", first_seen_after),
            ("keyword", keyword),
            ("keyword_scope", keyword_scope),
            ("location", location),
            ("workplace_type", workplace_type),
            ("location_match", location_match.map(|b| b.to_string())),
            ("comp_min", comp_min.map(|c| c.to_string())),
            ("has_comp", has_comp.map(|b| b.to_string())),
            ("open_only", open_only.map(|b| b.to_string())),
            ("include_dismissed", include_dismissed.map(|b| b.to_string())),
            ("source", source),
            ("department", department),
            ("sort", sort),
            ("limit", limit.map(|l| l.to_string())),
        ] {
            if let Some(v) = value {
                pairs.append_pair(key, &v);
            }
        }
        for v in stage3_verdict.into_iter().flatten() {
            pairs.append_pair("stage3_verdict", &v);
        }
        for v in stage2_result.into_iter().flatten() {
            pairs.append_pair("stage2_result", &v);
        }
    }
    let resp = state
        .client
        .get(req_url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    let total: i64 = resp
        .headers()
        .get("x-total-count")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let items: Vec<PostingSummary> = resp.json().await.map_err(|e| e.to_string())?;
    Ok(PostingsPage { items, total })
}

#[tauri::command]
async fn get_posting(state: tauri::State<'_, AppState>, id: i64) -> Result<PostingDetail, String> {
    request_json(&state.client, reqwest::Method::GET, &format!("/postings/{id}"), &[], None::<&()>).await
}

#[tauri::command]
async fn set_posting_note(
    state: tauri::State<'_, AppState>,
    id: i64,
    note: Option<String>,
) -> Result<PostingSummary, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/{id}/note"),
        &[("note", note)],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn similar_postings(
    state: tauri::State<'_, AppState>,
    id: i64,
    limit: Option<i64>,
) -> Result<Vec<SimilarPosting>, String> {
    request_json(
        &state.client,
        reqwest::Method::GET,
        &format!("/postings/{id}/similar"),
        &[("limit", limit.map(|l| l.to_string()))],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn hydrate_postings(
    state: tauri::State<'_, AppState>,
    company_id: Option<i64>,
    posting_ids: Option<Vec<i64>>,
    limit: Option<i64>,
) -> Result<HydrateResult, String> {
    // posting_ids is a repeated query param -- built directly rather than
    // through request_json's scalar-only query slice, same reasoning as
    // list_postings' stage2/3_result filters.
    let mut req_url = reqwest::Url::parse(&format!("{BASE_URL}/postings/hydrate")).map_err(|e| e.to_string())?;
    {
        let mut pairs = req_url.query_pairs_mut();
        if let Some(c) = company_id {
            pairs.append_pair("company_id", &c.to_string());
        }
        if let Some(l) = limit {
            pairs.append_pair("limit", &l.to_string());
        }
        for id in posting_ids.into_iter().flatten() {
            pairs.append_pair("posting_ids", &id.to_string());
        }
    }
    let resp = state
        .client
        .post(req_url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn star_posting(state: tauri::State<'_, AppState>, id: i64) -> Result<PostingSummary, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/{id}/star"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn unstar_posting(state: tauri::State<'_, AppState>, id: i64) -> Result<PostingSummary, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/{id}/unstar"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn dismiss_posting(state: tauri::State<'_, AppState>, id: i64) -> Result<PostingSummary, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/{id}/dismiss"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn undismiss_posting(state: tauri::State<'_, AppState>, id: i64) -> Result<PostingSummary, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/{id}/undismiss"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn analyze_posting(state: tauri::State<'_, AppState>, id: i64) -> Result<VerdictOut, String> {
    // Stage 3 -- a real Sonnet call, never automatic. No request body.
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/{id}/analyze"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn submit_stage3_batch(
    state: tauri::State<'_, AppState>,
    posting_ids: Vec<i64>,
    limit: Option<i64>,
    include_stale: Option<bool>,
) -> Result<AnalyzePendingSubmitted, String> {
    // posting_ids needs repeated query params (?posting_ids=1&posting_ids=2),
    // which the [(&str, Option<String>)]-shaped request_json helper can't
    // express (one value per key) -- built directly here instead.
    let mut url =
        reqwest::Url::parse(&format!("{BASE_URL}/postings/analyze-pending")).map_err(|e| e.to_string())?;
    {
        let mut pairs = url.query_pairs_mut();
        for id in &posting_ids {
            pairs.append_pair("posting_ids", &id.to_string());
        }
        if let Some(l) = limit {
            pairs.append_pair("limit", &l.to_string());
        }
        if let Some(s) = include_stale {
            pairs.append_pair("include_stale", &s.to_string());
        }
    }
    let resp = state
        .client
        .post(url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn submit_key_skills_refresh(
    state: tauri::State<'_, AppState>,
) -> Result<TriagePendingSubmitted, String> {
    request_json(&state.client, reqwest::Method::POST, "/postings/key-skills-refresh", &[], None::<&()>).await
}

#[tauri::command]
async fn submit_stage2_batch(
    state: tauri::State<'_, AppState>,
    posting_ids: Vec<i64>,
    limit: Option<i64>,
) -> Result<TriagePendingSubmitted, String> {
    // Mirrors submit_stage3_batch exactly -- there was no Stage 2 submit
    // command at all before this (only apply_stage2_batch, for checking on
    // a batch already submitted via curl/MCP).
    let mut url =
        reqwest::Url::parse(&format!("{BASE_URL}/postings/triage-pending")).map_err(|e| e.to_string())?;
    {
        let mut pairs = url.query_pairs_mut();
        for id in &posting_ids {
            pairs.append_pair("posting_ids", &id.to_string());
        }
        if let Some(l) = limit {
            pairs.append_pair("limit", &l.to_string());
        }
    }
    let resp = state
        .client
        .post(url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn triage_pending_preview(
    state: tauri::State<'_, AppState>,
    posting_ids: Vec<i64>,
    limit: Option<i64>,
) -> Result<BatchPreview, String> {
    let mut url = reqwest::Url::parse(&format!("{BASE_URL}/postings/triage-pending/preview"))
        .map_err(|e| e.to_string())?;
    {
        let mut pairs = url.query_pairs_mut();
        for id in &posting_ids {
            pairs.append_pair("posting_ids", &id.to_string());
        }
        if let Some(l) = limit {
            pairs.append_pair("limit", &l.to_string());
        }
    }
    let resp = state
        .client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn analyze_pending_preview(
    state: tauri::State<'_, AppState>,
    posting_ids: Vec<i64>,
    limit: Option<i64>,
    include_stale: Option<bool>,
) -> Result<BatchPreview, String> {
    let mut url = reqwest::Url::parse(&format!("{BASE_URL}/postings/analyze-pending/preview"))
        .map_err(|e| e.to_string())?;
    {
        let mut pairs = url.query_pairs_mut();
        for id in &posting_ids {
            pairs.append_pair("posting_ids", &id.to_string());
        }
        if let Some(l) = limit {
            pairs.append_pair("limit", &l.to_string());
        }
        if let Some(s) = include_stale {
            pairs.append_pair("include_stale", &s.to_string());
        }
    }
    let resp = state
        .client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_spend_summary(state: tauri::State<'_, AppState>) -> Result<SpendSummary, String> {
    request_json(&state.client, reqwest::Method::GET, "/postings/spend-summary", &[], None::<&()>).await
}

#[tauri::command]
async fn list_companies(state: tauri::State<'_, AppState>) -> Result<Vec<CompanyOut>, String> {
    request_json(&state.client, reqwest::Method::GET, "/companies", &[], None::<&()>).await
}

#[derive(Serialize)]
struct CreateCompanyBody<'a> {
    name: &'a str,
    careers_url: Option<&'a str>,
    ats_type: Option<&'a str>,
    notes: Option<&'a str>,
}

#[tauri::command]
async fn create_company(
    state: tauri::State<'_, AppState>,
    name: String,
    careers_url: Option<String>,
    ats_type: Option<String>,
    notes: Option<String>,
) -> Result<CompanyOut, String> {
    let body = CreateCompanyBody {
        name: &name,
        careers_url: careers_url.as_deref(),
        ats_type: ats_type.as_deref(),
        notes: notes.as_deref(),
    };
    request_json(&state.client, reqwest::Method::POST, "/companies", &[], Some(&body)).await
}

#[tauri::command]
async fn crawl_company(state: tauri::State<'_, AppState>, id: i64) -> Result<CrawlResult, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/companies/{id}/crawl"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn scrape_company(state: tauri::State<'_, AppState>, id: i64) -> Result<ScrapeResult, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/companies/{id}/scrape"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn refresh_all_companies(state: tauri::State<'_, AppState>) -> Result<RefreshAllResult, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        "/companies/refresh-all",
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn apply_stage3_batch(
    state: tauri::State<'_, AppState>,
    batch_id: String,
) -> Result<AnalyzePendingApplied, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/analyze-pending/{batch_id}/apply"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn apply_stage2_batch(
    state: tauri::State<'_, AppState>,
    batch_id: String,
) -> Result<TriagePendingApplied, String> {
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/triage-pending/{batch_id}/apply"),
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn list_batches(
    state: tauri::State<'_, AppState>,
    kind: Option<String>,
) -> Result<Vec<BatchSubmissionOut>, String> {
    request_json(&state.client, reqwest::Method::GET, "/batches", &[("kind", kind)], None::<&()>).await
}

#[tauri::command]
async fn get_application(
    state: tauri::State<'_, AppState>,
    posting_id: i64,
) -> Result<Option<ApplicationOut>, String> {
    request_json(
        &state.client,
        reqwest::Method::GET,
        &format!("/postings/{posting_id}/application"),
        &[],
        None::<&()>,
    )
    .await
}

#[derive(Serialize)]
struct StatusBody<'a> {
    status: &'a str,
}

#[derive(Serialize)]
struct UpdateApplicationBody<'a> {
    status: &'a str,
    resume_used: Option<&'a str>,
    date_applied: Option<&'a str>,
    follow_up_date: Option<&'a str>,
    notes: Option<&'a str>,
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn set_application_status(
    state: tauri::State<'_, AppState>,
    posting_id: i64,
    status: String,
    resume_used: Option<String>,
    date_applied: Option<String>,
    follow_up_date: Option<String>,
    notes: Option<String>,
) -> Result<ApplicationOut, String> {
    let body = UpdateApplicationBody {
        status: &status,
        resume_used: resume_used.as_deref(),
        date_applied: date_applied.as_deref(),
        follow_up_date: follow_up_date.as_deref(),
        notes: notes.as_deref(),
    };
    request_json(
        &state.client,
        reqwest::Method::POST,
        &format!("/postings/{posting_id}/application"),
        &[],
        Some(&body),
    )
    .await
}

#[tauri::command]
async fn list_applications(
    state: tauri::State<'_, AppState>,
    status: Option<String>,
    sort: Option<String>,
) -> Result<Vec<ApplicationOut>, String> {
    request_json(
        &state.client,
        reqwest::Method::GET,
        "/applications",
        &[("status", status), ("sort", sort)],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn export_applications_csv(
    state: tauri::State<'_, AppState>,
    dest_path: String,
) -> Result<(), String> {
    let url = format!("{BASE_URL}/applications/export.csv");
    let resp = state
        .client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    std::fs::write(&dest_path, &bytes).map_err(|e| format!("Could not write {dest_path}: {e}"))
}

#[tauri::command]
async fn get_current_profile(state: tauri::State<'_, AppState>) -> Result<ProfileSnapshotOut, String> {
    request_json(&state.client, reqwest::Method::GET, "/profile/current", &[], None::<&()>).await
}

#[tauri::command]
async fn list_profile_versions(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<ProfileVersionSummary>, String> {
    request_json(&state.client, reqwest::Method::GET, "/profile/versions", &[], None::<&()>).await
}

#[derive(Serialize)]
struct CreateProfileVersionBody {
    snapshot: serde_json::Value,
    changelog: String,
}

#[tauri::command]
async fn create_profile_version(
    state: tauri::State<'_, AppState>,
    snapshot: serde_json::Value,
    changelog: String,
) -> Result<ProfileSnapshotOut, String> {
    let body = CreateProfileVersionBody { snapshot, changelog };
    request_json(&state.client, reqwest::Method::POST, "/profile", &[], Some(&body)).await
}

#[derive(Serialize)]
struct ProfileIntakeChatBody {
    user_message: String,
}

#[tauri::command]
async fn profile_intake_chat(
    state: tauri::State<'_, AppState>,
    user_message: String,
) -> Result<ProfileIntakeChatResponse, String> {
    let body = ProfileIntakeChatBody { user_message };
    request_json(&state.client, reqwest::Method::POST, "/profile/intake-chat", &[], Some(&body)).await
}

#[tauri::command]
async fn parse_resume_file(
    state: tauri::State<'_, AppState>,
    src_path: String,
) -> Result<ParsedFileResponse, String> {
    let bytes = std::fs::read(&src_path).map_err(|e| format!("Could not read {src_path}: {e}"))?;
    let filename = std::path::Path::new(&src_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| src_path.clone());
    let part = reqwest::multipart::Part::bytes(bytes).file_name(filename);
    let form = reqwest::multipart::Form::new().part("file", part);

    let url = format!("{BASE_URL}/profile/intake-chat/parse-file");
    let resp = state
        .client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_intake_chat_history(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<ProfileIntakeMessageOut>, String> {
    request_json(
        &state.client,
        reqwest::Method::GET,
        "/profile/intake-chat/history",
        &[],
        None::<&()>,
    )
    .await
}

#[tauri::command]
async fn clear_intake_chat_history(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let _: serde_json::Value = request_json(
        &state.client,
        reqwest::Method::DELETE,
        "/profile/intake-chat/history",
        &[],
        None::<&()>,
    )
    .await?;
    Ok(())
}

#[tauri::command]
async fn list_roadmap(
    state: tauri::State<'_, AppState>,
    status: Option<String>,
) -> Result<Vec<SkillRoadmapOut>, String> {
    request_json(&state.client, reqwest::Method::GET, "/roadmap", &[("status", status)], None::<&()>).await
}

#[tauri::command]
async fn detect_skill_trends(
    state: tauri::State<'_, AppState>,
    threshold: i64,
    posting_ids: Option<Vec<i64>>,
) -> Result<Vec<SkillTrend>, String> {
    // posting_ids is a repeated query param -- same reasoning as
    // get_market_demand's.
    let mut url =
        reqwest::Url::parse(&format!("{BASE_URL}/roadmap/trends")).map_err(|e| e.to_string())?;
    {
        let mut pairs = url.query_pairs_mut();
        pairs.append_pair("threshold", &threshold.to_string());
        for id in posting_ids.into_iter().flatten() {
            pairs.append_pair("posting_ids", &id.to_string());
        }
    }
    let resp = state
        .client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Backend error: {}", resp.status()));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_readiness_view(state: tauri::State<'_, AppState>) -> Result<ReadinessView, String> {
    request_json(&state.client, reqwest::Method::GET, "/roadmap/readiness", &[], None::<&()>).await
}

#[tauri::command]
async fn preview_skill_judgments(
    state: tauri::State<'_, AppState>,
) -> Result<SkillJudgmentPreview, String> {
    request_json(&state.client, reqwest::Method::GET, "/roadmap/judge-skills/preview", &[], None::<&()>).await
}

#[tauri::command]
async fn judge_skills(state: tauri::State<'_, AppState>) -> Result<SkillJudgmentRun, String> {
    request_json(&state.client, reqwest::Method::POST, "/roadmap/judge-skills", &[], None::<&()>).await
}

#[tauri::command]
async fn get_market_demand(
    state: tauri::State<'_, AppState>,
    posting_ids: Option<Vec<i64>>,
    sort: String,
) -> Result<Vec<DemandCluster>, String> {
    // posting_ids is a repeated query param -- same reasoning as
    // list_postings' stage2/3_result filters.
    let mut url =
        reqwest::Url::parse(&format!("{BASE_URL}/roadmap/demand")).map_err(|e| e.to_string())?;
    {
        let mut pairs = url.query_pairs_mut();
        pairs.append_pair("sort", &sort);
        for id in posting_ids.into_iter().flatten() {
            pairs.append_pair("posting_ids", &id.to_string());
        }
    }
    let resp = state
        .client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Could not reach backend: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Backend returned {status}: {detail}"));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct GenerateRoadmapBriefBody<'a> {
    skill_or_language: &'a str,
    trigger_type: &'a str,
    trend_evidence: &'a [i64],
}

#[tauri::command]
async fn generate_roadmap_brief(
    state: tauri::State<'_, AppState>,
    skill_or_language: String,
    trigger_type: String,
    trend_evidence: Vec<i64>,
) -> Result<SkillRoadmapOut, String> {
    let body = GenerateRoadmapBriefBody {
        skill_or_language: &skill_or_language,
        trigger_type: &trigger_type,
        trend_evidence: &trend_evidence,
    };
    request_json(&state.client, reqwest::Method::POST, "/roadmap/brief", &[], Some(&body)).await
}

#[tauri::command]
async fn update_roadmap_status(
    state: tauri::State<'_, AppState>,
    roadmap_id: i64,
    status: String,
) -> Result<SkillRoadmapOut, String> {
    let body = StatusBody { status: &status };
    request_json(
        &state.client,
        reqwest::Method::PATCH,
        &format!("/roadmap/{roadmap_id}/status"),
        &[],
        Some(&body),
    )
    .await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .manage(AppState { client: build_client() })
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            list_postings,
            get_posting,
            set_posting_note,
            similar_postings,
            hydrate_postings,
            star_posting,
            unstar_posting,
            dismiss_posting,
            undismiss_posting,
            analyze_posting,
            submit_stage2_batch,
            submit_stage3_batch,
            triage_pending_preview,
            analyze_pending_preview,
            get_spend_summary,
            apply_stage3_batch,
            apply_stage2_batch,
            list_batches,
            list_companies,
            create_company,
            crawl_company,
            scrape_company,
            refresh_all_companies,
            get_application,
            set_application_status,
            list_applications,
            export_applications_csv,
            get_current_profile,
            list_profile_versions,
            create_profile_version,
            profile_intake_chat,
            parse_resume_file,
            get_intake_chat_history,
            clear_intake_chat_history,
            list_roadmap,
            detect_skill_trends,
            get_readiness_view,
            get_market_demand,
            preview_skill_judgments,
            judge_skills,
            submit_key_skills_refresh,
            generate_roadmap_brief,
            update_roadmap_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

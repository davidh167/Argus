// The only file that holds the API token or makes network requests.
// The content script and popup never see the token and never call fetch()
// directly -- see extension/manifest.json for the permission model.

const API_BASE = "http://127.0.0.1:8000";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Defense in depth: externally_connectable already blocks messages from
  // other extensions/websites at the platform level, but content scripts
  // share a messaging API with everything else in this extension's own
  // pages, so we still verify the sender is this extension.
  if (sender.id !== chrome.runtime.id) {
    return false;
  }

  if (message?.type === "EXTRACT") {
    extractFromTab(message.tabId)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // keep the message channel open for the async sendResponse
  }

  if (message?.type === "SUBMIT") {
    submitCapture(message.payload)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message?.type === "TEST_TOKEN") {
    testToken()
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message?.type === "LOOKUP") {
    lookupByUrl(message.url)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message?.type === "GET_DETAIL") {
    apiGet(`/postings/${message.postingId}`)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message?.type === "GET_APPLICATION") {
    apiGet(`/postings/${message.postingId}/application`)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message?.type === "STAR") {
    apiPost(`/postings/${message.postingId}/star`)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message?.type === "SET_STATUS") {
    apiPost(`/postings/${message.postingId}/application`, { status: message.status })
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  return false;
});

async function extractFromTab(tabId) {
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId },
    files: ["content-script.js"],
  });
  return { ok: true, data: result };
}

async function getToken() {
  const { token } = await chrome.storage.local.get("token");
  return token || null;
}

async function apiRequest(path, { method = "GET", body } = {}) {
  const token = await getToken();
  if (!token) {
    return {
      ok: false,
      error: "No API token set. Open the extension's options page and paste the token.",
    };
  }
  let resp;
  try {
    resp = await fetch(`${API_BASE}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
  } catch (err) {
    return { ok: false, error: `Could not reach ${API_BASE}. Is the backend running? (${err})` };
  }
  if (!resp.ok) {
    const detail = await resp.text().catch(() => "");
    return { ok: false, error: `Backend returned ${resp.status}: ${detail}` };
  }
  const text = await resp.text();
  return { ok: true, data: text ? JSON.parse(text) : null };
}

function apiGet(path) {
  return apiRequest(path);
}

function apiPost(path, body) {
  return apiRequest(path, { method: "POST", body });
}

async function submitCapture(payload) {
  return apiPost("/postings/capture", payload);
}

async function lookupByUrl(url) {
  const result = await apiGet(`/postings?url=${encodeURIComponent(url)}`);
  if (!result.ok) return result;
  return { ok: true, data: result.data[0] ?? null };
}

async function testToken() {
  const token = await getToken();
  if (!token) return { ok: false, error: "No token saved yet." };
  let resp;
  try {
    resp = await fetch(`${API_BASE}/postings`, {
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch (err) {
    return { ok: false, error: `Could not reach ${API_BASE}. Is the backend running? (${err})` };
  }
  if (resp.ok) return { ok: true };
  return { ok: false, error: `Server responded ${resp.status}` };
}

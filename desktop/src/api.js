// Vanilla JS, no npm/bundler -- app.withGlobalTauri in tauri.conf.json
// injects window.__TAURI__ directly instead of requiring the
// @tauri-apps/api npm package.
const { invoke } = window.__TAURI__.core;
const { openUrl } = window.__TAURI__.opener;

const statusEl = document.getElementById("status");

export function setStatus(text, tone = "") {
  statusEl.textContent = text;
  statusEl.className = tone ? `tone-${tone}` : "";
}

export async function call(command, args) {
  try {
    return await invoke(command, args);
  } catch (err) {
    setStatus(`Error: ${err}`, "error");
    throw err;
  }
}

// Routes a posting/apply link through the OS's default browser instead of
// Tauri's own webview -- see the plugin note in views/detail.js for why a
// plain <a target="_blank"> isn't used.
export async function openExternal(url) {
  try {
    await openUrl(url);
  } catch (err) {
    setStatus(`Could not open link: ${err}`, "error");
  }
}

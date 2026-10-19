const tokenInput = document.getElementById("token");
const statusEl = document.getElementById("status");

document.addEventListener("DOMContentLoaded", async () => {
  const { token } = await chrome.storage.local.get("token");
  if (token) tokenInput.value = token;
});

document.getElementById("saveBtn").addEventListener("click", async () => {
  await chrome.storage.local.set({ token: tokenInput.value.trim() });
  statusEl.textContent = "Saved.";
});

document.getElementById("testBtn").addEventListener("click", async () => {
  statusEl.textContent = "Testing...";
  const resp = await chrome.runtime.sendMessage({ type: "TEST_TOKEN" });
  statusEl.textContent = resp.ok ? "Token works." : `Failed: ${resp.error}`;
});

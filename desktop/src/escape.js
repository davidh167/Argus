// HTML-escape helper for every innerHTML template that interpolates data
// from the backend (scraped posting titles, LLM output, profile text).
const ENTITIES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ENTITIES[c]);
}

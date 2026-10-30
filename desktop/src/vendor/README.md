# Vendored frontend libraries

The frontend has no bundler, so third-party code is vendored as plain ES modules.
It lives under `desktop/src/` (not `desktop/vendor/`) because Tauri's `frontendDist` is `../src`
and only that directory is served. CSP stays `script-src 'self'` (via `default-src`); nothing
here needs `unsafe-inline` or `unsafe-eval`.

## lit-html.js

- Library: lit-html 3.3.3 (Google / Lit project), BSD-3-Clause
- Source: https://registry.npmjs.org/lit-html/-/lit-html-3.3.3.tgz, file `package/lit-html.js`
  (production build, single self-contained file, no imports)
- Tarball SHA-256: `ae004138a6327c362fee8e48db2a8ee9619a6ca3014c802857395e6ac447a0eb`
- `lit-html.js` SHA-256: `b878e7f95dec8a9b6e9b217faca6b6a11bad82ebc44c0caa77419ed83edd81f2`
- Why: its `html` tagged template escapes every `${}` value by context (text, attribute),
  replacing the former hand-written `esc()` helper. Rule: never build HTML with string
  concatenation or `innerHTML = \`...${data}\``; use `render(html\`...\`, el)`. Nested fragments
  must be `html\`\`` results too (plain strings are escaped as text).

To upgrade: download the new tarball, replace the file, update version and both hashes here.

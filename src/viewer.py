"""Dataset viewer — serves an HTML page for manual inspection of JSONL datasets.

Usage::

    uv run python -m src.viewer          # opens browser, lists data/*.jsonl
    uv run python -m src.viewer --port 8765
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse, unquote


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class _Handler(SimpleHTTPRequestHandler):
    """Serve static files from the project root and API endpoints."""
    upload_form: ClassVar[str] = ""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # Serve the viewer page.
        if path == "/" or path == "/index.html":
            self._serve_html()
            return

        # List available dataset files.
        if path == "/api/datasets":
            self._serve_json(self._list_datasets())
            return

        # Serve actual JSONL content.
        if path.startswith("/data/") and path.endswith(".jsonl"):
            self._serve_file(str(PROJECT_ROOT / path.lstrip("/")))
            return

        # Fall back to static file serving from project root.
        self._serve_file(str(PROJECT_ROOT / path.lstrip("/")))

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(_HTML_PAGE.encode("utf-8"))

    def _serve_json(self, obj):
        data = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, abs_path: str):
        try:
            with open(abs_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            ct = "application/json" if abs_path.endswith(".jsonl") else "text/plain"
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            self.send_error(404)

    def _list_datasets(self):
        if not DATA_DIR.exists():
            return []
        datasets = []
        for p in sorted(DATA_DIR.iterdir()):
            if p.is_dir():
                for f in sorted(p.glob("*.jsonl")):
                    datasets.append({
                        "name": f"{p.name}/{f.name}",
                        "path": f"/data/{p.name}/{f.name}",
                        "size": f.stat().st_size,
                    })
            elif p.suffix == ".jsonl":
                datasets.append({
                    "name": p.name,
                    "path": f"/data/{p.name}",
                    "size": p.stat().st_size,
                })
        return datasets


# --------------------------------------------------------------------------- #
# HTML page (single-file, all JS/CSS inlined or from CDN)
# --------------------------------------------------------------------------- #

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Viewer</title>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/github.min.css">
<style>
  :root {
    --bg: #fafafa;
    --fg: #222;
    --border: #d0d7de;
    --diff-removed: #ffebe9;
    --diff-added: #dafbe1;
    --diff-removed-word: #ff818266;
    --diff-added-word: #4ac26b66;
    --shadow: 0 1px 3px rgba(0,0,0,.08);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--fg); padding: 20px; }
  h1 { font-size: 20px; margin-bottom: 16px; }
  #toolbar { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
  #toolbar select, #toolbar button, #toolbar input { padding: 6px 10px; border: 1px solid var(--border);
    border-radius: 6px; font: inherit; background: #fff; cursor: pointer; }
  #toolbar button:hover { background: #f0f0f0; }
  #stats { font-size: 12px; color: #666; margin-left: 8px; }
  .sample { background: #fff; border: 1px solid var(--border); border-radius: 8px;
            margin-bottom: 12px; box-shadow: var(--shadow); overflow: hidden; }
  .sample-header { padding: 8px 12px; background: #f6f8fa; border-bottom: 1px solid var(--border);
                   font-size: 12px; color: #57606a; display: flex; justify-content: space-between; }
  .sample-header .edits { font-family: monospace; font-size: 11px; }
  .panes { display: grid; grid-template-columns: 1fr 1fr; }
  .pane { overflow: auto; max-height: 500px; }
  .pane+.pane { border-left: 1px solid var(--border); }
  .pane-label { font-size: 11px; font-weight: 600; padding: 4px 12px; background: #f6f8fa;
                border-bottom: 1px solid var(--border); color: #57606a; position: sticky; top: 0; z-index: 1; }
  .pane-label.fixed { color: #1a7f37; }
  .pane-label.corrupted { color: #cf222e; }
  .pane code { display: block; padding: 8px 12px; }
  .pane pre { margin: 0; }
  .no-samples { text-align: center; padding: 40px; color: #888; }
  .loading { text-align: center; padding: 40px; }
  .search { display: flex; gap: 4px; }
  .search input { width: 180px; }
  /* Diff highlighting */
  .hl-diff-removed { background: var(--diff-removed); }
  .hl-diff-added   { background: var(--diff-added); }
</style>
</head>
<body>
<h1>Dataset Viewer</h1>
<div id="toolbar">
  <select id="ds-select"><option value="">-- Select dataset --</option></select>
  <button id="btn-load" disabled>Load</button>
  <span class="search">
    <input id="search-input" type="text" placeholder="Filter by name..." disabled>
    <button id="btn-filter" disabled>Filter</button>
    <button id="btn-clear-filter" disabled>Show all</button>
  </span>
  <span id="stats"></span>
</div>
<div id="samples"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/diff-match-patch@1.0.5/index.js"></script>
<script>
/* ---- globals ---- */
let allSamples = [];
let currentDsPath = null;

/* ---- init ---- */
(async () => {
  const resp = await fetch("/api/datasets");
  const datasets = await resp.json();
  const sel = document.getElementById("ds-select");
  for (const ds of datasets) {
    const opt = document.createElement("option");
    opt.value = ds.path;
    opt.textContent = ds.name + " (" + formatSize(ds.size) + ")";
    sel.appendChild(opt);
  }
  sel.onchange = () => {
    document.getElementById("btn-load").disabled = !sel.value;
  };
  document.getElementById("btn-load").onclick = loadDataset;
  document.getElementById("btn-filter").onclick = filterSamples;
  document.getElementById("btn-clear-filter").onclick = clearFilter;
  document.getElementById("search-input").onkeydown = (e) => {
    if (e.key === "Enter") filterSamples();
  };

  // Re-highlight any <code> in the page.
  document.querySelectorAll("code.language-python").forEach(el => hljs.highlightElement(el));
})();

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + " KB";
  return (bytes/(1024*1024)).toFixed(1) + " MB";
}

/* ---- load JSONL ---- */
async function loadDataset() {
  const sel = document.getElementById("ds-select");
  if (!sel.value) return;
  currentDsPath = sel.value;
  document.getElementById("stats").textContent = "Loading...";
  document.getElementById("samples").innerHTML = '<div class="loading">Loading dataset...</div>';

  const resp = await fetch(currentDsPath);
  const text = await resp.text();
  const lines = text.trim().split("\n");
  allSamples = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    try { allSamples.push(JSON.parse(line)); }
    catch (e) { console.warn("bad line:", line.substring(0, 80)); }
  }
  renderSamples(allSamples);
  document.getElementById("search-input").disabled = false;
  document.getElementById("btn-filter").disabled = false;
  document.getElementById("btn-clear-filter").disabled = false;
}

/* ---- filtering ---- */
function filterSamples() {
  const q = document.getElementById("search-input").value.trim().toLowerCase();
  if (!q) { renderSamples(allSamples); return; }
  const filtered = allSamples.filter(s => {
    const searchStr = [s.fixed || "", s.code || "", (s.edits || []).map(e => e.original_name + " " + e.corrupted_name).join(" ")].join(" ").toLowerCase();
    return searchStr.includes(q);
  });
  renderSamples(filtered);
}

function clearFilter() {
  document.getElementById("search-input").value = "";
  renderSamples(allSamples);
}

/* ---- render ---- */
function renderSamples(samples) {
  const container = document.getElementById("samples");
  document.getElementById("stats").textContent =
    samples.length + " / " + allSamples.length + " samples";
  if (samples.length === 0) {
    container.innerHTML = '<div class="no-samples">No samples to show.</div>';
    return;
  }
  let html = "";
  for (let i = 0; i < samples.length; i++) {
    html += renderSample(samples[i], i);
  }
  container.innerHTML = html;
  // Apply highlight.js to all code blocks.
  document.querySelectorAll("code.language-python").forEach(el => hljs.highlightElement(el));
}

function renderSample(s, idx) {
  const original = s.fixed || "";
  const corrupted = s.code || "";
  const hasErrors = s.has_errors;
  const edits = s.edits || [];

  // Compute line diffs between original and corrupted.
  const [diffOrig, diffCorr] = sideBySideWithWordDiff(original, corrupted);

  let editInfo = "";
  if (edits.length > 0) {
    editInfo = edits.map(e =>
      `<span style="color:#cf222e">${esc(e.corrupted_name)}</span> → <span style="color:#1a7f37">${esc(e.original_name)}</span>`
    ).join(", ");
  }

  return `
  <div class="sample">
    <div class="sample-header">
      <span>Sample #${idx + 1} ${hasErrors ? "⛔" : "✅"}</span>
      <span class="edits">${editInfo}</span>
    </div>
    <div class="panes">
      <div class="pane">
        <div class="pane-label fixed">✦ Original (ground truth)</div>
        <pre><code class="language-python">${escapePreservingHighlights(diffOrig)}</code></pre>
      </div>
      <div class="pane">
        <div class="pane-label corrupted">✧ Corrupted (with typos)</div>
        <pre><code class="language-python">${escapePreservingHighlights(diffCorr)}</code></pre>
      </div>
    </div>
  </div>`;
}

/* ---- diff helpers ---- */
const dmp = new diff_match_patch();

/**
 * Compute side-by-side word-diff markup.
 * Both strings are split into lines and each line pair is word-diffed.
 * Returns [leftHtml, rightHtml] with <span class="hl-diff-*"> wrappers.
 */
function sideBySideWithWordDiff(original, corrupted) {
  const origLines = original.split("\n");
  const corrLines = corrupted.split("\n");
  const maxLines = Math.max(origLines.length, corrLines.length);

  const leftOut = [];
  const rightOut = [];

  for (let i = 0; i < maxLines; i++) {
    const ol = i < origLines.length ? origLines[i] : "";
    const cl = i < corrLines.length ? corrLines[i] : "";

    // If lines are identical, no highlighting.
    if (ol === cl) {
      leftOut.push(esc(ol));
      rightOut.push(esc(cl));
      continue;
    }

    // Do a character diff on the two lines.
    const diffs = dmp.diff_main(ol, cl);
    dmp.diff_cleanupSemantic(diffs);

    let leftLine = "";
    let rightLine = "";

    for (const [op, text] of diffs) {
      const escaped = esc(text);
      if (op === 0) {
        leftLine += escaped;
        rightLine += escaped;
      } else if (op === -1) {
        leftLine += `<span class="hl-diff-removed">${escaped}</span>`;
      } else { // op === 1
        rightLine += `<span class="hl-diff-added">${escaped}</span>`;
      }
    }

    leftOut.push(leftLine);
    rightOut.push(rightLine);
  }

  return [leftOut.join("\n"), rightOut.join("\n")];
}

/* ---- utilities ---- */
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Escapes HTML but preserves <span...> tags we've already added.
 * Used when embedding pre-highlighted diffs into <code> blocks.
 */
function escapePreservingHighlights(html) {
  // Temporarily protect <span> tags, escape the rest, then restore.
  const spans = [];
  const safe = html.replace(/<span[^>]*>.*?<\/span>/gs, (m) => {
    spans.push(m);
    return "\u0000SPAN" + (spans.length - 1) + "\u0000";
  });
  // No need to escape safe since all text was already escaped by sideBySideWithWordDiff.
  return safe.replace(/\u0000SPAN(\d+)\u0000/g, (_, i) => spans[+i]);
}
</script>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Dataset viewer server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    server = HTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"Serving dataset viewer at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
# %%

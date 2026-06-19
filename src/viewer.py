"""Dataset viewer — serves an HTML page for manual inspection of JSONL datasets
and interactive model evaluation.

Usage::

    uv run python -m src.viewer          # opens browser, lists data/*.jsonl
    uv run python -m src.viewer --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional
from urllib.parse import urlparse, unquote


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TEMPLATE_PATH = Path(__file__).resolve().parent / "viewer_template.html"

# Support both `python -m src.viewer` and `python src/viewer.py`.
try:
    from .models import MODEL_REGISTRY, make_model
    from .models.llm_api import _load_presets
    from .harness import _iter_jsonl, _process_sample, compute_metrics
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models import MODEL_REGISTRY, make_model  # type: ignore[no-redef]
    from models.llm_api import _load_presets  # type: ignore[no-redef]
    from harness import _iter_jsonl, _process_sample, compute_metrics  # type: ignore[no-redef]

# Load the HTML template at import time.
_HTML_PAGE = TEMPLATE_PATH.read_text(encoding="utf-8")

# Default preset config path.
_DEFAULT_PRESET_CONFIG = str(PROJECT_ROOT / "config" / "llm_presets.toml")


# --------------------------------------------------------------------------- #
# Background eval state (shared across requests).
# --------------------------------------------------------------------------- #

# Map run_id -> dict with keys: model_name, dataset, total, results,
# done, metrics, error, last_idx_sent.
_running_evals: Dict[str, Dict[str, Any]] = {}
_eval_lock = threading.Lock()


def _eval_runner_thread(
    run_id: str,
    dataset_path: str,
    model_name: str,
    max_samples: Optional[int],
    random_sample: bool,
    gector_model_dir: Optional[str] = None,
    preset: Optional[str] = None,
    llm_config: str = "config/llm_presets.toml",
) -> None:
    """Run evaluation in a background thread, updating ``_running_evals``.

    When ``random_sample`` is True and ``max_samples`` is set, samples are
    shuffled before selection rather than taking the first N.

    Parameters
    ----------
    gector_model_dir:
        Path to a GECToR checkpoint directory.  Required when
        ``model_name == "gector"``.
    """
    try:
        # Load and filter error samples.
        all_samples = list(_iter_jsonl(dataset_path))
        error_samples = [
            (s, i) for i, s in enumerate(all_samples) if s.get("has_errors", False)
        ]

        total_available = len(error_samples)
        if max_samples is not None and max_samples < total_available:
            if random_sample:
                rng = random.Random(42)
                rng.shuffle(error_samples)
            error_samples = error_samples[:max_samples]

        with _eval_lock:
            _running_evals[run_id]["total"] = len(error_samples)

        # Build model kwargs — GECToR needs model_dir.
        model_kwargs: dict = {}
        if model_name == "gector":
            if not gector_model_dir:
                raise ValueError(
                    "GECToR requires a checkpoint directory.  "
                    "Enter the path in the 'GECToR model dir' field."
                )
            model_kwargs["model_dir"] = gector_model_dir
        if model_name == "llm_api" and preset:
            model_kwargs["preset"] = preset
            model_kwargs["config_path"] = llm_config

        model = make_model(model_name, **model_kwargs)
        results: List[Any] = []

        t_start = time.perf_counter()

        max_parallel = getattr(model, "max_parallel_requests", 1)

        if max_parallel > 1:
            # Thread-pool path — concurrent HTTP calls for cloud APIs.
            with ThreadPoolExecutor(max_workers=max_parallel) as ex:
                futures = {
                    ex.submit(_process_sample, s, oi, model): oi
                    for s, oi in error_samples
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        orig_idx = futures[future]
                        print(f"[viewer] Sample {orig_idx} failed: {exc}")
                        continue
                    if result is not None:
                        results.append(result)
                    with _eval_lock:
                        _running_evals[run_id]["results"] = list(results)
            # Restore original ordering after thread-pool finish.
            results.sort(key=lambda r: r.sample_index)
            with _eval_lock:
                _running_evals[run_id]["results"] = list(results)
        else:
            # Serial path — local models or non-LLM.
            for idx, (sample, orig_idx) in enumerate(error_samples):
                result = _process_sample(sample, orig_idx, model)
                if result is not None:
                    results.append(result)
                with _eval_lock:
                    _running_evals[run_id]["results"] = list(results)

        # Compute final metrics.
        metrics = compute_metrics(results)

        elapsed = time.perf_counter() - t_start
        n = len(results)
        total_kb = sum(len(r.corrupted_code.encode("utf-8")) for r in results) / 1024.0

        with _eval_lock:
            _running_evals[run_id]["results"] = list(results)
            _running_evals[run_id]["metrics"] = {
                "total_samples": metrics.total_samples,
                "exact_match_rate": metrics.exact_match_rate,
                "identifier_precision": metrics.identifier_precision,
                "identifier_recall": metrics.identifier_recall,
                "identifier_f1": metrics.identifier_f1,
                "avg_normalized_edit_distance": metrics.avg_normalized_edit_distance,
                "total_time_seconds": elapsed,
                "avg_time_per_sample_seconds": elapsed / n if n else 0.0,
                "avg_time_per_kb_seconds": elapsed / total_kb if total_kb else 0.0,
            }
            _running_evals[run_id]["done"] = True

    except Exception as exc:
        with _eval_lock:
            _running_evals[run_id]["done"] = True
            _running_evals[run_id]["error"] = str(exc)


# --------------------------------------------------------------------------- #
# HTTP handler.
# --------------------------------------------------------------------------- #


class _Handler(SimpleHTTPRequestHandler):
    """Serve static files from the project root and API endpoints."""

    # JSONL files larger than this are served truncated at a newline boundary.
    MAX_JSONL_BYTES: ClassVar[int] = 10 * 1024 * 1024  # 10 MiB

    # ---- Logging suppression -------------------------------------------------

    def log_request(self, code='-', size='-'):
        """Suppress log lines for successful requests (1xx/2xx/3xx).

        Only 4xx (client error) and 5xx (server error) status codes are logged.
        """
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)

    # ---- GET -----------------------------------------------------------------

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

        # List available models.
        if path == "/api/models":
            self._serve_json(self._list_models())
            return

        # List available GECToR checkpoint directories.
        if path == "/api/gector_checkpoints":
            self._serve_json(self._list_gector_checkpoints())
            return

        # List available LLM presets.
        if path == "/api/presets":
            self._serve_json(self._list_presets())
            return

        # Poll eval progress.
        if path.startswith("/api/eval/status/"):
            run_id = path.split("/")[-1]
            self._serve_eval_status(run_id)
            return

        # Serve actual JSONL content (with size-aware truncation).
        if path.startswith("/data/") and path.endswith(".jsonl"):
            abs_path = str(PROJECT_ROOT / path.lstrip("/"))
            self._serve_jsonl(abs_path)
            return

        # Fall back to static file serving from project root.
        self._serve_file(str(PROJECT_ROOT / path.lstrip("/")))

    # ---- POST ----------------------------------------------------------------

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/eval/start":
            self._handle_eval_start()
            return

        self.send_error(404)

    # ---- HTML ----------------------------------------------------------------

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(_HTML_PAGE.encode("utf-8"))

    # ---- JSON helpers --------------------------------------------------------

    def _serve_json(self, obj):
        data = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- Data listings -------------------------------------------------------

    def _list_datasets(self):
        """Return dataset JSONL files and eval result JSON files."""
        entries: List[Dict[str, object]] = []
        if not DATA_DIR.exists():
            return entries
        for p in sorted(DATA_DIR.iterdir()):
            if p.is_dir():
                for f in sorted(p.glob("*.jsonl")):
                    entries.append({
                        "name": f"{p.name}/{f.name}",
                        "path": f"/data/{p.name}/{f.name}",
                        "size": f.stat().st_size,
                        "kind": "jsonl",
                    })
            elif p.suffix == ".jsonl":
                entries.append({
                    "name": p.name,
                    "path": f"/data/{p.name}",
                    "size": p.stat().st_size,
                    "kind": "jsonl",
                })
            elif p.suffix == ".json":
                # Detect eval result files: JSON with "metrics" key.
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        head = f.read(512)
                    if '"metrics"' in head and '"per_sample"' in head:
                        entries.append({
                            "name": f"📊 {p.name}",
                            "path": f"/data/{p.name}",
                            "size": p.stat().st_size,
                            "kind": "eval",
                        })
                except Exception:
                    pass
        return entries

    def _list_models(self):
        """Return available model names from the registry."""
        return list(MODEL_REGISTRY.keys())

    def _list_gector_checkpoints(self) -> List[Dict[str, str]]:
        """Scan ``models/`` for directories that look like GECToR checkpoints.

        A directory is considered a checkpoint if it contains ``vocab.txt``
        (written by every :meth:`~src.gector.model.GECToRModel.save_pretrained`
        call).
        """
        checkpoints: List[Dict[str, str]] = []
        if not MODELS_DIR.exists():
            return checkpoints
        for entry in sorted(MODELS_DIR.rglob("vocab.txt")):
            ckpt_dir = entry.parent
            # Use a path relative to the project root for display.
            try:
                rel = str(ckpt_dir.relative_to(PROJECT_ROOT))
            except ValueError:
                rel = str(ckpt_dir)
            checkpoints.append({"label": rel, "path": rel})
        return checkpoints
    def _list_presets(self):
        """Return available LLM preset names from the config file."""
        try:
            presets = _load_presets(_DEFAULT_PRESET_CONFIG)
            return sorted(presets.keys())
        except Exception:
            return []

    # ---- JSONL serving -------------------------------------------------------

    def _serve_jsonl(self, abs_path: str):
        """Serve a JSONL file, truncating if larger than MAX_JSONL_BYTES."""
        try:
            file_size = os.path.getsize(abs_path)
            with open(abs_path, "rb") as f:
                if file_size <= self.MAX_JSONL_BYTES:
                    data = f.read()
                    truncated = False
                else:
                    data = f.read(self.MAX_JSONL_BYTES)
                    nl_idx = data.rfind(b"\n")
                    if nl_idx >= 0:
                        data = data[: nl_idx + 1]
                    truncated = True

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            if truncated:
                self.send_header("X-Dataset-Truncated", "true")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            self.send_error(404)

    # ---- Static file fallback ------------------------------------------------

    def _serve_file(self, abs_path: str):
        """Serve an arbitrary static file (non-JSONL fallback)."""
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

    # ---- Eval start (POST) ---------------------------------------------------

    def _handle_eval_start(self):
        """Parse POST body and launch a background eval run."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            params = json.loads(body)
        except Exception:
            self.send_error(400, "Invalid JSON body")
            return

        dataset_path = params.get("dataset", "").strip()
        model_name = params.get("model", "").strip()
        max_samples_raw = params.get("max_samples", None)
        random_sample = params.get("random_sample", True)
        gector_model_dir = params.get("gector_model_dir", "").strip() or None
        preset = params.get("preset", None)
        llm_config = params.get("llm_config", _DEFAULT_PRESET_CONFIG)

        # Validate dataset path (must be within DATA_DIR).
        if not dataset_path:
            self.send_error(400, "Missing 'dataset' parameter")
            return
        abs_dataset = str(PROJECT_ROOT / dataset_path.lstrip("/"))
        if not os.path.isfile(abs_dataset):
            self.send_error(404, f"Dataset not found: {dataset_path}")
            return

        # Validate model.
        if model_name not in MODEL_REGISTRY:
            self.send_error(400, f"Unknown model: {model_name}")
            return

        # GECToR requires a checkpoint directory.
        if model_name == "gector" and not gector_model_dir:
            self.send_error(400, "GECToR requires 'gector_model_dir' parameter")
            return
        # Validate preset if model is llm_api.
        if isinstance(preset, str):
            preset = preset.strip()
        if model_name == "llm_api" and not preset:
            # Auto-select first preset.
            try:
                presets = _load_presets(llm_config)
                if presets:
                    preset = sorted(presets.keys())[0]
            except Exception:
                pass
            if not preset:
                self.send_error(400, "Missing 'preset' parameter for llm_api model")
                return

        # Parse max_samples.
        max_samples: Optional[int] = None
        if max_samples_raw is not None:
            try:
                max_samples = int(max_samples_raw)
                if max_samples < 1:
                    max_samples = None
            except (ValueError, TypeError):
                pass

        # Start the eval run.
        run_id = uuid.uuid4().hex[:12]
        with _eval_lock:
            _running_evals[run_id] = {
                "model_name": model_name,
                "dataset": dataset_path,
                "total": 0,
                "results": [],
                "done": False,
                "metrics": None,
                "error": None,
            }

        thread = threading.Thread(
            target=_eval_runner_thread,
            args=(run_id, abs_dataset, model_name, max_samples, random_sample,
                  gector_model_dir,                  preset, llm_config),
            daemon=True,
        )
        thread.start()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"run_id": run_id}).encode("utf-8"))

    # ---- Eval status (GET) ---------------------------------------------------

    def _serve_eval_status(self, run_id: str):
        """Return eval progress + new results since last poll.

        Query param ``?since=N`` controls the cursor (default 0).
        """
        with _eval_lock:
            state = _running_evals.get(run_id)

        if state is None:
            self.send_error(404, "Unknown run_id")
            return

        # Parse the ?since= cursor.
        parsed = urlparse(self.path)
        qs = parsed.query
        since = 0
        if qs:
            from urllib.parse import parse_qs
            params = parse_qs(qs)
            try:
                since = int(params.get("since", ["0"])[0])
            except (ValueError, TypeError):
                since = 0

        with _eval_lock:
            all_results = list(state["results"])
            done = state["done"]
            error = state["error"]
            metrics = state["metrics"]
            total = state["total"]

        new_results = all_results[since:]
        last_idx = since + len(new_results)

        payload = {
            "run_id": run_id,
            "done": done,
            "total": total,
            "processed": len(all_results),
            "new_results": [
                self._serialize_sample_result(r) for r in new_results
            ],
            "last_idx_sent": last_idx,
            "error": error,
        }
        if done:
            payload["metrics"] = metrics

        self._serve_json(payload)

    @staticmethod
    def _serialize_sample_result(result) -> Dict[str, object]:
        """Convert :class:`SampleResult` to a JSON-safe dict."""
        return {
            "index": result.sample_index,
            "exact_match": result.exact_match,
            "corrupted_code": result.corrupted_code,
            "predicted_code": result.predicted_code,
            "ground_truth_code": result.ground_truth_code,
            "model_fixes": result.model_fixes,
            "gt_original_names": result.gt_original_names,
            "gt_corrupted_names": result.gt_corrupted_names,
        }


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

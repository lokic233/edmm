#!/usr/bin/env python3
"""
EDMM Core Validation Benchmark (v2 - direct transformers)
Bypasses vLLM entirely to avoid V1 engine IPC hangs on multi-tenant H100 nodes.
Uses HuggingFace transformers with manual KV cache management to demonstrate:
1. The "Orchestration Bubble" — GPU drops to 0% utilization during tool execution loops
2. The "Radix Discontinuity Trap" — mid-prompt dynamic injection breaks prefix caching
"""

import gc
import os
import subprocess
import time
import uuid
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

NUM_ITERATIONS: int = 20
BASE_CONTEXT_TOKENS: int = 16384
SUFFIX_TOKENS: int = 2048
SLEEP_DURATION: float = 2.0
MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_MODEL_FALLBACKS: Tuple[str, ...] = (
    "/tmp/qwen7b",
    "/tmp/Qwen2.5-7B-Instruct",
    "/tmp/qwen25-7b-instruct",
)


def repeat_to_token_floor(tokenizer: Any, block: str, num_tokens: int) -> str:
    tokens_per_block = max(1, len(tokenizer.encode(block, add_special_tokens=False)))
    repetitions = (num_tokens // tokens_per_block) + 2
    text = block * repetitions
    while len(tokenizer.encode(text, add_special_tokens=False)) < num_tokens:
        text += block
    return text


_STATIC_CONTEXT_BLOCK = """\
class HTTPRequestHandler:
    \"\"\"Base handler for incoming HTTP requests with middleware support.\"\"\"

    _middleware_chain: list = []
    _error_handlers: dict = {}
    _route_cache: dict = {}

    def __init__(self, app, request, response_class=None):
        self.app = app
        self.request = request
        self.response_class = response_class or JSONResponse
        self._state = {}
        self._is_committed = False

    def dispatch(self, method: str, path: str, **kwargs):
        handler = self._resolve_route(method, path)
        if handler is None:
            raise RouteNotFoundError(f"No route for {method} {path}")
        try:
            for mw in self._middleware_chain:
                mw.process_request(self.request)
            result = handler(self.request, **kwargs)
            if not self._is_committed:
                self._is_committed = True
                return self.response_class(data=result, status=200)
        except ValidationError as exc:
            return self._handle_error(exc, status=422)
        except PermissionDeniedError as exc:
            return self._handle_error(exc, status=403)
        except Exception as exc:
            logger.exception("Unhandled error in %s %s", method, path)
            return self._handle_error(exc, status=500)

    def _resolve_route(self, method, path):
        cache_key = f"{method}:{path}"
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]
        for pattern, view_fn, allowed_methods in self.app.url_rules:
            if method in allowed_methods and pattern.match(path):
                self._route_cache[cache_key] = view_fn
                return view_fn
        return None

    def _handle_error(self, exc, status):
        handler = self._error_handlers.get(type(exc), self._default_error)
        return self.response_class(data={"error": str(exc)}, status=status)

    @staticmethod
    def _default_error(exc):
        return {"error": "Internal server error", "type": type(exc).__name__}


class QuerySetManager:
    \"\"\"Lazy-evaluated queryset with chainable filter, exclude, and prefetch.\"\"\"

    def __init__(self, model_class, using="default"):
        self._model = model_class
        self._db = using
        self._filters = []
        self._excludes = []
        self._prefetch = []
        self._order_by = []
        self._limit = None
        self._offset = 0
        self._cache = None

    def filter(self, **kwargs):
        clone = self._clone()
        clone._filters.append(kwargs)
        return clone

    def exclude(self, **kwargs):
        clone = self._clone()
        clone._excludes.append(kwargs)
        return clone

    def prefetch_related(self, *fields):
        clone = self._clone()
        clone._prefetch.extend(fields)
        return clone

    def order_by(self, *fields):
        clone = self._clone()
        clone._order_by = list(fields)
        return clone

    def _clone(self):
        qs = QuerySetManager(self._model, self._db)
        qs._filters = list(self._filters)
        qs._excludes = list(self._excludes)
        qs._prefetch = list(self._prefetch)
        qs._order_by = list(self._order_by)
        return qs

    def _build_query(self):
        sql_parts = [f"SELECT * FROM {self._model._meta.db_table}"]
        where_clauses = []
        for f in self._filters:
            for key, val in f.items():
                col, op = self._parse_lookup(key)
                where_clauses.append(f"{col} {op} %s")
        for e in self._excludes:
            for key, val in e.items():
                col, op = self._parse_lookup(key)
                where_clauses.append(f"NOT ({col} {op} %s)")
        if where_clauses:
            sql_parts.append("WHERE " + " AND ".join(where_clauses))
        if self._order_by:
            cols = ", ".join(
                f"{c.lstrip('-')} {'DESC' if c.startswith('-') else 'ASC'}"
                for c in self._order_by
            )
            sql_parts.append(f"ORDER BY {cols}")
        if self._limit is not None:
            sql_parts.append(f"LIMIT {self._limit}")
        if self._offset:
            sql_parts.append(f"OFFSET {self._offset}")
        return " ".join(sql_parts)

    @staticmethod
    def _parse_lookup(key):
        if "__" in key:
            field, lookup = key.rsplit("__", 1)
            ops = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<",
                   "exact": "=", "contains": "LIKE", "in": "IN"}
            return field, ops.get(lookup, "=")
        return key, "="

    def evaluate(self):
        if self._cache is None:
            conn = get_connection(self._db)
            query = self._build_query()
            self._cache = conn.execute(query).fetchall()
            if self._prefetch:
                self._do_prefetch(self._cache)
        return self._cache

    def _do_prefetch(self, results):
        for field_name in self._prefetch:
            related_model = getattr(self._model, field_name).related_model
            ids = [getattr(r, f"{field_name}_id") for r in results]
            related_qs = QuerySetManager(related_model).filter(id__in=ids)
            related_map = {r.id: r for r in related_qs.evaluate()}
            for r in results:
                setattr(r, f"_prefetched_{field_name}",
                        related_map.get(getattr(r, f"{field_name}_id")))


class MigrationRunner:
    \"\"\"Applies and rolls back database schema migrations in dependency order.\"\"\"

    def __init__(self, connection, migration_dir="./migrations"):
        self.connection = connection
        self.migration_dir = migration_dir
        self._applied = set()
        self._available = {}
        self._graph = {}

    def detect_migrations(self):
        import importlib, pathlib
        for path in sorted(pathlib.Path(self.migration_dir).glob("*.py")):
            mod = importlib.import_module(f"migrations.{path.stem}")
            self._available[path.stem] = mod
            deps = getattr(mod, "dependencies", [])
            self._graph[path.stem] = deps

    def get_applied(self):
        try:
            rows = self.connection.execute(
                "SELECT name FROM _migration_history ORDER BY applied_at"
            ).fetchall()
            self._applied = {r[0] for r in rows}
        except Exception:
            self.connection.execute(
                "CREATE TABLE _migration_history "
                "(name TEXT PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            self._applied = set()

    def apply_pending(self):
        order = self._topological_sort()
        for name in order:
            if name not in self._applied:
                mod = self._available[name]
                print(f"Applying migration: {name}")
                mod.upgrade(self.connection)
                self.connection.execute(
                    "INSERT INTO _migration_history (name) VALUES (%s)", (name,)
                )

    def rollback(self, target):
        order = list(reversed(self._topological_sort()))
        for name in order:
            if name in self._applied and name != target:
                mod = self._available[name]
                if hasattr(mod, "downgrade"):
                    print(f"Rolling back: {name}")
                    mod.downgrade(self.connection)
                    self.connection.execute(
                        "DELETE FROM _migration_history WHERE name = %s", (name,)
                    )
            if name == target:
                break

    def _topological_sort(self):
        visited, result = set(), []
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in self._graph.get(node, []):
                visit(dep)
            result.append(node)
        for node in self._graph:
            visit(node)
        return result
"""


def generate_static_context(tokenizer: Any, num_tokens: int) -> str:
    return repeat_to_token_floor(tokenizer, _STATIC_CONTEXT_BLOCK, num_tokens)


_SUFFIX_BLOCK = (
    "You are a senior software engineer performing an agentic code review. "
    "Given the repository source files, execution logs, and error traces loaded above, "
    "identify the root cause of the failing test suite. Produce a minimal unified diff "
    "(git diff format) that fixes the regression without altering the public API surface. "
    "Verify that your patch handles all edge cases visible in the stack trace, including "
    "null pointer dereferences, off-by-one index errors, and unclosed resource handles. "
    "Ensure backward compatibility with callers that rely on the previous return type. "
    "After generating the patch, list each modified file with a one-line rationale. "
)


def generate_suffix_block(tokenizer: Any, num_tokens: int) -> str:
    return repeat_to_token_floor(tokenizer, _SUFFIX_BLOCK, num_tokens)


def generate_dynamic_payload() -> str:
    ts = time.time_ns()
    uid = uuid.uuid4()
    return (
        f"Traceback (most recent call last):\n"
        f'  File "/srv/app/views/api.py", line 247, in dispatch_request\n'
        f"    result = handler(request, **bound_args)\n"
        f'  File "/srv/app/views/api.py", line 389, in update_resource\n'
        f"    validated = schema.load(request.json, partial=True)\n"
        f'  File "/srv/app/lib/schemas.py", line 112, in load\n'
        f"    self._run_validators(data)\n"
        f'  File "/srv/app/lib/schemas.py", line 78, in _run_validators\n'
        f"    raise ValidationError(errors)\n"
        f"marshmallow.exceptions.ValidationError: {{'field_config': ['Unknown field.']}}\n"
        f"\n"
        f"[tool_call_id={uid}] returned at {ts}\n"
        f"exit_code=1 runtime_ms=342 retries=0\n"
    )


def get_gpu_utilization(gpu_index: int) -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


def resolve_model_name() -> str:
    override = os.environ.get("EDMM_MODEL_NAME")
    if override:
        return override

    for path in LOCAL_MODEL_FALLBACKS:
        if os.path.isdir(path):
            return path

    return MODEL_NAME


def clone_kv_cache(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None

    if isinstance(past_key_values, DynamicCache):
        cloned_layers = tuple(
            (
                layer.keys.clone() if layer.keys is not None else None,
                layer.values.clone() if layer.values is not None else None,
            )
            for layer in past_key_values.layers
        )
        return DynamicCache(ddp_cache_data=cloned_layers)

    if isinstance(past_key_values, tuple):
        return tuple(clone_kv_cache(layer) for layer in past_key_values)

    if isinstance(past_key_values, list):
        return [clone_kv_cache(layer) for layer in past_key_values]

    if torch.is_tensor(past_key_values):
        return past_key_values.clone()

    if hasattr(past_key_values, "clone"):
        return past_key_values.clone()

    raise TypeError(f"Unsupported KV cache type: {type(past_key_values)!r}")


def tokenize_to_device(tokenizer: Any, text: str, device: str) -> torch.Tensor:
    return tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(
        device
    )


def build_benchmark_inputs(tokenizer: Any, device: str) -> Dict[str, Any]:
    base_text = generate_static_context(tokenizer, BASE_CONTEXT_TOKENS)
    suffix_text = generate_suffix_block(tokenizer, SUFFIX_TOKENS)
    query = "Summarize the key findings from the above context in one sentence."

    # Slice tensors after tokenization so all groups use identical measured
    # workloads regardless of tokenizer boundary merges.
    base_ids = tokenize_to_device(tokenizer, base_text, device)
    suffix_block_ids = tokenize_to_device(tokenizer, f"\n{suffix_text}", device)
    query_ids = tokenize_to_device(tokenizer, f"\n{query}", device)

    if base_ids.shape[1] < BASE_CONTEXT_TOKENS:
        raise RuntimeError(
            f"Generated base context has only {base_ids.shape[1]} tokens"
        )
    if suffix_block_ids.shape[1] < SUFFIX_TOKENS:
        raise RuntimeError(
            f"Generated suffix context has only {suffix_block_ids.shape[1]} tokens"
        )

    base_ids = base_ids[:, :BASE_CONTEXT_TOKENS].contiguous()
    suffix_ids = torch.cat(
        [suffix_block_ids[:, :SUFFIX_TOKENS].contiguous(), query_ids],
        dim=1,
    ).contiguous()

    return {
        "base_text": base_text,
        "suffix_text": suffix_text,
        "query": query,
        "base_ids": base_ids,
        "suffix_ids": suffix_ids,
    }


def build_full_recompute_ids(
    base_ids: torch.Tensor, dynamic_ids: torch.Tensor, suffix_ids: torch.Tensor
) -> torch.Tensor:
    return torch.cat([base_ids, dynamic_ids, suffix_ids], dim=1).contiguous()


def build_speculative_prefix_ids(
    base_ids: torch.Tensor, dynamic_ids: torch.Tensor
) -> torch.Tensor:
    return torch.cat([base_ids, dynamic_ids], dim=1).contiguous()


@torch.no_grad()
def prefill_and_cache(model: Any, input_ids: torch.Tensor) -> Tuple[float, Any]:
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    outputs = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, outputs.past_key_values


@torch.no_grad()
def prefill_with_cache(
    model: Any, new_ids: torch.Tensor, past_kv: Any
) -> Tuple[float, Any]:
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    outputs = model(input_ids=new_ids, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, outputs.past_key_values


def run_benchmark() -> None:
    gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "7"))
    device = "cuda:0"

    print(f"\n{'='*80}")
    print("EDMM Core Validation Benchmark (v2 - transformers)")
    print(f"GPU: {gpu_index} | Iterations: {NUM_ITERATIONS}")
    print(
        f"Base context: {BASE_CONTEXT_TOKENS} tokens | Suffix: {SUFFIX_TOKENS} tokens"
    )
    print(f"{'='*80}\n")

    model_name = resolve_model_name()
    print(f"Loading tokenizer and model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(
        f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB",
        flush=True,
    )

    benchmark_inputs = build_benchmark_inputs(tokenizer, device)
    base_ids = benchmark_inputs["base_ids"]
    suffix_ids = benchmark_inputs["suffix_ids"]

    base_len = base_ids.shape[1]
    suffix_len = suffix_ids.shape[1]
    print(f"Base prefix: {base_len} tokens, Suffix: {suffix_len} tokens", flush=True)

    results_a: List[Dict[str, Any]] = []
    results_b: List[Dict[str, Any]] = []
    results_c: List[Dict[str, Any]] = []
    bubble_utilizations: List[int] = []

    # ---- Warmup burn-in (untracked) ----
    print(
        "\n--- Warmup: 1 untracked pass per group (CUDA kernel compilation) ---",
        flush=True,
    )
    # Group A warmup
    _, wkv = prefill_and_cache(model, base_ids)
    prefill_with_cache(model, suffix_ids, clone_kv_cache(wkv))
    del wkv
    # Group B warmup
    dynamic_ids = tokenize_to_device(
        tokenizer, f"\n{generate_dynamic_payload()}", device
    )
    wids = build_full_recompute_ids(base_ids, dynamic_ids, suffix_ids)
    prefill_and_cache(model, wids)
    del dynamic_ids, wids
    # Group C warmup
    _, wkv = prefill_and_cache(model, base_ids)
    dynamic_ids = tokenize_to_device(
        tokenizer, f"\n{generate_dynamic_payload()}", device
    )
    wids = build_speculative_prefix_ids(base_ids, dynamic_ids)
    _, skv = prefill_and_cache(model, wids)
    prefill_with_cache(model, suffix_ids, clone_kv_cache(skv))
    del dynamic_ids, wkv, skv, wids
    gc.collect()
    torch.cuda.empty_cache()
    print("Warmup complete. Starting measured iterations.\n", flush=True)

    # ---- Group A: Cached Prefix + Incremental Suffix ----
    print("--- Group A: Cached Prefix + Incremental Suffix ---", flush=True)
    for i in range(NUM_ITERATIONS):
        t_cpu0 = time.perf_counter_ns()

        # Step 1: compute and cache the base prefix
        _, base_kv = prefill_and_cache(model, base_ids)

        # Step 2: sleep (simulating tool execution)
        time.sleep(SLEEP_DURATION)
        util_during = get_gpu_utilization(gpu_index)
        bubble_utilizations.append(util_during)

        t_cpu1 = time.perf_counter_ns()
        cpu_overhead_ms = (t_cpu1 - t_cpu0) / 1e6

        # Step 3: compute suffix using cached base KV (incremental)
        ttft_ms, _ = prefill_with_cache(model, suffix_ids, clone_kv_cache(base_kv))

        results_a.append(
            {
                "iteration": i + 1,
                "cpu_overhead_ms": cpu_overhead_ms,
                "ttft_ms": ttft_ms,
                "cache_status": "Hit",
                "gpu_util_during_sleep": util_during,
            }
        )
        print(
            f"  Iter {i+1:2d}: TTFT={ttft_ms:8.2f}ms  Cache=Hit  GPU_sleep={util_during}%",
            flush=True,
        )

        del base_kv
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Group B: Mid-Prompt Contamination (Full Recompute) ----
    # Dynamic content injected mid-prompt invalidates the entire prefix cache.
    # Must recompute all tokens from scratch.
    print("\n--- Group B: Mid-Prompt Contamination (full recompute) ---", flush=True)
    for i in range(NUM_ITERATIONS):
        # Step 1: compute and cache the base prefix (as if we had it cached)
        _, base_kv = prefill_and_cache(model, base_ids)

        # Step 2: sleep + dynamic payload arrives
        time.sleep(SLEEP_DURATION)
        dynamic_payload = generate_dynamic_payload()

        t_cpu0 = time.perf_counter_ns()
        dynamic_ids = tokenize_to_device(tokenizer, f"\n{dynamic_payload}", device)
        contaminated_ids = build_full_recompute_ids(base_ids, dynamic_ids, suffix_ids)
        t_cpu1 = time.perf_counter_ns()
        cpu_overhead_ms = (t_cpu1 - t_cpu0) / 1e6

        # Step 3: full recompute — cache is useless because mid-prompt changed
        ttft_ms, _ = prefill_and_cache(model, contaminated_ids)

        results_b.append(
            {
                "iteration": i + 1,
                "cpu_overhead_ms": cpu_overhead_ms,
                "ttft_ms": ttft_ms,
                "cache_status": "Miss",
            }
        )
        print(f"  Iter {i+1:2d}: TTFT={ttft_ms:8.2f}ms  Cache=Miss", flush=True)

        del dynamic_ids, contaminated_ids, base_kv
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Group C: EDMM Speculative Warm-Up ----
    # During sleep, speculatively compute the anticipated dynamic prefix.
    # When the orchestrator resumes, the same suffix+query workload is measured.
    print("\n--- Group C: EDMM Speculative Warm-Up ---", flush=True)
    for i in range(NUM_ITERATIONS):
        # Step 1: compute base prefix (initial state)
        _, base_kv = prefill_and_cache(model, base_ids)

        # Step 2: during sleep, build the anticipated dynamic prefix and compute it
        dynamic_payload = generate_dynamic_payload()
        dynamic_ids = tokenize_to_device(tokenizer, f"\n{dynamic_payload}", device)
        anticipated_ids = build_speculative_prefix_ids(base_ids, dynamic_ids)

        t_cpu0 = time.perf_counter_ns()

        # Speculative prefill: compute the anticipated prefix during idle time
        _, spec_kv = prefill_and_cache(model, anticipated_ids)
        time.sleep(SLEEP_DURATION)

        t_cpu1 = time.perf_counter_ns()
        cpu_overhead_ms = (t_cpu1 - t_cpu0) / 1e6

        # Step 3: append suffix using speculative KV directly (zero-copy —
        # spec_kv is not reused, so no clone needed).
        ttft_ms, _ = prefill_with_cache(model, suffix_ids, spec_kv)

        results_c.append(
            {
                "iteration": i + 1,
                "cpu_overhead_ms": cpu_overhead_ms,
                "ttft_ms": ttft_ms,
                "cache_status": "Hit",
            }
        )
        print(f"  Iter {i+1:2d}: TTFT={ttft_ms:8.2f}ms  Cache=Hit", flush=True)

        del dynamic_ids, anticipated_ids, base_kv, spec_kv
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}\n")

    def stats(values: List[float]) -> Tuple[float, float]:
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return mean, variance**0.5

    def trimmed_stats(values: List[float], pct: float = 0.1) -> Tuple[float, float]:
        s = sorted(values)
        trim = max(1, int(len(s) * pct))
        trimmed = s[trim:-trim]
        return stats(trimmed)

    ttft_a = [r["ttft_ms"] for r in results_a]
    ttft_b = [r["ttft_ms"] for r in results_b]
    ttft_c = [r["ttft_ms"] for r in results_c]

    mu_a, sigma_a = stats(ttft_a)
    mu_b, sigma_b = stats(ttft_b)
    mu_c, sigma_c = stats(ttft_c)

    tmu_a, tsig_a = trimmed_stats(ttft_a)
    tmu_b, tsig_b = trimmed_stats(ttft_b)
    tmu_c, tsig_c = trimmed_stats(ttft_c)

    print("| Group | Metric | Mean (ms) | Std Dev (ms) | Trimmed Mean | Trimmed SD |")
    print("|-------|--------|-----------|--------------|--------------|------------|")
    print(
        f"| A (Cached Prefix)       | TTFT | {mu_a:10.2f} | {sigma_a:12.2f} | {tmu_a:12.2f} | {tsig_a:10.2f} |"
    )
    print(
        f"| B (Full Recompute)      | TTFT | {mu_b:10.2f} | {sigma_b:12.2f} | {tmu_b:12.2f} | {tsig_b:10.2f} |"
    )
    print(
        f"| C (Speculative Prefill) | TTFT | {mu_c:10.2f} | {sigma_c:12.2f} | {tmu_c:12.2f} | {tsig_c:10.2f} |"
    )

    print(f"\n--- Orchestration Bubble ---")
    bubble_mu, bubble_sigma = stats([float(x) for x in bubble_utilizations])
    print(
        f"Mean GPU util during {SLEEP_DURATION}s sleep: {bubble_mu:.1f}%  (sigma={bubble_sigma:.1f}%)"
    )
    print(
        f"Iterations at 0%: {sum(1 for x in bubble_utilizations if x == 0)}/{NUM_ITERATIONS}"
    )

    print(f"\n--- Validation Criteria (using 10% trimmed means) ---")
    if tmu_a > 0:
        radix_ratio = tmu_b / tmu_a
        print(
            f"[{'PASS' if radix_ratio >= 2.5 else 'FAIL'}] Radix Discontinuity: B/A = {radix_ratio:.2f}x (threshold >= 2.5x)"
        )
        recovery_ratio = tmu_c / tmu_a
        print(
            f"[{'PASS' if recovery_ratio <= 1.15 else 'FAIL'}] Speculative Recovery: C/A = {recovery_ratio:.2f}x (threshold <= 1.15x)"
        )

    print(f"\n--- Detailed Results ---\n")
    print("| Group | Iter | TTFT (ms) | Cache |")
    print("|-------|------|-----------|-------|")
    for r in results_a:
        print(
            f"| A     | {r['iteration']:4d} | {r['ttft_ms']:9.2f} | {r['cache_status']:5s} |"
        )
    for r in results_b:
        print(
            f"| B     | {r['iteration']:4d} | {r['ttft_ms']:9.2f} | {r['cache_status']:5s} |"
        )
    for r in results_c:
        print(
            f"| C     | {r['iteration']:4d} | {r['ttft_ms']:9.2f} | {r['cache_status']:5s} |"
        )

    print(f"\n{'='*80}")
    print("Benchmark complete.")
    print(f"{'='*80}")

    # Write steady-state results to markdown file
    md_lines = [
        "# EDMM Steady-State Validation Results",
        "",
        f"**Model:** {model_name} (bf16) | **GPU:** NVIDIA H100 (index {gpu_index})",
        f"**Iterations:** {NUM_ITERATIONS} (after 1 untracked warmup pass per group)",
        f"**Base context:** {BASE_CONTEXT_TOKENS} tokens | **Suffix:** {suffix_len} tokens",
        f"**Sleep duration:** {SLEEP_DURATION}s",
        "",
        "## Summary (10% trimmed means — top/bottom outliers excluded)",
        "",
        "| Group | Description | Trimmed Mean (ms) | Trimmed SD (ms) | Raw Mean (ms) | Raw SD (ms) |",
        "|-------|-------------|-------------------|-----------------|---------------|-------------|",
        f"| A | Cached Prefix + Incremental Suffix | {tmu_a:.2f} | {tsig_a:.2f} | {mu_a:.2f} | {sigma_a:.2f} |",
        f"| B | Mid-Prompt Contamination (full recompute) | {tmu_b:.2f} | {tsig_b:.2f} | {mu_b:.2f} | {sigma_b:.2f} |",
        f"| C | EDMM Speculative Prefill + Incremental | {tmu_c:.2f} | {tsig_c:.2f} | {mu_c:.2f} | {sigma_c:.2f} |",
        "",
        "## Validation Criteria (10% trimmed means)",
        "",
        f"| Criterion | Result | Value | Threshold |",
        f"|-----------|--------|-------|-----------|",
        f"| Orchestration Bubble | {'PASS' if bubble_mu < 5.0 else 'FAIL'} | GPU util = {bubble_mu:.1f}% (sigma={bubble_sigma:.1f}%) | ~0% during sleep |",
    ]
    if tmu_a > 0:
        radix_ratio = tmu_b / tmu_a
        recovery_ratio = tmu_c / tmu_a
        md_lines.extend(
            [
                f"| Radix Discontinuity (B/A) | {'PASS' if radix_ratio >= 2.5 else 'FAIL'} | {radix_ratio:.2f}x | >= 2.5x |",
                f"| Speculative Recovery (C/A) | {'PASS' if recovery_ratio <= 1.15 else 'FAIL'} | {recovery_ratio:.2f}x | <= 1.15x |",
            ]
        )
    md_lines.extend(
        [
            "",
            "## Detailed Iteration Data",
            "",
            "| Group | Iter | TTFT (ms) | Cache Status |",
            "|-------|------|-----------|--------------|",
        ]
    )
    for r in results_a:
        md_lines.append(
            f"| A | {r['iteration']} | {r['ttft_ms']:.2f} | {r['cache_status']} |"
        )
    for r in results_b:
        md_lines.append(
            f"| B | {r['iteration']} | {r['ttft_ms']:.2f} | {r['cache_status']} |"
        )
    for r in results_c:
        md_lines.append(
            f"| C | {r['iteration']} | {r['ttft_ms']:.2f} | {r['cache_status']} |"
        )
    md_lines.append("")

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "edmm_steady_state_results.md"
    )
    with open(results_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nResults written to: {results_path}", flush=True)


if __name__ == "__main__":
    run_benchmark()

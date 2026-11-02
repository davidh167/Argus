"""Statistics and markdown for the Phase 0 baseline report."""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

from bench_scan import ScanResult


@dataclass
class Timing:
    name: str
    what: str
    samples_ms: list[float]
    warmup: int


@dataclass
class CaseResults:
    timings: list[Timing] = field(default_factory=list)
    # Facts about the work a case did (not timings), printed beside the table.
    facts: dict[str, str] = field(default_factory=dict)


WARM_P95_LIMIT_MS = 50.0
COLD_P95_LIMIT_MS = 1000.0
COLD_CASE = "find_similar_postings (cold)"
WARM_CASE = "find_similar_postings (warm)"


def percentile(samples: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method) of raw samples."""
    return float(np.percentile(samples, q))


def fmt_ms(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def environment(postgres_version: str) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "machine": f"{platform.platform()}, {platform.machine()}, {os.cpu_count()} logical CPUs",
        "python": platform.python_version(),
        "libraries": ", ".join(
            f"{p} {version(p)}" for p in ("numpy", "scipy", "bm25s", "sqlalchemy")
        ),
        "postgres": postgres_version,
        "git commit": commit,
    }


def timings_table(timings: list[Timing]) -> str:
    lines = [
        "| Case | Median ms | p95 ms | Min ms | Max ms | Timed runs | Warmup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in timings:
        s = t.samples_ms
        lines.append(
            f"| {t.name} | {fmt_ms(percentile(s, 50))} | {fmt_ms(percentile(s, 95))} "
            f"| {fmt_ms(min(s))} | {fmt_ms(max(s))} | {len(s)} | {t.warmup} |"
        )
    return "\n".join(lines)


def kill_criterion(timings: list[Timing]) -> str:
    by_name = {t.name: t for t in timings}
    warm = percentile(by_name[WARM_CASE].samples_ms, 95)
    cold = percentile(by_name[COLD_CASE].samples_ms, 95)
    warm_ok = warm < WARM_P95_LIMIT_MS
    cold_ok = cold < COLD_P95_LIMIT_MS
    return "\n".join(
        [
            "| Condition | Measured | Limit | Holds |",
            "|---|---:|---:|---|",
            f"| warm find_similar p95 | {fmt_ms(warm)} ms | < {WARM_P95_LIMIT_MS:.0f} ms "
            f"| {'yes' if warm_ok else 'no'} |",
            f"| cold find_similar p95 | {fmt_ms(cold)} ms | < {COLD_P95_LIMIT_MS:.0f} ms "
            f"| {'yes' if cold_ok else 'no'} |",
            "",
            f"Kill criterion (plan section 3, Phase 0) satisfied: {'yes' if warm_ok and cold_ok else 'no'}.",
        ]
    )


def scan_table(scan: ScanResult) -> str:
    def pct(part: int, whole: int) -> str:
        return f"{100 * part / whole:.2f}%" if whole else "n/a"

    rows = [
        ("postings", f"{scan.n_postings:,}"),
        (
            "postings with at least one non-ASCII word token",
            f"{scan.n_postings_non_ascii:,} ({pct(scan.n_postings_non_ascii, scan.n_postings)})",
        ),
        ("word tokens (Unicode `\\w+`)", f"{scan.n_tokens:,}"),
        (
            "tokens containing a non-ASCII word character",
            f"{scan.n_tokens_non_ascii:,} ({pct(scan.n_tokens_non_ascii, scan.n_tokens)})",
        ),
        ("distinct such tokens", f"{scan.n_distinct_non_ascii_tokens:,}"),
        (
            "postings whose top-10 similar list (order) differs, ASCII-only vs Unicode",
            f"{scan.n_top10_order_changed:,} ({pct(scan.n_top10_order_changed, scan.n_postings)})",
        ),
        (
            "postings whose top-10 similar set differs",
            f"{scan.n_top10_set_changed:,} ({pct(scan.n_top10_set_changed, scan.n_postings)})",
        ),
        (
            "postings whose top-1 similar posting differs",
            f"{scan.n_top1_changed:,} ({pct(scan.n_top1_changed, scan.n_postings)})",
        ),
        (
            "of the order-changed postings, own text contains non-ASCII tokens",
            f"{scan.n_order_changed_with_non_ascii_text:,}",
        ),
        (
            "of the set-changed postings, own text contains non-ASCII tokens",
            f"{scan.n_set_changed_with_non_ascii_text:,}",
        ),
    ]
    return "\n".join(["| Measure | Value |", "|---|---:|", *(f"| {k} | {v} |" for k, v in rows)])


def corpus_section(
    name: str,
    n_postings: int,
    results: CaseResults,
    scan: ScanResult,
    notes: dict[str, str],
) -> str:
    facts = {**notes, **results.facts}
    runs = min(len(t.samples_ms) for t in results.timings)
    if runs < 30:
        facts["timed runs"] = f"{runs} per case (fewer than 30: the cases are too slow at this N)"
    parts = [
        f"## Corpus: {name} (N = {n_postings:,})",
        "",
        "\n".join(f"- {k}: {v}" for k, v in facts.items()),
        "",
        "### Timings",
        "",
        timings_table(results.timings),
        "",
        "Case definitions:",
        "",
        "\n".join(f"- {t.name.strip()}: {t.what}" for t in results.timings),
        "",
        "### Phase 0 kill criterion",
        "",
        kill_criterion(results.timings),
        "",
        "### Non-ASCII tokenization scan (decision D4)",
        "",
        scan_table(scan),
    ]
    return "\n".join(parts)


def render_report(env: dict[str, str], sections: list[str]) -> str:
    header = [
        "# Python baseline for the native engine (Phase 0)",
        "",
        "\n".join(f"- {k}: {v}" for k, v in env.items()),
        "",
        "Medians and p95 are over the timed runs (p95: linear interpolation between order "
        "statistics, numpy default). Each case runs its warmup runs first, untimed. Times are wall "
        "clock around the call; every database call goes to a local throwaway Postgres.",
        "",
    ]
    return "\n".join(header) + "\n" + "\n\n".join(sections) + "\n"

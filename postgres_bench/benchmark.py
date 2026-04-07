#!/usr/bin/env python3
"""PostgreSQL configuration benchmark.

For each configuration variant the benchmark:
  1. Runs ``initdb`` once to create a template data directory (one-time setup).
  2. Per iteration: copies the template directory, starts postgres, runs the
     workload, stops postgres, and removes the copy.

This separates the immutable initialisation cost from the repeatable
start/stop signal so that ``start`` and ``stop`` are the primary metrics.

Usage:
    uv run postgres_bench/benchmark.py
    uv run postgres_bench/benchmark.py --iterations 5 --rows 10000
    uv run postgres_bench/benchmark.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tabulate import tabulate

CONFIGS_DIR = Path(__file__).parent / "configs"

# PostgreSQL superuser created by initdb (must match the OS user that
# runs initdb when no -U flag is passed).
_PG_USER = getpass.getuser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_pg_bin() -> Path:
    """Return the directory that contains ``initdb`` and ``pg_ctl``."""
    # Prefer versioned Debian/Ubuntu layout, then fall back to PATH.
    for version in range(20, 13, -1):
        candidate = Path(f"/usr/lib/postgresql/{version}/bin")
        if (candidate / "initdb").exists():
            return candidate
    for path_dir in os.environ.get("PATH", "").split(":"):
        if path_dir and (Path(path_dir) / "initdb").exists():
            return Path(path_dir)
    raise RuntimeError(
        "PostgreSQL binaries not found. "
        "Install PostgreSQL or add its bin directory to PATH."
    )


def find_free_port() -> int:
    """Ask the OS for a free TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_configs() -> dict[str, str]:
    """Read all ``*.conf`` files from the configs directory.

    Returns a dict mapping display-name → config-text, sorted by filename.
    Display name is derived by stripping leading digits and underscores
    from the file stem (e.g. ``01_baseline`` → ``baseline``).
    """
    configs: dict[str, str] = {}
    for conf_file in sorted(CONFIGS_DIR.glob("*.conf")):
        name = conf_file.stem.lstrip("0123456789_")
        configs[name] = conf_file.read_text()
    if not configs:
        raise RuntimeError(f"No *.conf files found in {CONFIGS_DIR}")
    return configs


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IterationResult:
    config_name: str
    iteration: int
    copy_s: float   # time to copy the template data directory
    start_s: float
    load_s: float
    stop_s: float

    @property
    def total_s(self) -> float:
        return self.copy_s + self.start_s + self.load_s + self.stop_s


# ---------------------------------------------------------------------------
# Database workload
# ---------------------------------------------------------------------------


async def _load_sample_data(dsn: str, num_rows: int) -> None:
    """Create a table and bulk-insert *num_rows* rows."""
    import asyncpg  # imported here so the module is usable without asyncpg on PATH checks

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            CREATE TABLE sample (
                id     SERIAL PRIMARY KEY,
                name   TEXT             NOT NULL,
                score  DOUBLE PRECISION NOT NULL,
                ts     TIMESTAMPTZ      DEFAULT now()
            )
            """
        )
        rows = [(f"item_{i:07d}", float(i) * 1.23) for i in range(num_rows)]
        await conn.executemany(
            "INSERT INTO sample (name, score) VALUES ($1, $2)", rows
        )
        count: int = await conn.fetchval("SELECT count(*) FROM sample")
        if count != num_rows:
            raise RuntimeError(f"Expected {num_rows} rows, got {count}")
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# One-time template setup
# ---------------------------------------------------------------------------


def setup_template(
    config_name: str,
    config_text: str,
    pg_bin: Path,
    root: Path,
) -> tuple[Path, float]:
    """Run ``initdb`` once and write the variant config.

    Returns the template directory path and the time taken (seconds).
    Port and socket path are deliberately omitted from the stored config —
    those runtime settings are injected fresh on every iteration copy.
    """
    template_dir = root / f"template_{config_name}"

    t0 = time.perf_counter()
    subprocess.run(
        [
            str(pg_bin / "initdb"),
            "-D", str(template_dir),
            "--auth=trust",
            "--no-instructions",
        ],
        check=True,
        capture_output=True,
    )
    # Overwrite postgresql.conf with the variant settings only.
    # Runtime settings (port, socket, logging) are added per-iteration.
    (template_dir / "postgresql.conf").write_text(config_text)
    init_s = time.perf_counter() - t0

    return template_dir, init_s


# ---------------------------------------------------------------------------
# Single benchmark cycle
# ---------------------------------------------------------------------------


def run_one_cycle(
    config_name: str,
    config_text: str,
    template_dir: Path,
    iteration: int,
    pg_bin: Path,
    num_rows: int,
) -> IterationResult:
    """Copy template → start → load → stop → cleanup."""

    tmpdir = Path(tempfile.mkdtemp(prefix="pgbench_"))

    try:
        # ── copy template data directory ─────────────────────────────────────
        data_dir = tmpdir / "data"
        t0 = time.perf_counter()
        shutil.copytree(template_dir, data_dir)
        copy_s = time.perf_counter() - t0

        # ── write runtime postgresql.conf overrides ──────────────────────────
        # Appended after the variant settings so they always win.
        port = find_free_port()
        socket_dir = tmpdir / "run"
        socket_dir.mkdir()
        runtime_conf = (
            "\n"
            "# --- benchmark runtime overrides ---\n"
            f"port                      = {port}\n"
            "listen_addresses          = '127.0.0.1'\n"
            f"unix_socket_directories   = '{socket_dir}'\n"
            "logging_collector         = off\n"
            "log_min_messages          = fatal\n"
            "log_min_error_statement   = fatal\n"
        )
        conf_path = data_dir / "postgresql.conf"
        conf_path.write_text(config_text + runtime_conf)

        # ── pg_ctl start ─────────────────────────────────────────────────────
        # Use DEVNULL for stdout/stderr to prevent the daemonized postgres
        # process from inheriting the pipes and blocking subprocess.run().
        t0 = time.perf_counter()
        subprocess.run(
            [
                str(pg_bin / "pg_ctl"), "start",
                "-D", str(data_dir),
                "-w",        # wait until ready to accept connections
                "-t", "60",  # timeout seconds
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        start_s = time.perf_counter() - t0

        # ── load data ────────────────────────────────────────────────────────
        dsn = f"postgresql://{_PG_USER}@127.0.0.1:{port}/postgres"
        t0 = time.perf_counter()
        asyncio.run(_load_sample_data(dsn, num_rows))
        load_s = time.perf_counter() - t0

        # ── pg_ctl stop ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        subprocess.run(
            [
                str(pg_bin / "pg_ctl"), "stop",
                "-D", str(data_dir),
                "-m", "fast",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        stop_s = time.perf_counter() - t0

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return IterationResult(config_name, iteration, copy_s, start_s, load_s, stop_s)


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(iterations: int, num_rows: int) -> list[IterationResult]:
    pg_bin = find_pg_bin()
    configs = load_configs()

    print(f"PostgreSQL binaries : {pg_bin}")
    print(f"Rows per iteration  : {num_rows:,}")
    print(f"Iterations/variant  : {iterations}")
    print(f"Variants            : {', '.join(configs)}")
    print()

    root = Path(tempfile.mkdtemp(prefix="pgbench_root_"))
    results: list[IterationResult] = []

    try:
        # ── one-time setup: initdb per variant ───────────────────────────────
        print("── Setup: initialising template data directories ────────────────")
        templates: dict[str, tuple[Path, float]] = {}
        for config_name, config_text in configs.items():
            print(f"  {config_name:<25}", end="", flush=True)
            template_dir, init_s = setup_template(config_name, config_text, pg_bin, root)
            templates[config_name] = (template_dir, init_s)
            print(f"  init={init_s:.2f}s")
        print()

        # ── repeated iterations ───────────────────────────────────────────────
        for i in range(1, iterations + 1):
            print(f"── Iteration {i}/{iterations} " + "─" * 40)
            for config_name, config_text in configs.items():
                template_dir, _ = templates[config_name]
                print(f"  {config_name:<25}", end="", flush=True)
                try:
                    r = run_one_cycle(
                        config_name, config_text, template_dir, i, pg_bin, num_rows
                    )
                    results.append(r)
                    print(
                        f"  copy={r.copy_s:.2f}s  start={r.start_s:.2f}s"
                        f"  load={r.load_s:.2f}s  stop={r.stop_s:.2f}s"
                        f"  → total={r.total_s:.2f}s"
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}")
            print()

    finally:
        shutil.rmtree(root, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Statistics & reporting
# ---------------------------------------------------------------------------

_PHASES = ("copy_s", "start_s", "load_s", "stop_s", "total_s")
_PHASE_LABELS = ("copy", "start", "load", "stop", "total")


def _stats(values: list[float]) -> tuple[float, float, float, float, float]:
    """Return (mean, median, stdev, min, max)."""
    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, median, stdev, min(values), max(values)


def print_statistics(results: list[IterationResult]) -> None:
    by_config: dict[str, list[IterationResult]] = {}
    for r in results:
        by_config.setdefault(r.config_name, []).append(r)

    # ── Summary table (mean ± stdev per phase) ──────────────────────────────
    headers = ["config"] + list(_PHASE_LABELS)
    rows = []
    for config_name, rs in by_config.items():
        row: list[str] = [config_name]
        for phase in _PHASES:
            vals = [getattr(r, phase) for r in rs]
            mean, _, stdev, *_ = _stats(vals)
            row.append(f"{mean:.3f} ± {stdev:.3f}")
        rows.append(row)

    print("=== Summary: mean ± stdev (seconds) ===")
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()

    # ── Detailed per-config, per-phase breakdown ─────────────────────────────
    print("=== Detailed statistics per variant ===")
    detail_rows = []
    for config_name, rs in by_config.items():
        for phase, label in zip(_PHASES, _PHASE_LABELS):
            vals = [getattr(r, phase) for r in rs]
            mean, median, stdev, vmin, vmax = _stats(vals)
            detail_rows.append(
                [
                    config_name,
                    label,
                    f"{mean:.3f}",
                    f"{median:.3f}",
                    f"{stdev:.3f}",
                    f"{vmin:.3f}",
                    f"{vmax:.3f}",
                ]
            )
    detail_headers = ["config", "phase", "mean", "median", "stdev", "min", "max"]
    print(tabulate(detail_rows, headers=detail_headers, tablefmt="github"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark multiple PostgreSQL configurations for minimal "
            "resource usage and fast startup.  Sacrifices durability."
        )
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of full cycles per configuration variant (default: 5)",
    )
    parser.add_argument(
        "--rows", "-r",
        type=int,
        default=10_000,
        metavar="N",
        help="Number of rows to insert per cycle (default: 10 000)",
    )
    args = parser.parse_args()

    if args.iterations < 2:
        parser.error("--iterations must be at least 2 to compute stdev")
    if args.rows < 1:
        parser.error("--rows must be at least 1")

    results = run_benchmark(args.iterations, args.rows)
    if results:
        print_statistics(results)


if __name__ == "__main__":
    main()

"""Command line.

  glymph run   --input raw.h5 --output out.h5ad            one library on one device
  glymph batch --manifest libs.tsv --outdir out --devices cuda:0,cuda:1,cuda:2,cuda:3
                                                            one library per GPU, in parallel

The batch manifest is a tab-separated file with a header and columns
  name  input  [cell_calls]  [unit_key]  [pool_key]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path


def _add_config_args(p: argparse.ArgumentParser):
    from .api import Config

    for f in fields(Config):
        if f.name == "device":
            continue
        default = f.default
        kind = type(default) if default is not None else (int if f.name == "expected_cells" else float)
        if kind is bool:
            p.add_argument(f"--{f.name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=default)
        else:
            p.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=default)


def _config_from(args, device: str):
    from .api import Config

    kw = {f.name: getattr(args, f.name) for f in fields(Config) if f.name != "device"}
    return Config(device=device, **kw)


def run_one(input, output, cfg, report=None, cell_calls=None, unit_key=None, pool_key=None, layout="auto", plot=None):
    from .api import remove_background
    from .io import read_barcode_list, read_counts

    t0 = time.perf_counter()
    adata = read_counts(input, layout)
    t_read = time.perf_counter() - t0
    calls = read_barcode_list(cell_calls) if cell_calls else None
    out, rep = remove_background(adata, unit_key or None, pool_key or None, calls, cfg)
    t1 = time.perf_counter()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output, compression="gzip")
    rep["timings"]["read_s"] = round(t_read, 3)
    rep["timings"]["write_s"] = round(time.perf_counter() - t1, 3)
    rep["timings"]["total_s"] = round(time.perf_counter() - t0, 3)
    rep["input"], rep["output"] = str(input), str(output)
    Path(report or str(output) + ".report.json").write_text(json.dumps(rep, indent=1))
    if plot:
        from .background import estimate_priors, plot_priors
        import scipy.sparse as sp

        X = sp.csr_matrix(adata.X)
        plot_priors(X, estimate_priors(X, low_count=cfg.low_count, expected_cells=cfg.expected_cells), plot)
    print(f"[glymph] {input}: {rep['n_cells']} cells, removed "
          f"{100 * rep['removed_fraction_in_cells']:.2f}% of their counts, {rep['timings']['total_s']} s")
    return rep


def _worker(device, jobs, results, args_dict):
    import queue

    args = argparse.Namespace(**args_dict)
    cfg = _config_from(args, device)
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        out = Path(args.outdir) / f"{job['name']}.glymph.h5ad"
        try:
            run_one(job["input"], out, cfg, None, job.get("cell_calls") or None,
                    job.get("unit_key") or None, job.get("pool_key") or None)
            results.put((job["name"], device, True, ""))
        except Exception as exc:  # keep the other libraries going
            print(f"[glymph] FAILED {job['name']} on {device}: {exc!r}"[:400])
            results.put((job["name"], device, False, type(exc).__name__))
            if "OutOfMemory" in type(exc).__name__:
                # The device is full or shared with another process: stop feeding it.
                # Libraries left in the queue are picked up by the other devices or by the retry pass.
                return


def main(argv=None):
    ap = argparse.ArgumentParser(prog="glymph", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one library on one device")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--report")
    r.add_argument("--cell-calls", help="file with one vendor-called barcode per line (weak labels)")
    r.add_argument("--unit-key", help="obs column: which barcodes share a soup")
    r.add_argument("--pool-key", help="obs column: which barcodes were amplified together")
    r.add_argument("--layout", default="auto", choices=["auto", "10x", "parse"])
    r.add_argument("--plot", help="write a barcode rank plot with the chosen thresholds")
    r.add_argument("--device", default="auto")
    _add_config_args(r)

    b = sub.add_parser("batch", help="many libraries, one per GPU")
    b.add_argument("--manifest", required=True)
    b.add_argument("--outdir", required=True)
    b.add_argument("--devices", default="auto", help="comma-separated, for example cuda:0,cuda:1")
    _add_config_args(b)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        run_one(args.input, args.output, _config_from(args, args.device), args.report, args.cell_calls,
                args.unit_key, args.pool_key, args.layout, args.plot)
        return

    import pandas as pd
    import torch
    import torch.multiprocessing as mp

    if args.devices == "auto":
        n = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(n)] or ["auto"]
    else:
        devices = args.devices.split(",")
    man = pd.read_csv(args.manifest, sep="\t").fillna("")
    # Largest inputs first so the GPUs finish at about the same time.
    man["_size"] = [Path(p).stat().st_size if Path(p).is_file() else 0 for p in man["input"]]
    man = man.sort_values("_size", ascending=False).drop(columns="_size")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    records = man.to_dict("records")
    t0 = time.perf_counter()
    done, attempts, pending, use = {}, [], records, list(devices)
    for attempt in (1, 2):
        # Pass 2 retries what failed in pass 1, only on devices that finished a library.
        jobs, results = ctx.Queue(), ctx.Queue()
        for rec in pending:
            jobs.put(rec)
        time.sleep(0.5)  # let the queue feeder thread flush before workers poll it
        procs = [ctx.Process(target=_worker, args=(d, jobs, results, vars(args))) for d in use]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        got = []
        while not results.empty():
            got.append(results.get())
        for name, dev, ok, err in got:
            attempts.append({"library": str(name), "device": dev, "ok": bool(ok), "error": err, "pass": attempt})
            if ok:
                done[str(name)] = dev
        pending = [r for r in records if str(r["name"]) not in done]
        healthy = sorted({a["device"] for a in attempts if a["ok"]})
        if not pending or not healthy or attempt == 2:
            break
        print(f"[glymph] retrying {[r['name'] for r in pending]} on {healthy}")
        use = healthy
    wall = round(time.perf_counter() - t0, 2)
    failed = [str(r["name"]) for r in pending]
    summary = {"n_libraries": len(man), "n_done": len(done), "failed": failed, "devices": devices,
               "retried": sorted({a["library"] for a in attempts if not a["ok"]}), "attempts": attempts, "wall_s": wall}
    (Path(args.outdir) / "batch_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[glymph] batch done: { {k: v for k, v in summary.items() if k != 'attempts'} }")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

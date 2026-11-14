"""Download the sound-effect library once: one short audio clip per entry of src/sfx/catalog.py.

    python scripts/build_sfx_library.py                 # everything that is missing
    python scripts/build_sfx_library.py --only fah bruh # just these ids
    python scripts/build_sfx_library.py --force         # re-download even sounds that already exist
    python scripts/build_sfx_library.py --list          # show the catalog and what is already downloaded

The audio goes to VIBECROP_SFX_DIR (default <repo>/sfx) and is NEVER committed (the sounds are copyrighted).
Safe to stop and run again: it only fetches what is missing. A few sounds may fail (YouTube search is not
exact); the summary lists them, and you can fix the search text in src/sfx/catalog.py and rerun.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.sfx import fetch, library  # noqa: E402
from src.sfx.catalog import CATALOG  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="only these catalog ids")
    ap.add_argument("--force", action="store_true", help="download again even if the sound exists")
    ap.add_argument("--list", action="store_true", help="list the catalog and exit")
    ap.add_argument("--workers", type=int, default=4, help="downloads at once (default 4)")
    args = ap.parse_args()

    have = {s["id"] for s in library.load() if library.available(s)}
    if args.list:
        for row in CATALOG:
            print(f"{'[x]' if row['id'] in have else '[ ]'} {row['id']:<28} {row['name']}")
        print(f"\n{len(have & {r['id'] for r in CATALOG})}/{len(CATALOG)} downloaded into {config.sfx_dir()}")
        return 0

    rows = [r for r in CATALOG if (not args.only or r["id"] in args.only) and (args.force or r["id"] not in have)]
    unknown = set(args.only or []) - {r["id"] for r in CATALOG}
    if unknown:
        print(f"Unknown id(s): {', '.join(sorted(unknown))}. Use --list to see the catalog.")
        return 2
    if not rows:
        print(f"Nothing to do: all {len(CATALOG)} sounds are already in {config.sfx_dir()}.")
        return 0
    print(f"Downloading {len(rows)} sound(s) into {config.sfx_dir()} ({args.workers} at a time)...")

    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch.build_catalog_entry, row): row for row in rows}
        for done, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                entry = fut.result()
                print(f"[{done}/{len(rows)}] ok    {row['id']:<28} {entry['duration']:.1f}s")
            except Exception as exc:  # noqa: BLE001 - one bad sound must not stop the others
                failed.append((row["id"], str(exc)))
                print(f"[{done}/{len(rows)}] FAIL  {row['id']:<28} {exc}")
    print(f"\nDone: {len(rows) - len(failed)} downloaded, {len(failed)} failed.")
    for lib_id, why in failed:
        print(f"  - {lib_id}: {why}")
    if failed:
        print("Edit the 'query' of the failed ones in src/sfx/catalog.py and run this script again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

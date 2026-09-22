#!/usr/bin/env python3
"""
aggregate.py — runs on GitHub Actions, computes the live FPL aggregates,
writes live.json and pushes it to Cloudflare KV.

    python3 aggregate.py                 # compute + write ./live.json
    python3 aggregate.py --upload        # ...and push to Cloudflare KV

Env vars for --upload:
    CF_ACCOUNT_ID, CF_KV_NAMESPACE_ID, CF_API_TOKEN
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time

from fpl_engine import (LiveEngine, get, CACHE_DIR)

# ---------------------------------------------------------------- config ----
N_TOP10K = int(os.environ.get("N_TOP10K", 10_000))   # "Live Average Top 10k"
N_RANDOM = int(os.environ.get("N_RANDOM", 3_000))    # overall "Live Average"
KV_KEY = "fpl-live"        # key the Worker reads
MIN_SAMPLE_BEFORE_PUBLISH = 200   # don't publish noise

HERE = os.path.dirname(os.path.abspath(__file__))


def current_gameweek() -> int:
    boot = get("/bootstrap-static/")
    for e in boot["events"]:
        if e["is_current"]:
            return e["id"]
    nxt = [e for e in boot["events"] if e["is_next"]]
    return nxt[0]["id"] if nxt else boot["events"][-1]["id"]


# Skip the pointless runs. The schedule fires 288x/day but a gameweek is only
# interesting from its deadline until a few days after it is checked.
# (FPL keeps `is_current` on the finished gameweek until the next deadline.)
SETTLE_DAYS = 5


def should_run() -> tuple[bool, str]:
    boot = get("/bootstrap-static/")
    cur = next((e for e in boot["events"] if e["is_current"]), None)
    if cur is None:
        return False, "no current gameweek"
    if not cur["finished"]:
        return True, f"GW{cur['id']} is live"
    try:
        dl = dt.datetime.strptime(cur["deadline_time"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError):
        return True, "unparseable deadline - running to be safe"
    age = dt.datetime.now(dt.timezone.utc) - dl.replace(tzinfo=dt.timezone.utc)
    if age > dt.timedelta(days=SETTLE_DAYS):
        return False, (f"GW{cur['id']} finished {age.days}d ago and is settled; "
                       f"next deadline {next((e['deadline_time'] for e in boot['events'] if e['is_next']), '?')}")
    return True, f"GW{cur['id']} finished {age.days}d ago - publishing final numbers"


def build_rank_table(scores: list[int], ranked_count: int) -> list[dict]:
    """For every achievable score, the estimated live rank.

    The app then looks its score up locally — no math on the client, and it
    stays correct even for scores we never sampled.
    """
    desc = sorted(scores, reverse=True)     # descending
    n = len(desc)
    # 'above' = how many sampled teams score strictly more than `pts`.
    # As pts falls, `above` only grows - so walk pts DOWNWARD and advance once.
    above = 0
    rows = {}
    for pts in range(200, -1, -1):
        while above < n and desc[above] > pts:
            above += 1
        rows[pts] = {
            "points": pts,
            "rank": max(1, round(above / n * ranked_count)),
            "pct": round(100.0 * above / n, 2),
        }
    return [rows[p] for p in range(0, 201)]


def compute(gw: int) -> dict:
    # seed is fixed per gameweek -> the picks cache is reused every run
    eng = LiveEngine(gw, n_top10k=N_TOP10K, n_random=N_RANDOM, seed=gw)
    print(f"[{now()}] GW{gw} | building picks cache ...", flush=True)
    t0 = time.time()
    eng.prepare()
    print(f"[{now()}] cache ready: {len(eng._cache):,} teams "
          f"({time.time()-t0:.0f}s)", flush=True)

    eng.refresh()

    if len(eng._random_scores) < MIN_SAMPLE_BEFORE_PUBLISH:
        raise SystemExit(f"sample too small ({len(eng._random_scores)}) - refusing to publish")

    avg = eng.live_average()
    lo, hi = eng.live_average_ci()
    ranked = eng.event["ranked_count"]

    return {
        "schema": 1,
        "gw": gw,
        "updated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "live": not eng.event["finished"],
        "average": round(avg, 2),
        "average_ci": [round(lo, 2), round(hi, 2)],
        "average_top10k": round(eng.live_average_top10k(), 2),
        "official_average_after_check": eng.event["average_entry_score"] if eng.event["finished"] else None,
        "ranked_count": ranked,
        "sample": {
            "random": len(eng._random_scores),
            "top10k": len(eng._top_scores),
        },
        "rank_table": build_rank_table(eng._random_scores, ranked),
    }


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")


def upload_to_kv(payload: dict) -> None:
    import requests
    acct = os.environ["CF_ACCOUNT_ID"]
    ns = os.environ["CF_KV_NAMESPACE_ID"]
    tok = os.environ["CF_API_TOKEN"]
    body = json.dumps(payload)
    hdr = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}

    # Two keys:
    #   fpl-live          -> always the current gameweek (cheap default)
    #   fpl-live-gw{N}    -> a permanent archive per gameweek, so past rounds
    #                        stay browsable after the season moves on.
    keys = [KV_KEY, f"{KV_KEY}-gw{payload['gw']}"]
    for key in keys:
        url = (f"https://api.cloudflare.com/client/v4/accounts/{acct}"
               f"/storage/kv/namespaces/{ns}/values/{key}")
        r = requests.put(url, headers=hdr, data=body, timeout=30)
        if r.status_code != 200:
            raise SystemExit(f"KV upload failed {r.status_code} for {key}: {r.text[:300]}")
        print(f"[{now()}] uploaded to Cloudflare KV key='{key}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="run even when the gameweek is long finished")
    args = ap.parse_args()

    if not args.force:
        ok, why = should_run()
        if not ok:
            print(f"[{now()}] skipping: {why}")
            return

    gw = args.gw or current_gameweek()
    payload = compute(gw)

    out = os.path.join(HERE, "live.json")
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    kb = os.path.getsize(out) / 1024
    print(f"[{now()}] wrote {out} ({kb:.1f} KB)")
    print(f"          average={payload['average']}  top10k={payload['average_top10k']}  "
          f"sample={payload['sample']}")

    if args.upload:
        needed = ("CF_ACCOUNT_ID", "CF_KV_NAMESPACE_ID", "CF_API_TOKEN")
        if all(os.environ.get(k) for k in needed):
            upload_to_kv(payload)
        else:
            missing = [k for k in needed if not os.environ.get(k)]
            print(f"          (skipped KV upload - missing secrets: {', '.join(missing)}. "
                  f"Add them in repo Settings > Secrets and variables > Actions)")
    else:
        print("          (skipped KV upload - pass --upload)")


if __name__ == "__main__":
    main()

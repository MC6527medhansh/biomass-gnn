# biomass-gnn/scripts/download_bigg_models.py
"""
Robust BiGG SBML downloader (parallel), with:
- polite headers,
- detailed status logs,
- tries both .xml and .xml.gz and decompresses if needed,
- follows redirects,
- adjustable concurrency.

Usage:
  python scripts/download_bigg_models.py --ids bigg_ids.txt --out data/models --workers 8
"""

import argparse, asyncio, os, gzip, io, sys
from typing import Optional

URL_CANDIDATES = [
    # Try HTTPS and HTTP, xml and gz
    "https://bigg.ucsd.edu/static/models/{mid}.xml",
    "https://bigg.ucsd.edu/static/models/{mid}.xml.gz",
    "http://bigg.ucsd.edu/static/models/{mid}.xml",
    "http://bigg.ucsd.edu/static/models/{mid}.xml.gz",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BiomassGNN/1.0; +https://example.org)"
}

async def fetch(session, url: str) -> Optional[bytes]:
    try:
        async with session.get(url, headers=HEADERS, allow_redirects=True) as r:
            if r.status == 200:
                return await r.read()
            else:
                print(f"[{r.status}] {url}")
                return None
    except Exception as e:
        print(f"[ERR] {url} -> {e}")
        return None

def maybe_decompress(mid: str, data: bytes, url: str) -> bytes:
    if url.endswith(".gz"):
        try:
            buf = io.BytesIO(data)
            with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
                return gz.read()
        except Exception as e:
            print(f"[decompress-fail] {mid}: {e}")
            return b""
    return data

async def download_one(session, mid: str, out_dir: str) -> bool:
    dest = os.path.join(out_dir, f"{mid}.xml")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {mid} already exists")
        return True

    for url in URL_CANDIDATES:
        url = url.format(mid=mid)
        data = await fetch(session, url)
        if data:
            data = maybe_decompress(mid, data, url)
            if not data:
                continue
            with open(dest, "wb") as f:
                f.write(data)
            print(f"[ok]   {mid}  <- {url}")
            return True

    print(f"[fail] {mid}")
    return False

async def runner(ids_path: str, out_dir: str, workers: int):
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=120)
    connector = aiohttp.TCPConnector(limit=max(1, workers))
    os.makedirs(out_dir, exist_ok=True)

    with open(ids_path, "r") as f:
        ids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    if not ids:
        print(f"[ERROR] No IDs in {ids_path}")
        return

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sem = asyncio.Semaphore(workers)

        async def bound(mid):
            async with sem:
                return await download_one(session, mid, out_dir)

        results = await asyncio.gather(*(bound(mid) for mid in ids))
    ok = sum(1 for r in results if r)
    print(f"Done. {ok}/{len(ids)} downloaded to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="Text file with BiGG IDs (one per line)")
    ap.add_argument("--out", required=True, help="Output folder (e.g., data/models)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    try:
        import aiohttp  # noqa
    except Exception:
        sys.exit("Install deps first: pip install aiohttp async-timeout")
    asyncio.run(runner(args.ids, args.out, args.workers))

if __name__ == "__main__":
    main()

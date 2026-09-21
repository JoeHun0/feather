#!/usr/bin/env python3
"""Fetch third-party CC0 assets into the gitignored cache under scratch/assets/.

Why a fetch script instead of committing the files
--------------------------------------------------
The assets are public domain, so committing them would be legal, but it would
put megabytes of binaries in git history for content that only feeds test
scenes. Instead each pack is **pinned**: an exact URL plus the SHA-256 of the
archive. A download that does not match is refused rather than extracted, so a
moved or altered file can never silently change a test scene. Only the files
the tools use are extracted, along with the pack's own licence.

Stdlib only, like the scene generators.

Usage:
    python3 tools/fetch_assets.py                  # fetch every pack not cached yet
    python3 tools/fetch_assets.py --zip PATH       # use an already-downloaded archive
    python3 tools/fetch_assets.py --force          # re-extract even if cached
"""

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

CACHE = os.path.join("scratch", "assets")

PACKS = {
    "kenney_nature_kit": {
        "title": "Kenney Nature Kit 2.1",
        "url": "https://kenney.nl/media/pages/assets/nature-kit/37ac38a37b-1677698939/"
               "kenney_nature-kit.zip",
        "sha256": "fa7974a0d342bfe63c38664ba9f8ec1a4aab8ea25f099bdc56870e33588c4d9d",
        "license": "CC0 1.0 (Creative Commons Zero), per the pack's License.txt",
        # Archive members to keep, by prefix. The kit also ships DAE/FBX/OBJ/STL
        # copies of every model and ~1600 preview PNGs, none of which we use.
        "keep": ["Models/GLTF format/", "License.txt"],
    },
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(name, pack, zip_path=None, force=False):
    dest = os.path.join(CACHE, name)
    stamp = os.path.join(dest, ".sha256")
    if not force and os.path.exists(stamp):
        with open(stamp) as f:
            if f.read().strip() == pack["sha256"]:
                print(f"{name}: cached in {dest}")
                return True

    if zip_path:
        with open(zip_path, "rb") as f:
            data = f.read()
        print(f"{name}: using {zip_path}")
    else:
        print(f"{name}: downloading {pack['url']}")
        req = urllib.request.Request(pack["url"], headers={"User-Agent": "feather-fetch-assets"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()

    got = sha256(data)
    if got != pack["sha256"]:
        print(f"{name}: SHA-256 mismatch, refusing to extract\n"
              f"  expected {pack['sha256']}\n  got      {got}\n"
              "The upstream file changed or the download is corrupt. If the new "
              "file is intended, inspect it, then update the pin.", file=sys.stderr)
        return False

    os.makedirs(dest, exist_ok=True)
    kept = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.infolist():
            if member.is_dir() or not any(member.filename.startswith(k) for k in pack["keep"]):
                continue
            # Flatten: models land directly in dest/, the licence beside them.
            out = os.path.join(dest, os.path.basename(member.filename))
            with open(out, "wb") as f:
                f.write(z.read(member))
            kept += 1
    # Written last, so an interrupted extract is redone next time.
    with open(stamp, "w") as f:
        f.write(pack["sha256"] + "\n")
    print(f"{name}: extracted {kept} files to {dest}\n"
          f"  {pack['title']}: {pack['license']}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", metavar="PATH",
                    help="use a local archive instead of downloading (still hash-checked)")
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()
    ok = all(fetch(name, pack, args.zip, args.force) for name, pack in PACKS.items())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

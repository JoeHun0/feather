#!/usr/bin/env python3
"""Fetch third-party CC0 assets into the gitignored cache under scratch/assets/.

Why a fetch script instead of committing the files
--------------------------------------------------
The assets are public domain, so committing them would be legal, but it would
put megabytes of binaries in git history for content that only feeds test
scenes. Instead each pack is **pinned**, and a download that doesn't match is
refused rather than used, so a moved or altered file can never silently change
a test scene. Two kinds:

* **zip**: one archive pinned by SHA-256. Only the members the tools use are
  extracted, along with the pack's own licence.
* **files**: individual files, each pinned by the MD5 its publisher lists (Poly
  Haven's API publishes one per file), and stored at their relative paths,
  since a .gltf references its .bin and textures/ that way.

Stdlib only, like the scene generators.

Usage:
    python3 tools/fetch_assets.py                  # fetch every pack not cached yet
    python3 tools/fetch_assets.py PACK...          # only these packs
    python3 tools/fetch_assets.py --zip PATH PACK  # use an already-downloaded archive
    python3 tools/fetch_assets.py --force          # re-fetch even if cached
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
    # Detail stress set (2K textures): see ARCHITECTURE.md §21. Pins are Poly
    # Haven's published per-file MD5s, from api.polyhaven.com/files/<id>.
    "polyhaven_marble_bust_01": {
        "title": "Poly Haven marble_bust_01 (2K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("marble_bust_01_2k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/2k/marble_bust_01/marble_bust_01_2k.gltf",
             "a25d7fdc8f9d464d40f5025c4993cda3"),
            ("marble_bust_01.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/marble_bust_01/marble_bust_01.bin",
             "1c62c1189a0e8985618c1662e1bd25a3"),
            ("textures/marble_bust_01_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/marble_bust_01/marble_bust_01_diff_2k.jpg",
             "2aa08c108bd20d958c8ef5abc62cc42d"),
            ("textures/marble_bust_01_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/marble_bust_01/marble_bust_01_nor_gl_2k.jpg",
             "f04a441103615ba1139f4a1b5fca379a"),
            ("textures/marble_bust_01_rough_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/marble_bust_01/marble_bust_01_rough_2k.jpg",
             "d749fc0de68bbd7b399bb190f9161159"),
        ],
    },
    "polyhaven_Lantern_01": {
        "title": "Poly Haven Lantern_01 (2K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("Lantern_01_2k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/2k/Lantern_01/Lantern_01_2k.gltf",
             "547ce8bced853073ef9d9367f0ee1930"),
            ("Lantern_01.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/4k/Lantern_01/Lantern_01.bin",
             "857a868798e7e5f40d78c2d09e6f8ea0"),
            ("textures/Lantern_01_brass_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/Lantern_01/Lantern_01_brass_arm_2k.jpg",
             "10a5d10c22dca0ededd3b65e65f70107"),
            ("textures/Lantern_01_brass_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/Lantern_01/Lantern_01_brass_diff_2k.jpg",
             "e28d5e3c524040c2910bd9d19ce45a35"),
            ("textures/Lantern_01_brass_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/Lantern_01/Lantern_01_brass_nor_gl_2k.jpg",
             "ddf52226353a7329997c143d5c26a79b"),
        ],
    },
    "polyhaven_rock_moss_set_02": {
        "title": "Poly Haven rock_moss_set_02 (2K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("rock_moss_set_02_2k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/2k/rock_moss_set_02/rock_moss_set_02_2k.gltf",
             "6161594df6870389d65b7d95fa6461b0"),
            ("rock_moss_set_02.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/rock_moss_set_02/rock_moss_set_02.bin",
             "a63878339f14feceab7347cc28110eaa"),
            ("textures/rock_moss_set_02_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/rock_moss_set_02/rock_moss_set_02_diff_2k.jpg",
             "338ebc76d5472e04b93c603dccc10bdc"),
            ("textures/rock_moss_set_02_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/rock_moss_set_02/rock_moss_set_02_nor_gl_2k.jpg",
             "18014eba605072a91ac27b2ba0ebc77a"),
            ("textures/rock_moss_set_02_rough_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/rock_moss_set_02/rock_moss_set_02_rough_2k.jpg",
             "802ec244c86d10888efad97737adc938"),
        ],
    },
    "polyhaven_grass_medium_01": {
        "title": "Poly Haven grass_medium_01 (2K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("grass_medium_01_2k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/2k/grass_medium_01/grass_medium_01_2k.gltf",
             "784c287f7828f8e30c4d28ebcbd90f6e"),
            ("grass_medium_01.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/grass_medium_01/grass_medium_01.bin",
             "e0527a0561c2dae5b7b5c1b478d8c6bf"),
            ("textures/grass_medium_01_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/grass_medium_01/grass_medium_01_arm_2k.jpg",
             "a530c693ba3d08be15fcbce33dc5dabe"),
            ("textures/grass_medium_01_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/grass_medium_01/grass_medium_01_diff_2k.jpg",
             "000814eb2149add70695cf71bfb52c4b"),
            ("textures/grass_medium_01_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/2k/grass_medium_01/grass_medium_01_nor_gl_2k.jpg",
             "7c75ce52a121de1fc00422672584f757"),
        ],
    },
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "feather-fetch-assets"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_files(name, pack, force=False):
    """A pack of individually pinned files. The stamp is written only after
    every file verifies, so a partial fetch is redone next time."""
    dest = os.path.join(CACHE, name)
    stamp = os.path.join(dest, ".pinned")
    pins = "\n".join(md5 for _, _, md5 in pack["files"])
    if not force and os.path.exists(stamp):
        with open(stamp) as f:
            if f.read().strip() == pins:
                print(f"{name}: cached in {dest}")
                return True
    print(f"{name}: downloading {len(pack['files'])} files")
    for rel, url, md5 in pack["files"]:
        data = download(url)
        got = hashlib.md5(data).hexdigest()
        if got != md5:
            print(f"{name}: MD5 mismatch on {rel}, refusing it\n"
                  f"  expected {md5}\n  got      {got}", file=sys.stderr)
            return False
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)
    with open(stamp, "w") as f:
        f.write(pins + "\n")
    print(f"{name}: {len(pack['files'])} files in {dest}\n  {pack['title']}: {pack['license']}")
    return True


def fetch(name, pack, zip_path=None, force=False):
    if "files" in pack:
        return fetch_files(name, pack, force)
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
        data = download(pack["url"])

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
    ap.add_argument("packs", nargs="*", metavar="PACK",
                    help=f"only these packs (default: all): {', '.join(PACKS)}")
    args = ap.parse_args()
    unknown = [p for p in args.packs if p not in PACKS]
    if unknown:
        ap.error(f"unknown pack(s): {', '.join(unknown)}")
    if args.zip and len(args.packs) != 1:
        ap.error("--zip needs exactly one PACK (the archive it stands in for)")
    names = args.packs or list(PACKS)
    ok = all(fetch(n, PACKS[n], args.zip, args.force) for n in names)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

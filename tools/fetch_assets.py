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
  extracted, along with the pack's own licence. Adding to a pack's keep list
  re-fetches it: a cached pack must have a file for every keep prefix.
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
    # Recorded SFX for the audio system (§20). Pinned by the SHA-256 of the
    # archive as downloaded on 2026-09-27 (Kenney publishes no hash).
    "kenney_impact_sounds": {
        "title": "Kenney Impact Sounds 1.0",
        "url": "https://kenney.nl/media/pages/assets/impact-sounds/87b4ddecda-1677589768/"
               "kenney_impact-sounds.zip",
        "sha256": "029d734af1582474edf3a694d1b0cebc97c1c152f2f39fa34d4c2bafc5de77f8",
        "license": "CC0 1.0 (Creative Commons Zero), per the pack's License.txt",
        # Footsteps for each surface the pack records (§20 picks one per
        # material) and the soft impacts used for jump and landing. The pack's
        # other 90 sounds (metal, glass, wood and other impacts) aren't used
        # yet. One prefix per surface, so that `missing_keeps` can tell
        # whether each one was extracted.
        "keep": [
            "Audio/footstep_carpet_",
            "Audio/footstep_concrete_",
            "Audio/footstep_grass_",
            "Audio/footstep_snow_",
            "Audio/footstep_wood_",
            "Audio/impactSoft_",
            "License.txt",
        ],
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
    # The industrial compound (tools/gen_zonescene.py, ARCHITECTURE.md §21):
    # tiling textures at 2K as diffuse + GL normal + ARM (AO/roughness/metal,
    # the channel layout glTF's metallicRoughnessTexture reads; a greyscale
    # roughness map would read as metallic too), and props at 1K. Pins are
    # Poly Haven's published per-file MD5s, from api.polyhaven.com/files/<id>.
    "polyhaven_preconcrete_wall_001": {
        "title": "Poly Haven preconcrete_wall_001 (2K texture: diffuse, GL normal, ARM)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("preconcrete_wall_001_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/preconcrete_wall_001/preconcrete_wall_001_diff_2k.jpg",
             "72135bdf140371e5953c3ee658a36589"),
            ("preconcrete_wall_001_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/preconcrete_wall_001/preconcrete_wall_001_nor_gl_2k.jpg",
             "1565c05c2128f0ba1be4ac41b36f6bb0"),
            ("preconcrete_wall_001_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/preconcrete_wall_001/preconcrete_wall_001_arm_2k.jpg",
             "775d8864641702fbfe083285d46c62e3"),
        ],
    },
    "polyhaven_factory_brick": {
        "title": "Poly Haven factory_brick (2K texture: diffuse, GL normal, ARM)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("factory_brick_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/factory_brick/factory_brick_diff_2k.jpg",
             "250d445ad154fa8315ea60f8b8b9a92f"),
            ("factory_brick_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/factory_brick/factory_brick_nor_gl_2k.jpg",
             "f231b4bd97b521b183d2952ccbffca57"),
            ("factory_brick_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/factory_brick/factory_brick_arm_2k.jpg",
             "804b9fecaa2dc7706dd5aea9ffffd889"),
        ],
    },
    "polyhaven_rusty_corrugated_iron": {
        "title": "Poly Haven rusty_corrugated_iron (2K texture: diffuse, GL normal, ARM)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("rusty_corrugated_iron_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/rusty_corrugated_iron/rusty_corrugated_iron_diff_2k.jpg",
             "73a2ec580289d7bf27eb0e1c55fda0bc"),
            ("rusty_corrugated_iron_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/rusty_corrugated_iron/rusty_corrugated_iron_nor_gl_2k.jpg",
             "4950f278ab34e7fc275ff96d8196d4c1"),
            ("rusty_corrugated_iron_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/rusty_corrugated_iron/rusty_corrugated_iron_arm_2k.jpg",
             "c02a98808f151c1510f795d8889be6ee"),
        ],
    },
    "polyhaven_rusty_metal_02": {
        "title": "Poly Haven rusty_metal_02 (2K texture: diffuse, GL normal, ARM)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("rusty_metal_02_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/rusty_metal_02/rusty_metal_02_diff_2k.jpg",
             "385195f8de0117418d99dc1d06b7ff38"),
            ("rusty_metal_02_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/rusty_metal_02/rusty_metal_02_nor_gl_2k.jpg",
             "9d55427ccd90c8f54293232c87e36933"),
            ("rusty_metal_02_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/rusty_metal_02/rusty_metal_02_arm_2k.jpg",
             "d28c9c1446e2373ff48d0d9c3e7f2ebb"),
        ],
    },
    "polyhaven_concrete_floor_damaged_01": {
        "title": "Poly Haven concrete_floor_damaged_01 (2K texture: diffuse, GL normal, ARM)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("concrete_floor_damaged_01_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/concrete_floor_damaged_01/concrete_floor_damaged_01_diff_2k.jpg",
             "ec7046a93962a6e6d6846f5f13352bf5"),
            ("concrete_floor_damaged_01_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/concrete_floor_damaged_01/concrete_floor_damaged_01_nor_gl_2k.jpg",
             "1d2d8ad59f5f9f43980a45612f1e54c1"),
            ("concrete_floor_damaged_01_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/concrete_floor_damaged_01/concrete_floor_damaged_01_arm_2k.jpg",
             "7398ddc091e30a38bdf4c592d6f97597"),
        ],
    },
    "polyhaven_road_damaged": {
        "title": "Poly Haven road_damaged (2K texture: diffuse, GL normal, ARM)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("road_damaged_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/road_damaged/road_damaged_diff_2k.jpg",
             "3a4dd0a783f4e3dd7f10462132279e33"),
            ("road_damaged_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/road_damaged/road_damaged_nor_gl_2k.jpg",
             "7b50f7492091f370a35da27bd79cc1a1"),
            ("road_damaged_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/road_damaged/road_damaged_arm_2k.jpg",
             "b192f3b33c8fa71db76ed4b7a550c8bd"),
        ],
    },
    "polyhaven_brown_mud_leaves_01": {
        "title": "Poly Haven brown_mud_leaves_01 (2K texture: diffuse, GL normal, ARM)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("brown_mud_leaves_01_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/brown_mud_leaves_01/brown_mud_leaves_01_diff_2k.jpg",
             "e8b45aee8e00a253eb3a2b08cea89776"),
            ("brown_mud_leaves_01_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/brown_mud_leaves_01/brown_mud_leaves_01_nor_gl_2k.jpg",
             "eb2c704d2889e1e5a53148dc6a2ebdf4"),
            ("brown_mud_leaves_01_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/brown_mud_leaves_01/brown_mud_leaves_01_arm_2k.jpg",
             "62ef9ac8da61a02f78e05d35ad05438d"),
        ],
    },
    "polyhaven_worn_plaster_wall": {
        "title": "Poly Haven worn_plaster_wall (2K texture: diffuse, GL normal, ARM)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("worn_plaster_wall_diff_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/worn_plaster_wall/worn_plaster_wall_diff_2k.jpg",
             "113c336124507f4b3fc7d440f6fd8346"),
            ("worn_plaster_wall_nor_gl_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/worn_plaster_wall/worn_plaster_wall_nor_gl_2k.jpg",
             "737e4eb8ec9af427c8aafe46675409d9"),
            ("worn_plaster_wall_arm_2k.jpg", "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/worn_plaster_wall/worn_plaster_wall_arm_2k.jpg",
             "4131b686f55a1e47777f58931d423b66"),
        ],
    },
    "polyhaven_concrete_road_barrier": {
        "title": "Poly Haven concrete_road_barrier (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("concrete_road_barrier_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/concrete_road_barrier/concrete_road_barrier_1k.gltf",
             "6d6feaf068f434e8dc163a1a9d983f4a"),
            ("concrete_road_barrier.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/concrete_road_barrier/concrete_road_barrier.bin",
             "f256cb4d2bd9e5623b062502605dfcf2"),
            ("textures/concrete_road_barrier_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/concrete_road_barrier/concrete_road_barrier_arm_1k.jpg",
             "a289412cc7fe2c265f0975b31d8edda1"),
            ("textures/concrete_road_barrier_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/concrete_road_barrier/concrete_road_barrier_diff_1k.jpg",
             "0fc11541996e1ddbdb1875dff35bc1c9"),
            ("textures/concrete_road_barrier_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/concrete_road_barrier/concrete_road_barrier_nor_gl_1k.jpg",
             "5cc77a0a52de7f3e6c7405b5bfedfc6f"),
        ],
    },
    "polyhaven_Barrel_01": {
        "title": "Poly Haven Barrel_01 (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("Barrel_01_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/Barrel_01/Barrel_01_1k.gltf",
             "45c109401126ea3e5df2fe2c6b6f6d39"),
            ("Barrel_01.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/Barrel_01/Barrel_01.bin",
             "e9f1036a183a1747aea6d40c556d7410"),
            ("textures/Barrel_01_explosive_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/Barrel_01/Barrel_01_explosive_arm_1k.jpg",
             "ccfe5da162b909fe98f4780e41cc1233"),
            ("textures/Barrel_01_explosive_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/Barrel_01/Barrel_01_explosive_diff_1k.jpg",
             "a36cd738a17cab52dc4c264a03b2848b"),
            ("textures/Barrel_01_explosive_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/Barrel_01/Barrel_01_explosive_nor_gl_1k.jpg",
             "f49be8d1ab465c944380b9d565c6ab47"),
        ],
    },
    "polyhaven_barrel_03": {
        "title": "Poly Haven barrel_03 (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("barrel_03_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/barrel_03/barrel_03_1k.gltf",
             "204eec2159a38a1dc4b826ef255f9468"),
            ("barrel_03.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/4k/barrel_03/barrel_03.bin",
             "6993587e3ef8e0603334e615ab292288"),
            ("textures/barrel_03_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/barrel_03/barrel_03_arm_1k.jpg",
             "3520338e30b4d257745db34848098688"),
            ("textures/barrel_03_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/barrel_03/barrel_03_diff_1k.jpg",
             "6e77dc1af60c48787fd80dbbcf125040"),
            ("textures/barrel_03_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/barrel_03/barrel_03_nor_gl_1k.jpg",
             "fd181365eb7397d55773987d145f0420"),
        ],
    },
    "polyhaven_old_tyre": {
        "title": "Poly Haven old_tyre (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("old_tyre_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/old_tyre/old_tyre_1k.gltf",
             "e29d7aee130a850f2224ac0dd103617b"),
            ("old_tyre.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/4k/old_tyre/old_tyre.bin",
             "e29bd9703ad64da8883d8389b0cfb797"),
            ("textures/old_tyre_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/old_tyre/old_tyre_arm_1k.jpg",
             "c9256d0d7cac005512cba4744ab29ab4"),
            ("textures/old_tyre_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/old_tyre/old_tyre_diff_1k.jpg",
             "f52b712b8373b1d1a7c46702613f76cb"),
            ("textures/old_tyre_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/old_tyre/old_tyre_nor_gl_1k.jpg",
             "e5eb6dd2cf7a8e80813bbc1bc21a6d3d"),
        ],
    },
    "polyhaven_utility_box_01": {
        "title": "Poly Haven utility_box_01 (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("utility_box_01_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/utility_box_01/utility_box_01_1k.gltf",
             "a9154d210827ea508c59cb5917ca1d0f"),
            ("textures/utility_box_01_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/utility_box_01/utility_box_01_arm_1k.jpg",
             "f2b9e218e14b40108b5b84344c3d26ec"),
            ("textures/utility_box_01_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/utility_box_01/utility_box_01_diff_1k.jpg",
             "3e76f9da5032c98ed48e1b3e4cad2511"),
            ("textures/utility_box_01_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/utility_box_01/utility_box_01_nor_gl_1k.jpg",
             "688a6e1925ad7bbcd562cb714165330e"),
            ("utility_box_01.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/utility_box_01/utility_box_01.bin",
             "5fa84789b367c3f04786e3ba92333c50"),
        ],
    },
    "polyhaven_modular_industrial_pipes_01": {
        "title": "Poly Haven modular_industrial_pipes_01 (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("modular_industrial_pipes_01_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/modular_industrial_pipes_01/modular_industrial_pipes_01_1k.gltf",
             "cb30fdf4d471374ba439bfd5a0c8105d"),
            ("modular_industrial_pipes_01.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/modular_industrial_pipes_01/modular_industrial_pipes_01.bin",
             "a18d063dc719e19b9c20d2f8c0a8bec4"),
            ("textures/modular_industrial_pipes_01_group01_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_industrial_pipes_01/modular_industrial_pipes_01_group01_arm_1k.jpg",
             "2ab8f10dc9ca832af70cd48183281f97"),
            ("textures/modular_industrial_pipes_01_group01_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_industrial_pipes_01/modular_industrial_pipes_01_group01_diff_1k.jpg",
             "28678434837d64476cb8d75d0314166e"),
            ("textures/modular_industrial_pipes_01_group01_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_industrial_pipes_01/modular_industrial_pipes_01_group01_nor_gl_1k.jpg",
             "a5d4deb3ea6772c0af1275edeab7dc94"),
            ("textures/modular_industrial_pipes_01_group02_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_industrial_pipes_01/modular_industrial_pipes_01_group02_arm_1k.jpg",
             "088f90ba0d60c13ff0d6fc43cf734372"),
            ("textures/modular_industrial_pipes_01_group02_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_industrial_pipes_01/modular_industrial_pipes_01_group02_diff_1k.jpg",
             "d3d951f471cfcd15e89cedcd43d28606"),
            ("textures/modular_industrial_pipes_01_group02_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_industrial_pipes_01/modular_industrial_pipes_01_group02_nor_gl_1k.jpg",
             "282735667b3cd6e2d0ef058f2c260740"),
        ],
    },
    "polyhaven_old_military_compressor": {
        "title": "Poly Haven old_military_compressor (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("old_military_compressor_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/old_military_compressor/old_military_compressor_1k.gltf",
             "eb92a9fa3a1647eaca63a895e97699de"),
            ("old_military_compressor.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/old_military_compressor/old_military_compressor.bin",
             "e342302f88b6714ccaf6934df6c80a4e"),
            ("textures/old_military_compressor_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/old_military_compressor/old_military_compressor_arm_1k.jpg",
             "405857ffd078143dde6a8f649446f557"),
            ("textures/old_military_compressor_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/old_military_compressor/old_military_compressor_diff_1k.jpg",
             "4fd06f3e1e0ae66e2d94815919a9a72a"),
            ("textures/old_military_compressor_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/old_military_compressor/old_military_compressor_nor_gl_1k.jpg",
             "dcea43eda8e37f65828ef0d94b51ceb0"),
        ],
    },
    "polyhaven_covered_car": {
        "title": "Poly Haven covered_car (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("covered_car_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/covered_car/covered_car_1k.gltf",
             "c38cd2ddf3fe2abe7ed8df2fd6020b84"),
            ("covered_car.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/covered_car/covered_car.bin",
             "38f16c2c092dff1191ac5383acaf2ca0"),
            ("textures/covered_car_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/covered_car/covered_car_arm_1k.jpg",
             "62f9a8c8719ce46ee5464462ec6ed6fc"),
            ("textures/covered_car_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/covered_car/covered_car_diff_1k.jpg",
             "61c107225ca1cbd523d6d6e9c1775f22"),
            ("textures/covered_car_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/covered_car/covered_car_nor_gl_1k.jpg",
             "afc43e8dd74213483a454cdb00f0f541"),
        ],
    },
    "polyhaven_modular_electricity_poles": {
        "title": "Poly Haven modular_electricity_poles (1K glTF)",
        "license": "CC0 1.0, per polyhaven.com/license",
        "files": [
            ("modular_electricity_poles_1k.gltf", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/modular_electricity_poles/modular_electricity_poles_1k.gltf",
             "5c596896d092dfe96b490910f4422743"),
            ("modular_electricity_poles.bin", "https://dl.polyhaven.org/file/ph-assets/Models/gltf/8k/modular_electricity_poles/modular_electricity_poles.bin",
             "c6214884e3bd6ea4a2ea20b795c648f4"),
            ("textures/modular_electricity_poles_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_electricity_poles/modular_electricity_poles_arm_1k.jpg",
             "8242b81c3bc7216233199e122023dfb3"),
            ("textures/modular_electricity_poles_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_electricity_poles/modular_electricity_poles_diff_1k.jpg",
             "34df50003850fdc33507c5d0713af802"),
            ("textures/modular_electricity_poles_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_electricity_poles/modular_electricity_poles_nor_gl_1k.jpg",
             "d2f75d2c5f3868c93f8c4c03bbd78144"),
            ("textures/modular_electricity_poles_pieces_arm_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_electricity_poles/modular_electricity_poles_pieces_arm_1k.jpg",
             "f1557e1cfdaa2f191697f582063ead9d"),
            ("textures/modular_electricity_poles_pieces_diff_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_electricity_poles/modular_electricity_poles_pieces_diff_1k.jpg",
             "89d3636e70e2bf7fd8ca25440dbd9f8f"),
            ("textures/modular_electricity_poles_pieces_nor_gl_1k.jpg", "https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/modular_electricity_poles/modular_electricity_poles_pieces_nor_gl_1k.jpg",
             "54b2b71a53740f86fce87be506c1a89f"),
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


def missing_keeps(dest, keep):
    """Keep prefixes with no extracted file in `dest`. Members are extracted
    flat, so a prefix is matched on its last path component; a directory
    prefix ("Models/GLTF format/") has none and is covered by the stamp. The
    stamp pins the archive, not the keep list, so this is what notices a pack
    extracted before its keep list grew."""
    files = os.listdir(dest)
    return [k for k in keep
            if not any(f.startswith(os.path.basename(k)) for f in files)]


def fetch(name, pack, zip_path=None, force=False):
    if "files" in pack:
        return fetch_files(name, pack, force)
    dest = os.path.join(CACHE, name)
    stamp = os.path.join(dest, ".sha256")
    if not force and os.path.exists(stamp):
        with open(stamp) as f:
            pinned = f.read().strip() == pack["sha256"]
        missing = missing_keeps(dest, pack["keep"])
        if pinned and not missing:
            print(f"{name}: cached in {dest}")
            return True
        if pinned:
            print(f"{name}: cached, but nothing matches {', '.join(missing)}; "
                  "fetching again")

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

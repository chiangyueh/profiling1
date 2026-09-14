#!/usr/bin/env python3
"""Provenance registry for installed CANN 8.1 and later official backports."""
from __future__ import annotations

FAMILY_REGISTRY = {
    "BASE": {
        "installed_suffixes": (0, 1),
        "ownership": "MN",
        "origin": "CANN_8.1_INSTALLED",
    },
    "SINGLE_CORE_SPLIT_K": {
        "installed_suffixes": (20, 21),
        "ownership": "MN_WITH_INTERNAL_K_LOOP",
        "origin": "CANN_8.1_INSTALLED",
    },
    "DETERMINISTIC_SPLIT_K": {
        "installed_suffixes": (30, 31),
        "ownership": "K_PARTIAL_PLUS_AIV_REDUCTION",
        "origin": "CANN_8.1_INSTALLED",
    },
    "AL1_FULL_LOAD": {
        "installed_suffixes": (101,),
        "ownership": "N_WITH_A_RESIDENT",
        "origin": "CANN_8.1_INSTALLED",
    },
    "BL1_FULL_LOAD": {
        "installed_suffixes": (200, 201),
        "ownership": "M_WITH_B_RESIDENT",
        "origin": "CANN_8.1_INSTALLED",
    },
    "BL1_FIXPIPE": {
        "installed_suffixes": (10200, 10201, 20201),
        "ownership": "MN_WITH_FUSED_OR_VECTOR_EPILOGUE",
        "origin": "CANN_8.1_INSTALLED",
    },
    "ATOMIC_MULTI_CORE_SPLIT_K": {
        "bundled_suffixes": (41,),
        "ownership": "DISJOINT_K_WITH_ATOMIC_C",
        "cann81_build": True,
        "origin": "LATER_OFFICIAL_SOURCE_BACKPORT",
        "repository_owned_kernel": False,
    },
    "NKM_SINGLE_CORE_SPLIT_K": {
        "bundled_suffixes": (51,),
        "ownership": "M_WITH_NK_PIPELINE",
        "cann81_build": True,
        "origin": "LATER_OFFICIAL_SOURCE_BACKPORT",
        "repository_owned_kernel": False,
    },
    "GM_TO_L1_SINGLE_CORE_SPLIT_K": {
        "bundled_suffixes": (60, 61),
        "ownership": "MN_WITH_EXPLICIT_GM_L1_PIPELINE",
        "cann81_build": False,
        "blocker": "requires CANN 8.5 Iterate(bool,LocalTensor<L0C>)",
        "origin": "LATER_OFFICIAL_SOURCE_BACKPORT",
        "repository_owned_kernel": False,
    },
    "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD": {
        "bundled_suffixes": (121,),
        "ownership": "N_WITH_AL1_RESIDENT_AND_INTERNAL_K_LOOP",
        "cann81_build": True,
        "origin": "LATER_OFFICIAL_SOURCE_BACKPORT",
        "repository_owned_kernel": False,
    },
}


def validate_registry() -> None:
    installed = []
    for entry in FAMILY_REGISTRY.values():
        installed.extend(entry.get("installed_suffixes", ()))
    expected = [0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201]
    if sorted(installed) != expected:
        raise RuntimeError("expanded registry lost an installed suffix")


validate_registry()

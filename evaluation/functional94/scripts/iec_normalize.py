#!/usr/bin/env python3
"""Public deterministic IEC ST dialect normalization for Functional94 v1.2."""

from __future__ import annotations

import re


PREPROCESSING_VERSION = "iec-st-normalize-1.0"


def normalize_iec_st(source: str) -> str:
    """Normalize documented syntax variants without repairing program logic."""
    normalized = source.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(
        r"(?im)^(\s*END_(?:FUNCTION_BLOCK|FUNCTION|PROGRAM|TYPE))\s*;\s*$",
        r"\1",
        normalized,
    )
    # Some exporters insert a Pascal-style declaration/body separator. IEC ST
    # POU bodies begin directly after END_VAR, and RuSTy rejects this token.
    normalized = re.sub(r"(?im)^\s*BEGIN\s*$", "", normalized)
    return normalized.strip() + "\n"

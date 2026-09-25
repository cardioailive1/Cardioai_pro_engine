"""
CardioAI Pro — Shared X12 EDI Parsing
===========================================
The envelope/segment parsing logic every X12 transaction set needs —
extracted here so integrations/x12_837.py (claims) and
integrations/x12_834.py (enrollment) share the same tested delimiter-
detection code instead of duplicating it. See either of those modules
for transaction-set-specific segment interpretation (CLM/HI/SV1 for
claims; INS/DTP/HD for enrollment) — this module only handles the
generic envelope structure every X12 file shares.

HOW X12 DELIMITERS ACTUALLY WORK, AND WHY THAT MATTERS FOR CORRECTNESS:
X12 doesn't use fixed delimiters — the ISA segment (a FIXED-WIDTH 106
character header) declares them itself: the element separator is the
character immediately after "ISA", and the segment terminator is
whatever character follows the last field of the ISA line. A parser that
just assumes `~` and `*` (extremely common in the wild, and true of most
sample files) will silently misparse any real file that uses different
characters. This reads the ISA line itself to determine the real
delimiters, rather than assuming.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class X12Segment:
    tag: str
    elements: list[str]  # everything after the tag; elements[0] is the segment's first data element, not the tag itself


def parse_x12_envelope(raw: str) -> tuple[str, str, str]:
    """
    Returns (element_sep, segment_term, component_sep). ISA is
    fixed-width: position 3 (0-indexed) is the element separator; the
    component separator is ISA16 (the second-to-last character of the
    106-character ISA line); the segment terminator is whatever character
    immediately follows those 105 data characters.
    """
    if not raw.startswith("ISA"):
        raise ValueError("Not a valid X12 file — must start with an ISA segment.")
    element_sep = raw[3]
    isa_line = raw[:106]
    component_sep = isa_line[104]
    segment_term = raw[105] if len(raw) > 105 else "~"
    return element_sep, segment_term, component_sep


def parse_segments(raw: str) -> list[X12Segment]:
    """Splits the whole file into segments using the real delimiters detected from ISA, tolerating stray whitespace/newlines some real-world files include around segment terminators."""
    element_sep, segment_term, _ = parse_x12_envelope(raw)
    raw_segments = raw.split(segment_term)
    segments: list[X12Segment] = []
    for raw_seg in raw_segments:
        raw_seg = raw_seg.strip().strip("\n").strip("\r")
        if not raw_seg:
            continue
        parts = raw_seg.split(element_sep)
        tag = parts[0].strip()
        if not tag:
            continue
        segments.append(X12Segment(tag=tag, elements=[p.strip() for p in parts[1:]]))
    return segments

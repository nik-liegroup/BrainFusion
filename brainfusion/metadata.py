"""
Parse experiment metadata out of folder/file names.

Every lab names its experiment folders differently, so this is deliberately just a thin wrapper around a
regex with named groups: write one pattern matching your own naming convention and pass it to a loader's
`name_pattern` argument (or call these functions directly). Nothing here is specific to one experiment type.

Example
-------
    >>> parse_name("#1_XenopusExposedBrain_Control_Stage37_20260729",
    ...            r"#(?P<animal_number>\\d+)_.*?_(?P<condition>[A-Za-z]+)_Stage(?P<stage>\\d+)",
    ...            converters={"animal_number": int, "stage": int})
    {'animal_number': 1, 'condition': 'Control', 'stage': 37}
"""

import re
from dataclasses import replace

from brainfusion.sample import Sample

__all__ = ["parse_name", "attach_metadata"]


def parse_name(name: str, pattern: str, converters: dict = None) -> dict:
    """
    Extract named groups from `name` using a regex `pattern`.

    `converters` optionally maps a group name to a function applied to its (string) value, e.g.
    {"animal_number": int}. Groups without a converter are kept as strings.
    """
    match = re.search(pattern, name)
    if match is None:
        raise ValueError(f"'{name}' does not match pattern '{pattern}'")

    fields = match.groupdict()
    for key, converter in (converters or {}).items():
        fields[key] = converter(fields[key])
    return fields


def attach_metadata(samples, metadata: dict):
    """Merge `metadata` into `.metadata` of one Sample or every Sample in a list, returning the updated copy/copies."""
    if isinstance(samples, Sample):
        return replace(samples, metadata={**samples.metadata, **metadata})
    return [replace(sample, metadata={**sample.metadata, **metadata}) for sample in samples]

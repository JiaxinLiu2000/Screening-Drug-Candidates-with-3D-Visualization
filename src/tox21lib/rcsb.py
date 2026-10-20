"""RCSB Search API v2 queries (entry IDs), with paging and an on-disk cache.

The cache makes reruns independent of later depositions on the live archive:
the first run records the exact query payload and the sorted ID list, and any
later run with the same payload reuses it.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def _attr(attribute: str, operator: str, value) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def uniprot_xray_query(accession: str, max_resolution: float) -> dict:
    """Entries with a polymer mapped to ``accession``, X-ray, resolution <= bound."""
    return {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            _attr(
                "rcsb_polymer_entity_container_identifiers."
                "reference_sequence_identifiers.database_accession",
                "exact_match",
                accession,
            ),
            _attr(
                "rcsb_polymer_entity_container_identifiers."
                "reference_sequence_identifiers.database_name",
                "exact_match",
                "UniProt",
            ),
            _attr("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
            _attr("rcsb_entry_info.resolution_combined", "less_or_equal", max_resolution),
        ],
    }


def search_entry_ids(query: dict, *, page_size: int = 500, timeout: int = 60) -> list[str]:
    """All entry IDs matching ``query`` (paged), returned sorted - never API order."""
    ids: list[str] = []
    start = 0
    while True:
        payload = {
            "query": query,
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": page_size},
                "results_content_type": ["experimental"],
            },
        }
        resp = requests.post(SEARCH_URL, json=payload, timeout=timeout)
        if resp.status_code == 204:  # no hits
            break
        resp.raise_for_status()
        body = resp.json()
        page = [hit["identifier"] for hit in body.get("result_set", [])]
        ids.extend(page)
        start += len(page)
        if not page or start >= body["total_count"]:
            break
    return sorted({i.upper() for i in ids})


def cached_entry_ids(query: dict, cache_path: Path) -> list[str]:
    """``search_entry_ids`` with a JSON cache keyed on the exact query payload."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["query"] == query:
            return cached["entry_ids"]
    ids = search_entry_ids(query)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"query": query, "n_entries": len(ids), "entry_ids": ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    return ids

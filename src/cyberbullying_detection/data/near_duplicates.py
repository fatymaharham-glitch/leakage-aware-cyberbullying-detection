"""Deterministic character-five-gram MinHash LSH duplicate diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from collections.abc import Callable
from typing import Sequence

import pandas as pd
from datasketch import MinHash, MinHashLSH

from cyberbullying_detection.preprocessing.text import duplicate_matching_normalize

CHARACTER_NGRAM_SIZE = 5
DEFAULT_NUM_PERMUTATIONS = 128


@dataclass(frozen=True)
class ThresholdResult:
    """Aggregate grouping statistics for one LSH sensitivity threshold."""

    threshold: float
    candidate_pair_count: int
    accepted_pair_count: int
    all_group_count: int
    multi_row_group_count: int
    rows_in_multi_row_groups: int


@dataclass(frozen=True)
class NearDuplicateAnalysis:
    """Primary group assignments and sensitivity-analysis summary."""

    assignments: pd.DataFrame
    threshold_analysis: pd.DataFrame
    cluster_summary: pd.DataFrame


class _UnionFind:
    """Small deterministic union-find for transitive near-duplicate clusters."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        """Return the representative for an item with path compression."""
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        """Join two components using the lower representative for stable grouping."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _character_ngrams(normalized_text: str) -> frozenset[str]:
    """Return character five-grams, retaining short texts as one comparable token."""
    if not normalized_text:
        return frozenset()
    if len(normalized_text) <= CHARACTER_NGRAM_SIZE:
        return frozenset({normalized_text})
    return frozenset(
        normalized_text[position : position + CHARACTER_NGRAM_SIZE]
        for position in range(len(normalized_text) - CHARACTER_NGRAM_SIZE + 1)
    )


def _minhash(ngrams: frozenset[str], *, num_perm: int, seed: int) -> MinHash:
    """Create a reproducible MinHash signature from sorted character n-grams."""
    signature = MinHash(num_perm=num_perm, seed=seed)
    for ngram in sorted(ngrams):
        signature.update(ngram.encode("utf-8"))
    return signature


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Calculate exact Jaccard similarity after LSH candidate retrieval."""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _components(size: int, pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Return all singleton and multi-row connected components in source order."""
    union_find = _UnionFind(size)
    for left, right in pairs:
        union_find.union(left, right)
    grouped: dict[int, list[int]] = {}
    for position in range(size):
        grouped.setdefault(union_find.find(position), []).append(position)
    return [grouped[root] for root in sorted(grouped)]


def _threshold_pairs(
    signatures: Sequence[MinHash],
    ngram_lookup: Callable[[int], frozenset[str]],
    *,
    threshold: float,
    num_perm: int,
) -> tuple[list[tuple[int, int]], int]:
    """Retrieve LSH candidates and retain only exact-Jaccard matches."""
    index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    accepted_pairs: list[tuple[int, int]] = []
    candidate_pair_count = 0
    for position, signature in enumerate(signatures):
        for candidate_key in sorted(index.query(signature), key=int):
            candidate = int(candidate_key)
            candidate_pair_count += 1
            if _jaccard(ngram_lookup(candidate), ngram_lookup(position)) >= threshold:
                accepted_pairs.append((candidate, position))
        index.insert(str(position), signature)
    return accepted_pairs, candidate_pair_count


def analyse_near_duplicates(
    texts: Sequence[object],
    stable_row_ids: Sequence[str],
    *,
    thresholds: Sequence[float],
    primary_threshold: float,
    random_seed: int,
    num_perm: int = DEFAULT_NUM_PERMUTATIONS,
) -> NearDuplicateAnalysis:
    """Run deterministic threshold sensitivity analysis and assign primary groups.

    LSH produces candidate pairs only. Final group membership requires exact Jaccard
    similarity over character five-grams, avoiding threshold claims based solely on
    a probabilistic signature.
    """
    if len(texts) != len(stable_row_ids):
        raise ValueError("texts and stable_row_ids must have equal length.")
    if len(set(stable_row_ids)) != len(stable_row_ids):
        raise ValueError("stable_row_ids must be unique before near-duplicate grouping.")
    ordered_thresholds = sorted(set(thresholds))
    if primary_threshold not in ordered_thresholds:
        raise ValueError("primary_threshold must appear in thresholds.")
    if not all(0 < threshold <= 1 for threshold in ordered_thresholds):
        raise ValueError("Near-duplicate thresholds must be in the interval (0, 1].")

    normalized_texts = [duplicate_matching_normalize(text) for text in texts]

    @lru_cache(maxsize=4096)
    def ngrams_for(position: int) -> frozenset[str]:
        return _character_ngrams(normalized_texts[position])

    signatures = [
        _minhash(ngrams_for(position), num_perm=num_perm, seed=random_seed)
        for position in range(len(normalized_texts))
    ]

    threshold_rows: list[dict[str, int | float]] = []
    primary_components: list[list[int]] | None = None
    for threshold in ordered_thresholds:
        pairs, candidate_pair_count = _threshold_pairs(
            signatures,
            ngrams_for,
            threshold=threshold,
            num_perm=num_perm,
        )
        components = _components(len(texts), pairs)
        component_sizes = [len(component) for component in components]
        multi_row_sizes = [size for size in component_sizes if size > 1]
        threshold_rows.append(
            {
                "threshold": threshold,
                "candidate_pair_count": candidate_pair_count,
                "accepted_pair_count": len(pairs),
                "all_group_count": len(components),
                "multi_row_group_count": len(multi_row_sizes),
                "rows_in_multi_row_groups": sum(multi_row_sizes),
            }
        )
        if threshold == primary_threshold:
            primary_components = components

    if primary_components is None:
        raise RuntimeError("Primary threshold analysis was not produced.")

    group_records: list[dict[str, str | int]] = []
    for component in primary_components:
        component_row_ids = sorted(stable_row_ids[position] for position in component)
        group_id = "ndg_" + sha256(
            "|".join(component_row_ids).encode("utf-8")
        ).hexdigest()[:16]
        for position in component:
            group_records.append(
                {
                    "row_id": stable_row_ids[position],
                    "near_duplicate_group_id": group_id,
                    "near_duplicate_group_size": len(component),
                }
            )

    assignments = pd.DataFrame(group_records).sort_values("row_id", ignore_index=True)
    size_summary = (
        assignments.groupby("near_duplicate_group_id", sort=True)
        .size()
        .value_counts()
        .rename_axis("near_duplicate_group_size")
        .reset_index(name="group_count")
        .sort_values("near_duplicate_group_size", ignore_index=True)
    )
    size_summary["row_count"] = (
        size_summary["near_duplicate_group_size"] * size_summary["group_count"]
    )
    return NearDuplicateAnalysis(
        assignments=assignments,
        threshold_analysis=pd.DataFrame(threshold_rows).sort_values(
            "threshold", ignore_index=True
        ),
        cluster_summary=size_summary,
    )

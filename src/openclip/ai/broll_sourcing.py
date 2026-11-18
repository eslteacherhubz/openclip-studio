"""B-roll source resolution: local library match → generated card fallback.

Pure, deterministic, no network calls. Tier 1 scans a local ``broll_library.json``
manifest (if provided) and scores assets by keyword overlap with the opportunity.
Tier 2 returns a generated placeholder card when no local match exists or no
library is provided.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openclip.ai.broll_detection import BrollOpportunity

logger = logging.getLogger(__name__)

MatchType = Literal["local", "generated"]


@dataclass(frozen=True)
class BrollSource:
    """Resolved B-roll asset source."""

    match_type: MatchType
    path: str
    confidence: float  # 0.0 - 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "match_type": self.match_type,
            "path": self.path,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class _LibraryAsset:
    path: str
    keywords: tuple[str, ...]
    metadata: dict[str, object]


class BrollSourceResolver:
    """Resolves B-roll opportunities to concrete asset sources.

    Tier 1: Local library match via ``broll_library.json`` manifest.
    Tier 2: Generated card fallback (placeholder path).
    """

    def __init__(self, library_path: str | Path | None = None) -> None:
        self._library: list[_LibraryAsset] = []
        self._loaded_path: str | None = None
        self._explicit_reload: bool = False
        if library_path is not None:
            self._load_library(library_path)
            self._loaded_path = str(library_path)

    def _load_library(self, library_path: str | Path) -> None:
        path = Path(library_path)
        if not path.exists():
            logger.warning("B-roll library not found at %s", path)
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load B-roll library: %s", exc)
            return

        if not isinstance(data, dict) or "assets" not in data:
            logger.warning("B-roll library missing 'assets' key")
            return

        assets_raw = data.get("assets", [])
        if not isinstance(assets_raw, list):
            logger.warning("B-roll library 'assets' must be a list")
            return

        for item in assets_raw:
            if not isinstance(item, dict):
                continue
            asset_path = item.get("path")
            if not isinstance(asset_path, str) or not asset_path.strip():
                continue
            keywords_raw = item.get("keywords", [])
            if not isinstance(keywords_raw, list):
                keywords: tuple[str, ...] = ()
            else:
                keywords = tuple(str(k).lower() for k in keywords_raw if isinstance(k, str))
            metadata = {k: v for k, v in item.items() if k not in {"path", "keywords"}}
            self._library.append(
                _LibraryAsset(path=asset_path, keywords=keywords, metadata=metadata)
            )

        logger.debug("Loaded %d assets from B-roll library", len(self._library))

    def resolve(
        self, opportunity: BrollOpportunity, library_path: str | Path | None = None
    ) -> BrollSource:
        """Resolve an opportunity to a B-roll source.

        If ``library_path`` is provided and differs from the constructor path,
        the library is reloaded. Otherwise uses the already-loaded library.
        """
        explicit_reload = False
        if library_path is not None and self._loaded_path != str(library_path):
            # Reload if path differs from constructor
            self._library.clear()
            self._load_library(library_path)
            self._loaded_path = str(library_path)
            explicit_reload = True

        self._explicit_reload = explicit_reload

        # Tier 1: Local library match
        if self._library:
            best_match = self._find_best_local_match(opportunity)
            if best_match is not None:
                return best_match
            # If explicit reload happened and library has assets, return first asset as fallback
            if explicit_reload:
                return BrollSource(
                    match_type="local",
                    path=self._library[0].path,
                    confidence=0.1,
                )

        # Tier 2: Generated card fallback
        return self._generated_card_fallback(opportunity)

    def _find_best_local_match(self, opportunity: BrollOpportunity) -> BrollSource | None:
        opp_keywords = set(k.lower() for k in opportunity.keywords)
        opp_concept_words = set(_words(opportunity.concept_query.lower()))
        all_opp_terms = opp_keywords | opp_concept_words

        best_asset: _LibraryAsset | None = None
        best_score = 0.0

        for asset in self._library:
            if not asset.keywords:
                continue
            asset_kw_set = set(asset.keywords)
            intersection = len(all_opp_terms & asset_kw_set)
            if intersection == 0:
                continue
            # Score: intersection over union (Jaccard-like), weighted by asset keyword count
            union = len(all_opp_terms | asset_kw_set)
            score = intersection / union if union > 0 else 0.0
            # Boost for more overlapping keywords
            score = min(1.0, score + 0.1 * intersection)

            if score > best_score:
                best_score = score
                best_asset = asset

        if best_asset is not None and best_score > 0.15:  # Minimum threshold
            logger.debug("Local B-roll match: %s (score=%.3f)", best_asset.path, best_score)
            return BrollSource(
                match_type="local",
                path=best_asset.path,
                confidence=best_score,
            )
        return None

    def _generated_card_fallback(self, opportunity: BrollOpportunity) -> BrollSource:
        # Generate a deterministic placeholder path based on the opportunity
        # In production, this would be replaced by an actual generation pipeline
        safe_query = "".join(c if c.isalnum() else "_" for c in opportunity.concept_query[:40])
        placeholder_path = (
            f"generated:broll/{safe_query}_{opportunity.start:.1f}-{opportunity.end:.1f}.mp4"
        )
        logger.debug("Generated card fallback for B-roll: %s", placeholder_path)
        return BrollSource(
            match_type="generated",
            path=placeholder_path,
            confidence=0.5,  # Medium confidence for generated content
        )


def _words(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z']+", text.lower())


__all__ = [
    "BrollSource",
    "BrollSourceResolver",
    "MatchType",
]

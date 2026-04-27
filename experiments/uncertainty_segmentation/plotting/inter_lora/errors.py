"""Custom exceptions for inter-LoRA comparison pipeline."""

from __future__ import annotations


class RankDiscoveryError(RuntimeError):
    """Fewer than the minimum required ranks discovered under ROOT."""


class MissingArtefactError(FileNotFoundError):
    """A required per-rank evaluation file is absent (raised under --strict)."""


class BaselineMismatchError(ValueError):
    """baseline_test_dice.csv differs by more than tolerance across ranks."""

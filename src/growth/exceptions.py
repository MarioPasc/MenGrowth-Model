# src/growth/exceptions.py
"""Custom exception hierarchy for the growth prediction framework."""


class GrowthModelError(Exception):
    """Base exception for growth model errors."""


class UncertaintyPropagationError(GrowthModelError):
    """Raised when uncertainty propagation fails or is misconfigured."""

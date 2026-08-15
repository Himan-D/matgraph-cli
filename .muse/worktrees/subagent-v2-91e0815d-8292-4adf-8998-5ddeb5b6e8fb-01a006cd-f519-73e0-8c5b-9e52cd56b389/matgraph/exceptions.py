"""Structured exceptions for MatGraph — no silent failures."""
from __future__ import annotations

class MatGraphError(Exception):
    """Base class for all MatGraph errors."""

class ConfigError(MatGraphError):
    """Missing or invalid configuration / API key."""

class DataNotFoundError(MatGraphError):
    """No data returned from Materials Project."""

class ModelLoadError(MatGraphError):
    """ML model could not be loaded."""

class ModelInferenceError(MatGraphError):
    """Inference failed."""

class ValidationError(MatGraphError):
    """Input validation failed."""

class AuthError(MatGraphError):
    """Authentication / authorization failure."""

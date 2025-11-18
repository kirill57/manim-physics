"""Compatibility helpers for different Manim versions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from manim import config

try:  # pragma: no cover - exercised implicitly via Manim
    from manim.constants import RendererType as _RendererType
except ImportError:  # pragma: no cover - fallback for very old releases
    class _RendererType(Enum):
        CAIRO = "cairo"
        OPENGL = "opengl"


RendererType = _RendererType
"""Public alias for the renderer enum used by Manim."""


def get_renderer_type(renderer: Any | None = None) -> RendererType:
    """Return the :class:`RendererType` for ``renderer`` or the current config."""

    value = config.renderer if renderer is None else renderer
    if isinstance(value, RendererType):
        return value
    try:
        return RendererType(value)
    except Exception:  # pragma: no cover - defensive fallback
        if str(value).lower() == RendererType.OPENGL.value:
            return RendererType.OPENGL
        return RendererType.CAIRO


def is_opengl_renderer(renderer: Any | None = None) -> bool:
    """Whether the given (or current) renderer corresponds to OpenGL."""

    return get_renderer_type(renderer) == RendererType.OPENGL


__all__ = ["RendererType", "get_renderer_type", "is_opengl_renderer"]

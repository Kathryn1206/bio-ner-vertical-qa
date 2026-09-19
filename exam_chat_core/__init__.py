"""Public interface for the exam-question answering package.

The lightweight demo backend is the default so a fresh clone can be executed
without private model artifacts. Set ``EXAM_CHAT_MODE=full`` to load the
original model-backed pipeline.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType


_BACKEND: ModuleType | None = None
_BACKEND_MODE: str | None = None


def get_backend_mode() -> str:
    """Return the configured backend mode (``demo`` or ``full``)."""

    mode = os.getenv("EXAM_CHAT_MODE", "demo").strip().lower()
    if mode not in {"demo", "full"}:
        raise ValueError("EXAM_CHAT_MODE must be either 'demo' or 'full'.")
    return mode


def _get_backend() -> ModuleType:
    global _BACKEND, _BACKEND_MODE

    mode = get_backend_mode()
    if _BACKEND is not None and _BACKEND_MODE == mode:
        return _BACKEND

    module_name = ".demo" if mode == "demo" else ".core_code"
    _BACKEND = importlib.import_module(module_name, __name__)
    _BACKEND_MODE = mode
    return _BACKEND


def init_exam_chat() -> None:
    """Initialize the selected backend."""

    backend = _get_backend()
    initializer = getattr(backend, "init_demo_chat", None)
    if initializer is not None:
        initializer()


def get_answer_from_exam_db(user_input: str) -> str:
    """Route a user query through the selected backend."""

    backend = _get_backend()
    return backend.get_answer_from_exam_db(user_input)


__all__ = ["get_backend_mode", "init_exam_chat", "get_answer_from_exam_db"]

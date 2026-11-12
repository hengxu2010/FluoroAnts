"""Interface language.

The texts in the source code are English.  A language pack maps each of them to
its translation; texts that carry values use numbered placeholders, as in
``T("Loaded: {0}", name)``.

The language is chosen once, when this module is first imported, because the
label tables of the other modules are built while they load.  In order of
priority: the ``FLUOROANTS_LANG`` environment variable (set by ``--lang``), the
language remembered from the Language menu, and the language of the system.
"""
from __future__ import annotations

import os
import sys

LANGUAGES: dict[str, str] = {"en": "English", "zh": "中文"}
ENVIRONMENT_VARIABLE = "FLUOROANTS_LANG"
_SETTINGS_KEY = "ui/language"


def _normalized(code: object) -> str | None:
    text = str(code or "").strip().lower()
    if text.startswith("zh"):
        return "zh"
    if text.startswith("en"):
        return "en"
    return None


def system_language() -> str:
    """Language of the operating system's interface: "zh" or "en"."""

    if sys.platform == "win32":
        try:
            import ctypes

            language_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()  # type: ignore[attr-defined]
            return "zh" if (language_id & 0x3FF) == 0x04 else "en"
        except Exception:
            pass
    for name in ("LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"):
        found = _normalized(os.environ.get(name))
        if found:
            return found
    try:
        from PySide6.QtCore import QLocale

        found = _normalized(QLocale.system().name())
        if found:
            return found
    except Exception:
        pass
    return "en"


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings("FluoroAnts", "FluoroAnts Universal")


def remembered_language() -> str | None:
    """The language chosen in the Language menu, or None to follow the system."""

    try:
        return _normalized(_settings().value(_SETTINGS_KEY, ""))
    except Exception:
        return None


def remember_language(code: str | None) -> None:
    """Store the language for the next start; None follows the system again."""

    settings = _settings()
    if code in LANGUAGES:
        settings.setValue(_SETTINGS_KEY, code)
    else:
        settings.remove(_SETTINGS_KEY)
    settings.sync()


def _choose() -> str:
    return (
        _normalized(os.environ.get(ENVIRONMENT_VARIABLE))
        or remembered_language()
        or system_language()
    )


_language = _choose()
_TEXT: dict[str, str] = {}
if _language == "zh":
    from .locale_zh import TEXT as _TEXT
_SOURCE_TEXT: dict[str, str] | None = None


def language() -> str:
    """Code of the language in use: "en" or "zh"."""

    return _language


def T(text: str, *values: object) -> str:
    """Return ``text`` in the interface language, with ``values`` filled in."""

    template = _TEXT.get(text, text)
    return template.format(*values) if values else template


def N_(text: str) -> str:
    """Mark a source text that serves as a stable key and is translated where it is shown."""

    return text


def translated(text: str, code: str, *values: object) -> str:
    """Return ``text`` in the language ``code``, whatever the interface language is."""

    template = text
    if code == "zh":
        from .locale_zh import TEXT

        template = TEXT.get(text, text)
    return template.format(*values) if values else template


def source_text(text: str) -> str:
    """Return the English source text of a text given in any supported language."""

    global _SOURCE_TEXT
    if _SOURCE_TEXT is None:
        from .locale_zh import TEXT

        _SOURCE_TEXT = {translated: source for source, translated in TEXT.items()}
    return _SOURCE_TEXT.get(text, text)

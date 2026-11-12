from __future__ import annotations

import argparse
import copy
import math
import os
import re
import sys
from pathlib import Path


def _language_from_command_line(arguments: list[str]) -> None:
    """Apply ``--lang`` before the program modules load.

    The label tables of the program are built while its modules are imported,
    so the language has to be known first.
    """

    for index, value in enumerate(arguments):
        if value == "--lang" and index + 1 < len(arguments):
            os.environ["FLUOROANTS_LANG"] = arguments[index + 1]
        elif value.startswith("--lang="):
            os.environ["FLUOROANTS_LANG"] = value.split("=", 1)[1]


_language_from_command_line(sys.argv[1:])

from fluoro_ants.i18n import LANGUAGES, T, source_text  # noqa: E402
from fluoro_ants.version import FLUOROANTS_RELEASE_VERSION  # noqa: E402
from fluoro_ants.quick_settings import (  # noqa: E402
    quick_settings_from_dict,
    segment_loaded_image_with_quick,
)
from fluoro_ants.universal_core import (  # noqa: E402
    AI_RECOGNITION_ENGINES,
    BUILTIN_PRESETS,
    EXPORT_BUNDLE_PREFIXED_FILES,
    EXPORT_LOCATION_MIRROR,
    EXPORT_LOCATION_MODES,
    EXPORT_LOCATION_SOURCE,
    UNIVERSAL_ALGORITHM_VERSION,
    batch_process,
    configure_opencv_threads,
    load_recipe_bundle,
    normalize_export_location_mode,
    scrub_local_paths,
)
from fluoro_ants.universal_core import (  # noqa: E402
    supported_image_files as supported_universal_image_files,
)


def _safe_error_message(error: BaseException, *paths: Path) -> str:
    message = str(error).strip() or error.__class__.__name__
    for path in paths:
        variants = {str(path)}
        try:
            expanded = path.expanduser()
        except (OSError, RuntimeError):
            expanded = path
        variants.add(str(expanded))
        try:
            variants.add(str(expanded.resolve()))
        except (OSError, RuntimeError):
            pass
        for value in sorted(variants, key=len, reverse=True):
            if value:
                message = message.replace(value, path.name or "selected path")
    message = re.sub(
        r"(?<![\w.])(?:[A-Za-z]:[\\/]|\\\\|/)(?:[^\s:;,]+[\\/])*[^\s:;,]*",
        T("local path"),
        message,
    )
    return message[:600]


def run_batch(
    input_path: Path,
    output_folder: Path | None,
    *,
    preset_name: str = "General balanced",
    recipe_path: Path | None = None,
    recursive: bool = False,
    export_location_mode: str = EXPORT_LOCATION_MIRROR,
    pixel_size: float | None = None,
    length_unit: str = "µm",
    edge_offset_px: int = 0,
) -> int:
    """Run the generic single-image workflow without the desktop interface."""

    configure_opencv_threads()
    raw_source = Path(input_path)
    raw_output = None if output_folder is None else Path(output_folder)
    try:
        location_mode = normalize_export_location_mode(export_location_mode)
        source = raw_source.expanduser().resolve()
        output_root = (
            None if raw_output is None else raw_output.expanduser().resolve()
        )
        if location_mode != EXPORT_LOCATION_SOURCE and output_root is None:
            raise ValueError("Mirror/flat export requires --output.")
        source_is_directory = source.is_dir()
        if source_is_directory:
            if (
                location_mode != EXPORT_LOCATION_SOURCE
                and output_root is not None
                and (output_root == source or source in output_root.parents)
            ):
                raise ValueError("Output folder must be outside the input folder tree.")
            paths = supported_universal_image_files(source, recursive=recursive)
        elif source.is_file():
            paths = [source]
        else:
            raise FileNotFoundError(
                f"Input path does not exist: {source.name or 'selected path'}"
            )
        if not paths:
            raise ValueError("No supported images were found.")
        relative_paths = {
            path: path.relative_to(source) if source_is_directory else Path(path.name)
            for path in paths
        }
    except Exception as exc:
        sensitive_paths = [raw_source]
        if raw_output is not None:
            sensitive_paths.append(raw_output)
        print(
            f"Could not prepare batch: {_safe_error_message(exc, *sensitive_paths)}",
            file=sys.stderr,
        )
        return 2

    try:
        segmenter = None
        export_quick_settings = None
        export_quick_base_params = None
        if recipe_path is not None:
            (
                _name,
                _description,
                params,
                quick_payload,
                quick_base,
            ) = load_recipe_bundle(recipe_path)
            if quick_payload is not None and quick_base is not None:
                try:
                    quick_settings = quick_settings_from_dict(quick_payload)
                except (TypeError, ValueError):
                    print(
                        "Quick settings are not compatible; using the recipe's "
                        "complete professional parameters.",
                        file=sys.stderr,
                    )
                else:
                    export_quick_settings = quick_payload
                    export_quick_base_params = quick_base

                    def segmenter(loaded, _params):
                        return segment_loaded_image_with_quick(
                            loaded,
                            quick_base,
                            quick_settings,
                            max_side=None,
                        )
        else:
            # A preset may be named in either interface language.
            preset_name = source_text(preset_name)
            if preset_name not in BUILTIN_PRESETS:
                available = T(", ").join(T(name) for name in BUILTIN_PRESETS)
                print(
                    f"Unknown preset: {preset_name}. Available presets: {available}",
                    file=sys.stderr,
                )
                return 2
            params = copy.deepcopy(BUILTIN_PRESETS[preset_name].params)
        if getattr(params, "recognition_engine", "classic") in AI_RECOGNITION_ENGINES:
            # A learned-model recipe never runs through a quick plan, even
            # when the recipe file also carries quick settings.
            segmenter = None
            export_quick_settings = None
            export_quick_base_params = None
        for index, path in enumerate(paths, start=1):
            print(
                f"[{index}/{len(paths)}] {relative_paths[path].as_posix()}",
                flush=True,
            )
        summary = batch_process(
            paths,
            params,
            output_root,
            relative_paths=relative_paths,
            export_location_mode=location_mode,
            export_bundle_mode=EXPORT_BUNDLE_PREFIXED_FILES,
            segmenter=segmenter,
            export_quick_settings=export_quick_settings,
            export_quick_base_params=export_quick_base_params,
            pixel_size=pixel_size,
            length_unit=length_unit,
            edge_offset_px=edge_offset_px,
        )
    except Exception as exc:
        sensitive_paths = [source]
        if output_root is not None:
            sensitive_paths.append(output_root)
        if recipe_path is not None:
            sensitive_paths.append(recipe_path)
        print(
            f"Batch failed: {_safe_error_message(exc, *sensitive_paths)}",
            file=sys.stderr,
        )
        return 1

    exported = summary.get("exported", [])
    failures = summary.get("failures", [])
    destination_text = (
        "beside each source image"
        if location_mode == EXPORT_LOCATION_SOURCE
        else "to the selected folder"
    )
    print(
        f"Exported {len(exported)}/{len(paths)} image(s) {destination_text}",
        flush=True,
    )
    if failures:
        print("Failed images:", file=sys.stderr)
        for item in failures:
            failure_paths = [source]
            if output_root is not None:
                failure_paths.append(output_root)
            safe_failure = _safe_error_message(
                RuntimeError(str(item.get("error", "unknown error"))),
                *failure_paths,
            )
            print(
                f"  - {item.get('source', 'unknown')}: "
                f"{safe_failure}",
                file=sys.stderr,
            )
        return 1
    return 0


def _positive_pixel_size(value: str) -> float:
    try:
        size = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(T(
            "The pixel size must be a finite number greater than 0",
        )) from exc
    if not math.isfinite(size) or size <= 0.0 or not math.isfinite(size * size) or size * size <= 0.0:
        raise argparse.ArgumentTypeError(T(
            "The pixel size must be a finite number greater than 0 from which areas can be "
            "computed reliably",
        ))
    return size


def _physical_length_unit(value: str) -> str:
    unit = str(value).strip()
    if not unit:
        raise argparse.ArgumentTypeError(T("The unit of length must not be empty"))
    if len(unit) > 32 or any(ord(character) < 32 for character in unit):
        raise argparse.ArgumentTypeError(T(
            "The unit of length must not exceed 32 characters or contain control characters",
        ))
    return unit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=T("FluoroAnts Universal: recognition of positive regions in 2-D images")
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            T(
                "FluoroAnts Universal {0} (recognition core {1})",
                FLUOROANTS_RELEASE_VERSION,
                UNIVERSAL_ALGORITHM_VERSION,
            )
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help=T("Image or folder to open at start; the input path in batch mode"),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=T("Process one image or a folder without the window"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=T("Root folder for batch output; not needed with --output-layout=source"),
    )
    parser.add_argument(
        "--output-layout",
        choices=tuple(sorted(EXPORT_LOCATION_MODES)),
        default=EXPORT_LOCATION_MIRROR,
        help=(
            T(
                "Output layout: mirror keeps the folder structure of the input (default); source "
                "writes beside each original image; flat puts all result files, prefixed with the "
                "image name, into the output root.",
            )
        ),
    )
    parser.add_argument(
        "--lang",
        choices=sorted(LANGUAGES),
        help=T("Language of the interface and of the texts in exported files; default: the stored choice, else the system language"),
    )
    parser.add_argument(
        "--preset",
        default="General balanced",
        help=T("Name of the built-in preset used in batch mode; default: General balanced"),
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        help=T("JSON recipe used in batch mode; overrides --preset"),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=T("Scan subfolders in batch mode"),
    )
    parser.add_argument(
        "--pixel-size",
        type=_positive_pixel_size,
        default=None,
        help=T(
            "Physical size of one pixel (used with --unit); adds physical-area columns to the "
            "exports.",
        ),
    )
    parser.add_argument(
        "--unit",
        type=_physical_length_unit,
        default="µm",
        # Keep --help printable in the legacy Windows GBK console.  The micro
        # sign itself is still the runtime default and remains accepted as an
        # explicit value; only the help label uses its ASCII spelling.
        help=T("Unit of the pixel size and of physical areas; default um (micrometres)."),
    )
    parser.add_argument(
        "--edge-offset",
        type=int,
        default=0,
        help=(
            T(
                "Edge distance of the selection in source pixels: positive expands, negative "
                "shrinks; applied to the final selection after recognition and not stored in the "
                "recipe. Default 0.",
            )
        ),
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help=T("List all built-in presets and exit"),
    )
    return parser


_PALETTE = (
    ("Window", "#11161d"),
    ("WindowText", "#e8eef5"),
    ("Base", "#0f141b"),
    ("AlternateBase", "#121a23"),
    ("Text", "#e8eef5"),
    ("Button", "#26313e"),
    ("ButtonText", "#e8eef5"),
    ("Highlight", "#1c6652"),
    ("HighlightedText", "#ffffff"),
    ("ToolTipBase", "#1c2530"),
    ("ToolTipText", "#e8eef5"),
    ("PlaceholderText", "#7d8a97"),
    ("Link", "#6fd6b2"),
)
_FONT_FAMILIES = {
    "zh": ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei"),
    "en": ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Noto Sans", "DejaVu Sans"),
}


def configure_application(app: object) -> None:
    """Give the whole program one look: widget style, colours, font, button texts.

    The window brings its own style sheet; this covers what a style sheet on a
    window cannot reach (tool tips, menus of other windows, standard dialogs).
    """

    from PySide6.QtCore import QLibraryInfo, QTranslator
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
    from PySide6.QtWidgets import QStyleFactory

    from fluoro_ants.i18n import language

    # The same widget style on every platform, so the dark theme is drawn alike.
    style = QStyleFactory.create("Fusion")
    if style is not None:
        app.setStyle(style)  # type: ignore[attr-defined]
    palette = QPalette()
    for role, colour in _PALETTE:
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(colour))
    for role in ("WindowText", "Text", "ButtonText"):
        palette.setColor(QPalette.ColorGroup.Disabled, getattr(QPalette.ColorRole, role), QColor("#65717e"))
    app.setPalette(palette)  # type: ignore[attr-defined]
    app.setStyleSheet(  # type: ignore[attr-defined]
        "QToolTip { background: #1c2530; color: #e8eef5; border: 1px solid #4d6579;"
        " padding: 5px 7px; font-size: 12px; }"
    )

    # A font that is installed anyway; nothing is bundled.  Other scripts,
    # such as Chinese file names in the English interface, fall back by themselves.
    installed = set(QFontDatabase.families())
    for family in _FONT_FAMILIES.get(language(), _FONT_FAMILIES["en"]):
        if family in installed:
            app.setFont(QFont(family, 10))  # type: ignore[attr-defined]
            break

    # Qt's own texts ("OK", "Cancel", "Yes", …) in the interface language.
    if language() == "zh":
        folder = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        for catalogue in ("qtbase_zh_CN", "qt_zh_CN"):
            translator = QTranslator(app)  # type: ignore[arg-type]
            if translator.load(catalogue, folder):
                app.installTranslator(translator)  # type: ignore[attr-defined]
                break


def _traceback_without_private_paths(exc_type, exc, tb) -> str:
    """Format a Python traceback for the crash log without the user's folders.

    Stack lines keep the program's own file names, with the home folder
    shortened to ``~``.  Message lines can quote image locations, so every
    local path in them is replaced.
    """
    import traceback

    home = str(Path.home())
    lines: list[str] = []
    for chunk in traceback.format_exception(exc_type, exc, tb):
        for line in chunk.splitlines(keepends=True):
            if line.startswith(" "):
                if len(home) > 1:
                    line = line.replace(home, "~")
            elif not line.startswith(("Traceback ", "During handling", "The above exception")):
                line = scrub_local_paths(line)
            lines.append(line)
    return "".join(lines)


def _enable_crash_log() -> None:
    """Leave evidence when the process dies below Python.

    A native crash inside OpenCV/Qt kills the window with no traceback
    and no dialog - the user just sees it vanish.  faulthandler writes the C-level
    stack of every thread to this file before the process goes down, and the
    excepthook catches anything Python-level that escapes a worker.

    The log stays on this computer (``FluoroAnts/crash.log`` in the per-user
    application-data or cache folder); nothing is ever sent anywhere.
    """
    import faulthandler, os, sys, datetime
    try:
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") \
            or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        log_dir = Path(base) / "FluoroAnts"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "crash.log"
        # Every start appends a header, so an append-only log grows without
        # bound over months of use.  Roll it over once it passes 2 MB; the
        # previous generation is kept as crash.log.1 for diagnosis.
        try:
            if log_path.is_file() and log_path.stat().st_size > 2 * 1024 * 1024:
                log_path.replace(log_dir / "crash.log.1")
        except OSError:
            pass
        handle = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
        handle.write(f"\n===== start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} pid={os.getpid()} =====\n")
        faulthandler.enable(file=handle, all_threads=True)

        def hook(exc_type, exc, tb):
            handle.write(f"--- python exception {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ---\n")
            handle.write(_traceback_without_private_paths(exc_type, exc, tb))
            sys.__excepthook__(exc_type, exc, tb)

        sys.excepthook = hook
        try:
            import threading
            threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)
        except Exception:
            pass
    except Exception:
        pass  # logging must never be the thing that breaks startup


def main(argv: list[str] | None = None) -> int:
    _enable_crash_log()
    configure_opencv_threads()
    args = build_parser().parse_args(argv)
    if args.list_presets:
        for index, preset in enumerate(BUILTIN_PRESETS.values(), start=1):
            print(
                T(
                    "{0:02d}. {1} | {2} — {3}",
                    index,
                    T(preset.category),
                    T(preset.name),
                    T(preset.description),
                )
            )
        return 0
    if args.batch:
        if args.path is None:
            raise SystemExit("--batch requires an image/folder")
        if args.output_layout != EXPORT_LOCATION_SOURCE and args.output is None:
            raise SystemExit(
                "--batch with mirror/flat layout requires --output; "
                "source layout writes beside each source image"
            )
        return run_batch(
            args.path,
            args.output,
            preset_name=args.preset,
            recipe_path=args.recipe,
            recursive=args.recursive,
            export_location_mode=args.output_layout,
            pixel_size=args.pixel_size,
            length_unit=args.unit,
            edge_offset_px=args.edge_offset,
        )

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "PySide6 is not installed. Install it with: pip install PySide6",
            file=sys.stderr,
        )
        return 3

    app = QApplication([sys.argv[0]])
    app.setApplicationName("FluoroAnts Universal")
    app.setOrganizationName("FluoroAnts")
    configure_application(app)
    # The window performs path resolution inside its guarded open routine so a
    # broken link or inaccessible startup path is reported without a traceback.
    initial = args.path.expanduser() if args.path else None
    from fluoro_ants.universal_gui import UniversalFluoroWindow

    window = UniversalFluoroWindow(initial)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import copy
import fnmatch
import json
import math
import re
import sys
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    QDir,
    QProcess,
    QByteArray,
    QEvent,
    QObject,
    QPointF,
    QRect,
    QRectF,
    QRunnable,
    QSettings,
    QSize,
    QStandardPaths,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QGuiApplication,
    QImage,
    QKeySequence,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolBox,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .quick_settings import (
    QUICK_BACKGROUND_LABELS,
    QUICK_BORDER_LABELS,
    QUICK_CONTOUR_LABELS,
    QUICK_EDGE_LABELS,
    QUICK_GAP_LABELS,
    QUICK_HOLE_LABELS,
    QUICK_IMAGING_LABELS,
    QUICK_INTENSITY_LABELS,
    QUICK_LOGIC_LABELS,
    QUICK_MULTI_FIELDS,
    QUICK_NOISE_LABELS,
    QUICK_OPENING_LABELS,
    QUICK_POLARITY_LABELS,
    QUICK_SCALE_LABELS,
    QUICK_SENSITIVITY_LABELS,
    QUICK_SHAPE_LABELS,
    QUICK_SHAPE_STRICTNESS_LABELS,
    QUICK_SIGNAL_LABELS,
    QUICK_SIZE_LABELS,
    QUICK_SPECK_LABELS,
    QUICK_TOUCHING_LABELS,
    QuickSettings,
    compile_quick_plan,
    normalize_quick_settings,
    quick_setting_values,
    quick_settings_from_dict,
    quick_settings_summary,
    quick_settings_to_dict,
    quick_plan_key,
    result_matches_quick_plan,
    resolve_quick_settings,
    segment_loaded_image_with_quick,
)
from . import i18n
from .i18n import T
from .quality_control import assess_segmentation
from .version import AUTHOR, FLUOROANTS_RELEASE_VERSION, contact_address
from .universal_core import (
    BACKGROUND_CEILING_MODE_LABELS,
    BACKGROUND_METHOD_LABELS,
    BUILTIN_PRESETS,
    DEFAULT_EXPORT_ARTIFACTS,
    DENOISE_METHOD_LABELS,
    EXPORT_ARTIFACT_BACKGROUND_CORRECTED_RESPONSE,
    EXPORT_ARTIFACT_IDS,
    EXPORT_ARTIFACT_MERGE,
    EXPORT_ARTIFACT_OBJECT_LABELS,
    EXPORT_ARTIFACT_OBJECTS,
    EXPORT_ARTIFACT_POSITIVE_MASK,
    EXPORT_ARTIFACT_PROJECT,
    EXPORT_ARTIFACT_QUALITY_REPORT,
    EXPORT_ARTIFACT_RECIPE,
    EXPORT_ARTIFACT_SUMMARY,
    EXPORT_BUNDLE_PREFIXED_FILES,
    EXPORT_LOCATION_FLAT,
    EXPORT_LOCATION_MIRROR,
    EXPORT_LOCATION_SOURCE,
    INTENSITY_SPACE_LABELS,
    MAX_EDGE_OFFSET_PX,
    MERGE_BASE_MODE_ANALYSIS_INPUT,
    MERGE_BASE_MODE_SOURCE_ORIGINAL,
    OBJECT_SHAPE_PROFILE_LABELS,
    OBJECT_SHAPE_STRICTNESS_LABELS,
    SIGNAL_MODE_LABELS,
    SUPPORTED_IMAGE_SUFFIXES,
    THRESHOLD_METHOD_LABELS,
    UNIVERSAL_ALGORITHM_VERSION,
    CrosstalkCorrection,
    ExportReceipt,
    LoadedImage,
    UniversalParams,
    UniversalResult,
    apply_crosstalk_correction,
    apply_edge_offset,
    configure_opencv_threads,
    contours_from_mask,
    display_channel_for_signal_mode,
    export_bundle_base_name,
    export_bundle_exists,
    export_result,
    image_to_rgb_original_values,
    image_to_rgb_display,
    AI_RECOGNITION_ENGINES,
    load_image,
    load_recipe_bundle,
    load_recipe_crosstalk_settings,
    normalize_export_location_mode,
    normalize_merge_base_mode,
    params_from_dict,
    params_to_dict,
    rebuild_result_from_mask,
    render_overlay,
    resolve_export_parent,
    restore_ai_result_sidecars,
    resized_mask,
    save_recipe,
    scalar_to_rgb_display,
    select_rgb_display_channel,
    segment_loaded_image,
    supported_image_files,
)

_IMAGE_FILTER = T("Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg *.webp);;All files (*)")
_PREVIEW_MAX_SIDE = 2200
_DISPLAY_MAX_SIDE = 2600
_PREVIEW_PIXELS = 8_000_000
# Marching ants are a display aid, not analysis data.  A highly fragmented mask
# can otherwise create hundreds of thousands of OpenCV contour objects and Qt
# path elements on the GUI thread.  Keep the exact mask/result/export intact,
# but use a bounded level of detail for the on-screen path when its boundary is
# too complex to animate responsively.
_DISPLAY_CONTOUR_EDGE_BUDGET = 60_000
_DISPLAY_CONTOUR_COUNT_BUDGET = 12_000
_DISPLAY_CONTOUR_POINT_BUDGET = 48_000
_DISPLAY_CONTOUR_SCAN_ROWS = 256
_SPATIAL_SCALE_MODE_LABELS = {
    "relative_grid": T("Relative to the frame"),
    "pixel_locked": T("Fixed source pixels (cropping / magnification change)"),
}
# What the statistics show for the signal actually used; a single-plane image
# reports "grayscale", which is not one of the selectable modes.
_SIGNAL_MODE_DISPLAY = {**SIGNAL_MODE_LABELS, "grayscale": T("Greyscale")}
_SIGNAL_MODE_BY_DISPLAY_CHANNEL = {
    "r": "red",
    "g": "green",
    "b": "blue",
}

_SELECTION_MASK_FILTER = T("Binary mask (*.tif *.tiff *.png *.bmp);;All files (*)")
_RESUME_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")

# Per-image processing state shown in the queue's status column.  The order
# here is also the sort order for "Processing status" (unfinished work first).
_QUEUE_STATUS_ORDER = (
    "failed",
    "pending",
    "running",
    "previewed",
    "analyzed",
    "edited",
    "exported",
)
_QUEUE_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "pending": (T("Pending"), "#7f8e9c"),
    "running": (T("Processing…"), "#f2c94c"),
    "previewed": (T("Previewed"), "#8fb3d9"),
    "analyzed": (T("Analysed"), "#6fc3ff"),
    "edited": (T("Edited"), "#e3b341"),
    "exported": (T("Exported"), "#59ddb0"),
    "failed": (T("Failed"), "#ff7b72"),
}
_QUEUE_SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("name", T("File name")),
    ("path", T("Folder + name")),
    ("mtime", T("Modification time")),
    ("size", T("File size")),
    ("status", T("Processing status")),
    ("added", T("Order added")),
)
_NATURAL_SORT_RE = re.compile(r"(\d+)")


def _natural_key(text: str) -> tuple[object, ...]:
    """Sort key that orders embedded numbers numerically (img_2 < img_10)."""

    parts: list[object] = []
    for index, piece in enumerate(_NATURAL_SORT_RE.split(str(text).casefold())):
        if index % 2:
            parts.append((0, int(piece), len(piece)))
        else:
            parts.append((1, piece))
    return tuple(parts)


def _queue_filter_matches(label: str, filter_text: str) -> bool:
    """Case-insensitive match of every space-separated term against ``label``.

    A term containing ``*`` or ``?`` is a wildcard pattern (``*_ch01*``); a
    leading ``-`` excludes matches (``-DAPI``).  Terms are matched against
    the folder-relative display text, so a channel name selects that channel
    across every imported folder.
    """

    haystack = label.casefold()
    for raw_term in filter_text.split():
        negate = raw_term.startswith("-") and len(raw_term) > 1
        term = raw_term[1:] if negate else raw_term
        term = term.casefold()
        if any(character in term for character in "*?["):
            pattern = term if term.startswith("*") else f"*{term}"
            pattern = pattern if pattern.endswith("*") else f"{pattern}*"
            hit = fnmatch.fnmatchcase(haystack, pattern)
        else:
            hit = term in haystack
        if hit == negate:
            return False
    return True


def _validated_edge_offset(value: object) -> int:
    """Clamp-free validation of a signed edge distance in source pixels."""

    if isinstance(value, bool):
        raise TypeError(T("The edge distance must be a whole number of pixels"))
    offset = int(value)  # type: ignore[call-overload]
    if abs(offset) > MAX_EDGE_OFFSET_PX:
        raise ValueError(T("The edge distance must be within ±{0} pixels", MAX_EDGE_OFFSET_PX))
    return offset


def _pristine_result_copy(result: UniversalResult) -> UniversalResult:
    """A shallow copy whose Mask is private, so in-place brush edits on the
    displayed result can never leak into a kept reference copy."""

    duplicate = copy.copy(result)
    duplicate.mask = np.array(result.mask, dtype=bool, copy=True)
    if getattr(result, "review_mask", None) is not None:
        duplicate.review_mask = np.array(result.review_mask, dtype=bool, copy=True)
    if getattr(result, "review_base_mask", None) is not None:
        duplicate.review_base_mask = np.array(result.review_base_mask, dtype=bool, copy=True)
    duplicate.ai_metadata = copy.deepcopy(getattr(result, "ai_metadata", None))
    return duplicate


@dataclass(frozen=True)
class SelectionReviewBundle:
    """A clean review base paired with the exported final binary selection."""

    review_path: Path | None
    base_path: Path
    mask_path: Path


class StableParameterScrollArea(QScrollArea):
    """Keep a parameter's wheel event from also scrolling its whole panel."""

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        target = QApplication.widgetAt(event.globalPosition().toPoint())
        widget = target
        while widget is not None:
            if isinstance(widget, (QAbstractSpinBox, QComboBox, QSlider)):
                event.accept()
                return
            if widget is self:
                break
            widget = widget.parentWidget()
        super().wheelEvent(event)


def _casefold_sibling(parent: Path, filename: str) -> Path | None:
    requested = filename.casefold()
    for child in parent.iterdir():
        if child.is_file() and child.name.casefold() == requested:
            return child.resolve()
    return None


class ReviewSourceNotFound(FileNotFoundError):
    """The export record names an original image that is not next to the export."""

    def __init__(self, review: Path, mask: Path, file_name: str) -> None:
        super().__init__(
            T(
                "The original image {0} was not found beside the exported files; please point to "
                "the original image of this result.",
                file_name,
            )
        )
        self.review_path = review
        self.mask_path = mask
        self.file_name = file_name


_REVIEW_IMAGE_SUFFIXES = ("_ai_ants_16bit.tif", "_ai_review_overlay.png")


def resolve_selection_review_bundle(
    review_path: str | Path,
) -> SelectionReviewBundle:
    """Resolve one exported review image to its original image and its Mask.

    The outline pixels themselves are never reverse-engineered.  Continuing an
    analysis needs the lossless binary ``positive_mask`` written beside the
    review image, plus the original image.  Export records hold the original's
    file name, not its location, so it is looked up next to the export (and one
    folder above it); ``ReviewSourceNotFound`` lets the caller ask for it.
    """

    review = Path(review_path).expanduser().resolve()
    if not review.is_file():
        raise FileNotFoundError(T("The chosen review image does not exist."))
    parent = review.parent
    name = review.name
    folded = name.casefold()
    matched_suffix = next(
        (suffix for suffix in _REVIEW_IMAGE_SUFFIXES if folded.endswith(suffix)),
        None,
    )
    if matched_suffix is None:
        raise ValueError(
            T(
                "Choose a review image exported by FluoroAnts (*_ai_ants_16bit.tif or "
                "*_ai_review_overlay.png).",
            )
        )
    prefix = name[: -len(matched_suffix)]
    mask = next(
        (
            sibling
            for candidate in (f"{prefix}_positive_mask.tif", f"{prefix}_positive_mask.tiff")
            if (sibling := _casefold_sibling(parent, candidate)) is not None
        ),
        None,
    )
    if mask is None:
        raise FileNotFoundError(
            T(
                "No matching *_positive_mask.tif was found in the same folder; the true selection "
                "cannot be restored reliably from the review image alone.",
            )
        )

    source_stem = re.sub(r"(?i)_FluoroAnts(?:_\d{3,})?$", "", prefix)
    base_names = [f"{source_stem}{ext}" for ext in _RESUME_IMAGE_EXTENSIONS]
    recorded_name = ""
    record = mask.with_name(f"{prefix}_ai_analysis.json")
    if record.is_file():
        try:
            source = json.loads(record.read_text(encoding="utf-8")).get("ai_metadata", {}).get("source_image")
        except (OSError, ValueError):
            source = None
        if isinstance(source, Mapping):
            recorded_name = Path(str(source.get("file_name") or "")).name
            # Records written by earlier builds stored the full location.
            former_location = source.get("path")
            if isinstance(former_location, str) and former_location:
                candidate = Path(former_location).expanduser()
                recorded_name = recorded_name or candidate.name
                if candidate.is_file():
                    return SelectionReviewBundle(review, candidate.resolve(), mask)
    if recorded_name:
        base_names.insert(0, recorded_name)

    exported = {review, mask}
    for folder in (parent, parent.parent):
        if not folder.is_dir():
            continue
        for candidate in dict.fromkeys(base_names):
            sibling = _casefold_sibling(folder, candidate)
            if sibling is not None and sibling not in exported:
                return SelectionReviewBundle(review, sibling, mask)
    raise ReviewSourceNotFound(review, mask, recorded_name or base_names[0])


def load_selection_review_mask(
    bundle: SelectionReviewBundle,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read and strictly validate the binary Mask used for continued editing."""

    loaded = load_image(bundle.mask_path)
    image = np.asarray(loaded.image)
    if image.ndim == 2:
        plane = image
    elif image.ndim == 3 and image.shape[2] == 1:
        plane = image[..., 0]
    elif image.ndim == 3 and image.shape[2] >= 3:
        planes = image[..., :3]
        if not (
            np.array_equal(planes[..., 0], planes[..., 1])
            and np.array_equal(planes[..., 0], planes[..., 2])
        ):
            raise ValueError(T("The companion mask is a colour image, not a binary selection."))
        plane = planes[..., 0]
    else:
        raise ValueError(T("The companion mask must be a 2-D binary image."))
    if np.issubdtype(plane.dtype, np.floating) and not bool(np.isfinite(plane).all()):
        raise ValueError(T("The companion mask contains NaN or infinite values."))
    values = np.unique(plane)
    if (
        values.size > 2
        or (values.size == 2 and float(values[0]) != 0.0)
        or (values.size == 1 and float(values[0]) < 0.0)
    ):
        raise ValueError(T("The companion mask is not a binary image of 0 and positive values."))
    mask = np.asarray(plane > 0, dtype=bool)
    if expected_shape is not None and mask.shape != tuple(expected_shape):
        raise ValueError(
            T(
                "The mask size {0}×{1} differs from the base image size {2}×{3}.",
                mask.shape[1],
                mask.shape[0],
                expected_shape[1],
                expected_shape[0],
            )
        )
    return mask


def _mask_boundary_exceeds_budget(mask: np.ndarray, budget: int) -> bool:
    """Return early when a binary mask has too many display boundary edges.

    The scan is chunked so a large image needs only a small temporary boolean
    array.  This check runs before ``findContours`` and prevents a pathological
    salt-and-pepper mask from allocating one Python/NumPy object per fragment.
    """

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(T("The selection mask must be a 2-D array."))
    height, width = source.shape
    if height == 0 or width == 0:
        return False

    limit = max(0, int(budget))
    edge_count = int(np.count_nonzero(source[0]))
    if height > 1:
        edge_count += int(np.count_nonzero(source[-1]))
    edge_count += int(np.count_nonzero(source[:, 0]))
    if width > 1:
        edge_count += int(np.count_nonzero(source[:, -1]))
    if edge_count > limit:
        return True

    for y0 in range(0, height, _DISPLAY_CONTOUR_SCAN_ROWS):
        y1 = min(height, y0 + _DISPLAY_CONTOUR_SCAN_ROWS)
        block = source[y0:y1]
        if width > 1:
            edge_count += int(np.count_nonzero(block[:, 1:] != block[:, :-1]))
            if edge_count > limit:
                return True
        vertical_start = max(1, y0)
        if vertical_start < y1:
            edge_count += int(
                np.count_nonzero(
                    source[vertical_start:y1]
                    != source[vertical_start - 1 : y1 - 1]
                )
            )
            if edge_count > limit:
                return True
    return False


def _max_pool_mask_2x(mask: np.ndarray) -> np.ndarray:
    """Reduce a mask by 2 while retaining every selected source pixel."""

    source = np.asarray(mask, dtype=bool)
    pooled = source[0::2, 0::2].copy()
    lower = source[1::2, 0::2]
    right = source[0::2, 1::2]
    diagonal = source[1::2, 1::2]
    pooled[: lower.shape[0], : lower.shape[1]] |= lower
    pooled[: right.shape[0], : right.shape[1]] |= right
    pooled[: diagonal.shape[0], : diagonal.shape[1]] |= diagonal
    return pooled


def _bounded_display_contours(
    mask: np.ndarray,
) -> tuple[list[np.ndarray], float, float, int]:
    """Build bounded-complexity contours for the on-screen marching ants.

    Returns contours, x/y coordinate scales and the number of 2x reductions.
    Downsampling is used only for the display path; callers retain the original
    mask for editing, measurements and export.
    """

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(T("The selection mask must be a 2-D array."))
    original_height, original_width = source.shape
    display_mask = source
    reductions = 0

    while True:
        too_many_edges = _mask_boundary_exceeds_budget(
            display_mask,
            _DISPLAY_CONTOUR_EDGE_BUDGET,
        )
        if not too_many_edges:
            contours = contours_from_mask(display_mask, simplify=0.75)
            point_count = sum(len(contour) for contour in contours)
            if (
                len(contours) <= _DISPLAY_CONTOUR_COUNT_BUDGET
                and point_count <= _DISPLAY_CONTOUR_POINT_BUDGET
            ):
                break
        if display_mask.shape == (1, 1):
            contours = contours_from_mask(display_mask, simplify=0.75)
            break
        display_mask = _max_pool_mask_2x(display_mask)
        reductions += 1

    display_height, display_width = display_mask.shape
    scale_x = (
        float(original_width - 1) / float(display_width - 1)
        if original_width > 1 and display_width > 1
        else 1.0
    )
    scale_y = (
        float(original_height - 1) / float(display_height - 1)
        if original_height > 1 and display_height > 1
        else 1.0
    )
    return contours, scale_x, scale_y, reductions


def _select_display_channel(rgb: np.ndarray, channel: str) -> np.ndarray:
    """Compatibility wrapper around the shared analysis/export view rule."""

    return select_rgb_display_channel(rgb, channel)


def _analysis_input_signature(loaded: LoadedImage) -> str:
    return str(
        loaded.metadata.get(
            "analysis_input_signature",
            loaded.metadata.get("sha256", ""),
        )
    )


# Small glyphs for the style sheet (arrows, check mark), referred to as "fa:name.png".
QDir.addSearchPath("fa", str(Path(__file__).with_name("assets")))

# Start of a note in parentheses, in either interface language.
_PARENTHETICAL = re.compile(r"\s*[（(]")


def _tag_result_analysis_input(
    result: UniversalResult,
    loaded: LoadedImage,
) -> UniversalResult:
    """Bind a result to the exact raw/corrected input used by its worker."""

    setattr(result, "analysis_input_signature", _analysis_input_signature(loaded))
    preprocessing = loaded.metadata.get("analysis_preprocessing")
    if isinstance(preprocessing, Mapping):
        target_mode = str(preprocessing.get("target_signal_mode_used", "")).strip()
        if target_mode:
            # Keep a note in parentheses, such as "(union of 3 branches)".
            current_mode = str(result.signal_mode_used)
            note = _PARENTHETICAL.search(current_mode)
            suffix = current_mode[note.start():] if note else ""
            result.signal_mode_used = target_mode + suffix
    return result


_CONFIG_LOCATION = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
_USER_PRESET_FILE = (
    Path(_CONFIG_LOCATION) / "universal_presets.json"
    if _CONFIG_LOCATION
    else Path.home() / ".config" / "FluoroAnts" / "universal_presets.json"
)

_EXPORT_ARTIFACT_GROUPS: tuple[
    tuple[str, str, tuple[tuple[str, str, str], ...]], ...
] = (
    (
        T("Common results"),
        T(
            "For everyday checking, the selection mask and a quick summary; this group is ticked "
            "by default on first use.",
        ),
        (
            (
                EXPORT_ARTIFACT_POSITIVE_MASK,
                T("Positive selection mask"),
                T(
                    "name_FluoroAnts_positive_mask.tif · black-and-white binary image for further "
                    "use in ImageJ, Photoshop and other image software",
                ),
            ),
            (
                EXPORT_ARTIFACT_MERGE,
                T("Merge image with selection"),
                T(
                    "name_FluoroAnts_merge.png · original or corrected base (your choice) with "
                    "the final selection outlined, no fill",
                ),
            ),
            (
                EXPORT_ARTIFACT_SUMMARY,
                T("Summary table of the whole image"),
                T("name_FluoroAnts_summary.csv · object count, positive area and quality overview"),
            ),
        ),
    ),
    (
        T("Object data"),
        T("For object numbers, per-object data or quality checks."),
        (
            (
                EXPORT_ARTIFACT_OBJECT_LABELS,
                T("Object label image"),
                T(
                    "name_FluoroAnts_object_labels.tif / name_FluoroAnts_object_labels.npy · "
                    "every positive object has its own number",
                ),
            ),
            (
                EXPORT_ARTIFACT_OBJECTS,
                T("Table of every positive object"),
                T(
                    "name_FluoroAnts_objects.csv · per-object area, shape, position, intensity "
                    "and more",
                ),
            ),
            (
                EXPORT_ARTIFACT_QUALITY_REPORT,
                T("Quality check record"),
                T(
                    "name_FluoroAnts_quality_report.json · for reviewing unusual areas, border "
                    "objects and saturation",
                ),
            ),
        ),
    ),
    (
        T("Reproduction and diagnostics"),
        T("Tick these when parameters must be reproduced or the algorithm's response examined."),
        (
            (
                EXPORT_ARTIFACT_RECIPE,
                T("Recipe of this recognition"),
                T(
                    "name_FluoroAnts_recipe.json · complete recognition parameters that can be "
                    "imported again",
                ),
            ),
            (
                EXPORT_ARTIFACT_PROJECT,
                T("Complete project record"),
                T(
                    "name_FluoroAnts_project.json · versions, parameters, result summary and "
                    "index of the files written",
                ),
            ),
            (
                EXPORT_ARTIFACT_BACKGROUND_CORRECTED_RESPONSE,
                T("Background-corrected response image"),
                T(
                    "name_FluoroAnts_background_corrected_response.png · for advanced "
                    "troubleshooting",
                ),
            ),
        ),
    ),
)
_EXPORT_ARTIFACT_LABELS = {
    artifact_id: label
    for _group, _description, items in _EXPORT_ARTIFACT_GROUPS
    for artifact_id, label, _details in items
}
_COMMON_EXPORT_ARTIFACTS = frozenset(
    {
        EXPORT_ARTIFACT_POSITIVE_MASK,
        EXPORT_ARTIFACT_MERGE,
        EXPORT_ARTIFACT_SUMMARY,
    }
)
_COMMON_EXPORT_ITEMS = _EXPORT_ARTIFACT_GROUPS[0][2]
_ADVANCED_EXPORT_ITEMS = tuple(
    item
    for _group_title, _group_description, items in _EXPORT_ARTIFACT_GROUPS[1:]
    for item in items
)
_ALL_EXPORT_ARTIFACTS = frozenset(DEFAULT_EXPORT_ARTIFACTS)
_OUTLINE_EXPORT_ARTIFACTS = frozenset({EXPORT_ARTIFACT_MERGE})

if set(_EXPORT_ARTIFACT_LABELS) != set(EXPORT_ARTIFACT_IDS):
    raise RuntimeError(T("The export options must cover every artifact registered in the core"))
if len(_COMMON_EXPORT_ITEMS) != 3 or len(_ADVANCED_EXPORT_ITEMS) != 6:
    raise RuntimeError(T("The export options must keep 3 common items and 6 other items"))

_ManualEdit = tuple[int, int, int, int, np.ndarray, np.ndarray]
_ManualRollback = tuple[
    int,
    int,
    int,
    int,
    np.ndarray,
    list[_ManualEdit],
    list[_ManualEdit],
    bool,
    bool,
]

_PROFESSIONAL_CONTROL_ATTRS: dict[str, str] = {
    "scale_mode": "scale_mode_combo",
    "reference_short_side": "reference_short_side_spin",
    "signal_mode": "signal_combo",
    "invert": "polarity_combo",
    "red_weight": "red_weight_spin",
    "green_weight": "green_weight_spin",
    "blue_weight": "blue_weight_spin",
    "black_percentile": "black_percentile_spin",
    "white_percentile": "white_percentile_spin",
    "gamma": "gamma_spin",
    "intensity_space": "intensity_space_combo",
    "rank_tail_power": "rank_tail_power_spin",
    "snr_seed_sigma": "snr_seed_sigma_spin",
    "denoise_method": "denoise_method_combo",
    "denoise_strength": "denoise_strength_spin",
    "background_method": "background_method_combo",
    "background_radius": "background_radius_spin",
    "background_strength": "background_strength_spin",
    "threshold_method": "threshold_method_combo",
    "sensitivity": "sensitivity_spin",
    "manual_threshold": "manual_threshold_spin",
    "threshold_percentile": "threshold_percentile_spin",
    "local_window": "local_window_spin",
    "local_k": "local_k_spin",
    "grow_ratio": "grow_ratio_spin",
    "open_radius": "open_radius_spin",
    "close_radius": "close_radius_spin",
    "erode_radius": "erode_radius_spin",
    "dilate_radius": "dilate_radius_spin",
    "hole_area": "hole_area_spin",
    "min_area": "min_area_spin",
    "max_area": "max_area_spin",
    "remove_border": "remove_border_check",
    "split_touching": "split_touching_check",
    "watershed_peak_distance": "peak_distance_spin",
    "watershed_peak_height": "peak_height_spin",
    "min_circularity": "min_circularity_spin",
    "min_solidity": "min_solidity_spin",
    "max_eccentricity": "max_eccentricity_spin",
    "max_aspect_ratio": "max_aspect_ratio_spin",
    "shape_profile": "shape_profile_combo",
    "shape_profile_strictness": "shape_profile_strictness_combo",
    "min_thickness": "min_thickness_spin",
    "max_thickness": "max_thickness_spin",
    "min_mean_intensity": "min_mean_intensity_spin",
    "min_response_confidence": "min_response_confidence_spin",
    "noise_reference_value": "noise_reference_value_spin",
    "background_ceiling_mode": "background_ceiling_mode_combo",
    "background_ceiling_scale": "background_ceiling_scale_spin",
    "background_ceiling": "background_ceiling_spin",
}

if set(_PROFESSIONAL_CONTROL_ATTRS) | {"recognition_engine", "learned_sensitivity"} != {
    field.name for field in dataclass_fields(UniversalParams)
}:
    raise RuntimeError(T(
        "The registry of professional parameter controls must cover UniversalParams completely",
    ))

_PROFESSIONAL_PARAMETER_COUNT = len(_PROFESSIONAL_CONTROL_ATTRS)


def _safe_error_message(error: BaseException, *paths: Path) -> str:
    """Return an actionable error without exposing local directory structure."""

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
        replacement = path.name or T("selected location")
        for value in sorted(variants, key=len, reverse=True):
            if value:
                message = message.replace(value, replacement)
    message = re.sub(
        r"(?<![\w.])(?:[A-Za-z]:[\\/]|\\\\|/)(?:[^\s:;,]+[\\/])*[^\s:;,]*",
        T("local path"),
        message,
    )
    return message[:600]


class NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float(left) < float(right)
        return super().__lt__(other)


class QueueTree(QTreeWidget):
    """Image queue shown either as the folder tree the files came from, or flat.

    The surrounding window addresses the queue by *row*, where a row is an
    index into the window's ordered list of visible queue paths.  A tree has
    no rows, so this class keeps the row-addressed API the rest of the window
    already uses and maps it onto leaf items.  Folder rows are structure only:
    they are not selectable, so ``selectedItems`` can never hand the caller
    something that is not an image, and "remove selected" cannot silently mean
    "remove a directory".  The second column carries each image's processing
    state so a long batch can be followed and resumed at a glance.
    """

    currentRowChanged = Signal(int)

    _ROW_ROLE = Qt.UserRole + 701
    STATUS_COLUMN_WIDTH = 66

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setColumnCount(2)
        self.setUniformRowHeights(True)
        self.setExpandsOnDoubleClick(False)
        self.setIndentation(13)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Indentation eats width, and these file names differ near their end.
        # Let the column keep its natural width and scroll instead of eliding
        # the part that tells two channels of one ROI apart.
        self.setTextElideMode(Qt.ElideNone)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.resizeSection(1, self.STATUS_COLUMN_WIDTH)
        self._leaves: list[QTreeWidgetItem] = []
        self._collapsed: set[str] = set()
        self._emitted_row = -1
        self.currentItemChanged.connect(self._on_current_item_changed)
        self.itemExpanded.connect(self._remember_expansion)
        self.itemCollapsed.connect(self._remember_expansion)

    # -- row-addressed API used by the window ----------------------------
    def count(self) -> int:
        return len(self._leaves)

    def item(self, row: int) -> QTreeWidgetItem | None:
        if 0 <= row < len(self._leaves):
            return self._leaves[row]
        return None

    def row(self, item: QTreeWidgetItem | None) -> int:
        if item is None:
            return -1
        value = item.data(0, self._ROW_ROLE)
        return int(value) if value is not None else -1

    def currentRow(self) -> int:
        """The queue row on screen, which a folder never changes.

        Callers use this to mean "which image is current".  A folder is
        structure, not an image, so when one is the current *item* the answer is
        still the last image row — otherwise clicking a directory would read as
        "no image selected" and disable the actions that act on it.
        """

        row = self.row(self.currentItem())
        if row >= 0:
            return row
        if 0 <= self._emitted_row < len(self._leaves):
            return self._emitted_row
        return -1

    def setCurrentRow(self, row: int) -> None:
        target = self.item(row)
        if target is None:
            self.setCurrentItem(None)
            return
        parent = target.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        self.setCurrentItem(target)
        self.scrollToItem(target)

    def selectedItems(self) -> list[QTreeWidgetItem]:
        return [item for item in super().selectedItems() if self.row(item) >= 0]

    # -- construction -----------------------------------------------------
    def rebuild(
        self,
        segments_for_row: Sequence[Sequence[str]],
        tooltips: Sequence[str],
        statuses: Sequence[tuple[str, str, str]] | None = None,
        *,
        current_row: int,
        flat: bool = False,
    ) -> None:
        """Replace the whole tree from ``segments_for_row``.

        Each entry is the path segments of one queue row, deepest last.  In
        flat mode every row becomes one top-level leaf labelled with its full
        relative path, so a global sort (for example by file name across all
        folders) is shown exactly in processing order.  The tree is rebuilt
        rather than patched so that a newly imported folder can re-parent rows
        that were previously shown flat, which incremental insertion cannot
        do.  Collapsed folders stay collapsed across a rebuild.
        """

        blocked = self.blockSignals(True)
        try:
            self.clear()
            self._leaves = []
            folders: dict[tuple[str, ...], QTreeWidgetItem] = {}
            for row, segments in enumerate(segments_for_row):
                parts = [str(part) for part in segments if str(part)]
                if not parts:
                    parts = [T("(unnamed)")]
                parent: QTreeWidgetItem | None = None
                if flat:
                    leaf = QTreeWidgetItem(self, ["/".join(parts), ""])
                else:
                    for depth in range(len(parts) - 1):
                        key = tuple(parts[: depth + 1])
                        node = folders.get(key)
                        if node is None:
                            node = (
                                QTreeWidgetItem(self, [parts[depth], ""])
                                if parent is None
                                else QTreeWidgetItem(parent, [parts[depth], ""])
                            )
                            # Structure only: never selectable, so "remove
                            # selected" always means images and only images.
                            node.setFlags(Qt.ItemIsEnabled)
                            node.setForeground(0, QColor("#93a6bb"))
                            font = node.font(0)
                            font.setBold(True)
                            node.setFont(0, font)
                            node.setToolTip(0, "/".join(key))
                            folders[key] = node
                        parent = node
                    leaf = (
                        QTreeWidgetItem(self, [parts[-1], ""])
                        if parent is None
                        else QTreeWidgetItem(parent, [parts[-1], ""])
                    )
                leaf.setData(0, self._ROW_ROLE, row)
                if row < len(tooltips):
                    leaf.setToolTip(0, tooltips[row])
                if statuses is not None and row < len(statuses):
                    self._apply_status(leaf, *statuses[row])
                self._leaves.append(leaf)
            for key, node in folders.items():
                node.setExpanded("/".join(key) not in self._collapsed)
        finally:
            self.blockSignals(blocked)
        if 0 <= current_row < len(self._leaves):
            self.setCurrentRow(current_row)
            self._emitted_row = current_row
        else:
            # The previously current image is no longer a row (filtered out
            # or removed): forget it, so that the next click on any row is
            # reported instead of being mistaken for "still the same row".
            self._emitted_row = -1

    @staticmethod
    def _apply_status(item: QTreeWidgetItem, text: str, color: str, tooltip: str) -> None:
        item.setText(1, text)
        item.setForeground(1, QColor(color))
        item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
        item.setToolTip(1, tooltip)

    def set_row_status(self, row: int, text: str, color: str, tooltip: str) -> None:
        """Update one row's status column in place (no rebuild)."""

        item = self.item(row)
        if item is not None:
            self._apply_status(item, text, color, tooltip)

    def mark_active_row(self, row: int) -> None:
        """Give the image actually loaded a stronger mark than mere selection.

        Selection is a multi-item concept used for removal; exactly one queued
        image is on screen.  Conflating the two makes a multi-select look like
        it changed the displayed image, so the loaded row is marked separately.
        """

        for index, leaf in enumerate(self._leaves):
            active = index == row
            font = leaf.font(0)
            font.setBold(active)
            leaf.setFont(0, font)
            leaf.setForeground(
                0,
                QColor("#8ef0c8") if active else QColor("#d6e2ee"),
            )
            # A background rather than a text marker: the label is a file name
            # the user reads and compares, so it must not change as it is
            # loaded, and the tree's indentation must stay aligned.
            background = QColor(32, 128, 96, 90) if active else QColor(0, 0, 0, 0)
            leaf.setBackground(0, background)
            leaf.setBackground(1, background)

    # -- internals ---------------------------------------------------------
    def _remember_expansion(self, item: QTreeWidgetItem) -> None:
        if self.row(item) >= 0:
            return
        parts: list[str] = []
        node: QTreeWidgetItem | None = item
        while node is not None:
            parts.append(node.text(0))
            node = node.parent()
        key = "/".join(reversed(parts))
        if item.isExpanded():
            self._collapsed.discard(key)
        else:
            self._collapsed.add(key)

    def _on_current_item_changed(
        self,
        current: QTreeWidgetItem | None,
        previous: QTreeWidgetItem | None,
    ) -> None:
        row = self.row(current)
        # A folder is not a row.  Keeping the previous row avoids reporting -1
        # and tearing down the loaded image when the user clicks a directory.
        if row < 0:
            return
        if row == self._emitted_row:
            return
        self._emitted_row = row
        self.currentRowChanged.emit(row)


class FlexibleComboBox(QComboBox):
    """Combo box that may be narrower than its longest item.

    The closed box shows as much of the current item as fits, so a long label
    cannot force a side panel wider than the window allows; the list that opens
    is always wide enough for every item.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(8)

    def showPopup(self) -> None:  # noqa: N802 - Qt API
        view = self.view()
        needed = view.sizeHintForColumn(0) + view.verticalScrollBar().sizeHint().width() + 14
        view.setMinimumWidth(max(self.width(), needed))
        super().showPopup()

    def event(self, event) -> bool:  # type: ignore[no-untyped-def]
        # Without a tool tip of its own, a box too narrow for its text shows the text in full.
        if event.type() == QEvent.ToolTip and not self.toolTip():
            text = self.currentText()
            if self.fontMetrics().horizontalAdvance(text) > self.width() - 38:
                QToolTip.showText(event.globalPos(), text, self)
                return True
        return super().event(event)


class PresetCombo(FlexibleComboBox):
    """Drop-down preset selector with the small API the window relied on.

    The former always-visible chip wall listed all 24 built-in presets and
    consumed most of the panel; a grouped drop-down shows the current preset
    in one line and keeps the frequently used controls below it reachable
    without scrolling.
    """

    choice_activated = Signal(object)

    def __init__(self, *, accessible_name: str = T("Recognition preset"), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._keys: list[tuple[str, str] | None] = []
        self.setAccessibleName(accessible_name)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(18)
        self.activated.connect(self._on_activated)

    def setChoices(  # noqa: N802 - Qt-style API
        self,
        choices: Iterable[tuple[tuple[str, str], str, str]],
        tooltips: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        """Fill the list from (key, label, group); a separator divides the groups."""

        blocked = self.blockSignals(True)
        try:
            self.clear()
            self._keys = []
            last_group: str | None = None
            for key, label, group in choices:
                if last_group is not None and group != last_group:
                    self.insertSeparator(self.count())
                    self._keys.append(None)
                last_group = group
                self.addItem(label)
                self._keys.append((str(key[0]), str(key[1])))
                if tooltips and key in tooltips:
                    self.setItemData(self.count() - 1, tooltips[key], Qt.ToolTipRole)
        finally:
            self.blockSignals(blocked)

    def currentData(self, role: int = Qt.UserRole) -> tuple[str, str] | None:  # noqa: N802
        index = self.currentIndex()
        if 0 <= index < len(self._keys):
            return self._keys[index]
        return None

    def setCurrentData(self, value: object) -> bool:  # noqa: N802 - Qt-style API
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return False
        wanted = (str(value[0]), str(value[1]))
        for index, key in enumerate(self._keys):
            if key == wanted:
                blocked = self.blockSignals(True)
                try:
                    self.setCurrentIndex(index)
                finally:
                    self.blockSignals(blocked)
                return True
        return False

    def buttonForData(self, value: object) -> None:  # noqa: N802 - compatibility
        return None

    def _on_activated(self, index: int) -> None:
        if 0 <= index < len(self._keys) and self._keys[index] is not None:
            self.choice_activated.emit(self._keys[index])


class SignedSpinBox(QSpinBox):
    """Integer spin box that always shows the sign (``+3 px`` / ``-2 px``)."""

    def textFromValue(self, value: int) -> str:  # noqa: N802 - Qt API
        return f"{int(value):+d}" if value else "0"

    def valueFromText(self, text: str) -> int:  # noqa: N802 - Qt API
        cleaned = text.replace(self.prefix(), "").replace(self.suffix(), "").strip()
        cleaned = cleaned.replace("＋", "+").replace("−", "-").replace("－", "-")
        try:
            return int(cleaned)
        except ValueError:
            return int(self.value())


class FlowLayout(QLayout):
    """Compact wrapping layout for always-visible option buttons."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        horizontal_spacing: int = 5,
        vertical_spacing: int = 5,
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._horizontal_spacing = horizontal_spacing
        self._vertical_spacing = vertical_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt API
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt API
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt API
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt API
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        return self._do_layout(QRect(0, 0, max(0, width), 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(),
            margins.top(),
            -margins.right(),
            -margins.bottom(),
        )
        x = effective.x()
        y = effective.y()
        line_height = 0
        line: list[tuple[QLayoutItem, int, QSize]] = []

        def place_line() -> None:
            # Items of different heights (a label next to a button) are
            # centred within their row instead of hugging its top edge.
            for line_item, item_x, size in line:
                offset = (line_height - size.height()) // 2
                line_item.setGeometry(QRect(item_x, y + offset, size.width(), size.height()))

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._horizontal_spacing
            if line_height > 0 and next_x - self._horizontal_spacing > effective.right() + 1:
                if not test_only:
                    place_line()
                line = []
                x = effective.x()
                y += line_height + self._vertical_spacing
                next_x = x + hint.width() + self._horizontal_spacing
                line_height = 0
            line.append((item, x, hint))
            x = next_x
            line_height = max(line_height, hint.height())
        if not test_only:
            place_line()
        return y + line_height - rect.y() + margins.bottom()


class CanvasToolStrip(QFrame):
    """One-row tool strip that wraps onto a second row when the canvas is
    narrow, so no tool is ever clipped off the picture's edge."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.flow = FlowLayout(self, horizontal_spacing=2, vertical_spacing=3)
        self.flow.setContentsMargins(6, 3, 6, 3)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        margins = self.flow.contentsMargins()
        # The style-sheet border is outside the layout's rectangle.
        frame = self.contentsMargins()
        width = margins.left() + margins.right() + frame.left() + frame.right()
        height = 0
        count = self.flow.count()
        for index in range(count):
            item = self.flow.itemAt(index)
            if item is None:
                continue
            hint = item.sizeHint()
            width += hint.width() + (2 if index < count - 1 else 0)
            height = max(height, hint.height())
        return QSize(
            width,
            height + margins.top() + margins.bottom() + frame.top() + frame.bottom(),
        )

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        frame = self.contentsMargins()
        return self.flow.minimumSize() + QSize(
            frame.left() + frame.right(), frame.top() + frame.bottom()
        )


class ChoiceButtonGroup(QWidget):
    """Always-visible single choice, or an explicit checkbox-like OR group."""

    value_changed = Signal(object)
    activated = Signal(object)

    def __init__(
        self,
        choices: Mapping[Hashable, str] | Iterable[tuple[Hashable, str]] | None = None,
        *,
        accessible_name: str = T("Options"),
        multi_select: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._accessible_name = accessible_name
        self._multi_select = bool(multi_select)
        self._current_value: Hashable | None = None
        self._buttons: dict[Hashable, QRadioButton | QCheckBox] = {}
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(not self._multi_select)
        self._flow = FlowLayout(self)
        self.setAccessibleName(accessible_name)
        self.setAccessibleDescription(
            T(
                "Several can be ticked at once; click again to clear; with none ticked the preset "
                "is followed.",
            )
            if self._multi_select
            else T("Choose one per row.")
        )
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        if choices is not None:
            items = choices.items() if isinstance(choices, Mapping) else choices
            self.setChoices(items)

    def setChoices(
        self,
        choices: Iterable[tuple[Hashable, str]],
    ) -> None:  # noqa: N802 - Qt-style API
        previous = self._current_value
        while self._flow.count():
            item = self._flow.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                self._button_group.removeButton(widget)
                widget.hide()
                widget.deleteLater()
        self._buttons = {}
        self._current_value = None
        first_value: Hashable | None = None
        for value, label in choices:
            if first_value is None:
                first_value = value
            button = (
                QCheckBox(str(label), self)
                if self._multi_select
                else QRadioButton(str(label), self)
            )
            button.setProperty("choiceButton", True)
            button.setAccessibleName(T("{0}: {1}", self._accessible_name, label))
            button.setToolTip(str(label))
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.clicked.connect(
                lambda _checked=False, selected=value: self._button_clicked(selected)
            )
            self._buttons[value] = button
            self._button_group.addButton(button)
            self._flow.addWidget(button)
        if self._multi_select:
            target: object = previous if previous is not None else "follow"
            if not self.setCurrentData(target) and first_value is not None:
                self.setCurrentData(first_value)
        else:
            target = previous if previous in self._buttons else first_value
            if target is not None:
                self.setCurrentData(target)
        self.updateGeometry()

    def _button_clicked(self, value: Hashable) -> None:
        if self._multi_select:
            previous = self._current_value
            clicked = self._buttons[value]
            if value == "follow":
                for candidate, button in self._buttons.items():
                    button.setChecked(candidate == "follow")
            elif clicked.isChecked():
                follow = self._buttons.get("follow")
                if follow is not None:
                    follow.setChecked(False)
                if value == "any":
                    for candidate, button in self._buttons.items():
                        if candidate not in {"follow", "any"}:
                            button.setChecked(False)
                else:
                    any_button = self._buttons.get("any")
                    if any_button is not None:
                        any_button.setChecked(False)
            selected = self.checkedData()
            if not selected:
                follow = self._buttons.get("follow")
                if follow is not None:
                    follow.setChecked(True)
                selected = ("follow",) if follow is not None else ()
            self._current_value = "+".join(str(item) for item in selected)
            if self._current_value != previous:
                self.value_changed.emit(self._current_value)
            self.activated.emit(value)
            return
        changed = value != self._current_value
        self._current_value = value
        if changed:
            self.value_changed.emit(value)
        self.activated.emit(value)

    def currentData(self) -> Hashable | None:  # noqa: N802 - Qt-style API
        return self._current_value

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        return self._flow.heightForWidth(max(0, width))

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        preferred_width = 360
        return QSize(preferred_width, self.heightForWidth(preferred_width))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        height = max(
            (button.minimumSizeHint().height() for button in self._buttons.values()),
            default=0,
        )
        return QSize(0, height)

    def setCurrentData(self, value: object) -> bool:  # noqa: N802 - Qt-style API
        if self._multi_select:
            if isinstance(value, str):
                requested = tuple(item for item in value.split("+") if item)
            elif isinstance(value, Iterable):
                requested = tuple(value)
            else:
                requested = (value,)
            if not requested:
                requested = ("follow",)
            if any(item not in self._buttons for item in requested):
                return False
            if "follow" in requested and len(requested) > 1:
                requested = tuple(item for item in requested if item != "follow")
            if "any" in requested:
                requested = ("any",)
            for candidate, button in self._buttons.items():
                button.setChecked(candidate in requested)
            selected = self.checkedData()
            if not selected and "follow" in self._buttons:
                self._buttons["follow"].setChecked(True)
                selected = ("follow",)
            self._current_value = "+".join(str(item) for item in selected)
            return True
        button = self._buttons.get(value)
        if button is None:
            return False
        button.setChecked(True)
        self._current_value = value
        return True

    def buttonForData(  # noqa: N802
        self,
        value: Hashable,
    ) -> QRadioButton | QCheckBox | None:
        return self._buttons.get(value)

    def checkedData(self) -> tuple[Hashable, ...]:  # noqa: N802 - Qt-style API
        return tuple(
            value for value, button in self._buttons.items() if button.isChecked()
        )

    def setChoiceToolTip(self, text: str) -> None:  # noqa: N802 - Qt-style API
        self.setToolTip(text)
        for button in self._buttons.values():
            button.setToolTip(text)
            button.setAccessibleDescription(text)


@dataclass(frozen=True)
class ExportPlan:
    """One fully confirmed export request, frozen before background work."""

    output_root: Path | None
    artifacts: frozenset[str]
    outline_expand_px: int
    merge_base_mode: str
    location_mode: str


class ExportOptionsDialog(QDialog):
    """Shared, non-technical artifact chooser for current and batch export."""

    def __init__(
        self,
        selected_artifacts: Iterable[str] | None = None,
        *,
        context_text: str = "",
        batch_count: int | None = None,
        outline_expand_px: int = 0,
        crosstalk_available: bool = False,
        merge_base_mode: str = MERGE_BASE_MODE_ANALYSIS_INPUT,
        export_location_mode: str = EXPORT_LOCATION_MIRROR,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(T("Choose what to export"))
        self.setModal(True)
        self.setMinimumWidth(600)
        self.artifact_checks: dict[str, QCheckBox] = {}
        self._batch_count = batch_count
        self._crosstalk_available = bool(crosstalk_available)
        initial_merge_base_mode = normalize_merge_base_mode(merge_base_mode)
        initial_location_mode = normalize_export_location_mode(export_location_mode)
        if initial_location_mode == EXPORT_LOCATION_FLAT:
            initial_location_mode = EXPORT_LOCATION_MIRROR

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        heading = QLabel(
            T("Batch export of {0:,} images", batch_count) if batch_count is not None else T("Export the current image"),
            self,
        )
        heading.setStyleSheet("font-size: 17px; font-weight: 700;")
        layout.addWidget(heading)
        if context_text:
            context_label = QLabel(context_text, self)
            context_label.setWordWrap(True)
            context_label.setObjectName("mutedText")
            layout.addWidget(context_label)
        instruction = QLabel(
            T("Tick what to save, then choose where to save it."),
            self,
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        self.location_group = QGroupBox(T("Where to save the results"), self)
        location_layout = QVBoxLayout(self.location_group)
        location_layout.setContentsMargins(12, 18, 12, 10)
        location_layout.setSpacing(6)
        self.location_mirror_radio = QRadioButton(
            T("Choose an output root; keep the imported folder structure (recommended)"),
            self.location_group,
        )
        self.location_source_radio = QRadioButton(
            T("Export into the folder of each original image"),
            self.location_group,
        )
        self.location_buttons = QButtonGroup(self.location_group)
        self.location_buttons.setExclusive(True)
        self.location_buttons.addButton(self.location_mirror_radio)
        self.location_buttons.addButton(self.location_source_radio)
        if initial_location_mode == EXPORT_LOCATION_SOURCE:
            self.location_source_radio.setChecked(True)
        else:
            self.location_mirror_radio.setChecked(True)
        location_layout.addWidget(self.location_mirror_radio)
        location_layout.addWidget(self.location_source_radio)
        self.location_note = QLabel(
            T(
                "Either way the result files are written directly into the folder concerned, "
                "named \"name_FluoroAnts_item\" to avoid clashes; no per-image subfolder is "
                "created and the original images are never overwritten. These result files are "
                "ignored automatically when a folder is imported recursively again.",
            ),
            self.location_group,
        )
        self.location_note.setWordWrap(True)
        self.location_note.setObjectName("mutedText")
        location_layout.addWidget(self.location_note)
        layout.addWidget(self.location_group)

        def add_artifact_check(
            parent_widget: QWidget,
            target_layout: QVBoxLayout,
            artifact: tuple[str, str, str],
        ) -> None:
            artifact_id, label, details = artifact
            check = QCheckBox(label, parent_widget)
            check.setAccessibleName(T("Export: {0}", label))
            check.setAccessibleDescription(details)
            check.setToolTip(details)
            check.stateChanged.connect(self._update_selection_state)
            target_layout.addWidget(check)
            self.artifact_checks[artifact_id] = check

        common_group = QGroupBox(T("Common exports"), self)
        common_group.setAccessibleDescription(T(
            "Common results for everyday viewing, the selection mask and the summary",
        ))
        common_layout = QVBoxLayout(common_group)
        common_layout.setContentsMargins(12, 18, 12, 10)
        common_layout.setSpacing(6)
        for artifact in _COMMON_EXPORT_ITEMS:
            add_artifact_check(common_group, common_layout, artifact)
        layout.addWidget(common_group)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(8)
        self.common_button = QPushButton(T("Common only"), self)
        self.all_button = QPushButton(T("Tick all"), self)
        self.common_button.setToolTip(T("Select only the 3 common items above"))
        self.all_button.setToolTip(
            T("Select all {0} export files this version supports", len(_ALL_EXPORT_ARTIFACTS))
        )
        for button in (self.common_button, self.all_button):
            quick_row.addWidget(button)
        quick_row.addStretch(1)
        layout.addLayout(quick_row)

        outline_row = QHBoxLayout()
        outline_row.setSpacing(8)
        self.outline_expand_label = QLabel(T("Outline expansion"), self)
        self.outline_expand_spin = QSpinBox(self)
        self.outline_expand_spin.setRange(0, 512)
        self.outline_expand_spin.setSuffix(" px")
        self.outline_expand_spin.setValue(
            max(0, min(512, int(outline_expand_px)))
        )
        self.outline_expand_label.setBuddy(self.outline_expand_spin)
        outline_description = (
            T("Changes only the outline in the exported Merge image, not the mask or the area.")
        )
        self.outline_expand_spin.setToolTip(outline_description)
        self.outline_expand_spin.setAccessibleName(T(
            "Outline expansion in the exported Merge image",
        ))
        self.outline_expand_spin.setAccessibleDescription(outline_description)
        self.outline_expand_note = QLabel(outline_description, self)
        self.outline_expand_note.setWordWrap(True)
        self.outline_expand_note.setObjectName("mutedText")
        outline_row.addWidget(self.outline_expand_label)
        outline_row.addWidget(self.outline_expand_spin)
        outline_row.addWidget(self.outline_expand_note, 1)
        layout.addLayout(outline_row)

        self.merge_base_group = QGroupBox(T("Base of the Merge / review image"), self)
        merge_base_layout = QVBoxLayout(self.merge_base_group)
        merge_base_layout.setContentsMargins(12, 18, 12, 10)
        merge_base_layout.setSpacing(6)
        self.merge_analysis_radio = QRadioButton(
            T("Base after crosstalk correction (as exported now)"),
            self.merge_base_group,
        )
        self.merge_original_radio = QRadioButton(
            T("Original image as base"),
            self.merge_base_group,
        )
        self.merge_base_buttons = QButtonGroup(self.merge_base_group)
        self.merge_base_buttons.setExclusive(True)
        self.merge_base_buttons.addButton(self.merge_analysis_radio)
        self.merge_base_buttons.addButton(self.merge_original_radio)
        if initial_merge_base_mode == MERGE_BASE_MODE_SOURCE_ORIGINAL:
            self.merge_original_radio.setChecked(True)
        else:
            self.merge_analysis_radio.setChecked(True)
        merge_base_layout.addWidget(self.merge_analysis_radio)
        merge_base_layout.addWidget(self.merge_original_radio)
        self.merge_base_note = QLabel(
            T(
                "Changes only the background of the Merge image; the black-and-white outline "
                "still comes from the same final mask, and the area, the statistics and the "
                "exported mask are unchanged.",
            ),
            self.merge_base_group,
        )
        merge_base_description = self.merge_base_note.text()
        for radio in (self.merge_analysis_radio, self.merge_original_radio):
            radio.setToolTip(merge_base_description)
            radio.setAccessibleDescription(merge_base_description)
        self.merge_base_note.setWordWrap(True)
        self.merge_base_note.setObjectName("mutedText")
        merge_base_layout.addWidget(self.merge_base_note)
        self.merge_base_group.setVisible(False)
        layout.addWidget(self.merge_base_group)

        self.advanced_toggle = QPushButton(self)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setAccessibleName(T("Show or hide the other export items"))
        self.advanced_toggle.toggled.connect(self._toggle_advanced_section)
        layout.addWidget(self.advanced_toggle)

        self.advanced_content = QWidget(self)
        advanced_layout = QVBoxLayout(self.advanced_content)
        advanced_layout.setContentsMargins(12, 2, 12, 4)
        advanced_layout.setSpacing(6)
        for artifact in _ADVANCED_EXPORT_ITEMS:
            add_artifact_check(self.advanced_content, advanced_layout, artifact)
        self.advanced_content.setVisible(False)
        layout.addWidget(self.advanced_content)

        self.selection_count_label = QLabel(self)
        self.selection_count_label.setWordWrap(True)
        layout.addWidget(self.selection_count_label)
        self.selection_warning = QLabel(T("Choose at least one item to export."), self)
        self.selection_warning.setStyleSheet("color: #b42318; font-weight: 600;")
        self.selection_warning.setWordWrap(True)
        layout.addWidget(self.selection_warning)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel, self)
        self.start_button = self.button_box.addButton(
            T("Next: choose where to save"),
            QDialogButtonBox.AcceptRole,
        )
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.location_mirror_radio.toggled.connect(self._update_location_state)
        self.location_source_radio.toggled.connect(self._update_location_state)

        self.common_button.clicked.connect(
            lambda: self.set_selected_artifacts(_COMMON_EXPORT_ARTIFACTS)
        )
        self.all_button.clicked.connect(
            lambda: self.set_selected_artifacts(_ALL_EXPORT_ARTIFACTS)
        )

        initial = (
            _COMMON_EXPORT_ARTIFACTS
            if selected_artifacts is None
            else frozenset(selected_artifacts) & EXPORT_ARTIFACT_IDS
        )
        self.set_selected_artifacts(initial)
        self._update_location_state()

    def selected_artifacts(self) -> frozenset[str]:
        return frozenset(
            artifact_id
            for artifact_id, check in self.artifact_checks.items()
            if check.isChecked()
        )

    def outline_expand_px(self) -> int:
        return int(self.outline_expand_spin.value())

    def export_location_mode(self) -> str:
        return (
            EXPORT_LOCATION_SOURCE
            if self.location_source_radio.isChecked()
            else EXPORT_LOCATION_MIRROR
        )

    def merge_base_mode(self) -> str:
        if not self._crosstalk_available:
            return MERGE_BASE_MODE_ANALYSIS_INPUT
        return (
            MERGE_BASE_MODE_SOURCE_ORIGINAL
            if self.merge_original_radio.isChecked()
            else MERGE_BASE_MODE_ANALYSIS_INPUT
        )

    def set_selected_artifacts(self, artifacts: Iterable[str]) -> None:
        requested = frozenset(artifacts) & EXPORT_ARTIFACT_IDS
        for artifact_id, check in self.artifact_checks.items():
            check.setChecked(artifact_id in requested)
        self._update_selection_state()

    def _toggle_advanced_section(self, expanded: bool) -> None:
        self.advanced_content.setVisible(bool(expanded))
        self._update_advanced_toggle_text()
        self._fit_height()

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802 - Qt API
        super().showEvent(event)
        self._fit_height()

    def _fit_height(self) -> None:
        """Make the dialog as tall as its wrapped texts need at the current width."""

        layout = self.layout()
        if layout is None:
            return
        layout.activate()
        needed = layout.totalHeightForWidth(self.width()) if layout.hasHeightForWidth() else self.sizeHint().height()
        if needed != self.height():
            self.resize(self.width(), max(needed, self.minimumSizeHint().height()))

    def _update_advanced_toggle_text(self) -> None:
        selected_advanced = len(
            self.selected_artifacts() - _COMMON_EXPORT_ARTIFACTS
        )
        arrow = "▼" if self.advanced_toggle.isChecked() else "▶"
        self.advanced_toggle.setText(
            T(
                "{0} Other exports ({1} items, {2} selected)",
                arrow,
                len(_ADVANCED_EXPORT_ITEMS),
                selected_advanced,
            )
        )

    def _update_selection_state(self, _state: int | None = None) -> None:
        selected = self.selected_artifacts()
        count = len(selected)
        suffix = (
            T("; in a batch export every image gets these {0} item(s)", count)
            if self._batch_count is not None and count
            else ""
        )
        self.selection_count_label.setText(T("{0} item(s) selected{1}", count, suffix))
        self._update_advanced_toggle_text()
        outline_enabled = bool(selected.intersection(_OUTLINE_EXPORT_ARTIFACTS))
        self.outline_expand_label.setEnabled(outline_enabled)
        self.outline_expand_spin.setEnabled(outline_enabled)
        self.merge_base_group.setVisible(
            bool(self._crosstalk_available and outline_enabled)
        )
        self.selection_warning.setVisible(count == 0)
        self.start_button.setEnabled(count > 0)

    def _update_location_state(self, _checked: bool | None = None) -> None:
        self.start_button.setText(
            T("Start export into the original folders")
            if self.export_location_mode() == EXPORT_LOCATION_SOURCE
            else T("Next: choose the output root")
        )

    def accept(self) -> None:
        if not self.selected_artifacts():
            self._update_selection_state()
            return
        super().accept()


class AboutDialog(QDialog):
    """Version, purpose, what the program stores, and whom to contact."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(T("About FluoroAnts Universal"))
        self.setModal(True)
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 22, 26, 18)
        layout.setSpacing(10)

        title = QLabel("FluoroAnts Universal", self)
        title.setObjectName("aboutTitle")
        layout.addWidget(title)
        version = QLabel(
            T(
                "Version {0} · recognition core {1} · {2}",
                FLUOROANTS_RELEASE_VERSION,
                UNIVERSAL_ALGORITHM_VERSION,
                _ai_engine_label("learned_v6"),
            ),
            self,
        )
        version.setObjectName("mutedText")
        version.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(version)

        purpose = QLabel(
            T(
                "Recognition of positive regions in 2-D fluorescence images, with manual review "
                "and quantitative export.\nFor research use; not for clinical diagnosis."
            ),
            self,
        )
        purpose.setWordWrap(True)
        layout.addWidget(purpose)

        privacy = QLabel(
            T(
                "This program does not connect to the internet and collects no data. Interface "
                "settings, personal presets and the crash log stay on this computer; exported "
                "files record the file name of an image, never its folder."
            ),
            self,
        )
        privacy.setObjectName("infoCard")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)

        contact = QLabel(
            T("Author: {0}", AUTHOR) + "\n" + T("E-mail: {0}", contact_address()),
            self,
        )
        contact.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(contact)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802 - Qt API
        super().showEvent(event)
        layout = self.layout()
        if layout is not None and layout.hasHeightForWidth():
            self.resize(self.width(), layout.totalHeightForWidth(self.width()))


class TaskSignals(QObject):
    result = Signal(object)
    error = Signal(object)
    progress = Signal(object)
    finished = Signal(object)


class LoadWorker(QRunnable):
    """Decode one image and prepare its display frame off the GUI thread.

    The percentile stretch of a large frame used to run inside the loaded
    slot and could freeze the window for a noticeable moment; the worker now
    hands back the ready-to-show RGB view together with the decoded image.
    """

    def __init__(
        self,
        token: int,
        path: Path,
        *,
        preserve_original_values: bool = False,
    ) -> None:
        super().__init__()
        self.token = token
        self.path = path
        self.preserve_original_values = bool(preserve_original_values)
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            loaded = load_image(self.path)
            if self.preserve_original_values:
                display, scale = image_to_rgb_original_values(
                    loaded.image,
                    max_side=_DISPLAY_MAX_SIDE,
                )
            else:
                display, scale = image_to_rgb_display(
                    loaded.image,
                    max_side=_DISPLAY_MAX_SIDE,
                )
            self.signals.result.emit((self.token, loaded, display, scale))
        except Exception as exc:
            self.signals.error.emit((self.token, _safe_error_message(exc, self.path)))
        finally:
            self.signals.finished.emit(("load", self.token))


class CrosstalkWorker(QRunnable):
    """Load one paired reference and build one reusable corrected input."""

    def __init__(
        self,
        token: int,
        serial: int,
        source: LoadedImage,
        reference_path: Path,
        target_params: UniversalParams,
        correction: CrosstalkCorrection,
        analysis_key: tuple[object, ...],
        *,
        cached_reference: LoadedImage | None = None,
    ) -> None:
        super().__init__()
        self.token = token
        self.serial = serial
        self.source = source
        self.reference_path = reference_path
        self.target_params = copy.deepcopy(target_params)
        self.correction = copy.deepcopy(correction)
        self.analysis_key = analysis_key
        self.cached_reference = cached_reference
        self.signals = TaskSignals()

    @staticmethod
    def _cache_is_current(loaded: LoadedImage, path: Path) -> bool:
        if loaded.path != path:
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        return bool(
            int(stat.st_size) == int(loaded.metadata.get("size", -1))
            and int(stat.st_mtime_ns) == int(loaded.metadata.get("mtime_ns", -1))
        )

    @Slot()
    def run(self) -> None:
        try:
            reference = self.cached_reference
            if reference is None or not self._cache_is_current(
                reference,
                self.reference_path,
            ):
                reference = load_image(self.reference_path)
            corrected = apply_crosstalk_correction(
                self.source,
                reference,
                self.target_params,
                self.correction,
            )
            corrected_display, corrected_scale = image_to_rgb_display(
                corrected.image,
                max_side=_DISPLAY_MAX_SIDE,
            )
            reference_display, reference_scale = image_to_rgb_display(
                reference.image,
                max_side=_DISPLAY_MAX_SIDE,
            )
            self.signals.result.emit(
                (
                    self.token,
                    self.serial,
                    corrected,
                    reference,
                    corrected_display,
                    corrected_scale,
                    reference_display,
                    reference_scale,
                    self.analysis_key,
                )
            )
        except Exception as exc:
            self.signals.error.emit(
                (
                    self.token,
                    self.serial,
                    _safe_error_message(
                        exc,
                        self.source.path,
                        self.reference_path,
                    ),
                )
            )
        finally:
            self.signals.finished.emit(("crosstalk", self.serial))


AI_ENGINE_LABELS = {
    "learned_v6": T("Learned model v6.3"),
}


def _uses_learned_engine(params: UniversalParams) -> bool:
    return getattr(params, "recognition_engine", "classic") in AI_RECOGNITION_ENGINES


def _ai_engine_label(engine: str) -> str:
    return AI_ENGINE_LABELS.get(engine, T("Learned model"))


class SegmentWorker(QRunnable):
    def __init__(
        self,
        token: int,
        serial: int,
        loaded: LoadedImage,
        params: UniversalParams,
        max_side: int | None,
        *,
        quick_base_params: UniversalParams | None = None,
        quick_settings: QuickSettings | None = None,
        mask: np.ndarray | None = None,
        previous: UniversalResult | None = None,
        edge_offset_px: int = 0,
    ) -> None:
        super().__init__()
        self.token = token
        self.serial = serial
        self.loaded = loaded
        self.params = copy.deepcopy(params)
        self.max_side = max_side
        self.quick_base_params = copy.deepcopy(quick_base_params)
        self.quick_settings = copy.deepcopy(quick_settings)
        self.mask = None if mask is None else np.asarray(mask, dtype=bool).copy()
        self.previous = previous
        # Post-recognition boundary offset.  Applied only to automatic
        # segmentations; a manual rebuild keeps whatever offset its mask
        # already carries (documented on the previous result).
        self.edge_offset_px = _validated_edge_offset(edge_offset_px)
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            if self.mask is None:
                has_quick_base = self.quick_base_params is not None
                has_quick_settings = self.quick_settings is not None
                if not _uses_learned_engine(self.params) and has_quick_base != has_quick_settings:
                    raise ValueError(T("The recognition plan of the quick settings is incomplete."))
                if _uses_learned_engine(self.params):
                    result = segment_loaded_image(self.loaded, self.params, max_side=None)
                elif has_quick_base and has_quick_settings:
                    assert self.quick_base_params is not None
                    assert self.quick_settings is not None
                    result = segment_loaded_image_with_quick(
                        self.loaded,
                        self.quick_base_params,
                        self.quick_settings,
                        max_side=self.max_side,
                    )
                else:
                    result = segment_loaded_image(
                        self.loaded,
                        self.params,
                        max_side=self.max_side,
                    )
                if self.edge_offset_px:
                    result = apply_edge_offset(result, self.edge_offset_px)
            else:
                result = rebuild_result_from_mask(
                    self.loaded,
                    self.mask,
                    self.params,
                    self.previous,
                )
            result = _tag_result_analysis_input(result, self.loaded)
            self.signals.result.emit((self.token, self.serial, result))
        except Exception as exc:
            self.signals.error.emit(
                (self.token, self.serial, _safe_error_message(exc, self.loaded.path))
            )
        finally:
            self.signals.finished.emit(("segment", self.serial))


class EdgeOffsetWorker(QRunnable):
    """Re-apply the edge distance to a restored (imported) selection.

    An imported Mask has no recognition to re-run, so a changed edge distance
    is computed from the pristine imported result each time rather than
    cumulatively from the previous offset.
    """

    def __init__(
        self,
        token: int,
        serial: int,
        loaded: LoadedImage,
        base_result: UniversalResult,
        edge_offset_px: int,
    ) -> None:
        super().__init__()
        self.token = token
        self.serial = serial
        self.loaded = loaded
        self.base_result = base_result
        self.edge_offset_px = _validated_edge_offset(edge_offset_px)
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = apply_edge_offset(self.base_result, self.edge_offset_px)
            if result is self.base_result:
                result = copy.copy(result)
                result.mask = np.array(self.base_result.mask, dtype=bool, copy=True)
            result = _tag_result_analysis_input(result, self.loaded)
            self.signals.result.emit((self.token, self.serial, result))
        except Exception as exc:
            self.signals.error.emit(
                (self.token, self.serial, _safe_error_message(exc, self.loaded.path))
            )
        finally:
            self.signals.finished.emit(("segment", self.serial))


class ExportWorker(QRunnable):
    def __init__(
        self,
        loaded: LoadedImage,
        params: UniversalParams,
        result: UniversalResult | None,
        output_root: Path | None,
        *,
        relative_path: str | Path | None = None,
        export_location_mode: str = EXPORT_LOCATION_FLAT,
        artifacts: Iterable[str] | None = None,
        outline_expand_px: int = 0,
        merge_base_mode: str = MERGE_BASE_MODE_ANALYSIS_INPUT,
        merge_source_loaded: LoadedImage | None = None,
        quick_base_params: UniversalParams | None = None,
        quick_settings: QuickSettings | None = None,
        edge_offset_px: int = 0,
    ) -> None:
        super().__init__()
        self.loaded = loaded
        self.params = copy.deepcopy(params)
        self.result = result
        self.edge_offset_px = _validated_edge_offset(edge_offset_px)
        self.location_mode = normalize_export_location_mode(export_location_mode)
        self.output_root = (
            None if output_root is None else Path(output_root).expanduser().resolve()
        )
        self.export_parent = resolve_export_parent(
            loaded.path,
            self.output_root,
            relative_path=relative_path,
            location_mode=self.location_mode,
        )
        self.allowed_output_root = (
            loaded.path.expanduser().resolve().parent
            if self.location_mode == EXPORT_LOCATION_SOURCE
            else self.output_root
        )
        self.artifacts = frozenset(
            DEFAULT_EXPORT_ARTIFACTS if artifacts is None else artifacts
        )
        self.outline_expand_px = max(0, min(512, int(outline_expand_px)))
        self.merge_base_mode = normalize_merge_base_mode(merge_base_mode)
        self.merge_source_loaded = merge_source_loaded
        self.quick_base_params = copy.deepcopy(quick_base_params)
        self.quick_settings = copy.deepcopy(quick_settings)
        self.signals = TaskSignals()

    def _edge_offset_matches(self, result: UniversalResult) -> bool:
        """A hand-corrected Mask is always exported as it is on screen.

        The window re-runs recognition (after confirmation) whenever the edge
        distance changes, so a manual result on screen already carries the
        current offset; never discard a user's corrections at export time.
        """

        if bool(result.manually_edited):
            return True
        return int(getattr(result, "edge_offset_px", 0)) == self.edge_offset_px

    @Slot()
    def run(self) -> None:
        try:
            result = self.result
            has_quick_base = self.quick_base_params is not None
            has_quick_settings = self.quick_settings is not None
            if not _uses_learned_engine(self.params) and has_quick_base != has_quick_settings:
                raise ValueError(T(
                    "The recognition plan of the quick settings used for the export is incomplete.",
                ))
            if _uses_learned_engine(self.params):
                self.loaded = self.merge_source_loaded or self.loaded
                if (
                    result is None
                    or result.is_preview
                    or result.params.recognition_engine != self.params.recognition_engine
                    or result.params.learned_sensitivity != self.params.learned_sensitivity
                    or getattr(result, "analysis_input_signature", "")
                    != _analysis_input_signature(self.loaded)
                    or not self._edge_offset_matches(result)
                ):
                    result = segment_loaded_image(self.loaded, self.params, max_side=None)
                    if self.edge_offset_px:
                        result = apply_edge_offset(result, self.edge_offset_px)
            elif has_quick_base and has_quick_settings:
                assert self.quick_base_params is not None
                assert self.quick_settings is not None
                can_reuse = bool(
                    result is not None
                    and not result.is_preview
                    and result.mask.shape == self.loaded.shape
                    and getattr(result, "analysis_input_signature", "")
                    == _analysis_input_signature(self.loaded)
                    and result_matches_quick_plan(
                        result,
                        self.quick_base_params,
                        self.quick_settings,
                        self.loaded.shape,
                    )
                    and self._edge_offset_matches(result)
                )
                if not can_reuse:
                    result = segment_loaded_image_with_quick(
                        self.loaded,
                        self.quick_base_params,
                        self.quick_settings,
                        max_side=None,
                    )
                    if self.edge_offset_px:
                        result = apply_edge_offset(result, self.edge_offset_px)
            else:
                validated = self.params.validated(self.loaded.shape)
                if (
                    result is None
                    or result.is_preview
                    or getattr(result, "analysis_input_signature", "")
                    != _analysis_input_signature(self.loaded)
                    or result.params.validated(self.loaded.shape) != validated
                    or not self._edge_offset_matches(result)
                ):
                    result = segment_loaded_image(self.loaded, validated, max_side=None)
                    if self.edge_offset_px:
                        result = apply_edge_offset(result, self.edge_offset_px)
            result = _tag_result_analysis_input(result, self.loaded)
            export_quick = (
                quick_settings_to_dict(self.quick_settings)
                if not _uses_learned_engine(self.params) and self.quick_settings is not None
                and self.quick_settings.override_count
                else None
            )
            output = export_result(
                self.loaded,
                result,
                self.export_parent,
                artifacts=self.artifacts,
                outline_expand_px=self.outline_expand_px,
                merge_base_mode=self.merge_base_mode,
                merge_source_loaded=self.merge_source_loaded,
                preserve_merge_original_values=bool(
                    self.loaded.metadata.get(
                        "_preserve_merge_original_values",
                        False,
                    )
                ),
                quick_settings=export_quick,
                quick_base_params=(
                    self.quick_base_params if export_quick is not None else None
                ),
                allowed_output_root=self.allowed_output_root,
                bundle_mode=EXPORT_BUNDLE_PREFIXED_FILES,
            )
            if not isinstance(output, ExportReceipt):
                raise RuntimeError(T("The export returned no receipt for its files"))
            self.signals.result.emit(output)
        except Exception as exc:
            self.signals.error.emit(
                (
                    "export",
                    _safe_error_message(exc, self.loaded.path, self.export_parent),
                )
            )
        finally:
            self.signals.finished.emit(("export", 0))


class BatchWorker(QRunnable):
    def __init__(
        self,
        paths: list[Path],
        params: UniversalParams,
        output_root: Path | None,
        *,
        relative_paths: Mapping[Path, str | Path] | None = None,
        export_location_mode: str = EXPORT_LOCATION_FLAT,
        artifacts: Iterable[str] | None = None,
        outline_expand_px: int = 0,
        merge_base_mode: str = MERGE_BASE_MODE_ANALYSIS_INPUT,
        quick_base_params: UniversalParams | None = None,
        quick_settings: QuickSettings | None = None,
        crosstalk_pairs: Mapping[Path, Path] | None = None,
        crosstalk_correction: CrosstalkCorrection | None = None,
        edge_offset_px: int = 0,
    ) -> None:
        super().__init__()
        self.paths = [Path(path).expanduser().resolve() for path in paths]
        self.params = copy.deepcopy(params)
        self.quick_base_params = copy.deepcopy(quick_base_params)
        self.quick_settings = copy.deepcopy(quick_settings)
        if not _uses_learned_engine(self.params) and (self.quick_base_params is None) != (self.quick_settings is None):
            raise ValueError(T(
                "The recognition plan of the quick settings used for the batch export is "
                "incomplete.",
            ))
        self.edge_offset_px = _validated_edge_offset(edge_offset_px)
        self.crosstalk_pairs = {
            Path(target).expanduser().resolve(): Path(reference).expanduser().resolve()
            for target, reference in (crosstalk_pairs or {}).items()
        }
        self.crosstalk_correction = None if _uses_learned_engine(self.params) else copy.deepcopy(crosstalk_correction)
        self.location_mode = normalize_export_location_mode(export_location_mode)
        self.output_root = (
            None if output_root is None else Path(output_root).expanduser().resolve()
        )
        supplied_relative_paths = {
            Path(path).expanduser().resolve(): Path(relative)
            for path, relative in (relative_paths or {}).items()
        }
        self.relative_paths: dict[Path, Path] = {}
        self.output_parents: dict[Path, Path] = {}
        self.allowed_output_roots: dict[Path, Path | None] = {}
        for path in self.paths:
            relative = supplied_relative_paths.get(path, Path(path.name))
            self.output_parents[path] = resolve_export_parent(
                path,
                self.output_root,
                relative_path=relative,
                location_mode=self.location_mode,
            )
            self.relative_paths[path] = relative.parent / path.name
            self.allowed_output_roots[path] = (
                path.parent
                if self.location_mode == EXPORT_LOCATION_SOURCE
                else self.output_root
            )
        self.artifacts = frozenset(
            DEFAULT_EXPORT_ARTIFACTS if artifacts is None else artifacts
        )
        self.outline_expand_px = max(0, min(512, int(outline_expand_px)))
        self.merge_base_mode = normalize_merge_base_mode(merge_base_mode)
        self.cancelled = False
        self.signals = TaskSignals()

    def cancel(self) -> None:
        self.cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            self._run_batch()
        except Exception as exc:
            self.signals.result.emit(
                {
                    "exported": [],
                    "failures": [
                        {
                            "source": T("Batch task"),
                            "error": _safe_error_message(exc, *self.paths),
                        }
                    ],
                    "cancelled": self.cancelled,
                }
            )
        finally:
            self.signals.finished.emit(("batch", 0))

    def _run_batch(self) -> None:
        exported: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        total = len(self.paths)
        frozen_inputs: dict[Path, tuple[str, str, int, int, int, int]] = {}
        if self.crosstalk_correction is not None:
            seen_references: dict[Path, Path] = {}
            for index, path in enumerate(self.paths, start=1):
                if self.cancelled:
                    break
                source_label = self.relative_paths[path].as_posix()
                self.signals.progress.emit(("preflight", index - 1, total, source_label))
                reference_path = self.crosstalk_pairs.get(path.resolve())
                try:
                    if reference_path is None:
                        raise ValueError(T(
                            "No crosstalk reference image has been paired with the target image "
                            "yet: {0}",
                            path.name,
                        ))
                    if reference_path == path.resolve():
                        raise ValueError(T(
                            "A target image cannot be its own crosstalk reference image",
                        ))
                    previous_target = seen_references.get(reference_path)
                    if previous_target is not None:
                        raise ValueError(
                            T(
                                "A crosstalk reference image is paired with more than one target: "
                                "{0} is already used for {1}",
                                reference_path.name,
                                previous_target.name,
                            )
                        )
                    seen_references[reference_path] = path
                    source_probe = load_image(path)
                    reference_probe = load_image(reference_path)
                    if source_probe.shape != reference_probe.shape:
                        raise ValueError(
                            T(
                                "The target image and the crosstalk reference image differ in "
                                "size: {0}×{1} versus {2}×{3}",
                                source_probe.shape[1],
                                source_probe.shape[0],
                                reference_probe.shape[1],
                                reference_probe.shape[0],
                            )
                        )
                    if source_probe.metadata["sha256"] == reference_probe.metadata["sha256"]:
                        raise ValueError(T(
                            "The target image and the crosstalk reference image have identical "
                            "content; pairing refused",
                        ))
                    frozen_inputs[path.resolve()] = (
                        str(source_probe.metadata["sha256"]),
                        str(reference_probe.metadata["sha256"]),
                        int(source_probe.metadata["size"]),
                        int(reference_probe.metadata["size"]),
                        int(source_probe.metadata["mtime_ns"]),
                        int(reference_probe.metadata["mtime_ns"]),
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "source": source_label,
                            "error": _safe_error_message(
                                exc,
                                path,
                                *(
                                    (reference_path,)
                                    if reference_path is not None
                                    else ()
                                ),
                            ),
                        }
                    )
                self.signals.progress.emit(("preflight", index, total, source_label))
            if self.cancelled or failures:
                self.signals.result.emit(
                    {
                        "exported": [],
                        "failures": failures,
                        "cancelled": self.cancelled,
                        "preflight_failed": bool(failures),
                    }
                )
                return

        # Decode the next image on a helper thread while the current one is
        # analysed and exported (both are CPU-heavy but release the GIL in
        # OpenCV/NumPy), so a folder of large TIFF/PNG files no longer pays
        # the full decode time serially for every image.  Errors from the
        # prefetch surface inside that image's own try block below.
        prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fa-prefetch")
        prefetched: Future[LoadedImage] | None = None
        try:
            for index, path in enumerate(self.paths, start=1):
                if self.cancelled:
                    break
                source_label = self.relative_paths[path].as_posix()
                self.signals.progress.emit(("process", index - 1, total, source_label))
                self.signals.progress.emit(("item", str(path), "running", ""))
                current_future = prefetched
                prefetched = (
                    prefetch_pool.submit(load_image, self.paths[index])
                    if index < total
                    else None
                )
                try:
                    source_loaded = (
                        current_future.result()
                        if current_future is not None
                        else load_image(path)
                    )
                    loaded = source_loaded
                    if self.crosstalk_correction is not None and not _uses_learned_engine(self.params):
                        reference_path = self.crosstalk_pairs.get(path.resolve())
                        if reference_path is None:
                            raise ValueError(T(
                                "No crosstalk reference image has been paired with the target "
                                "image yet: {0}",
                                path.name,
                            ))
                        reference_loaded = load_image(reference_path)
                        frozen = frozen_inputs[path.resolve()]
                        current = (
                            str(source_loaded.metadata["sha256"]),
                            str(reference_loaded.metadata["sha256"]),
                            int(source_loaded.metadata["size"]),
                            int(reference_loaded.metadata["size"]),
                            int(source_loaded.metadata["mtime_ns"]),
                            int(reference_loaded.metadata["mtime_ns"]),
                        )
                        if current != frozen:
                            raise RuntimeError(T(
                                "The target image or the crosstalk reference image changed after "
                                "the batch pre-check",
                            ))
                        loaded = apply_crosstalk_correction(
                            source_loaded,
                            reference_loaded,
                            self.params,
                            self.crosstalk_correction,
                        )
                    params = self.params
                    if _uses_learned_engine(params):
                        result = segment_loaded_image(source_loaded, params, max_side=None)
                    elif self.quick_base_params is not None and self.quick_settings is not None:
                        result = segment_loaded_image_with_quick(
                            loaded,
                            self.quick_base_params,
                            self.quick_settings,
                            max_side=None,
                        )
                    else:
                        result = segment_loaded_image(loaded, params, max_side=None)
                    if self.edge_offset_px:
                        result = apply_edge_offset(result, self.edge_offset_px)
                    result = _tag_result_analysis_input(result, loaded)
                    export_quick = (
                        quick_settings_to_dict(self.quick_settings)
                        if not _uses_learned_engine(params) and self.quick_settings is not None
                        and self.quick_settings.override_count
                        else None
                    )
                    output = export_result(
                        loaded,
                        result,
                        self.output_parents[path],
                        artifacts=self.artifacts,
                        outline_expand_px=self.outline_expand_px,
                        merge_base_mode=self.merge_base_mode,
                        merge_source_loaded=source_loaded,
                        quick_settings=export_quick,
                        quick_base_params=(
                            self.quick_base_params if export_quick is not None else None
                        ),
                        allowed_output_root=self.allowed_output_roots[path],
                        bundle_mode=EXPORT_BUNDLE_PREFIXED_FILES,
                    )
                    if not isinstance(output, ExportReceipt):
                        raise RuntimeError(T("The batch export returned no receipt for its files"))
                    output_label = output.prefix
                    if (
                        self.location_mode != EXPORT_LOCATION_SOURCE
                        and self.output_root is not None
                    ):
                        output_label = output.label_path.relative_to(
                            self.output_root
                        ).as_posix()
                    file_labels = (
                        [path.name for path in output.files]
                        if self.location_mode == EXPORT_LOCATION_SOURCE
                        or self.output_root is None
                        else [
                            path.relative_to(self.output_root).as_posix()
                            for path in output.files
                        ]
                    )
                    exported.append(
                        {
                            "source": source_label,
                            # `output` remains as a compatible logical prefix label;
                            # `files` contains the actual safe relative paths.
                            "output": output_label,
                            "output_prefix": output_label,
                            "files": file_labels,
                        }
                    )
                    self.signals.progress.emit(
                        ("item", str(path), "exported", str(output.prefix))
                    )
                except Exception as exc:
                    message = _safe_error_message(exc, path)
                    failures.append({"source": source_label, "error": message})
                    self.signals.progress.emit(("item", str(path), "failed", message))
                self.signals.progress.emit(("process", index, total, source_label))
        finally:
            if prefetched is not None:
                prefetched.cancel()
            prefetch_pool.shutdown(wait=True, cancel_futures=True)
        self.signals.result.emit(
            {"exported": exported, "failures": failures, "cancelled": self.cancelled}
        )


class UniversalImageView(QGraphicsView):
    stroke_requested = Signal(object)
    stroke_cancelled = Signal()
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(10, 13, 18))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFrameShape(QFrame.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        # The side panels need 280 + 390 px; this keeps the whole workbench
        # inside the window's 1040 px minimum width without overlap.
        self.setMinimumSize(360, 320)
        # Hold Space to pan temporarily without leaving the add/erase tool.
        self._space_pan = False
        self._space_prev_mode = "pan"
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._contour_items: list[QGraphicsPathItem] = []
        self._dash_items: list[QGraphicsPathItem] = []
        self._cached_mask: np.ndarray | None = None
        self._cached_path = QPainterPath()
        self._display_mask_cache: (
            tuple[np.ndarray, tuple[int, int], np.ndarray] | None
        ) = None
        self._image_shape: tuple[int, int] | None = None
        self._edit_mode = "pan"
        self._brush_radius = 6
        self._stroke_points: list[tuple[float, float]] = []
        self._temporary_stroke: QGraphicsPathItem | None = None
        self._focus_item: QGraphicsRectItem | None = None
        self._dash_phase = 0.0
        self._ants_timer = QTimer(self)
        self._ants_timer.setInterval(80)
        self._ants_timer.timeout.connect(self._advance_ants)
        # The timer only runs while a selection is actually on screen; a static
        # 12.5 Hz repaint of the whole contour when idle wasted CPU and battery.
        # Floating tool strips / hints drawn over the canvas.  They are children
        # of the view itself rather than of the viewport: panning scrolls the
        # viewport's children along with the scene, which would drag a toolbar
        # off screen, whereas siblings of the viewport stay put.
        self._overlays: list[tuple[QWidget, str]] = []
        self.viewport().installEventFilter(self)

    # ---------- floating overlays ----------
    OVERLAY_MARGIN = 8

    def add_overlay(self, widget: QWidget, anchor: str = "top-left") -> None:
        """Float ``widget`` over the canvas without taking any layout space.

        ``anchor`` is one of ``top-left``, ``top-right``, ``bottom-left``,
        ``bottom-right``, ``top-center`` or ``bottom-center``.  The overlay is
        re-placed whenever the viewport changes size (window resize, scroll bar
        appearing) so it always follows the picture area.
        """

        widget.setParent(self)
        self._overlays.append((widget, anchor))
        # A label inside the overlay growing (zoom 9% -> 100%) must widen it.
        widget.installEventFilter(self)
        widget.show()
        widget.raise_()
        self._layout_overlays()

    def _layout_overlays(self) -> None:
        area = self.viewport().geometry()
        margin = self.OVERLAY_MARGIN
        available = max(40, area.width() - 2 * margin)
        for widget, anchor in self._overlays:
            if not widget.isVisibleTo(self):
                continue
            hint = widget.sizeHint()
            width = min(hint.width(), available)
            if widget.hasHeightForWidth():
                if isinstance(widget, QLabel) and widget.wordWrap():
                    # A wrapped label reports a narrow preferred width.  Give it
                    # the narrowest width that needs the fewest lines here.
                    fewest = widget.heightForWidth(available)
                    low, high = width, available
                    while low < high:
                        middle = (low + high) // 2
                        if widget.heightForWidth(middle) <= fewest:
                            high = middle
                        else:
                            low = middle + 1
                    width = low
                height = max(1, widget.heightForWidth(width))
            else:
                height = hint.height()
            if "right" in anchor:
                x = area.right() + 1 - margin - width
            elif "center" in anchor:
                x = area.left() + (area.width() - width) // 2
            else:
                x = area.left() + margin
            if anchor == "center":
                y = area.top() + max(margin, (area.height() - height) // 2)
            elif "bottom" in anchor:
                y = area.bottom() + 1 - margin - height
            else:
                y = area.top() + margin
            widget.setGeometry(x, y, width, height)
            widget.raise_()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched is self.viewport() and event.type() in (QEvent.Resize, QEvent.Show):
            self._layout_overlays()
        elif event.type() == QEvent.LayoutRequest and any(
            watched is widget for widget, _anchor in self._overlays
        ):
            self._layout_overlays()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._layout_overlays()

    def set_rgb_image(self, rgb: np.ndarray, *, fit: bool = False) -> None:
        contiguous = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width = contiguous.shape[:2]
        image = QImage(
            contiguous.data,
            width,
            height,
            contiguous.strides[0],
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        if self._pixmap_item is None:
            self._pixmap_item = self.scene().addPixmap(pixmap)
            self._pixmap_item.setZValue(0)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._image_shape = (height, width)
        self.scene().setSceneRect(0, 0, width, height)
        if fit:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self.zoom_changed.emit(self.current_zoom())

    def display_mask_for(
        self,
        mask: np.ndarray | None,
        target_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Return ``mask`` resampled to the display grid, cached per mask.

        The window re-renders on every view toggle and parameter edit.  The
        result mask object is stable between edits (edits mutate it in place
        and then call ``invalidate_contour_cache``), so both the resampled
        display mask and the marching-ants path built from it can be reused
        until the mask really changes.
        """

        if mask is None:
            return None
        cached = self._display_mask_cache
        if (
            cached is not None
            and cached[0] is mask
            and cached[1] == tuple(target_shape)
        ):
            return cached[2]
        display_mask = resized_mask(mask, tuple(target_shape))
        self._display_mask_cache = (mask, tuple(target_shape), display_mask)
        return display_mask

    def set_contour(self, mask: np.ndarray | None, visible: bool = True, *, positive_color: QColor | None = None) -> None:
        for item in self._contour_items:
            self.scene().removeItem(item)
        self._contour_items.clear()
        self._dash_items.clear()
        if mask is None or not visible:
            self._ants_timer.stop()
            return
        if self._cached_mask is mask:
            painter_path = self._cached_path
        elif not bool(mask.any()):
            self._ants_timer.stop()
            return
        else:
            painter_path = QPainterPath()
            contours, scale_x, scale_y, _ = _bounded_display_contours(mask)
            for contour in contours:
                points = contour.reshape(-1, 2)
                if len(points) == 0:
                    continue
                painter_path.moveTo(
                    float(points[0, 0]) * scale_x,
                    float(points[0, 1]) * scale_y,
                )
                for point in points[1:]:
                    painter_path.lineTo(
                        float(point[0]) * scale_x,
                        float(point[1]) * scale_y,
                    )
                if len(points) >= 3:
                    painter_path.closeSubpath()
            self._cached_mask = mask
            self._cached_path = painter_path
        if painter_path.isEmpty():
            self._ants_timer.stop()
            return
        # Two equal-width, complementary dashed strokes form true black/white
        # marching ants.  The former wide solid black under-stroke could look
        # like a heavy outline rather than alternating selection marks.
        black_ants = QGraphicsPathItem(painter_path)
        black_pen = QPen(QColor(0, 0, 0), 2.2)
        black_pen.setCosmetic(True)
        black_pen.setCapStyle(Qt.FlatCap)
        black_pen.setDashPattern([4.0, 4.0])
        black_ants.setPen(black_pen)
        black_ants.setZValue(10)
        self.scene().addItem(black_ants)

        white_ants = QGraphicsPathItem(painter_path)
        white_pen = QPen(positive_color or QColor(255, 255, 255), 2.2)
        white_pen.setCosmetic(True)
        white_pen.setCapStyle(Qt.FlatCap)
        white_pen.setDashPattern([4.0, 4.0])
        white_pen.setDashOffset(4.0)
        white_ants.setPen(white_pen)
        white_ants.setZValue(11)
        self.scene().addItem(white_ants)
        self._contour_items.extend((black_ants, white_ants))
        self._dash_items.extend((black_ants, white_ants))
        # Every tick restrokes the whole dashed path.  A very fragmented
        # selection (tens of thousands of path elements) is animated a little
        # more slowly so the repaint does not monopolise the GUI thread.
        elements = painter_path.elementCount()
        self._ants_timer.setInterval(
            80 if elements <= 6_000 else 120 if elements <= 20_000 else 200
        )
        if not self._ants_timer.isActive():
            self._ants_timer.start()

    def invalidate_contour_cache(self) -> None:
        self._cached_mask = None
        self._cached_path = QPainterPath()
        self._display_mask_cache = None

    def set_edit_context(self, mode: str, brush_radius: int) -> None:
        mode = mode if mode in {"pan", "add", "erase"} else "pan"
        if mode != self._edit_mode:
            self.cancel_stroke()
        self._edit_mode = mode
        self._brush_radius = max(1, int(brush_radius))
        if mode == "pan":
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)

    def cancel_stroke(self) -> None:
        """Discard any unfinished manual stroke when a tool or image changes."""

        if self._temporary_stroke is not None:
            self.scene().removeItem(self._temporary_stroke)
            self._temporary_stroke = None
        self._stroke_points.clear()

    def reset_zoom(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self.zoom_changed.emit(self.current_zoom())

    def current_zoom(self) -> float:
        return float(self.transform().m11())

    def _apply_zoom(self, factor: float) -> None:
        if self._pixmap_item is None:
            return
        current = self.transform().m11()
        if (factor > 1 and current < 40) or (factor < 1 and current > 0.03):
            self.scale(factor, factor)
            self.zoom_changed.emit(self.current_zoom())

    def zoom_in(self) -> None:
        self._apply_zoom(1.18)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / 1.18)

    def focus_bbox(self, x: float, y: float, width: float, height: float) -> None:
        """Center one object and mark its bounding box for manual review."""

        self.clear_focus()
        if self._pixmap_item is None or self._image_shape is None:
            return
        image_height, image_width = self._image_shape
        x0 = float(np.clip(x, 0.0, max(0.0, image_width - 1.0)))
        y0 = float(np.clip(y, 0.0, max(0.0, image_height - 1.0)))
        box_width = float(np.clip(width, 1.0, max(1.0, image_width - x0)))
        box_height = float(np.clip(height, 1.0, max(1.0, image_height - y0)))
        rect = QRectF(x0, y0, box_width, box_height)
        focus_pen = QPen(QColor(255, 205, 55), 2.4)
        focus_pen.setCosmetic(True)
        item = self.scene().addRect(rect, focus_pen, QBrush(Qt.NoBrush))
        item.setZValue(20)
        self._focus_item = item
        margin = max(18.0, max(box_width, box_height) * 0.7)
        focus_rect = rect.adjusted(-margin, -margin, margin, margin).intersected(self.sceneRect())
        if not focus_rect.isEmpty():
            self.fitInView(focus_rect, Qt.KeepAspectRatio)
            self.zoom_changed.emit(self.current_zoom())

    def clear_focus(self) -> None:
        if self._focus_item is not None:
            self.scene().removeItem(self._focus_item)
            self._focus_item = None

    def _inside_image(self, point: QPointF) -> bool:
        if self._image_shape is None:
            return False
        height, width = self._image_shape
        return 0 <= point.x() < width and 0 <= point.y() < height

    def _update_temporary_stroke(self) -> None:
        if not self._stroke_points:
            return
        if self._temporary_stroke is not None:
            self.scene().removeItem(self._temporary_stroke)
        path = QPainterPath()
        path.moveTo(*self._stroke_points[0])
        for x, y in self._stroke_points[1:]:
            path.lineTo(x, y)
        item = QGraphicsPathItem(path)
        color = QColor(65, 235, 165, 155) if self._edit_mode == "add" else QColor(255, 85, 85, 170)
        pen = QPen(color, self._brush_radius * 2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        item.setPen(pen)
        item.setZValue(20)
        self.scene().addItem(item)
        self._temporary_stroke = item

    def _finish_stroke(self) -> None:
        if self._temporary_stroke is not None:
            self.scene().removeItem(self._temporary_stroke)
            self._temporary_stroke = None
        # Emit even for an empty attempt.  The selected add/erase tool remains
        # active, so the user can make several corrections without reselecting it.
        self.stroke_requested.emit(
            (list(self._stroke_points), self._edit_mode == "add", self._brush_radius)
        )
        self._stroke_points.clear()

    def _advance_ants(self) -> None:
        window = self.window()
        if not self.isVisible() or (window is not None and window.isMinimized()):
            # Nothing on screen (window minimised or hidden): skip the
            # restroke but keep the timer so the ants resume when shown.
            return
        self._dash_phase = (self._dash_phase + 0.9) % 8.0
        for index, item in enumerate(self._dash_items):
            pen = item.pen()
            pen.setDashOffset(self._dash_phase + (4.0 if index % 2 else 0.0))
            item.setPen(pen)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._pixmap_item is None:
            return
        self._apply_zoom(1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18)

    def _pan_active(self) -> bool:
        return self._edit_mode == "pan" or self._space_pan

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._pan_active() or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        point = self.mapToScene(event.position().toPoint())
        if not self._inside_image(point):
            return
        self._stroke_points = [(point.x(), point.y())]
        self._update_temporary_stroke()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._pan_active() or not self._stroke_points or not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        point = self.mapToScene(event.position().toPoint())
        if self._inside_image(point):
            previous = self._stroke_points[-1]
            if (point.x() - previous[0]) ** 2 + (point.y() - previous[1]) ** 2 >= 0.5:
                self._stroke_points.append((point.x(), point.y()))
                self._update_temporary_stroke()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            not self._pan_active()
            and self._edit_mode != "pan"
            and event.button() == Qt.LeftButton
        ):
            self._finish_stroke()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() == Qt.Key_Escape and self._edit_mode != "pan":
            self.cancel_stroke()
            self.stroke_cancelled.emit()
            event.accept()
            return
        if (
            event.key() == Qt.Key_Space
            and not event.isAutoRepeat()
            and self._edit_mode != "pan"
            and not self._space_pan
        ):
            # Temporary hand-pan while a brush tool stays selected.
            self._space_pan = True
            self._space_prev_mode = self._edit_mode
            self.cancel_stroke()
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            event.key() == Qt.Key_Space
            and not event.isAutoRepeat()
            and self._space_pan
        ):
            self._space_pan = False
            if self._space_prev_mode in {"add", "erase"}:
                self.setDragMode(QGraphicsView.NoDrag)
                self.viewport().setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        had_unfinished_stroke = bool(self._stroke_points or self._temporary_stroke is not None)
        if had_unfinished_stroke:
            self.cancel_stroke()
        if self._space_pan:
            # Key-release is not guaranteed after focus moves to another
            # window.  Restore the selected brush instead of leaving a hidden
            # temporary-pan state behind.
            self._space_pan = False
            if self._edit_mode in {"add", "erase"}:
                self.setDragMode(QGraphicsView.NoDrag)
                self.viewport().setCursor(Qt.CrossCursor)
        super().focusOutEvent(event)


class UniversalFluoroWindow(QMainWindow):
    """Generic 2-D fluorescence-positive segmentation workbench."""

    def __init__(
        self,
        initial_path: Path | None = None,
        *,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        configure_opencv_threads()
        self.setWindowTitle(
            T("FluoroAnts Universal {0} · positive-signal recognition", FLUOROANTS_RELEASE_VERSION)
        )
        # Keep the complete workbench usable inside the effective client area
        # of a 1366x768 display (after the Windows taskbar and title bar).
        self.setMinimumSize(1040, 620)
        self.settings = (
            settings
            if settings is not None
            else QSettings("FluoroAnts", "FluoroAnts Universal")
        )

        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(2)
        self.paths: list[Path] = []
        self._queue_display_text: dict[Path, str] = {}
        self._queue_relative_text: dict[Path, str] = {}
        self._queue_import_roots: dict[Path, Path] = {}
        self._selection_resume_bundles: dict[Path, SelectionReviewBundle] = {}
        # Queue presentation.  ``self.paths`` is the canonical order (sorted by
        # the chosen criterion); ``self._visible_paths`` is the filtered subset
        # actually shown, and it is that list — in that order — which a batch
        # processes.  Queue rows index ``self._visible_paths``.
        self._visible_paths: list[Path] = []
        self._queue_status: dict[Path, tuple[str, str, float]] = {}
        self._queue_file_stats: dict[Path, tuple[int, int]] = {}
        self._queue_added_order: dict[Path, int] = {}
        self._queue_added_counter = 0
        self._queue_export_target: dict[Path, tuple[str, Path | None]] = {}
        self._queue_filter_text = ""
        self._queue_sort_key = "path"
        self._queue_sort_desc = False
        self._queue_view_mode = "tree"
        self._queue_filter_timer = QTimer(self)
        self._queue_filter_timer.setSingleShot(True)
        self._queue_filter_timer.setInterval(120)
        self._queue_filter_timer.timeout.connect(self._apply_queue_filter_now)
        # Edge distance: a post-recognition boundary offset (source pixels).
        # It is neither a recipe parameter nor a manual correction, so it lives
        # outside the preset/quick/professional state and the undo history.
        self._edge_offset_value = 0
        self._edge_offset_ui_state: tuple[bool, int] = (False, 0)
        self._edge_offset_serial: int | None = None
        self._pending_edge_rebuild = False
        self._resume_base_result: UniversalResult | None = None
        self._loading_path: Path | None = None
        self.source_loaded: LoadedImage | None = None
        self.loaded: LoadedImage | None = None
        self.result: UniversalResult | None = None
        self.base_display: np.ndarray | None = None
        self.analysis_display: np.ndarray | None = None
        self.crosstalk_reference_display: np.ndarray | None = None
        self.display_scale = 1.0
        self._crosstalk_reference_paths: dict[Path, Path] = {}
        self._crosstalk_reference_loaded: LoadedImage | None = None
        self._crosstalk_analysis_key: tuple[object, ...] | None = None
        self._crosstalk_serial = 0
        self._crosstalk_busy = False
        self._crosstalk_error: str | None = None
        self._pending_crosstalk_full: bool | None = None
        self._accepted_crosstalk_enabled = False
        self._accepted_crosstalk_settings = CrosstalkCorrection()
        self.load_token = 0
        self.segment_serial = 0
        # QRunnable has no safe force-stop.  Keep at most one segmentation
        # worker in the pool and coalesce rapid parameter changes into one
        # latest request instead of building an unbounded stale-work queue.
        self._active_segment_serial: int | None = None
        self._pending_segment_full: bool | None = None
        self.manual_serial: int | None = None
        self._suppress_controls = False
        self._export_prompt_open = False
        self._active_export = False
        self._last_export_artifacts = _COMMON_EXPORT_ARTIFACTS
        self._last_export_outline_expand_px = 0
        self._last_export_merge_base_mode = MERGE_BASE_MODE_ANALYSIS_INPUT
        self._last_export_location_mode = EXPORT_LOCATION_MIRROR
        self._active_export_artifacts: frozenset[str] | None = None
        self._active_export_location_mode: str | None = None
        self._batch_export_location_mode: str | None = None
        self._batch_export_root: Path | None = None
        self._active_export_path: Path | None = None
        self._active_export_root: Path | None = None
        self._load_busy = False
        self._segment_busy = False
        self._manual_dirty = False
        self._manual_busy = False
        self._batch_worker: BatchWorker | None = None
        self._batch_dialog: QProgressDialog | None = None
        self._export_dialog: QProgressDialog | None = None
        self._undo_stack: list[_ManualEdit] = []
        self._redo_stack: list[_ManualEdit] = []
        self._manual_rebuild_rollback: _ManualRollback | None = None
        self.user_presets: dict[str, dict[str, object]] = {}
        self._recognition_engine = "classic"
        self._quick_base_params = copy.deepcopy(BUILTIN_PRESETS["General balanced"].params)
        self._quick_settings = QuickSettings()
        self._quick_plan_error: str | None = None
        self._parameter_source = "preset"
        self._applied_preset_key: tuple[str, str] = ("builtin", "General balanced")

        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.setInterval(260)
        self.auto_timer.timeout.connect(self._run_auto_preview)

        self._build_actions()
        self._build_ui()
        self._apply_style()
        self._load_user_presets()
        self._refresh_preset_choices(selected=("builtin", "General balanced"))
        self._set_quick_controls(self._quick_settings)
        self._set_controls_from_params(BUILTIN_PRESETS["General balanced"].params)
        self._update_parameter_mode_label()
        self._update_action_state()
        self._restore_ui_state()
        self.setAcceptDrops(True)
        self.image_view.setAcceptDrops(False)   # drops are handled by the window
        QTimer.singleShot(0, self.image_view.setFocus)

        if initial_path is not None:
            QTimer.singleShot(0, lambda value=Path(initial_path): self._open_initial(value))

    # ---------- window state persistence ----------
    def _apply_default_geometry(self) -> None:
        """Size the window to fit the screen on first run (never off-screen)."""

        screen = self.screen() or QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        width, height = 1540, 930
        if available is not None:
            width = min(width, available.width() - 40)
            height = min(height, available.height() - 40)
        self.resize(max(self.minimumWidth(), width), max(self.minimumHeight(), height))
        if available is not None:
            frame = self.frameGeometry()
            frame.moveCenter(available.center())
            self.move(frame.topLeft())

    def _restore_ui_state(self) -> None:
        """Restore geometry, toolbars, splitter and remembered preferences."""

        # Automatic preview is a session-only switch that always starts
        # enabled: a remembered "off" would make the next launch look frozen,
        # because changed settings invalidate the Mask without recomputing it.
        previous_auto_signals = self.auto_check.blockSignals(True)
        try:
            self.auto_check.setChecked(True)
        finally:
            self.auto_check.blockSignals(previous_auto_signals)

        geometry = self.settings.value("window/geometry")
        restored = False
        if isinstance(geometry, (QByteArray, bytes, bytearray)):
            restored = bool(self.restoreGeometry(QByteArray(geometry)))
        state = self.settings.value("window/state")
        if isinstance(state, (QByteArray, bytes, bytearray)):
            self.restoreState(QByteArray(state))
        # Settings created on another monitor can otherwise restore the whole
        # window beyond the current desktop.
        # Re-centre and shrink such geometry while respecting the responsive
        # minimum size.
        screen = self.screen() or QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        if restored and available is not None:
            frame = self.frameGeometry()
            if not available.contains(frame):
                target_width = min(
                    max(self.minimumWidth(), frame.width()),
                    max(self.minimumWidth(), available.width() - 20),
                )
                target_height = min(
                    max(self.minimumHeight(), frame.height()),
                    max(self.minimumHeight(), available.height() - 20),
                )
                self.resize(target_width, target_height)
                centred = self.frameGeometry()
                centred.moveCenter(available.center())
                self.move(centred.topLeft())
        if not restored:
            self._apply_default_geometry()
        sizes = self.settings.value("window/splitter")
        if sizes and hasattr(self, "_main_splitter"):
            try:
                self._main_splitter.setSizes([int(v) for v in sizes])
            except (TypeError, ValueError):
                pass
        brush = self.settings.value("tools/brush_radius")
        if brush is not None:
            try:
                self.brush_radius_spin.setValue(int(brush))
            except (TypeError, ValueError):
                pass
        self._restore_panel_state()
        self._set_combo_data(
            self.recognition_engine_combo,
            str(self.settings.value("analysis/recognition_engine", "classic")),
        )
        self.ai_review_check.setChecked(
            self._settings_bool(self.settings.value("analysis/show_ai_review"), True)
        )

    @staticmethod
    def _settings_bool(value: object, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _settings_int(value: object, default: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    def _restore_panel_state(self) -> None:
        """Queue filter/sort, collapsible sections and the edge distance."""

        settings = self.settings
        previous = self._suppress_controls
        self._suppress_controls = True
        try:
            self.queue_filter_toggle.setChecked(
                self._settings_bool(settings.value("queue/filter_visible"), True)
            )
            sort_key = str(settings.value("queue/sort_key") or "path")
            if sort_key not in {key for key, _label in _QUEUE_SORT_OPTIONS}:
                sort_key = "path"
            self._set_combo_data(self.queue_sort_combo, sort_key)
            self.queue_sort_direction.setChecked(
                self._settings_bool(settings.value("queue/sort_desc"), False)
            )
            self.queue_view_toggle.setChecked(
                str(settings.value("queue/view_mode") or "tree") == "list"
            )
            self._queue_sort_key = sort_key
            self._queue_sort_desc = self.queue_sort_direction.isChecked()
            self._queue_view_mode = "list" if self.queue_view_toggle.isChecked() else "tree"
            self._refresh_queue_control_text()
            self.view_group_toggle.setChecked(
                self._settings_bool(settings.value("left/view_group_expanded"), False)
            )
            self.stats_details_toggle.setChecked(
                self._settings_bool(settings.value("right/stats_expanded"), False)
            )
            self.plan_details_toggle.setChecked(
                self._settings_bool(settings.value("right/plan_expanded"), False)
            )
            enabled = self._settings_bool(settings.value("edge/enabled"), False)
            offset = self._settings_int(settings.value("edge/offset_px"), 0)
            offset = max(-MAX_EDGE_OFFSET_PX, min(MAX_EDGE_OFFSET_PX, offset))
            self.edge_offset_check.setChecked(enabled)
            self.edge_offset_spin.setValue(offset)
            self.edge_offset_spin.setEnabled(enabled)
            self._edge_offset_ui_state = (enabled, offset)
            self._edge_offset_value = offset if enabled else 0
            self._update_edge_offset_summary()
        finally:
            self._suppress_controls = previous
        self._fit_left_controls()

    def _save_ui_state(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        if hasattr(self, "_main_splitter"):
            self.settings.setValue("window/splitter", self._main_splitter.sizes())
        self.settings.setValue("tools/brush_radius", int(self.brush_radius_spin.value()))
        self.settings.setValue("queue/filter_visible", bool(self.queue_filter_toggle.isChecked()))
        self.settings.setValue("queue/sort_key", str(self._queue_sort_key))
        self.settings.setValue("queue/sort_desc", bool(self._queue_sort_desc))
        self.settings.setValue("queue/view_mode", str(self._queue_view_mode))
        self.settings.setValue("left/view_group_expanded", bool(self.view_group_toggle.isChecked()))
        self.settings.setValue("right/stats_expanded", bool(self.stats_details_toggle.isChecked()))
        self.settings.setValue("right/plan_expanded", bool(self.plan_details_toggle.isChecked()))
        self.settings.setValue("edge/enabled", bool(self.edge_offset_check.isChecked()))
        self.settings.setValue("edge/offset_px", int(self.edge_offset_spin.value()))
        self.settings.setValue("analysis/recognition_engine", self._recognition_engine)
        self.settings.setValue("analysis/show_ai_review", self.ai_review_check.isChecked())

    # ---------- construction ----------
    def _build_actions(self) -> None:
        self.open_files_action = QAction(T("Open images"), self)
        self.open_files_action.setShortcut(QKeySequence.Open)
        self.open_files_action.triggered.connect(self._choose_files)

        self.open_folder_action = QAction(T("Open folder"), self)
        self.open_folder_action.setShortcut("Ctrl+Shift+O")
        self.open_folder_action.triggered.connect(self._choose_folder)

        self.continue_selection_action = QAction(T("Choose base image + mask manually"), self)
        self.continue_selection_action.setShortcut("Ctrl+Alt+O")
        self.continue_selection_action.setToolTip(
            T(
                "Choose a clean base image and its binary mask one after the other, without "
                "relying on file names",
            )
        )
        self.continue_selection_action.triggered.connect(
            self._choose_selection_review_to_continue
        )

        self.remove_action = QAction(T("Remove selected"), self)
        self.remove_action.setShortcut(QKeySequence.Delete)
        self.remove_action.triggered.connect(self._remove_selected)

        self.export_action = QAction(T("Export current"), self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self._export_current)

        self.batch_action = QAction(T("Batch export"), self)
        self.batch_action.setShortcut("Ctrl+Shift+E")
        self.batch_action.triggered.connect(self._export_batch)

        self.import_recipe_action = QAction(T("Import recipe"), self)
        self.import_recipe_action.triggered.connect(self._import_recipe)

        self.export_recipe_action = QAction(T("Export recipe"), self)
        self.export_recipe_action.triggered.connect(self._export_recipe)

        self.undo_action = QAction(T("Undo"), self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self._undo_manual)

        self.redo_action = QAction(T("Redo"), self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self._redo_manual)

        self.fit_action = QAction(T("Fit to window"), self)
        self.fit_action.setShortcut("0")
        self.fit_action.triggered.connect(lambda: self.image_view.reset_zoom())

        self.full_action = QAction(T("Full-resolution analysis"), self)
        self.full_action.setShortcut("F5")
        self.full_action.triggered.connect(lambda: self._start_segmentation(full=True))

        self.zoom_in_action = QAction(T("Zoom in"), self)
        self.zoom_in_action.setShortcuts([QKeySequence.ZoomIn, QKeySequence("Ctrl+=")])
        self.zoom_in_action.triggered.connect(lambda: self.image_view.zoom_in())

        self.zoom_out_action = QAction(T("Zoom out"), self)
        self.zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        self.zoom_out_action.triggered.connect(lambda: self.image_view.zoom_out())

        self.brush_smaller_action = QAction(T("Smaller brush"), self)
        self.brush_smaller_action.setShortcut("[")
        self.brush_smaller_action.triggered.connect(lambda: self._adjust_brush(-2))

        self.brush_larger_action = QAction(T("Larger brush"), self)
        self.brush_larger_action.setShortcut("]")
        self.brush_larger_action.triggered.connect(lambda: self._adjust_brush(2))

        self.about_action = QAction(T("About"), self)
        self.about_action.triggered.connect(self._show_about)

        # Short labels for the floating canvas strip (QToolButton shows the
        # action's iconText); menus keep the full text.
        self.undo_action.setIconText(T("Undo"))
        self.redo_action.setIconText(T("Redo"))
        self.fit_action.setIconText(T("Fit"))
        self.zoom_in_action.setIconText("+")
        self.zoom_out_action.setIconText("−")
        self.undo_action.setToolTip(T("Undo the last manual edit (Ctrl+Z)"))
        self.redo_action.setToolTip(T("Redo the manual edit (Ctrl+Y)"))
        self.fit_action.setToolTip(T("Fit the whole image to the window (0)"))
        self.zoom_in_action.setToolTip(T("Zoom in (Ctrl++)"))
        self.zoom_out_action.setToolTip(T("Zoom out (Ctrl+-)"))

        for action in (
            self.zoom_in_action,
            self.zoom_out_action,
            self.brush_smaller_action,
            self.brush_larger_action,
        ):
            self.addAction(action)

        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.pan_action = QAction(T("Pan"), self)
        self.pan_action.setShortcut("V")
        self.add_action = QAction(T("Add positive"), self)
        self.add_action.setShortcut("B")
        self.erase_action = QAction(T("Erase positive"), self)
        self.erase_action.setShortcut("E")
        for action in (self.pan_action, self.add_action, self.erase_action):
            action.setCheckable(True)
        self.pan_action.setToolTip(T("Drag the view; also leaves the current manual brush (V)"))
        self.add_action.setToolTip(T(
            "Add positive continuously; stays selected, click again or press Esc to leave (B)",
        ))
        self.erase_action.setToolTip(
            T("Erase positive continuously; stays selected, click again or press Esc to leave (E)")
        )
        self.pan_action.setIconText(T("Pan"))
        self.add_action.setIconText(T("Add"))
        self.erase_action.setIconText(T("Erase"))
        for action in (self.pan_action, self.add_action, self.erase_action):
            action.setStatusTip(action.toolTip())
        for action, mode in (
            (self.pan_action, "pan"),
            (self.add_action, "add"),
            (self.erase_action, "erase"),
        ):
            self.mode_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, value=mode: self._tool_action_triggered(value)
            )
        self.pan_action.setChecked(True)

    def _build_ui(self) -> None:
        # One compact row of file/export commands.  The former second toolbar
        # (review and edit tools) now floats over the canvas, so the top of the
        # window no longer spends two rows of height on buttons.
        toolbar = QToolBar(T("File and export"), self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(toolbar)
        for action in (
            self.open_files_action,
            self.open_folder_action,
            self.continue_selection_action,
            self.remove_action,
        ):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.export_action)
        toolbar.addAction(self.batch_action)
        toolbar.addSeparator()
        toolbar.addAction(self.import_recipe_action)
        toolbar.addAction(self.export_recipe_action)
        self.main_toolbar = toolbar

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setObjectName("mainSplitter")
        splitter.setChildrenCollapsible(False)
        self._main_splitter = splitter
        splitter.addWidget(self._build_left_panel())

        center = QWidget(self)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self.image_view = UniversalImageView(center)
        self.image_view.stroke_requested.connect(self._apply_stroke)
        self.image_view.stroke_cancelled.connect(lambda: self._set_edit_mode("pan"))
        self.image_view.zoom_changed.connect(self._on_zoom_changed)
        center_layout.addWidget(self.image_view, 1)
        self._build_canvas_overlay()
        splitter.addWidget(center)
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([330, 760, 450])
        self.setCentralWidget(splitter)

        self._build_menu_bar()

        status = QStatusBar(self)
        self.setStatusBar(status)
        self.status_label = QLabel(T("Open a 2-D greyscale or colour image to start"))
        self.statusBar().addWidget(self.status_label, 1)
        self.algorithm_label = QLabel(T("Algorithm {0}", UNIVERSAL_ALGORITHM_VERSION))
        self.algorithm_label.setObjectName("mutedText")
        self.statusBar().addPermanentWidget(self.algorithm_label)

        # A field moves below its label when a row is too narrow for both.
        for form in self.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.WrapLongRows)

    def _build_canvas_overlay(self) -> None:
        """Tool strip and hint that float over the picture instead of stacking
        above/below it, so they follow the canvas and cost no window height."""

        strip = CanvasToolStrip(self.image_view)
        strip.setObjectName("canvasToolStrip")
        strip_layout = strip.flow

        def tool_button(action: QAction) -> QToolButton:
            button = QToolButton(strip)
            button.setObjectName("canvasTool")
            button.setDefaultAction(action)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.setAutoRaise(True)
            button.setFocusPolicy(Qt.NoFocus)
            strip_layout.addWidget(button)
            return button

        def separator() -> None:
            line = QFrame(strip)
            line.setObjectName("canvasToolSeparator")
            line.setFrameShape(QFrame.VLine)
            line.setFixedSize(7, 22)
            strip_layout.addWidget(line)

        self.pan_tool_button = tool_button(self.pan_action)
        self.add_tool_button = tool_button(self.add_action)
        self.erase_tool_button = tool_button(self.erase_action)
        separator()
        brush_label = QLabel(T("Brush"), strip)
        brush_label.setObjectName("canvasStripLabel")
        strip_layout.addWidget(brush_label)
        self.brush_radius_spin = self._spin(1, 512, 6, suffix=" px")
        self.brush_radius_spin.setObjectName("canvasSpin")
        self.brush_radius_spin.setParent(strip)
        self.brush_radius_spin.setFixedWidth(70)
        self.brush_radius_spin.setToolTip(T("Brush radius (source pixels); adjust with [ and ]"))
        self.brush_radius_spin.setFocusPolicy(Qt.ClickFocus)
        self.brush_radius_spin.valueChanged.connect(
            lambda: self._set_edit_mode(self._current_edit_mode())
        )
        strip_layout.addWidget(self.brush_radius_spin)
        separator()
        self.undo_tool_button = tool_button(self.undo_action)
        self.redo_tool_button = tool_button(self.redo_action)
        separator()
        self.fit_tool_button = tool_button(self.fit_action)
        self.zoom_out_tool_button = tool_button(self.zoom_out_action)
        self.zoom_in_tool_button = tool_button(self.zoom_in_action)
        self.zoom_label = QLabel("—", strip)
        self.zoom_label.setObjectName("canvasStripLabel")
        self.zoom_label.setMinimumWidth(40)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setToolTip(T("Current zoom"))
        strip_layout.addWidget(self.zoom_label)
        self.canvas_tool_strip = strip
        self.image_view.add_overlay(strip, "top-left")

        self.canvas_hint = QLabel(
            T(
                "Scroll to zoom · drag to pan (hold Space while a brush is active) · manual edits "
                "need a full-resolution result",
            ),
            self.image_view,
        )
        self.canvas_hint.setObjectName("canvasHint")
        self.canvas_hint.setWordWrap(True)
        self.canvas_hint.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.image_view.add_overlay(self.canvas_hint, "bottom-left")

        # Shown while the queue is empty: what to do first.
        empty = QFrame(self.image_view)
        empty.setObjectName("emptyState")
        empty.setFixedWidth(440)
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(28, 24, 28, 22)
        empty_layout.setSpacing(10)
        empty_title = QLabel(T("Open images to begin"), empty)
        empty_title.setObjectName("emptyTitle")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_text = QLabel(
            T(
                "Drop image files or a folder here, or use the buttons below.\n"
                "2-D greyscale or colour images: TIFF, PNG, JPEG, BMP, WebP."
            ),
            empty,
        )
        empty_text.setObjectName("mutedText")
        empty_text.setAlignment(Qt.AlignCenter)
        empty_text.setWordWrap(True)
        empty_buttons = QHBoxLayout()
        empty_buttons.setSpacing(8)
        open_files_button = QPushButton(T("Open images…"), empty)
        open_files_button.setObjectName("primaryButton")
        open_files_button.clicked.connect(self._choose_files)
        open_folder_button = QPushButton(T("Open folder…"), empty)
        open_folder_button.clicked.connect(self._choose_folder)
        empty_buttons.addStretch(1)
        empty_buttons.addWidget(open_files_button)
        empty_buttons.addWidget(open_folder_button)
        empty_buttons.addStretch(1)
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_text)
        empty_layout.addLayout(empty_buttons)
        self.empty_state = empty
        self.image_view.add_overlay(empty, "center")

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._fit_left_controls()

    def _build_menu_bar(self) -> None:
        """A conventional menu bar mirroring the toolbars for discoverability."""

        bar = self.menuBar()
        file_menu = bar.addMenu(T("File"))
        for action in (
            self.open_files_action,
            self.open_folder_action,
            self.continue_selection_action,
            self.remove_action,
        ):
            file_menu.addAction(action)
        file_menu.addSeparator()
        for action in (self.import_recipe_action, self.export_recipe_action):
            file_menu.addAction(action)

        export_menu = bar.addMenu(T("Export"))
        for action in (self.export_action, self.batch_action):
            export_menu.addAction(action)

        edit_menu = bar.addMenu(T("Edit"))
        for action in (self.undo_action, self.redo_action):
            edit_menu.addAction(action)
        edit_menu.addSeparator()
        for action in (self.pan_action, self.add_action, self.erase_action):
            edit_menu.addAction(action)
        edit_menu.addSeparator()
        for action in (self.brush_smaller_action, self.brush_larger_action):
            edit_menu.addAction(action)

        view_menu = bar.addMenu(T("View"))
        for action in (
            self.fit_action,
            self.zoom_in_action,
            self.zoom_out_action,
            self.full_action,
        ):
            view_menu.addAction(action)
        view_menu.addSeparator()
        for text, button in (
            (T("Queue filter / sort bar"), self.queue_filter_toggle),
            (T("View and analysis channel"), self.view_group_toggle),
            (T("Detailed statistics"), self.stats_details_toggle),
            (T("Parameter status and execution plan"), self.plan_details_toggle),
        ):
            action = QAction(text, self)
            action.setCheckable(True)
            action.setChecked(button.isChecked())
            action.toggled.connect(button.setChecked)
            button.toggled.connect(action.setChecked)
            view_menu.addAction(action)

        self._build_language_menu(bar)
        help_menu = bar.addMenu(T("Help"))
        help_menu.addAction(self.about_action)

    def _on_zoom_changed(self, factor: float) -> None:
        self.zoom_label.setText(f"{factor * 100:.0f}%")

    def _adjust_brush(self, delta: int) -> None:
        self.brush_radius_spin.setValue(
            int(np.clip(self.brush_radius_spin.value() + delta, 1, 512))
        )

    def _show_about(self) -> None:
        dialog = AboutDialog(self)
        dialog.exec()
        dialog.deleteLater()

    # ---------- language ----------
    def _build_language_menu(self, bar) -> None:  # type: ignore[no-untyped-def]
        """A menu that is readable whatever language the interface is in."""

        menu = bar.addMenu(T("Language (语言)"))
        group = QActionGroup(self)
        group.setExclusive(True)
        remembered = i18n.remembered_language()
        choices: list[tuple[str | None, str]] = [(None, T("Follow the system"))]
        choices.extend(i18n.LANGUAGES.items())
        for code, label in choices:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(code == remembered)
            action.triggered.connect(lambda _checked=False, value=code: self._language_chosen(value))
            group.addAction(action)
            menu.addAction(action)

    def _language_chosen(self, code: str | None) -> None:
        try:
            i18n.remember_language(code)
        except Exception as exc:
            QMessageBox.warning(self, T("Language"), _safe_error_message(exc))
            return
        target = code or i18n.system_language()
        if target == i18n.language():
            return
        # Both languages, so the question can be read before and after the change.
        question = i18n.N_("The interface language changes when FluoroAnts starts again. Restart now?")
        text = "\n\n".join(i18n.translated(question, language) for language in (i18n.language(), target))
        answer = QMessageBox.question(
            self,
            T("Language"),
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes and not self._restart():
            QMessageBox.information(
                self,
                T("Language"),
                T("The language will change the next time FluoroAnts starts."),
            )

    def _restart(self) -> bool:
        """Close this window and start the program again; False if that is not possible now."""

        program = sys.argv[0] if sys.argv else ""
        if not program or not Path(program).is_file():
            return False
        if not self.close():
            return False
        return bool(QProcess.startDetached(sys.executable, [str(Path(program).resolve())]))

    # ---------- files dropped on the window ----------
    @staticmethod
    def _dropped_paths(mime) -> list[Path]:  # type: ignore[no-untyped-def]
        if not mime.hasUrls():
            return []
        return [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802 - Qt API
        if self.open_files_action.isEnabled() and self._dropped_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802 - Qt API
        paths = self._dropped_paths(event.mimeData())
        if not paths or not self.open_files_action.isEnabled():
            event.ignore()
            return
        event.acceptProposedAction()
        files = [
            path for path in paths
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        ]
        folders = [path for path in paths if path.is_dir()]
        for folder in folders:
            self._open_folder(folder)
        if files:
            self._add_paths(files)
        elif not folders:
            QMessageBox.information(
                self,
                T("No images"),
                T("None of the dropped files is an image this program can read."),
            )

    def _build_left_panel(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(430)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        header = QWidget(panel)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        title = QLabel(T("Image queue"), header)
        title.setObjectName("sectionTitle")
        self.queue_count_label = QLabel(T("0 images"), header)
        self.queue_count_label.setObjectName("countBadge")
        self.queue_count_label.setAlignment(Qt.AlignCenter)
        self.queue_filter_toggle = QToolButton(header)
        self.queue_filter_toggle.setObjectName("headerToggle")
        self.queue_filter_toggle.setText(T("Filter / sort"))
        self.queue_filter_toggle.setCheckable(True)
        self.queue_filter_toggle.setChecked(True)
        self.queue_filter_toggle.setFocusPolicy(Qt.NoFocus)
        self.queue_filter_toggle.setToolTip(
            T(
                "Show or hide the filter and sort bar. The current filter and sort order stay in "
                "effect while it is hidden.",
            )
        )
        self.queue_filter_toggle.toggled.connect(self._toggle_queue_filter_bar)
        header_layout.addWidget(title)
        header_layout.addWidget(self.queue_count_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.queue_filter_toggle)
        layout.addWidget(header)

        filter_bar = QFrame(panel)
        filter_bar.setObjectName("queueFilterBar")
        filter_layout = QVBoxLayout(filter_bar)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(4)
        self.queue_filter_edit = QLineEdit(filter_bar)
        self.queue_filter_edit.setObjectName("queueFilterEdit")
        self.queue_filter_edit.setClearButtonEnabled(True)
        self.queue_filter_edit.setPlaceholderText(T("Filter by file name: ch01   -ch00   *S1*"))
        self.queue_filter_edit.setToolTip(
            T(
                "Type part of a file name to filter (case-insensitive; relative paths match "
                "too).\nSeparate several words with spaces: all must match; \"-word\" excludes; "
                "the wildcards * and ? are supported.\nExample: to analyse one channel only, type "
                "its name; to leave out a group as well, add -A03.",
            )
        )
        self.queue_filter_edit.textChanged.connect(self._queue_filter_text_changed)
        filter_layout.addWidget(self.queue_filter_edit)

        sort_row = QWidget(filter_bar)
        sort_layout = QHBoxLayout(sort_row)
        sort_layout.setContentsMargins(0, 0, 0, 0)
        sort_layout.setSpacing(4)
        sort_label = QLabel(T("Sort"), sort_row)
        sort_label.setObjectName("mutedText")
        self.queue_sort_combo = FlexibleComboBox(sort_row)
        for key, label in _QUEUE_SORT_OPTIONS:
            self.queue_sort_combo.addItem(label, key)
        self.queue_sort_combo.setToolTip(
            T(
                "The order shown in the queue is the order of batch processing.\nTree mode groups "
                "by folder and sorts within each group; list mode sorts across all folders.",
            )
        )
        self.queue_sort_combo.currentIndexChanged.connect(self._queue_sort_changed)
        self.queue_sort_direction = QToolButton(sort_row)
        self.queue_sort_direction.setObjectName("compactToggle")
        self.queue_sort_direction.setCheckable(True)
        self.queue_sort_direction.setFocusPolicy(Qt.NoFocus)
        self.queue_sort_direction.setToolTip(T("Switch ascending / descending"))
        self.queue_sort_direction.toggled.connect(self._queue_sort_changed)
        self.queue_view_toggle = QToolButton(sort_row)
        self.queue_view_toggle.setObjectName("compactToggle")
        self.queue_view_toggle.setCheckable(True)
        self.queue_view_toggle.setFocusPolicy(Qt.NoFocus)
        self.queue_view_toggle.setToolTip(
            T(
                "Tree: grouped by folder.\nList: all files in one flat list sorted across folders "
                "(for example, when analysing one channel only, filtering puts that channel of "
                "every folder into one column sorted by file name).",
            )
        )
        self.queue_view_toggle.toggled.connect(self._queue_view_mode_changed)
        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self.queue_sort_combo, 1)
        sort_layout.addWidget(self.queue_sort_direction)
        sort_layout.addWidget(self.queue_view_toggle)
        filter_layout.addWidget(sort_row)
        self.queue_filter_bar = filter_bar
        layout.addWidget(filter_bar)
        self._refresh_queue_control_text()

        self.queue_filter_summary = QLabel("", panel)
        self.queue_filter_summary.setObjectName("filterSummary")
        self.queue_filter_summary.setWordWrap(True)
        self.queue_filter_summary.setVisible(False)
        layout.addWidget(self.queue_filter_summary)

        self.queue_list = QueueTree(panel)
        self.queue_list.setObjectName("imageQueue")
        self.queue_list.setMinimumHeight(150)
        self.queue_list.setToolTip(
            T(
                "Several images can be selected; the image on display has a green background. The "
                "right column shows the status of each image.",
            )
        )
        self.queue_list.currentRowChanged.connect(self._queue_row_changed)
        self.queue_list.itemSelectionChanged.connect(self._update_action_state)
        layout.addWidget(self.queue_list, 1)

        control_scroll = StableParameterScrollArea(panel)
        control_scroll.setObjectName("leftControlScroll")
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        control_body = QWidget(control_scroll)
        control_body.setObjectName("leftControlBody")
        control_body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_body)
        control_layout.setContentsMargins(0, 0, 4, 0)
        control_layout.setSpacing(6)

        self.view_group_toggle = QPushButton(T("▸ View and analysis channel"), control_body)
        self.view_group_toggle.setObjectName("sectionToggle")
        self.view_group_toggle.setCheckable(True)
        self.view_group_toggle.setToolTip(T(
            "Show or hide the viewing options: display mode, analysis channel, selection outline "
            "and more",
        ))
        self.view_group_toggle.toggled.connect(self._toggle_view_group)
        control_layout.addWidget(self.view_group_toggle)

        view_group = QFrame(control_body)
        view_group.setObjectName("collapsibleCard")
        view_layout = QFormLayout(view_group)
        view_layout.setContentsMargins(8, 8, 8, 8)
        view_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        view_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        view_layout.setSpacing(6)

        review_row = QWidget(view_group)
        review_row_layout = QHBoxLayout(review_row)
        review_row_layout.setContentsMargins(0, 0, 0, 0)
        review_row_layout.setSpacing(5)
        self.review_original_button = QPushButton(T("Review on original"), review_row)
        self.review_corrected_button = QPushButton(T("Review on corrected"), review_row)
        self.review_original_button.setObjectName("segmentedButton")
        self.review_corrected_button.setObjectName("segmentedButton")
        self.review_original_button.setCheckable(True)
        self.review_corrected_button.setCheckable(True)
        self.review_original_button.setToolTip(
            T(
                "Review the final selection, obtained from the actual analysis input, on the "
                "unmodified original image.",
            )
        )
        self.review_corrected_button.setToolTip(
            T("Review the same final selection on the analysis input after crosstalk correction.")
        )
        self.review_button_group = QButtonGroup(self)
        self.review_button_group.setExclusive(True)
        self.review_button_group.addButton(self.review_original_button)
        self.review_button_group.addButton(self.review_corrected_button)
        self.review_original_button.setChecked(True)
        self.review_corrected_button.setEnabled(False)
        self.review_original_button.clicked.connect(
            lambda: self._set_view_mode("overlay")
        )
        self.review_corrected_button.clicked.connect(
            lambda: self._set_view_mode("corrected_overlay")
        )
        review_row_layout.addWidget(self.review_original_button, 1)
        review_row_layout.addWidget(self.review_corrected_button, 1)
        view_layout.addRow(review_row)

        self.view_mode_combo = FlexibleComboBox(view_group)
        for text, key in (
            (T("Original + final selection (recommended)"), "overlay"),
            (T("Corrected input + final selection"), "corrected_overlay"),
            (T("Original only (no selection)"), "original"),
            (T("Crosstalk-corrected input only"), "corrected"),
            (T("Crosstalk reference image"), "crosstalk_reference"),
            (T("Extracted signal"), "signal"),
            (T("Background-corrected response"), "response"),
            (T("Binary mask"), "mask"),
        ):
            self.view_mode_combo.addItem(text, key)
        self.view_mode_combo.currentIndexChanged.connect(self._view_mode_changed)
        view_layout.addRow(T("Display"), self.view_mode_combo)
        self.display_channel_combo = FlexibleComboBox(view_group)
        for text, key in (
            (T("RGB composite"), "rgb"),
            (T("Red channel"), "r"),
            (T("Green channel"), "g"),
            (T("Blue channel"), "b"),
        ):
            self.display_channel_combo.addItem(text, key)
        self.display_channel_combo.setToolTip(
            T(
                "Chooses the channel that is actually analysed; after a change the old mask is "
                "void at once and the image is analysed again on that channel. The red, green and "
                "blue views keep their colour, and the exported Merge image uses the same base.",
            )
        )
        self.display_channel_combo.currentIndexChanged.connect(
            self._analysis_channel_changed
        )
        view_layout.addRow(T("Analysis channel"), self.display_channel_combo)
        self.contour_check = QCheckBox(T("Selection outline"), view_group)
        self.contour_check.setToolTip(T(
            "Shows or hides the selection boundary; the mask is not added to, erased or changed.",
        ))
        self.contour_check.setChecked(True)
        self.contour_check.toggled.connect(lambda: self._render_display())
        view_layout.addRow(self.contour_check)
        self.ai_review_check = QCheckBox(T(
            "Yellow review area (shown separately)",
        ), view_group)
        self.ai_review_check.setChecked(True)
        self.ai_review_check.setToolTip(T(
            "Display only; the yellow review area is not counted in the area of the green main "
            "selection.",
        ))
        self.ai_review_check.toggled.connect(lambda: self._render_display())
        view_layout.addRow(self.ai_review_check)
        self.object_id_check = QCheckBox(T("Object numbers (≤400)"), view_group)
        self.object_id_check.setToolTip(T(
            "Changes the display only; the mask is not added to, erased or changed.",
        ))
        self.object_id_check.toggled.connect(lambda: self._render_display())
        view_layout.addRow(self.object_id_check)
        self._update_display_channel_state()
        view_group.setVisible(False)
        self.view_group = view_group
        control_layout.addWidget(view_group)
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_view_group_toggle_text)
        self.display_channel_combo.currentIndexChanged.connect(
            self._refresh_view_group_toggle_text
        )
        self._refresh_view_group_toggle_text()
        control_layout.addWidget(self._build_crosstalk_group(control_body))

        self.image_info = QLabel(T("No image loaded"), control_body)
        self.image_info.setObjectName("infoCard")
        self.image_info.setWordWrap(True)
        control_layout.addWidget(self.image_info)
        control_scroll.setWidget(control_body)
        self.left_control_scroll = control_scroll
        self.left_control_body = control_body
        control_body.installEventFilter(self)
        layout.addWidget(control_scroll, 0)
        return panel

    # ---------- left panel helpers ----------
    def _toggle_queue_filter_bar(self, visible: bool) -> None:
        self.queue_filter_bar.setVisible(bool(visible))
        self.queue_filter_toggle.setText(T("Filter / sort") if visible else T("Filter / sort ▸"))
        self._update_queue_summary()

    def _toggle_view_group(self, expanded: bool) -> None:
        self.view_group.setVisible(bool(expanded))
        self._refresh_view_group_toggle_text()
        self._fit_left_controls()

    def _refresh_view_group_toggle_text(self, *_args: object) -> None:
        if not hasattr(self, "view_group_toggle"):
            return
        expanded = self.view_group_toggle.isChecked()
        if expanded:
            self.view_group_toggle.setText(T("▾ View and analysis channel"))
            return
        view_text = _PARENTHETICAL.split(str(self.view_mode_combo.currentText() or ""), maxsplit=1)[0]
        channel_text = str(self.display_channel_combo.currentText() or "")
        # The summary is shortened with an ellipsis when the panel is narrow.
        summary = T("▸ View · {0} · {1}", view_text, channel_text)
        available = max(60, self.view_group_toggle.width() - 20)
        self.view_group_toggle.setText(
            self.view_group_toggle.fontMetrics().elidedText(summary, Qt.ElideRight, available)
        )

    def _fit_left_controls(self, *_args: object) -> None:
        """Give the lower-left cards only the height they need (bounded), so
        the queue keeps every remaining pixel."""

        scroll = getattr(self, "left_control_scroll", None)
        body = getattr(self, "left_control_body", None)
        if scroll is None or body is None:
            return
        layout = body.layout()
        layout.activate()
        width = max(120, scroll.viewport().width())
        if layout.hasHeightForWidth():
            # Word-wrapped labels are taller than their one-line size hint.
            wanted = int(layout.totalHeightForWidth(width)) + 2
        else:
            wanted = int(body.sizeHint().height()) + 2
        # Leave the queue at least ~40 % of the window at every supported size.
        limit = max(180, min(440, int(self.height() * 0.42)))
        target = max(40, min(wanted, limit))
        if scroll.height() != target or scroll.maximumHeight() != target:
            scroll.setFixedHeight(target)
        self._refresh_view_group_toggle_text()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched is getattr(self, "left_control_body", None) and event.type() == QEvent.LayoutRequest:
            QTimer.singleShot(0, self._fit_left_controls)
        return super().eventFilter(watched, event)

    def _refresh_queue_control_text(self) -> None:
        if hasattr(self, "queue_sort_direction"):
            self.queue_sort_direction.setText(
                T("↓ Descending") if self.queue_sort_direction.isChecked() else T("↑ Ascending")
            )
        if hasattr(self, "queue_view_toggle"):
            self.queue_view_toggle.setText(
                T("List") if self.queue_view_toggle.isChecked() else T("Tree")
            )

    def _build_crosstalk_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox(T("Enable crosstalk correction"), parent)
        group.setCheckable(True)
        group.setChecked(False)
        group.setToolTip(T(
            "The original image is not modified; only the scalar signal passed to recognition is "
            "reduced, pixel by pixel.",
        ))
        layout = QVBoxLayout(group)
        layout.setSpacing(5)

        self.crosstalk_pair_summary_label = QLabel(
            T("Off · the original image is analysed directly"),
            group,
        )
        self.crosstalk_pair_summary_label.setObjectName("compactStatus")
        self.crosstalk_pair_summary_label.setWordWrap(True)
        layout.addWidget(self.crosstalk_pair_summary_label)

        body = QWidget(group)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 2, 0, 0)
        body_layout.setSpacing(6)

        self.crosstalk_reference_label = QLabel(T("Current image: no reference image chosen"), body)
        self.crosstalk_reference_label.setObjectName("mutedText")
        self.crosstalk_reference_label.setWordWrap(True)
        body_layout.addWidget(self.crosstalk_reference_label)

        reference_buttons = QHBoxLayout()
        self.crosstalk_choose_button = QPushButton(T("Choose reference image…"), body)
        self.crosstalk_choose_button.clicked.connect(self._choose_crosstalk_reference)
        self.crosstalk_clear_button = QPushButton(T("Clear"), body)
        self.crosstalk_clear_button.clicked.connect(self._clear_crosstalk_reference)
        reference_buttons.addWidget(self.crosstalk_choose_button, 1)
        reference_buttons.addWidget(self.crosstalk_clear_button)
        body_layout.addLayout(reference_buttons)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.crosstalk_coefficient_spin = QDoubleSpinBox(body)
        self.crosstalk_coefficient_spin.setRange(0.0, 5.0)
        self.crosstalk_coefficient_spin.setDecimals(2)
        self.crosstalk_coefficient_spin.setSingleStep(0.05)
        self.crosstalk_coefficient_spin.setValue(0.30)
        self.crosstalk_coefficient_spin.setSuffix(" ×")
        self.crosstalk_coefficient_spin.setKeyboardTracking(False)
        self.crosstalk_coefficient_spin.valueChanged.connect(
            self._crosstalk_settings_changed
        )
        form.addRow(T("Coefficient"), self.crosstalk_coefficient_spin)
        body_layout.addLayout(form)

        self.crosstalk_details_toggle = QPushButton(T(
            "▸ More correction settings and batch pairing",
        ), body)
        self.crosstalk_details_toggle.setObjectName("sectionToggle")
        self.crosstalk_details_toggle.setCheckable(True)
        self.crosstalk_details_toggle.toggled.connect(
            self._toggle_crosstalk_details
        )
        body_layout.addWidget(self.crosstalk_details_toggle)

        details = QWidget(body)
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(6)
        details_form = QFormLayout()
        details_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        details_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.crosstalk_reference_channel_combo = FlexibleComboBox(details)
        for text, key in (
            (T("Greyscale / automatic"), "auto"),
            (T("Red channel"), "red"),
            (T("Green channel"), "green"),
            (T("Blue channel"), "blue"),
            (T("RGB maximum"), "max_rgb"),
        ):
            self.crosstalk_reference_channel_combo.addItem(text, key)
        self.crosstalk_reference_channel_combo.currentIndexChanged.connect(
            self._crosstalk_settings_changed
        )
        details_form.addRow(T("Reference channel"), self.crosstalk_reference_channel_combo)
        details_layout.addLayout(details_form)

        self.crosstalk_background_check = QCheckBox(
            T("Protect the reference background (recommended)"),
            details,
        )
        self.crosstalk_background_check.setToolTip(
            T(
                "Subtracts only the reference signal above its own background, which lowers the "
                "risk of darkening the whole background.",
            )
        )
        self.crosstalk_background_check.setChecked(True)
        self.crosstalk_background_check.toggled.connect(
            self._crosstalk_settings_changed
        )
        details_layout.addWidget(self.crosstalk_background_check)

        self.crosstalk_formula_label = QLabel("", details)
        self.crosstalk_formula_label.setObjectName("mutedText")
        self.crosstalk_formula_label.setWordWrap(True)
        details_layout.addWidget(self.crosstalk_formula_label)

        self.crosstalk_batch_pair_button = QPushButton(T("Pair in batch by relative path…"), details)
        self.crosstalk_batch_pair_button.clicked.connect(
            self._pair_crosstalk_references_by_folder
        )
        details_layout.addWidget(self.crosstalk_batch_pair_button)

        warning = QLabel(
            T(
                "Target and reference image must show the same field, have exactly the same size "
                "and be registered; the program does not rescale or align them.",
            ),
            details,
        )
        warning.setObjectName("mutedText")
        warning.setWordWrap(True)
        details_layout.addWidget(warning)
        details.setVisible(False)
        self.crosstalk_details = details
        body_layout.addWidget(details)

        body.setVisible(False)
        self.crosstalk_body = body
        layout.addWidget(body)

        self.crosstalk_group = group
        self._update_crosstalk_formula()
        group.toggled.connect(self._toggle_crosstalk_body)
        group.toggled.connect(self._crosstalk_enabled_changed)
        return group

    def _toggle_crosstalk_body(self, enabled: bool) -> None:
        self.crosstalk_body.setVisible(bool(enabled))
        self._update_crosstalk_reference_state()
        self._fit_left_controls()

    def _toggle_crosstalk_details(self, expanded: bool) -> None:
        self.crosstalk_details.setVisible(bool(expanded))
        self.crosstalk_details_toggle.setText(
            ("▾" if expanded else "▸") + T(" More correction settings and batch pairing")
        )
        self._fit_left_controls()

    def _build_right_panel(self) -> QWidget:
        outer = QFrame(self)
        outer.setObjectName("sidePanel")
        outer.setMinimumWidth(390)
        outer.setMaximumWidth(540)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = StableParameterScrollArea(outer)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget(scroll)
        # The body takes the width of the panel; its rows wrap instead of
        # pushing content out of sight when texts are long.
        body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)

        # Order by how often each card is touched: choose a recipe, run and
        # read the result, adjust the edge distance, then the quick target
        # options.  Everything below the fold is either an occasional tweak
        # ("More") or the full professional parameter set.
        self.preset_group = self._build_preset_group()
        self.run_group = self._build_run_group()
        self.edge_offset_group = self._build_edge_offset_group()
        self.quick_target_group = self._build_quick_target_group()
        self.quick_more_section = self._build_quick_more_section()
        self.professional_section = self._build_professional_section()
        layout.addWidget(self.preset_group)
        layout.addWidget(self.run_group)
        layout.addWidget(self.edge_offset_group)
        layout.addWidget(self.quick_target_group)
        layout.addWidget(self.quick_more_section)
        layout.addWidget(self.professional_section)
        layout.addStretch(1)
        scroll.setWidget(body)
        self.right_control_scroll = scroll
        outer_layout.addWidget(scroll)
        return outer

    def _build_preset_group(self) -> QGroupBox:
        group = QGroupBox(T("Recognition preset"), self)
        layout = QVBoxLayout(group)
        layout.setSpacing(5)
        self.recognition_engine_combo = FlexibleComboBox(group)
        self.recognition_engine_combo.setAccessibleName(T("Recognition mode"))
        self.recognition_engine_combo.addItem(T("Learned model v6.3"), "learned_v6")
        self.recognition_engine_combo.addItem(T("Classic recognition"), "classic")
        self.recognition_engine_combo.currentIndexChanged.connect(self._recognition_engine_changed)
        layout.addWidget(self.recognition_engine_combo)
        self.ai_mode_hint = QLabel(
            T(
                "Learned model · finds small targets clearly brighter than their immediate "
                "background; any image type, size and bit depth. The green main selection and "
                "the yellow review area are counted separately. The brightness and shape "
                "parameters below take no part in this mode.",
            ), group
        )
        self.ai_mode_hint.setWordWrap(True)
        self.ai_mode_hint.setVisible(False)
        self.ai_mode_hint.setObjectName("noteCard")
        layout.addWidget(self.ai_mode_hint)
        self.learned_sensitivity_row = QWidget(group)
        sensitivity_layout = QHBoxLayout(self.learned_sensitivity_row)
        sensitivity_layout.setContentsMargins(0, 0, 0, 0)
        sensitivity_layout.setSpacing(6)
        sensitivity_layout.addWidget(QLabel(T("Sensitivity"), self.learned_sensitivity_row))
        self.learned_sensitivity_spin = QDoubleSpinBox(self.learned_sensitivity_row)
        self.learned_sensitivity_spin.setAccessibleName(T("Sensitivity of the learned model"))
        self.learned_sensitivity_spin.setRange(-3.0, 3.0)
        self.learned_sensitivity_spin.setSingleStep(0.25)
        self.learned_sensitivity_spin.setDecimals(2)
        self.learned_sensitivity_spin.setMinimumWidth(84)
        self.learned_sensitivity_spin.setToolTip(T(
            "0 is the threshold the model was calibrated with. Higher values also accept "
            "candidates that score lower (more targets, more risk of false ones); lower values "
            "are stricter. The yellow review band moves along.",
        ))
        self.learned_sensitivity_spin.valueChanged.connect(self._learned_sensitivity_changed)
        sensitivity_layout.addWidget(self.learned_sensitivity_spin)
        self.learned_sensitivity_note = QLabel("", self.learned_sensitivity_row)
        self.learned_sensitivity_note.setObjectName("mutedText")
        sensitivity_layout.addWidget(self.learned_sensitivity_note, 1)
        self.learned_sensitivity_row.setVisible(False)
        layout.addWidget(self.learned_sensitivity_row)
        self._update_learned_sensitivity_note()
        row = QWidget(group)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(5)
        self.preset_choices = PresetCombo(accessible_name=T("Recognition preset"), parent=row)
        self.preset_choices.setToolTip(
            T(
                "A preset is applied as soon as it is chosen; choose the current preset again to "
                "restore its defaults.",
            )
        )
        self.preset_choices.choice_activated.connect(self._preset_selected)
        self.save_preset_button = QPushButton(T("Save"), row)
        self.save_preset_button.setToolTip(T("Save all current parameters as a personal preset"))
        self.save_preset_button.clicked.connect(self._save_user_preset)
        self.delete_preset_button = QPushButton(T("Delete"), row)
        self.delete_preset_button.setToolTip(T("Delete the selected personal preset"))
        self.delete_preset_button.clicked.connect(self._delete_user_preset)
        row_layout.addWidget(self.preset_choices, 1)
        row_layout.addWidget(self.save_preset_button)
        row_layout.addWidget(self.delete_preset_button)
        layout.addWidget(row)
        self.preset_description = QLabel("", group)
        self.preset_description.setObjectName("mutedText")
        self.preset_description.setWordWrap(True)
        layout.addWidget(self.preset_description)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox(T("Analysis and results"), self)
        layout = QVBoxLayout(group)
        layout.setSpacing(5)

        self.analysis_badge = QLabel(T("Waiting for analysis"), group)
        self.analysis_badge.setObjectName("analysisBadge")
        self.analysis_badge.setAlignment(Qt.AlignCenter)
        self.analysis_badge.setWordWrap(True)
        layout.addWidget(self.analysis_badge)

        self.analysis_context_label = QLabel(T("Selection source: not analysed yet"), group)
        self.analysis_context_label.setObjectName("analysisContext")
        self.analysis_context_label.setAlignment(Qt.AlignCenter)
        self.analysis_context_label.setWordWrap(True)
        layout.addWidget(self.analysis_context_label)

        run_row = QWidget(group)
        run_layout = QHBoxLayout(run_row)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(6)
        self.auto_check = QCheckBox(T("Auto preview"), run_row)
        self.auto_check.setChecked(True)
        self.auto_check.setToolTip(T(
            "Regenerates the quick preview automatically after a parameter or the edge distance "
            "changes",
        ))
        self.auto_check.toggled.connect(self._auto_preview_toggled)
        self.preview_button = QPushButton(T("Update preview"), run_row)
        self.preview_button.setToolTip(T("Regenerate the quick preview with the current settings"))
        self.preview_button.clicked.connect(
            lambda: self._start_segmentation(full=False)
        )
        self.full_button = QPushButton(T("Full resolution"), run_row)
        self.full_button.setObjectName("primaryButton")
        self.full_button.setToolTip(T(
            "Run the full-resolution analysis; manual edits and exports are based on the "
            "full-resolution result (F5)",
        ))
        self.full_button.clicked.connect(lambda: self._start_segmentation(full=True))
        run_layout.addWidget(self.auto_check)
        run_layout.addStretch(1)
        run_layout.addWidget(self.preview_button)
        run_layout.addWidget(self.full_button)
        layout.addWidget(run_row)

        self.stats_summary_label = QLabel(T("No result yet"), group)
        self.stats_summary_label.setObjectName("statsSummary")
        self.stats_summary_label.setWordWrap(True)
        self.stats_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.stats_summary_label)

        detail_row = QWidget(group)
        detail_layout = QHBoxLayout(detail_row)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(6)
        self.stats_details_toggle = QPushButton(T("▸ Detailed statistics"), detail_row)
        self.stats_details_toggle.setObjectName("sectionToggle")
        self.stats_details_toggle.setCheckable(True)
        self.stats_details_toggle.toggled.connect(self._toggle_stats_details)
        self.object_table_button = QPushButton(T("Object details"), detail_row)
        self.object_table_button.setToolTip(T(
            "View the measurements of each object; locate or exclude single objects",
        ))
        self.object_table_button.clicked.connect(self._show_object_table)
        detail_layout.addWidget(self.stats_details_toggle, 1)
        detail_layout.addWidget(self.object_table_button)
        layout.addWidget(detail_row)

        self.stats_label = QLabel(T("No result yet"), group)
        self.stats_label.setObjectName("statsCard")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.stats_label.setVisible(False)
        layout.addWidget(self.stats_label)

        self.plan_details_toggle = QPushButton(T("▸ Parameter status and execution plan"), group)
        self.plan_details_toggle.setObjectName("sectionToggle")
        self.plan_details_toggle.setCheckable(True)
        self.plan_details_toggle.toggled.connect(self._toggle_plan_details)
        layout.addWidget(self.plan_details_toggle)
        plan_box = QWidget(group)
        plan_layout = QVBoxLayout(plan_box)
        plan_layout.setContentsMargins(0, 0, 0, 0)
        plan_layout.setSpacing(4)
        self.parameter_mode_label = QLabel(T("Follows the chosen preset completely"), plan_box)
        self.parameter_mode_label.setObjectName("mutedText")
        self.parameter_mode_label.setWordWrap(True)
        plan_layout.addWidget(self.parameter_mode_label)
        self.quick_plan_label = QLabel(T("The execution plan has not been compiled yet"), plan_box)
        self.quick_plan_label.setObjectName("mutedText")
        self.quick_plan_label.setWordWrap(True)
        self.quick_plan_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        plan_layout.addWidget(self.quick_plan_label)
        plan_box.setVisible(False)
        self.plan_details_box = plan_box
        layout.addWidget(plan_box)
        return group

    def _toggle_stats_details(self, expanded: bool) -> None:
        self.stats_label.setVisible(bool(expanded))
        self.stats_details_toggle.setText(("▾" if expanded else "▸") + T(" Detailed statistics"))

    def _toggle_plan_details(self, expanded: bool) -> None:
        self.plan_details_box.setVisible(bool(expanded))
        self.plan_details_toggle.setText(
            ("▾" if expanded else "▸") + T(" Parameter status and execution plan")
        )

    def _build_edge_offset_group(self) -> QGroupBox:
        group = QGroupBox(T(
            "Edge distance · expand / shrink after recognition",
        ), self)
        group.setToolTip(
            T(
                "Offsets the boundary of the recognized selection as a whole (exact Euclidean "
                "distance).\nIt is not a recognition parameter: it is not written into presets or "
                "recipes and does not mark the preset as \"custom\".\nIt is not a manual edit "
                "either: it is not in the undo history and does not trigger the \"unexported "
                "manual edits\" prompt.\nExport current and batch export both use the offset "
                "selection.",
            )
        )
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        row = QWidget(group)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        self.edge_offset_check = QCheckBox(T("Enable"), row)
        self.edge_offset_check.setToolTip(T(
            "When off, the original recognized boundary is used; turning it on again restores the "
            "last distance",
        ))
        self.edge_offset_check.toggled.connect(self._edge_offset_changed)
        self.edge_offset_spin = SignedSpinBox(row)
        self.edge_offset_spin.setRange(-MAX_EDGE_OFFSET_PX, MAX_EDGE_OFFSET_PX)
        self.edge_offset_spin.setValue(0)
        self.edge_offset_spin.setSingleStep(1)
        self.edge_offset_spin.setSuffix(" px")
        self.edge_offset_spin.setKeyboardTracking(False)
        self.edge_offset_spin.setEnabled(False)
        self.edge_offset_spin.setMinimumWidth(96)
        self.edge_offset_spin.setToolTip(
            T(
                "Positive values expand, negative values shrink, in source pixels; the preview is "
                "scaled accordingly.",
            )
        )
        self.edge_offset_spin.valueChanged.connect(self._edge_offset_changed)
        self.edge_offset_status = QLabel(T("Off"), row)
        self.edge_offset_status.setObjectName("compactStatus")
        row_layout.addWidget(self.edge_offset_check)
        row_layout.addWidget(self.edge_offset_spin)
        row_layout.addWidget(self.edge_offset_status, 1)
        layout.addWidget(row)
        note = QLabel(
            T(
                "Acts on the recognition result only: not written into the preset and not a "
                "manual edit; applies to exports and batches alike.",
            ),
            group,
        )
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addWidget(note)
        return group

    def _build_quick_target_group(self) -> QGroupBox:
        group = QGroupBox(T("Quick settings · target"), self)
        group.setToolTip(
            T(
                "\"Detection logic\", \"Shape tendency\" and \"Target size\" allow several "
                "choices in one row, whose masks are united; click again to clear. The other rows "
                "take one choice. Each of several choices is recognized separately and the "
                "results are united, which usually finds more and takes longer.",
            )
        )
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        self._quick_controls: dict[str, ChoiceButtonGroup] = {}

        _, self.quick_logic_choice = self._add_quick_choice(
            layout, group, "logic", T("Detection logic (combinable)"), QUICK_LOGIC_LABELS
        )
        _, self.quick_signal_choice = self._add_quick_choice(
            layout, group, "signal", T("Signal source"), QUICK_SIGNAL_LABELS
        )
        _, self.quick_polarity_choice = self._add_quick_choice(
            layout, group, "polarity", T("Target brightness"), QUICK_POLARITY_LABELS
        )
        _, self.quick_sensitivity_choice = self._add_quick_choice(
            layout, group, "sensitivity", T("Detection sensitivity"), QUICK_SENSITIVITY_LABELS
        )
        _, self.quick_shape_choice = self._add_quick_choice(
            layout, group, "shape", T("Shape tendency (combinable)"), QUICK_SHAPE_LABELS
        )
        self.quick_size_label, self.quick_size_choice = self._add_quick_choice(
            layout, group, "size", T("Target size (combinable)"), QUICK_SIZE_LABELS
        )
        return group

    def _build_quick_more_group(self) -> QGroupBox:
        group = QGroupBox(T("Quick settings · imaging, outline and cleanup"), self)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.quick_shape_strictness_label, self.quick_shape_strictness_choice = (
            self._add_quick_choice(
                layout,
                group,
                "shape_strictness",
                T("Shape requirement"),
                QUICK_SHAPE_STRICTNESS_LABELS,
            )
        )
        _, self.quick_intensity_choice = self._add_quick_choice(
            layout, group, "intensity", T("Positive intensity filter"), QUICK_INTENSITY_LABELS
        )
        _, self.quick_imaging_choice = self._add_quick_choice(
            layout, group, "imaging", T("Brightness handling"), QUICK_IMAGING_LABELS
        )
        _, self.quick_scale_choice = self._add_quick_choice(
            layout, group, "scale", T("Size reference"), QUICK_SCALE_LABELS
        )
        scale_note = QLabel(
            T(
                "\"Relative to the frame\" suits screenshots or resampled versions of the same "
                "field; cropping, stitching or a change of magnification cannot be told from "
                "width and height alone, so use \"Fixed source pixels\" for those.",
            ),
            group,
        )
        scale_note.setObjectName("mutedText")
        scale_note.setWordWrap(True)
        layout.addWidget(scale_note)

        definitions = (
            ("background", T("Background"), QUICK_BACKGROUND_LABELS),
            ("noise", T("Denoising"), QUICK_NOISE_LABELS),
            ("edge", T("Boundary extent"), QUICK_EDGE_LABELS),
            ("speck_cleanup", T("Speck cleanup"), QUICK_SPECK_LABELS),
            ("opening", T("Despeckle"), QUICK_OPENING_LABELS),
            ("gap_connect", T("Connect gaps"), QUICK_GAP_LABELS),
            ("contour_adjust", T("Outline adjustment"), QUICK_CONTOUR_LABELS),
            ("holes", T("Inner holes"), QUICK_HOLE_LABELS),
            ("touching", T("Touching objects"), QUICK_TOUCHING_LABELS),
            ("border", T("Border objects"), QUICK_BORDER_LABELS),
        )
        for field_name, label, choices in definitions:
            label_widget, _choice = self._add_quick_choice(
                layout, group, field_name, label, choices
            )
            if field_name == "edge":
                self.quick_edge_label = label_widget
            elif field_name == "holes":
                self.quick_holes_label = label_widget
            elif field_name == "touching":
                self.quick_touching_label = label_widget

        self.quick_edge_choice = self._quick_controls["edge"]
        self.quick_holes_choice = self._quick_controls["holes"]
        self.quick_touching_choice = self._quick_controls["touching"]
        note = QLabel(
            T(
                "When background texture joins into large areas, try \"Texture-background "
                "suppression\" first; if the targets are mainly tiny spots, add \"Sparse tiny "
                "spots + Dense tiny spots\"; shape is judged by compactness, elongation, "
                "thickness, holes and irregularity.",
            ),
            group,
        )
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addWidget(note)
        return group

    def _build_quick_more_section(self) -> QWidget:
        section = QWidget(self)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.quick_more_toggle = QPushButton(T(
            "▶ More quick options (imaging, outline and cleanup)",
        ), section)
        self.quick_more_toggle.setCheckable(True)
        self.quick_more_toggle.toggled.connect(self._toggle_quick_more_section)
        layout.addWidget(self.quick_more_toggle)
        self.quick_more_group = self._build_quick_more_group()
        self.quick_more_group.setVisible(False)
        layout.addWidget(self.quick_more_group)
        return section

    def _toggle_quick_more_section(self, expanded: bool) -> None:
        self.quick_more_group.setVisible(bool(expanded))
        self.quick_more_toggle.setText(
            ("▼" if expanded else "▶") + T(" More quick options (imaging, outline and cleanup)")
        )

    def _add_quick_choice(
        self,
        layout: QVBoxLayout,
        parent: QWidget,
        field_name: str,
        label_text: str,
        choices: dict[str, str],
    ) -> tuple[QLabel, ChoiceButtonGroup]:
        label = QLabel(label_text, parent)
        label.setObjectName("choiceFieldLabel")
        label.setWordWrap(True)
        choice = ChoiceButtonGroup(
            choices,
            accessible_name=label_text,
            multi_select=field_name in QUICK_MULTI_FIELDS,
            parent=parent,
        )
        choice.value_changed.connect(self._on_quick_parameter_changed)
        self._quick_controls[field_name] = choice
        layout.addWidget(label)
        layout.addWidget(choice)
        return label, choice

    def _build_professional_section(self) -> QWidget:
        section = QWidget(self)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.professional_toggle = QPushButton(
            T("▶ Professional parameters ({0})", _PROFESSIONAL_PARAMETER_COUNT),
            section,
        )
        self.professional_toggle.setCheckable(True)
        self.professional_toggle.toggled.connect(self._toggle_professional_section)
        layout.addWidget(self.professional_toggle)

        self.professional_content = QWidget(section)
        content_layout = QVBoxLayout(self.professional_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.addWidget(self._build_basic_group())
        content_layout.addWidget(self._build_advanced_box())
        self.professional_content.setVisible(False)
        layout.addWidget(self.professional_content)
        return section

    def _toggle_professional_section(self, expanded: bool) -> None:
        self.professional_content.setVisible(bool(expanded))
        self.professional_toggle.setText(
            ("▼" if expanded else "▶")
            + T(" Professional parameters ({0})", _PROFESSIONAL_PARAMETER_COUNT)
        )

    def _build_basic_group(self) -> QGroupBox:
        group = QGroupBox(T("Common professional parameters"), self)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.scale_mode_combo = self._make_combo(_SPATIAL_SCALE_MODE_LABELS)
        self.reference_short_side_spin = self._spin(
            256, 8192, 1024, step=128, suffix=" px"
        )
        self.scale_mode_combo.setToolTip(
            T(
                "Relative scaling suits a frame that was resampled as a whole; fixed source "
                "pixels suit cropping, stitching or a change of magnification.",
            )
        )
        self.reference_short_side_spin.setToolTip(
            T(
                "The canonical short side for analysis in relative mode. A larger value keeps "
                "smaller detail but is slower.",
            )
        )
        form.addRow(T("Spatial scale mode"), self.scale_mode_combo)
        form.addRow(T("Relative reference short side"), self.reference_short_side_spin)

        self.signal_combo = self._make_combo(SIGNAL_MODE_LABELS)
        form.addRow(T("Signal extraction"), self.signal_combo)
        self.polarity_combo = FlexibleComboBox(group)
        self.polarity_combo.addItem(T("Bright targets / dark background"), False)
        self.polarity_combo.addItem(T("Dark targets / bright background"), True)
        form.addRow(T("Positive polarity"), self.polarity_combo)

        self.sensitivity_slider = QSlider(Qt.Horizontal, group)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_spin = self._spin(0, 100, 50, suffix="")
        self.sensitivity_slider.setValue(self.sensitivity_spin.value())
        sensitivity_row = QWidget(group)
        sensitivity_layout = QHBoxLayout(sensitivity_row)
        sensitivity_layout.setContentsMargins(0, 0, 0, 0)
        sensitivity_layout.addWidget(self.sensitivity_slider, 1)
        sensitivity_layout.addWidget(self.sensitivity_spin)
        form.addRow(T("Sensitivity"), sensitivity_row)
        self.sensitivity_slider.valueChanged.connect(self.sensitivity_spin.setValue)
        self.sensitivity_spin.valueChanged.connect(self.sensitivity_slider.setValue)

        self.background_radius_spin = self._spin(0, 2048, 24, suffix=" px")
        form.addRow(T("Background scale"), self.background_radius_spin)
        self.denoise_strength_spin = self._double_spin(0.0, 30.0, 0.8, 0.1, 2, " px")
        form.addRow(T("Denoising strength"), self.denoise_strength_spin)
        self.grow_ratio_spin = self._double_spin(0.01, 1.0, 0.60, 0.01, 2, "")
        form.addRow(T("Boundary growth ratio"), self.grow_ratio_spin)
        self.min_area_spin = self._spin(0, 100_000_000, 8, suffix=" px²")
        form.addRow(T("Minimum area"), self.min_area_spin)
        self.max_area_spin = self._spin(0, 100_000_000, 0, suffix=" px²")
        self.max_area_spin.setSpecialValueText(T("No limit"))
        form.addRow(T("Maximum area"), self.max_area_spin)
        self.close_radius_spin = self._spin(0, 256, 1, suffix=" px")
        form.addRow(T("Connect / closing"), self.close_radius_spin)
        self.hole_area_spin = self._spin(0, 100_000_000, 4, suffix=" px²")
        form.addRow(T("Hole-fill area"), self.hole_area_spin)

        controls = [
            self.scale_mode_combo,
            self.reference_short_side_spin,
            self.signal_combo,
            self.polarity_combo,
            self.sensitivity_spin,
            self.background_radius_spin,
            self.denoise_strength_spin,
            self.grow_ratio_spin,
            self.min_area_spin,
            self.max_area_spin,
            self.close_radius_spin,
            self.hole_area_spin,
        ]
        for widget in controls:
            self._connect_parameter_widget(widget)

        return group

    def _build_advanced_box(self) -> QToolBox:
        box = QToolBox(self)
        box.setObjectName("advancedBox")
        box.addItem(self._build_signal_advanced(), T("Signal and dynamic range"))
        box.addItem(self._build_preprocess_advanced(), T("Denoising and background correction"))
        box.addItem(self._build_threshold_advanced(), T("Threshold and local recognition"))
        box.addItem(self._build_morphology_advanced(), T("Morphology and object separation"))
        box.addItem(self._build_filter_advanced(), T("Object shape filters"))
        return box

    def _build_signal_advanced(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.intensity_space_combo = self._make_combo(INTENSITY_SPACE_LABELS)
        self.rank_tail_power_spin = self._double_spin(
            1.0, 12.0, 4.0, 1.0, 1, ""
        )
        self.intensity_space_combo.setToolTip(
            T(
                "Robust SNR judges by local effect size and may return an empty mask; relative "
                "rank is insensitive to overall brightness and gamma changes; percentile linear "
                "stretches brightness between the black and white percentiles, with gamma.",
            )
        )
        self.rank_tail_power_spin.setToolTip(
            T(
                "Prior on the share of targets in relative rank: 2 for large areas, 4 general, 8 "
                "for sparse targets or spots.",
            )
        )
        self.black_percentile_spin = self._double_spin(0.0, 49.9, 0.2, 0.1, 2, "%")
        self.white_percentile_spin = self._double_spin(50.0, 100.0, 99.8, 0.1, 2, "%")
        self.gamma_spin = self._double_spin(0.05, 8.0, 1.0, 0.05, 2, "")
        self.red_weight_spin = self._double_spin(0.0, 10.0, 1.0, 0.1, 2, "")
        self.green_weight_spin = self._double_spin(0.0, 10.0, 1.0, 0.1, 2, "")
        self.blue_weight_spin = self._double_spin(0.0, 10.0, 1.0, 0.1, 2, "")
        for label, widget in (
            (T("Intensity space"), self.intensity_space_combo),
            (T("Relative signal sparsity"), self.rank_tail_power_spin),
            (T("Black-point percentile"), self.black_percentile_spin),
            (T("White-point percentile"), self.white_percentile_spin),
            ("Gamma", self.gamma_spin),
            (T("Red weight"), self.red_weight_spin),
            (T("Green weight"), self.green_weight_spin),
            (T("Blue weight"), self.blue_weight_spin),
        ):
            form.addRow(label, widget)
            self._connect_parameter_widget(widget)
        return page

    def _build_preprocess_advanced(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.denoise_method_combo = self._make_combo(DENOISE_METHOD_LABELS)
        self.background_method_combo = self._make_combo(BACKGROUND_METHOD_LABELS)
        self.background_strength_spin = self._double_spin(0.0, 3.0, 1.0, 0.05, 2, "")
        form.addRow(T("Denoising method"), self.denoise_method_combo)
        form.addRow(T("Background method"), self.background_method_combo)
        form.addRow(T("Background subtraction strength"), self.background_strength_spin)
        for widget in (
            self.denoise_method_combo,
            self.background_method_combo,
            self.background_strength_spin,
        ):
            self._connect_parameter_widget(widget)
        return page

    def _build_threshold_advanced(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.threshold_method_combo = self._make_combo(THRESHOLD_METHOD_LABELS)
        self.snr_seed_sigma_spin = self._double_spin(2.0, 12.0, 5.0, 0.25, 2, " σ")
        self.snr_seed_sigma_spin.setToolTip(
            T(
                "How many times the local noise of this image a strong positive seed must exceed "
                "at least; larger values are stricter.",
            )
        )
        self.manual_threshold_spin = self._double_spin(0.0, 1.0, 0.25, 0.01, 3, "")
        self.threshold_percentile_spin = self._double_spin(0.0, 100.0, 98.5, 0.1, 2, "%")
        self.local_window_spin = self._spin(3, 4095, 51, step=2, suffix=" px")
        self.local_k_spin = self._double_spin(-2.0, 2.0, 0.20, 0.02, 3, "")
        for label, widget in (
            (T("Threshold method"), self.threshold_method_combo),
            (T("Positive evidence threshold"), self.snr_seed_sigma_spin),
            (T("Manual threshold"), self.manual_threshold_spin),
            (T("Percentile threshold"), self.threshold_percentile_spin),
            (T("Local window"), self.local_window_spin),
            (T("Local k"), self.local_k_spin),
        ):
            form.addRow(label, widget)
            self._connect_parameter_widget(widget)
        return page

    def _build_morphology_advanced(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.open_radius_spin = self._spin(0, 256, 0, suffix=" px")
        self.erode_radius_spin = self._spin(0, 128, 0, suffix=" px")
        self.dilate_radius_spin = self._spin(0, 128, 0, suffix=" px")
        self.remove_border_check = QCheckBox(T("Remove objects touching the image border"), page)
        self.split_touching_check = QCheckBox(T("Separate touching objects by watershed"), page)
        self.peak_distance_spin = self._spin(1, 256, 6, suffix=" px")
        self.peak_height_spin = self._double_spin(0.02, 0.95, 0.28, 0.01, 2, "")
        for label, widget in (
            (T("Opening"), self.open_radius_spin),
            (T("Erosion"), self.erode_radius_spin),
            (T("Dilation"), self.dilate_radius_spin),
        ):
            form.addRow(label, widget)
        form.addRow(self.remove_border_check)
        form.addRow(self.split_touching_check)
        form.addRow(T("Minimum peak distance"), self.peak_distance_spin)
        form.addRow(T("Relative peak height"), self.peak_height_spin)
        for widget in (
            self.open_radius_spin,
            self.erode_radius_spin,
            self.dilate_radius_spin,
            self.remove_border_check,
            self.split_touching_check,
            self.peak_distance_spin,
            self.peak_height_spin,
        ):
            self._connect_parameter_widget(widget)
        return page

    def _build_filter_advanced(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.shape_profile_combo = self._make_combo(OBJECT_SHAPE_PROFILE_LABELS)
        self.shape_profile_strictness_combo = self._make_combo(
            OBJECT_SHAPE_STRICTNESS_LABELS
        )
        self.min_thickness_spin = self._double_spin(
            0.0, 4096.0, 0.0, 0.5, 2, " px"
        )
        self.min_thickness_spin.setSpecialValueText(T("No limit"))
        self.max_thickness_spin = self._double_spin(
            0.0, 4096.0, 0.0, 0.5, 2, " px"
        )
        self.max_thickness_spin.setSpecialValueText(T("No limit"))
        self.min_circularity_spin = self._double_spin(0.0, 1.0, 0.0, 0.02, 3, "")
        self.min_solidity_spin = self._double_spin(0.0, 1.0, 0.0, 0.02, 3, "")
        self.max_eccentricity_spin = self._double_spin(0.0, 1.0, 1.0, 0.02, 3, "")
        self.max_aspect_ratio_spin = self._double_spin(1.0, 1000.0, 1000.0, 0.5, 2, "")
        self.min_mean_intensity_spin = self._double_spin(0.0, 1.0, 0.0, 0.01, 3, "")
        self.min_response_confidence_spin = self._double_spin(
            0.0, 5.0, 0.0, 0.05, 2, ""
        )
        self.min_response_confidence_spin.setSpecialValueText(T("Off"))
        self.min_response_confidence_spin.setToolTip(
            T(
                "Filters out whole weak objects by their mean background-corrected response "
                "relative to this image's automatic response threshold; the outlines of kept "
                "objects are not shrunk further.",
            )
        )
        self.noise_reference_value_spin = self._double_spin(
            0.0, 1_000_000.0, 0.0, 10.0, 1, ""
        )
        self.noise_reference_value_spin.setSpecialValueText(T("Off (noise of this image)"))
        self.noise_reference_value_spin.setToolTip(
            T(
                "Noise scale shared by the whole batch, in grey levels of the source image.\nWhen "
                "off (the default), the \"SNR evidence\" criterion uses the residual noise of "
                "[each image itself], so within a batch the threshold rises automatically in "
                "fields with much background texture and falls in quiet ones; when that texture "
                "is itself an autofluorescence artefact, this is exactly the wrong "
                "direction.\nWith a value above 0 the criterion becomes an absolute contrast "
                "shared by the batch: seed threshold = \"SNR seed multiple\" × this value, "
                "identical for every image.\nThis is meaningful only when the whole batch shares "
                "one exposure and one export curve; calibrate the value on blank control images.",
            )
        )
        self.background_ceiling_mode_combo = self._make_combo(
            BACKGROUND_CEILING_MODE_LABELS
        )
        self.background_ceiling_mode_combo.setToolTip(
            T(
                "How autofluorescent compartments are excluded.\n[Absolute grey-level ceiling] "
                "decides by the absolute value below and requires the whole batch to share one "
                "exposure and one export curve; images whose brightness was adjusted one by one "
                "cannot use it.\n[Automatic per image] splits the broad background of each image "
                "into two modes and excludes the brighter one; it does not rely on comparability "
                "across images and suits existing images with differing exposure or exported "
                "brightness.\nBoth exclude by the compartment an object lies in, not by the "
                "object's brightness rank within the image.\nTargets inside excluded compartments "
                "cannot be counted; say so in reports.",
            )
        )
        self.background_ceiling_scale_spin = self._spin(0, 2048, 0, suffix=" px")
        self.background_ceiling_scale_spin.setSpecialValueText(T("Follow background scale"))
        self.background_ceiling_scale_spin.setToolTip(
            T(
                "Spatial scale (radius, in pixels) used to decide what is an autofluorescent "
                "compartment.\nIt is not the \"background correction scale\" above: that one is "
                "the background at object level, this one is the size of a compartment.\nBroad "
                "continuous bands of high background can use a larger value; thin bright bands "
                "only a few dozen pixels wide need a smaller one, otherwise the decision blurs "
                "and swallows normal areas around them (this preset defaults to 14 px).\n0 "
                "derives the scale from the background correction scale.",
            )
        )
        self.background_ceiling_spin = self._double_spin(
            0.0, 1_000_000.0, 0.0, 100.0, 1, ""
        )
        self.background_ceiling_spin.setSpecialValueText(T("Off"))
        self.background_ceiling_spin.setToolTip(
            T(
                "Absolute ceiling of the broad-scale local background, in grey levels of the "
                "source image.\nRegions whose background is brighter than this value are excluded "
                "as a whole before thresholding; this removes \"uniformly glowing\" compartments "
                "such as autofluorescent bands, folds or thickened edges.\nThe criterion is how "
                "bright the compartment of an object is, not the object's rank within the image, "
                "so it also requires the whole batch to share the same grey scale.",
            )
        )
        for label, widget in (
            (T("Target shape profile"), self.shape_profile_combo),
            (T("Shape matching strictness"), self.shape_profile_strictness_combo),
            (T("Minimum mean thickness"), self.min_thickness_spin),
            (T("Maximum mean thickness"), self.max_thickness_spin),
            (T("Minimum circularity"), self.min_circularity_spin),
            (T("Minimum solidity"), self.min_solidity_spin),
            (T("Maximum eccentricity"), self.max_eccentricity_spin),
            (T("Maximum aspect ratio"), self.max_aspect_ratio_spin),
            (T("Minimum mean intensity"), self.min_mean_intensity_spin),
            (T("Minimum object response confidence"), self.min_response_confidence_spin),
            (T(
                "Shared noise scale (absolute threshold for the batch)",
            ), self.noise_reference_value_spin),
            (T("Exclusion of autofluorescent compartments"), self.background_ceiling_mode_combo),
            (T("Compartment scale"), self.background_ceiling_scale_spin),
            (T(
                "Background brightness ceiling (absolute grey level)",
            ), self.background_ceiling_spin),
        ):
            form.addRow(label, widget)
            self._connect_parameter_widget(widget)
        self.shape_compatibility_label = QLabel(page)
        self.shape_compatibility_label.setObjectName("mutedText")
        self.shape_compatibility_label.setWordWrap(True)
        form.addRow(self.shape_compatibility_label)
        return page

    def _spin(
        self,
        minimum: int,
        maximum: int,
        value: int,
        *,
        step: int = 1,
        suffix: str = "",
    ) -> QSpinBox:
        widget = QSpinBox(self)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setSuffix(suffix)
        widget.setKeyboardTracking(False)
        return widget

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
        suffix: str,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox(self)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        widget.setValue(value)
        widget.setSuffix(suffix)
        widget.setKeyboardTracking(False)
        return widget

    def _make_combo(self, choices: dict[str, str]) -> QComboBox:
        combo = FlexibleComboBox(self)
        for key, label in choices.items():
            combo.addItem(label, key)
        return combo

    def _connect_parameter_widget(self, widget: QWidget) -> None:
        if isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._on_parameter_changed)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(self._on_parameter_changed)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self._on_parameter_changed)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #11161d;
                color: #e8eef5;
                font-size: 13px;
            }
            QToolBar#mainToolbar {
                background: #171e27;
                border: none;
                border-bottom: 1px solid #2b3542;
                spacing: 2px;
                padding: 1px 6px;
            }
            QToolBar#mainToolbar::separator { width: 1px; background: #2b3542; margin: 4px 4px; }
            QToolButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 5px;
                padding: 4px 8px;
            }
            QToolButton:hover { background: #26313e; border-color: #354454; }
            QToolButton:checked { background: #174f42; border-color: #2fb98b; }
            QToolButton:disabled { color: #65717e; }
            QFrame#canvasToolStrip {
                background: rgba(14, 19, 26, 218);
                border: 1px solid rgba(88, 108, 128, 170);
                border-radius: 8px;
            }
            QFrame#canvasToolStrip QLabel { background: transparent; }
            QFrame#canvasToolSeparator {
                background: transparent;
                border: none;
                border-left: 1px solid rgba(88, 108, 128, 170);
                margin: 2px 3px;
            }
            QToolButton#canvasTool {
                padding: 3px 5px;
                font-size: 12px;
                color: #e3ebf3;
            }
            QToolButton#canvasTool:checked { background: #1b6a53; border-color: #47d3a2; color: #ffffff; }
            QToolButton#canvasTool:disabled { color: #5f6c79; }
            QLabel#canvasStripLabel { color: #a9b8c6; font-size: 11px; padding: 0 2px; }
            QSpinBox#canvasSpin { padding: 1px 3px; min-height: 18px; font-size: 12px; }
            QLabel#canvasHint {
                background: rgba(14, 19, 26, 190);
                border: 1px solid rgba(88, 108, 128, 120);
                border-radius: 6px;
                color: #a3b2c0;
                font-size: 11px;
                padding: 3px 8px;
            }
            QToolButton#compactToggle {
                background: #18212b;
                border: 1px solid #354150;
                border-radius: 4px;
                padding: 3px 7px;
                font-size: 12px;
                color: #c9d5e0;
            }
            QToolButton#compactToggle:hover { background: #263747; border-color: #4d6579; }
            QToolButton#compactToggle:checked { background: #17634f; border-color: #59ddb0; color: #ffffff; }
            QToolButton#compactToggle:disabled { color: #65717e; background: #1b222b; }
            QToolButton#headerToggle {
                background: transparent;
                border: 1px solid #324151;
                border-radius: 4px;
                padding: 2px 7px;
                font-size: 12px;
                color: #b9c8d6;
            }
            QToolButton#headerToggle:hover { background: #202b36; }
            QToolButton#headerToggle:checked { background: #1d2833; border-color: #4d6579; color: #e3ebf3; }
            QLineEdit#queueFilterEdit { padding: 4px 6px; }
            QLabel#filterSummary { color: #8fe0be; font-size: 11px; padding: 0 2px; }
            QFrame#collapsibleCard {
                background: #111820;
                border: 1px solid #2b3542;
                border-radius: 7px;
            }
            QFrame#sidePanel { background: #151b23; }
            QLabel#sectionTitle { font-size: 15px; font-weight: 700; color: #ffffff; }
            QLabel#mutedText { color: #93a1af; font-size: 12px; }
            QLabel#countBadge {
                background: #23313d;
                border: 1px solid #354657;
                border-radius: 9px;
                color: #b9c8d6;
                font-size: 11px;
                padding: 2px 7px;
            }
            QLabel#compactStatus {
                background: #111820;
                border: 1px solid #2b3947;
                border-radius: 5px;
                color: #aebdcc;
                font-size: 12px;
                padding: 5px 7px;
            }
            QLabel#choiceFieldLabel {
                color: #dce7f2;
                font-weight: 650;
                padding-top: 3px;
            }
            QRadioButton[choiceButton="true"], QCheckBox[choiceButton="true"] {
                background: #18212b;
                border: 2px solid #354150;
                border-radius: 6px;
                color: #d7e0e9;
                padding: 5px 8px;
                font-weight: 500;
            }
            QRadioButton[choiceButton="true"]:hover, QCheckBox[choiceButton="true"]:hover {
                background: #263747;
                border-color: #4d6579;
            }
            QRadioButton[choiceButton="true"]:checked, QCheckBox[choiceButton="true"]:checked {
                background: #17634f;
                border-color: #59ddb0;
                color: #ffffff;
                font-weight: 700;
            }
            QRadioButton[choiceButton="true"]:focus, QCheckBox[choiceButton="true"]:focus {
                border-color: #9ce9d0;
            }
            QRadioButton[choiceButton="true"]:disabled, QCheckBox[choiceButton="true"]:disabled {
                background: #171d24;
                border-color: #2b3540;
                color: #697681;
            }
            QRadioButton[choiceButton="true"]:checked:disabled, QCheckBox[choiceButton="true"]:checked:disabled {
                background: #23483f;
                border-color: #52756c;
                color: #aabdb7;
            }
            QRadioButton[choiceButton="true"]::indicator {
                width: 12px;
                height: 12px;
                border: 1px solid #71808f;
                border-radius: 7px;
                background: #0f141b;
            }
            QRadioButton[choiceButton="true"]::indicator:checked {
                border: 3px solid #9ce9d0;
                background: #2fb98b;
            }
            QCheckBox[choiceButton="true"]::indicator {
                width: 12px;
                height: 12px;
                border: 1px solid #71808f;
                border-radius: 3px;
                background: #0f141b;
            }
            QCheckBox[choiceButton="true"]::indicator:checked {
                border: 3px solid #9ce9d0;
                background: #2fb98b;
            }
            QLabel#infoCard, QLabel#statsCard {
                background: #0f141b;
                border: 1px solid #2b3542;
                border-radius: 7px;
                padding: 7px 9px;
                color: #cbd6e2;
            }
            QLabel#infoCard { font-size: 12px; padding: 5px 8px; }
            QLabel#statsSummary {
                background: #0f141b;
                border: 1px solid #2b3542;
                border-radius: 6px;
                padding: 5px 8px;
                color: #cbd6e2;
                font-size: 12px;
            }
            QLabel#analysisBadge {
                background: #24303b;
                border: 1px solid #354454;
                border-radius: 6px;
                padding: 5px;
                font-weight: 600;
            }
            QLabel#analysisContext {
                color: #9fb0bf;
                font-size: 11px;
                padding: 0 3px;
            }
            QScrollArea#leftControlScroll, QWidget#leftControlBody {
                background: transparent;
                border: none;
            }
            QGroupBox {
                border: 1px solid #2b3542;
                border-radius: 7px;
                margin-top: 10px;
                padding-top: 6px;
                font-weight: 650;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 9px;
                padding: 0 5px;
                color: #dce7f2;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #0f141b;
                border: 1px solid #354150;
                border-radius: 5px;
                padding: 4px 7px;
                selection-background-color: #23795f;
            }
            QComboBox, QSpinBox, QDoubleSpinBox { min-height: 20px; }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                border: none;
                width: 22px;
            }
            QComboBox::down-arrow { image: url(fa:down.png); width: 10px; height: 10px; }
            QComboBox::down-arrow:disabled { image: url(fa:down_off.png); }
            QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
                color: #65717e;
                border-color: #2b3540;
                background: #131920;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #47d3a2; }
            QSpinBox, QDoubleSpinBox { padding-right: 20px; }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                border: none;
                border-left: 1px solid #2b3542;
                border-top-right-radius: 4px;
                margin: 1px 1px 0 0;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                border: none;
                border-left: 1px solid #2b3542;
                border-bottom-right-radius: 4px;
                margin: 0 1px 1px 0;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #26313e; }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(fa:up.png); width: 8px; height: 8px; }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(fa:down.png); width: 8px; height: 8px; }
            QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
            QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off { image: url(fa:up_off.png); }
            QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off,
            QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off { image: url(fa:down_off.png); }
            QComboBox QAbstractItemView {
                background: #0f141b;
                border: 1px solid #354150;
                selection-background-color: #1c6652;
                padding: 2px;
            }
            QTreeWidget {
                background: #0f141b;
                border: 1px solid #354150;
                border-radius: 5px;
                padding: 2px;
                outline: none;
                font-size: 12px;
            }
            QTreeWidget::item { padding: 2px 3px; border-radius: 3px; }
            QTreeWidget::item:hover { background: #182634; }
            QTreeWidget::item:selected { background: #1c6652; color: #ffffff; }
            QTreeWidget::item:selected:active { background: #23795f; }
            QTreeWidget::branch:hover { background: transparent; }
            QPushButton {
                background: #26313e;
                border: 1px solid #3b4959;
                border-radius: 5px;
                padding: 6px 9px;
            }
            QPushButton:hover { background: #314051; }
            QPushButton:disabled { color: #65717e; background: #1b222b; }
            QPushButton#primaryButton {
                background: #208060;
                border-color: #32b88c;
                color: white;
                font-weight: 650;
            }
            QPushButton#primaryButton:hover { background: #26966f; }
            QPushButton#segmentedButton {
                background: #18212b;
                border-color: #354454;
                padding: 6px 5px;
            }
            QPushButton#segmentedButton:checked {
                background: #17634f;
                border-color: #59ddb0;
                color: #ffffff;
                font-weight: 650;
            }
            QPushButton#segmentedButton:disabled {
                background: #171d24;
                border-color: #2b3540;
                color: #697681;
            }
            QPushButton#sectionToggle {
                background: transparent;
                border-color: #324151;
                color: #b9c8d6;
                text-align: left;
            }
            QPushButton#sectionToggle:hover { background: #202b36; }
            QSlider::groove:horizontal { height: 5px; background: #2a3541; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 15px; margin: -5px 0; border-radius: 7px;
                background: #48c89b; border: 1px solid #9ce9d0;
            }
            QToolBox::tab {
                background: #1b232d;
                border: 1px solid #2b3542;
                border-radius: 5px;
                padding: 0 9px;
                font-weight: 600;
            }
            QToolBoxButton { min-height: 28px; }
            QToolBox::tab:selected { background: #24313d; color: #70d9b5; }
            QScrollBar:vertical { background: transparent; width: 11px; margin: 0; }
            QScrollBar::handle:vertical {
                background: #344251; min-height: 28px; border-radius: 4px; margin: 1px 2px;
            }
            QScrollBar:horizontal { background: transparent; height: 11px; margin: 0; }
            QScrollBar::handle:horizontal {
                background: #344251; min-width: 28px; border-radius: 4px; margin: 2px 1px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #4a5f73; }
            QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; background: none; }
            QScrollBar::add-page, QScrollBar::sub-page { background: none; }
            QAbstractScrollArea::corner { background: transparent; border: none; }
            QStatusBar { background: #0d1218; border-top: 1px solid #29333f; }
            QStatusBar::item { border: none; }
            QStatusBar QLabel { padding: 2px 8px; }
            QSplitter::handle { background: #27313d; width: 1px; }
            QSplitter::handle:hover { background: #3d8f77; }

            QLabel#noteCard {
                background: #151d26;
                border: 1px solid #2b3542;
                border-left: 3px solid #d29922;
                border-radius: 5px;
                color: #cbd6e2;
                font-size: 12px;
                padding: 6px 9px;
            }
            QFrame#emptyState {
                background: rgba(20, 27, 36, 235);
                border: 1px solid #354657;
                border-radius: 12px;
            }
            QFrame#emptyState QLabel { background: transparent; }
            QLabel#emptyTitle { font-size: 18px; font-weight: 700; color: #ffffff; }
            QLabel#aboutTitle { font-size: 21px; font-weight: 700; color: #ffffff; }
            QCheckBox, QRadioButton { spacing: 7px; }
            QCheckBox::indicator, QGroupBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5d6d7e;
                border-radius: 3px;
                background: #0f141b;
            }
            QCheckBox::indicator:hover, QGroupBox::indicator:hover { border-color: #8fa3b5; }
            QCheckBox::indicator:checked, QGroupBox::indicator:checked {
                background: #2fb98b;
                border-color: #59ddb0;
                image: url(fa:check.png);
            }
            QCheckBox::indicator:disabled, QGroupBox::indicator:disabled {
                border-color: #333e4a;
                background: #151b22;
            }
            QCheckBox::indicator:checked:disabled, QGroupBox::indicator:checked:disabled {
                background: #2b4a42;
                border-color: #3f6359;
                image: url(fa:check_off.png);
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5d6d7e;
                border-radius: 8px;
                background: #0f141b;
            }
            QRadioButton::indicator:hover { border-color: #8fa3b5; }
            QRadioButton::indicator:checked {
                width: 8px;
                height: 8px;
                border: 4px solid #2fb98b;
                background: #ffffff;
            }
            QRadioButton::indicator:disabled { border-color: #333e4a; background: #151b22; }
            QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled { color: #65717e; }
            QCheckBox[choiceButton="true"]::indicator:checked { image: none; }
            QGroupBox::indicator { margin-left: 2px; }

            QMenuBar { background: #11161d; border-bottom: 1px solid #1f2731; padding: 1px 4px; }
            QMenuBar::item { background: transparent; padding: 4px 9px; border-radius: 4px; }
            QMenuBar::item:selected, QMenuBar::item:pressed { background: #26313e; }
            QMenu { background: #171e27; border: 1px solid #354150; padding: 4px; }
            QMenu::item { padding: 5px 26px 5px 24px; border-radius: 4px; }
            QMenu::item:selected { background: #1c6652; color: #ffffff; }
            QMenu::item:disabled { color: #65717e; }
            QMenu::separator { height: 1px; background: #2b3542; margin: 4px 6px; }
            QMenu::indicator { width: 12px; height: 12px; left: 7px; }
            QMenu::indicator:checked { image: url(fa:check.png); }

            QTableWidget, QTableView {
                background: #0f141b;
                alternate-background-color: #121a23;
                border: 1px solid #354150;
                border-radius: 5px;
                gridline-color: #232d38;
                selection-background-color: #1c6652;
                selection-color: #ffffff;
            }
            QHeaderView { background: #18212b; }
            QHeaderView::section {
                background: #18212b;
                color: #c9d5e0;
                border: none;
                border-right: 1px solid #2b3542;
                border-bottom: 1px solid #2b3542;
                padding: 4px 6px;
                font-weight: 600;
            }
            QTableCornerButton::section { background: #18212b; border: none; }
            QProgressBar {
                background: #0f141b;
                border: 1px solid #354150;
                border-radius: 5px;
                text-align: center;
                min-height: 16px;
            }
            QProgressBar::chunk { background: #2fb98b; border-radius: 4px; }
            QPushButton:pressed { background: #1f2933; }
            QPushButton:focus { border-color: #59ddb0; }
            QPushButton:default { border-color: #32b88c; }
            QToolButton:focus { border-color: #47d3a2; }
            """
        )

    # ---------- parameters and presets ----------
    @staticmethod
    def _preset_key(value: object) -> tuple[str, str] | None:
        """Normalise a preset key received from a UI signal or stored recipe."""
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None
        kind, name = value
        if not isinstance(kind, str) or not isinstance(name, str):
            return None
        return kind, name

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _quick_settings_from_controls(self) -> QuickSettings:
        payload = {
            name: str(choice.currentData() or "follow")
            for name, choice in self._quick_controls.items()
        }
        return quick_settings_from_dict(payload)

    def _set_quick_controls(self, settings: QuickSettings) -> None:
        previous = self._suppress_controls
        self._suppress_controls = True
        try:
            for name, value in quick_settings_to_dict(settings).items():
                self._quick_controls[name].setCurrentData(value)
        finally:
            self._suppress_controls = previous
        self._update_quick_dynamic_controls()

    @staticmethod
    def _set_quick_options_enabled(
        choice: ChoiceButtonGroup,
        values: Iterable[str],
        enabled: bool,
    ) -> None:
        for value in values:
            button = choice.buttonForData(value)
            if button is not None:
                button.setEnabled(enabled)

    def _update_quick_dynamic_controls(self) -> None:
        if not hasattr(self, "quick_shape_choice"):
            return
        shape = str(self.quick_shape_choice.currentData() or "follow")
        selected_shapes = quick_setting_values("shape", shape)
        base_shape = str(getattr(self._quick_base_params, "shape_profile", "any"))
        effective_shapes = (
            (base_shape,) if selected_shapes == ("follow",) else selected_shapes
        )
        all_fiber = effective_shapes == ("fiber",)
        mixed_fiber = "fiber" in effective_shapes and len(effective_shapes) > 1
        self.quick_size_label.setText(
            T("Fibre thickness (mean thickness)")
            if all_fiber
            else T("Target size / fibre thickness")
            if mixed_fiber
            else T("Target size")
        )
        self.quick_size_choice.setChoiceToolTip(
            T(
                "Strip-like targets are graded by their mean thickness after recognition, not "
                "limited by total length.",
            )
            if all_fiber
            else T(
                "Each shape branch is converted on its own: fibres by thickness, other objects by "
                "area.",
            )
            if mixed_fiber
            else T(
                "Converted by the short side of the current image; rotating the image does not "
                "change the scale.",
            )
        )
        shape_filter_enabled = any(
            item in {"round", "ellipse", "fiber", "patch", "ring"}
            for item in selected_shapes
        )
        self.quick_shape_strictness_choice.setEnabled(shape_filter_enabled)
        self.quick_shape_strictness_label.setText(
            T("Shape requirement")
            if shape_filter_enabled
            else T("Shape requirement (needs a specific shape)")
        )
        self.quick_shape_strictness_choice.setChoiceToolTip(
            T("Applies to all five specific shapes: round, elliptic, fibre, patch and ring.")
        )

        ring_mode = effective_shapes == ("ring",)
        continuous_mode = bool(effective_shapes) and set(effective_shapes).issubset(
            {"fiber", "patch", "ring"}
        )
        # Keep cross-field conflicts selectable so the plan card can explain
        # them.  The compiler blocks execution instead of silently unchecking
        # or rewriting the user's intent.
        self._set_quick_options_enabled(
            self.quick_holes_choice,
            ("small", "medium", "large"),
            True,
        )
        self._set_quick_options_enabled(
            self.quick_touching_choice,
            ("conservative", "balanced", "strong"),
            True,
        )
        self.quick_holes_label.setText(
            T("Inner holes (always kept in ring mode)")
            if ring_mode
            else T("Inner holes")
        )
        self.quick_holes_choice.setChoiceToolTip(
            T(
                "Ring targets must keep their central hole; if hole filling is chosen as well, "
                "the execution plan reports the conflict.",
            )
            if ring_mode
            else T("Controls whether small holes inside objects are filled.")
        )
        self.quick_touching_label.setText(
            T("Touching objects (continuous shapes stay whole)")
            if continuous_mode
            else T("Touching objects")
        )
        self.quick_touching_choice.setChoiceToolTip(
            T(
                "Fibres, patches and rings are continuous structures; if the watershed is chosen "
                "explicitly, the execution plan reports the conflict.",
            )
            if continuous_mode
            else T("Separates touching compact objects by a distance watershed.")
        )

        self.quick_edge_choice.setEnabled(True)
        self.quick_edge_label.setText(T("Boundary extent"))
        self.quick_edge_choice.setChoiceToolTip(
            T(
                "Controls how far clear cores extend into weaker edges; the local threshold for "
                "low contrast supports it as well.",
            )
        )
        if hasattr(self, "quick_scale_choice"):
            self.quick_scale_choice.setChoiceToolTip(
                T(
                    "Choose relative mode when the same frame was only rescaled as a whole; "
                    "choose fixed source pixels after cropping, stitching, a change of "
                    "magnification, or when objects keep their size in pixels.",
                )
            )

    def _update_parameter_mode_label(self) -> None:
        if not hasattr(self, "parameter_mode_label"):
            return
        plan = self._refresh_quick_plan_label()
        count = self._quick_settings.override_count
        if self._ai_mode():
            self.parameter_mode_label.setText(T(
                "The learned model has fixed parameters; the main selection can be added to or "
                "erased by hand, and the yellow review area is kept separately.",
            ))
            return
        if self._quick_plan_error:
            self.parameter_mode_label.setText(
                T(
                    "The current combination has conflicts and the old mask is void; fix the "
                    "choices shown in red in the execution plan before analysing or exporting.",
                )
            )
            return
        if self._parameter_source == "quick" and count:
            branches = len(plan.branches) if plan is not None else 1
            self.parameter_mode_label.setText(
                T("{0} quick option(s) enabled, {1} recognition branch(es); ", count, branches)
                + (
                    T(
                        "the final mask is the union of the branches and the professional panel "
                        "shows a representative branch. ",
                    )
                    if branches > 1
                    else T(
                        "the professional parameters were computed completely and automatically. ",
                    )
                )
                + T(
                    "Changing a professional parameter makes the branch on display the new "
                    "professional base.",
                )
            )
        elif self._parameter_source == "professional":
            self.parameter_mode_label.setText(
                T(
                    "Custom professional parameters are in use; the quick options all follow this "
                    "professional base.",
                )
            )
        elif self._parameter_source == "imported":
            self.parameter_mode_label.setText(
                T(
                    "An external professional recipe is loaded; choose any quick option to adjust "
                    "on top of it.",
                )
            )
        else:
            self.parameter_mode_label.setText(
                T(
                    "The chosen preset is used completely; quick options can override it item by "
                    "item.",
                )
            )

    def _refresh_quick_plan_label(self):
        """Compile once for visible diagnostics without rewriting user choices."""

        if not hasattr(self, "quick_plan_label"):
            return None
        if self._ai_mode():
            self._quick_plan_error = None
            self.quick_plan_label.setStyleSheet("")
            self.quick_plan_label.setText(T(
                "Learned-model mode: the classic quick settings take no part in recognition.",
            ))
            return None
        try:
            plan = compile_quick_plan(
                self._quick_base_params,
                self._quick_settings,
                self.loaded.shape if self.loaded is not None else None,
            )
        except (TypeError, ValueError) as exc:
            self._quick_plan_error = str(exc)
            self.quick_plan_label.setStyleSheet("color: #ff7b72;")
            self.quick_plan_label.setText(T("The execution plan cannot be built: {0}", exc))
            return None
        if plan.errors:
            self._quick_plan_error = T("; ").join(issue.message for issue in plan.errors)
            self.quick_plan_label.setStyleSheet("color: #ff7b72;")
            lines = [T("{0} conflict(s) found:", len(plan.errors))]
            lines.extend(f"• {issue.message}" for issue in plan.errors[:5])
            if len(plan.errors) > 5:
                lines.append(T("• {0} more", len(plan.errors) - 5))
            self.quick_plan_label.setText("\n".join(lines))
            return plan
        self._quick_plan_error = None
        derivations = tuple(
            dict.fromkeys(
                item
                for branch in plan.branches
                for item in branch.derivations
            )
        )
        self.quick_plan_label.setStyleSheet(
            "color: #d29922;" if plan.warnings else ""
        )
        lines = [
            T(
                "Execution plan: {0} distinct branch(es); each is recognized and the masks are "
                "united.",
                len(plan.branches),
            )
        ]
        if derivations:
            lines.append(T("Derived automatically: ") + T("; ").join(derivations[:3]))
        if plan.warnings:
            lines.append(T("Note: ") + T("; ").join(issue.message for issue in plan.warnings[:3]))
        self.quick_plan_label.setText("\n".join(lines))
        return plan

    def _on_quick_parameter_changed(self, *_args: object) -> None:
        if self._suppress_controls or self._ai_mode():
            return
        proposed = self._quick_settings_from_controls()
        plan = compile_quick_plan(
            self._quick_base_params,
            proposed,
            self.loaded.shape if self.loaded is not None else None,
        )
        if self.result is not None and self.result.manually_edited:
            if not self._confirm_discard_manual(T(
                "Changing the quick settings clears the current manual edits.",
            )):
                self._set_quick_controls(self._quick_settings)
                return
            self.result.manually_edited = False
            self._clear_manual_history()

        self._quick_settings = proposed
        self._parameter_source = "quick" if proposed.override_count else "professional"
        self._select_custom_preset()
        if plan.errors:
            self.preset_description.setStyleSheet("color: #ff7b72;")
            self.preset_description.setText(
                T(
                    "The current quick settings conflict; the program has not cleared or "
                    "overridden any choice for you.",
                )
            )
            self._invalidate_result_after_parameter_change()
            self._update_parameter_mode_label()
            self._update_action_state()
            return
        params = plan.branches[0].params
        branches = len(plan.branches)
        self._set_controls_from_params(params)
        description = (
            T("Quick settings: ")
            + quick_settings_summary(proposed)
            + (
                T(
                    ". {0} branches will run and their masks will be united; the professional "
                    "panel shows the first branch.",
                    branches,
                )
                if branches > 1
                else T(". The professional parameters were updated accordingly.")
            )
        )
        derivations = tuple(
            dict.fromkeys(item for branch in plan.branches for item in branch.derivations)
        )
        if derivations:
            description += T(" Derived automatically: ") + T("; ").join(derivations[:3]) + T(".")
        if plan.warnings:
            description += T(" Note: ") + T("; ").join(
                issue.message for issue in plan.warnings[:3]
            ) + T(".")
        self.preset_description.setStyleSheet("")
        self.preset_description.setText(description)
        self._update_parameter_mode_label()
        self._invalidate_result_after_parameter_change()
        if self.auto_check.isChecked() and self.loaded is not None:
            self.auto_timer.start()
        self._update_action_state()

    def _auto_preview_toggled(self, enabled: bool) -> None:
        if not enabled:
            self.auto_timer.stop()
            return
        if (
            self.loaded is not None
            and self.result is None
            and not self._quick_plan_error
            and not self._load_busy
            and not self._segment_busy
            and not self._manual_busy
        ):
            self.auto_timer.start()

    def _ai_mode(self) -> bool:
        return getattr(self, "_recognition_engine", "classic") in AI_RECOGNITION_ENGINES

    def _recognition_engine_changed(self, _index: int = -1) -> None:
        if self._suppress_controls:
            return
        engine = str(self.recognition_engine_combo.currentData() or "classic")
        if engine == self._recognition_engine:
            return
        if self.result is not None and self.result.manually_edited:
            if not self._confirm_discard_manual(T(
                "Switching the recognition mode clears the manual edits that have not been "
                "exported yet.",
            )):
                previous = self.recognition_engine_combo.blockSignals(True)
                self._set_combo_data(self.recognition_engine_combo, self._recognition_engine)
                self.recognition_engine_combo.blockSignals(previous)
                return
        self._recognition_engine = engine
        self._quick_base_params.recognition_engine = engine
        self._quick_plan_error = None
        self._reset_crosstalk_analysis_input()
        self._invalidate_result_after_parameter_change()
        self._sync_engine_controls()
        self._update_parameter_mode_label()
        self._update_action_state()
        if self.loaded is not None and self.auto_check.isChecked():
            self.auto_timer.start()

    def _update_learned_sensitivity_note(self) -> None:
        value = float(self.learned_sensitivity_spin.value())
        if abs(value) < 0.005:
            text = T("as calibrated")
        elif value > 0:
            text = T("accepts more candidates")
        else:
            text = T("stricter")
        self.learned_sensitivity_note.setText(text)

    def _learned_sensitivity_changed(self, *_args: object) -> None:
        self._update_learned_sensitivity_note()
        if self._suppress_controls:
            return
        if self.result is not None and self.result.manually_edited:
            if not self._confirm_discard_manual(T(
                "Changing the sensitivity clears the current manual edits.",
            )):
                previous = self._suppress_controls
                self._suppress_controls = True
                try:
                    self.learned_sensitivity_spin.setValue(
                        float(self.result.params.learned_sensitivity)
                    )
                finally:
                    self._suppress_controls = previous
                self._update_learned_sensitivity_note()
                return
            self.result.manually_edited = False
            self._clear_manual_history()
        self._quick_base_params.learned_sensitivity = float(self.learned_sensitivity_spin.value())
        if not self._ai_mode():
            return
        self._invalidate_result_after_parameter_change()
        self._update_action_state()
        if self.loaded is not None and self.auto_check.isChecked():
            self.auto_timer.start()

    def _sync_engine_controls(self) -> None:
        ai = self._ai_mode()
        if hasattr(self, "learned_sensitivity_row"):
            self.learned_sensitivity_row.setVisible(ai)
        if hasattr(self, "preset_description"):
            self._update_preset_description()
        if hasattr(self, "algorithm_label"):
            self.algorithm_label.setText(
                T("Learned model v6.3") if ai else T("Algorithm {0}", UNIVERSAL_ALGORITHM_VERSION)
            )
        if hasattr(self, "ai_mode_hint"):
            self.ai_mode_hint.setVisible(ai)
            self.ai_mode_hint.setText(
                T(
                    "Finds small targets that are clearly brighter than their immediate "
                    "background and lie in a quiet neighbourhood. Any image type, size and bit "
                    "depth; a colour image is read by its brightest channel. The "
                    "green main selection and the yellow review area are counted separately; "
                    "yellow marks candidates scoring close to the threshold. A reference channel "
                    "is optional and used to subtract crosstalk. The brightness and shape "
                    "parameters take no part in this mode.",
                )
            )
        if hasattr(self, "ai_review_check"):
            self.ai_review_check.setVisible(ai)
        if hasattr(self, "preview_button"):
            self.preview_button.setText(T("Update AI selection") if ai else T("Update preview"))
            self.preview_button.setToolTip(T(
                "The learned model always recognizes at the original resolution.",
            ) if ai else "")
        self._update_queue_summary()

    def _params_from_controls(self) -> UniversalParams:
        values: dict[str, object] = {}
        for field_name, attribute in _PROFESSIONAL_CONTROL_ATTRS.items():
            widget = getattr(self, attribute)
            if isinstance(widget, QComboBox):
                values[field_name] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                values[field_name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[field_name] = widget.value()
            else:
                raise TypeError(T("Unsupported professional parameter control: {0}", attribute))
        values["recognition_engine"] = self._recognition_engine
        values["learned_sensitivity"] = float(self.learned_sensitivity_spin.value())
        params = UniversalParams(**values)  # type: ignore[arg-type]
        return params.validated(self.loaded.shape if self.loaded is not None else None)

    def _set_controls_from_params(self, params: UniversalParams) -> None:
        params = params.validated(self.loaded.shape if self.loaded is not None else None)
        previous = self._suppress_controls
        self._suppress_controls = True
        try:
            self._recognition_engine = getattr(params, "recognition_engine", "classic")
            self._set_combo_data(self.recognition_engine_combo, self._recognition_engine)
            self.learned_sensitivity_spin.setValue(float(params.learned_sensitivity))
            self._update_learned_sensitivity_note()
            self.background_radius_spin.setMaximum(
                128 if params.background_method in {"vesselness", "log"} else 2048
            )
            for field_name, attribute in _PROFESSIONAL_CONTROL_ATTRS.items():
                widget = getattr(self, attribute)
                value = getattr(params, field_name)
                if isinstance(widget, QComboBox):
                    self._set_combo_data(widget, value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                else:
                    raise TypeError(T("Unsupported professional parameter control: {0}", attribute))
            # The slider is not part of the dataclass control map.  Keep it
            # explicitly aligned even when the spin box already held the target
            # value and therefore emitted no valueChanged signal.
            target_sensitivity = int(self.sensitivity_spin.value())
            slider_signals = self.sensitivity_slider.blockSignals(True)
            try:
                self.sensitivity_slider.setValue(target_sensitivity)
            finally:
                self.sensitivity_slider.blockSignals(slider_signals)
        finally:
            self._suppress_controls = previous
        self._update_dynamic_controls()
        self._sync_analysis_channel_from_signal(params.signal_mode)
        self._sync_engine_controls()

    def _on_parameter_changed(self, *_args: object) -> None:
        if self._suppress_controls or self._ai_mode():
            return
        self._update_dynamic_controls()
        if self.result is not None and self.result.manually_edited:
            if not self._confirm_discard_manual(T(
                "Changing a recognition parameter clears the current manual edits.",
            )):
                self._set_controls_from_params(self.result.params)
                return
            self.result.manually_edited = False
            self._clear_manual_history()
        current = self._params_from_controls()
        self._sync_analysis_channel_from_signal(current.signal_mode)
        self._quick_base_params = copy.deepcopy(current)
        self._quick_settings = quick_settings_from_dict({})
        self._set_quick_controls(self._quick_settings)
        self._parameter_source = "professional"
        self._select_custom_preset()
        self._update_parameter_mode_label()
        self._invalidate_result_after_parameter_change()
        if self.auto_check.isChecked() and self.loaded is not None:
            self.auto_timer.start()
        self._update_action_state()

    def _select_custom_preset(self) -> None:
        previous = self._suppress_controls
        self._suppress_controls = True
        try:
            self.preset_choices.setCurrentData(("custom", ""))
            self._applied_preset_key = ("custom", "")
            self.preset_description.setStyleSheet("")
            self.preset_description.setText(T(
                "The current parameters differ from the preset; they can be saved as a personal "
                "preset.",
            ))
        finally:
            self._suppress_controls = previous

    @staticmethod
    def _parse_user_preset_file(path: Path) -> dict[str, dict[str, object]] | None:
        """Parse one preset file, returning None only on read/structure failure.

        A valid-but-empty file returns {}, so an intentionally emptied library is
        not mistaken for corruption and does not trigger the .bak fallback.
        """

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        presets = payload.get("presets", {})
        if not isinstance(presets, dict):
            return None
        parsed: dict[str, dict[str, object]] = {}
        for name, item in presets.items():
            if not isinstance(name, str) or not isinstance(item, dict):
                continue
            params_payload = item.get("params")
            if isinstance(params_payload, dict):
                try:
                    params_from_dict(params_payload)
                except (TypeError, ValueError):
                    continue
                stored: dict[str, object] = {
                    "description": str(item.get("description", T("Personally saved preset"))),
                    "params": params_payload,
                }
                quick_payload = item.get("quick_settings")
                base_payload = item.get("base_params")
                if isinstance(quick_payload, dict) and isinstance(base_payload, dict):
                    try:
                        quick_settings_from_dict(quick_payload)
                        params_from_dict(base_payload)
                    except (TypeError, ValueError):
                        pass
                    else:
                        stored["quick_settings"] = quick_payload
                        stored["base_params"] = base_payload
                parsed[name] = stored
        return parsed

    def _load_user_presets(self) -> None:
        # Prefer the live file; fall back to the last-good .bak if it is missing
        # or corrupt, so a truncated/damaged file never silently wipes the library.
        self.user_presets = {}
        for candidate in (
            _USER_PRESET_FILE,
            _USER_PRESET_FILE.with_name(_USER_PRESET_FILE.name + ".bak"),
        ):
            if not candidate.is_file():
                continue
            parsed = self._parse_user_preset_file(candidate)
            if parsed is not None:
                self.user_presets = parsed
                return

    def _write_user_presets(self) -> None:
        _USER_PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 2, "presets": self.user_presets}
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        # Write to a sibling temp file then atomically replace, so an interrupted
        # save can never leave a half-written (corrupt) preset library behind.
        temporary = _USER_PRESET_FILE.with_name(_USER_PRESET_FILE.name + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(_USER_PRESET_FILE)
        # Mirror the just-committed good content to .bak for load-time recovery.
        try:
            _USER_PRESET_FILE.with_name(_USER_PRESET_FILE.name + ".bak").write_text(
                serialized, encoding="utf-8"
            )
        except OSError:
            pass

    def _refresh_preset_choices(self, selected: tuple[str, str] | None = None) -> None:
        current = self._preset_key(
            selected if selected is not None else self.preset_choices.currentData()
        ) or ("custom", "")
        choices: list[tuple[tuple[str, str], str, str]] = [
            (("custom", ""), T("Custom parameters"), "")
        ]
        choices.extend(
            (("builtin", key), T("{0} | {1}", T(preset.category), T(preset.name)), preset.category)
            for key, preset in BUILTIN_PRESETS.items()
        )
        choices.extend(
            (("user", name), T("Personal | {0}", name), "user")
            for name in sorted(self.user_presets, key=str.casefold)
        )
        valid_keys = {key for key, _label, _group in choices}
        if current not in valid_keys:
            current = ("custom", "")
        tooltips: dict[tuple[str, str], str] = {
            ("custom", ""): T(
                "Combine all parameters freely; a JSON recipe can be exported for reproduction or "
                "batch processing.",
            )
        }
        tooltips.update(
            (("builtin", key), T(preset.description))
            for key, preset in BUILTIN_PRESETS.items()
        )
        tooltips.update(
            (("user", name), str(item.get("description", T("Personal preset"))))
            for name, item in self.user_presets.items()
        )
        previous = self._suppress_controls
        self._suppress_controls = True
        try:
            self.preset_choices.setChoices(choices, tooltips)
            self.preset_choices.setCurrentData(current)
            self._applied_preset_key = current
        finally:
            self._suppress_controls = previous
        self._update_preset_description()
        self._update_action_state()

    def _update_preset_description(self) -> None:
        if self._ai_mode():
            self.preset_description.setStyleSheet("")
            self.preset_description.setText(
                T(
                    "Current recognition: {0}. The preset below takes effect only after switching "
                    "back to \"Classic recognition\".",
                    _ai_engine_label(self._recognition_engine),
                )
            )
            return
        data = self._preset_key(self.preset_choices.currentData())
        if data is None:
            self.preset_description.setStyleSheet("")
            self.preset_description.setText("")
            return
        kind, name = data
        if kind == "builtin" and name in BUILTIN_PRESETS:
            specialized = name in {"Spot enhancement", "Fibre ridge enhancement"}
            self.preset_description.setText(
                T("Current preset: {0}. {1}", T(name), T(BUILTIN_PRESETS[name].description))
            )
            self.preset_description.setStyleSheet(
                "color: #9a5b00; font-weight: 600;" if specialized else ""
            )
        elif kind == "user" and name in self.user_presets:
            self.preset_description.setStyleSheet("")
            self.preset_description.setText(
                T("Current preset: {0}. ", name)
                + str(self.user_presets[name].get("description", T("Personal preset")))
            )
        else:
            self.preset_description.setStyleSheet("")
            self.preset_description.setText(T(
                "Combine all parameters freely; a JSON recipe can be exported for reproduction or "
                "batch processing.",
            ))

    def _preset_selected(self, selected: object) -> None:
        if self._suppress_controls:
            return
        data = self._preset_key(selected)
        if data is None:
            self.preset_choices.setCurrentData(self._applied_preset_key)
            return
        kind, name = data
        if kind == "custom":
            self._applied_preset_key = data
            self._update_preset_description()
            self._update_action_state()
            return
        if self.result is not None and self.result.manually_edited:
            if not self._confirm_discard_manual(T(
                "Switching the preset clears the current manual edits.",
            )):
                self.preset_choices.setCurrentData(self._applied_preset_key)
                self._update_preset_description()
                return
            self._invalidate_result_after_parameter_change()
        loaded_quick: QuickSettings | None = None
        base_params: UniversalParams | None = None
        try:
            if kind == "builtin":
                params = copy.deepcopy(BUILTIN_PRESETS[str(name)].params)
            else:
                item = self.user_presets[str(name)]
                params = params_from_dict(item["params"])
                quick_payload = item.get("quick_settings")
                base_payload = item.get("base_params")
                if isinstance(quick_payload, dict) and isinstance(base_payload, dict):
                    loaded_quick = quick_settings_from_dict(quick_payload)
                    base_params = params_from_dict(base_payload)
                    loaded_quick, _ = normalize_quick_settings(
                        loaded_quick,
                        base_params.shape_profile,
                    )
                    params = resolve_quick_settings(
                        base_params,
                        loaded_quick,
                        self.loaded.shape if self.loaded is not None else None,
                    )
        except Exception as exc:
            self.preset_choices.setCurrentData(self._applied_preset_key)
            self._update_preset_description()
            QMessageBox.critical(self, T("Invalid preset"), _safe_error_message(exc))
            return
        self._quick_base_params = copy.deepcopy(base_params or params)
        self._quick_settings = loaded_quick or (
            QuickSettings() if kind == "builtin" else quick_settings_from_dict({})
        )
        self._parameter_source = (
            "quick" if self._quick_settings.override_count else "preset"
        )
        self._set_quick_controls(self._quick_settings)
        self._set_controls_from_params(params)
        self._applied_preset_key = data
        self.preset_choices.setCurrentData(data)
        self._update_preset_description()
        self._update_parameter_mode_label()
        self._update_action_state()
        if self.loaded is not None:
            self.auto_timer.stop()
            self._start_segmentation(full=False)

    def _save_user_preset(self) -> None:
        name, accepted = QInputDialog.getText(self, T("Save personal preset"), T("Preset name:"))
        name = name.strip()
        if not accepted or not name:
            return
        if name in BUILTIN_PRESETS:
            QMessageBox.warning(self, T("Name not available"), T(
                "This name is the same as a built-in preset; please choose another.",
            ))
            return
        if name in self.user_presets:
            answer = QMessageBox.question(
                self,
                T("Overwrite preset"),
                T("The personal preset \"{0}\" already exists. Overwrite it?", name),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        description, accepted_description = QInputDialog.getText(
            self,
            T("Preset description"),
            T("Optional description:"),
            text=T("Personally saved recognition preset"),
        )
        if not accepted_description:
            description = T("Personally saved recognition preset")
        stored: dict[str, object] = {
            "description": description.strip() or T("Personally saved recognition preset"),
            "params": params_to_dict(self._params_from_controls()),
        }
        if not self._ai_mode() and self._quick_settings.override_count:
            stored["quick_settings"] = quick_settings_to_dict(self._quick_settings)
            stored["base_params"] = params_to_dict(self._quick_base_params)
        self.user_presets[name] = stored
        try:
            self._write_user_presets()
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Saving failed"),
                _safe_error_message(exc, _USER_PRESET_FILE),
            )
            return
        self._refresh_preset_choices(selected=("user", name))
        self.status_label.setText(T("Personal preset saved: {0}", name))

    def _delete_user_preset(self) -> None:
        data = self._preset_key(self.preset_choices.currentData())
        if data is None or data[0] != "user":
            QMessageBox.information(self, T("Cannot delete"), T(
                "Built-in presets are read-only; choose a personal preset.",
            ))
            return
        name = str(data[1])
        answer = QMessageBox.question(
            self,
            T("Delete personal preset"),
            T("Delete \"{0}\"?", name),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.user_presets.pop(name, None)
        try:
            self._write_user_presets()
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Deleting failed"),
                _safe_error_message(exc, _USER_PRESET_FILE),
            )
            return
        self._refresh_preset_choices(selected=("custom", ""))

    def _import_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            T("Import recognition recipe"),
            "",
            T("JSON recipe (*.json);;All files (*)"),
        )
        if not path:
            return
        quick_fallback = False
        imported_crosstalk: CrosstalkCorrection | None = None
        try:
            name, description, params, quick_payload, quick_base = load_recipe_bundle(path)
            imported_crosstalk = load_recipe_crosstalk_settings(path)
            imported_quick = None
            if not _uses_learned_engine(params) and quick_payload is not None and quick_base is not None:
                try:
                    imported_quick = quick_settings_from_dict(quick_payload)
                except (TypeError, ValueError):
                    quick_fallback = True
            if imported_quick is not None and quick_base is not None:
                imported_quick, _ = normalize_quick_settings(
                    imported_quick,
                    quick_base.shape_profile,
                )
                params = resolve_quick_settings(
                    quick_base,
                    imported_quick,
                    self.loaded.shape if self.loaded is not None else None,
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Import failed"),
                _safe_error_message(exc, Path(path)),
            )
            return
        if self.result is not None and self.result.manually_edited:
            if not self._confirm_discard_manual(T(
                "Importing a recipe clears the current manual edits.",
            )):
                return
            self._invalidate_result_after_parameter_change()
        self._quick_base_params = copy.deepcopy(
            quick_base
            if imported_quick is not None and quick_base is not None
            else params
        )
        self._quick_settings = imported_quick or quick_settings_from_dict({})
        self._parameter_source = (
            "quick" if self._quick_settings.override_count else "imported"
        )
        self._set_quick_controls(self._quick_settings)
        self._set_controls_from_params(params)
        crosstalk_note = ""
        if imported_crosstalk is not None and not self._ai_mode():
            self._set_crosstalk_controls(imported_crosstalk)
            previous_group = self.crosstalk_group.blockSignals(True)
            self.crosstalk_group.setChecked(True)
            self.crosstalk_group.blockSignals(previous_group)
            self._accepted_crosstalk_enabled = True
            self._accepted_crosstalk_settings = imported_crosstalk
            self._reset_crosstalk_analysis_input()
            crosstalk_note = T(
                " The crosstalk coefficient and reference channel were restored; reference images "
                "are still paired per image.",
            )
        self._select_custom_preset()
        fallback_note = (
            T(
                " The quick settings were not compatible; the complete professional parameters of "
                "the recipe are used instead.",
            )
            if quick_fallback
            else ""
        )
        self.preset_description.setStyleSheet("")
        self.preset_description.setText(
            T("Imported: {0}. {1}{2}{3}", name, description, fallback_note, crosstalk_note)
        )
        self._update_parameter_mode_label()
        self._update_action_state()
        if self.loaded is not None:
            self.auto_timer.stop()
            self._start_segmentation(full=False)

    def _export_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            T("Export recognition recipe"),
            "FluoroAnts_recipe.json",
            T("JSON recipe (*.json)"),
        )
        if not path:
            return
        try:
            save_recipe(
                path,
                self._params_from_controls(),
                name=T("Custom recipe"),
                description=T("Exported by FluoroAnts Universal"),
                quick_settings=(
                    quick_settings_to_dict(self._quick_settings)
                    if not self._ai_mode() and self._quick_settings.override_count
                    else None
                ),
                quick_base_params=(
                    self._quick_base_params
                    if not self._ai_mode() and self._quick_settings.override_count
                    else None
                ),
                analysis_preprocessing=(
                    self._current_crosstalk_recipe_metadata()
                    if self.crosstalk_group.isChecked()
                    else None
                ),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Export failed"),
                _safe_error_message(exc, Path(path)),
            )
            return
        self.status_label.setText(T("Recipe exported: {0}", Path(path).name))

    # ---------- pre-analysis crosstalk correction ----------
    def _current_crosstalk_settings(self) -> CrosstalkCorrection:
        return CrosstalkCorrection(
            coefficient=float(self.crosstalk_coefficient_spin.value()),
            reference_signal_mode=str(
                self.crosstalk_reference_channel_combo.currentData() or "auto"
            ),
            preserve_reference_background=self.crosstalk_background_check.isChecked(),
        ).validated()

    def _current_crosstalk_recipe_metadata(self) -> dict[str, object]:
        settings = self._current_crosstalk_settings()
        target = self._current_source_path()
        reference = (
            self._crosstalk_reference_paths.get(target)
            if target is not None
            else None
        )
        return {
            "type": "crosstalk_subtraction",
            "version": 1,
            "formula": "target - coefficient * reference",
            "coefficient": settings.coefficient,
            "target_signal_mode_requested": self._params_from_controls().signal_mode,
            "reference_file_name": reference.name if reference is not None else None,
            "reference_signal_mode_requested": settings.reference_signal_mode,
            "preserve_reference_background": settings.preserve_reference_background,
            "reference_background_percentile": settings.background_percentile,
            "size_policy": "strict_same_shape_no_resize",
        }

    def _set_crosstalk_controls(self, settings: CrosstalkCorrection) -> None:
        previous_combo = self.crosstalk_reference_channel_combo.blockSignals(True)
        previous_spin = self.crosstalk_coefficient_spin.blockSignals(True)
        previous_background = self.crosstalk_background_check.blockSignals(True)
        try:
            index = self.crosstalk_reference_channel_combo.findData(
                settings.reference_signal_mode
            )
            if index >= 0:
                self.crosstalk_reference_channel_combo.setCurrentIndex(index)
            self.crosstalk_coefficient_spin.setValue(settings.coefficient)
            self.crosstalk_background_check.setChecked(
                settings.preserve_reference_background
            )
        finally:
            self.crosstalk_reference_channel_combo.blockSignals(previous_combo)
            self.crosstalk_coefficient_spin.blockSignals(previous_spin)
            self.crosstalk_background_check.blockSignals(previous_background)
        self._update_crosstalk_formula()

    def _update_crosstalk_formula(self) -> None:
        if not hasattr(self, "crosstalk_formula_label"):
            return
        coefficient = float(self.crosstalk_coefficient_spin.value())
        reference_term = (
            T("max(reference channel − reference background, 0)")
            if self.crosstalk_background_check.isChecked()
            else T("reference channel")
        )
        self.crosstalk_formula_label.setText(
            T(
                "Corrected input = current analysis channel − {0:.2f} × {1}; negative values are "
                "set to 0 before recognition.",
                coefficient,
                reference_term,
            )
        )

    def _update_queue_summary(self) -> None:
        if not hasattr(self, "queue_count_label"):
            return
        total = len(self.paths)
        visible = len(self._visible_paths)
        paired = sum(path in self._crosstalk_reference_paths for path in self.paths)
        filtered = bool(self._queue_filter_text.strip()) and total > 0
        self.queue_count_label.setText(
            T("{0:,} / {1:,} images", visible, total) if filtered else T("{0:,} images", total)
        )
        self.queue_count_label.setToolTip(
            (
                T(
                    "{0:,} image(s) in the queue; {1:,} shown with the current filter.\n",
                    total,
                    visible,
                )
                if filtered
                else T("{0:,} image(s) in the queue.\n", total)
            )
            + T("Crosstalk references paired: {0:,}/{1:,}", paired, total)
            if total
            else T("The queue is empty")
        )
        if hasattr(self, "queue_filter_summary"):
            if filtered:
                mode = T("as a list across folders") if self._queue_view_mode == "list" else T("as a tree by folder")
                self.queue_filter_summary.setText(
                    T(
                        "Filter \"{0}\": showing {1:,} / {2:,} image(s) {3}; a batch processes "
                        "only these {4:,} (in the current order).",
                        self._queue_filter_text.strip(),
                        visible,
                        total,
                        mode,
                        visible,
                    )
                )
                self.queue_filter_summary.setVisible(True)
            else:
                self.queue_filter_summary.setVisible(False)

    # ---------- queue order, filter and status ----------
    def _queue_paths(self) -> list[Path]:
        """Images a batch processes: the visible (filtered) rows, in order."""

        return list(self._visible_paths)

    def _queue_label_for_path(self, path: Path) -> str:
        """Folder-relative text used for filtering and path sorting."""

        return str(self._queue_relative_text.get(path) or path.name)

    def _queue_segments_for_path(self, path: Path, prefix_root: bool) -> list[str]:
        root = self._queue_import_roots.get(path, path.parent)
        try:
            parts = list(path.relative_to(root).parts)
        except ValueError:
            text = self._queue_display_text.get(path) or path.name
            parts = [part for part in str(text).split("/") if part]
        if not parts:
            parts = [path.name]
        if prefix_root:
            parts = [root.name or str(root)] + parts
        return parts

    def _queue_prefix_roots(self) -> bool:
        roots = {
            self._queue_import_roots.get(path, path.parent) for path in self.paths
        }
        return len(roots) > 1

    def _queue_status_rank(self, path: Path) -> int:
        state = self._queue_status.get(path, ("pending", "", 0.0))[0]
        try:
            return _QUEUE_STATUS_ORDER.index(state)
        except ValueError:
            return len(_QUEUE_STATUS_ORDER)

    def _apply_queue_sort(self) -> None:
        """Reorder ``self.paths`` by the chosen criterion.

        The displayed order is the processing order, so the sort lives on the
        canonical list rather than on the view.  In tree mode the folder
        structure is kept (files are ordered inside their folder); in list
        mode the criterion is applied across every folder at once.
        """

        key = self._queue_sort_key
        prefix_root = self._queue_prefix_roots()
        segments = {
            path: self._queue_segments_for_path(path, prefix_root) for path in self.paths
        }

        def folder_key(path: Path) -> tuple[tuple[object, ...], ...]:
            return tuple(_natural_key(part) for part in segments[path][:-1])

        def name_key(path: Path) -> tuple[object, ...]:
            return _natural_key(path.name)

        def value_key(path: Path) -> tuple[object, ...]:
            if key == "name":
                return (name_key(path),)
            if key == "mtime":
                return (self._queue_file_stats.get(path, (0, 0))[0],)
            if key == "size":
                return (self._queue_file_stats.get(path, (0, 0))[1],)
            if key == "status":
                return (self._queue_status_rank(path),)
            if key == "added":
                return (self._queue_added_order.get(path, 0),)
            return (folder_key(path), name_key(path))

        ordered = sorted(self.paths, key=lambda path: (folder_key(path), name_key(path)))
        ordered.sort(key=value_key, reverse=self._queue_sort_desc)
        if self._queue_view_mode != "list" and key != "path":
            # Stable: files keep their criterion order inside each folder.
            ordered.sort(key=folder_key)
        self.paths = ordered

    def _apply_queue_filter(self) -> None:
        text = self._queue_filter_text.strip()
        if not text:
            self._visible_paths = list(self.paths)
            return
        self._visible_paths = [
            path
            for path in self.paths
            if _queue_filter_matches(self._queue_label_for_path(path), text)
            or _queue_filter_matches(path.name, text)
        ]

    def _refresh_queue(self, *, keep_path: Path | None = None) -> None:
        """Re-sort, re-filter and rebuild the tree, keeping the current image."""

        self._apply_queue_sort()
        self._apply_queue_filter()
        current = keep_path
        if current is None and self.loaded is not None:
            current = self.loaded.path
        if current is None:
            row = self.queue_list.currentRow()
            if 0 <= row < len(self._visible_paths):
                current = self._visible_paths[row]
        current_row = (
            self._visible_paths.index(current)
            if current is not None and current in self._visible_paths
            else -1
        )
        self._rebuild_queue_tree(current_row=current_row)
        self._update_queue_summary()

    def _queue_filter_text_changed(self, text: str) -> None:
        self._queue_filter_text = str(text)
        self._queue_filter_timer.start()

    def _apply_queue_filter_now(self) -> None:
        self._queue_filter_timer.stop()
        if not self.paths:
            self._update_queue_summary()
            return
        self._refresh_queue()
        self._update_action_state()

    def _queue_sort_changed(self, *_args: object) -> None:
        if self._suppress_controls:
            return
        self._queue_sort_key = str(self.queue_sort_combo.currentData() or "path")
        self._queue_sort_desc = bool(self.queue_sort_direction.isChecked())
        self._refresh_queue_control_text()
        if self.paths:
            self._refresh_queue()
            self._update_action_state()

    def _queue_view_mode_changed(self, flat: bool) -> None:
        if self._suppress_controls:
            return
        self._queue_view_mode = "list" if flat else "tree"
        self._refresh_queue_control_text()
        if self.paths:
            self._refresh_queue()
            self._update_action_state()

    def _queue_status_tuple(self, path: Path) -> tuple[str, str, str]:
        state, detail, stamp = self._queue_status.get(path, ("pending", "", 0.0))
        label, color = _QUEUE_STATUS_LABELS.get(state, _QUEUE_STATUS_LABELS["pending"])
        tooltip = label
        if detail:
            tooltip += T(": {0}", detail)
        if stamp:
            tooltip += "\n" + time.strftime("%H:%M:%S", time.localtime(stamp))
        return label, color, tooltip

    def _set_queue_status(self, path: Path | None, state: str, detail: str = "") -> None:
        """Record what step ``path`` reached and show it in the status column."""

        if path is None or state not in _QUEUE_STATUS_LABELS:
            return
        path = Path(path)
        if path not in self._queue_added_order:
            return
        self._queue_status[path] = (state, str(detail), time.time())
        try:
            row = self._visible_paths.index(path)
        except ValueError:
            row = -1
        if row >= 0:
            self.queue_list.set_row_status(row, *self._queue_status_tuple(path))
            if state == "running":
                item = self.queue_list.item(row)
                if item is not None:
                    self.queue_list.scrollToItem(item)
        if self._queue_sort_key == "status" and self._batch_worker is None:
            # Re-sorting during a batch would move the row under the user's
            # eyes for every image; the order is refreshed when it finishes.
            self._refresh_queue()

    def _update_analysis_context_label(self) -> None:
        if not hasattr(self, "analysis_context_label"):
            return
        if self._ai_mode():
            self.analysis_context_label.setText(
                T("Learned model · green main / yellow review")
            )
            return
        target = self._current_source_path()
        if target is None:
            self.analysis_context_label.setText(T("Selection source: not analysed yet"))
            return
        channel = str(self.display_channel_combo.currentText() or T("current channel"))
        enabled = bool(self.crosstalk_group.isChecked())
        if not enabled:
            self.analysis_context_label.setText(T("Selection source: original image · {0}", channel))
            return
        reference = self._crosstalk_reference_paths.get(target)
        if reference is None:
            self.analysis_context_label.setText(T(
                "Selection source: waiting for the crosstalk reference image",
            ))
            return
        coefficient = float(self.crosstalk_coefficient_spin.value())
        state = T("corrected") if self._crosstalk_ready_for_params() else T("waiting for update")
        self.analysis_context_label.setText(
            T(
                "Selection source: {0} · {1} − {2:.2f}× {3}",
                state,
                channel,
                coefficient,
                reference.name,
            )
        )

    def _current_source_path(self) -> Path | None:
        if self.source_loaded is not None:
            return self.source_loaded.path
        if self.loaded is not None:
            return self.loaded.path
        return None

    def _queue_tooltip_for_path(self, path: Path) -> str:
        reference = self._crosstalk_reference_paths.get(path)
        if reference is None:
            return str(path)
        return T("{0}\nCrosstalk reference: {1}", path, reference)

    def _queue_segments(self) -> list[list[str]]:
        """Folder segments for every visible row, deepest last.

        The hierarchy comes from the path relative to the folder the user
        actually opened.  When several different import roots are in the
        queue their relative texts can collide (two folders both containing
        ``R13/…``), so in that case each row is prefixed with its own root
        name; the tooltip still carries the absolute path.
        """

        prefix_root = self._queue_prefix_roots()
        return [
            self._queue_segments_for_path(path, prefix_root)
            for path in self._visible_paths
        ]

    def _rebuild_queue_tree(self, *, current_row: int | None = None) -> None:
        if current_row is None:
            current_row = self.queue_list.currentRow()
        blocked = self.queue_list.blockSignals(True)
        try:
            self.queue_list.rebuild(
                self._queue_segments(),
                [self._queue_tooltip_for_path(path) for path in self._visible_paths],
                [self._queue_status_tuple(path) for path in self._visible_paths],
                current_row=current_row,
                flat=self._queue_view_mode == "list",
            )
        finally:
            self.queue_list.blockSignals(blocked)
        self._mark_active_queue_row()

    def _mark_active_queue_row(self) -> None:
        active = -1
        if self.loaded is not None:
            active = next(
                (
                    index
                    for index, candidate in enumerate(self._visible_paths)
                    if candidate == self.loaded.path
                ),
                -1,
            )
        self.queue_list.mark_active_row(active)

    def _refresh_queue_tooltip(self, path: Path) -> None:
        try:
            row = self._visible_paths.index(path)
        except ValueError:
            return
        item = self.queue_list.item(row)
        if item is not None:
            item.setToolTip(0, self._queue_tooltip_for_path(path))

    def _update_crosstalk_reference_state(self) -> None:
        if not hasattr(self, "crosstalk_reference_label"):
            return
        # Recipe import and self-check setup may change the check state while
        # signals are blocked; keep the visual expansion aligned regardless.
        self.crosstalk_body.setVisible(self.crosstalk_group.isChecked())
        target = self._current_source_path()
        reference = (
            self._crosstalk_reference_paths.get(target)
            if target is not None
            else None
        )
        if target is None:
            text = T("Current image: none loaded")
        elif reference is None:
            text = T("{0}: no reference image chosen", target.name)
        else:
            text = T("{0}\nSubtracting: {1}", target.name, reference.name)
        if self._crosstalk_error:
            text += T("\nStatus: {0}", self._crosstalk_error)
        elif self._crosstalk_busy:
            text += T("\nStatus: generating the corrected input…")
        elif self._crosstalk_analysis_key is not None:
            text += T("\nStatus: corrected input ready")
        self.crosstalk_reference_label.setText(text)
        self.crosstalk_reference_label.setToolTip(
            "" if reference is None else str(reference)
        )
        self.crosstalk_clear_button.setEnabled(reference is not None)
        self.crosstalk_batch_pair_button.setEnabled(bool(self.paths))
        paired = sum(path in self._crosstalk_reference_paths for path in self.paths)
        if not self.crosstalk_group.isChecked():
            pair_text = T("Off · the original image is analysed directly")
        elif not self.paths:
            pair_text = T("Enabled · add target images first")
        else:
            pair_text = T("Enabled · queue pairing {0:,}/{1:,}", paired, len(self.paths))
        self.crosstalk_pair_summary_label.setText(pair_text)
        self._update_queue_summary()
        self._update_analysis_context_label()
        self._update_view_mode_availability()

    def _discard_manual_for_crosstalk(self, message: str) -> bool:
        if self.result is None or not self.result.manually_edited:
            return True
        if not self._confirm_discard_manual(message):
            return False
        self.result.manually_edited = False
        self._clear_manual_history()
        return True

    def _reset_crosstalk_analysis_input(self) -> None:
        self.auto_timer.stop()
        self._crosstalk_serial += 1
        self._crosstalk_busy = False
        self._crosstalk_error = None
        self._pending_crosstalk_full = None
        self._crosstalk_analysis_key = None
        self._crosstalk_reference_loaded = None
        self.loaded = self.source_loaded
        self.analysis_display = self.base_display
        self.crosstalk_reference_display = None
        self._invalidate_result_after_parameter_change()
        self._update_crosstalk_reference_state()

    def _crosstalk_enabled_changed(self, enabled: bool) -> None:
        if self._suppress_controls:
            return
        enabled = bool(enabled)
        if not self._discard_manual_for_crosstalk(
            T("Turning crosstalk correction on or off clears the current manual edits.")
        ):
            previous = self.crosstalk_group.blockSignals(True)
            self.crosstalk_group.setChecked(self._accepted_crosstalk_enabled)
            self.crosstalk_group.blockSignals(previous)
            return
        self._accepted_crosstalk_enabled = enabled
        self._accepted_crosstalk_settings = self._current_crosstalk_settings()
        self._reset_crosstalk_analysis_input()
        self._update_crosstalk_formula()
        if self.source_loaded is not None:
            if enabled and self.source_loaded.path not in self._crosstalk_reference_paths:
                self.analysis_badge.setText(T(
                    "Crosstalk correction · waiting for the reference image",
                ))
                self.status_label.setText(T(
                    "Choose a crosstalk reference image of the same field for the current target "
                    "image",
                ))
            elif not self._quick_plan_error and self.auto_check.isChecked():
                self.auto_timer.start()
            elif enabled:
                self.analysis_badge.setText(T("Crosstalk settings changed · waiting to be applied"))
                self.status_label.setText(T("Click \"Update preview\" or turn on auto preview"))
        self._update_action_state()

    def _crosstalk_settings_changed(self, *_args: object) -> None:
        if self._suppress_controls:
            return
        self._update_crosstalk_formula()
        proposed = self._current_crosstalk_settings()
        if not self._discard_manual_for_crosstalk(
            T("Changing the crosstalk correction settings clears the current manual edits.")
        ):
            self._set_crosstalk_controls(self._accepted_crosstalk_settings)
            return
        self._accepted_crosstalk_settings = proposed
        if not self.crosstalk_group.isChecked():
            return
        self._reset_crosstalk_analysis_input()
        if (
            self.source_loaded is not None
            and self.source_loaded.path in self._crosstalk_reference_paths
            and not self._quick_plan_error
            and self.auto_check.isChecked()
        ):
            self.auto_timer.start()
        elif self.source_loaded is not None:
            self.analysis_badge.setText(T("Crosstalk settings changed · waiting to be applied"))
            self.status_label.setText(T("Click \"Update preview\" or turn on auto preview"))
        self._update_action_state()

    def _choose_crosstalk_reference(self) -> None:
        target = self._current_source_path()
        if target is None:
            QMessageBox.information(
                self,
                T("No target image loaded"),
                T("Open the target image to be corrected in the queue first."),
            )
            return
        initial = str(target.parent)
        value, _ = QFileDialog.getOpenFileName(
            self,
            T("Choose the crosstalk reference image of the same field"),
            initial,
            _IMAGE_FILTER,
        )
        if not value:
            return
        reference = Path(value).expanduser().resolve()
        if reference == target:
            QMessageBox.warning(self, T("Cannot use the same file"), T(
                "A target image cannot be its own crosstalk reference image.",
            ))
            return
        if not self._discard_manual_for_crosstalk(
            T("Replacing the crosstalk reference image clears the current manual edits.")
        ):
            return
        self._crosstalk_reference_paths[target] = reference
        self._refresh_queue_tooltip(target)
        self._reset_crosstalk_analysis_input()
        self._update_crosstalk_reference_state()
        if (
            self.crosstalk_group.isChecked()
            and not self._quick_plan_error
            and self.auto_check.isChecked()
        ):
            self.auto_timer.start()
        elif self.crosstalk_group.isChecked():
            self.analysis_badge.setText(T("Crosstalk reference set · waiting to be applied"))
            self.status_label.setText(T("Click \"Update preview\" to generate the corrected input"))
        self._update_action_state()

    def _clear_crosstalk_reference(self) -> None:
        target = self._current_source_path()
        if target is None or target not in self._crosstalk_reference_paths:
            return
        if not self._discard_manual_for_crosstalk(
            T("Clearing the crosstalk reference image clears the current manual edits.")
        ):
            return
        self._crosstalk_reference_paths.pop(target, None)
        self._refresh_queue_tooltip(target)
        self._reset_crosstalk_analysis_input()
        self.analysis_badge.setText(T("Crosstalk correction · waiting for the reference image"))
        self.status_label.setText(T(
            "No crosstalk reference image is set for the current target image",
        ))
        self._update_action_state()

    def _pair_crosstalk_references_by_folder(self) -> None:
        if not self.paths:
            QMessageBox.information(self, T("The queue is empty"), T("Add the target images to the queue first."))
            return
        initial = str(self.paths[0].parent)
        value = QFileDialog.getExistingDirectory(
            self,
            T("Choose the folder that holds the crosstalk reference images (subfolders included)"),
            initial,
        )
        if not value:
            return
        reference_root = Path(value).expanduser().resolve()
        try:
            candidates = supported_image_files(reference_root, recursive=True)
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Reading the reference folder failed"),
                _safe_error_message(exc, reference_root),
            )
            return
        candidate_set = {candidate.resolve() for candidate in candidates}

        proposed: dict[Path, Path] = {}
        used_references: set[Path] = set()
        unresolved: list[Path] = []
        conflicts: list[Path] = []
        for target in self.paths:
            relative_text = self._queue_relative_text.get(target, target.name)
            exact = (reference_root / Path(relative_text)).resolve()
            if exact not in candidate_set or exact == target:
                unresolved.append(target)
            elif exact in used_references:
                conflicts.append(target)
            else:
                proposed[target] = exact
                used_references.add(exact)

        if not self._confirm_crosstalk_pairing_plan(
            reference_root,
            proposed,
            unresolved,
            conflicts,
        ):
            return

        current = self._current_source_path()
        current_changes = bool(
            current is not None
            and self._crosstalk_reference_paths.get(current) != proposed.get(current)
        )
        if current_changes and not self._discard_manual_for_crosstalk(
            T(
                "Updating the crosstalk reference pairing in a batch clears the current manual "
                "edits.",
            )
        ):
            return
        # One confirmed folder plan atomically replaces every queue mapping.
        # Otherwise an unresolved row could silently retain a reference from a
        # previously chosen folder even though the confirmation table says it
        # is not paired.
        for target in self.paths:
            self._crosstalk_reference_paths.pop(target, None)
        self._crosstalk_reference_paths.update(proposed)
        for target in self.paths:
            self._refresh_queue_tooltip(target)
        if current_changes:
            self._reset_crosstalk_analysis_input()
            if (
                self.crosstalk_group.isChecked()
                and not self._quick_plan_error
                and self.auto_check.isChecked()
            ):
                self.auto_timer.start()
            elif self.crosstalk_group.isChecked():
                self.analysis_badge.setText(T("Crosstalk pairing updated · waiting to be applied"))
                self.status_label.setText(T(
                    "Click \"Update preview\" to use the new batch pairing",
                ))
        self._update_crosstalk_reference_state()
        details = [T("Paired: {0:,} image(s)", len(proposed))]
        if unresolved:
            details.append(T(
                "No reference at the same relative path: {0:,} image(s)",
                len(unresolved),
            ))
        if conflicts:
            details.append(T(
                "Reference already used by another target: {0:,} image(s)",
                len(conflicts),
            ))
        if unresolved or conflicts:
            details.append(T(
                "With crosstalk correction on, unpaired items are never analysed silently as "
                "uncorrected originals in a batch.",
            ))
        QMessageBox.information(self, T("Batch pairing of crosstalk references"), "\n".join(details))
        self._update_action_state()

    def _confirm_crosstalk_pairing_plan(
        self,
        reference_root: Path,
        proposed: Mapping[Path, Path],
        unresolved: Iterable[Path],
        conflicts: Iterable[Path],
    ) -> bool:
        unresolved_set = set(unresolved)
        conflict_set = set(conflicts)
        dialog = QDialog(self)
        dialog.setWindowTitle(T("Confirm the crosstalk reference pairing"))
        dialog.resize(920, 620)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            T(
                "The program pairs one to one by the relative path inside the imported target "
                "folder only; it does not guess from sort order or similar file names. Confirm "
                "that every row really shows the same field.",
            ),
            dialog,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QTableWidget(len(self.paths), 3, dialog)
        table.setHorizontalHeaderLabels((T("Target image"), T("Crosstalk reference image"), T("Status")))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        for row, target in enumerate(self.paths):
            target_item = QTableWidgetItem(
                self._queue_display_text.get(target, target.name)
            )
            target_item.setToolTip(str(target))
            reference = proposed.get(target)
            if reference is not None:
                try:
                    reference_text = reference.relative_to(reference_root).as_posix()
                except ValueError:
                    reference_text = reference.name
                status = T("Paired")
            elif target in conflict_set:
                reference_text = "—"
                status = T("Reference used twice, not paired")
            elif target in unresolved_set:
                reference_text = "—"
                status = T("No file at this relative path")
            else:
                reference_text = "—"
                status = T("Not paired")
            reference_item = QTableWidgetItem(reference_text)
            if reference is not None:
                reference_item.setToolTip(str(reference))
            status_item = QTableWidgetItem(status)
            if reference is None:
                status_item.setForeground(QColor("#ff7b72"))
            table.setItem(row, 0, target_item)
            table.setItem(row, 1, reference_item)
            table.setItem(row, 2, status_item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, 1)
        summary = QLabel(
            T(
                "{0:,} pair(s) can be applied; {1:,} item(s) are not paired.",
                len(proposed),
                len(unresolved_set) + len(conflict_set),
            ),
            dialog,
        )
        layout.addWidget(summary)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            dialog,
        )
        apply_button = buttons.button(QDialogButtonBox.Ok)
        apply_button.setText(T("Apply the confirmed pairs"))
        apply_button.setEnabled(bool(proposed))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        return dialog.exec() == QDialog.Accepted

    def _crosstalk_analysis_key_for(
        self,
        params: UniversalParams,
        reference_path: Path,
    ) -> tuple[object, ...]:
        assert self.source_loaded is not None
        settings = self._current_crosstalk_settings()
        target_signal_key: tuple[object, ...] = (params.signal_mode,)
        if params.signal_mode == "custom_rgb":
            target_signal_key += (
                float(params.red_weight),
                float(params.green_weight),
                float(params.blue_weight),
            )
        return (
            self.source_loaded.path,
            reference_path,
            target_signal_key,
            settings.coefficient,
            settings.reference_signal_mode,
            settings.preserve_reference_background,
            settings.background_percentile,
        )

    def _crosstalk_ready_for_params(
        self,
        params: UniversalParams | None = None,
    ) -> bool:
        """Return true only when the cached subtraction matches current inputs."""

        if self._ai_mode() or not self.crosstalk_group.isChecked():
            return True
        if (
            self.source_loaded is None
            or self.loaded is None
            or self._crosstalk_analysis_key is None
        ):
            return False
        reference_path = self._crosstalk_reference_paths.get(self.source_loaded.path)
        preprocessing = self.loaded.metadata.get("analysis_preprocessing")
        if (
            reference_path is None
            or not isinstance(preprocessing, Mapping)
            or preprocessing.get("type") != "crosstalk_subtraction"
            or not str(self.loaded.metadata.get("analysis_input_signature", ""))
        ):
            return False
        try:
            current_params = params or self._params_from_controls()
            expected = self._crosstalk_analysis_key_for(current_params, reference_path)
        except (TypeError, ValueError):
            return False
        return tuple(self._crosstalk_analysis_key) == expected

    def _ensure_crosstalk_analysis_input(
        self,
        params: UniversalParams,
        *,
        full: bool,
    ) -> bool:
        if self.source_loaded is None:
            return False
        if self._ai_mode() or not self.crosstalk_group.isChecked():
            self.loaded = self.source_loaded
            self.analysis_display = self.base_display
            return True
        reference_path = self._crosstalk_reference_paths.get(self.source_loaded.path)
        if reference_path is None:
            self._crosstalk_error = T("No reference image chosen")
            self.analysis_badge.setText(T("Crosstalk correction · waiting for the reference image"))
            self.status_label.setText(T(
                "Choose the crosstalk reference image of the same field first",
            ))
            self._update_crosstalk_reference_state()
            self._update_action_state()
            return False
        analysis_key = self._crosstalk_analysis_key_for(params, reference_path)
        if self._crosstalk_ready_for_params(params):
            return True
        if self._crosstalk_busy:
            self._pending_crosstalk_full = bool(
                full or self._pending_crosstalk_full
            )
            return False

        self._crosstalk_serial += 1
        serial = self._crosstalk_serial
        self._crosstalk_busy = True
        self._crosstalk_error = None
        self._pending_crosstalk_full = bool(full)
        self.analysis_badge.setText(T("Generating the crosstalk-corrected input…"))
        self.status_label.setText(
            T("Subtracting the reference image: {0}", reference_path.name)
        )
        cached_reference = self._crosstalk_reference_loaded
        if cached_reference is not None and cached_reference.path != reference_path:
            cached_reference = None
        worker = CrosstalkWorker(
            self.load_token,
            serial,
            self.source_loaded,
            reference_path,
            params,
            self._current_crosstalk_settings(),
            analysis_key,
            cached_reference=cached_reference,
        )
        worker.signals.result.connect(self._on_crosstalk_result)
        worker.signals.error.connect(self._on_crosstalk_error)
        worker.signals.finished.connect(self._on_crosstalk_finished)
        self.thread_pool.start(worker)
        self._update_crosstalk_reference_state()
        self._update_action_state()
        return False

    @Slot(object)
    def _on_crosstalk_result(self, payload: object) -> None:
        (
            token,
            serial,
            corrected,
            reference,
            corrected_display,
            corrected_scale,
            reference_display,
            reference_scale,
            analysis_key,
        ) = payload  # type: ignore[misc]
        if int(token) != self.load_token or int(serial) != self._crosstalk_serial:
            return
        self._crosstalk_busy = False
        self._crosstalk_error = None
        self.loaded = corrected
        self._crosstalk_reference_loaded = reference
        self.analysis_display = corrected_display
        self.display_scale = float(corrected_scale)
        self.crosstalk_reference_display = reference_display
        self._crosstalk_analysis_key = tuple(analysis_key)
        pending_full = bool(self._pending_crosstalk_full)
        self._pending_crosstalk_full = None
        self._update_crosstalk_reference_state()
        self._render_display()
        self.status_label.setText(
            T("Crosstalk-corrected input generated: {0} subtracted", reference.path.name)
        )
        self._update_action_state()
        if not self._quick_plan_error:
            self._start_segmentation(full=pending_full)

    @Slot(object)
    def _on_crosstalk_error(self, payload: object) -> None:
        token, serial, details = payload  # type: ignore[misc]
        if int(token) != self.load_token or int(serial) != self._crosstalk_serial:
            return
        self._crosstalk_busy = False
        self._pending_crosstalk_full = None
        self._crosstalk_analysis_key = None
        self._crosstalk_error = str(details)
        self.loaded = self.source_loaded
        self.analysis_display = self.base_display
        self.result = None
        self.analysis_badge.setText(T("Crosstalk correction failed"))
        self.status_label.setText(T(
            "No uncorrected analysis was run; fix the reference image or the settings",
        ))
        self._update_crosstalk_reference_state()
        self._render_display()
        QMessageBox.critical(self, T("Crosstalk correction failed"), str(details))
        self._update_action_state()

    @Slot(object)
    def _on_crosstalk_finished(self, payload: object) -> None:
        _kind, serial = payload  # type: ignore[misc]
        if int(serial) == self._crosstalk_serial and self._crosstalk_busy:
            self._crosstalk_busy = False
            self._update_crosstalk_reference_state()
            self._update_action_state()

    # ---------- image queue and loading ----------
    def _open_initial(self, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
            if resolved.is_file():
                self._add_paths([resolved])
            elif resolved.is_dir():
                self._add_paths(
                    supported_image_files(resolved, recursive=True),
                    display_root=resolved,
                )
            else:
                raise FileNotFoundError(T("The path does not exist: {0}", resolved.name or T("selected location")))
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Cannot open"),
                _safe_error_message(exc, Path(path)),
            )

    def _choose_files(self) -> None:
        initial = str(self.loaded.path.parent if self.loaded is not None else Path.home())
        values, _ = QFileDialog.getOpenFileNames(self, T("Open images"), initial, _IMAGE_FILTER)
        if values:
            self._add_paths(Path(value) for value in values)

    def _choose_selection_review_to_continue(self) -> None:
        """Continue one selection by manually pairing its clean base and Mask."""

        if (
            self._load_busy
            or self._segment_busy
            or self._manual_busy
            or self._active_export
            or self._batch_worker is not None
            or self._export_prompt_open
        ):
            return
        initial = str(self.loaded.path.parent if self.loaded is not None else Path.home())
        base_value, _ = QFileDialog.getOpenFileName(
            self,
            T("Step 1 of 2: choose the clean base image without a selection outline"),
            initial,
            _IMAGE_FILTER,
        )
        if not base_value:
            return
        base_path = Path(base_value).expanduser().resolve()
        mask_value, _ = QFileDialog.getOpenFileName(
            self,
            T("Step 2 of 2: choose the binary mask that belongs to the base image"),
            str(base_path.parent),
            _SELECTION_MASK_FILTER,
        )
        if not mask_value:
            return
        mask_path = Path(mask_value).expanduser().resolve()
        try:
            if base_path == mask_path:
                raise ValueError(T(
                    "The clean base image and the binary mask cannot be the same file.",
                ))
            bundle = SelectionReviewBundle(
                review_path=None,
                base_path=base_path,
                mask_path=mask_path,
            )
            base_probe = load_image(bundle.base_path)
            load_selection_review_mask(bundle, base_probe.shape)
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Cannot continue the analysis"),
                _safe_error_message(exc, base_path, mask_path),
            )
            return

        if not self._confirm_discard_manual(
            T(
                "Importing an existing selection switches to its base image and discards the "
                "manual edits that have not been exported yet.",
            )
        ):
            return

        self._selection_resume_bundles[base_path] = bundle
        previous_signals = self.queue_list.blockSignals(True)
        try:
            self._add_paths([base_path])
            if base_path not in self._visible_paths and self._queue_filter_text.strip():
                # The user asked for this exact image; a stale filter must not
                # hide the row they are about to work on.
                self.queue_filter_edit.clear()
                self._queue_filter_text = ""
                self._queue_filter_timer.stop()
                self._refresh_queue(keep_path=base_path)
            row = next(
                index
                for index, candidate in enumerate(self._visible_paths)
                if candidate == base_path
            )
            self.queue_list.setCurrentRow(row)
            item = self.queue_list.item(row)
            if item is not None:
                item.setToolTip(
                    0,
                    T("{0}\nMask paired by hand: {1}", base_path, bundle.mask_path.name),
                )
        finally:
            self.queue_list.blockSignals(previous_signals)
        self._load_path(base_path)

    def _ask_for_review_source(
        self,
        missing: ReviewSourceNotFound,
    ) -> SelectionReviewBundle | None:
        """Let the user point to the original image of an exported result.

        Exports name the original image but do not store where it was, so a
        result kept in a separate output folder needs this one extra step.  The
        content fingerprint in the record is checked when the image is loaded.
        """

        value, _ = QFileDialog.getOpenFileName(
            self,
            T("Choose the original image of this result: {0}", missing.file_name),
            str(missing.review_path.parent),
            _IMAGE_FILTER,
        )
        if not value:
            return None
        base_path = Path(value).expanduser().resolve()
        if base_path in {missing.review_path, missing.mask_path}:
            raise ValueError(T(
                "The original image cannot be the exported review image or the mask itself.",
            ))
        return SelectionReviewBundle(missing.review_path, base_path, missing.mask_path)

    def _choose_folder(self) -> None:
        initial = str(self.loaded.path.parent if self.loaded is not None else Path.home())
        value = QFileDialog.getExistingDirectory(self, T("Open image folder (subfolders included)"), initial)
        if value:
            self._open_folder(Path(value))

    def _open_folder(self, value: Path) -> None:
        """Add every readable image below a folder to the queue."""

        try:
            paths = supported_image_files(value, recursive=True)
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Reading failed"),
                _safe_error_message(exc, Path(value)),
            )
            return
        if not paths:
            QMessageBox.information(self, T("No images"), T(
                "This folder contains no 2-D images that can be read.",
            ))
            return
        if len(paths) > 1000:
            answer = QMessageBox.question(
                self,
                T("Many images"),
                T(
                    "{0:,} images were found. Adding all of them to the queue may be slow. "
                    "Continue?",
                    len(paths),
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._add_paths(paths, display_root=Path(value))

    @staticmethod
    def _queue_text_for_path(path: Path, display_root: Path | None) -> str:
        if display_root is not None:
            try:
                return path.relative_to(display_root).as_posix()
            except ValueError:
                pass
        parent_name = path.parent.name
        return f"{parent_name}/{path.name}" if parent_name else path.name

    @staticmethod
    def _unique_import_root_labels(roots: Iterable[Path]) -> dict[Path, Path]:
        """Build the shortest stable, non-absolute labels for import roots."""

        ordered = sorted(
            {Path(root).expanduser().resolve() for root in roots},
            key=lambda item: str(item).casefold(),
        )
        root_parts: dict[Path, tuple[str, ...]] = {}
        for root in ordered:
            ignored = {"", root.anchor, root.drive}
            parts = tuple(part for part in root.parts if part not in ignored)
            root_parts[root] = parts or (root.name or T("Import root"),)
        depths = {root: 1 for root in ordered}
        while True:
            grouped: dict[str, list[Path]] = {}
            for root in ordered:
                parts = root_parts[root]
                label = Path(*parts[-depths[root] :]).as_posix().casefold()
                grouped.setdefault(label, []).append(root)
            collisions = [group for group in grouped.values() if len(group) > 1]
            if not collisions:
                break
            changed = False
            for group in collisions:
                for root in group:
                    if depths[root] < len(root_parts[root]):
                        depths[root] += 1
                        changed = True
            if not changed:
                break

        labels = {
            root: Path(*root_parts[root][-depths[root] :]) for root in ordered
        }
        grouped_labels: dict[str, list[Path]] = {}
        for root, label in labels.items():
            grouped_labels.setdefault(label.as_posix().casefold(), []).append(root)
        for group in grouped_labels.values():
            if len(group) < 2:
                continue
            for index, root in enumerate(group, start=1):
                label = labels[root]
                labels[root] = (
                    label
                    if index == 1
                    else label.parent / f"{label.name}_{index:03d}"
                )
        # A root namespace must occupy exactly one path component; otherwise
        # one label could be a prefix of another and become ambiguous once the
        # source-relative path is appended.  Preserve the readable suffix with
        # a middle-dot separator, then deterministically suffix rare text
        # collisions (for example when a real folder name already contains it).
        flattened = {
            root: Path(" · ".join(labels[root].parts)) for root in ordered
        }
        grouped_flattened: dict[str, list[Path]] = {}
        for root, label in flattened.items():
            grouped_flattened.setdefault(label.name.casefold(), []).append(root)
        for group in grouped_flattened.values():
            if len(group) < 2:
                continue
            for index, root in enumerate(group, start=1):
                label = flattened[root]
                flattened[root] = (
                    label if index == 1 else Path(f"{label.name}_{index:03d}")
                )
        return flattened

    def _export_relative_paths(self, paths: Iterable[Path]) -> dict[Path, Path]:
        """Return collision-free mirror labels for the current import context."""

        resolved_paths = [Path(path).expanduser().resolve() for path in paths]
        relative_paths = {
            path: Path(self._queue_relative_text.get(path, path.name))
            for path in resolved_paths
        }
        def collision_key(path: Path, relative: Path) -> str:
            return (
                f"{relative.parent.as_posix().casefold()}|"
                f"{export_bundle_base_name(path).casefold()}"
            )

        groups: dict[str, list[Path]] = {}
        for path, relative in relative_paths.items():
            groups.setdefault(collision_key(path, relative), []).append(path)
        has_cross_root_collision = False
        for group in groups.values():
            group_roots = {
                Path(self._queue_import_roots.get(path, path.parent))
                .expanduser()
                .resolve()
                for path in group
            }
            if len(group) > 1 and len(group_roots) > 1:
                has_cross_root_collision = True
                break
        if has_cross_root_collision:
            roots = {
                Path(self._queue_import_roots.get(path, path.parent))
                .expanduser()
                .resolve()
                for path in resolved_paths
            }
            root_labels = self._unique_import_root_labels(roots)
            for path in resolved_paths:
                root = (
                    Path(self._queue_import_roots.get(path, path.parent))
                    .expanduser()
                    .resolve()
                )
                relative_paths[path] = root_labels[root] / relative_paths[path]

        # Defensive final pass: case-insensitive destinations can receive files
        # from a case-sensitive source.  Allocate a deterministic extra parent
        # for any labels that still collide, checking against all other labels.
        final_groups: dict[str, list[Path]] = {}
        for path, relative in relative_paths.items():
            final_groups.setdefault(collision_key(path, relative), []).append(path)
        occupied = {
            collision_key(group[0], relative)
            for group in final_groups.values()
            if len(group) == 1
            for relative in (relative_paths[group[0]],)
        }
        for group in final_groups.values():
            if len(group) < 2:
                continue
            group_roots = {
                Path(self._queue_import_roots.get(path, path.parent))
                .expanduser()
                .resolve()
                for path in group
            }
            if len(group_roots) < 2:
                # Same-directory sources with the same safe stem (for example
                # sample.tif and sample.png) stay in that directory. The atomic
                # reservation assigns _002 instead of inventing a result folder.
                continue
            for path in sorted(group, key=lambda item: str(item).casefold()):
                base = relative_paths[path]
                index = 1
                while True:
                    candidate = Path(T("import_{0:03d}", index)) / base
                    key = collision_key(path, candidate)
                    if key not in occupied:
                        relative_paths[path] = candidate
                        occupied.add(key)
                        break
                    index += 1
        return relative_paths

    def _add_paths(
        self,
        values: Iterable[str | Path],
        *,
        display_root: str | Path | None = None,
    ) -> int:
        """Add unique paths while keeping queue-only display context separate."""

        resolved_display_root = (
            Path(display_root).expanduser().resolve()
            if display_root is not None
            else None
        )
        existing = {str(path).casefold(): path for path in self.paths}
        added = 0
        resumed_path: Path | None = None
        for value in values:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                continue
            if resolved_display_root is None and path.name.casefold().endswith(_REVIEW_IMAGE_SUFFIXES):
                try:
                    try:
                        bundle = resolve_selection_review_bundle(path)
                    except ReviewSourceNotFound as missing:
                        bundle = self._ask_for_review_source(missing)
                        if bundle is None:
                            continue
                    load_selection_review_mask(bundle)
                except Exception as exc:
                    QMessageBox.critical(self, T("Cannot restore the AI result"), _safe_error_message(exc, path))
                    continue
                if self.result is not None and self.result.manually_edited:
                    if not self._confirm_discard_manual(T(
                        "Opening an AI result switches the selection and clears the manual edits "
                        "that have not been exported yet.",
                    )):
                        continue
                path = bundle.base_path
                self._selection_resume_bundles[path] = bundle
                resumed_path = path
            key = str(path).casefold()
            if key in existing:
                existing_path = existing[key]
                if resolved_display_root is not None:
                    try:
                        relative_text = existing_path.relative_to(
                            resolved_display_root
                        ).as_posix()
                    except ValueError:
                        pass
                    else:
                        previous_root = self._queue_import_roots.get(
                            existing_path,
                            existing_path.parent,
                        )
                        # Prefer the broadest explicitly selected folder.  A
                        # later narrow folder must not discard richer context.
                        if (
                            previous_root == existing_path.parent
                            or resolved_display_root == previous_root
                            or resolved_display_root in previous_root.parents
                        ):
                            self._queue_relative_text[existing_path] = relative_text
                            self._queue_import_roots[existing_path] = resolved_display_root
                            self._queue_display_text[existing_path] = relative_text
                continue
            self.paths.append(path)
            display_text = self._queue_text_for_path(path, resolved_display_root)
            self._queue_display_text[path] = display_text
            import_root = path.parent
            if resolved_display_root is not None:
                try:
                    relative_text = path.relative_to(resolved_display_root).as_posix()
                except ValueError:
                    relative_text = path.name
                else:
                    import_root = resolved_display_root
            else:
                relative_text = path.name
            self._queue_relative_text[path] = relative_text
            self._queue_import_roots[path] = import_root
            try:
                stat = path.stat()
                self._queue_file_stats[path] = (int(stat.st_mtime_ns), int(stat.st_size))
            except OSError:
                self._queue_file_stats[path] = (0, 0)
            self._queue_added_counter += 1
            self._queue_added_order[path] = self._queue_added_counter
            existing[key] = path
            added += 1
        had_current = self.queue_list.currentRow() >= 0
        self._refresh_queue()
        if added and not had_current and self._visible_paths:
            self.queue_list.setCurrentRow(0)
        elif resumed_path is not None:
            if self._queue_filter_text.strip():
                self.queue_filter_edit.clear()
                self._queue_filter_text = ""
                self._refresh_queue(keep_path=resumed_path)
            row = next((i for i, p in enumerate(self._visible_paths) if p == resumed_path), -1)
            if row >= 0:
                self.queue_list.setCurrentRow(row)
                if self._loading_path != resumed_path:
                    self._load_path(resumed_path)
        visible = len(self._visible_paths)
        self.status_label.setText(
            T("{0:,} image(s) in the queue", len(self.paths))
            + (T(", {0:,} shown after filtering", visible) if visible != len(self.paths) else "")
        )
        self._update_queue_summary()
        self._update_action_state()
        return added

    def _forget_queue_path(self, path: Path) -> None:
        self._queue_display_text.pop(path, None)
        self._queue_relative_text.pop(path, None)
        self._queue_import_roots.pop(path, None)
        self._selection_resume_bundles.pop(path, None)
        self._crosstalk_reference_paths.pop(path, None)
        self._queue_status.pop(path, None)
        self._queue_file_stats.pop(path, None)
        self._queue_added_order.pop(path, None)
        self._queue_export_target.pop(path, None)

    def _remove_selected(self) -> None:
        rows = sorted({self.queue_list.row(item) for item in self.queue_list.selectedItems()}, reverse=True)
        rows = [row for row in rows if 0 <= row < len(self._visible_paths)]
        if not rows:
            return
        removed_paths = [self._visible_paths[row] for row in rows]
        current_path = self.loaded.path if self.loaded is not None else None
        if current_path is not None and current_path in removed_paths:
            if not self._confirm_discard_manual(T(
                "Removing the current image discards its manual edits that have not been exported "
                "yet.",
            )):
                return
            # Agreed once; the follow-up row change must not ask again.
            self._manual_dirty = False
        removed_set = set(removed_paths)
        self.paths = [path for path in self.paths if path not in removed_set]
        for removed_path in removed_paths:
            self._forget_queue_path(removed_path)
        self._refresh_queue()
        if not self.paths:
            self._clear_current_image()
        elif self.queue_list.currentRow() < 0 and self._visible_paths:
            self.queue_list.setCurrentRow(min(rows[-1], len(self._visible_paths) - 1))
        self._update_action_state()
        self._update_crosstalk_reference_state()

    def _queue_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._visible_paths):
            return
        path = self._visible_paths[row]
        if self.loaded is not None and path == self.loaded.path:
            return
        if self.loaded is not None and not self._confirm_discard_manual(
            T(
                "Switching images discards the manual edits of the current image that have not "
                "been exported yet.",
            )
        ):
            previous_row = next(
                (
                    index
                    for index, candidate in enumerate(self._visible_paths)
                    if candidate == self.loaded.path
                ),
                -1,
            )
            self.queue_list.blockSignals(True)
            self.queue_list.setCurrentRow(previous_row)
            self.queue_list.blockSignals(False)
            return
        self._load_path(path)

    def _reset_image_view_for_transition(self) -> None:
        """Clear transient tools and stale graphics before an image transition."""

        self.pan_action.setChecked(True)
        self.image_view.set_edit_context("pan", 1)
        self.image_view.cancel_stroke()
        self.image_view.clear_focus()
        self.image_view.invalidate_contour_cache()
        self.image_view.set_contour(None)
        self.image_view.set_rgb_image(np.zeros((480, 720, 3), dtype=np.uint8), fit=True)

    def _load_path(self, path: Path) -> None:
        self.auto_timer.stop()
        self.load_token += 1
        self._crosstalk_serial += 1
        self._pending_segment_full = None
        token = self.load_token
        self.source_loaded = None
        self.loaded = None
        self.result = None
        self._mark_active_queue_row()
        self.base_display = None
        self.analysis_display = None
        self.crosstalk_reference_display = None
        self._crosstalk_reference_loaded = None
        self._crosstalk_analysis_key = None
        self._crosstalk_busy = False
        self._crosstalk_error = None
        self._pending_crosstalk_full = None
        self._resume_base_result = None
        self._edge_offset_serial = None
        self._pending_edge_rebuild = False
        self._loading_path: Path | None = path
        self._clear_manual_history()
        self._reset_image_view_for_transition()
        self._load_busy = True
        self.image_info.setText(T("Reading {0}", path.name))
        self.stats_label.setText(T("Waiting for analysis"))
        self.stats_summary_label.setText(T("Waiting for analysis"))
        self.analysis_badge.setText(T("Reading…"))
        self.status_label.setText(T("Reading: {0}", path.name))
        self._update_crosstalk_reference_state()
        self._update_action_state()
        worker = LoadWorker(
            token,
            path,
            preserve_original_values=(
                self._selection_resume_bundles.get(path.resolve()) is not None
            ),
        )
        worker.signals.result.connect(self._on_loaded)
        worker.signals.error.connect(self._on_load_error)
        worker.signals.finished.connect(self._on_load_finished)
        self.thread_pool.start(worker)

    @Slot(object)
    def _on_loaded(self, payload: object) -> None:
        token, loaded, display, display_scale = payload  # type: ignore[misc]
        if int(token) != self.load_token:
            return
        self._load_busy = False
        self.source_loaded = loaded
        self.loaded = loaded
        self.result = None
        self._mark_active_queue_row()
        resume_bundle = self._selection_resume_bundles.get(loaded.path.resolve())
        if resume_bundle is not None:
            loaded.metadata["_preserve_merge_original_values"] = True
        if resume_bundle is not None and self.crosstalk_group.isChecked():
            previous_group = self.crosstalk_group.blockSignals(True)
            self.crosstalk_group.setChecked(False)
            self.crosstalk_group.blockSignals(previous_group)
            self._accepted_crosstalk_enabled = False
        plan = None if self._ai_mode() else compile_quick_plan(
            self._quick_base_params,
            self._quick_settings,
            loaded.shape,
        )
        if plan is not None and plan.is_valid and plan.branches:
            self._set_controls_from_params(plan.branches[0].params)
        # The worker prepared the frame for the same resume decision (it was
        # taken from the same bundle map before the load started).
        self.base_display = display
        self.display_scale = float(display_scale)
        self.analysis_display = self.base_display
        channels = T("greyscale") if loaded.image.ndim == 2 else T("{0}-channel colour", loaded.image.shape[2])
        self.image_info.setText(
            f"{loaded.path.name} · {loaded.shape[1]:,} × {loaded.shape[0]:,} px · "
            f"{loaded.image.dtype} {channels} · "
            f"{int(loaded.metadata['size']) / (1024 * 1024):.2f} MiB"
        )
        self.image_info.setToolTip(str(loaded.path))
        self._render_display(fit=True)
        self.status_label.setText(T("Loaded: {0}", loaded.path.name))
        self._update_crosstalk_reference_state()
        self._update_parameter_mode_label()
        if resume_bundle is not None:
            try:
                imported_mask = load_selection_review_mask(
                    resume_bundle,
                    loaded.shape,
                )
                resume_params = self._params_from_controls()
                # First restore the exact user mask without running any model.
                # Companion AI records then restore its learned provenance.
                resume_params.recognition_engine = "classic"
                self._quick_base_params = copy.deepcopy(resume_params)
                self._quick_settings = QuickSettings()
                self._quick_plan_error = None
                self._parameter_source = "professional"
                self._set_quick_controls(self._quick_settings)
                self._select_custom_preset()
                resumed = rebuild_result_from_mask(
                    loaded,
                    imported_mask,
                    resume_params,
                )
                resumed = restore_ai_result_sidecars(resume_bundle.mask_path, loaded, resumed)
                self._quick_base_params = copy.deepcopy(resumed.params)
                self._set_controls_from_params(resumed.params)
                resumed = _tag_result_analysis_input(resumed, loaded)
                setattr(
                    resumed,
                    "quick_plan_key",
                    quick_plan_key(
                        self._quick_base_params,
                        self._quick_settings,
                        loaded.shape,
                    ),
                )
                setattr(resumed, "quick_variant_count", 1)
            except Exception as exc:
                self.analysis_badge.setText(T("Restoring the existing selection failed"))
                self.status_label.setText(T(
                    "The base image or the mask failed verification; automatic recognition was "
                    "not started",
                ))
                self._set_queue_status(loaded.path, "failed", T("Restoring the existing selection failed"))
                sensitive_paths = [
                    resume_bundle.mask_path,
                    resume_bundle.base_path,
                ]
                if resume_bundle.review_path is not None:
                    sensitive_paths.append(resume_bundle.review_path)
                QMessageBox.critical(
                    self,
                    T("Cannot restore the existing selection"),
                    _safe_error_message(exc, *sensitive_paths),
                )
                self._update_action_state()
                return
            # Keep a private copy of the imported selection: the edge distance
            # is always recomputed from this pristine Mask, never cumulatively.
            self._resume_base_result = _pristine_result_copy(resumed)
            self.result = resumed
            self._clear_manual_history()
            self.image_view.invalidate_contour_cache()
            self._render_display(fit=True)
            self._update_stats()
            self._update_parameter_mode_label()
            self.image_info.setText(
                self.image_info.text()
                + T("\nSelection restored: {0}", resume_bundle.mask_path.name)
            )
            self.analysis_badge.setText(T(
                "AI main selection and review area restored · ready for editing",
            ) if _uses_learned_engine(resumed.params) else T("Selection restored · ready for editing"))
            self.status_label.setText(
                T(
                    "Clean base image and existing mask loaded; use the brush, the edge distance "
                    "or object exclusion, or export again",
                )
            )
            self._set_queue_status(loaded.path, "analyzed", T("Selection restored"))
            self._update_action_state()
            if self._effective_edge_offset():
                self._start_edge_offset_rebuild()
            return
        if self._ai_mode():
            self._start_segmentation(full=True)
            return
        if plan is not None and plan.is_valid:
            if (
                self.crosstalk_group.isChecked()
                and loaded.path not in self._crosstalk_reference_paths
            ):
                self.analysis_badge.setText(T(
                    "Crosstalk correction · waiting for the reference image",
                ))
                self.stats_label.setText(T(
                    "No crosstalk reference image is paired with the current target image; the "
                    "original image was not analysed.",
                ))
                self.stats_summary_label.setText(T("Waiting for the crosstalk reference image"))
                self.status_label.setText(T(
                    "Choose a reference image of the same field for the current target image",
                ))
                self._update_action_state()
            else:
                self._start_segmentation(full=False)
        else:
            self.analysis_badge.setText(T("Settings conflict · waiting for a fix"))
            self.stats_label.setText(T(
                "The current execution plan has conflicts; fix the choices before analysing.",
            ))
            self.stats_summary_label.setText(T("Settings conflict, waiting for a fix"))
            self._update_action_state()

    @Slot(object)
    def _on_load_error(self, payload: object) -> None:
        token, details = payload  # type: ignore[misc]
        if int(token) != self.load_token:
            return
        self._load_busy = False
        self.analysis_badge.setText(T("Reading failed"))
        self.status_label.setText(T("Reading the image failed"))
        self._set_queue_status(self._loading_path, "failed", T("Reading failed: {0}", details))
        QMessageBox.critical(self, T("Reading failed"), str(details))
        self._update_action_state()

    def _clear_current_image(self) -> None:
        self.load_token += 1
        self._crosstalk_serial += 1
        self._pending_segment_full = None
        self.source_loaded = None
        self.loaded = None
        self.result = None
        self._resume_base_result = None
        self._edge_offset_serial = None
        self._pending_edge_rebuild = False
        self.base_display = None
        self.analysis_display = None
        self.crosstalk_reference_display = None
        self._crosstalk_reference_loaded = None
        self._crosstalk_analysis_key = None
        self._crosstalk_busy = False
        self._crosstalk_error = None
        self._pending_crosstalk_full = None
        self._clear_manual_history()
        self._reset_image_view_for_transition()
        self.image_info.setText(T("No image loaded"))
        self.image_info.setToolTip("")
        self.stats_label.setText(T("No result yet"))
        self.stats_summary_label.setText(T("No result yet"))
        self.analysis_badge.setText(T("Waiting for analysis"))
        self.status_label.setText(T("Open a 2-D greyscale or colour image to start"))
        self._update_crosstalk_reference_state()
        self._update_action_state()

    # ---------- segmentation ----------
    def _invalidate_result_after_parameter_change(self) -> None:
        """Make stale masks unusable immediately after any parameter change."""

        self.auto_timer.stop()
        active_serial = self._active_segment_serial
        if active_serial is not None and self.segment_serial == active_serial:
            # Running workers cannot be force-stopped safely.  Invalidate their
            # serial immediately, then let the finish handler launch only the
            # newest debounced request.
            self.segment_serial += 1
        self._pending_segment_full = None
        self._segment_busy = active_serial is not None
        self._clear_manual_history()
        self.result = None
        # Any recipe change abandons an imported selection (it can only be
        # re-recognised from the base image now).
        self._resume_base_result = None
        self._pending_edge_rebuild = False
        self.image_view.invalidate_contour_cache()
        self.image_view.set_contour(None)
        self.image_view.clear_focus()
        if self.base_display is not None:
            self._render_display()
        if self.loaded is not None:
            if self.auto_check.isChecked():
                self.analysis_badge.setText(T("Parameters changed · updating automatically"))
                self.stats_label.setText(T(
                    "The old selection is void; updating the selection automatically with the "
                    "latest parameters.",
                ))
                self.stats_summary_label.setText(T("Old selection void, updating automatically…"))
                self.status_label.setText(T(
                    "Parameters changed, updating the selection automatically",
                ))
            else:
                self.analysis_badge.setText(T("Parameters changed · click to update the selection"))
                self.stats_label.setText(
                    T(
                        "The old selection is void; click \"Update preview\" or \"Full "
                        "resolution\" to generate a new one.",
                    )
                )
                self.stats_summary_label.setText(T(
                    "Old selection void, waiting for \"Update preview\"",
                ))
                self.status_label.setText(
                    T(
                        "The program is not stuck: auto preview is off; click \"Update preview\" "
                        "or \"Full resolution\"",
                    )
                )

    def _run_auto_preview(self) -> None:
        if self.loaded is not None and not self._quick_plan_error:
            self._start_segmentation(full=False)

    def _start_segmentation(self, *, full: bool) -> None:
        self.auto_timer.stop()
        if self.source_loaded is None or self._quick_plan_error or self._export_prompt_open:
            if self._quick_plan_error:
                self.status_label.setText(T(
                    "The execution plan has conflicts; the analysis was not started",
                ))
            return
        discarded_manual = False
        if self.result is not None and self.result.manually_edited:
            if not self._confirm_discard_manual(T(
                "Running the automatic analysis again clears the current manual edits.",
            )):
                return
            discarded_manual = True
        params = self._params_from_controls()
        if not self._ensure_crosstalk_analysis_input(params, full=full):
            return
        assert self.loaded is not None
        params_changed = bool(
            self.result is not None
            and not self._ai_mode()
            and not result_matches_quick_plan(
                self.result,
                self._quick_base_params,
                self._quick_settings,
                self.loaded.shape,
            )
        )
        if discarded_manual or params_changed:
            self._invalidate_result_after_parameter_change()
        # A new automatic analysis replaces the editable Mask and therefore
        # explicitly exits any persistent manual drawing tool.
        self._set_edit_mode("pan")
        if self._active_segment_serial is not None:
            # A running QRunnable cannot be cancelled.  Mark it stale and keep
            # just the most recent request; its finish signal will start that
            # request with the then-current controls.
            if self.segment_serial == self._active_segment_serial:
                self.segment_serial += 1
            self._pending_segment_full = bool(full)
            self.manual_serial = None
            self._manual_busy = False
            self._segment_busy = True
            self.analysis_badge.setText(T("Parameters updated · waiting for the latest analysis"))
            self.status_label.setText(
                T(
                    "Finishing the previous computation; then only the latest parameters are run, "
                    "old tasks do not pile up",
                )
            )
            self._update_action_state()
            return

        self.segment_serial += 1
        serial = self.segment_serial
        self._pending_segment_full = None
        self.manual_serial = None
        max_side: int | None
        needs_preview = (
            self.loaded.shape[0] * self.loaded.shape[1] > _PREVIEW_PIXELS
            or max(self.loaded.shape) > _PREVIEW_MAX_SIDE
        )
        max_side = None if self._ai_mode() or full or not needs_preview else _PREVIEW_MAX_SIDE
        if max_side is None:
            self.analysis_badge.setText(T("Computing at full resolution…"))
            self.status_label.setText(T("Running the full-resolution analysis"))
        else:
            self.analysis_badge.setText(T("Computing the quick preview…"))
            self.status_label.setText(T(
                "Generating the quick preview; exports use the full resolution automatically",
            ))
        self._segment_busy = True
        self.preview_button.setEnabled(False)
        self.full_button.setEnabled(False)
        self._active_segment_serial = serial
        self._set_queue_status(self.loaded.path, "running", T("Full-resolution analysis") if max_side is None else T("Quick preview"))
        try:
            worker = SegmentWorker(
                self.load_token,
                serial,
                self.loaded,
                params,
                max_side,
                quick_base_params=self._quick_base_params,
                quick_settings=self._quick_settings,
                edge_offset_px=self._effective_edge_offset(),
            )
            worker.signals.result.connect(self._on_segment_result)
            worker.signals.error.connect(self._on_segment_error)
            worker.signals.finished.connect(self._on_segment_finished)
            self.thread_pool.start(worker)
        except Exception:
            self._active_segment_serial = None
            self._segment_busy = False
            raise
        self._update_action_state()

    # ---------- edge distance (post-recognition boundary offset) ----------
    def _effective_edge_offset(self) -> int:
        if not hasattr(self, "edge_offset_check"):
            return 0
        if not self.edge_offset_check.isChecked():
            return 0
        return int(self.edge_offset_spin.value())

    def _update_edge_offset_summary(self) -> None:
        if not hasattr(self, "edge_offset_status"):
            return
        offset = self._effective_edge_offset()
        if not self.edge_offset_check.isChecked():
            self.edge_offset_status.setText(T("Off · original recognized boundary"))
        elif offset == 0:
            self.edge_offset_status.setText(T("0 px · same as the original boundary"))
        else:
            direction = T("Expand") if offset > 0 else T("Shrink")
            self.edge_offset_status.setText(f"{direction} {abs(offset)} px")

    def _set_edge_offset_controls(self, enabled: bool, offset: int) -> None:
        previous = self._suppress_controls
        self._suppress_controls = True
        try:
            self.edge_offset_check.setChecked(bool(enabled))
            self.edge_offset_spin.setValue(int(offset))
            self.edge_offset_spin.setEnabled(bool(enabled))
        finally:
            self._suppress_controls = previous
        self._update_edge_offset_summary()

    def _edge_offset_changed(self, *_args: object) -> None:
        if self._suppress_controls:
            return
        enabled = bool(self.edge_offset_check.isChecked())
        self.edge_offset_spin.setEnabled(enabled)
        offset = self._effective_edge_offset()
        if offset == self._edge_offset_value:
            self._edge_offset_ui_state = (enabled, int(self.edge_offset_spin.value()))
            self._update_edge_offset_summary()
            return
        if self.result is not None and self.result.manually_edited and self._manual_dirty:
            message = (
                T(
                    "Changing the edge distance recomputes from the restored original mask and "
                    "clears the manual edits made since.",
                )
                if self._resume_base_result is not None
                else T(
                    "Changing the edge distance regenerates the selection from the recognition "
                    "result and clears the current manual edits.",
                )
            )
            if not self._confirm_discard_manual(message):
                self._set_edge_offset_controls(*self._edge_offset_ui_state)
                return
        self._edge_offset_value = offset
        self._edge_offset_ui_state = (enabled, int(self.edge_offset_spin.value()))
        self._update_edge_offset_summary()
        if self.loaded is None:
            self._update_action_state()
            return
        if self._resume_base_result is not None:
            self._start_edge_offset_rebuild()
            return
        self._invalidate_result_after_parameter_change()
        if self.auto_check.isChecked():
            self.analysis_badge.setText(T("Edge distance changed · updating automatically"))
            self.stats_summary_label.setText(T(
                "Recomputing the selection with the new edge distance…",
            ))
            self.status_label.setText(T(
                "Edge distance changed, recomputing the selection boundary",
            ))
            self.auto_timer.start()
        else:
            self.analysis_badge.setText(T("Edge distance changed · click to update the selection"))
            self.stats_summary_label.setText(T(
                "Edge distance changed, waiting for \"Update preview\"",
            ))
            self.status_label.setText(
                T(
                    "Edge distance changed; auto preview is off, click \"Update preview\" or "
                    "\"Full resolution\"",
                )
            )
        self._update_action_state()

    def _start_edge_offset_rebuild(self) -> None:
        """Recompute an imported selection's boundary from its pristine Mask."""

        base = self._resume_base_result
        if base is None or self.loaded is None or self._export_prompt_open:
            return
        if self._active_segment_serial is not None:
            # A running worker cannot be cancelled; run once more with the
            # then-current distance when it finishes.
            self._pending_edge_rebuild = True
            self._segment_busy = True
            self.analysis_badge.setText(T("Edge distance updated · waiting to recompute"))
            self._update_action_state()
            return
        self._pending_edge_rebuild = False
        self.auto_timer.stop()
        self._pending_segment_full = None
        self.segment_serial += 1
        serial = self.segment_serial
        self._edge_offset_serial = serial
        self.manual_serial = None
        self._clear_manual_history()
        self._segment_busy = True
        self.analysis_badge.setText(T("Recomputing the restored selection with the edge distance…"))
        self.status_label.setText(T("Recomputing the selection boundary from the original mask"))
        self.image_view.set_edit_context("pan", 1)
        self._active_segment_serial = serial
        try:
            worker = EdgeOffsetWorker(
                self.load_token,
                serial,
                self.loaded,
                base,
                self._effective_edge_offset(),
            )
            worker.signals.result.connect(self._on_segment_result)
            worker.signals.error.connect(self._on_segment_error)
            worker.signals.finished.connect(self._on_segment_finished)
            self.thread_pool.start(worker)
        except Exception:
            self._active_segment_serial = None
            self._edge_offset_serial = None
            self._segment_busy = False
            raise
        self._update_action_state()

    @Slot(object)
    def _on_segment_result(self, payload: object) -> None:
        token, serial, result = payload  # type: ignore[misc]
        if int(token) != self.load_token or int(serial) != self.segment_serial:
            return
        is_manual = self.manual_serial == int(serial)
        is_edge = self._edge_offset_serial == int(serial)
        self.result = result
        if is_manual:
            self._manual_rebuild_rollback = None
        else:
            self._clear_manual_history()
            if not is_edge:
                # A fresh recognition supersedes any imported selection.
                self._resume_base_result = None
        self.image_view.invalidate_contour_cache()
        self._render_display()
        self._update_stats()
        correction_prefix = (
            T("Crosstalk corrected · ")
            if not self._ai_mode() and self.crosstalk_group.isChecked()
            and self._crosstalk_ready_for_params()
            else ""
        )
        if _uses_learned_engine(result.params):
            correction_prefix = T("Learned model · ")
        offset = int(getattr(result, "edge_offset_px", 0))
        offset_suffix = T(" · edge {0:+d} px", offset) if offset else ""
        if self._resume_base_result is not None and not is_manual:
            self.analysis_badge.setText(T(
                "Selection restored · ready for editing{0}",
                offset_suffix,
            ))
            self.status_label.setText(
                T(
                    "The restored selection was recomputed with the edge distance; continue with "
                    "the brush or export",
                )
                if offset
                else T("The original mask boundary was restored; continue with the brush or export")
            )
            self._set_queue_status(self.loaded.path if self.loaded else None, "analyzed", T("Selection restored"))
        elif result.manually_edited:
            self.analysis_badge.setText(
                T("{0}Full resolution · manually edited{1}", correction_prefix, offset_suffix)
            )
            self.status_label.setText(T(
                "Object measurements were recomputed after the manual edit; remember to export",
            ))
            self._set_queue_status(self.loaded.path if self.loaded else None, "edited")
        elif result.is_preview:
            self.analysis_badge.setText(
                T("{0}Quick preview · {1:.1f}%{2}", correction_prefix, result.analysis_scale * 100, offset_suffix)
            )
            self.status_label.setText(T(
                "This is a reduced preview; exports and manual edits use the full resolution",
            ))
            self._set_queue_status(self.loaded.path if self.loaded else None, "previewed")
        else:
            self.analysis_badge.setText(T("{0}Full-resolution result{1}", correction_prefix, offset_suffix))
            self.status_label.setText(T("Full-resolution analysis finished"))
            self._set_queue_status(self.loaded.path if self.loaded else None, "analyzed")
        self._update_action_state()

    @Slot(object)
    def _on_segment_error(self, payload: object) -> None:
        token, serial, details = payload  # type: ignore[misc]
        if int(token) != self.load_token or int(serial) != self.segment_serial:
            return
        self._manual_busy = False
        is_manual = self.manual_serial == int(serial)
        is_edge = self._edge_offset_serial == int(serial)
        restored = False
        if is_manual:
            restored = self._restore_pending_manual_edit()
            self.manual_serial = None
        if is_edge:
            self._edge_offset_serial = None
        if restored:
            self.analysis_badge.setText(
                T("Full resolution · manually edited")
                if self.result is not None and self.result.manually_edited
                else T("Full-resolution result")
            )
            self.status_label.setText(T(
                "Recomputing after the manual edit failed; the operation was rolled back safely",
            ))
            QMessageBox.critical(self, T("Manual recompute failed (rolled back)"), str(details))
        elif is_edge:
            self.analysis_badge.setText(T("Recomputing the edge distance failed"))
            self.status_label.setText(T(
                "Recomputing the edge distance failed; the previous selection is still shown",
            ))
            QMessageBox.critical(self, T("Recomputing the edge distance failed"), str(details))
        else:
            self._manual_rebuild_rollback = None
            self.analysis_badge.setText(T("Analysis failed"))
            self.status_label.setText(T("Analysis failed"))
            self._set_queue_status(
                self.loaded.path if self.loaded else None, "failed", str(details)
            )
            QMessageBox.critical(self, T("Analysis failed"), str(details))
        self._update_action_state()

    # ---------- rendering and statistics ----------
    def _sync_analysis_channel_from_signal(
        self,
        signal_mode: str,
        *,
        render: bool = True,
    ) -> None:
        """Keep the left analysis channel aligned with effective parameters."""

        channel = display_channel_for_signal_mode(signal_mode)
        index = self.display_channel_combo.findData(channel)
        if index < 0 or self.display_channel_combo.currentIndex() == index:
            return
        previous = self.display_channel_combo.blockSignals(True)
        try:
            self.display_channel_combo.setCurrentIndex(index)
        finally:
            self.display_channel_combo.blockSignals(previous)
        if render and self.base_display is not None:
            self._render_display()

    def _analysis_channel_changed(self, _index: int = -1) -> None:
        """Make the left channel authoritative for view, analysis and export."""

        self._render_display()
        self._update_analysis_context_label()
        if self._suppress_controls or not hasattr(self, "quick_signal_choice"):
            return
        channel = str(self.display_channel_combo.currentData() or "rgb")
        requested_signal = _SIGNAL_MODE_BY_DISPLAY_CHANNEL.get(channel, "auto")
        current_signal = str(self.signal_combo.currentData() or "auto")
        if current_signal != requested_signal:
            self.quick_signal_choice.setCurrentData(requested_signal)
            self._on_quick_parameter_changed()

        # A rejected manual-edit confirmation restores the former quick state.
        # Mirror that effective state back to the left selector rather than
        # leaving a view label which no longer matches the active parameters.
        effective_signal = str(self.signal_combo.currentData() or "auto")
        self._sync_analysis_channel_from_signal(effective_signal)
        if (
            effective_signal == requested_signal
            and self.loaded is not None
            and not self._quick_plan_error
        ):
            # Channel changes are direct input changes, so reanalyse even when
            # general automatic preview is disabled.
            self.auto_timer.start()

    def _set_view_mode(self, mode: str) -> None:
        index = self.view_mode_combo.findData(mode)
        if index < 0 or self.view_mode_combo.currentIndex() == index:
            self._sync_review_buttons()
            return
        self.view_mode_combo.setCurrentIndex(index)

    def _sync_review_buttons(self) -> None:
        if not hasattr(self, "review_button_group"):
            return
        mode = str(self.view_mode_combo.currentData() or "overlay")
        previous_original = self.review_original_button.blockSignals(True)
        previous_corrected = self.review_corrected_button.blockSignals(True)
        self.review_button_group.setExclusive(False)
        try:
            self.review_original_button.setChecked(mode == "overlay")
            self.review_corrected_button.setChecked(mode == "corrected_overlay")
        finally:
            self.review_button_group.setExclusive(True)
            self.review_original_button.blockSignals(previous_original)
            self.review_corrected_button.blockSignals(previous_corrected)

    def _update_view_mode_availability(self) -> None:
        if not hasattr(self, "view_mode_combo"):
            return
        corrected_ready = bool(
            hasattr(self, "crosstalk_group")
            and self.crosstalk_group.isChecked()
            and self._crosstalk_ready_for_params()
            and self.analysis_display is not None
        )
        reference_ready = bool(
            corrected_ready and self.crosstalk_reference_display is not None
        )
        model = self.view_mode_combo.model()
        for key, enabled in (
            ("corrected_overlay", corrected_ready),
            ("corrected", corrected_ready),
            ("crosstalk_reference", reference_ready),
        ):
            index = self.view_mode_combo.findData(key)
            item = model.item(index) if index >= 0 and hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(bool(enabled))
                item.setToolTip(
                    ""
                    if enabled
                    else T(
                        "Available after crosstalk correction is enabled, a reference image is "
                        "set and the preview is updated.",
                    )
                )
        self.review_corrected_button.setEnabled(corrected_ready)
        current_mode = str(self.view_mode_combo.currentData() or "overlay")
        if (
            current_mode in {"corrected_overlay", "corrected"}
            and not corrected_ready
        ) or (current_mode == "crosstalk_reference" and not reference_ready):
            previous = self.view_mode_combo.blockSignals(True)
            try:
                self.view_mode_combo.setCurrentIndex(
                    self.view_mode_combo.findData("overlay")
                )
            finally:
                self.view_mode_combo.blockSignals(previous)
            self._render_display()
        self._sync_review_buttons()

    def _update_display_channel_state(self) -> None:
        mode = str(self.view_mode_combo.currentData() or "overlay")
        self.display_channel_combo.setEnabled(not self._ai_mode())
        if mode in {"overlay", "corrected_overlay", "original", "corrected"}:
            self.display_channel_combo.setToolTip(
                T(
                    "Chooses the channel that is actually analysed; red, green and blue are shown "
                    "in their colour. The main review can show the same final selection on the "
                    "original image or on the corrected input; the export dialog chooses the "
                    "original or the crosstalk-corrected base independently.",
                )
            )
        elif mode == "crosstalk_reference":
            self.display_channel_combo.setToolTip(
                T(
                    "This still decides the channel of the target image that is actually "
                    "analysed; the display of the crosstalk reference image uses the separate "
                    "reference channel below.",
                )
            )
        elif mode in {"signal", "response"}:
            self.display_channel_combo.setToolTip(
                T(
                    "Chooses the channel that is actually analysed; the processed signal or "
                    "response is shown in the matching red, green or blue as well. A change "
                    "analyses again, and the exported Merge image uses the same channel.",
                )
            )
        else:
            self.display_channel_combo.setToolTip(
                T(
                    "The mask is a binary result; this still decides the channel that is actually "
                    "analysed and the exported Merge image.",
                )
            )

    def _view_mode_changed(self, _index: int = -1) -> None:
        self._update_display_channel_state()
        self._sync_review_buttons()
        self._render_display()

    def _render_display(self, *, fit: bool = False) -> None:
        if self.base_display is None:
            return
        mode = str(self.view_mode_combo.currentData() or "overlay")
        channel = str(self.display_channel_combo.currentData() or "rgb")
        # Keep the selected analysis channel visible in every non-binary view.
        # This also prevents a stale full-RGB frame while a channel change is
        # waiting for its replacement analysis result.
        analysis_base = (
            self.analysis_display
            if self.analysis_display is not None
            else self.base_display
        )
        selected_base = (
            self.base_display
            if mode in {"overlay", "original"}
            else analysis_base
        )
        base_rgb = _select_display_channel(selected_base, channel)
        target_shape = self.base_display.shape[:2]
        display_mask: np.ndarray | None = None
        if self.result is not None:
            display_mask = self.image_view.display_mask_for(
                self.result.mask,
                target_shape,
            )

        if mode == "crosstalk_reference":
            if self.crosstalk_reference_display is None:
                rgb = base_rgb
            else:
                reference_mode = str(
                    self.crosstalk_reference_channel_combo.currentData() or "auto"
                )
                reference_channel = display_channel_for_signal_mode(reference_mode)
                rgb = _select_display_channel(
                    self.crosstalk_reference_display,
                    reference_channel,
                )
        elif mode == "corrected":
            rgb = _select_display_channel(analysis_base, channel)
        elif mode == "signal" and self.result is not None:
            rgb = _select_display_channel(
                scalar_to_rgb_display(self.result.signal, target_shape=target_shape),
                channel,
            )
        elif mode == "response" and self.result is not None:
            rgb = _select_display_channel(
                scalar_to_rgb_display(self.result.response, target_shape=target_shape),
                channel,
            )
        elif mode == "mask":
            rgb = np.zeros_like(self.base_display)
            if display_mask is not None:
                rgb[display_mask] = np.asarray((75, 230, 170), dtype=np.uint8)
        elif mode in {"overlay", "corrected_overlay"} and self.result is not None:
            rgb = render_overlay(
                base_rgb,
                self.result.mask,
                fill=False,
                contour=False,
                object_records=(self.result.objects if self.object_id_check.isChecked() else None),
                object_scale=self.display_scale,
            )
        else:
            rgb = base_rgb

        review = getattr(self.result, "review_mask", None) if self.result is not None else None
        if (
            review is not None
            and self.ai_review_check.isChecked()
            and mode in {"overlay", "corrected_overlay", "mask"}
        ):
            shown_review = resized_mask(np.asarray(review, dtype=bool) & ~self.result.mask, target_shape)
            rgb = np.array(rgb, copy=True)
            if mode == "mask":
                rgb[shown_review] = (255, 205, 45)
            else:
                # A restrained yellow tint marks candidates without hiding the
                # original microscopy pixels; it never changes the binary mask.
                rgb[shown_review] = (
                    rgb[shown_review].astype(np.float32) * .45
                    + np.array([255, 205, 45], dtype=np.float32) * .55
                ).astype(np.uint8)
        self.image_view.set_rgb_image(rgb, fit=fit)
        show_contour = bool(
            self.contour_check.isChecked()
            and self.result is not None
            and mode in {"overlay", "corrected_overlay"}
        )
        self.image_view.set_contour(
            display_mask,
            visible=show_contour,
            positive_color=QColor(75, 230, 170) if self.result is not None and _uses_learned_engine(self.result.params) else None,
        )
        selected_edit_mode = self._current_edit_mode()
        if self._can_edit_mask():
            display_radius = max(
                1,
                int(round(self.brush_radius_spin.value() * self.display_scale)),
            )
            self.image_view.set_edit_context(selected_edit_mode, display_radius)
        else:
            # A manual statistics worker renders its rebuilt result before its
            # finished signal releases the busy state.  Suspend drawing without
            # changing the checked persistent add/erase QAction; the finish
            # handler's action-state refresh restores that tool immediately.
            self.image_view.set_edit_context("pan", 1)

    def _update_stats(self) -> None:
        if self.loaded is None or self.result is None:
            self.stats_label.setText(T("No result yet"))
            self.stats_summary_label.setText(T("No result yet"))
            return
        result = self.result
        source_pixels = self.loaded.shape[0] * self.loaded.shape[1]
        if _uses_learned_engine(result.params):
            main_pixels = int(np.count_nonzero(result.mask))
            review = getattr(result, "review_mask", None)
            review_pixels = int(np.count_nonzero(np.asarray(review, dtype=bool) & ~result.mask)) if review is not None else 0
            edited = T(" · manually edited") if result.manually_edited else ""
            metadata = result.ai_metadata or {}
            reference = metadata.get("reference_image")
            reference_name = (
                Path(str(reference.get("path") or reference.get("file_name") or "")).name
                if metadata.get("reference_used") and isinstance(reference, Mapping)
                else ""
            )
            reference_line = (
                T("Reference channel: {0} (paired by file name)", reference_name)
                if reference_name
                else T("Reference channel: none (no file with a matching name in the folder)")
            )
            input_line = T("Input: {0}", str(metadata.get("target_plane") or "—"))
            if abs(float(metadata.get("sensitivity") or 0.0)) >= 0.005:
                input_line += "\n" + T(
                    "Sensitivity　{0:+.2f} (main threshold {1:.2f} instead of {2:.2f})",
                    float(metadata["sensitivity"]),
                    float(metadata["main_threshold"]),
                    float(metadata.get("model_threshold", metadata["main_threshold"])),
                )
            lossy_note = (
                T(
                    "\nJPEG is lossy: small targets can come out differently than from the "
                    "original file.",
                )
                if self.loaded.path.suffix.casefold() in {".jpg", ".jpeg"}
                else ""
            )
            self.stats_summary_label.setText(
                T(
                    "Green main selection {0:,} px² ({1:.3f}%)\nYellow for review {2:,} px² "
                    "(counted separately){3}",
                    main_pixels,
                    100 * main_pixels / source_pixels,
                    review_pixels,
                    edited,
                )
                + lossy_note
            )
            self.stats_label.setText(
                T(
                    "{0}\nMain selection area: {1:,} px²\nReview area: {2:,} px²\nConnected "
                    "components of the main selection: {3:,} (touching targets count as "
                    "one)\nEdge distance: {4:+d} px\nProcessing time: {5:.3f} s\nOriginal "
                    "resolution; yellow is not included in the green area.\nSaturated pixels in "
                    "the original: {6:.2f}% (brightness was clipped there at acquisition; treat "
                    "the result with care)\nThe model score is not a biological probability of "
                    "being positive.\n",
                    _ai_engine_label(result.params.recognition_engine)
                    + "\n" + input_line + "\n" + reference_line,
                    main_pixels,
                    review_pixels,
                    result.object_count,
                    int(getattr(result, 'edge_offset_px', 0)),
                    result.processing_seconds,
                    100 * float(getattr(result, 'input_saturation_fraction', 0.0)),
                )
                + (T(
                    "Edges of large bright stripe-like structures are a known source of false "
                    "positives; review such fields image by image.",
                ))
                + edited
            )
            return
        positive_pixels = int(round(result.positive_fraction * source_pixels))
        areas = np.asarray([float(item["area_px"]) for item in result.objects], dtype=float)
        median_area = float(np.median(areas)) if areas.size else 0.0
        mean_intensities = [float(item["normalized_mean"]) for item in result.objects]
        mean_intensity = float(np.mean(mean_intensities)) if mean_intensities else 0.0
        response_confidences = [
            float(item.get("response_confidence", 0.0)) for item in result.objects
        ]
        mean_response_confidence = (
            float(np.mean(response_confidences)) if response_confidences else 0.0
        )
        border_count = sum(bool(item["touches_border"]) for item in result.objects)
        threshold = "—" if not math.isfinite(result.threshold_value) else f"{result.threshold_value:.4f}"
        mode_label = _SIGNAL_MODE_DISPLAY.get(result.signal_mode_used, result.signal_mode_used)
        quality = T("Preview estimate") if result.is_preview else T("Full resolution")
        if result.manually_edited:
            quality += T(" / manually edited")
        quality_objects = result.objects
        if result.is_preview:
            area_scale = result.analysis_scale * result.analysis_scale
            quality_objects = [
                {**item, "area_px": float(item.get("area_px", 0.0)) * area_scale}
                for item in result.objects
            ]
        preset_key = self._preset_key(self.preset_choices.currentData())
        recipe_name = preset_key[1] if preset_key is not None else ""
        quality_assessment = assess_segmentation(
            result.mask,
            quality_objects,
            mixed_channels=False,
            signal_mode_used=result.signal_mode_used,
            recipe_name=recipe_name,
            structure_method=result.params.background_method,
            confidence_floor=result.params.min_response_confidence,
            extra_warnings=getattr(result, "intensity_warnings", ()),
            force_warning=bool(
                getattr(result, "input_saturation_fraction", 0.0) >= 0.10
                or getattr(result, "input_high_plateau_fraction", 0.0) >= 0.10
                or getattr(result, "input_largest_tie_fraction", 0.0) >= 0.50
            ),
        )
        all_notes = tuple(dict.fromkeys(quality_assessment.warnings))
        notes = list(all_notes[:4])
        if len(all_notes) > 4:
            notes.append(T(
                "{0} more note(s); see quality_report.json after export",
                len(all_notes) - 4,
            ))
        note_text = "" if not notes else "\n" + "\n".join(T("Review note: {0}", item) for item in notes)
        offset = int(getattr(result, "edge_offset_px", 0))
        grid_offset = int(getattr(result, "edge_offset_grid_px", 0))
        if offset:
            edge_text = T("Edge distance: {0:+d} px", offset)
            if result.is_preview:
                edge_text += T(" (preview grid {0:+d} px)", grid_offset)
            edge_text += "\n"
        else:
            edge_text = ""
        self.stats_label.setText(
            T(
                "Objects: {0:,}\nPositive area: {1:,} px²\nPositive fraction: {2:.4f}%\nMedian "
                "area: {3:.2f} px²\nMean normalized intensity: {4:.4f}\nMean object confidence: "
                "{5:.3f}\nObjects touching the border: {6:,}\nThreshold: {7}\nSignal mode: "
                "{8}\n{9}Processing time: {10:.3f} s\nResult level: {11}\nQuality check: {12} "
                "(heuristic, not ground truth){13}",
                result.object_count,
                positive_pixels,
                result.positive_fraction * 100,
                median_area,
                mean_intensity,
                mean_response_confidence,
                border_count,
                threshold,
                mode_label,
                edge_text,
                result.processing_seconds,
                quality,
                quality_assessment.level_label,
                note_text,
            )
        )
        summary_lines = [
            T(
                "Objects {0:,} · positive {1:.3f}% · median area {2:.1f} px² · on border {3:,}",
                result.object_count,
                result.positive_fraction * 100,
                median_area,
                border_count,
            ),
            f"{quality}"
            + (T(" · edge {0:+d} px", offset) if offset else "")
            + T(" · quality check: {0}", quality_assessment.level_label)
            + (T(" · {0} review note(s)", len(all_notes)) if all_notes else ""),
        ]
        if all_notes:
            summary_lines.append(T("Review note: {0}", all_notes[0]))
        self.stats_summary_label.setText("\n".join(summary_lines))

    # ---------- manual correction ----------
    def _current_edit_mode(self) -> str:
        checked = self.mode_group.checkedAction()
        if checked is self.add_action:
            return "add"
        if checked is self.erase_action:
            return "erase"
        return "pan"

    def _tool_action_triggered(self, mode: str) -> None:
        # A second click on a selected drawing tool is an explicit cancel.  The
        # QActionGroup itself remains exclusive and always ends with pan checked.
        if mode in {"add", "erase"} and self.image_view._edit_mode == mode:
            self._set_edit_mode("pan")
            return
        self._set_edit_mode(mode)

    def _can_edit_mask(self) -> bool:
        return bool(
            self.loaded is not None
            and self.result is not None
            and not self.result.is_preview
            and not self._load_busy
            and not self._segment_busy
            and not self._manual_busy
            and self._batch_worker is None
            and not self._active_export
            and not self._export_prompt_open
        )

    def _set_edit_mode(self, mode: str) -> None:
        can_edit = self._can_edit_mask()
        if mode in {"add", "erase"} and not can_edit:
            if self.loaded is not None and self.result is not None and self.result.is_preview:
                self.status_label.setText(T(
                    "Manual edits need a full-resolution result; click \"Full resolution\"",
                ))
            mode = "pan"
        actions = {"pan": self.pan_action, "add": self.add_action, "erase": self.erase_action}
        chosen = actions.get(mode, self.pan_action)
        if not chosen.isChecked():
            chosen.setChecked(True)
        display_radius = max(1, int(round(self.brush_radius_spin.value() * self.display_scale)))
        self.image_view.set_edit_context(mode, display_radius)
        self._update_action_state()

    def _capture_manual_rollback(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        restore_patch: np.ndarray,
    ) -> _ManualRollback:
        assert self.result is not None
        return (
            x0,
            y0,
            x1,
            y1,
            np.asarray(restore_patch, dtype=bool).copy(),
            list(self._undo_stack),
            list(self._redo_stack),
            bool(self._manual_dirty),
            bool(self.result.manually_edited),
        )

    def _restore_pending_manual_edit(self) -> bool:
        rollback = self._manual_rebuild_rollback
        self._manual_rebuild_rollback = None
        if rollback is None or self.result is None:
            return False
        x0, y0, x1, y1, patch, undo_stack, redo_stack, was_dirty, was_manual = rollback
        self.result.mask[y0:y1, x0:x1] = patch
        self.result.manually_edited = was_manual
        self._undo_stack[:] = undo_stack
        self._redo_stack[:] = redo_stack
        self._manual_dirty = was_dirty
        self.image_view.invalidate_contour_cache()
        self._render_display()
        self._update_stats()
        return True

    def _apply_stroke(self, payload: object) -> None:
        try:
            if not self._can_edit_mask():
                return
            assert self.loaded is not None
            assert self.result is not None
            points_display, add_value, _display_radius = payload  # type: ignore[misc]
            scale = max(self.display_scale, 1e-9)
            points = [
                (int(round(float(x) / scale)), int(round(float(y) / scale)))
                for x, y in points_display
            ]
            if not points:
                return
            radius = max(1, int(self.brush_radius_spin.value()))
            height, width = self.loaded.shape
            x0 = max(0, min(x for x, _ in points) - radius - 2)
            y0 = max(0, min(y for _, y in points) - radius - 2)
            x1 = min(width, max(x for x, _ in points) + radius + 3)
            y1 = min(height, max(y for _, y in points) + radius + 3)
            if x1 <= x0 or y1 <= y0:
                return
            before = self.result.mask[y0:y1, x0:x1].astype(bool, copy=True)
            canvas = before.astype(np.uint8) * 255
            relative = np.asarray([(x - x0, y - y0) for x, y in points], dtype=np.int32)
            value = 255 if bool(add_value) else 0
            thickness = radius * 2 + 1
            if len(relative) == 1:
                cv2.circle(canvas, tuple(relative[0]), radius, value, -1, lineType=cv2.LINE_8)
            else:
                cv2.polylines(
                    canvas,
                    [relative.reshape(-1, 1, 2)],
                    False,
                    value,
                    thickness=thickness,
                    lineType=cv2.LINE_8,
                )
                cv2.circle(canvas, tuple(relative[0]), radius, value, -1, lineType=cv2.LINE_8)
                cv2.circle(canvas, tuple(relative[-1]), radius, value, -1, lineType=cv2.LINE_8)
            after = canvas > 0
            if np.array_equal(before, after):
                return
            rollback = self._capture_manual_rollback(x0, y0, x1, y1, before)
            self.result.mask[y0:y1, x0:x1] = after
            self.result.manually_edited = True
            self._manual_dirty = True
            self._undo_stack.append((x0, y0, x1, y1, before, after.copy()))
            if len(self._undo_stack) > 60:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
            self.image_view.invalidate_contour_cache()
            self._render_display()
            self._start_manual_rebuild(rollback)
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Manual edit failed"),
                _safe_error_message(exc, self.loaded.path),
            )

    def _start_manual_rebuild(self, rollback: _ManualRollback | None = None) -> None:
        if (
            self.loaded is None
            or self.result is None
            or self.result.is_preview
            or self._export_prompt_open
            or self._active_segment_serial is not None
        ):
            return
        self._manual_rebuild_rollback = rollback
        self.segment_serial += 1
        serial = self.segment_serial
        self.manual_serial = serial
        self._manual_busy = True
        self._segment_busy = True
        self.analysis_badge.setText(T("Recomputing the manual result…"))
        self.status_label.setText(T("Updating connected objects and shape statistics"))
        # Temporarily stop drawing while the Mask statistics are rebuilt, but
        # keep the selected add/erase QAction so it resumes when the worker ends.
        self.image_view.set_edit_context("pan", 1)
        try:
            worker = SegmentWorker(
                self.load_token,
                serial,
                self.loaded,
                self._params_from_controls(),
                None,
                mask=self.result.mask,
                previous=self.result,
            )
            worker.signals.result.connect(self._on_segment_result)
            worker.signals.error.connect(self._on_segment_error)
            worker.signals.finished.connect(self._on_segment_finished)
            self._active_segment_serial = serial
            self.thread_pool.start(worker)
            self._update_action_state()
        except Exception:
            self._active_segment_serial = None
            self._segment_busy = False
            self._manual_busy = False
            self.manual_serial = None
            self._restore_pending_manual_edit()
            raise

    def _undo_manual(self) -> None:
        if not self._can_edit_mask() or not self._undo_stack:
            return
        assert self.result is not None
        edit = self._undo_stack[-1]
        x0, y0, x1, y1, before, after = edit
        rollback = self._capture_manual_rollback(x0, y0, x1, y1, after)
        self._undo_stack.pop()
        self.result.mask[y0:y1, x0:x1] = before
        self.result.manually_edited = True
        self._manual_dirty = True
        self._redo_stack.append((x0, y0, x1, y1, before, after))
        self.image_view.invalidate_contour_cache()
        self._render_display()
        self._start_manual_rebuild(rollback)

    def _redo_manual(self) -> None:
        if not self._can_edit_mask() or not self._redo_stack:
            return
        assert self.result is not None
        edit = self._redo_stack[-1]
        x0, y0, x1, y1, before, after = edit
        rollback = self._capture_manual_rollback(x0, y0, x1, y1, before)
        self._redo_stack.pop()
        self.result.mask[y0:y1, x0:x1] = after
        self.result.manually_edited = True
        self._manual_dirty = True
        self._undo_stack.append((x0, y0, x1, y1, before, after))
        self.image_view.invalidate_contour_cache()
        self._render_display()
        self._start_manual_rebuild(rollback)

    def _clear_manual_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.manual_serial = None
        self._manual_busy = False
        self._manual_dirty = False
        self._manual_rebuild_rollback = None

    def _confirm_discard_manual(self, message: str) -> bool:
        if not self._manual_dirty:
            return True
        answer = QMessageBox.question(
            self,
            T("Manual edits not exported yet"),
            message + T("\n\nContinue?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    # ---------- export ----------
    def _default_output_parent(self) -> Path:
        if self.loaded is not None:
            parent = self.loaded.path.parent
            return parent.parent if parent.parent != parent else parent
        return Path.home()

    def _choose_output_root(self, title: str, paths: list[Path]) -> Path | None:
        value = QFileDialog.getExistingDirectory(self, title, str(self._default_output_parent()))
        if not value:
            return None
        root = Path(value).expanduser().resolve()
        import_roots = {
            self._queue_import_roots.get(
                Path(path).expanduser().resolve(),
                Path(path).expanduser().resolve().parent,
            )
            for path in paths
        }
        conflicts = [
            source_root
            for source_root in import_roots
            if root == source_root or source_root in root.parents
        ]
        if conflicts:
            QMessageBox.warning(
                self,
                T("Unsafe output location"),
                T(
                    "The mirror output root cannot lie inside the imported folder.\nChoose a "
                    "location outside the imported folder, or go back and choose \"Export into "
                    "the folder of each original image\".",
                ),
            )
            return None
        return root

    def _choose_export_plan(
        self,
        *,
        paths: list[Path],
        output_title: str,
        context_text: str,
        batch_count: int | None = None,
        crosstalk_available: bool = False,
    ) -> ExportPlan | None:
        """Choose content before destination and remember only a started plan."""

        if self._export_prompt_open:
            return None
        self._export_prompt_open = True
        self._update_action_state()
        try:
            dialog = ExportOptionsDialog(
                self._last_export_artifacts,
                context_text=context_text,
                batch_count=batch_count,
                outline_expand_px=self._last_export_outline_expand_px,
                crosstalk_available=crosstalk_available,
                merge_base_mode=self._last_export_merge_base_mode,
                export_location_mode=self._last_export_location_mode,
                parent=self,
            )
            if dialog.exec() != QDialog.Accepted:
                return None
            artifacts = dialog.selected_artifacts()
            if not artifacts:
                return None
            outline_expand_px = dialog.outline_expand_px()
            merge_base_mode = dialog.merge_base_mode()
            location_mode = dialog.export_location_mode()
            root: Path | None = None
            if location_mode == EXPORT_LOCATION_MIRROR:
                root = self._choose_output_root(output_title, paths)
                if root is None:
                    return None
            self._last_export_artifacts = frozenset(artifacts)
            self._last_export_outline_expand_px = int(outline_expand_px)
            self._last_export_location_mode = location_mode
            if crosstalk_available:
                self._last_export_merge_base_mode = merge_base_mode
            return ExportPlan(
                output_root=root,
                artifacts=self._last_export_artifacts,
                outline_expand_px=self._last_export_outline_expand_px,
                merge_base_mode=merge_base_mode,
                location_mode=location_mode,
            )
        finally:
            self._export_prompt_open = False
            self._update_action_state()

    def _export_current(self) -> None:
        crosstalk_ready = self._crosstalk_ready_for_params()
        if (
            self.loaded is None
            or self._quick_plan_error
            or self._crosstalk_error
            or not crosstalk_ready
            or self._load_busy
            or self._segment_busy
            or self._manual_busy
            or self._crosstalk_busy
            or self._active_export
            or self._batch_worker is not None
            or self._export_prompt_open
        ):
            return
        plan = self._choose_export_plan(
            paths=[self.loaded.path],
            output_title=T("Choose the output folder for the current result"),
            context_text=T("Current image: {0}", self.loaded.path.name),
            crosstalk_available=self.crosstalk_group.isChecked() and not self._ai_mode(),
        )
        if plan is None:
            return
        params = self._params_from_controls()
        result = self.result
        current_path = self.loaded.path.expanduser().resolve()
        relative_path = self._export_relative_paths(self.paths).get(
            current_path,
            Path(self._queue_relative_text.get(current_path, current_path.name)),
        )
        try:
            worker = ExportWorker(
                self.loaded,
                params,
                result,
                plan.output_root,
                relative_path=relative_path,
                export_location_mode=plan.location_mode,
                artifacts=plan.artifacts,
                outline_expand_px=plan.outline_expand_px,
                merge_base_mode=plan.merge_base_mode,
                merge_source_loaded=self.source_loaded,
                quick_base_params=self._quick_base_params,
                quick_settings=self._quick_settings,
                edge_offset_px=self._effective_edge_offset(),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Invalid export location"),
                _safe_error_message(exc, self.loaded.path),
            )
            return
        self._active_export = True
        self._active_export_artifacts = plan.artifacts
        self._active_export_location_mode = plan.location_mode
        self._active_export_path = current_path
        self._active_export_root = plan.output_root
        self._set_queue_status(current_path, "running", T("Export current"))
        dialog = QProgressDialog(T(
            "Generating the full-resolution result and exporting…",
        ), "", 0, 0, self)
        dialog.setWindowTitle(T("Export current result"))
        dialog.setCancelButton(None)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.show()
        self._export_dialog = dialog
        worker.signals.result.connect(self._export_done)
        worker.signals.error.connect(self._export_error)
        worker.signals.finished.connect(self._export_finished)
        self.thread_pool.start(worker)
        self.status_label.setText(T("Exporting the current image"))
        self._update_action_state()

    @Slot(object)
    def _export_done(self, output: object) -> None:
        if not isinstance(output, ExportReceipt):
            raise TypeError(T("The export-finished signal lacks the receipt for its files"))
        # Only lossless mask-bearing artifacts preserve manual pixel edits.
        # Exporting a recipe, response image, report, or summary alone must not
        # suppress the later "unsaved manual changes" warning.
        if self._active_export_artifacts and self._active_export_artifacts.intersection(
            {EXPORT_ARTIFACT_POSITIVE_MASK, EXPORT_ARTIFACT_OBJECT_LABELS}
        ):
            self._manual_dirty = False
        exported_path = getattr(self, "_active_export_path", None)
        if exported_path is not None:
            self._set_queue_status(exported_path, "exported", str(output.prefix))
            self._queue_export_target[exported_path] = (
                str(self._active_export_location_mode or ""),
                getattr(self, "_active_export_root", None),
            )
        self.status_label.setText(
            T("Export finished: {0} ({1:,} files)", output.prefix, len(output.files))
        )
        if self._active_export_location_mode == EXPORT_LOCATION_SOURCE:
            location_label = T("the folder of the original image")
        elif output.parent.name:
            location_label = T("the folder \"{0}\"", output.parent.name)
        else:
            location_label = T("the chosen output folder")
        QMessageBox.information(
            self,
            T("Export finished"),
            T(
                "{1:,} result file(s) were written to {0}.\nFile name prefix: {2}",
                location_label,
                len(output.files),
                output.prefix,
            ),
        )

    @Slot(object)
    def _export_error(self, payload: object) -> None:
        _kind, details = payload  # type: ignore[misc]
        self.status_label.setText(T("Export failed"))
        exported_path = getattr(self, "_active_export_path", None)
        if exported_path is not None:
            self._set_queue_status(exported_path, "failed", T("Export failed: {0}", details))
        QMessageBox.critical(self, T("Export failed"), str(details))

    @Slot(object)
    def _export_finished(self, _payload: object) -> None:
        self._active_export = False
        if self._export_dialog is not None:
            self._export_dialog.close()
            self._export_dialog = None
        self._active_export_artifacts = None
        self._active_export_location_mode = None
        self._active_export_path = None
        self._active_export_root = None
        self._update_action_state()

    def _batch_already_exported(
        self,
        paths: list[Path],
        plan: ExportPlan,
        relative_paths: Mapping[Path, Path],
    ) -> list[Path]:
        """Queue images whose result bundle already exists for this plan.

        Two signals are combined: what this session exported to the *same*
        destination (recorded per image), and — for the mirror / source
        layouts, where a bundle location is unique per image — a bundle
        already present on disk from an earlier, interrupted batch.
        """

        done: list[Path] = []
        target = (str(plan.location_mode), plan.output_root)
        on_disk_check = plan.location_mode in {EXPORT_LOCATION_MIRROR, EXPORT_LOCATION_SOURCE}
        for path in paths:
            if self._queue_export_target.get(path) == target and (
                self._queue_status.get(path, ("", "", 0.0))[0] == "exported"
            ):
                done.append(path)
                continue
            if not on_disk_check:
                continue
            try:
                parent = resolve_export_parent(
                    path,
                    plan.output_root,
                    relative_path=relative_paths.get(path, Path(path.name)),
                    location_mode=plan.location_mode,
                )
                if export_bundle_exists(parent, path):
                    done.append(path)
            except Exception:
                continue
        return done

    def _export_batch(self) -> None:
        queue_paths = self._queue_paths()
        crosstalk_enabled = bool(self.crosstalk_group.isChecked()) and not self._ai_mode()
        missing_pairs = [
            path for path in queue_paths if path not in self._crosstalk_reference_paths
        ] if crosstalk_enabled else []
        invalid_pairs = [
            path
            for path in queue_paths
            if crosstalk_enabled
            and (
                not self._crosstalk_reference_paths[path].is_file()
                or self._crosstalk_reference_paths[path] == path
            )
        ] if not missing_pairs else []
        if missing_pairs or invalid_pairs:
            QMessageBox.warning(
                self,
                T("Crosstalk reference pairing incomplete"),
                (
                    T(
                        "Not paired: {0:,} image(s); invalid reference file: {1:,} image(s).\nThe "
                        "batch was not started, so that uncorrected originals are never used "
                        "silently.",
                        len(missing_pairs),
                        len(invalid_pairs),
                    )
                ),
            )
            return
        if (
            not queue_paths
            or self._quick_plan_error
            or self._crosstalk_error
            or self._load_busy
            or self._segment_busy
            or self._manual_busy
            or self._crosstalk_busy
            or self._active_export
            or self._batch_worker is not None
            or self._export_prompt_open
        ):
            return
        export_safety_paths = list(queue_paths)
        if crosstalk_enabled:
            export_safety_paths.extend(
                self._crosstalk_reference_paths[path] for path in queue_paths
            )
        filtered = len(queue_paths) != len(self.paths)
        plan = self._choose_export_plan(
            paths=list(dict.fromkeys(export_safety_paths)),
            output_title=T("Choose the output folder for the batch results"),
            context_text=(
                T(
                    "Processes the {0:,} image(s) shown after filtering, in the current queue "
                    "order ({1:,} in the queue); each gets the same ticked items.",
                    len(queue_paths),
                    len(self.paths),
                )
                if filtered
                else T(
                    "Processes every image in the current queue order; each gets the same ticked "
                    "items.",
                )
            ),
            batch_count=len(queue_paths),
            crosstalk_available=crosstalk_enabled,
        )
        if plan is None:
            return
        # Mirror labels are derived from the whole queue so that a filtered
        # batch lands in exactly the same sub-folders as an unfiltered one.
        relative_paths = self._export_relative_paths(self.paths)
        already_done = self._batch_already_exported(queue_paths, plan, relative_paths)
        batch_paths = list(queue_paths)
        if already_done:
            remaining = [path for path in queue_paths if path not in set(already_done)]
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(T("Continue the previous batch?"))
            if remaining:
                box.setText(
                    T(
                        "{0:,} image(s) of the queue already have results at this output "
                        "location; {1:,} have not been exported yet.",
                        len(already_done),
                        len(remaining),
                    )
                )
                box.setInformativeText(
                    T(
                        "\"Only those not exported\" skips images that already have results and "
                        "continues where the batch stopped; \"Process all again\" recognizes "
                        "every image again and writes new result files.",
                    )
                )
                continue_button = box.addButton(
                    T("Only the {0:,} not exported", len(remaining)), QMessageBox.AcceptRole
                )
                redo_button = box.addButton(T("Process all again"), QMessageBox.DestructiveRole)
            else:
                box.setText(
                    T(
                        "All {0:,} image(s) shown after filtering already have results at this "
                        "output location.",
                        len(queue_paths),
                    )
                )
                box.setInformativeText(T("Process all of them again?"))
                continue_button = None
                redo_button = box.addButton(T("Process all again"), QMessageBox.DestructiveRole)
            cancel_button = box.addButton(T("Cancel"), QMessageBox.RejectRole)
            box.setDefaultButton(continue_button or cancel_button)
            box.setEscapeButton(cancel_button)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_button or clicked is None:
                return
            if clicked is continue_button:
                batch_paths = remaining
            elif clicked is redo_button:
                batch_paths = list(queue_paths)
        if not batch_paths:
            return
        skipped_target = (str(plan.location_mode), plan.output_root)
        for path in queue_paths:
            if path not in set(batch_paths):
                # Carried over from an earlier run: show it as done so the
                # queue reads as one continuous job.
                self._set_queue_status(path, "exported", T("Result exists, skipped this time"))
                self._queue_export_target[path] = skipped_target
        if self.result is not None and self.result.manually_edited:
            QMessageBox.information(
                self,
                T("About batch export"),
                T(
                    "A batch applies the current parameters and the edge distance to every "
                    "original image of the queue; manual edits of the current image (brush or "
                    "object exclusion) are not copied to other images.",
                ),
            )
        params = self._params_from_controls()
        try:
            worker = BatchWorker(
                batch_paths,
                params,
                plan.output_root,
                relative_paths=relative_paths,
                export_location_mode=plan.location_mode,
                artifacts=plan.artifacts,
                outline_expand_px=plan.outline_expand_px,
                merge_base_mode=plan.merge_base_mode,
                quick_base_params=self._quick_base_params,
                quick_settings=self._quick_settings,
                crosstalk_pairs=(
                    {
                        path: self._crosstalk_reference_paths[path]
                        for path in batch_paths
                    }
                    if crosstalk_enabled
                    else None
                ),
                crosstalk_correction=(
                    self._current_crosstalk_settings()
                    if crosstalk_enabled
                    else None
                ),
                edge_offset_px=self._effective_edge_offset(),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                T("Invalid batch export location"),
                _safe_error_message(exc, *batch_paths),
            )
            return
        self._batch_worker = worker
        self._batch_export_location_mode = plan.location_mode
        self._batch_export_root = plan.output_root
        dialog = QProgressDialog(T("Preparing the batch…"), T("Cancel"), 0, len(batch_paths), self)
        dialog.setWindowTitle(T("Batch export"))
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.canceled.connect(worker.cancel)
        dialog.show()
        self._batch_dialog = dialog
        worker.signals.progress.connect(self._batch_progress)
        worker.signals.result.connect(self._batch_done)
        worker.signals.finished.connect(self._batch_finished)
        self.thread_pool.start(worker)
        self.status_label.setText(
            T("Starting the batch of {0:,} image(s)", len(batch_paths))
            + (
                T(" (skipping {0:,} already exported)", len(queue_paths) - len(batch_paths))
                if len(batch_paths) != len(queue_paths)
                else ""
            )
        )
        self._update_action_state()

    @Slot(object)
    def _batch_progress(self, payload: object) -> None:
        values = tuple(payload)  # type: ignore[arg-type]
        if values and values[0] == "item" and len(values) == 4:
            _marker, path_text, state, detail = values
            path = Path(str(path_text))
            self._set_queue_status(path, str(state), str(detail))
            if str(state) == "exported":
                self._queue_export_target[path] = (
                    str(self._batch_export_location_mode or ""),
                    getattr(self, "_batch_export_root", None),
                )
            return
        if len(values) == 4:
            phase, index, total, path = values
        else:
            index, total, path = values
            phase = "process"
        prefix = T("Pairing pre-check") if str(phase) == "preflight" else T("Batch")
        source_label = str(path)
        dialog = self._batch_dialog
        if dialog is not None:
            dialog.setLabelText(
                f"{prefix} [{int(index):,}/{int(total):,}] {source_label}"
            )
            dialog.setMaximum(int(total))
            # A modal QProgressDialog processes events inside setValue(), so
            # the batch may finish (and drop the dialog) before this returns.
            dialog.setValue(int(index))
        self.status_label.setText(
            T("{0} {1:,}/{2:,}: {3}", prefix, int(index), int(total), source_label)
        )

    @Slot(object)
    def _batch_done(self, payload: object) -> None:
        summary = dict(payload)  # type: ignore[arg-type]
        exported = list(summary.get("exported", []))
        failures = list(summary.get("failures", []))
        cancelled = bool(summary.get("cancelled", False))
        preflight_failed = bool(summary.get("preflight_failed", False))
        lines = [
            T("Succeeded: {0:,} image(s)", len(exported)),
            T("Failed: {0:,} image(s)", len(failures)),
        ]
        if cancelled:
            lines.append(T("The task was cancelled; the results already finished are kept."))
        if preflight_failed:
            lines.append(T(
                "The crosstalk pairing pre-check failed, so no image was exported this time.",
            ))
        if failures:
            lines.append(T("\nExamples of failures:"))
            for item in failures[:8]:
                lines.append(
                    T("• {0}: {1}", item.get('source', T("unknown")), item.get('error', T("unknown error")))
                )
            if len(failures) > 8:
                lines.append(T("…and {0} more", len(failures) - 8))
        self.status_label.setText(T(
            "Batch finished: {0:,} succeeded, {1:,} failed",
            len(exported),
            len(failures),
        ))
        QMessageBox.information(self, T("Batch result"), "\n".join(lines))

    @Slot(object)
    def _batch_finished(self, _payload: object) -> None:
        if self._batch_dialog is not None:
            self._batch_dialog.close()
            self._batch_dialog = None
        self._batch_worker = None
        self._batch_export_location_mode = None
        self._batch_export_root = None
        if self._queue_sort_key == "status":
            self._refresh_queue()
        self._update_action_state()

    # ---------- state and auxiliary windows ----------
    def _update_action_state(self) -> None:
        if self._ai_mode():
            self._update_preset_description()
        if hasattr(self, "empty_state"):
            queue_empty = not self.paths
            if self.empty_state.isVisible() != queue_empty:
                self.empty_state.setVisible(queue_empty)
                self.canvas_hint.setVisible(not queue_empty)
                self.image_view._layout_overlays()
        loaded = self.loaded is not None
        has_result = self.result is not None
        crosstalk_enabled = bool(
            hasattr(self, "crosstalk_group") and self.crosstalk_group.isChecked()
        ) and not self._ai_mode()
        crosstalk_ready = self._crosstalk_ready_for_params()
        crosstalk_configured = bool(
            not crosstalk_enabled
            or (
                self.source_loaded is not None
                and self.source_loaded.path in self._crosstalk_reference_paths
            )
        )
        queue_paths = self._queue_paths()
        batch_crosstalk_ready = bool(
            not crosstalk_enabled
            or all(path in self._crosstalk_reference_paths for path in queue_paths)
        )
        has_resumed_selection = any(
            path in self._selection_resume_bundles for path in queue_paths
        )
        plan_valid = not bool(self._quick_plan_error) and not bool(self._crosstalk_error)
        busy = bool(
            self._load_busy
            or self._segment_busy
            or self._manual_busy
            or self._crosstalk_busy
        )
        batch_busy = self._batch_worker is not None
        export_busy = self._active_export
        prompt_busy = self._export_prompt_open
        can_edit = self._can_edit_mask()
        controls_enabled = not busy and not batch_busy and not export_busy and not prompt_busy
        for widget in (
            self.preset_group,
            self.quick_target_group,
            self.quick_more_group,
            self.professional_section,
            self.edge_offset_group,
            self.crosstalk_group,
        ):
            widget.setEnabled(controls_enabled)
        if self._ai_mode():
            for widget in (self.preset_choices, self.quick_target_group, self.quick_more_group,
                           self.professional_section, self.crosstalk_group, self.display_channel_combo):
                widget.setEnabled(False)
        else:
            self.preset_choices.setEnabled(controls_enabled)
        self.display_channel_combo.setEnabled(controls_enabled and not self._ai_mode())
        # The run card stays enabled so its result summary remains readable
        # and selectable while a computation is in flight; its buttons are
        # gated individually below.
        self.auto_check.setEnabled(not batch_busy and not export_busy and not prompt_busy)
        queue_mutation_enabled = not batch_busy and not export_busy and not prompt_busy
        self.open_files_action.setEnabled(queue_mutation_enabled)
        self.open_folder_action.setEnabled(queue_mutation_enabled)
        self.continue_selection_action.setEnabled(queue_mutation_enabled and not busy)
        self.queue_list.setEnabled(queue_mutation_enabled)
        for widget in (
            self.queue_filter_edit,
            self.queue_sort_combo,
            self.queue_sort_direction,
            self.queue_view_toggle,
        ):
            widget.setEnabled(queue_mutation_enabled)
        self.remove_action.setEnabled(
            bool(self.queue_list.selectedItems())
            and not busy
            and not batch_busy
            and not export_busy
            and not prompt_busy
        )
        self.export_action.setEnabled(
            loaded
            and plan_valid
            and crosstalk_ready
            and not busy
            and not export_busy
            and not batch_busy
            and not prompt_busy
        )
        self.batch_action.setToolTip(
            T(
                "An existing selection is restored one image at a time; use \"Export current\" so "
                "that a batch does not recognize the base image again.",
            )
            if has_resumed_selection
            else ""
        )
        self.batch_action.setEnabled(
            bool(queue_paths)
            and plan_valid
            and batch_crosstalk_ready
            and not busy
            and not export_busy
            and not batch_busy
            and not prompt_busy
            and not has_resumed_selection
        )
        self.import_recipe_action.setEnabled(
            not busy and not export_busy and not batch_busy and not prompt_busy
        )
        self.export_recipe_action.setEnabled(
            plan_valid and not export_busy and not batch_busy and not prompt_busy
        )
        can_start_analysis = bool(
            loaded
            and plan_valid
            and crosstalk_configured
            and not busy
            and not export_busy
            and not batch_busy
            and not prompt_busy
        )
        self.preview_button.setEnabled(can_start_analysis)
        self.full_button.setEnabled(can_start_analysis)
        self.full_action.setEnabled(self.full_button.isEnabled())
        self.add_action.setEnabled(can_edit)
        self.erase_action.setEnabled(can_edit)
        self.undo_action.setEnabled(can_edit and bool(self._undo_stack))
        self.redo_action.setEnabled(can_edit and bool(self._redo_stack))
        self.object_table_button.setEnabled(
            has_result
            and not busy
            and not batch_busy
            and not export_busy
            and not prompt_busy
        )
        selected_preset = self._preset_key(self.preset_choices.currentData())
        self.delete_preset_button.setEnabled(
            selected_preset is not None and selected_preset[0] == "user"
        )
        self._update_queue_summary()
        self._update_analysis_context_label()
        self._update_view_mode_availability()
        selected_edit_mode = self._current_edit_mode()
        if can_edit:
            display_radius = max(
                1,
                int(round(self.brush_radius_spin.value() * self.display_scale)),
            )
            self.image_view.set_edit_context(selected_edit_mode, display_radius)
        elif selected_edit_mode != "pan":
            # Busy rebuilds suspend drawing without changing the user's selected
            # persistent tool. Explicit analysis and image transitions choose
            # pan themselves.
            self.image_view.set_edit_context("pan", 1)
        self._update_dynamic_controls()

    def _update_dynamic_controls(self) -> None:
        relative_scale = str(self.scale_mode_combo.currentData()) == "relative_grid"
        self.reference_short_side_spin.setEnabled(relative_scale)
        intensity_space = str(self.intensity_space_combo.currentData())
        relative_intensity = intensity_space == "relative_rank"
        snr_intensity = intensity_space == "robust_snr"
        self.rank_tail_power_spin.setEnabled(relative_intensity)
        for widget in (
            self.black_percentile_spin,
            self.white_percentile_spin,
            self.gamma_spin,
        ):
            widget.setEnabled(not (relative_intensity or snr_intensity))
            widget.setToolTip(
                T(
                    "Robust SNR and relative rank use neither the black/white percentiles nor "
                    "gamma; they can be edited after switching to percentile linear.",
                )
                if relative_intensity or snr_intensity
                else ""
            )
        custom_rgb = str(self.signal_combo.currentData()) == "custom_rgb"
        for widget in (self.red_weight_spin, self.green_weight_spin, self.blue_weight_spin):
            widget.setEnabled(custom_rgb)

        self.denoise_strength_spin.setEnabled(
            str(self.denoise_method_combo.currentData()) != "none"
        )
        background_method = str(self.background_method_combo.currentData())
        background_enabled = background_method != "none"
        specialized_structure = background_method in {"vesselness", "log"}
        # setMaximum() may clamp a large radius and emit valueChanged.  The
        # surrounding method-change handler already reads the clamped value;
        # suppressing this nested signal prevents duplicate invalidation and
        # duplicate preview scheduling.
        radius_signals = self.background_radius_spin.blockSignals(True)
        try:
            self.background_radius_spin.setMaximum(
                128 if specialized_structure else 2048
            )
        finally:
            self.background_radius_spin.blockSignals(radius_signals)
        self.background_radius_spin.setEnabled(background_enabled)
        self.background_method_combo.setEnabled(not snr_intensity)
        self.background_strength_spin.setEnabled(background_enabled and not snr_intensity)
        structure_tip = (
            T(
                "The safe scale limit of the special structure methods is 128 px. A strength "
                "below 1 blends the original signal back in; above 1 it suppresses weak structure "
                "responses further.",
            )
            if specialized_structure
            else ""
        )
        self.background_radius_spin.setToolTip(structure_tip)
        self.background_strength_spin.setToolTip(structure_tip)

        method = str(self.threshold_method_combo.currentData())
        self.threshold_method_combo.setEnabled(not snr_intensity)
        self.snr_seed_sigma_spin.setEnabled(snr_intensity)
        self.manual_threshold_spin.setEnabled(method == "manual")
        self.threshold_percentile_spin.setEnabled(method == "percentile")
        local_enabled = method in {"local_mean", "local_gaussian", "mean_std", "sauvola"}
        self.local_window_spin.setEnabled(local_enabled)
        self.local_k_spin.setEnabled(local_enabled)
        self.grow_ratio_spin.setEnabled(
            method
            in {
                "hysteresis_otsu",
                "snr_hysteresis",
                "manual",
                "percentile",
                "local_mean",
                "local_gaussian",
                "mean_std",
                "sauvola",
            }
        )

        shape_profile = str(self.shape_profile_combo.currentData() or "any")
        concrete_shape = shape_profile != "any"
        continuous_shape = shape_profile in {"fiber", "patch", "ring"}
        ring_shape = shape_profile == "ring"
        self.shape_profile_strictness_combo.setEnabled(concrete_shape)

        if ring_shape and self.hole_area_spin.value() != 0:
            was_blocked = self.hole_area_spin.blockSignals(True)
            self.hole_area_spin.setValue(0)
            self.hole_area_spin.blockSignals(was_blocked)
        self.hole_area_spin.setEnabled(not ring_shape)
        self.hole_area_spin.setToolTip(
            T("The ring profile always keeps inner holes.")
            if ring_shape
            else T(
                "0 fills no extra holes; any other value is the largest hole area that may be "
                "filled.",
            )
        )
        if continuous_shape and self.split_touching_check.isChecked():
            was_blocked = self.split_touching_check.blockSignals(True)
            self.split_touching_check.setChecked(False)
            self.split_touching_check.blockSignals(was_blocked)
        self.split_touching_check.setEnabled(not continuous_shape)
        self.split_touching_check.setToolTip(
            T(
                "Fibre, patch and ring objects stay continuous; the watershed meant for compact "
                "objects is not used.",
            )
            if continuous_shape
            else T("Tries to separate touching compact objects at distance peaks.")
        )
        if ring_shape:
            compatibility_text = (
                T(
                    "Ring mode always keeps the central hole and turns off the watershed, so that "
                    "rings are neither filled nor cut up.",
                )
            )
        elif continuous_shape:
            compatibility_text = (
                T(
                    "Fibres and patches are treated as continuous structures with the watershed "
                    "always off; thickness, holes and outline values can still be adjusted "
                    "separately.",
                )
            )
        elif concrete_shape:
            compatibility_text = (
                T(
                    "Shape strictness combines circularity, solidity, elongation, holes and other "
                    "features; the numeric filters below are applied on top.",
                )
            )
        else:
            compatibility_text = (
                T(
                    "With any shape, only the numeric filters below are used, and separation of "
                    "touching objects can be enabled as needed.",
                )
            )
        self.shape_compatibility_label.setText(compatibility_text)

        split_enabled = (
            not continuous_shape and self.split_touching_check.isChecked()
        )
        self.peak_distance_spin.setEnabled(split_enabled)
        self.peak_height_spin.setEnabled(split_enabled)
        self._update_quick_dynamic_controls()

    @Slot(object)
    def _on_load_finished(self, payload: object) -> None:
        _kind, token = payload  # type: ignore[misc]
        if int(token) == self.load_token:
            self._load_busy = False
            self._update_action_state()

    @Slot(object)
    def _on_segment_finished(self, payload: object) -> None:
        _kind, serial = payload  # type: ignore[misc]
        finished_serial = int(serial)
        if finished_serial != self._active_segment_serial:
            return

        self._active_segment_serial = None
        pending_full = self._pending_segment_full
        self._pending_segment_full = None
        self._segment_busy = False
        if self.manual_serial == finished_serial:
            self.manual_serial = None
            self._manual_busy = False
        if self._edge_offset_serial == finished_serial:
            self._edge_offset_serial = None

        if pending_full is not None:
            # Start synchronously from the GUI thread.  The old worker has
            # already emitted all signals and left the active slot, so this
            # cannot recurse or add more than one runnable to the pool.
            self._start_segmentation(full=pending_full)
            return
        if self._pending_edge_rebuild:
            self._pending_edge_rebuild = False
            if (
                self._resume_base_result is not None
                and int(getattr(self.result, "edge_offset_px", 0)) != self._effective_edge_offset()
            ):
                self._start_edge_offset_rebuild()
                return
        self._update_action_state()

    def _show_object_table(self) -> None:
        if (
            self.result is None
            or self._load_busy
            or self._segment_busy
            or self._manual_busy
            or self._batch_worker is not None
            or self._active_export
            or self._export_prompt_open
        ):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(T("Manual review of candidate objects"))
        dialog.resize(1320, 680)
        layout = QVBoxLayout(dialog)
        # Building one widget per cell for tens of thousands of objects freezes
        # the UI for seconds and spikes memory.  Page the complete object set
        # instead of silently dropping small objects, because those are often
        # exactly the false positives that need manual exclusion.
        total_objects = len(self.result.objects)
        page_size = 2000
        page_count = max(1, int(math.ceil(total_objects / page_size)))
        note = QLabel(dialog)
        note.setWordWrap(True)
        layout.addWidget(note)
        navigation = QHBoxLayout()
        navigation.addWidget(QLabel(T("Show first:"), dialog))
        order_combo = FlexibleComboBox(dialog)
        order_combo.addItem(T("Suspected false positives (low confidence, small area)"), "review")
        order_combo.addItem(T("Small objects"), "small")
        order_combo.addItem(T("Large objects"), "large")
        order_combo.addItem(T("Object ID"), "id")
        navigation.addWidget(order_combo, 1)
        navigation.addWidget(QLabel(T("Page:"), dialog))
        page_spin = QSpinBox(dialog)
        page_spin.setRange(1, page_count)
        page_spin.setValue(1)
        page_spin.setEnabled(page_count > 1)
        navigation.addWidget(page_spin)
        page_total_label = QLabel(f"/ {page_count}", dialog)
        navigation.addWidget(page_total_label)
        layout.addLayout(navigation)
        columns = (
            ("id", "ID"),
            ("area_px", T("Area px²")),
            ("shape_class", T("Shape")),
            ("shape_tags", T("Compatible shapes")),
            ("mean_thickness_px", T("Mean thickness px")),
            ("elongation_index", T("Elongation index")),
            ("max_hole_fraction", T("Largest hole fraction")),
            ("normalized_mean", T("Normalized mean")),
            ("response_confidence", T("Object confidence")),
            ("saturated_fraction", T("Saturated fraction")),
            ("circularity", T("Circularity")),
            ("solidity", T("Solidity")),
            ("aspect_ratio", T("Aspect ratio")),
            ("touches_border", T("On border")),
        )
        table = QTableWidget(0, len(columns), dialog)
        table.setHorizontalHeaderLabels([title for _key, title in columns])
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setSortingEnabled(False)
        record_by_id: dict[int, dict[str, object]] = {}

        def finite_number(record: Mapping[str, object], key: str, fallback: float) -> float:
            raw_value = record.get(key, fallback)
            if raw_value is None or raw_value == "":
                return fallback
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return fallback
            return value if math.isfinite(value) else fallback

        ordered_cache: dict[str, list[dict[str, object]]] = {}

        def ordered_objects() -> list[dict[str, object]]:
            mode = str(order_combo.currentData() or "review")
            cached = ordered_cache.get(mode)
            if cached is not None:
                return cached
            records = list(self.result.objects)
            if mode == "small":
                ordered = sorted(
                    records,
                    key=lambda item: finite_number(item, "area_px", math.inf),
                )
            elif mode == "large":
                ordered = sorted(
                    records,
                    key=lambda item: finite_number(item, "area_px", -math.inf),
                    reverse=True,
                )
            elif mode == "id":
                ordered = sorted(
                    records,
                    key=lambda item: finite_number(item, "id", math.inf),
                )
            else:
                ordered = sorted(
                    records,
                    key=lambda item: (
                        finite_number(item, "response_confidence", math.inf),
                        finite_number(item, "area_px", math.inf),
                        finite_number(item, "id", math.inf),
                    ),
                )
            ordered_cache[mode] = ordered
            return ordered

        def refresh_page(_value: int | None = None) -> None:
            records = ordered_objects()
            page = page_spin.value()
            start = (page - 1) * page_size
            display_objects = records[start : start + page_size]
            table.blockSignals(True)
            table.setSortingEnabled(False)
            table.clearContents()
            table.setRowCount(len(display_objects))
            record_by_id.clear()
            for row, record in enumerate(display_objects):
                record_by_id[int(record.get("id", start + row + 1))] = record
                for column, (key, _title) in enumerate(columns):
                    value = record.get(key, "")
                    text = f"{value:.6g}" if isinstance(value, float) else str(value)
                    item = (
                        NumericTableWidgetItem(text)
                        if isinstance(value, (int, float))
                        else QTableWidgetItem(text)
                    )
                    if isinstance(value, (int, float)):
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        item.setData(Qt.UserRole, float(value))
                    table.setItem(row, column, item)
            table.setSortingEnabled(True)
            table.blockSignals(False)
            shown_from = start + 1 if display_objects else 0
            shown_to = start + len(display_objects)
            note.setText(
                T(
                    "{0:,} candidate objects in total, showing {1:,}–{2:,}; every object can be "
                    "reviewed page by page. Click a column header to sort this page; selecting a "
                    "row locates the object on the canvas; several rows can be selected and "
                    "excluded, and \"Undo\" in the main window restores them. Manual review is "
                    "not ground truth.",
                    total_objects,
                    shown_from,
                    shown_to,
                )
            )

        def change_order(_index: int | None = None) -> None:
            if page_spin.value() != 1:
                page_spin.setValue(1)
            else:
                refresh_page()

        order_combo.currentIndexChanged.connect(change_order)
        page_spin.valueChanged.connect(refresh_page)
        refresh_page()
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, dialog)
        exclude_button = buttons.addButton(T("Exclude selected objects"), QDialogButtonBox.ActionRole)
        can_exclude = (
            self.loaded is not None
            and not self.result.is_preview
            and not self._manual_busy
            and not self._segment_busy
        )
        exclude_button.setEnabled(can_exclude)
        if not can_exclude:
            exclude_button.setToolTip(T(
                "Excluding objects needs a full-resolution result; wait for the current "
                "computation to finish.",
            ))

        def selected_ids() -> list[int]:
            values: list[int] = []
            for model_index in table.selectionModel().selectedRows():
                item = table.item(model_index.row(), 0)
                if item is not None:
                    try:
                        values.append(int(float(item.text())))
                    except ValueError:
                        continue
            return sorted(set(values))

        def focus_selected() -> None:
            values = selected_ids()
            if not values:
                self.image_view.clear_focus()
                return
            record = record_by_id.get(values[0])
            if record is None:
                return
            scale = max(self.display_scale, 1e-9)
            self.image_view.focus_bbox(
                float(record.get("bbox_x_px", 0.0)) * scale,
                float(record.get("bbox_y_px", 0.0)) * scale,
                float(record.get("bbox_width_px", 1.0)) * scale,
                float(record.get("bbox_height_px", 1.0)) * scale,
            )

        def exclude_selected() -> None:
            values = selected_ids()
            if not values:
                QMessageBox.information(dialog, T("No object selected"), T(
                    "Select one or more candidate objects in the table first.",
                ))
                return
            answer = QMessageBox.question(
                dialog,
                T("Exclude candidate objects"),
                T(
                    "{0} selected object(s) will be excluded from the current mask.\nThis is a "
                    "manual review step and can be undone in the main window. Continue?",
                    len(values),
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            if self._exclude_object_ids(values):
                dialog.accept()

        table.itemSelectionChanged.connect(focus_selected)
        exclude_button.clicked.connect(exclude_selected)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        self.image_view.clear_focus()

    def _exclude_object_ids(self, object_ids: Iterable[int]) -> bool:
        if not self._can_edit_mask():
            return False
        assert self.result is not None
        requested = {int(value) for value in object_ids if int(value) > 0}
        records = [item for item in self.result.objects if int(item.get("id", 0)) in requested]
        if not records:
            return False
        height, width = self.result.mask.shape
        x0 = max(0, min(int(item.get("bbox_x_px", 0)) for item in records))
        y0 = max(0, min(int(item.get("bbox_y_px", 0)) for item in records))
        x1 = min(
            width,
            max(
                int(item.get("bbox_x_px", 0)) + int(item.get("bbox_width_px", 1))
                for item in records
            ),
        )
        y1 = min(
            height,
            max(
                int(item.get("bbox_y_px", 0)) + int(item.get("bbox_height_px", 1))
                for item in records
            ),
        )
        if x1 <= x0 or y1 <= y0:
            return False
        before = self.result.mask[y0:y1, x0:x1].astype(bool, copy=True)
        label_patch = self.result.labels[y0:y1, x0:x1]
        after = before & ~np.isin(label_patch, np.asarray(sorted(requested), dtype=label_patch.dtype))
        if np.array_equal(before, after):
            return False
        rollback = self._capture_manual_rollback(x0, y0, x1, y1, before)
        self.result.mask[y0:y1, x0:x1] = after
        self.result.manually_edited = True
        self._manual_dirty = True
        self._undo_stack.append((x0, y0, x1, y1, before, after.copy()))
        if len(self._undo_stack) > 60:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self.image_view.invalidate_contour_cache()
        self._render_display()
        self._start_manual_rebuild(rollback)
        return True

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not self._confirm_discard_manual(T(
            "Closing the program discards the manual edits that have not been exported yet.",
        )):
            event.ignore()
            return
        background_busy = bool(
            self._load_busy
            or self._segment_busy
            or self._manual_busy
            or self._active_segment_serial is not None
            or self._pending_segment_full is not None
            or self._crosstalk_busy
            or self._active_export
            or self._batch_worker is not None
            or self.thread_pool.activeThreadCount() > 0
        )
        if background_busy:
            if self._batch_worker is not None:
                self._batch_worker.cancel()
            # Closing is an explicit request not to start more analysis.  The
            # active runnable still finishes safely, but a coalesced follow-up
            # must not be launched behind the user's back.
            self.auto_timer.stop()
            self._pending_segment_full = None
            if (
                self._active_segment_serial is not None
                and self.segment_serial == self._active_segment_serial
            ):
                self.segment_serial += 1
            self.status_label.setText(T(
                "Background tasks are still finishing safely; the program can be closed when they "
                "are done",
            ))
            QMessageBox.information(
                self,
                T("Finishing background tasks"),
                T(
                    "Reading, analysis or export tasks are still running. The program finishes "
                    "them safely first; the batch has been asked to cancel. Please close the "
                    "window again in a moment.",
                ),
            )
            event.ignore()
            return
        self._save_ui_state()
        self.auto_timer.stop()
        self.image_view._ants_timer.stop()
        self.load_token += 1
        self.segment_serial += 1
        # The busy guard above ensures no active worker is abandoned while the
        # Python interpreter is being dismantled.  Clear only stale queued work.
        self.thread_pool.clear()
        self.thread_pool.waitForDone(1000)
        event.accept()


__all__ = ["ExportOptionsDialog", "UniversalFluoroWindow", "UniversalImageView"]

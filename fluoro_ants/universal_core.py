from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import operator
import os
import re
import stat
import time
from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# Abort the native decoder before a decompression bomb finishes allocating.
# universal_core is the first module every real entrypoint imports, so this env
# var is read in time for cv2's own sanity check.  The friendly 80,000,000-pixel
# message in _read_bytes still applies; this is only a hard backstop set at twice
# that limit so the decoder fails fast on hostile files instead of exhausting RAM.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 * 80_000_000))

import cv2  # noqa: E402 - must follow the OPENCV_IO_MAX_IMAGE_PIXELS backstop above
import numpy as np  # noqa: E402

from .i18n import N_, T
from .version import FLUOROANTS_RELEASE_VERSION

from .quality_control import assess_segmentation

UNIVERSAL_ALGORITHM_VERSION = "2.7.1"
CLASSIC_ENGINE = "classic"
LEARNED_ENGINE = "learned_v6"
AI_RECOGNITION_ENGINES = frozenset({LEARNED_ENGINE})
# Unit of the learned response map: local-contrast z score; candidates start at 3.5.
AI_RESPONSE_REFERENCE = {LEARNED_ENGINE: 3.5}


def normalize_engine_id(engine: object) -> object:
    """Return the current identifier for an engine name read from a file.

    Earlier builds stored the learned engine under other names that also end
    in ``_v6``; recipes and export records written by them still load.
    """

    if (
        isinstance(engine, str)
        and engine != CLASSIC_ENGINE
        and engine not in AI_RECOGNITION_ENGINES
        and engine.endswith("_v6")
    ):
        return LEARNED_ENGINE
    return engine


def ai_engine_module(engine: str):
    """Return the module that implements a learned engine (currently one)."""
    if engine not in AI_RECOGNITION_ENGINES:
        raise ValueError(T("Unknown recognition engine: {0}", engine))
    from . import learned_v6 as module
    return module
RECIPE_SCHEMA_VERSION = 3
QUICK_SCHEMA_VERSION = 3
CROSSTALK_CORRECTION_VERSION = 1
MERGE_IMAGE_FILENAME = "merge.png"
# Earlier builds could also write the merge image under these two names.  They
# are no longer produced, but files with these names are still recognised as
# exports so they are never queued as input images.
_FORMER_MERGE_IMAGE_FILENAMES = ("merge_with_selection.png", "positive_overlay.png")
MERGE_BASE_MODE_ANALYSIS_INPUT = "analysis_input"
MERGE_BASE_MODE_SOURCE_ORIGINAL = "source_original"
MERGE_BASE_MODES = frozenset(
    {MERGE_BASE_MODE_ANALYSIS_INPUT, MERGE_BASE_MODE_SOURCE_ORIGINAL}
)
EXPORT_LOCATION_FLAT = "flat"
EXPORT_LOCATION_MIRROR = "mirror"
EXPORT_LOCATION_SOURCE = "source"
EXPORT_LOCATION_MODES = frozenset(
    {EXPORT_LOCATION_FLAT, EXPORT_LOCATION_MIRROR, EXPORT_LOCATION_SOURCE}
)
EXPORT_BUNDLE_DIRECTORY = "directory"
EXPORT_BUNDLE_PREFIXED_FILES = "prefixed_files"
EXPORT_BUNDLE_MODES = frozenset(
    {EXPORT_BUNDLE_DIRECTORY, EXPORT_BUNDLE_PREFIXED_FILES}
)

# Stable IDs used by the GUI, CLI and batch API to request individual export
# artifacts.  Keep these identifiers independent of filenames: object labels
# deliberately switch between TIFF and NPY according to the label range.
EXPORT_ARTIFACT_POSITIVE_MASK = "positive_mask"
EXPORT_ARTIFACT_OBJECT_LABELS = "object_labels"
EXPORT_ARTIFACT_MERGE = "merge"
EXPORT_ARTIFACT_BACKGROUND_CORRECTED_RESPONSE = "background_corrected_response"
EXPORT_ARTIFACT_QUALITY_REPORT = "quality_report"
EXPORT_ARTIFACT_OBJECTS = "objects"
EXPORT_ARTIFACT_SUMMARY = "summary"
EXPORT_ARTIFACT_RECIPE = "recipe"
EXPORT_ARTIFACT_PROJECT = "project"

# Tuple order is the canonical presentation/write order.  ``None`` at either
# public export entrypoint expands to this complete set.
DEFAULT_EXPORT_ARTIFACTS: tuple[str, ...] = (
    EXPORT_ARTIFACT_POSITIVE_MASK,
    EXPORT_ARTIFACT_OBJECT_LABELS,
    EXPORT_ARTIFACT_MERGE,
    EXPORT_ARTIFACT_BACKGROUND_CORRECTED_RESPONSE,
    EXPORT_ARTIFACT_QUALITY_REPORT,
    EXPORT_ARTIFACT_OBJECTS,
    EXPORT_ARTIFACT_SUMMARY,
    EXPORT_ARTIFACT_RECIPE,
    EXPORT_ARTIFACT_PROJECT,
)
EXPORT_ARTIFACT_IDS = frozenset(DEFAULT_EXPORT_ARTIFACTS)

MAX_ENCODED_IMAGE_BYTES = 1024 * 1024 * 1024
MAX_DECODED_IMAGE_PIXELS = 80_000_000
# Crosstalk subtraction holds the target and reference signals as float32 at
# the same time.  Keeping a separate ceiling avoids turning two otherwise valid
# 80 MP inputs into a multi-gigabyte allocation with an unhelpful crash.
MAX_CROSSTALK_CORRECTION_PIXELS = 40_000_000
MAX_CROSSTALK_COEFFICIENT = 5.0
CROSSTALK_REFERENCE_SIGNAL_MODES = frozenset(
    {"auto", "average", "max_rgb", "red", "green", "blue"}
)
SUPPORTED_IMAGE_SUFFIXES = {
    ".tif",
    ".tiff",
    ".png",
    ".bmp",
    ".jpg",
    ".jpeg",
    ".webp",
}

SIGNAL_MODE_LABELS: dict[str, str] = {
    "auto": T("Auto"),
    "average": T("RGB average"),
    "luma": T("Luma-weighted"),
    "max_rgb": T("RGB maximum"),
    "red": T("Red channel"),
    "green": T("Green channel"),
    "blue": T("Blue channel"),
    "hsv_value": T("HSV value"),
    "hsv_saturation": T("HSV saturation"),
    "custom_rgb": T("Custom RGB weights"),
}

INTENSITY_SPACE_LABELS: dict[str, str] = {
    "percentile_linear": T("Percentile linear (black/white point + gamma)"),
    "relative_rank": T("Relative rank (robust to global brightness / gamma)"),
    "robust_snr": T("Robust SNR (may return an empty selection)"),
}

DENOISE_METHOD_LABELS: dict[str, str] = {
    "none": T("No denoising"),
    "gaussian": T("Gaussian filter"),
    "median": T("Median filter"),
    "bilateral": T("Bilateral filter"),
    "nl_means": T("Non-local means"),
}

BACKGROUND_METHOD_LABELS: dict[str, str] = {
    "none": T("No correction"),
    "morph_open": T("Morphological background"),
    "gaussian": T("Large-scale Gaussian background"),
    "multiscale": T("Multi-scale background correction"),
    "flat_field": T("Relative flat-field correction"),
    "dog": T("Difference-of-Gaussians enhancement"),
    "vesselness": T("Tubular ridge enhancement"),
    "log": T("Spot enhancement (LoG)"),
}

THRESHOLD_METHOD_LABELS: dict[str, str] = {
    "snr_hysteresis": T("Fixed-SNR hysteresis"),
    "hysteresis_otsu": T("Otsu hysteresis growth"),
    "otsu": T("Otsu global threshold"),
    "triangle": T("Triangle global threshold"),
    "manual": T("Manual normalized threshold"),
    "percentile": T("Intensity percentile threshold"),
    "local_mean": T("Local mean threshold"),
    "local_gaussian": T("Local Gaussian threshold"),
    "mean_std": T("Local mean + standard deviation"),
    "sauvola": T("Sauvola local threshold"),
}

OBJECT_SHAPE_PROFILE_LABELS: dict[str, str] = {
    "any": T("Any / numeric filters only"),
    "round": T("Round / compact"),
    "ellipse": T("Elliptic / spindle"),
    "fiber": T("Elongated / strip / fibre"),
    "patch": T("Patch / irregular"),
    "ring": T("Ring / hole in the centre"),
}

OBJECT_SHAPE_STRICTNESS_LABELS: dict[str, str] = {
    "loose": T("Loose (allows neighbouring shapes)"),
    "standard": T("Standard"),
    "strict": T("Strict"),
}

BACKGROUND_CEILING_MODE_LABELS: dict[str, str] = {
    "off": T("Off"),
    "absolute": T("Absolute grey-level ceiling (whole batch shares one exposure)"),
    "auto": T("Automatic per image (two-mode split of the broad background)"),
}

SPATIAL_SCALE_MODES = frozenset({"relative_grid", "pixel_locked"})

_EXPORT_DIRECTORY_RE = re.compile(r"(?i)^.+_fluoroants(?:_\d{3,})?$")
_EXPORT_STAGING_DIRECTORY_RE = re.compile(
    r"(?i)^\..+_fluoroants(?:_\d{3,})?\.exporting$"
)
_EXPORT_MARKER_FILENAMES = frozenset(
    {
        "INCOMPLETE.txt",
        "positive_mask.tif",
        "object_labels.tif",
        "object_labels.npy",
        MERGE_IMAGE_FILENAME,
        *_FORMER_MERGE_IMAGE_FILENAMES,
        "background_corrected_response.png",
        "quality_report.json",
        "objects.csv",
        "summary.csv",
        "recipe.json",
        "project.json",
    }
)
_PREFIXED_EXPORT_IMAGE_RE = re.compile(
    r"(?i)^.+_fluoroants(?:_\d{3,})?_(?:"
    + "|".join(
        re.escape(name)
        for name in sorted(
            {
                "positive_mask.tif",
                "object_labels.tif",
                MERGE_IMAGE_FILENAME,
                *_FORMER_MERGE_IMAGE_FILENAMES,
                "background_corrected_response.png",
            }
        )
    )
    + r")$"
)
_KERNEL_CACHE: dict[int, np.ndarray] = {}

_add_reduce = np.add.reduce
_maximum_reduce = np.maximum.reduce


def _exact_mean(values: np.ndarray, count: int) -> float:
    """``float(values.mean())`` for a 1-D array, without ``np.mean``'s overhead.

    ``ndarray.mean`` sums with the array's own dtype (pairwise, in C), divides
    that scalar by an ``intp`` count (which promotes float32 to float64) and
    casts the quotient back to the array dtype.  Reproducing those exact steps
    keeps every exported per-object statistic bit-for-bit identical while
    skipping the Python-level bookkeeping that dominated the old hot path.
    """

    kind = values.dtype
    if kind == np.float32:
        return float(np.float32(np.float64(_add_reduce(values)) / np.float64(count)))
    if kind == np.float64:
        return float(np.float64(_add_reduce(values)) / np.float64(count))
    return float(values.mean())


# Hessian/LoG filters otherwise create several full-frame float32 temporaries.
# Small images keep the single-pass path; larger frames are processed in
# overlapping tiles so a 4K or larger source does not briefly require more than
# a gigabyte of working memory.
_STRUCTURE_TILE_THRESHOLD_PIXELS = 2_500_000
_STRUCTURE_TILE_SIDE = 1024

# Matches Windows drive paths, UNC shares and POSIX paths so batch/error text can
# be scrubbed of user directories at the source, independent of the caller.
_LOCAL_PATH_RE = re.compile(
    r"(?<![\w.])(?:[A-Za-z]:[\\/]|\\\\|/)(?:[^\s:;,]+[\\/])*[^\s:;,]*"
)


def scrub_local_paths(message: str) -> str:
    """Replace absolute local paths in a message with a neutral placeholder."""

    return _LOCAL_PATH_RE.sub(T("local path"), str(message))


def default_opencv_thread_limit() -> int:
    """Worker count that keeps one core free for the desktop event loop.

    Every OpenCV kernel used here (blur, morphology, resize, distance
    transform, labelling, thresholding, denoising) is row-partitioned and
    produces bit-identical output whatever the thread count, so raising the
    limit on multi-core machines only changes speed, never results.
    """

    cores = os.cpu_count() or 2
    return max(2, min(8, cores - 1))


def configure_opencv_threads(max_threads: int | None = None) -> int:
    """Apply the OpenCV worker limit; ``None`` picks the desktop default."""

    if max_threads is None:
        max_threads = default_opencv_thread_limit()
    limit = max(1, min(int(max_threads), 32))
    if cv2.getNumThreads() != limit:
        cv2.setNumThreads(limit)
    return int(cv2.getNumThreads())


# Apply the desktop-safe default before decoding or segmentation starts.
configure_opencv_threads()


def contours_from_mask(mask: np.ndarray, simplify: float = 0.7) -> list[np.ndarray]:
    """Return optionally simplified contours for a two-dimensional mask."""

    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if simplify <= 0:
        return contours
    return [
        contour
        if len(contour) < 3
        else cv2.approxPolyDP(contour, simplify, True)
        for contour in contours
    ]


def adjust_mask_boundary(mask: np.ndarray, signed_pixels: int) -> np.ndarray:
    """Expand or contract a 2-D mask by an exact source-pixel distance.

    Positive values expand the foreground and negative values contract it.
    Distances are measured between pixel centres using the Euclidean metric;
    pixels outside the image are treated as background during contraction.
    The input is never modified and the returned array is always boolean with
    the same shape.
    """

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(T("The selection mask must be a 2-D array"))
    if isinstance(signed_pixels, (bool, np.bool_)):
        raise TypeError(T(
            "The expand/shrink distance of the selection must be a whole number of pixels",
        ))
    try:
        delta = operator.index(signed_pixels)
    except TypeError as exc:
        raise TypeError(T(
            "The expand/shrink distance of the selection must be a whole number of pixels",
        )) from exc

    if delta == 0 or not bool(source.any()):
        return source.copy()

    radius = abs(delta)
    if delta > 0:
        # distanceTransform measures distance to zero.  Inverting the Mask makes
        # every selected pixel a zero-distance seed and naturally clips growth
        # at the canvas boundary.
        distance = cv2.distanceTransform(
            (~source).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        return source | (distance <= float(radius))

    # Add an explicit zero border so a selection touching (or filling) the
    # canvas contracts from the canvas edge just like every other boundary.
    padded = np.pad(
        source.astype(np.uint8),
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=0,
    )
    distance = cv2.distanceTransform(
        padded,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )[1:-1, 1:-1]
    return source & (distance > float(radius))


@dataclass
class UniversalParams:
    """Complete, serializable recipe for one 2-D positive-signal segmentation."""

    # Spatial parameters are source pixels unless a recipe opts into the
    # canonical reference grid, as every built-in recipe does.
    scale_mode: str = "pixel_locked"
    reference_short_side: int = 1024

    signal_mode: str = "auto"
    invert: bool = False
    red_weight: float = 1.0
    green_weight: float = 1.0
    blue_weight: float = 1.0

    # Built-in recipes choose relative_rank or robust_snr explicitly.
    intensity_space: str = "percentile_linear"
    rank_tail_power: float = 4.0
    # Seed evidence in robust residual-noise standard deviations.  It is only
    # used by the robust_snr / snr_hysteresis path.
    snr_seed_sigma: float = 5.0

    black_percentile: float = 0.2
    white_percentile: float = 99.8
    gamma: float = 1.0

    denoise_method: str = "gaussian"
    denoise_strength: float = 0.8

    background_method: str = "multiscale"
    background_radius: int = 24
    background_strength: float = 1.0

    threshold_method: str = "hysteresis_otsu"
    sensitivity: int = 50
    manual_threshold: float = 0.25
    threshold_percentile: float = 98.5
    local_window: int = 51
    local_k: float = 0.20
    grow_ratio: float = 0.60

    open_radius: int = 0
    close_radius: int = 1
    erode_radius: int = 0
    dilate_radius: int = 0
    hole_area: int = 4

    min_area: int = 8
    max_area: int = 0
    remove_border: bool = False

    split_touching: bool = False
    watershed_peak_distance: int = 6
    watershed_peak_height: float = 0.28

    shape_profile: str = "any"
    shape_profile_strictness: str = "standard"
    min_thickness: float = 0.0
    max_thickness: float = 0.0
    min_circularity: float = 0.0
    min_solidity: float = 0.0
    max_eccentricity: float = 1.0
    # 1000 is effectively unbounded for normal images while still giving the
    # professional control a finite, serializable ceiling.  The former default
    # of 100 silently discarded perfectly valid ultra-thin fibers even when the
    # user selected "No limit".
    max_aspect_ratio: float = 1000.0
    min_mean_intensity: float = 0.0
    # Whole-object confidence after background correction.  The score is the
    # object's mean response divided by the image-wide Otsu response reference.
    # Zero disables the gate so older recipes remain bit-for-bit compatible.
    min_response_confidence: float = 0.0

    # --- Shared absolute decision scale ---------------------------------------
    # robust_snr normally divides the local residual by *this image's own*
    # residual MAD.  That is an absolute rule within one image but an adaptive
    # one across a batch: two fields of the same sample, imaged and exported
    # identically, get different decision thresholds purely because one of them
    # contains more textured background.  Where that texture is autofluorescence
    # rather than signal, the effect runs backwards — the detector becomes
    # permissive exactly where the background is quiet and strict where the
    # artefact is loud.
    #
    # When ``noise_reference_value`` is greater than zero it replaces the
    # per-image MAD with one fixed scale expressed in *source grey levels*, so
    # the seed becomes the absolute contrast ``snr_seed_sigma * value`` and is
    # identical for every image in the batch.  Zero keeps the per-image
    # behaviour.
    noise_reference_value: float = 0.0

    # Compartment gate on the *broad-scale* local background.  Broad glow is
    # sample autofluorescence, folds, thickened edges and similar continuously
    # bright regions; it is a property of the compartment, not of any object
    # inside it, so objects are excluded by where they sit.
    #
    #   "off"       - disabled (default)
    #   "absolute"  - exclude background brighter than ``background_ceiling``
    #                 grey levels.  Requires the whole batch to share one
    #                 exposure and one export curve.
    #   "auto"      - split this image's own broad background into its two
    #                 modes (Otsu) and exclude the bright one.  This is NOT a
    #                 fixed percentile: it finds the actual dim/bright
    #                 boundary, so a field with no bright compartment keeps
    #                 almost everything.  Use it when the images were adjusted
    #                 one by one and no shared grey axis exists.
    background_ceiling_mode: str = "off"
    background_ceiling: float = 0.0

    # Spatial scale, as a radius in pixels, at which the compartment is judged.
    # This is deliberately independent of ``background_radius``: the object
    # background scale and the compartment scale are different physical things.
    # A broad continuous band is hundreds of pixels across and wants a large
    # scale; fine bright strands are only tens of pixels across, and judging
    # them at the band scale smears the gate into amorphous blobs that swallow
    # the surrounding field instead of following the strands.  Zero keeps the
    # legacy behaviour of deriving it from ``background_radius``.
    background_ceiling_scale: int = 0
    # Keep the default classic path and old recipes unchanged. The learned
    # engine always processes the original supported source grid.
    recognition_engine: str = "classic"
    # Learned model only: how far the main threshold is moved from its
    # calibrated value.  Positive accepts candidates that score lower.
    learned_sensitivity: float = 0.0

    def validated(
        self,
        image_shape: tuple[int, int] | None = None,
    ) -> "UniversalParams":
        if self.recognition_engine not in {CLASSIC_ENGINE, *AI_RECOGNITION_ENGINES}:
            raise ValueError(T(
                "Unknown recognition mode; choose \"Classic recognition\" or \"Learned model\".",
            ))
        if image_shape is None:
            max_radius = 2048
            max_area = 100_000_000
            min_side = 4097
        else:
            height, width = (max(1, int(v)) for v in image_shape[:2])
            min_side = min(height, width)
            max_radius = max(1, min(2048, max(height, width) // 2))
            max_area = height * width

        black = float(np.clip(self.black_percentile, 0.0, 49.9))
        white = float(np.clip(self.white_percentile, 50.0, 100.0))
        if white <= black + 0.01:
            white = min(100.0, black + 0.01)

        local_window = max(3, min(int(self.local_window), 4095, max(3, min_side)))
        if local_window % 2 == 0:
            local_window = local_window - 1 if local_window >= min_side else local_window + 1
        local_window = max(3, local_window)

        signal_mode = (
            self.signal_mode if self.signal_mode in SIGNAL_MODE_LABELS else "auto"
        )
        intensity_space = (
            self.intensity_space
            if self.intensity_space in INTENSITY_SPACE_LABELS
            else "percentile_linear"
        )
        denoise_method = (
            self.denoise_method
            if self.denoise_method in DENOISE_METHOD_LABELS
            else "gaussian"
        )
        background_method = (
            self.background_method
            if self.background_method in BACKGROUND_METHOD_LABELS
            else "multiscale"
        )
        if intensity_space == "robust_snr" and background_method != "none":
            background_method = "gaussian"
        background_radius_limit = (
            min(max_radius, 128)
            if background_method in {"vesselness", "log"}
            else max_radius
        )
        threshold_method = (
            self.threshold_method
            if self.threshold_method in THRESHOLD_METHOD_LABELS
            else "hysteresis_otsu"
        )
        if intensity_space == "robust_snr":
            threshold_method = "snr_hysteresis"

        min_area_value = max(0, min(int(self.min_area), max_area))
        max_area_value = max(0, min(int(self.max_area), max_area))
        if max_area_value and max_area_value < min_area_value:
            max_area_value = min_area_value
        shape_profile = (
            self.shape_profile
            if self.shape_profile in OBJECT_SHAPE_PROFILE_LABELS
            else "any"
        )
        shape_profile_strictness = (
            self.shape_profile_strictness
            if self.shape_profile_strictness in OBJECT_SHAPE_STRICTNESS_LABELS
            else "standard"
        )
        scale_mode = (
            self.scale_mode
            if self.scale_mode in SPATIAL_SCALE_MODES
            else "pixel_locked"
        )
        reference_short_side = max(
            16,
            min(int(self.reference_short_side), 32768),
        )
        min_thickness = float(np.clip(self.min_thickness, 0.0, max_radius * 2))
        max_thickness = float(np.clip(self.max_thickness, 0.0, max_radius * 2))
        if max_thickness and max_thickness < min_thickness:
            max_thickness = min_thickness

        return replace(
            self,
            scale_mode=scale_mode,
            reference_short_side=reference_short_side,
            signal_mode=signal_mode,
            invert=bool(self.invert),
            red_weight=float(np.clip(self.red_weight, 0.0, 10.0)),
            green_weight=float(np.clip(self.green_weight, 0.0, 10.0)),
            blue_weight=float(np.clip(self.blue_weight, 0.0, 10.0)),
            intensity_space=intensity_space,
            rank_tail_power=float(np.clip(self.rank_tail_power, 1.0, 12.0)),
            snr_seed_sigma=float(np.clip(self.snr_seed_sigma, 2.0, 12.0)),
            black_percentile=black,
            white_percentile=white,
            gamma=float(np.clip(self.gamma, 0.05, 8.0)),
            denoise_method=denoise_method,
            denoise_strength=float(np.clip(self.denoise_strength, 0.0, 30.0)),
            background_method=background_method,
            background_radius=max(
                0,
                min(int(self.background_radius), background_radius_limit),
            ),
            background_strength=(
                1.0
                if intensity_space == "robust_snr"
                else float(np.clip(self.background_strength, 0.0, 3.0))
            ),
            threshold_method=threshold_method,
            sensitivity=max(0, min(int(self.sensitivity), 100)),
            manual_threshold=float(np.clip(self.manual_threshold, 0.0, 1.0)),
            threshold_percentile=float(
                np.clip(self.threshold_percentile, 0.0, 100.0)
            ),
            local_window=local_window,
            local_k=float(np.clip(self.local_k, -2.0, 2.0)),
            grow_ratio=float(np.clip(self.grow_ratio, 0.01, 1.0)),
            open_radius=max(0, min(int(self.open_radius), min(max_radius, 256))),
            close_radius=max(0, min(int(self.close_radius), min(max_radius, 256))),
            erode_radius=max(0, min(int(self.erode_radius), min(max_radius, 128))),
            dilate_radius=max(0, min(int(self.dilate_radius), min(max_radius, 128))),
            hole_area=max(0, min(int(self.hole_area), max_area)),
            min_area=min_area_value,
            max_area=max_area_value,
            remove_border=bool(self.remove_border),
            split_touching=bool(self.split_touching),
            watershed_peak_distance=max(
                1,
                min(int(self.watershed_peak_distance), min(max_radius, 256)),
            ),
            watershed_peak_height=float(
                np.clip(self.watershed_peak_height, 0.02, 0.95)
            ),
            shape_profile=shape_profile,
            shape_profile_strictness=shape_profile_strictness,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            min_circularity=float(np.clip(self.min_circularity, 0.0, 1.0)),
            min_solidity=float(np.clip(self.min_solidity, 0.0, 1.0)),
            max_eccentricity=float(np.clip(self.max_eccentricity, 0.0, 1.0)),
            max_aspect_ratio=float(np.clip(self.max_aspect_ratio, 1.0, 1000.0)),
            min_mean_intensity=float(
                np.clip(self.min_mean_intensity, 0.0, 1.0)
            ),
            min_response_confidence=float(
                np.clip(self.min_response_confidence, 0.0, 5.0)
            ),
            noise_reference_value=(
                0.0
                if not math.isfinite(float(self.noise_reference_value))
                or float(self.noise_reference_value) <= 0.0
                else float(np.clip(self.noise_reference_value, 1e-6, 1e9))
            ),
            background_ceiling_mode=(
                self.background_ceiling_mode
                if self.background_ceiling_mode in BACKGROUND_CEILING_MODE_LABELS
                else "off"
            ),
            background_ceiling_scale=max(
                0,
                min(int(self.background_ceiling_scale), max_radius),
            ),
            background_ceiling=(
                0.0
                if not math.isfinite(float(self.background_ceiling))
                or float(self.background_ceiling) <= 0.0
                else float(np.clip(self.background_ceiling, 1e-6, 1e9))
            ),
        )

    def scaled_spatial(self, scale: float) -> "UniversalParams":
        """Scale every spatial quantity once; preserve dimensionless controls.

        ``scale`` maps the coordinate system in which this recipe is expressed
        to the coordinate system where segmentation will run.  Lengths use the
        linear factor and areas use its square.  The scale mode and reference
        grid remain recipe metadata and are deliberately not rewritten.
        """

        scale = float(np.clip(scale, 1e-4, 1e4))
        area_scale = scale * scale

        def radius(value: int, minimum_if_enabled: int = 0) -> int:
            if value <= 0:
                return 0
            scaled = int(math.floor(value * scale + 0.5))
            return max(minimum_if_enabled, scaled)

        def area(value: int) -> int:
            if value <= 0:
                return 0
            return max(1, int(round(value * area_scale)))

        local_window = max(3, int(round(self.local_window * scale)))
        if local_window % 2 == 0:
            local_window += 1
        return replace(
            self,
            denoise_strength=self.denoise_strength * scale,
            background_radius=radius(self.background_radius, 1),
            local_window=local_window,
            open_radius=radius(self.open_radius),
            close_radius=radius(self.close_radius),
            erode_radius=radius(self.erode_radius),
            dilate_radius=radius(self.dilate_radius),
            hole_area=area(self.hole_area),
            min_area=area(self.min_area),
            max_area=area(self.max_area),
            watershed_peak_distance=radius(self.watershed_peak_distance, 1),
            background_ceiling_scale=radius(self.background_ceiling_scale, 1),
            min_thickness=self.min_thickness * scale,
            max_thickness=self.max_thickness * scale,
        )

    def scaled_for_preview(self, scale: float) -> "UniversalParams":
        """Backward-compatible downscale for the legacy preview path."""

        return self.scaled_spatial(float(np.clip(scale, 1e-4, 1.0)))


def enforce_shape_compatibility(params: UniversalParams) -> UniversalParams:
    """Apply topology guarantees promised by the selected shape profile.

    This lives in the core rather than only in the GUI so imported professional
    recipes, batch jobs, and direct API calls cannot accidentally fill a ring or
    watershed a continuous fiber/patch into fragments.
    """

    if params.shape_profile not in {"fiber", "patch", "ring"}:
        return params
    changes: dict[str, object] = {"split_touching": False}
    if params.shape_profile == "ring":
        changes["hole_area"] = 0
    return replace(params, **changes)


@dataclass(frozen=True)
class PresetDefinition:
    name: str
    category: str
    description: str
    params: UniversalParams


@dataclass
class LoadedImage:
    path: Path
    image: np.ndarray
    metadata: dict[str, object]

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])


@dataclass(frozen=True)
class ExportReceipt:
    """Describe one flat file bundle written directly into its parent folder."""

    parent: Path
    prefix: str
    files: tuple[Path, ...]

    @property
    def label_path(self) -> Path:
        """Return a display-only path identifying the shared file prefix."""

        return self.parent / self.prefix


@dataclass(frozen=True)
class CrosstalkCorrection:
    """One reproducible pre-analysis subtraction setting.

    The reference image is supplied separately so the immutable settings can
    be reused for a folder of paired target/reference images.
    """

    coefficient: float = 0.30
    reference_signal_mode: str = "auto"
    preserve_reference_background: bool = True
    background_percentile: float = 10.0

    def validated(self) -> "CrosstalkCorrection":
        if isinstance(self.coefficient, (bool, np.bool_)):
            raise TypeError(T("The crosstalk coefficient must be a number between 0 and 5"))
        coefficient = float(self.coefficient)
        if not math.isfinite(coefficient) or not 0.0 <= coefficient <= MAX_CROSSTALK_COEFFICIENT:
            raise ValueError(T("The crosstalk coefficient must be between 0 and 5"))
        signal_mode = str(self.reference_signal_mode).strip().casefold()
        if signal_mode not in CROSSTALK_REFERENCE_SIGNAL_MODES:
            choices = T(", ").join(sorted(CROSSTALK_REFERENCE_SIGNAL_MODES))
            raise ValueError(T(
                "Unknown crosstalk reference channel: {0}; allowed values: {1}",
                signal_mode or T("empty"),
                choices,
            ))
        if not isinstance(self.preserve_reference_background, (bool, np.bool_)):
            raise TypeError(T("Background protection must be a boolean"))
        if isinstance(self.background_percentile, (bool, np.bool_)):
            raise TypeError(T(
                "The background percentile of the crosstalk reference must be a number between 0 "
                "and 50",
            ))
        percentile = float(self.background_percentile)
        if not math.isfinite(percentile) or not 0.0 <= percentile <= 50.0:
            raise ValueError(T(
                "The background percentile of the crosstalk reference must be between 0 and 50",
            ))
        return replace(
            self,
            coefficient=coefficient,
            reference_signal_mode=signal_mode,
            preserve_reference_background=bool(self.preserve_reference_background),
            background_percentile=percentile,
        )


@dataclass
class UniversalResult:
    mask: np.ndarray
    labels: np.ndarray
    objects: list[dict[str, object]]
    signal: np.ndarray
    normalized_signal: np.ndarray
    response: np.ndarray
    threshold_value: float
    low_threshold: float
    positive_fraction: float
    params: UniversalParams
    signal_mode_used: str
    input_black_value: float
    input_white_value: float
    processing_seconds: float
    analysis_scale: float = 1.0
    source_shape: tuple[int, int] | None = None
    manually_edited: bool = False
    # Input diagnostics are advisory only and never alter the mask.  Defaults
    # keep external/manual UniversalResult construction source compatible.
    input_saturation_fraction: float = 0.0
    input_high_plateau_fraction: float = 0.0
    input_largest_tie_fraction: float = 0.0
    input_unique_levels: int = 0
    intensity_warnings: tuple[str, ...] = ()
    # Selection edge offset applied *after* recognition (see
    # ``apply_edge_offset``).  It is a post-processing display/export setting,
    # deliberately not part of ``UniversalParams`` and never written into a
    # recipe.  ``edge_offset_px`` is the requested source-pixel distance;
    # ``edge_offset_grid_px`` is what was actually applied on this result's
    # analysis grid (a coarse preview may round a small offset to zero).
    edge_offset_px: int = 0
    edge_offset_grid_px: int = 0
    review_mask: np.ndarray | None = None
    ai_metadata: dict[str, object] | None = None
    # Unoccluded review base survives brushing a reviewed pixel into the main
    # selection and undoing that edit. It is never part of main measurements.
    review_base_mask: np.ndarray | None = None

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def is_preview(self) -> bool:
        if self.source_shape is not None:
            source_shape = tuple(int(value) for value in self.source_shape[:2])
            if tuple(int(value) for value in self.mask.shape[:2]) != source_shape:
                return True
        return self.analysis_scale < 0.999


_BASE = UniversalParams(
    scale_mode="relative_grid",
    reference_short_side=1024,
    intensity_space="relative_rank",
)


def _preset(name: str, category: str, description: str, **changes: Any) -> PresetDefinition:
    return PresetDefinition(
        name=name,
        category=category,
        description=description,
        params=replace(_BASE, **changes),
    )


BUILTIN_PRESETS: dict[str, PresetDefinition] = {
    definition.name: definition
    for definition in (
        _preset(
            N_("General balanced"),
            N_("General"),
            N_(
                "Judges clearly bright targets against the local background noise; a negative "
                "image, or one without enough evidence, may give an empty selection.",
            ),
            intensity_space="robust_snr",
            background_method="gaussian",
            background_radius=48,
            background_strength=1.0,
            threshold_method="snr_hysteresis",
            snr_seed_sigma=5.0,
            sensitivity=50,
            grow_ratio=0.44,
            min_area=12,
            min_mean_intensity=0.0,
            min_response_confidence=0.0,
        ),
        _preset(
            N_("Strict positive"),
            N_("General"),
            N_(
                "Raises the SNR evidence threshold and tightens edges; keeps clear positives "
                "first.",
            ),
            intensity_space="robust_snr",
            background_method="gaussian",
            background_radius=48,
            background_strength=1.0,
            threshold_method="snr_hysteresis",
            snr_seed_sigma=6.0,
            sensitivity=50,
            grow_ratio=0.47,
            min_area=12,
            min_mean_intensity=0.0,
            min_response_confidence=0.0,
        ),
        _preset(
            N_("Weak signal first"),
            N_("General"),
            N_("Lowers the threshold and widens weak edges; for images with slightly low SNR."),
            sensitivity=72,
            grow_ratio=0.43,
            denoise_strength=0.55,
            background_radius=32,
            min_area=4,
        ),
        _preset(
            N_("Very low false positives"),
            N_("General"),
            N_(
                "Uses a very high SNR evidence threshold and structural cleanup; for negative "
                "controls or when high specificity is needed.",
            ),
            intensity_space="robust_snr",
            background_method="gaussian",
            background_radius=48,
            background_strength=1.0,
            threshold_method="snr_hysteresis",
            snr_seed_sigma=7.0,
            sensitivity=50,
            grow_ratio=0.57,
            denoise_method="median",
            denoise_strength=1.2,
            open_radius=1,
            min_area=20,
            min_mean_intensity=0.0,
            min_response_confidence=0.0,
        ),
        _preset(
            N_("Sparse tiny spots"),
            N_("Spots"),
            N_("Keeps a small number of small, well-separated bright spots."),
            rank_tail_power=8.0,
            background_radius=10,
            denoise_strength=0.45,
            sensitivity=56,
            grow_ratio=0.56,
            min_area=2,
            max_area=180,
            close_radius=0,
            hole_area=0,
        ),
        _preset(
            N_("Dense tiny spots"),
            N_("Spots"),
            N_("For many adjacent small spots; uses a distance watershed to help split them."),
            rank_tail_power=8.0,
            background_radius=9,
            denoise_method="median",
            denoise_strength=0.8,
            sensitivity=59,
            grow_ratio=0.62,
            min_area=2,
            max_area=240,
            close_radius=0,
            split_touching=True,
            watershed_peak_distance=3,
            watershed_peak_height=0.22,
        ),
        _preset(
            N_("Faint small spots"),
            N_("Spots"),
            N_(
                "Looks for small spots of low intensity first; check false positives against a "
                "negative control.",
            ),
            background_radius=12,
            denoise_strength=0.35,
            sensitivity=78,
            grow_ratio=0.48,
            min_area=2,
            max_area=160,
            close_radius=0,
        ),
        _preset(
            N_("Spot enhancement"),
            N_("Special · spots"),
            N_(
                "Special tool: use only when the targets are known to be spots. Scale-matched LoG "
                "enhancement brings out spot cores, but may also respond to spotty texture or to "
                "bright cores inside larger targets.",
            ),
            background_method="log",
            background_radius=4,
            background_strength=1.0,
            denoise_method="gaussian",
            denoise_strength=0.35,
            sensitivity=55,
            grow_ratio=0.50,
            min_area=2,
            max_area=400,
            close_radius=0,
            hole_area=0,
        ),
        _preset(
            N_("Medium compact regions"),
            N_("Regions"),
            N_("For compact objects of medium size, round or elliptic."),
            background_radius=28,
            denoise_strength=1.0,
            sensitivity=50,
            min_area=25,
            max_area=8000,
            close_radius=2,
            hole_area=30,
            shape_profile="round",
            shape_profile_strictness="loose",
            min_circularity=0.12,
            min_solidity=0.58,
        ),
        _preset(
            N_("Large masses"),
            N_("Regions"),
            N_("Uses a larger background scale and keeps wide continuous regions."),
            background_method="gaussian",
            background_radius=96,
            denoise_strength=1.6,
            sensitivity=57,
            grow_ratio=0.42,
            min_area=220,
            close_radius=4,
            hole_area=300,
        ),
        _preset(
            N_("Diffuse regions"),
            N_("Regions"),
            N_("For large positive areas with blurred boundaries and uneven inner intensity."),
            background_method="gaussian",
            background_radius=144,
            background_strength=0.85,
            denoise_strength=2.0,
            sensitivity=69,
            grow_ratio=0.34,
            min_area=100,
            close_radius=3,
            hole_area=800,
        ),
        _preset(
            N_("Fine filaments"),
            N_("Fibres"),
            N_("Reduces smoothing and morphological damage to keep very thin linear structures."),
            background_method="gaussian",
            background_radius=36,
            denoise_strength=0.25,
            sensitivity=68,
            grow_ratio=0.40,
            min_area=3,
            close_radius=1,
            hole_area=0,
            shape_profile="fiber",
        ),
        _preset(
            N_("Coarse fibre network"),
            N_("Fibres"),
            N_("For wider, interconnected fibres or networks."),
            background_method="gaussian",
            background_radius=104,
            denoise_strength=0.8,
            sensitivity=61,
            grow_ratio=0.41,
            min_area=40,
            close_radius=3,
            hole_area=0,
        ),
        _preset(
            N_("Fibre ridge enhancement"),
            N_("Special · fibres"),
            N_(
                "Special tool: use only when the targets are known to be fibres, tubes or "
                "networks. The Frangi ridge filter may also respond to target outlines, bright "
                "edges and curved boundaries.",
            ),
            background_method="vesselness",
            background_radius=8,
            background_strength=1.0,
            denoise_method="gaussian",
            denoise_strength=0.35,
            sensitivity=56,
            grow_ratio=0.45,
            min_area=6,
            close_radius=1,
            hole_area=0,
            shape_profile="fiber",
        ),
        _preset(
            N_("Ring structures"),
            N_("Shape"),
            N_("Keeps holes inside objects instead of filling them solid by default."),
            background_radius=34,
            denoise_strength=0.6,
            sensitivity=57,
            grow_ratio=0.52,
            min_area=18,
            close_radius=1,
            hole_area=0,
            shape_profile="ring",
        ),
        _preset(
            N_("Separate touching objects"),
            N_("Shape"),
            N_("Splits touching compact objects at distance peaks."),
            background_radius=26,
            denoise_strength=1.0,
            sensitivity=52,
            grow_ratio=0.62,
            min_area=18,
            close_radius=1,
            split_touching=True,
            watershed_peak_distance=7,
            watershed_peak_height=0.30,
        ),
        _preset(
            N_("Uneven illumination"),
            N_("Image quality"),
            N_(
                "Large-scale multi-background correction; for images with gradients or local "
                "differences in illumination.",
            ),
            background_method="multiscale",
            background_radius=144,
            background_strength=1.15,
            denoise_strength=0.9,
            sensitivity=56,
            min_area=8,
        ),
        _preset(
            N_("High background fluorescence"),
            N_("Image quality"),
            N_(
                "Uses the relative flat-field response to suppress overall brightness and "
                "autofluorescent background.",
            ),
            background_method="flat_field",
            background_radius=72,
            background_strength=1.0,
            denoise_strength=1.0,
            sensitivity=48,
            grow_ratio=0.58,
            min_area=10,
            min_response_confidence=1.00,
        ),
        _preset(
            N_("High granular noise"),
            N_("Image quality"),
            N_("Median filter with opening; reduces random bright grains."),
            denoise_method="median",
            denoise_strength=1.8,
            background_radius=24,
            sensitivity=42,
            grow_ratio=0.66,
            open_radius=1,
            close_radius=1,
            min_area=7,
            min_response_confidence=0.90,
        ),
        _preset(
            N_("Low contrast"),
            N_("Image quality"),
            N_(
                "Local mean and standard-deviation threshold; for images where positives and "
                "background differ little.",
            ),
            black_percentile=0.5,
            white_percentile=99.5,
            gamma=0.82,
            background_method="gaussian",
            background_radius=64,
            threshold_method="mean_std",
            sensitivity=68,
            local_window=61,
            local_k=0.35,
            min_area=6,
            min_response_confidence=1.00,
        ),
        _preset(
            N_("Overexposed"),
            N_("Image quality"),
            N_(
                "Tightens edges and the dynamic range; reduces over-expansion around saturated "
                "regions.",
            ),
            black_percentile=0.5,
            white_percentile=98.8,
            gamma=1.25,
            background_method="gaussian",
            background_radius=80,
            sensitivity=34,
            grow_ratio=0.74,
            min_area=15,
            close_radius=1,
        ),
        _preset(
            N_("Dark target recognition"),
            N_("Polarity"),
            N_(
                "Recognizes dark targets after inversion; for bright backgrounds or inverted "
                "images.",
            ),
            invert=True,
            background_radius=32,
            sensitivity=52,
            grow_ratio=0.58,
            min_area=8,
        ),
        _preset(
            N_("Long-wavelength channel · shared absolute threshold"),
            N_("Special · comparable across images"),
            N_(
                "Special tool: for channels whose main artefact is autofluorescence, when the "
                "whole batch shares one exposure and one export curve. It replaces \"so many "
                "times this image's noise\" with an absolute contrast shared by the batch, and "
                "excludes compartments whose broad background is too bright. Calibrate the "
                "\"shared noise scale\" and the \"background brightness ceiling\" on blank "
                "control images first; otherwise this preset has no effect.",
            ),
            intensity_space="robust_snr",
            background_method="gaussian",
            background_radius=48,
            background_strength=1.0,
            threshold_method="snr_hysteresis",
            snr_seed_sigma=6.0,
            sensitivity=50,
            grow_ratio=0.50,
            denoise_method="gaussian",
            denoise_strength=0.8,
            min_area=12,
            close_radius=1,
            hole_area=4,
            # Deliberately left at zero: the two controls below are the whole
            # point of this preset and they are meaningless until they have been
            # calibrated on *this* batch.  A shipped default would be a number
            # from somebody else's microscope.
            noise_reference_value=0.0,
            background_ceiling_mode="absolute",
            background_ceiling=0.0,
        ),
        _preset(
            N_("Long-wavelength channel · exclude high-background compartments"),
            N_("Special · comparable across images"),
            N_(
                "Special tool: for channels with strong autofluorescence, when exposure or "
                "exported brightness differs between images. It does not assume that grey levels "
                "are comparable across images; within each image it splits the broad background "
                "into two modes and excludes the brighter one (continuously bright "
                "high-background compartments such as autofluorescent bands, folds or thickened "
                "edges), judging only in the remaining compartments. Note: targets inside "
                "excluded compartments cannot be counted, and reports must say so.",
            ),
            intensity_space="robust_snr",
            background_method="gaussian",
            background_radius=48,
            background_strength=1.0,
            threshold_method="snr_hysteresis",
            snr_seed_sigma=6.0,
            sensitivity=50,
            grow_ratio=0.50,
            denoise_method="gaussian",
            denoise_strength=0.8,
            min_area=12,
            close_radius=1,
            hole_area=4,
            background_ceiling_mode="auto",
            # Chosen so the excluded components come out the size of the
            # structures they are meant to be.  On the calibration set the
            # excluded blobs measured 64 px median / 148 px p90 equivalent
            # diameter, matching the bright strands measured by hand there.
            # The object-background scale above is deliberately left much
            # larger; judging the compartment at that scale smeared the gate
            # into amorphous blobs that swallowed the surrounding field instead
            # of tracking the strands.  Broad continuous bands tolerate a larger
            # value, fine strands need a small one — re-measure whenever the
            # sample or the optics change.
            background_ceiling_scale=14,
        ),
    )
}

if len(BUILTIN_PRESETS) != 24:
    raise RuntimeError("Exactly 24 built-in presets are required")


def params_to_dict(params: UniversalParams) -> dict[str, object]:
    return asdict(enforce_shape_compatibility(params.validated()))


def params_from_dict(payload: Mapping[str, object]) -> UniversalParams:
    allowed = {field.name for field in fields(UniversalParams)}
    values = {key: value for key, value in payload.items() if key in allowed}
    if "recognition_engine" in values:
        values["recognition_engine"] = normalize_engine_id(values["recognition_engine"])
    return enforce_shape_compatibility(
        UniversalParams(**values).validated()  # type: ignore[arg-type]
    )


def save_recipe(
    path: str | Path,
    params: UniversalParams,
    *,
    name: str = T("Custom recipe"),
    description: str = "",
    quick_settings: Mapping[str, object] | None = None,
    quick_base_params: UniversalParams | None = None,
    analysis_preprocessing: Mapping[str, object] | None = None,
) -> Path:
    recipe_path = Path(path).expanduser().resolve()
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "algorithm_version": UNIVERSAL_ALGORITHM_VERSION,
        "name": str(name).strip() or T("Custom recipe"),
        "description": str(description).strip(),
        "params": params_to_dict(params),
    }
    if quick_settings is not None and quick_base_params is not None:
        payload.update(
            {
                "mode": "quick",
                "quick_schema_version": QUICK_SCHEMA_VERSION,
                "quick_settings": dict(quick_settings),
                "quick_base_params": params_to_dict(quick_base_params),
            }
        )
    else:
        payload["mode"] = "professional"
    if analysis_preprocessing is not None:
        public_keys = (
            "type",
            "version",
            "formula",
            "coefficient",
            "target_signal_mode_requested",
            "reference_file_name",
            "reference_signal_mode_requested",
            "preserve_reference_background",
            "reference_background_percentile",
            "size_policy",
        )
        sanitized = {
            key: analysis_preprocessing[key]
            for key in public_keys
            if key in analysis_preprocessing
        }
        if sanitized.get("type") == "crosstalk_subtraction":
            payload["analysis_preprocessing"] = sanitized
    recipe_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return recipe_path


def load_recipe_crosstalk_settings(
    path: str | Path,
) -> CrosstalkCorrection | None:
    """Read optional correction settings without binding a local reference path."""

    recipe_path = Path(path).expanduser().resolve()
    payload = json.loads(recipe_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        return None
    preprocessing = payload.get("analysis_preprocessing")
    if not isinstance(preprocessing, Mapping):
        return None
    if str(preprocessing.get("type", "")) != "crosstalk_subtraction":
        return None
    coefficient = preprocessing.get("coefficient", 0.30)
    if isinstance(coefficient, (bool, np.bool_)) or not isinstance(
        coefficient,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(T("The crosstalk coefficient in the recipe must be a number"))
    reference_signal_mode = preprocessing.get(
        "reference_signal_mode_requested",
        "auto",
    )
    if not isinstance(reference_signal_mode, str):
        raise ValueError(T("The crosstalk reference channel in the recipe must be a string"))
    preserve_background = preprocessing.get("preserve_reference_background", True)
    if not isinstance(preserve_background, (bool, np.bool_)):
        raise ValueError(T(
            "The crosstalk background-protection field in the recipe must be a boolean",
        ))
    background_percentile = preprocessing.get(
        "reference_background_percentile",
        10.0,
    )
    if isinstance(background_percentile, (bool, np.bool_)) or not isinstance(
        background_percentile,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(T("The crosstalk background percentile in the recipe must be a number"))
    return CrosstalkCorrection(
        coefficient=float(coefficient),
        reference_signal_mode=reference_signal_mode,
        preserve_reference_background=bool(preserve_background),
        background_percentile=float(background_percentile),
    ).validated()


_OLDEST_READABLE_ALGORITHM = (2, 7)
_UNSUPPORTED_RECIPE_MESSAGE = (
    T(
        "This recipe file was saved by an earlier version and is no longer read; set it up again "
        "in the current version and save a new recipe.",
    )
)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", text)
    return tuple(int(part) for part in parts[:3])


def load_recipe_bundle(
    path: str | Path,
) -> tuple[
    str,
    str,
    UniversalParams,
    dict[str, object] | None,
    UniversalParams | None,
]:
    """Read a recipe: (name, description, params, quick settings, quick base).

    Recipes written by versions whose parameters meant something different are
    refused, so a recipe never loads with silently changed results.
    """

    recipe_path = Path(path).expanduser().resolve()
    payload = json.loads(recipe_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(T("The top level of the recipe JSON must be an object"))
    params_payload = payload.get("params", payload)
    if not isinstance(params_payload, dict):
        raise ValueError(T("The recipe lacks a params object"))
    saved_by = _version_tuple(str(payload.get("algorithm_version", "")))
    if saved_by and saved_by < _OLDEST_READABLE_ALGORITHM:
        raise ValueError(_UNSUPPORTED_RECIPE_MESSAGE)
    if params_payload.get("scale_mode") not in (None, *SPATIAL_SCALE_MODES):
        raise ValueError(_UNSUPPORTED_RECIPE_MESSAGE)
    name = str(payload.get("name", recipe_path.stem)).strip() or recipe_path.stem
    description = str(payload.get("description", "")).strip()
    params = params_from_dict(params_payload)
    try:
        raw_radius = float(params_payload.get("background_radius", params.background_radius))
    except (TypeError, ValueError):
        raw_radius = params.background_radius
    if params.background_method in {"vesselness", "log"} and raw_radius > 128.0:
        description = (
            T(
                "{0} The structure scale was limited to 128 px so that a very large convolution "
                "range cannot exhaust memory.",
                description,
            )
        ).strip()

    quick_payload = payload.get("quick_settings")
    base_payload = payload.get("quick_base_params")
    if (
        payload.get("mode") == "quick"
        and isinstance(quick_payload, dict)
        and isinstance(base_payload, dict)
    ):
        if payload.get("quick_schema_version") != QUICK_SCHEMA_VERSION:
            raise ValueError(_UNSUPPORTED_RECIPE_MESSAGE)
        if base_payload.get("scale_mode") not in (None, *SPATIAL_SCALE_MODES):
            raise ValueError(_UNSUPPORTED_RECIPE_MESSAGE)
        try:
            base_params = params_from_dict(base_payload)
        except (TypeError, ValueError):
            base_params = None
        if base_params is not None:
            return name, description, params, dict(quick_payload), base_params
    return name, description, params, None, None


def load_recipe(path: str | Path) -> tuple[str, str, UniversalParams]:
    name, description, params, _quick, _base = load_recipe_bundle(path)
    return name, description, params


def _path_is_link_or_reparse_point(path: Path) -> bool:
    """Identify path-redirecting links/junctions without rejecting cloud files."""

    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = int(getattr(metadata, "st_file_attributes", 0))
    if not (reparse_flag and file_attributes & reparse_flag):
        return False
    # Windows uses reparse points both for path redirection (symlinks,
    # junctions/mount points) and for non-redirecting services such as OneDrive
    # placeholders, WOF compression and data deduplication.  Only tags carrying
    # the documented Name Surrogate bit redirect path lookup.  st_reparse_tag
    # exists on supported Windows Python versions; retain the conservative
    # fallback solely for runtimes that expose the attribute flag without a tag.
    reparse_tag = int(getattr(metadata, "st_reparse_tag", 0))
    if reparse_tag:
        return bool(reparse_tag & 0x20000000)
    return True


def supported_image_files(
    folder: str | Path,
    *,
    recursive: bool = True,
) -> list[Path]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(T("The image folder does not exist: {0}", root.name or T("selected location")))

    def is_export_directory(directory: Path, filenames: set[str]) -> bool:
        if _EXPORT_STAGING_DIRECTORY_RE.fullmatch(directory.name):
            return True
        if not _EXPORT_DIRECTORY_RE.fullmatch(directory.name):
            return False
        if "INCOMPLETE.txt" in filenames:
            return True
        if "project.json" in filenames and "objects.csv" in filenames:
            return True
        # A selective export may contain neither project.json nor objects.csv.
        # Require both the directory naming contract and one exact artifact name
        # so a legitimate source folder containing an image called merge.png is
        # not rejected.  os.walk can then prune confirmed outputs immediately.
        return bool(filenames.intersection(_EXPORT_MARKER_FILENAMES))

    output: list[Path] = []
    scan_error: OSError | None = None

    def remember_scan_error(error: OSError) -> None:
        nonlocal scan_error
        scan_error = error

    for current_value, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=remember_scan_error,
        followlinks=False,
    ):
        if scan_error is not None:
            raise scan_error
        current = Path(current_value)
        filenames = set(file_names)
        if is_export_directory(current, filenames):
            directory_names.clear()
            continue
        if not recursive:
            directory_names.clear()
        else:
            # os.walk(followlinks=False) covers ordinary directory symlinks,
            # but explicitly prune all links/reparse points as well so Windows
            # junction behavior cannot vary across supported Python releases.
            safe_directory_names: list[str] = []
            for directory_name in directory_names:
                directory = current / directory_name
                try:
                    if not _path_is_link_or_reparse_point(directory):
                        safe_directory_names.append(directory_name)
                except FileNotFoundError:
                    continue
            directory_names[:] = safe_directory_names
        for filename in file_names:
            path = current / filename
            if path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            # New flat bundles keep the import tree but place artifacts beside
            # the source.  Exclude only the exact reserved output namespace;
            # ordinary files such as merge.png or R01_signal.tif remain valid.
            if _PREFIXED_EXPORT_IMAGE_RE.fullmatch(filename):
                continue
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            # Never follow a file link out of the selected import root.  Hard
            # links remain regular files and are safe because the directory
            # entry itself is inside the root.
            if stat.S_ISREG(metadata.st_mode) and not _path_is_link_or_reparse_point(path):
                output.append(path)
    if scan_error is not None:
        raise scan_error
    return sorted(output, key=lambda item: str(item).casefold())


def _tiff_main_ifd_count(payload: np.ndarray, *, limit: int = 2) -> int | None:
    """Count top-level TIFF image-file directories without decoding pixels.

    OpenCV can decode the first page of a damaged multi-page TIFF even when its
    multi-page probe fails.  The independent IFD-chain check below lets the
    single-frame fallback run only when the container itself proves that it has
    exactly one top-level page.  Both classic TIFF and BigTIFF are supported.
    ``None`` means that the container metadata is malformed or unsupported.
    """

    raw = memoryview(np.ascontiguousarray(payload)).cast("B")
    length = len(raw)
    if length < 8:
        return None
    marker = bytes(raw[:2])
    if marker == b"II":
        byteorder = "little"
    elif marker == b"MM":
        byteorder = "big"
    else:
        return None

    def read_uint(offset: int, size: int) -> int | None:
        if offset < 0 or size <= 0 or offset > length - size:
            return None
        return int.from_bytes(raw[offset : offset + size], byteorder=byteorder)

    magic = read_uint(2, 2)
    if magic == 42:
        count_size, entry_size, offset_size = 2, 12, 4
        first_ifd = read_uint(4, 4)
    elif magic == 43:
        if read_uint(4, 2) != 8 or read_uint(6, 2) != 0 or length < 16:
            return None
        count_size, entry_size, offset_size = 8, 20, 8
        first_ifd = read_uint(8, 8)
    else:
        return None
    if first_ifd is None:
        return None

    count = 0
    current = first_ifd
    seen: set[int] = set()
    while current:
        if current in seen:
            return None
        seen.add(current)
        entry_count = read_uint(current, count_size)
        if entry_count is None:
            return None
        next_offset_position = current + count_size + entry_count * entry_size
        next_ifd = read_uint(next_offset_position, offset_size)
        if next_ifd is None:
            return None
        count += 1
        if count >= max(1, int(limit)):
            return count
        current = next_ifd
    return count


def _read_bytes(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    try:
        stat_before = path.stat()
    except OSError as exc:
        raise OSError(T("Cannot read the image information: {0}", path.name)) from exc
    if int(stat_before.st_size) > MAX_ENCODED_IMAGE_BYTES:
        raise ValueError(
            T(
                "The image file exceeds the 1 GiB safety limit: {0}. Crop it or export it in "
                "tiles first.",
                path.name,
            )
        )
    try:
        payload = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise OSError(T("Cannot read the image: {0}", path.name)) from exc
    if path.suffix.casefold() in {".tif", ".tiff"}:
        metadata_page_count = _tiff_main_ifd_count(payload)
        if metadata_page_count is not None and metadata_page_count > 1:
            raise ValueError(
                T(
                    "This version handles 2-D single-page images only; export one plane of a "
                    "multi-page TIFF / Z-stack first: {0}",
                    path.name,
                )
            )
        decoded, pages = cv2.imdecodemulti(
            payload,
            cv2.IMREAD_UNCHANGED,
            range=(0, 2),
        )
        if decoded and len(pages) > 1:
            raise ValueError(
                T(
                    "This version handles 2-D single-page images only; export one plane of a "
                    "multi-page TIFF / Z-stack first: {0}",
                    path.name,
                )
            )
        image = pages[0] if decoded and pages else None
        if image is None:
            # Some valid single-page TIFF encodings/compressions decode with the
            # single-frame reader even when the multi-page probe returns nothing.
            # Do not infer "single page" from a failed decoder probe: a damaged
            # later page can make imdecodemulti fail while imdecode still returns
            # page one.  The independent TIFF IFD chain must prove one page.
            if metadata_page_count != 1:
                raise ValueError(T(
                    "The TIFF page structure is damaged, or the file cannot be confirmed to be a "
                    "single-page image: {0}",
                    path.name,
                ))
            image = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(T("The image is damaged or its format is not supported: {0}", path.name))
    if image.ndim not in (2, 3):
        raise ValueError(T(
            "A 2-D greyscale or colour image is needed; the actual shape is {0}: {1}",
            image.shape,
            path.name,
        ))
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(T(
            "Interleaved images with {0} channels are not supported: {1}",
            image.shape[2],
            path.name,
        ))
    pixel_count = int(image.shape[0]) * int(image.shape[1])
    if pixel_count > MAX_DECODED_IMAGE_PIXELS:
        raise ValueError(
            T(
                "The image has {0:,} pixels, more than the safety limit of 80,000,000 pixels.",
                pixel_count,
            )
        )
    stat_after = path.stat()
    if (
        int(stat_after.st_size) != int(stat_before.st_size)
        or int(stat_after.st_mtime_ns) != int(stat_before.st_mtime_ns)
    ):
        raise RuntimeError(T(
            "The image changed while it was being read; please try again: {0}",
            path.name,
        ))
    metadata: dict[str, object] = {
        "size": int(stat_after.st_size),
        "mtime_ns": int(stat_after.st_mtime_ns),
        "sha256": hashlib.sha256(memoryview(payload)).hexdigest(),
        "dtype": str(image.dtype),
        "shape": [int(v) for v in image.shape],
        "color_order": "BGR" if image.ndim == 3 else "GRAY",
    }
    return image, metadata


def load_image(path: str | Path) -> LoadedImage:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(T("The image does not exist: {0}", image_path.name or T("selected file")))
    if image_path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(T("Unsupported image format: {0}", image_path.suffix))
    image, metadata = _read_bytes(image_path)
    return LoadedImage(path=image_path, image=image, metadata=metadata)


def _sample_pixels(image: np.ndarray, max_pixels: int = 500_000) -> np.ndarray:
    height, width = image.shape[:2]
    count = height * width
    if count <= max_pixels:
        return image.reshape(-1, *image.shape[2:]) if image.ndim > 2 else image.ravel()
    stride = max(1, int(math.ceil(math.sqrt(count / max_pixels))))
    sampled = image[::stride, ::stride]
    return sampled.reshape(-1, *sampled.shape[2:]) if sampled.ndim > 2 else sampled.ravel()


def extract_signal(
    image: np.ndarray,
    params: UniversalParams,
) -> tuple[np.ndarray, str]:
    params = params.validated(tuple(int(v) for v in image.shape[:2]))
    if image.ndim == 2:
        return image.astype(np.float32, copy=False), "grayscale"
    if image.shape[2] == 1:
        return image[..., 0].astype(np.float32, copy=True), "grayscale"

    b = image[..., 0]
    g = image[..., 1]
    r = image[..., 2]
    mode = params.signal_mode
    if mode == "auto":
        sample = _sample_pixels(image[..., :3])
        sample = sample.astype(np.float64, copy=False)
        # One partition serves all three per-channel order statistics; the
        # selected values are exactly those of three separate calls.
        channel_high, channel_low, channel_centres = np.percentile(
            sample,
            (99.5, 5.0, 50.0),
            axis=0,
        )
        channel_ranges = channel_high - channel_low
        # Compare channel centres against a translation-free pooled dynamic
        # range.  The former mean/max-mean ratio changed when the same offset
        # was added to all RGB channels, so merely brightening a picture could
        # switch auto mode between average and max_rgb.
        pooled_low, pooled_high = np.percentile(sample.reshape(-1), (5.0, 99.5))
        pooled_span = float(pooled_high - pooled_low)
        centre_spread = float(channel_centres.max() - channel_centres.min())
        nearly_gray = (
            centre_spread <= 1e-12
            if pooled_span <= 1e-12
            else centre_spread <= pooled_span * 0.015
        )
        if nearly_gray:
            mode = "average"
        else:
            ordered = np.argsort(channel_ranges)
            best = int(ordered[-1])
            second = float(channel_ranges[ordered[-2]])
            if float(channel_ranges[best]) >= max(second * 1.45, 1e-12):
                mode = ("blue", "green", "red")[best]
            else:
                mode = "max_rgb"

    if mode == "blue":
        signal = b.astype(np.float32, copy=True)
    elif mode == "green":
        signal = g.astype(np.float32, copy=True)
    elif mode == "red":
        signal = r.astype(np.float32, copy=True)
    elif mode in {"max_rgb", "hsv_value"}:
        signal = b.astype(np.float32, copy=True)
        np.maximum(signal, g, out=signal)
        np.maximum(signal, r, out=signal)
    elif mode == "average":
        signal = b.astype(np.float32, copy=True)
        np.add(signal, g, out=signal, casting="unsafe")
        np.add(signal, r, out=signal, casting="unsafe")
        signal *= np.float32(1.0 / 3.0)
    elif mode == "luma":
        signal = b.astype(np.float32, copy=True) * np.float32(0.0722)
        signal += g.astype(np.float32, copy=False) * np.float32(0.7152)
        signal += r.astype(np.float32, copy=False) * np.float32(0.2126)
    elif mode == "hsv_saturation":
        maximum = b.astype(np.float32, copy=True)
        np.maximum(maximum, g, out=maximum)
        np.maximum(maximum, r, out=maximum)
        minimum = b.astype(np.float32, copy=True)
        np.minimum(minimum, g, out=minimum)
        np.minimum(minimum, r, out=minimum)
        dynamic_scale = float(np.percentile(_sample_pixels(maximum), 99.9))
        dynamic_scale = max(dynamic_scale, 1.0)
        signal = (maximum - minimum) / np.maximum(maximum, np.float32(1e-6))
        signal *= np.float32(dynamic_scale)
    elif mode == "custom_rgb":
        weights = np.asarray(
            [params.blue_weight, params.green_weight, params.red_weight],
            dtype=np.float32,
        )
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-8:
            weights[:] = 1.0 / 3.0
        else:
            weights /= weight_sum
        signal = b.astype(np.float32, copy=True) * weights[0]
        signal += g.astype(np.float32, copy=False) * weights[1]
        signal += r.astype(np.float32, copy=False) * weights[2]
    else:
        raise ValueError(T("Unknown signal extraction mode: {0}", mode))
    return signal, mode


def _finite_sample(image: np.ndarray, max_pixels: int = 1_000_000) -> np.ndarray:
    sample = _sample_pixels(image, max_pixels=max_pixels).astype(np.float64, copy=False)
    finite = sample[np.isfinite(sample)]
    return finite if finite.size else np.asarray([0.0], dtype=np.float64)


def _signal_native_scale(source_dtype: np.dtype[Any], signal: np.ndarray) -> float:
    """Return a stable full-scale value for cross-bit-depth subtraction."""

    dtype = np.dtype(source_dtype)
    if np.issubdtype(dtype, np.bool_):
        return 1.0
    if np.issubdtype(dtype, np.integer):
        return max(1.0, float(np.iinfo(dtype).max))
    finite = _finite_sample(signal, max_pixels=1_000_000)
    observed = float(np.max(np.abs(finite), initial=0.0))
    # Most floating-point microscopy TIFFs use 0..1.  Preserve that shared
    # scale, while still allowing larger non-negative floating inputs.
    return 1.0 if observed <= 1.0 + 1e-6 else observed


def apply_crosstalk_correction(
    target: LoadedImage,
    reference: LoadedImage,
    target_params: UniversalParams,
    correction: CrosstalkCorrection,
) -> LoadedImage:
    """Build the scalar image analysed after target − k × reference.

    Integer target/reference images are first put onto their native 0..1
    ranges, so an 8-bit reference can be paired with a 16-bit target without a
    hidden 257-fold mismatch.  No resizing or registration is attempted.
    """

    settings = correction.validated()
    target_path = target.path.expanduser().resolve()
    reference_path = reference.path.expanduser().resolve()
    if target_path == reference_path:
        raise ValueError(T(
            "The crosstalk reference image must not be the same file as the current target image",
        ))
    if (
        str(target.metadata.get("sha256", ""))
        and str(target.metadata.get("sha256", ""))
        == str(reference.metadata.get("sha256", ""))
    ):
        raise ValueError(T(
            "The target image and the crosstalk reference image have identical content; "
            "correction refused",
        ))
    if target.shape != reference.shape:
        raise ValueError(
            T(
                "The crosstalk reference image must have exactly the size of the target image and "
                "be registered to it: target {0}×{1}, reference {2}×{3}",
                target.shape[1],
                target.shape[0],
                reference.shape[1],
                reference.shape[0],
            )
        )
    if (
        reference.image.ndim == 3
        and reference.image.shape[2] >= 3
        and settings.reference_signal_mode == "auto"
    ):
        sample = _sample_pixels(reference.image[..., :3], max_pixels=500_000)
        if sample.ndim == 2 and sample.shape[1] >= 3:
            channel_difference = float(
                np.max(
                    np.abs(
                        sample[:, :3].astype(np.float64, copy=False)
                        - sample[:, :1].astype(np.float64, copy=False)
                    ),
                    initial=0.0,
                )
            )
            if channel_difference > 0.0:
                raise ValueError(
                    T(
                        "The crosstalk reference image is a colour image; choose red, green, blue "
                        "or RGB maximum explicitly instead of letting automatic mode guess the "
                        "reference channel",
                    )
                )
    pixel_count = int(target.shape[0]) * int(target.shape[1])
    if pixel_count > MAX_CROSSTALK_CORRECTION_PIXELS:
        raise ValueError(
            T(
                "The images for crosstalk correction have {0:,} pixels, more than the two-image "
                "safety limit of {1:,} pixels; crop, tile or downscale them first.",
                pixel_count,
                MAX_CROSSTALK_CORRECTION_PIXELS,
            )
        )

    validated_target_params = target_params.validated(target.shape)
    target_extracted, target_mode_used = extract_signal(
        target.image,
        validated_target_params,
    )
    target_signal = np.array(target_extracted, dtype=np.float32, copy=True, order="C")

    reference_params = replace(
        validated_target_params,
        signal_mode=settings.reference_signal_mode,
        invert=False,
    )
    reference_extracted, reference_mode_used = extract_signal(
        reference.image,
        reference_params,
    )
    reference_signal = np.array(
        reference_extracted,
        dtype=np.float32,
        copy=True,
        order="C",
    )

    target_scale = _signal_native_scale(target.image.dtype, target_signal)
    reference_scale = _signal_native_scale(reference.image.dtype, reference_signal)
    reference_signal /= np.float32(reference_scale)
    reference_background = 0.0
    if settings.preserve_reference_background:
        reference_background = float(
            np.percentile(
                _finite_sample(reference_signal, max_pixels=1_000_000),
                settings.background_percentile,
            )
        )
        reference_signal -= np.float32(reference_background)
        np.maximum(reference_signal, np.float32(0.0), out=reference_signal)
    reference_signal *= np.float32(settings.coefficient * target_scale)

    np.subtract(target_signal, reference_signal, out=target_signal)
    clipped_fraction = float(np.count_nonzero(target_signal < 0.0) / max(1, pixel_count))
    np.clip(target_signal, np.float32(0.0), np.float32(target_scale), out=target_signal)

    public_metadata: dict[str, object] = {
        "type": "crosstalk_subtraction",
        "version": CROSSTALK_CORRECTION_VERSION,
        "formula": "target - coefficient * reference",
        "coefficient": settings.coefficient,
        "target_signal_mode_requested": validated_target_params.signal_mode,
        "target_signal_mode_used": target_mode_used,
        "reference_file_name": reference.path.name,
        "reference_signal_mode_requested": settings.reference_signal_mode,
        "reference_signal_mode_used": reference_mode_used,
        "preserve_reference_background": settings.preserve_reference_background,
        "reference_background_percentile": (
            settings.background_percentile
            if settings.preserve_reference_background
            else None
        ),
        "reference_background_normalized": (
            reference_background if settings.preserve_reference_background else None
        ),
        "zero_clipped_fraction": clipped_fraction,
        "size_policy": "strict_same_shape_no_resize",
    }
    signature_payload = {
        "target_sha256": str(target.metadata.get("sha256", "")),
        "reference_sha256": str(reference.metadata.get("sha256", "")),
        **public_metadata,
    }
    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata = dict(target.metadata)
    metadata["analysis_preprocessing"] = public_metadata
    metadata["analysis_input_signature"] = signature
    # The absolute reference path is private runtime state used only to reject
    # an export if the paired image changes after analysis.  It is deliberately
    # excluded from project.json and every CSV.
    metadata["_crosstalk_reference"] = {
        "path": str(reference.path),
        "size": int(reference.metadata["size"]),
        "mtime_ns": int(reference.metadata["mtime_ns"]),
        "sha256": str(reference.metadata["sha256"]),
    }
    return LoadedImage(
        path=target.path,
        image=target_signal,
        metadata=metadata,
    )


_RELATIVE_RANK_ANCHOR_COUNT = 1025


_MIDRANK_BLOCK_ROWS = 256
_MIDRANK_MAX_LUT_LEVELS = 65536


def _integer_lut_levels(signal: np.ndarray, sample: np.ndarray) -> int | None:
    """Return ``max + 1`` when every pixel is an integer in ``[0, 65535]``.

    Camera data (8/16-bit TIFF/PNG) reaches the rank transform as float32 with
    at most 65536 distinct values, so the transform can be evaluated once per
    grey level instead of once per pixel.  The check scans in row blocks and
    stops at the first non-integer block, so genuinely floating-point images
    pay for only one block.
    """

    if signal.dtype.kind in "iub":
        low = int(signal.min(initial=0))
        high = int(signal.max(initial=0))
        if low < 0 or high >= _MIDRANK_MAX_LUT_LEVELS:
            return None
        return high + 1
    if signal.dtype.kind != "f" or sample.size == 0:
        return None
    if float(sample.min()) < 0.0 or float(sample.max()) >= _MIDRANK_MAX_LUT_LEVELS:
        return None
    flat = signal.reshape(-1) if signal.ndim != 2 else signal
    high = 0.0
    for y0 in range(0, flat.shape[0], _MIDRANK_BLOCK_ROWS):
        block = flat[y0 : y0 + _MIDRANK_BLOCK_ROWS]
        truncated = np.trunc(block)
        if not np.array_equal(block, truncated):
            return None
        block_high = float(truncated.max(initial=0.0))
        if float(truncated.min(initial=0.0)) < 0.0 or block_high >= _MIDRANK_MAX_LUT_LEVELS:
            return None
        high = max(high, block_high)
    return int(high) + 1


def _empirical_midrank(
    signal: np.ndarray,
    *,
    invert: bool = False,
    tail_power: float = 1.0,
    finite: np.ndarray | None = None,
    lut: tuple[np.ndarray, np.ndarray] | None = None,
    return_lut: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """Return a deterministic empirical-CDF coordinate for ``signal``.

    Quantile plateaus receive their interval mid-rank.  This matters for 8-bit
    screenshots: assigning every tied value the upper rank systematically
    brightens quantized input and makes the final mask less stable.  Any
    strictly increasing point transform has the same ranks, subject only to
    destructive clipping/quantization ties.

    The transform is a pure per-pixel function of the anchor table, so it is
    evaluated either through a per-grey-level lookup table (integer-valued
    input, or an input that is itself ``table[index]`` as described by
    ``lut``) or in bounded row blocks (floating input).  Both produce exactly
    the float32 values of one whole-frame ``np.interp`` call while avoiding
    its two full-frame float64 temporaries.  With ``return_lut`` the function
    also returns ``(table, index)`` with ``result == table[index]`` when such a
    representation exists, so a second rank pass can be chained on the table.
    """

    sample = _finite_sample(signal) if finite is None else finite
    probabilities = np.linspace(
        0.0,
        100.0,
        _RELATIVE_RANK_ANCHOR_COUNT,
        dtype=np.float64,
    )
    anchors = np.percentile(sample, probabilities)
    values, first = np.unique(anchors, return_index=True)
    power = float(np.clip(tail_power, 1.0, 12.0))
    apply_power = abs(power - 1.0) > 1e-6
    if values.size <= 1:
        ranked = np.full(signal.shape, 0.5, dtype=np.float32)
        if invert:
            ranked = np.float32(1.0) - ranked
        if apply_power:
            ranked = np.power(ranked, np.float32(power), dtype=np.float32)
        ranked = ranked.astype(np.float32, copy=False)
        return (ranked, None) if return_lut else ranked

    last = np.r_[first[1:] - 1, _RELATIVE_RANK_ANCHOR_COUNT - 1]
    midranks = (
        (first.astype(np.float64) + last.astype(np.float64))
        * 0.5
        / float(_RELATIVE_RANK_ANCHOR_COUNT - 1)
    )

    source_table: np.ndarray | None = None
    index: np.ndarray | None = None
    if lut is not None:
        source_table, index = lut
    else:
        levels = _integer_lut_levels(signal, sample)
        if levels is not None:
            source_table = np.arange(levels, dtype=np.float64)
            index = signal.astype(np.uint8 if levels <= 256 else np.uint16, copy=False)
    if source_table is not None and index is not None:
        table = np.interp(
            source_table.astype(np.float64, copy=False),
            values,
            midranks,
            left=0.0,
            right=1.0,
        ).astype(np.float32)
        # Polarity and tail power are element-wise, so applying them to the
        # table gives the same float32 result as applying them to every pixel.
        if invert:
            table = np.float32(1.0) - table
        if apply_power:
            table = np.power(table, np.float32(power), dtype=np.float32)
        if index.dtype == np.uint8:
            # cv2.LUT is a plain 256-entry gather, ~3x faster than fancy
            # indexing on a 24 MP frame, and yields the identical float32 plane.
            table256 = np.zeros(256, dtype=np.float32)
            table256[: table.size] = table
            table = table256
            ranked = cv2.LUT(index, table)
        else:
            ranked = table[index]
        return (ranked, (table, index)) if return_lut else ranked

    ranked = np.empty(signal.shape, dtype=np.float32)
    source = signal if signal.ndim == 2 else signal.reshape(signal.shape[0], -1)
    target = ranked if ranked.ndim == 2 else ranked.reshape(ranked.shape[0], -1)
    for y0 in range(0, source.shape[0], _MIDRANK_BLOCK_ROWS):
        block = source[y0 : y0 + _MIDRANK_BLOCK_ROWS]
        target[y0 : y0 + _MIDRANK_BLOCK_ROWS] = np.interp(
            block.astype(np.float64, copy=False),
            values,
            midranks,
            left=0.0,
            right=1.0,
        )
    if invert:
        np.subtract(np.float32(1.0), ranked, out=ranked)
    if apply_power:
        np.power(ranked, np.float32(power), out=ranked, dtype=np.float32)
    return (ranked, None) if return_lut else ranked


def normalize_signal(
    signal: np.ndarray,
    params: UniversalParams,
) -> tuple[np.ndarray, float, float]:
    finite = _finite_sample(signal)
    if params.intensity_space == "relative_rank":
        low, high = (
            float(value)
            for value in np.percentile(
                finite,
                (params.black_percentile, params.white_percentile),
            )
        )
        normalized = _empirical_midrank(
            signal,
            invert=params.invert,
            tail_power=params.rank_tail_power,
            finite=finite,
        )
        return normalized, low, high

    # Keep this legacy percentile-linear branch byte-for-byte equivalent to
    # 2.5.x.  In particular, do not consolidate its fallbacks with the rank
    # branch or change float32 operation order.
    low, high = (
        float(value)
        for value in np.percentile(
            finite,
            (params.black_percentile, params.white_percentile),
        )
    )
    if not math.isfinite(low):
        low = 0.0
    if not math.isfinite(high) or high <= low:
        low = float(finite.min(initial=0.0))
        high = float(finite.max(initial=1.0))
    if high <= low:
        high = low + 1.0
    normalized = (signal.astype(np.float32, copy=False) - np.float32(low)) / np.float32(
        high - low
    )
    np.clip(normalized, 0.0, 1.0, out=normalized)
    if params.invert:
        normalized = np.float32(1.0) - normalized
    if abs(params.gamma - 1.0) > 1e-6:
        normalized = np.power(normalized, np.float32(params.gamma), dtype=np.float32)
    return normalized.astype(np.float32, copy=False), low, high


def _input_intensity_diagnostics(
    signal: np.ndarray,
    source_dtype: np.dtype[Any] | type[Any] | None,
) -> tuple[float, float, float, int, tuple[str, ...]]:
    """Measure saturation and tied-value plateaus without changing the mask."""

    sample = _finite_sample(signal, max_pixels=500_000)
    sample_count = max(1, int(sample.size))
    values, counts = np.unique(sample, return_counts=True)
    unique_levels = int(values.size)
    largest_tie_fraction = float(counts.max(initial=0) / sample_count)
    observed_max = float(values[-1]) if values.size else 0.0
    high_plateau_fraction = float(np.count_nonzero(sample == observed_max) / sample_count)

    saturation_fraction = 0.0
    if source_dtype is not None:
        dtype = np.dtype(source_dtype)
        if np.issubdtype(dtype, np.integer):
            nominal_max = float(np.iinfo(dtype).max)
            tolerance = max(1e-6, nominal_max * 1e-7)
            saturation_fraction = float(
                np.count_nonzero(sample >= nominal_max - tolerance) / sample_count
            )

    warnings: list[str] = []
    if saturation_fraction >= 0.10:
        warnings.append(
            T(
                "Severe saturation: about {0:.1%} of the sampled pixels are at the upper limit of "
                "the data type, so the order of pixel intensities is lost; the result is for "
                "review only.",
                saturation_fraction,
            )
        )
    elif saturation_fraction >= 0.01:
        warnings.append(
            T(
                "Saturated highlights detected: about {0:.1%} of the sampled pixels are at the "
                "upper limit of the data type.",
                saturation_fraction,
            )
        )
    elif high_plateau_fraction >= 0.10:
        warnings.append(
            T(
                "Severe highlight plateau: about {0:.1%} of the sampled pixels share the same "
                "highest value; relative rank cannot recover the order inside the plateau.",
                high_plateau_fraction,
            )
        )

    if (
        largest_tie_fraction >= 0.50
        and largest_tie_fraction > high_plateau_fraction + 1e-12
    ):
        warnings.append(
            T(
                "Severe tie plateau: the largest plateau of equal grey level covers about "
                "{0:.1%}; check for background clipping or quantization.",
                largest_tie_fraction,
            )
        )
    if sample_count >= 1024 and unique_levels <= 256:
        warnings.append(
            T(
                "Coarse grey-level quantization: only {0} grey levels in the sample; faint "
                "boundaries may be less consistent than in the original high-bit-depth image.",
                unique_levels,
            )
        )
    return (
        saturation_fraction,
        high_plateau_fraction,
        largest_tie_fraction,
        unique_levels,
        tuple(warnings),
    )


def _ellipse_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = _KERNEL_CACHE.get(radius)
    if kernel is None:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        _KERNEL_CACHE[radius] = kernel
    return kernel


# Ellipse openings wider than this radius run through the exact rectangular
# band decomposition below instead of OpenCV's point-by-point ellipse filter.
_ELLIPSE_DECOMPOSE_MIN_RADIUS = 20
_ELLIPSE_BAND_CACHE: dict[int, tuple[tuple[int, int], ...] | None] = {}


def _ellipse_bands(radius: int) -> tuple[tuple[int, int], ...] | None:
    """Decompose the digital ellipse into a union of centred rectangles.

    Every row ``dy`` of OpenCV's ellipse element is a centred run of
    half-width ``w(dy)``, non-increasing in ``|dy|``.  For each distinct
    half-width ``w`` the rows with ``w(dy) >= w`` form a centred band
    ``|dy| <= d``, so the element is exactly the union of the rectangles
    ``[-w, w] x [-d, d]``.  Erosion (dilation) by a union is the pixel-wise
    minimum (maximum) of the erosions (dilations) by the parts, and each part
    is a separable rectangle that OpenCV filters in O(w + d) per pixel
    instead of O(w * d).  ``None`` means the element did not have the expected
    structure and the caller must use the plain kernel.
    """

    radius = max(1, int(radius))
    if radius in _ELLIPSE_BAND_CACHE:
        return _ELLIPSE_BAND_CACHE[radius]
    kernel = _ellipse_kernel(radius)
    half_widths: dict[int, int] = {}
    valid = True
    for row in range(kernel.shape[0]):
        columns = np.flatnonzero(kernel[row])
        if columns.size == 0:
            continue
        left = radius - int(columns[0])
        right = int(columns[-1]) - radius
        if left != right or int(columns[-1] - columns[0]) + 1 != columns.size:
            valid = False
            break
        half_widths[row - radius] = left
    if valid:
        for offset in range(1, radius + 1):
            if half_widths.get(offset) != half_widths.get(-offset):
                valid = False
                break
        widths = [half_widths[offset] for offset in range(0, radius + 1) if offset in half_widths]
        if any(widths[index] < widths[index + 1] for index in range(len(widths) - 1)):
            valid = False
    bands: tuple[tuple[int, int], ...] | None = None
    if valid and half_widths:
        result: list[tuple[int, int]] = []
        for width in sorted(set(half_widths.values()), reverse=True):
            depth = max(
                offset for offset, value in half_widths.items() if value >= width
            )
            result.append((width, depth))
        bands = tuple(result)
    _ELLIPSE_BAND_CACHE[radius] = bands
    return bands


def _ellipse_open_reflect(work: np.ndarray, radius: int) -> np.ndarray:
    """``cv2.morphologyEx(work, MORPH_OPEN, ellipse(radius), BORDER_REFLECT)``.

    Exact: min/max filters involve no rounding, and the band decomposition
    reproduces the same neighbourhood (including the reflected border) as the
    full ellipse.  It is roughly twice as fast from radius 24 upwards and the
    gap widens with the radius; small elements keep OpenCV's direct filter.
    """

    radius = max(1, int(radius))
    bands = (
        _ellipse_bands(radius)
        if radius >= _ELLIPSE_DECOMPOSE_MIN_RADIUS
        else None
    )
    if bands is None:
        return cv2.morphologyEx(
            work,
            cv2.MORPH_OPEN,
            _ellipse_kernel(radius),
            borderType=cv2.BORDER_REFLECT,
        )
    eroded: np.ndarray | None = None
    for width, depth in bands:
        rows = cv2.erode(
            work,
            np.ones((1, 2 * width + 1), dtype=np.uint8),
            borderType=cv2.BORDER_REFLECT,
        )
        band = cv2.erode(
            rows,
            np.ones((2 * depth + 1, 1), dtype=np.uint8),
            borderType=cv2.BORDER_REFLECT,
        )
        eroded = band if eroded is None else np.minimum(eroded, band, out=eroded)
    assert eroded is not None
    dilated: np.ndarray | None = None
    for width, depth in bands:
        rows = cv2.dilate(
            eroded,
            np.ones((1, 2 * width + 1), dtype=np.uint8),
            borderType=cv2.BORDER_REFLECT,
        )
        band = cv2.dilate(
            rows,
            np.ones((2 * depth + 1, 1), dtype=np.uint8),
            borderType=cv2.BORDER_REFLECT,
        )
        dilated = band if dilated is None else np.maximum(dilated, band, out=dilated)
    assert dilated is not None
    return dilated


def _denoise(
    image: np.ndarray,
    method: str,
    strength: float,
    *,
    encode_offset: float = 0.0,
    encode_scale: float = 1.0,
) -> np.ndarray:
    """Denoise a float image.

    ``encode_offset``/``encode_scale`` describe the intensity window that maps to
    the 8-bit encode used by the wide-kernel median and non-local-means filters,
    and to the bilateral colour sigma.  The defaults ``(0.0, 1.0)`` reproduce the
    historical unit-interval behaviour exactly for callers whose data is already
    normalized to ``[0, 1]``.  The robust-SNR path works on raw-scale (or
    inverted) intensities and passes the data's own ``[min, max]`` window so those
    filters no longer collapse every pixel to the encode ceiling.
    """

    if method == "none" or strength <= 0.0:
        return image.astype(np.float32, copy=False)
    work = image.astype(np.float32, copy=False)
    scale = float(encode_scale) if math.isfinite(encode_scale) and encode_scale > 0.0 else 1.0
    offset = float(encode_offset) if math.isfinite(encode_offset) else 0.0

    def _encode8(values: np.ndarray) -> np.ndarray:
        normed = (values - np.float32(offset)) / np.float32(scale)
        return np.rint(np.clip(normed, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _decode8(values: np.ndarray) -> np.ndarray:
        return values.astype(np.float32) / np.float32(255.0) * np.float32(scale) + np.float32(offset)

    if method == "gaussian":
        return cv2.GaussianBlur(
            work,
            (0, 0),
            sigmaX=max(strength, 0.05),
            sigmaY=max(strength, 0.05),
            borderType=cv2.BORDER_REFLECT,
        )
    if method == "median":
        radius = max(1, int(round(strength)))
        kernel = min(63, radius * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        # OpenCV only accepts float32/uint16 inputs for 3x3 and 5x5
        # median filters.  The public parameter range intentionally permits
        # much wider kernels, so use an 8-bit working copy above 5x5 and map
        # the result back to the float domain via the encode window.
        if kernel <= 5:
            return cv2.medianBlur(work, kernel)
        return _decode8(cv2.medianBlur(_encode8(work), kernel))
    if method == "bilateral":
        radius = max(1, min(20, int(round(strength))))
        return cv2.bilateralFilter(
            work,
            radius * 2 + 1,
            sigmaColor=max(0.03, min(0.5, strength * 0.035)) * scale,
            sigmaSpace=max(1.0, strength * 1.5),
            borderType=cv2.BORDER_REFLECT,
        )
    if method == "nl_means":
        filtered = cv2.fastNlMeansDenoising(
            _encode8(work),
            None,
            h=max(1.0, strength * 2.5),
            templateWindowSize=7,
            searchWindowSize=21,
        )
        return _decode8(filtered)
    raise ValueError(T("Unknown denoising method: {0}", method))


def _hessian_eigenvalues(
    image: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return scale-normalized Hessian eigenvalues, ordered by ascending |value|."""

    smoothed = cv2.GaussianBlur(
        image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )
    scale = np.float32(sigma * sigma)
    dxx = cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, ksize=3, borderType=cv2.BORDER_REFLECT) * scale
    dyy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, ksize=3, borderType=cv2.BORDER_REFLECT) * scale
    dxy = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, ksize=3, borderType=cv2.BORDER_REFLECT) * scale
    half_trace = (dxx + dyy) * np.float32(0.5)
    delta = np.sqrt(
        ((dxx - dyy) * np.float32(0.5)) ** 2 + dxy * dxy, dtype=np.float32
    )
    lambda_plus = half_trace + delta
    lambda_minus = half_trace - delta
    larger_first = np.abs(lambda_plus) >= np.abs(lambda_minus)
    mu_small = np.where(larger_first, lambda_minus, lambda_plus).astype(np.float32)
    mu_large = np.where(larger_first, lambda_plus, lambda_minus).astype(np.float32)
    return mu_small, mu_large


def _gaussian_support_margin(sigma: float) -> int:
    """Conservative Gaussian support plus the one-pixel derivative stencil."""

    return max(2, int(math.ceil(4.0 * float(sigma))) + 2)


def _structure_tiles(
    shape: tuple[int, int],
    margin: int,
) -> Iterable[tuple[slice, slice, slice, slice]]:
    """Yield source and core slices for seamless overlapping tile processing."""

    height, width = (int(value) for value in shape[:2])
    for y0 in range(0, height, _STRUCTURE_TILE_SIDE):
        y1 = min(height, y0 + _STRUCTURE_TILE_SIDE)
        source_y0 = max(0, y0 - margin)
        source_y1 = min(height, y1 + margin)
        core_y = slice(y0 - source_y0, y1 - source_y0)
        for x0 in range(0, width, _STRUCTURE_TILE_SIDE):
            x1 = min(width, x0 + _STRUCTURE_TILE_SIDE)
            source_x0 = max(0, x0 - margin)
            source_x1 = min(width, x1 + margin)
            core_x = slice(x0 - source_x0, x1 - source_x0)
            yield (
                slice(source_y0, source_y1),
                slice(source_x0, source_x1),
                core_y,
                core_x,
            )


def _vesselness_from_eigenvalues(
    mu_small: np.ndarray,
    mu_large: np.ndarray,
    c_value: np.float32,
) -> np.ndarray:
    """Return one Frangi-scale response from ordered Hessian eigenvalues."""

    beta = np.float32(0.5)
    structureness = np.sqrt(
        mu_small * mu_small + mu_large * mu_large,
        dtype=np.float32,
    )
    safe_large = np.where(
        np.abs(mu_large) < 1e-8,
        np.float32(1e-8),
        mu_large,
    )
    blobness = mu_small / safe_large
    vessel = np.exp(
        -(blobness * blobness) / (np.float32(2.0) * beta * beta)
    ) * (
        np.float32(1.0)
        - np.exp(
            -(structureness * structureness)
            / (np.float32(2.0) * c_value * c_value)
        )
    )
    return np.where(mu_large <= 0.0, vessel, np.float32(0.0)).astype(np.float32)


def _apply_structure_strength(
    image: np.ndarray,
    response: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Make the shared strength control meaningful for normalized structure filters.

    A plain scalar multiplier is cancelled by the robust normalization performed
    by the segmentation pipeline.  Below 1.0 we instead blend the original signal
    back in; above 1.0 we progressively suppress weak structure responses.  The
    default 1.0 path is returned untouched.
    """

    value = float(np.clip(strength, 0.0, 3.0))
    if abs(value - 1.0) <= 1e-8:
        return response.astype(np.float32, copy=False)
    if value <= 0.0:
        return image.astype(np.float32, copy=False)
    normalized = robust_normalize(response)
    if value < 1.0:
        return (
            normalized * np.float32(value)
            + image.astype(np.float32, copy=False) * np.float32(1.0 - value)
        ).astype(np.float32, copy=False)
    return np.power(normalized, np.float32(value), dtype=np.float32)


def _vesselness_response_tiled(
    image: np.ndarray,
    sigmas: tuple[float, ...],
    strength: float,
) -> np.ndarray:
    """Memory-bounded two-pass Frangi response for large images."""

    best = np.zeros(image.shape, dtype=np.float32)
    for sigma in sigmas:
        margin = _gaussian_support_margin(sigma)
        tiles = tuple(_structure_tiles(image.shape[:2], margin))
        structure_max = 0.0
        # The original filter derives c from the whole scale.  A first tiled
        # pass recovers that same global reference without retaining full-frame
        # Hessian arrays.
        for source_y, source_x, core_y, core_x in tiles:
            mu_small, mu_large = _hessian_eigenvalues(
                image[source_y, source_x],
                sigma,
            )
            structureness = np.sqrt(
                mu_small * mu_small + mu_large * mu_large,
                dtype=np.float32,
            )
            structure_max = max(
                structure_max,
                float(structureness[core_y, core_x].max(initial=0.0)),
            )
        c_value = np.float32(max(1e-6, 0.5 * structure_max))
        for source_y, source_x, core_y, core_x in tiles:
            mu_small, mu_large = _hessian_eigenvalues(
                image[source_y, source_x],
                sigma,
            )
            vessel = _vesselness_from_eigenvalues(mu_small, mu_large, c_value)
            output_y = slice(source_y.start + core_y.start, source_y.start + core_y.stop)
            output_x = slice(source_x.start + core_x.start, source_x.start + core_x.stop)
            np.maximum(
                best[output_y, output_x],
                vessel[core_y, core_x],
                out=best[output_y, output_x],
            )
    return _apply_structure_strength(image, best, strength)


def _vesselness_response(image: np.ndarray, radius: int, strength: float) -> np.ndarray:
    """Frangi-style 2-D ridge filter enhancing bright tubular/fiber structures.

    normalize_signal already applied the polarity/invert setting upstream, so the
    positive signal is bright here; only bright ridges (large eigenvalue < 0)
    contribute.  Three scales derived from the radius are combined pixel-wise so a
    single control tracks the target fiber/vessel thickness.  Deterministic.
    """

    base_sigma = max(1.0, radius * 0.5)
    sigmas = tuple(max(1.0, base_sigma * factor) for factor in (0.75, 1.0, 1.5))
    if image.size > _STRUCTURE_TILE_THRESHOLD_PIXELS:
        return _vesselness_response_tiled(image, sigmas, strength)
    beta = np.float32(0.5)
    best: np.ndarray | None = None
    for sigma in sigmas:
        mu_small, mu_large = _hessian_eigenvalues(image, sigma)
        structureness = np.sqrt(
            mu_small * mu_small + mu_large * mu_large, dtype=np.float32
        )
        c_value = np.float32(max(1e-6, 0.5 * float(structureness.max(initial=0.0))))
        safe_large = np.where(np.abs(mu_large) < 1e-8, np.float32(1e-8), mu_large)
        blobness = mu_small / safe_large
        vessel = np.exp(
            -(blobness * blobness) / (np.float32(2.0) * beta * beta)
        ) * (
            np.float32(1.0)
            - np.exp(-(structureness * structureness) / (np.float32(2.0) * c_value * c_value))
        )
        vessel = np.where(mu_large <= 0.0, vessel, np.float32(0.0)).astype(np.float32)
        best = vessel if best is None else np.maximum(best, vessel)
    response = np.maximum(best, 0.0).astype(np.float32)
    return _apply_structure_strength(image, response, strength)


def _log_blob_response(image: np.ndarray, radius: int, strength: float) -> np.ndarray:
    """Scale-normalized Laplacian-of-Gaussian enhancing bright blobs/spots.

    The blob core has a strongly negative Laplacian; negating and multiplying by
    sigma**2 turns each spot centre into a positive response the standard
    threshold/watershed stage can pick and count.  Deterministic.
    """

    sigma = max(1.0, float(radius) * 0.7071)  # sigma ~= blob_radius / sqrt(2)
    if image.size <= _STRUCTURE_TILE_THRESHOLD_PIXELS:
        smoothed = cv2.GaussianBlur(
            image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
        )
        laplacian = cv2.Laplacian(
            smoothed,
            cv2.CV_32F,
            ksize=3,
            borderType=cv2.BORDER_REFLECT,
        )
        response = np.maximum(
            -laplacian * np.float32(sigma * sigma),
            0.0,
        ).astype(np.float32)
    else:
        response = np.zeros(image.shape, dtype=np.float32)
        margin = _gaussian_support_margin(sigma)
        for source_y, source_x, core_y, core_x in _structure_tiles(
            image.shape[:2],
            margin,
        ):
            tile = image[source_y, source_x]
            smoothed = cv2.GaussianBlur(
                tile,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT,
            )
            laplacian = cv2.Laplacian(
                smoothed,
                cv2.CV_32F,
                ksize=3,
                borderType=cv2.BORDER_REFLECT,
            )
            tile_response = np.maximum(
                -laplacian * np.float32(sigma * sigma),
                0.0,
            ).astype(np.float32)
            output_y = slice(source_y.start + core_y.start, source_y.start + core_y.stop)
            output_x = slice(source_x.start + core_x.start, source_x.start + core_x.stop)
            response[output_y, output_x] = tile_response[core_y, core_x]
    return _apply_structure_strength(image, response, strength)


def _background_response(
    image: np.ndarray,
    method: str,
    radius: int,
    strength: float,
) -> np.ndarray:
    work = image.astype(np.float32, copy=False)
    if method == "none" or radius <= 0 or strength <= 0:
        return np.maximum(work, 0.0)

    radius = max(1, int(radius))
    # Very large morphology kernels are both memory-heavy and disproportionately
    # slow. Above 128 px, use a smooth low-frequency approximation so the UI can
    # expose a genuinely wide background range without risking multi-gigabyte
    # temporary kernels.
    if method == "morph_open":
        if radius <= 128:
            background = _ellipse_open_reflect(work, radius)
        else:
            background = cv2.GaussianBlur(
                work,
                (0, 0),
                sigmaX=max(1.0, radius * 0.50),
                sigmaY=max(1.0, radius * 0.50),
                borderType=cv2.BORDER_REFLECT,
            )
        return np.maximum(work - np.float32(strength) * background, 0.0)

    if method == "vesselness":
        return _vesselness_response(work, radius, strength)
    if method == "log":
        return _log_blob_response(work, radius, strength)

    sigma = max(1.0, radius * 0.55)
    gaussian_background = cv2.GaussianBlur(
        work,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    if method == "gaussian":
        return np.maximum(work - np.float32(strength) * gaussian_background, 0.0)
    if method == "dog":
        fine_sigma = max(0.5, min(4.0, radius * 0.08))
        fine = cv2.GaussianBlur(
            work,
            (0, 0),
            sigmaX=fine_sigma,
            sigmaY=fine_sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        return np.maximum(fine - np.float32(strength) * gaussian_background, 0.0)
    if method == "flat_field":
        ratio = (work + np.float32(0.02)) / (gaussian_background + np.float32(0.02))
        response = np.maximum(ratio - np.float32(1.0), 0.0)
        if strength != 1.0:
            response *= np.float32(strength)
        return response
    if method == "multiscale":
        if radius <= 128:
            fine_background = _ellipse_open_reflect(work, radius)
        else:
            fine_background = cv2.GaussianBlur(
                work,
                (0, 0),
                sigmaX=max(1.0, radius * 0.32),
                sigmaY=max(1.0, radius * 0.32),
                borderType=cv2.BORDER_REFLECT,
            )
        fine = np.maximum(work - np.float32(strength) * fine_background, 0.0)
        coarse = np.maximum(work - np.float32(strength) * gaussian_background, 0.0)
        return np.maximum(fine, coarse)
    raise ValueError(T("Unknown background correction method: {0}", method))


def _robust_snr_response(
    signal: np.ndarray,
    params: UniversalParams,
    *,
    noise_reference_signal: np.ndarray | None = None,
) -> np.ndarray:
    """Return positive local evidence in robust background-noise sigmas.

    This path deliberately retains intensity effect size.  It never ranks the
    pixels and never stretches the residual tail to a fixed occupancy.  A
    uniform offset is removed by the local (or global) background, while a
    positive linear gain is cancelled by the residual MAD.  Consequently an
    image with no measurable residual variation can produce an empty mask.
    """

    work = signal.astype(np.float32, copy=False)
    if params.invert:
        work = np.negative(work, dtype=np.float32)
    finite_signal = _finite_sample(work)
    finite_centre = float(np.median(finite_signal))
    work_replaced = not bool(np.all(np.isfinite(work)))
    if work_replaced:
        work = np.nan_to_num(
            work,
            copy=True,
            nan=finite_centre,
            posinf=finite_centre,
            neginf=finite_centre,
        )
    noise_work = work
    if noise_reference_signal is not None:
        noise_work = noise_reference_signal.astype(np.float32, copy=False)
        if params.invert:
            noise_work = np.negative(noise_work, dtype=np.float32)
        noise_finite = _finite_sample(noise_work)
        noise_centre = float(np.median(noise_finite))
        if not bool(np.all(np.isfinite(noise_work))):
            noise_work = np.nan_to_num(
                noise_work,
                copy=True,
                nan=noise_centre,
                posinf=noise_centre,
                neginf=noise_centre,
            )

    encoded_noise_floor = 0.0
    if params.denoise_method in ("median", "nl_means", "bilateral"):
        # These filters use an 8-bit encode / intensity-difference sigma that
        # assume a known value window.  The robust-SNR signal is raw-scale (and
        # possibly inverted, hence negative), so use a robust value window.
        # Full min/max is unsafe: one saturated pixel can compress the real
        # background noise into one or two 8-bit levels and manufacture huge
        # SNR values.  Central quantiles retain real objects (tail values clip
        # to the endpoints) while keeping normal noise resolvable.
        # The sample of ``work`` is unchanged unless non-finite pixels were
        # replaced above, so reuse it instead of resampling the full frame.
        finite_window = _finite_sample(work) if work_replaced else finite_signal
        work_lo, work_hi = (
            float(value)
            for value in np.percentile(finite_window, (0.1, 99.9))
        )
        work_span = work_hi - work_lo
        if not math.isfinite(work_span) or work_span <= 0.0:
            work_lo = float(np.min(finite_window))
            work_hi = float(np.max(finite_window))
            work_span = work_hi - work_lo
        if not math.isfinite(work_span) or work_span <= 0.0:
            work_lo, work_span = finite_centre, 1.0
        denoised = _denoise(
            work,
            params.denoise_method,
            params.denoise_strength,
            encode_offset=work_lo,
            encode_scale=work_span,
        )
        median_radius = max(1, int(round(params.denoise_strength)))
        uses_8bit_encode = params.denoise_method == "nl_means" or (
            params.denoise_method == "median" and median_radius * 2 + 1 > 5
        )
        if uses_8bit_encode:
            # Estimate sensor/background noise from adjacent raw-pixel
            # differences.  A wide median can leave large, flat quantization
            # plateaus, making the MAD of the *filtered* residual exactly zero;
            # using only one 8-bit code step then promotes harmless plateau
            # edges to 10–20 sigma.  The adjacent-difference estimator is
            # translation invariant, robust to sparse real objects and sampled
            # to stay memory bounded on very large images.
            height, width = noise_work.shape[:2]
            max_pairs_per_axis = 250_000
            differences: list[np.ndarray] = []
            if width > 1:
                row_step = max(
                    1,
                    int(math.ceil(height * (width - 1) / max_pairs_per_axis)),
                )
                differences.append(
                    np.diff(noise_work[::row_step, :], axis=1).reshape(-1)
                )
            if height > 1:
                column_step = max(
                    1,
                    int(math.ceil(width * (height - 1) / max_pairs_per_axis)),
                )
                differences.append(
                    np.diff(noise_work[:, ::column_step], axis=0).reshape(-1)
                )
            raw_noise_sigma = 0.0
            if differences:
                adjacent = np.concatenate(differences).astype(np.float32, copy=False)
                adjacent = adjacent[np.isfinite(adjacent)]
                if adjacent.size:
                    adjacent_centre = float(np.median(adjacent))
                    adjacent_mad = float(
                        np.median(np.abs(adjacent - np.float32(adjacent_centre)))
                    )
                    raw_noise_sigma = 1.4826 * adjacent_mad / math.sqrt(2.0)
                    if not math.isfinite(raw_noise_sigma) or raw_noise_sigma <= 0.0:
                        adjacent_p16, adjacent_p84 = np.percentile(
                            adjacent,
                            (16.0, 84.0),
                        )
                        raw_noise_sigma = (
                            float(adjacent_p84 - adjacent_p16)
                            * 0.5
                            / math.sqrt(2.0)
                        )
            if not math.isfinite(raw_noise_sigma) or raw_noise_sigma <= 0.0:
                raw_noise_sigma = 0.0
            encoded_noise_floor = max(
                work_span / 255.0,
                # A deliberately conservative fraction of native adjacent
                # noise.  This still leaves a real 5-sigma object above the
                # decision seed after denoising, while suppressing isolated
                # interpolation/quantization ridges in otherwise signal-free
                # backgrounds.
                raw_noise_sigma * 0.60,
            )
    else:
        denoised = _denoise(work, params.denoise_method, params.denoise_strength)
    if params.background_method == "none" or params.background_radius <= 0:
        background_level = float(np.median(_finite_sample(denoised)))
        residual = denoised - np.float32(background_level)
    else:
        # robust_snr has one intentionally simple local-background model.
        # Other background_method names are compatibility metadata here; every
        # enabled local correction uses the same exact Gaussian subtraction.
        # In particular, do not multiply the background by background_strength:
        # a value other than one would reintroduce sensitivity to uniform
        # brightness offsets.
        sigma = max(1.0, float(params.background_radius) * 0.55)
        background = cv2.GaussianBlur(
            denoised,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        residual = denoised - background

    finite_residual = _finite_sample(residual)
    centre = float(np.median(finite_residual))
    mad = float(np.median(np.abs(finite_residual - centre)))
    if math.isfinite(mad) and mad > 0.0:
        noise_sigma = 1.4826 * mad
    else:
        p16, p84 = np.percentile(finite_residual, (16.0, 84.0))
        noise_sigma = float(p84 - p16) * 0.5
    if not math.isfinite(noise_sigma) or noise_sigma <= 0.0:
        # A noiseless synthetic/quantized field may contain a sparse, perfectly
        # real object while more than 84% of the residual is exactly zero.  In
        # that case the central robust scales are both zero, but the residual
        # dynamic range still proves that contrast exists.  Use only a tiny
        # numerical floor; a truly constant field has zero dynamic range and
        # remains an empty response.
        residual_min = float(finite_residual.min(initial=0.0))
        residual_max = float(finite_residual.max(initial=0.0))
        residual_span = residual_max - residual_min
        if not math.isfinite(residual_span) or residual_span <= 0.0:
            return np.zeros(signal.shape, dtype=np.float32)
        magnitude = max(abs(residual_min), abs(residual_max), 1.0)
        noise_sigma = max(
            residual_span * 1e-6,
            float(np.finfo(np.float32).eps) * magnitude * 16.0,
        )
    if encoded_noise_floor > 0.0 and math.isfinite(encoded_noise_floor):
        # The decoded 8-bit working image cannot justify a noise estimate below
        # one code step.  Enforcing that floor prevents quantization plateaus
        # from turning a lone clipped pixel into widespread false positives.
        noise_sigma = max(noise_sigma, encoded_noise_floor)

    if params.noise_reference_value > 0.0:
        # Shared absolute decision scale.  The residual is still the same local
        # high-pass, but it is now divided by one fixed grey-level scale for the
        # whole batch instead of by this image's own texture.  ``snr_seed_sigma``
        # therefore names an absolute contrast, identical in every field, which
        # is the only way a count from a quiet cortical field and a count from a
        # texture-rich field can be placed on the same axis.
        noise_sigma = float(params.noise_reference_value)
        if params.invert:
            # ``work`` was negated above; the residual keeps the source scale, so
            # the reference needs no sign change, only the same magnitude.
            noise_sigma = abs(noise_sigma)
        del measured_noise_sigma

    response = (
        residual.astype(np.float32, copy=False) - np.float32(centre)
    ) / np.float32(noise_sigma)
    np.maximum(response, np.float32(0.0), out=response)
    if not bool(np.all(np.isfinite(response))):
        response = np.nan_to_num(
            response,
            copy=False,
            nan=0.0,
            posinf=float(np.finfo(np.float32).max),
            neginf=0.0,
        )
    return response.astype(np.float32, copy=False)


def _broad_background_map(
    signal: np.ndarray,
    background_radius: int,
    compartment_scale: int = 0,
) -> np.ndarray:
    """Local background at the scale of the compartment being judged.

    ``compartment_scale`` is a radius in pixels.  When it is zero the scale is
    derived from ``background_radius`` for backward compatibility.  Both are
    scaled with the analysis grid by ``scaled_spatial``, so a preview and a
    full-resolution run gate the same regions.
    """

    effective = int(compartment_scale) if compartment_scale > 0 else max(
        1, int(background_radius)
    )
    sigma = max(2.0, float(effective) * 1.4)
    return cv2.GaussianBlur(
        signal.astype(np.float32, copy=False),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )


def _auto_background_ceiling(broad: np.ndarray) -> float | None:
    """Split this image's broad background into two modes and return the cut.

    Otsu on the broad background, not a fixed percentile.  The distinction
    matters: a percentile rule would always remove the same *fraction* of every
    field, which is a rank rule in disguise and would keep excluding parts of a
    field that contains no bright compartment at all.  Otsu instead looks for an
    actual bimodal boundary, so a uniform field yields a cut at its own extreme
    and almost nothing is excluded.

    Returns ``None`` when the background carries no usable dynamic range, which
    leaves the response untouched rather than guessing.
    """

    sample = _finite_sample(broad, max_pixels=400_000)
    low, high = (float(v) for v in np.percentile(sample, (0.5, 99.5)))
    span = high - low
    if not math.isfinite(span) or span <= 0.0:
        return None
    quantised = np.clip((broad - np.float32(low)) / np.float32(span), 0.0, 1.0)
    level, _ = cv2.threshold(
        (quantised * 255.0).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return low + (float(level) / 255.0) * span


def _apply_background_ceiling(
    response: np.ndarray,
    signal: np.ndarray,
    params: UniversalParams,
) -> np.ndarray:
    """Zero the response inside the bright broad-background compartment.

    Broad glow is sample autofluorescence, folds, thickened edges and similar
    continuously bright regions.  Their brightness is a property of the
    compartment rather than of any object inside it, so the ceiling is an
    absolute grey level shared by the whole batch, exactly like
    ``noise_reference_value``.  Objects are excluded by where they sit, never by
    how they rank inside their own image.
    """

    mode = params.background_ceiling_mode
    if mode == "off":
        return response
    broad = _broad_background_map(
        signal,
        params.background_radius,
        params.background_ceiling_scale,
    )
    if mode == "absolute":
        if params.background_ceiling <= 0.0:
            return response
        allowed = broad <= np.float32(params.background_ceiling)
    else:
        ceiling = _auto_background_ceiling(broad)
        if ceiling is None:
            return response
        allowed = broad <= np.float32(ceiling)
    if not bool(np.any(allowed)):
        return np.zeros(response.shape, dtype=np.float32)
    gated = response.astype(np.float32, copy=True)
    gated[~allowed] = np.float32(0.0)
    return gated


def robust_normalize(image: np.ndarray, percentile: float = 99.9) -> np.ndarray:
    finite = _finite_sample(image)
    low, high = (float(value) for value in np.percentile(finite, (0.0, percentile)))
    low = max(0.0, low)
    if not math.isfinite(high) or high <= low:
        high = float(finite.max(initial=1.0))
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    output = (image.astype(np.float32, copy=False) - np.float32(low)) / np.float32(
        high - low
    )
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)


def _encode_unit_interval(image: np.ndarray, full_scale: float) -> np.ndarray:
    """``np.rint(np.clip(image, 0, 1) * full_scale)`` with one working buffer.

    Multiplying first and clipping to ``[0, full_scale]`` afterwards yields the
    same float32 values (``full_scale`` is exactly representable and the clip
    bounds map exactly), so the three passes can run in place on a single
    temporary instead of allocating a full frame per step.
    """

    work = image * full_scale
    if work is image:  # pragma: no cover - defensive; ``*`` always allocates
        work = np.array(work)
    np.clip(work, 0.0, full_scale, out=work)
    np.rint(work, out=work)
    return work


def _global_threshold(image: np.ndarray, method: str) -> float:
    if not bool(np.any(image > 0)):
        return float("inf")
    encoded = _encode_unit_interval(image, 65535.0).astype(np.uint16)
    flag = cv2.THRESH_OTSU if method == "otsu" else cv2.THRESH_TRIANGLE
    try:
        threshold, _ = cv2.threshold(
            encoded,
            0,
            65535,
            cv2.THRESH_BINARY | flag,
        )
        value = float(threshold) / 65535.0
    except cv2.error:
        encoded8 = _encode_unit_interval(image, 255.0).astype(np.uint8)
        threshold, _ = cv2.threshold(
            encoded8,
            0,
            255,
            cv2.THRESH_BINARY | flag,
        )
        value = float(threshold) / 255.0
    if value <= 0:
        positive = image[image > 0]
        if positive.size == 0:
            return float("inf")
        value = float(np.percentile(positive, 95.0))
    return value


def _hysteresis_mask(image: np.ndarray, low: float, high: float) -> np.ndarray:
    if not math.isfinite(high):
        return np.zeros(image.shape, dtype=bool)
    seeds = image >= high
    candidates = image >= low
    if not bool(seeds.any()):
        return np.zeros(image.shape, dtype=bool)
    count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(labels[seeds])] = True
    keep[0] = False
    return keep[labels]


def _hysteresis_candidates(
    seeds: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Keep candidate components that contain at least one strong seed."""

    if not bool(seeds.any()):
        return np.zeros(seeds.shape, dtype=bool)
    count, labels = cv2.connectedComponents(
        candidates.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(labels[seeds])] = True
    keep[0] = False
    return keep[labels]


def _box_mean_std(image: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    mean = cv2.boxFilter(
        image,
        ddepth=cv2.CV_32F,
        ksize=(window, window),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    square_mean = cv2.boxFilter(
        image * image,
        ddepth=cv2.CV_32F,
        ksize=(window, window),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    variance = np.maximum(square_mean - mean * mean, 0.0)
    return mean, np.sqrt(variance, dtype=np.float32)


def _snr_seed_high(params: UniversalParams) -> float:
    """Fixed high-z seed for the robust-SNR hysteresis threshold.

    It depends only on the recipe (seed sigma and sensitivity), never on the
    image, so it is also the correct object response-confidence reference on any
    analysis grid and for manual rebuilds — unlike an Otsu value recomputed on a
    resized sigma-scale response, which a unit-interval Otsu would clip.
    """

    return float(params.snr_seed_sigma) * 2.0 ** (
        (50.0 - float(params.sensitivity)) / 70.0
    )


def _threshold_mask(
    response: np.ndarray,
    params: UniversalParams,
) -> tuple[np.ndarray, float, float, float | None]:
    # The fourth return value is the evidence reference used for whole-object
    # confidence.  It is the fixed high-z seed for robust_snr, the image-wide
    # Otsu response threshold when that value was already computed, and None
    # when the caller must retain the legacy fallback.
    method = params.threshold_method
    sensitivity_factor = 2.0 ** ((50.0 - params.sensitivity) / 35.0)
    sensitivity_factor = float(np.clip(sensitivity_factor, 0.30, 3.20))

    if method == "snr_hysteresis":
        high = _snr_seed_high(params)
        low = high * float(params.grow_ratio)
        return _hysteresis_mask(response, low, high), high, low, high

    if method in {"hysteresis_otsu", "otsu", "triangle"}:
        base_method = "triangle" if method == "triangle" else "otsu"
        auto = _global_threshold(response, base_method)
        otsu_reference = auto if base_method == "otsu" else None
        high = auto * sensitivity_factor
        low = high * params.grow_ratio
        if method == "hysteresis_otsu":
            return _hysteresis_mask(response, low, high), float(high), float(low), otsu_reference
        return response >= high, float(high), float(high), otsu_reference

    if method == "manual":
        high = params.manual_threshold
        low = high * params.grow_ratio
        if params.grow_ratio < 0.999:
            return _hysteresis_mask(response, low, high), float(high), float(low), None
        return response >= high, float(high), float(high), None

    if method == "percentile":
        positive = response[response > 0]
        if positive.size == 0:
            return np.zeros(response.shape, dtype=bool), float("inf"), float("inf"), None
        effective_percentile = float(
            np.clip(
                params.threshold_percentile + (50.0 - params.sensitivity) * 0.22,
                0.0,
                100.0,
            )
        )
        high = float(np.percentile(positive, effective_percentile))
        low = high * params.grow_ratio
        return _hysteresis_mask(response, low, high), high, low, None

    window = params.local_window
    sensitivity_offset = (50.0 - params.sensitivity) / 100.0
    local_std: np.ndarray | None = None
    if method == "local_gaussian":
        local_mean = cv2.GaussianBlur(
            response,
            (window, window),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_REFLECT,
        )
        threshold_map = local_mean + np.float32(
            params.local_k + sensitivity_offset * 0.28
        )
    elif method == "local_mean":
        local_mean = cv2.boxFilter(
            response,
            ddepth=cv2.CV_32F,
            ksize=(window, window),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        threshold_map = local_mean + np.float32(
            params.local_k + sensitivity_offset * 0.28
        )
    elif method in {"mean_std", "sauvola"}:
        local_mean, local_std = _box_mean_std(response, window)
        if method == "mean_std":
            effective_k = params.local_k + sensitivity_offset * 0.9
            threshold_map = local_mean + np.float32(effective_k) * local_std
        else:
            effective_k = float(np.clip(params.local_k + sensitivity_offset * 0.45, -1.0, 1.0))
            threshold_map = local_mean * (
                np.float32(1.0)
                + np.float32(effective_k)
                * (local_std / np.float32(0.5) - np.float32(1.0))
            )
    else:
        raise ValueError(T("Unknown threshold method: {0}", method))

    # Local methods previously ignored grow_ratio, so the simple "core / full"
    # control appeared to work but produced an identical mask.  Grow from the
    # spatially varying strong threshold into a slightly lower local threshold.
    # The contrast-aware margin avoids multiplying low background thresholds
    # directly, which can otherwise connect most of an uneven field at once.
    threshold_map = np.clip(threshold_map, 0.0, 1.0)
    positive_response = response > np.float32(np.finfo(np.float32).eps)
    strong = positive_response & (response >= threshold_map)
    if params.grow_ratio < 0.999:
        if local_std is None:
            _, local_std = _box_mean_std(response, window)
        contrast_margin = np.maximum(
            np.abs(threshold_map - local_mean),
            np.float32(0.35) * local_std,
        )
        low_map = np.clip(
            threshold_map
            - np.float32(1.0 - params.grow_ratio) * contrast_margin,
            0.0,
            1.0,
        )
        candidates = positive_response & (response >= low_map)
        mask = _hysteresis_candidates(strong, candidates)
        low_value = float(np.mean(low_map, dtype=np.float64))
    else:
        mask = strong
        low_value = float(np.mean(threshold_map, dtype=np.float64))
    threshold_value = float(np.mean(threshold_map, dtype=np.float64))
    return mask, threshold_value, low_value, None


def _filter_snr_evidence_components(
    mask: np.ndarray,
    response: np.ndarray,
    high: float,
    low: float,
) -> np.ndarray:
    """Reject grown components whose aggregate z evidence is too weak."""

    if not bool(mask.any()):
        return np.zeros(mask.shape, dtype=bool)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64, copy=False)
    sums = np.bincount(
        labels.ravel(),
        weights=response.ravel(),
        minlength=count,
    )
    means = np.divide(
        sums,
        np.maximum(areas, 1.0),
        out=np.zeros(count, dtype=np.float64),
    )
    mean_floor = max(float(low), 0.48 * float(high))
    p90_floor = 0.64 * float(high)
    keep = means >= mean_floor
    keep[0] = False

    stats_list = stats.tolist()
    for label_id in np.flatnonzero(keep).tolist():
        x, y, width, height, _area = stats_list[label_id]
        local_labels = labels[y : y + height, x : x + width]
        component_values = response[y : y + height, x : x + width][
            local_labels == label_id
        ]
        if component_values.size == 0 or _percentile_1d(
            component_values,
            90.0,
        ) < p90_floor:
            keep[label_id] = False
    return keep[labels]


def _percentile_1d(values: np.ndarray, q: float) -> float:
    """``float(np.percentile(values, q))`` for a 1-D array, ~10x cheaper.

    This mirrors numpy's default ``linear`` method for a scalar ``q`` exactly:
    the same virtual index, the same two order statistics selected by a
    partition, and the same interpolation arithmetic in the array's own dtype
    (numpy treats a Python-float ``q`` as a weak scalar, so a float32 input is
    interpolated in float32).  It exists because the SNR evidence gate calls
    this once per candidate component.
    """

    count = int(values.size)
    quantile = np.true_divide(q, 100)
    virtual = (count - 1) * quantile
    previous = math.floor(virtual)
    following = previous + 1
    if virtual >= count - 1:
        previous = following = count - 1
        gamma = float(virtual + 1)
    elif virtual < 0:
        previous = following = 0
        gamma = float(virtual)
    else:
        gamma = float(virtual - previous)
    if following < count:
        partitioned = np.partition(values, (previous, following))
    else:
        partitioned = np.partition(values, previous)
    lower = partitioned[previous]
    upper = partitioned[following]
    difference = upper - lower
    if gamma >= 0.5:
        return float(upper - difference * (1 - gamma))
    return float(lower + difference * gamma)


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0 or not bool(mask.any()):
        return mask.astype(bool, copy=False)
    inverse = (~mask).astype(np.uint8)
    # Foreground objects use 8-connectivity; the complementary background must
    # use 4-connectivity to avoid diagonal leaks through a valid closed rim.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    is_border = np.zeros(count, dtype=bool)
    is_border[border_labels] = True
    areas = stats[:, cv2.CC_STAT_AREA]
    fill = (areas <= max_hole_area) & (~is_border)
    fill[0] = False
    return mask | fill[labels]


def _apply_morphology(mask: np.ndarray, params: UniversalParams) -> np.ndarray:
    output = mask.astype(np.uint8)
    if params.open_radius > 0 and bool(output.any()):
        output = cv2.morphologyEx(
            output,
            cv2.MORPH_OPEN,
            _ellipse_kernel(params.open_radius),
        )
    if params.close_radius > 0 and bool(output.any()):
        output = cv2.morphologyEx(
            output,
            cv2.MORPH_CLOSE,
            _ellipse_kernel(params.close_radius),
        )
    if params.erode_radius > 0 and bool(output.any()):
        output = cv2.erode(output, _ellipse_kernel(params.erode_radius))
    if params.dilate_radius > 0 and bool(output.any()):
        output = cv2.dilate(output, _ellipse_kernel(params.dilate_radius))
    return _fill_small_holes(output.astype(bool), params.hole_area)


def _remove_border_components(mask: np.ndarray) -> np.ndarray:
    """Remove whole parent components before any watershed partitioning."""

    if not bool(mask.any()):
        return mask.astype(bool, copy=False)
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    border_ids = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    remove = np.zeros(int(labels.max(initial=0)) + 1, dtype=bool)
    remove[border_ids] = True
    remove[0] = False
    return mask & (~remove[labels])


def _repair_partition_connectivity(
    local_mask: np.ndarray,
    nearest: np.ndarray,
    active_seeds: np.ndarray,
) -> np.ndarray:
    """Make every seed partition singly connected without dropping mask pixels."""

    nearest_ids = np.unique(nearest[local_mask])
    partition = np.zeros(local_mask.shape, dtype=np.int32)
    for partition_id, nearest_id in enumerate(nearest_ids.tolist(), start=1):
        partition[local_mask & (nearest == nearest_id)] = partition_id

    # Euclidean Voronoi cells can intersect a concave mask in disconnected
    # islands. Keep the connected part that owns the seed; the remaining pixels
    # are reassigned below through paths constrained to the source mask.
    for partition_id in range(1, len(nearest_ids) + 1):
        region = partition == partition_id
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            region.astype(np.uint8),
            connectivity=8,
        )
        if count <= 2:
            continue
        seed_components = np.unique(components[region & active_seeds])
        seed_components = seed_components[seed_components > 0]
        if seed_components.size:
            main_component = int(
                seed_components[
                    np.argmax(stats[seed_components, cv2.CC_STAT_AREA])
                ]
            )
        else:
            main_component = int(
                1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            )
        partition[region & (components != main_component)] = 0

    missing = local_mask & (partition == 0)
    if not bool(missing.any()):
        return partition

    height, width = local_mask.shape
    frontier: list[tuple[int, int, int]] = []
    for y, x in np.argwhere(missing).tolist():
        y0 = max(0, y - 1)
        y1 = min(height, y + 2)
        x0 = max(0, x - 1)
        x1 = min(width, x + 2)
        neighbours = partition[y0:y1, x0:x1]
        positive = neighbours[neighbours > 0]
        if positive.size:
            frontier.append((y, x, int(positive.min())))

    queue: deque[tuple[int, int, int]] = deque()
    for y, x, partition_id in frontier:
        if partition[y, x] == 0:
            partition[y, x] = partition_id
            queue.append((y, x, partition_id))
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while queue:
        y, x, partition_id = queue.popleft()
        for dy, dx in offsets:
            neighbour_y = y + dy
            neighbour_x = x + dx
            if (
                0 <= neighbour_y < height
                and 0 <= neighbour_x < width
                and local_mask[neighbour_y, neighbour_x]
                and partition[neighbour_y, neighbour_x] == 0
            ):
                partition[neighbour_y, neighbour_x] = partition_id
                queue.append((neighbour_y, neighbour_x, partition_id))

    if bool(np.any(local_mask & (partition == 0))):
        raise RuntimeError(T(
            "The watershed connectivity repair failed to cover the original object",
        ))
    return partition


def _continuous_geometry_evidence(component: np.ndarray) -> tuple[float, float]:
    """Return covariance aspect ratio and outer solidity for split protection."""

    y, x = np.nonzero(component)
    if x.size >= 2:
        covariance = np.asarray(
            np.cov(np.vstack((x, y)), bias=True),
            dtype=np.float64,
        )
    else:
        covariance = np.zeros((2, 2), dtype=np.float64)
    covariance += np.eye(2, dtype=np.float64) / 12.0
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    aspect_ratio = math.sqrt(
        float(eigenvalues[0]) / max(float(eigenvalues[1]), 1e-12)
    )

    contours, _ = cv2.findContours(
        component.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return aspect_ratio, 1.0
    points = np.concatenate(contours, axis=0)
    outer_area = sum(float(abs(cv2.contourArea(contour))) for contour in contours)
    hull_area = (
        float(cv2.contourArea(cv2.convexHull(points)))
        if len(points) >= 3
        else outer_area
    )
    outer_solidity = min(1.0, outer_area / hull_area) if hull_area > 0 else 1.0
    return aspect_ratio, outer_solidity


def _distance_split_labels(
    mask: np.ndarray,
    peak_distance: int,
    peak_height: float,
    shape_profile: str = "any",
) -> np.ndarray:
    mask8 = mask.astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        mask8,
        connectivity=8,
    )
    if count <= 1:
        return np.zeros(mask.shape, dtype=np.int32)

    output = np.zeros(mask.shape, dtype=np.int32)
    next_label = 1
    radius = max(1, int(peak_distance))
    continuous_profiles = {"fiber", "patch", "ring"}

    # Split each original connected component inside its own bounding box.
    # A single global Voronoi can assign pixels to a seed from a neighbouring,
    # disconnected component and silently merge their labels.
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        local_components = components[y : y + height, x : x + width]
        local_mask = local_components == component_id
        target = output[y : y + height, x : x + width]

        # One padded copy and one distance transform serve the hole test, the
        # thickness metrics and the watershed peaks below (each used to build
        # its own copy and transform).
        padded_mask = _pad_uint8_border(local_mask)
        hole_area = _internal_hole_stats_padded(padded_mask)[0]
        hole_fraction = float(hole_area / max(area + hole_area, 1))
        padded_distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)
        mean_thickness, max_thickness, elongation = _thickness_from_distance(
            local_mask,
            padded_distance,
            area,
        )
        aspect_ratio = 1.0
        outer_solidity = 1.0
        if shape_profile == "any" and elongation >= 8.0:
            aspect_ratio, outer_solidity = _continuous_geometry_evidence(local_mask)
        protect_continuous = (
            hole_fraction >= 0.025
            or shape_profile in continuous_profiles
            or (
                shape_profile == "any"
                and elongation >= 8.0
                and (
                    outer_solidity <= 0.60
                    or (
                        aspect_ratio >= 3.0
                        and outer_solidity >= 0.92
                        and max_thickness / max(mean_thickness, 1.0) <= 1.18
                    )
                )
            )
        )
        if protect_continuous:
            target[local_mask] = next_label
            next_label += 1
            continue

        distance = padded_distance[1:-1, 1:-1]
        component_max = float(distance.max(initial=0.0))
        local_max = cv2.dilate(distance, _ellipse_kernel(radius))
        peaks = (
            local_mask
            & (distance >= local_max - np.float32(1e-5))
            & (distance >= component_max * np.float32(peak_height))
            & (distance > 0)
        )
        if not bool(peaks.any()):
            peak_y, peak_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            peaks[peak_y, peak_x] = True
        seed_count, seed_labels = cv2.connectedComponents(
            peaks.astype(np.uint8),
            connectivity=8,
        )
        if seed_count <= 2:
            target[local_mask] = next_label
            next_label += 1
            continue

        # Local distance labels retain every source pixel while guaranteeing
        # that no final label can cross its original parent component.
        seed_input = np.ones(local_mask.shape, dtype=np.uint8)
        seed_input[seed_labels > 0] = 0
        _, nearest = cv2.distanceTransformWithLabels(
            seed_input,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_CCOMP,
        )
        nearest_ids = np.unique(nearest[local_mask])
        active_seeds = seed_labels > 0
        if len(nearest_ids) >= 3:
            # A multi-lobed union can create a high but geometrically invalid
            # peak at the overlap centre.  Its Voronoi cell is much smaller
            # than the disk implied by the peak radius (for three touching
            # circles this used to create a fourth, artificial object).  Drop
            # only those unsupported seeds and reassign their pixels to the
            # remaining genuine peaks; never reduce a split below two parts.
            seed_to_nearest: dict[int, int] = {}
            peak_radius: dict[int, float] = {}
            # Group seed pixels by label in a single pass instead of scanning the
            # whole array once per seed with np.where(seed_labels == seed_id).
            # Stable sort keeps row-major order inside each group, so the argmax
            # (first-maximum) selection below is identical to the per-seed version.
            seed_pixel_y, seed_pixel_x = np.nonzero(seed_labels)
            seed_pixel_id = seed_labels[seed_pixel_y, seed_pixel_x]
            group_order = np.argsort(seed_pixel_id, kind="stable")
            grouped_y = seed_pixel_y[group_order]
            grouped_x = seed_pixel_x[group_order]
            grouped_id = seed_pixel_id[group_order]
            unique_seed_ids, group_starts = np.unique(grouped_id, return_index=True)
            group_bounds = group_starts.tolist() + [int(grouped_id.size)]
            for group_index, seed_id in enumerate(unique_seed_ids.tolist()):
                seed_y = grouped_y[group_bounds[group_index] : group_bounds[group_index + 1]]
                seed_x = grouped_x[group_bounds[group_index] : group_bounds[group_index + 1]]
                best = int(np.argmax(distance[seed_y, seed_x]))
                nearest_id = int(nearest[seed_y[best], seed_x[best]])
                seed_to_nearest[int(seed_id)] = nearest_id
                peak_radius[nearest_id] = max(
                    peak_radius.get(nearest_id, 0.0),
                    float(distance[seed_y[best], seed_x[best]]),
                )
            assigned_areas = {
                int(nearest_id): int(np.count_nonzero(local_mask & (nearest == nearest_id)))
                for nearest_id in nearest_ids.tolist()
            }
            unsupported = {
                nearest_id
                for nearest_id, assigned_area in assigned_areas.items()
                if assigned_area
                / max(math.pi * peak_radius.get(nearest_id, 0.0) ** 2, 1.0)
                < 0.38
            }
            if unsupported and len(nearest_ids) - len(unsupported) >= 2:
                retained_seed_ids = [
                    seed_id
                    for seed_id, nearest_id in seed_to_nearest.items()
                    if nearest_id not in unsupported
                ]
                retained_seeds = np.isin(seed_labels, retained_seed_ids)
                seed_input = np.ones(local_mask.shape, dtype=np.uint8)
                seed_input[retained_seeds] = 0
                _, nearest = cv2.distanceTransformWithLabels(
                    seed_input,
                    cv2.DIST_L2,
                    5,
                    labelType=cv2.DIST_LABEL_CCOMP,
                )
                nearest_ids = np.unique(nearest[local_mask])
                active_seeds = retained_seeds

        partition = _repair_partition_connectivity(
            local_mask,
            nearest,
            active_seeds,
        )
        for local_label in range(1, int(partition.max(initial=0)) + 1):
            selected = partition == local_label
            if bool(selected.any()):
                target[selected] = next_label
                next_label += 1
    return output


LabelBoxes = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _component_labels_with_boxes(
    mask: np.ndarray,
    params: UniversalParams,
) -> tuple[np.ndarray, LabelBoxes | None]:
    """Label the mask and, when labelling is plain, return per-label boxes.

    ``cv2.connectedComponentsWithStats`` runs the same labelling algorithm as
    ``cv2.connectedComponents`` (identical label numbering) and produces the
    exact bounding boxes and pixel counts that ``_label_bounding_boxes`` would
    otherwise recompute from a full-frame ``nonzero`` scan.
    """

    if params.split_touching and bool(mask.any()):
        return (
            _distance_split_labels(
                mask,
                params.watershed_peak_distance,
                params.watershed_peak_height,
                params.shape_profile,
            ),
            None,
        )
    return _plain_component_labels_with_boxes(mask)


def _plain_component_labels_with_boxes(
    mask: np.ndarray,
) -> tuple[np.ndarray, LabelBoxes | None]:
    """8-connected labels plus exact per-label boxes from one OpenCV call."""

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        _mask_as_uint8(mask),
        connectivity=8,
    )
    labels = labels.astype(np.int32, copy=False)
    if count <= 1:
        return labels, None
    left = stats[:, cv2.CC_STAT_LEFT].astype(np.int64)
    top = stats[:, cv2.CC_STAT_TOP].astype(np.int64)
    width = stats[:, cv2.CC_STAT_WIDTH].astype(np.int64)
    height = stats[:, cv2.CC_STAT_HEIGHT].astype(np.int64)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    # Row 0 describes the background; _label_bounding_boxes leaves it as an
    # empty box, and no caller reads it, so only the foreground rows matter.
    areas[0] = 0
    boxes: LabelBoxes = (areas, left, top, left + width - 1, top + height - 1)
    return labels, boxes


def _pad_uint8_border(mask: np.ndarray, border: int = 1) -> np.ndarray:
    """Zero-border pad for a boolean/uint8 component.

    Byte-for-byte identical to ``np.pad(mask.astype(np.uint8), border)`` but far
    cheaper on the per-object hot path: ``np.pad``'s generic ``_as_pairs`` setup
    dominates when it is called tens of thousands of times per segmentation.
    """

    source = mask.astype(np.uint8, copy=False)
    h, w = source.shape
    out = np.zeros((h + 2 * border, w + 2 * border), dtype=np.uint8)
    out[border : border + h, border : border + w] = source
    return out


def _internal_hole_stats_padded(padded: np.ndarray) -> tuple[int, int, int]:
    """Hole statistics for a component that already carries a zero border."""

    inverse = 1 - padded
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    if count <= 1:
        return 0, 0, 0
    border_label = int(labels[0, 0])
    hole_areas: list[int] = []
    for index in range(1, count):
        if index != border_label:
            hole_areas.append(int(stats[index, cv2.CC_STAT_AREA]))
    return sum(hole_areas), max(hole_areas, default=0), len(hole_areas)


def _thickness_from_distance(
    component: np.ndarray,
    padded_distance: np.ndarray,
    area: int,
) -> tuple[float, float, float]:
    """Thickness metrics from a component's zero-padded distance transform."""

    if area <= 0:
        return 0.0, 0.0, 0.0
    distance = padded_distance[1:-1, 1:-1]
    values = distance[component]
    # For a long strip, four times the mean interior distance approximates its
    # physical width. area / width² stays high for curved fibers, unlike a
    # covariance aspect ratio which collapses when the fiber bends.
    mean_thickness = max(1.0, 4.0 * _exact_mean(values, int(values.size)))
    max_thickness = max(1.0, 2.0 * float(_maximum_reduce(values, initial=0.0)))
    elongation = float(area / max(mean_thickness * mean_thickness, 1.0))
    return mean_thickness, max_thickness, elongation


def _shape_class(area: int, point_area: int, profiles: list[str]) -> str:
    if "ring" in profiles:
        return T("Ring")
    if "fiber" in profiles:
        return T("Elongated/filament")
    if area <= point_area:
        return T("Spot")
    if "round" in profiles:
        return T("Round/compact")
    if "ellipse" in profiles:
        return T("Elliptic/spindle")
    if "patch" in profiles:
        return T("Patch/irregular")
    return T("Mixed/mass")


def _label_bounding_boxes(labels: np.ndarray) -> tuple[np.ndarray, ...]:
    max_label = int(labels.max(initial=0))
    if max_label <= 0:
        empty = np.zeros(1, dtype=np.int64)
        return empty, empty, empty, empty, empty
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs].astype(np.int64, copy=False)
    areas = np.bincount(ids, minlength=max_label + 1).astype(np.int64, copy=False)
    x_min = np.full(max_label + 1, labels.shape[1], dtype=np.int64)
    y_min = np.full(max_label + 1, labels.shape[0], dtype=np.int64)
    x_max = np.full(max_label + 1, -1, dtype=np.int64)
    y_max = np.full(max_label + 1, -1, dtype=np.int64)
    # Four indexed reductions use far less memory than sorting every foreground
    # pixel.  On a dense 4K mask the former argsort path briefly allocated several
    # additional full-frame int64 arrays and could exhaust desktop RAM.
    np.minimum.at(x_min, ids, xs)
    np.minimum.at(y_min, ids, ys)
    np.maximum.at(x_max, ids, xs)
    np.maximum.at(y_max, ids, ys)
    return areas, x_min, y_min, x_max, y_max


# Shape-profile rule tables for the per-object hot path, one row of thresholds
# per strictness level.  Keeping them at module level avoids rebuilding five
# dict literals for every object.
_ROUND_RULES = {
    "loose": (0.35, 0.65, 3.0, 4.5, 0.20),
    "standard": (0.58, 0.82, 2.1, 3.5, 0.10),
    "strict": (0.72, 0.90, 1.6, 2.4, 0.05),
}
_ELLIPSE_RULES = {
    "loose": (1.15, 12.0, 0.08, 0.40, 12.0, 0.20),
    "standard": (1.35, 8.0, 0.15, 0.82, 7.0, 0.10),
    "strict": (1.65, 5.0, 0.25, 0.90, 6.0, 0.05),
}
_FIBER_RULES = {
    "loose": (0.85, 0.20, 5.0, 2.2, 0.80),
    "standard": (0.65, 0.10, 7.0, 3.0, 0.90),
    "strict": (0.45, 0.05, 10.0, 4.5, 0.96),
}
_PATCH_RULES = {
    "loose": (0.40, 12.0, 0.96, 0.65),
    "standard": (0.25, 7.0, 0.90, 0.45),
    "strict": (0.15, 5.0, 0.80, 0.32),
}
_RING_RULES = {
    "loose": (0.03, 12.0),
    "standard": (0.08, 8.0),
    "strict": (0.18, 5.0),
}


def _profile_matches(
    profile: str,
    level: str,
    outer_circularity: float,
    aspect_ratio: float,
    eccentricity: float,
    outer_solidity: float,
    max_hole_fraction: float,
    elongation_index: float,
) -> bool:
    """Whether an object's shape measurements fit one profile at one strictness."""

    if profile == "round":
        circularity, solidity, aspect, elongation, hole = _ROUND_RULES[level]
        return (
            outer_circularity >= circularity
            and outer_solidity >= solidity
            and aspect_ratio <= aspect
            and elongation_index <= elongation
            and max_hole_fraction <= hole
        )
    if profile == "ellipse":
        min_aspect, max_aspect, circularity, solidity, elongation, hole = (
            _ELLIPSE_RULES[level]
        )
        return (
            min_aspect <= aspect_ratio <= max_aspect
            and outer_circularity >= circularity
            and outer_solidity >= solidity
            and elongation_index <= elongation
            and max_hole_fraction <= hole
        )
    if profile == "fiber":
        circularity, hole, elongation, aspect, eccentricity_limit = _FIBER_RULES[level]
        return (
            outer_circularity <= circularity
            and max_hole_fraction <= hole
            and (
                elongation_index >= elongation
                or (aspect_ratio >= aspect and eccentricity >= eccentricity_limit)
            )
        )
    if profile == "patch":
        hole, elongation, solidity, circularity = _PATCH_RULES[level]
        return (
            max_hole_fraction <= hole
            and elongation_index <= elongation
            and (outer_solidity <= solidity or outer_circularity <= circularity)
        )
    if profile == "ring":
        hole, aspect = _RING_RULES[level]
        return max_hole_fraction >= hole and aspect_ratio <= aspect
    return True


_PROFILE_ORDER = ("round", "ellipse", "fiber", "patch", "ring")
_EYE_OVER_12 = np.eye(2, dtype=np.float64) / 12.0
_ZERO_COVARIANCE = np.zeros((2, 2), dtype=np.float64)
def _measure_label(
    label_id: int,
    labels: np.ndarray,
    raw_flat: np.ndarray,
    normalized_flat: np.ndarray,
    response_flat: np.ndarray,
    bbox: tuple[int, int, int, int],
    area: int,
    saturation_level: float,
    response_reference: float,
    point_area: int,
) -> tuple[dict[str, object], np.ndarray, tuple[float, float, float, float]]:
    """Measure one labelled object.

    ``raw_flat``/``normalized_flat``/``response_flat`` are the C-contiguous
    full-frame planes flattened once by the caller, so every object gathers its
    pixel values with a single indexed take instead of three boolean-mask
    extractions over a strided crop.  Every statistic is computed with the same
    floating-point operations, in the same order, as before.

    The pixel covariance (already regularised by the unit-square term) is
    returned separately so the caller can solve all 2×2 eigenproblems in one
    LAPACK call; ``_finish_record`` fills in the axis-derived fields.  The
    trailing tuple carries the *unrounded* shape terms that the profile
    classification must see (the record stores their rounded copies).
    """

    x0, y0, x1, y1 = bbox
    frame_height, frame_width = labels.shape
    component = labels[y0:y1, x0:x1] == label_id
    height = y1 - y0
    width = x1 - x0
    yy, xx = np.nonzero(component)
    count = int(xx.size)
    if count == 0:
        # Cannot happen for a label reported by _label_bounding_boxes; keep the
        # legacy behaviour of an all-empty record for robustness.
        component_index = np.zeros(0, dtype=np.intp)
    else:
        component_index = (yy + y0) * frame_width + (xx + x0)
    raw_values = raw_flat[component_index]
    normalized_values = normalized_flat[component_index]
    response_values = response_flat[component_index]

    contours, hierarchy = cv2.findContours(
        component.view(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    lengths = [cv2.arcLength(contour, True) for contour in contours]
    perimeter = float(sum(lengths))
    outer_perimeter = 0.0
    contour_area = 0.0
    outer_contour_area = 0.0
    external_contours: list[np.ndarray] = []
    if hierarchy is not None:
        parents = hierarchy[0]
        for contour_index, contour in enumerate(contours):
            value = float(abs(cv2.contourArea(contour)))
            if int(parents[contour_index][3]) < 0:
                contour_area += value
                outer_contour_area += value
                outer_perimeter += float(lengths[contour_index])
                external_contours.append(contour)
            else:
                contour_area -= value
    contour_area = max(contour_area, 0.0)
    circularity = (
        min(1.0, 4.0 * math.pi * contour_area / (perimeter * perimeter))
        if perimeter > 0
        else 0.0
    )
    outer_circularity = (
        min(
            1.0,
            4.0
            * math.pi
            * outer_contour_area
            / (outer_perimeter * outer_perimeter),
        )
        if outer_perimeter > 0
        else 0.0
    )
    if len(external_contours) == 1:
        all_points = external_contours[0]
    elif external_contours:
        all_points = np.concatenate(external_contours, axis=0)
    else:
        all_points = np.empty((0, 1, 2), dtype=np.int32)
    hull_area = (
        float(cv2.contourArea(cv2.convexHull(all_points)))
        if len(all_points) >= 3
        else contour_area
    )
    solidity = min(1.0, contour_area / hull_area) if hull_area > 0 else 1.0
    outer_solidity = (
        min(1.0, outer_contour_area / hull_area) if hull_area > 0 else 1.0
    )

    # Centroid: the mean of exactly representable integers is one correctly
    # rounded division of the exact integer sum, identical to the old float64
    # pairwise mean.
    if count:
        sum_x = int(_add_reduce(xx))
        sum_y = int(_add_reduce(yy))
        centroid_x = (sum_x + count * x0) / count
        centroid_y = (sum_y + count * y0) / count
    else:
        sum_x = sum_y = 0
        centroid_x = float(x0)
        centroid_y = float(y0)
    if count >= 2:
        # Step-for-step replica of ``np.cov(np.vstack((xx, yy)), bias=True)``
        # without its generic argument handling: centre the float64 rows by
        # their exact means, form the Gram matrix with the same BLAS call and
        # scale by the reciprocal count.
        coordinates = np.empty((2, count), dtype=np.float64)
        coordinates[0] = xx
        coordinates[1] = yy
        coordinates[0] -= sum_x / count
        coordinates[1] -= sum_y / count
        covariance = np.dot(coordinates, coordinates.T.conj())
        covariance *= np.true_divide(1, count)
        covariance += _EYE_OVER_12
    else:
        covariance = _ZERO_COVARIANCE + _EYE_OVER_12

    # One zero-padded copy serves both the hole search and the distance
    # transform (the former code padded the same component twice).
    padded = np.zeros((height + 2, width + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = component
    hole_count_total, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        1 - padded,
        connectivity=4,
    )
    if hole_count_total <= 1:
        hole_area = max_hole_area = hole_count = 0
    else:
        border_label = int(hole_labels[0, 0])
        hole_areas = hole_stats[1:, cv2.CC_STAT_AREA]
        hole_count = int(hole_count_total - 1)
        if border_label > 0:
            hole_areas = hole_areas.copy()
            hole_areas[border_label - 1] = 0
            hole_count -= 1
        if hole_count > 0:
            hole_area = int(_add_reduce(hole_areas))
            max_hole_area = int(_maximum_reduce(hole_areas))
        else:
            hole_area = max_hole_area = 0
    hole_fraction = float(hole_area / max(area + hole_area, 1))
    max_hole_fraction = float(max_hole_area / max(area + hole_area, 1))

    if count > 0:
        padded_distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        distance_values = padded_distance.ravel()[
            (yy + 1) * (width + 2) + (xx + 1)
        ]
        # For a long strip, four times the mean interior distance approximates
        # its physical width. area / width² stays high for curved fibers, unlike
        # a covariance aspect ratio which collapses when the fiber bends.
        mean_thickness = max(1.0, 4.0 * _exact_mean(distance_values, count))
        max_thickness = max(
            1.0,
            2.0 * float(_maximum_reduce(distance_values, initial=0.0)),
        )
        elongation_index = float(
            count / max(mean_thickness * mean_thickness, 1.0)
        )
        response_mean = _exact_mean(response_values, count)
        raw_mean = _exact_mean(raw_values, count)
        normalized_mean = _exact_mean(normalized_values, count)
        saturated_fraction = (
            int(np.count_nonzero(raw_values >= saturation_level)) / count
        )
    else:
        mean_thickness = max_thickness = elongation_index = 0.0
        response_mean = float(response_values.mean())
        raw_mean = float(raw_values.mean())
        normalized_mean = float(normalized_values.mean())
        saturated_fraction = 0.0
    response_confidence = (
        response_mean / response_reference
        if math.isfinite(response_reference) and response_reference > 1e-8
        else 0.0
    )
    touches_border = bool(
        x0 == 0 or y0 == 0 or x1 >= frame_width or y1 >= frame_height
    )
    record: dict[str, object] = {
        "id": label_id,
        "area_px": int(area),
        "perimeter_px": round(perimeter, 4),
        "outer_perimeter_px": round(outer_perimeter, 4),
        "circularity": round(float(circularity), 6),
        "outer_circularity": round(float(outer_circularity), 6),
        "major_axis_px": 0.0,
        "minor_axis_px": 0.0,
        "aspect_ratio": 0.0,
        "eccentricity": 0.0,
        "solidity": round(float(solidity), 6),
        "outer_solidity": round(float(outer_solidity), 6),
        "hole_fraction": round(hole_fraction, 6),
        "max_hole_fraction": round(max_hole_fraction, 6),
        "hole_count": hole_count,
        "mean_thickness_px": round(mean_thickness, 4),
        "max_thickness_px": round(max_thickness, 4),
        "elongation_index": round(elongation_index, 6),
        "shape_class": "",
        "shape_tags": "",
        "centroid_x_px": round(centroid_x, 4),
        "centroid_y_px": round(centroid_y, 4),
        "bbox_x_px": int(x0),
        "bbox_y_px": int(y0),
        "bbox_width_px": int(x1 - x0),
        "bbox_height_px": int(y1 - y0),
        "touches_border": touches_border,
        "raw_mean": round(raw_mean, 6),
        "raw_max": round(float(_maximum_reduce(raw_values, initial=0.0)), 6),
        "raw_integrated": round(
            float(_add_reduce(raw_values, dtype=np.float64)),
            6,
        ),
        "normalized_mean": round(normalized_mean, 8),
        "normalized_max": round(
            float(_maximum_reduce(normalized_values, initial=0.0)),
            8,
        ),
        "response_mean": round(response_mean, 8),
        "response_max": round(
            float(_maximum_reduce(response_values, initial=0.0)),
            8,
        ),
        "response_confidence": round(float(response_confidence), 8),
        "saturated_fraction": round(float(saturated_fraction), 8),
    }
    return (
        record,
        covariance,
        (outer_circularity, outer_solidity, max_hole_fraction, elongation_index),
    )


def _finish_record(
    record: dict[str, object],
    eigenvalues: np.ndarray,
    *,
    unrounded_outer_circularity: float,
    unrounded_outer_solidity: float,
    unrounded_max_hole_fraction: float,
    unrounded_elongation_index: float,
    point_area: int,
) -> tuple[float, float]:
    """Fill the eigen-derived fields and shape classes; return (aspect, ecc)."""

    major_axis = float(4.0 * math.sqrt(eigenvalues[0]))
    minor_axis = float(4.0 * math.sqrt(eigenvalues[1]))
    aspect_ratio = major_axis / max(minor_axis, 1e-6)
    eccentricity = math.sqrt(
        max(0.0, 1.0 - eigenvalues[1] / max(eigenvalues[0], 1e-12))
    )
    compatible_profiles = [
        profile
        for profile in _PROFILE_ORDER
        if _profile_matches(
            profile,
            "standard",
            unrounded_outer_circularity,
            aspect_ratio,
            eccentricity,
            unrounded_outer_solidity,
            unrounded_max_hole_fraction,
            unrounded_elongation_index,
        )
    ]
    area = int(record["area_px"])
    shape_tags = [
        OBJECT_SHAPE_PROFILE_LABELS[profile] for profile in compatible_profiles
    ]
    if area <= point_area:
        shape_tags.insert(0, T("Spot (by image scale)"))
    record["major_axis_px"] = round(major_axis, 4)
    record["minor_axis_px"] = round(minor_axis, 4)
    record["aspect_ratio"] = round(float(aspect_ratio), 6)
    record["eccentricity"] = round(float(eccentricity), 6)
    record["shape_class"] = _shape_class(area, point_area, compatible_profiles)
    record["shape_tags"] = T("; ").join(shape_tags) if shape_tags else T("Mixed/unclassified")
    return aspect_ratio, eccentricity


def _reusable_record(
    label_id: int,
    labels: np.ndarray,
    previous_labels: np.ndarray,
    previous_by_id: Mapping[int, Mapping[str, object]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    area: int,
) -> dict[str, object] | None:
    """Return a copy of the previous record whose pixel set equals this label's.

    The candidate is the previous label under this component's first pixel;
    equal bounding box and area are checked first (cheap) and then the two
    pixel sets are compared inside the box, so a coincidental match can never
    slip through.  ``None`` means the object must be measured afresh.
    """

    first_row = labels[y0, x0:x1]
    first_columns = np.flatnonzero(first_row == label_id)
    if first_columns.size == 0:  # pragma: no cover - bbox rows always contain the label
        return None
    previous_id = int(previous_labels[y0, x0 + int(first_columns[0])])
    if previous_id <= 0:
        return None
    previous = previous_by_id.get(previous_id)
    if previous is None:
        return None
    try:
        if (
            int(previous["bbox_x_px"]) != x0
            or int(previous["bbox_y_px"]) != y0
            or int(previous["bbox_width_px"]) != x1 - x0
            or int(previous["bbox_height_px"]) != y1 - y0
            or int(previous["area_px"]) != area
        ):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    current_component = labels[y0:y1, x0:x1] == label_id
    previous_component = previous_labels[y0:y1, x0:x1] == previous_id
    if not np.array_equal(current_component, previous_component):
        return None
    record = dict(previous)
    record["id"] = label_id
    return record


def _flat_plane(plane: np.ndarray) -> np.ndarray:
    """Return a 1-D view (or, for a strided input, a copy) of a 2-D plane."""

    return np.ascontiguousarray(plane).reshape(-1)


def measure_and_filter_labels(
    labels: np.ndarray,
    raw_signal: np.ndarray,
    normalized_signal: np.ndarray,
    response: np.ndarray,
    params: UniversalParams,
    *,
    apply_filters: bool = True,
    response_reference: float | None = None,
    reusable: tuple[np.ndarray, Sequence[Mapping[str, object]]] | None = None,
    bounding_boxes: LabelBoxes | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Measure every labelled object and apply the recipe's object filters.

    ``reusable`` may carry ``(previous_labels, previous_records)`` measured on
    the *same* raw/normalized/response planes with the same reference.  Any
    current component whose pixel set is identical to a previous object's is
    then given a copy of that object's record instead of being re-measured.
    Every record is a pure function of the pixel set and those planes, so the
    copy is exactly what a fresh measurement would produce; this is what makes
    a brush stroke on a large image cost only the objects it touched.

    ``bounding_boxes`` are the per-label ``(areas, x_min, y_min, x_max, y_max)``
    arrays when the caller already has them from the labelling step.
    """

    labels = labels.astype(np.int32, copy=False)
    if bounding_boxes is not None and len(bounding_boxes[0]) > 0:
        max_label = int(len(bounding_boxes[0]) - 1)
    else:
        max_label = int(labels.max(initial=0))
    if max_label <= 0:
        return (
            np.zeros(labels.shape, dtype=bool),
            np.zeros(labels.shape, dtype=np.int32),
            [],
        )
    if bounding_boxes is not None:
        areas, x_min, y_min, x_max, y_max = bounding_boxes
    else:
        areas, x_min, y_min, x_max, y_max = _label_bounding_boxes(labels)
    previous_labels: np.ndarray | None = None
    previous_by_id: dict[int, Mapping[str, object]] = {}
    if reusable is not None:
        candidate_labels, candidate_records = reusable
        candidate_labels = np.asarray(candidate_labels)
        if candidate_labels.shape == labels.shape:
            previous_labels = candidate_labels
            for record in candidate_records:
                try:
                    previous_by_id[int(record["id"])] = record
                except (KeyError, TypeError, ValueError):
                    continue
        if not previous_by_id:
            previous_labels = None
    raw_max_observed = float(raw_signal.max(initial=0.0))
    # Reuse the Otsu response threshold the caller already computed when it is the
    # same value (otsu / hysteresis_otsu paths); otherwise compute it here exactly
    # as before.  Passing the caller's value is bit-identical: it is
    # _global_threshold(response, "otsu") on the same response array.
    if response_reference is None:
        response_reference = _global_threshold(response, "otsu")
    else:
        response_reference = float(response_reference)
    saturation_level = next(
        (
            float(level)
            for level in (255, 1023, 4095, 16383, 65535, 2**32 - 1)
            if raw_max_observed <= level
        ),
        raw_max_observed,
    )
    point_area = max(
        1,
        int(round(24.0 * (min(labels.shape[:2]) / 1024.0) ** 2)),
    )
    raw_flat = _flat_plane(raw_signal)
    normalized_flat = _flat_plane(normalized_signal)
    response_flat = _flat_plane(response)

    # Area is the only keep-term available before measurement, and it decides
    # the fate of every component regardless of the other filters.  Skip the
    # expensive _measure_label (contours, convex hull, distance transform) for
    # components the area gate already rejects.  This is the exact negation of
    # the area terms in the keep test below, so the retained set, records, and
    # renumbered ids stay identical.  The rebuild path (apply_filters=False)
    # still measures every component.
    label_areas = areas[1 : max_label + 1]
    candidate = label_areas > 0
    if apply_filters:
        candidate &= label_areas >= params.min_area
        if params.max_area > 0:
            candidate &= label_areas <= params.max_area
    candidate_ids = (np.flatnonzero(candidate) + 1).tolist()

    x_min_list = x_min.tolist()
    y_min_list = y_min.tolist()
    x_max_list = x_max.tolist()
    y_max_list = y_max.tolist()
    areas_list = areas.tolist()
    measured: list[dict[str, object]] = []
    shape_terms: list[tuple[float, float, float, float] | None] = []
    covariances: list[np.ndarray] = []
    for label_id in candidate_ids:
        area = areas_list[label_id]
        x0 = x_min_list[label_id]
        y0 = y_min_list[label_id]
        x1 = x_max_list[label_id] + 1
        y1 = y_max_list[label_id] + 1
        if previous_labels is not None:
            reused = _reusable_record(
                label_id,
                labels,
                previous_labels,
                previous_by_id,
                x0,
                y0,
                x1,
                y1,
                area,
            )
            if reused is not None:
                measured.append(reused)
                shape_terms.append(None)
                continue
        record, covariance, terms = _measure_label(
            label_id,
            labels,
            raw_flat,
            normalized_flat,
            response_flat,
            (x0, y0, x1, y1),
            area,
            saturation_level,
            response_reference,
            point_area,
        )
        measured.append(record)
        shape_terms.append(terms)
        covariances.append(covariance)

    # All symmetric 2×2 eigenproblems in one LAPACK call.  numpy applies the
    # same routine to every matrix of the stack, so each object's eigenvalues
    # are identical to a per-object ``eigvalsh`` call.
    if covariances:
        eigen_stack = np.linalg.eigvalsh(np.stack(covariances))
        eigen_stack = np.sort(np.maximum(eigen_stack, 0.0), axis=1)[:, ::-1]
    else:
        eigen_stack = np.zeros((0, 2), dtype=np.float64)
    eigen_slot = 0

    records: list[dict[str, object]] = []
    keep_ids: list[int] = []
    profile = params.shape_profile
    strictness = (
        params.shape_profile_strictness
        if params.shape_profile_strictness in OBJECT_SHAPE_STRICTNESS_LABELS
        else "standard"
    )
    min_thickness = params.min_thickness
    max_thickness = params.max_thickness
    min_circularity = params.min_circularity
    min_solidity = params.min_solidity
    max_eccentricity = params.max_eccentricity
    max_aspect_ratio = params.max_aspect_ratio
    min_mean_intensity = params.min_mean_intensity
    min_response_confidence = params.min_response_confidence
    remove_border = params.remove_border
    for index, record in enumerate(measured):
        terms = shape_terms[index]
        if terms is not None:
            eigenvalues = eigen_stack[eigen_slot]
            eigen_slot += 1
            _finish_record(
                record,
                eigenvalues,
                unrounded_outer_circularity=terms[0],
                unrounded_outer_solidity=terms[1],
                unrounded_max_hole_fraction=terms[2],
                unrounded_elongation_index=terms[3],
                point_area=point_area,
            )
        keep = True
        if apply_filters:
            profile_match = profile == "any" or _profile_matches(
                profile,
                strictness,
                float(record["outer_circularity"]),
                float(record["aspect_ratio"]),
                float(record["eccentricity"]),
                float(record["outer_solidity"]),
                float(record["max_hole_fraction"]),
                float(record["elongation_index"]),
            )
            thickness = float(record["mean_thickness_px"])
            keep = (
                profile_match
                and thickness >= min_thickness
                and (max_thickness <= 0 or thickness <= max_thickness)
                and float(record["circularity"]) >= min_circularity
                and float(record["solidity"]) >= min_solidity
                and float(record["eccentricity"]) <= max_eccentricity
                and float(record["aspect_ratio"]) <= max_aspect_ratio
                and float(record["normalized_mean"]) >= min_mean_intensity
                and float(record["response_confidence"]) >= min_response_confidence
                and (not remove_border or not bool(record["touches_border"]))
            )
        if keep:
            keep_ids.append(int(record["id"]))
            records.append(record)

    if len(keep_ids) == max_label:
        # Every label 1..max_label survived, so the renumbering is the
        # identity and the label plane can be returned as it is.
        final_labels = labels
    else:
        mapping = np.zeros(max_label + 1, dtype=np.int32)
        if keep_ids:
            mapping[np.asarray(keep_ids, dtype=np.int64)] = np.arange(
                1,
                len(keep_ids) + 1,
                dtype=np.int32,
            )
        final_labels = mapping[labels]
    final_mask = final_labels > 0
    for new_id, record in enumerate(records, start=1):
        record["id"] = new_id
    return final_mask, final_labels, records


def _rescale_preview_objects(
    objects: list[dict[str, object]],
    scale: float,
) -> list[dict[str, object]]:
    if abs(float(scale) - 1.0) <= 0.001:
        return objects
    scale = max(float(scale), 1e-6)
    linear_keys = (
        "perimeter_px",
        "outer_perimeter_px",
        "major_axis_px",
        "minor_axis_px",
        "mean_thickness_px",
        "max_thickness_px",
        "centroid_x_px",
        "centroid_y_px",
        "bbox_x_px",
        "bbox_y_px",
        "bbox_width_px",
        "bbox_height_px",
    )
    area_factor = 1.0 / (scale * scale)
    linear_factor = 1.0 / scale
    output: list[dict[str, object]] = []
    for record in objects:
        converted = dict(record)
        converted["area_px"] = int(round(float(record["area_px"]) * area_factor))
        converted["raw_integrated"] = round(
            float(record["raw_integrated"]) * area_factor,
            6,
        )
        for key in linear_keys:
            value = float(record[key]) * linear_factor
            converted[key] = int(round(value)) if key.startswith("bbox_") else round(value, 4)
        output.append(converted)
    return output


def _segment_prepared_signal(
    signal: np.ndarray,
    normalized: np.ndarray,
    params: UniversalParams,
    *,
    mode_used: str,
    black_value: float,
    white_value: float,
    started: float,
    analysis_scale: float = 1.0,
    source_shape: tuple[int, int] | None = None,
    source_params: UniversalParams | None = None,
    diagnostics: tuple[float, float, float, int, tuple[str, ...]] | None = None,
    noise_reference_signal: np.ndarray | None = None,
) -> UniversalResult:
    """Run the common response/threshold/object half on a prepared signal."""

    shape = tuple(int(v) for v in normalized.shape[:2])
    params = enforce_shape_compatibility(params.validated(shape))
    robust_snr = params.intensity_space == "robust_snr"
    if robust_snr:
        # Decision evidence comes directly from the affine-invariant raw-signal
        # residual.  Do not rank it, stretch its upper tail, or run Otsu.
        response = _robust_snr_response(
            signal,
            params,
            noise_reference_signal=noise_reference_signal,
        )
    else:
        denoised = _denoise(
            normalized,
            params.denoise_method,
            params.denoise_strength,
        )
        response_raw = _background_response(
            denoised,
            params.background_method,
            params.background_radius,
            params.background_strength,
        )
        response = robust_normalize(response_raw)
    response = _apply_background_ceiling(response, signal, params)
    mask, threshold_value, low_threshold, response_reference = _threshold_mask(
        response,
        params,
    )
    mask = _apply_morphology(mask, params)
    if robust_snr:
        mask = _filter_snr_evidence_components(
            mask,
            response,
            threshold_value,
            low_threshold,
        )
    if params.remove_border:
        mask = _remove_border_components(mask)
    labels, label_boxes = _component_labels_with_boxes(mask, params)
    final_mask, final_labels, objects = measure_and_filter_labels(
        labels,
        signal,
        normalized,
        response,
        params,
        apply_filters=True,
        response_reference=response_reference,
        bounding_boxes=label_boxes,
    )
    # The legacy preview path only downsamples, while a relative reference grid
    # may legitimately upscale a small screenshot.  Preserve the real mapping
    # so object measurements can always be converted back to source pixels.
    scale = float(np.clip(analysis_scale, 1e-6, 1e6))
    objects = _rescale_preview_objects(objects, scale)
    effective_params = (
        enforce_shape_compatibility(source_params.validated(source_shape))
        if source_params
        else params
    )
    diagnostic_values = diagnostics or (0.0, 0.0, 0.0, 0, ())
    return UniversalResult(
        mask=final_mask,
        labels=final_labels,
        objects=objects,
        signal=signal,
        normalized_signal=normalized,
        response=response,
        threshold_value=float(threshold_value),
        low_threshold=float(low_threshold),
        positive_fraction=float(final_mask.mean()),
        params=effective_params,
        signal_mode_used=mode_used,
        input_black_value=float(black_value),
        input_white_value=float(white_value),
        processing_seconds=float(time.perf_counter() - started),
        analysis_scale=scale,
        source_shape=source_shape or shape,
        manually_edited=False,
        input_saturation_fraction=float(diagnostic_values[0]),
        input_high_plateau_fraction=float(diagnostic_values[1]),
        input_largest_tie_fraction=float(diagnostic_values[2]),
        input_unique_levels=int(diagnostic_values[3]),
        intensity_warnings=tuple(diagnostic_values[4]),
    )


def segment_image(
    image: np.ndarray,
    params: UniversalParams,
    *,
    analysis_scale: float = 1.0,
    source_shape: tuple[int, int] | None = None,
    source_params: UniversalParams | None = None,
) -> UniversalResult:
    if params.recognition_engine != "classic":
        raise ValueError(T(
            "Learned mode needs the path of the original image; run it through the image loading "
            "entry point.",
        ))
    started = time.perf_counter()
    shape = tuple(int(v) for v in image.shape[:2])
    params = enforce_shape_compatibility(params.validated(shape))
    signal, mode_used = extract_signal(image, params)
    normalized, black_value, white_value = normalize_signal(signal, params)
    diagnostics = _input_intensity_diagnostics(signal, image.dtype)
    return _segment_prepared_signal(
        signal,
        normalized,
        params,
        mode_used=mode_used,
        black_value=black_value,
        white_value=white_value,
        started=started,
        analysis_scale=analysis_scale,
        source_shape=source_shape,
        source_params=source_params,
        diagnostics=diagnostics,
    )


def _relative_grid_plan(
    source_shape: tuple[int, int],
    reference_short_side: int,
    max_side: int | None,
) -> tuple[tuple[int, int], float, float]:
    """Return analysis shape, source->analysis scale and reference->grid scale."""

    source_height, source_width = (max(1, int(value)) for value in source_shape[:2])
    reference = max(16, int(reference_short_side))
    source_short_side = min(source_height, source_width)
    scale = float(reference) / float(source_short_side)

    # ``max_side`` is a preview/performance ceiling, not a second semantic
    # scale.  It only reduces the analysis grid and the canonical parameters are
    # materialized to that grid once below.
    if max_side is not None:
        scale = min(scale, float(max(1, int(max_side))) / max(source_shape))

    # A malicious panorama plus an extreme reference size must not turn the
    # fixed-short-side grid into an allocation larger than the decoder limit.
    projected_pixels = float(source_height) * float(source_width) * scale * scale
    if projected_pixels > MAX_DECODED_IMAGE_PIXELS:
        scale *= math.sqrt(MAX_DECODED_IMAGE_PIXELS / projected_pixels)

    analysis_width = max(1, int(round(source_width * scale)))
    analysis_height = max(1, int(round(source_height * scale)))
    analysis_shape = (analysis_height, analysis_width)
    source_to_analysis = math.sqrt(
        (analysis_height / source_height) * (analysis_width / source_width)
    )
    reference_to_analysis = min(analysis_shape) / float(reference)
    return analysis_shape, source_to_analysis, reference_to_analysis


def _resize_float_map(
    image: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    if image.shape[:2] == target_shape:
        return image.astype(np.float32, copy=False)
    shrinking = target_shape[0] < image.shape[0] or target_shape[1] < image.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(
        image.astype(np.float32, copy=False),
        (target_shape[1], target_shape[0]),
        interpolation=interpolation,
    ).astype(np.float32, copy=False)


def _restore_relative_result(
    loaded: LoadedImage,
    analysis_result: UniversalResult,
    recipe_params: UniversalParams,
    started: float,
    *,
    source_signal: np.ndarray | None = None,
) -> UniversalResult:
    """Restore a canonical-grid result to a self-consistent source-size result."""

    source_shape = loaded.shape
    if analysis_result.mask.shape == source_shape:
        analysis_result.params = recipe_params
        analysis_result.analysis_scale = 1.0
        analysis_result.source_shape = source_shape
        analysis_result.processing_seconds = float(time.perf_counter() - started)
        return analysis_result

    mapped_labels = resized_labels(analysis_result.labels, source_shape)
    mapped_mask = resized_mask(analysis_result.mask, source_shape)
    # Nearest-neighbour label restoration should match the restored mask.  The
    # explicit intersection also protects against a corrupt/non-contiguous label
    # image without inventing foreground pixels.  Multiplying by the 0/1 mask
    # zeroes exactly the same pixels as the former masked assignment.
    if mapped_labels is analysis_result.labels:
        mapped_labels = mapped_labels.copy()
    np.multiply(mapped_labels, mapped_mask, out=mapped_labels, casting="unsafe")
    mapped_mask = mapped_labels > 0
    normalized = _resize_float_map(analysis_result.normalized_signal, source_shape)
    response = _resize_float_map(analysis_result.response, source_shape)

    # Object raw-intensity statistics should use source pixels, while normalized
    # and response statistics stay tied to the exact canonical decision field.
    signal_params = recipe_params
    if analysis_result.signal_mode_used in SIGNAL_MODE_LABELS:
        signal_params = replace(
            recipe_params,
            signal_mode=analysis_result.signal_mode_used,
        )
    if source_signal is not None and source_signal.shape[:2] == source_shape:
        # The caller already extracted exactly this plane (same image, same
        # resolved signal mode); reuse it instead of a second full-frame pass.
        signal = source_signal
    else:
        signal, _source_mode = extract_signal(loaded.image, signal_params)
    # For robust-SNR recipes the response is in noise sigmas, so the confidence
    # reference must be the recipe's fixed seed, not a unit-interval Otsu of the
    # resized response (which would clip and make export disagree with preview).
    restore_reference = (
        _snr_seed_high(recipe_params)
        if recipe_params.threshold_method == "snr_hysteresis"
        else None
    )
    final_mask, final_labels, objects = measure_and_filter_labels(
        mapped_labels,
        signal,
        normalized,
        response,
        recipe_params,
        apply_filters=False,
        response_reference=restore_reference,
    )
    return UniversalResult(
        mask=final_mask,
        labels=final_labels,
        objects=objects,
        signal=signal,
        normalized_signal=normalized,
        response=response,
        threshold_value=analysis_result.threshold_value,
        low_threshold=analysis_result.low_threshold,
        positive_fraction=float(final_mask.mean()),
        params=recipe_params,
        signal_mode_used=analysis_result.signal_mode_used,
        input_black_value=analysis_result.input_black_value,
        input_white_value=analysis_result.input_white_value,
        processing_seconds=float(time.perf_counter() - started),
        analysis_scale=1.0,
        source_shape=source_shape,
        manually_edited=False,
        input_saturation_fraction=analysis_result.input_saturation_fraction,
        input_high_plateau_fraction=analysis_result.input_high_plateau_fraction,
        input_largest_tie_fraction=analysis_result.input_largest_tie_fraction,
        input_unique_levels=analysis_result.input_unique_levels,
        intensity_warnings=analysis_result.intensity_warnings,
    )


def _segment_loaded_relative_grid(
    loaded: LoadedImage,
    params: UniversalParams,
    *,
    max_side: int | None,
) -> UniversalResult:
    started = time.perf_counter()
    source_shape = loaded.shape
    recipe_params = enforce_shape_compatibility(params.validated())
    analysis_shape, source_to_analysis, reference_to_analysis = _relative_grid_plan(
        source_shape,
        recipe_params.reference_short_side,
        max_side,
    )
    analysis_params = recipe_params.scaled_spatial(
        reference_to_analysis
    ).validated(analysis_shape)

    if recipe_params.intensity_space == "robust_snr":
        # Preserve raw, linearly transformed effect size on the canonical grid.
        # Ranking before or after this resize would manufacture a bright tail
        # even for an otherwise negative field, so this branch never creates an
        # ordinal plane.
        source_signal, mode_used = extract_signal(loaded.image, recipe_params)
        source_finite = _finite_sample(source_signal)
        black_value, white_value = (
            float(value)
            for value in np.percentile(
                source_finite,
                (recipe_params.black_percentile, recipe_params.white_percentile),
            )
        )
        analysis_signal = _resize_float_map(source_signal, analysis_shape)
        normalized, _analysis_black, _analysis_white = normalize_signal(
            analysis_signal,
            analysis_params,
        )
        result = _segment_prepared_signal(
            analysis_signal,
            normalized,
            analysis_params,
            mode_used=mode_used,
            black_value=black_value,
            white_value=white_value,
            started=started,
            analysis_scale=source_to_analysis,
            source_shape=source_shape,
            noise_reference_signal=(
                source_signal if source_to_analysis > 1.0 else None
            ),
            diagnostics=_input_intensity_diagnostics(
                source_signal,
                loaded.image.dtype,
            ),
        )
        result.params = recipe_params
        result.processing_seconds = float(time.perf_counter() - started)
        if max_side is not None:
            return result
        return _restore_relative_result(
            loaded,
            result,
            recipe_params,
            started,
            source_signal=source_signal,
        )

    if recipe_params.intensity_space == "relative_rank":
        # Canonical relative chain (order is intentional): rank the native
        # signal first so a monotonic display transform is removed before any
        # interpolation; resize that ordinal plane; then mid-rank again to
        # absorb interpolation/tie-distribution changes and apply the tail
        # occupancy prior.  A single rank after resize is measurably less Gamma
        # invariant because averaging and a nonlinear point transform do not
        # commute.
        source_signal, mode_used = extract_signal(loaded.image, recipe_params)
        source_finite = _finite_sample(source_signal)
        black_value, white_value = (
            float(value)
            for value in np.percentile(
                source_finite,
                (recipe_params.black_percentile, recipe_params.white_percentile),
            )
        )
        native_ordinal, native_lut = _empirical_midrank(
            source_signal,
            invert=recipe_params.invert,
            tail_power=1.0,
            finite=source_finite,
            return_lut=True,
        )
        if analysis_shape == source_shape:
            analysis_ordinal = native_ordinal
            analysis_signal = source_signal.astype(np.float32, copy=False)
        else:
            native_lut = None
            interpolation = (
                cv2.INTER_AREA if source_to_analysis < 1.0 else cv2.INTER_CUBIC
            )
            target_size = (analysis_shape[1], analysis_shape[0])
            analysis_ordinal = cv2.resize(
                native_ordinal,
                target_size,
                interpolation=interpolation,
            ).astype(np.float32, copy=False)
            analysis_signal = cv2.resize(
                source_signal.astype(np.float32, copy=False),
                target_size,
                interpolation=interpolation,
            ).astype(np.float32, copy=False)
        normalized = _empirical_midrank(
            analysis_ordinal,
            invert=False,
            tail_power=recipe_params.rank_tail_power,
            lut=native_lut,
        )
        result = _segment_prepared_signal(
            analysis_signal,
            normalized,
            analysis_params,
            mode_used=mode_used,
            black_value=black_value,
            white_value=white_value,
            started=started,
            analysis_scale=source_to_analysis,
            source_shape=source_shape,
            diagnostics=_input_intensity_diagnostics(
                source_signal,
                loaded.image.dtype,
            ),
        )
        result.params = recipe_params
        result.processing_seconds = float(time.perf_counter() - started)
        if max_side is not None:
            return result
        return _restore_relative_result(
            loaded,
            result,
            recipe_params,
            started,
            source_signal=source_signal,
        )

    if analysis_shape == source_shape:
        analysis_image = loaded.image
    else:
        interpolation = (
            cv2.INTER_AREA if source_to_analysis < 1.0 else cv2.INTER_CUBIC
        )
        analysis_image = cv2.resize(
            loaded.image,
            (analysis_shape[1], analysis_shape[0]),
            interpolation=interpolation,
        )

    result = segment_image(
        analysis_image,
        analysis_params,
        analysis_scale=source_to_analysis,
        source_shape=source_shape,
    )
    result.params = recipe_params
    result.processing_seconds = float(time.perf_counter() - started)
    if max_side is not None:
        return result
    return _restore_relative_result(loaded, result, recipe_params, started)


def segment_loaded_image(
    loaded: LoadedImage,
    params: UniversalParams,
    *,
    max_side: int | None = None,
    reference_loaded: LoadedImage | None = None,
) -> UniversalResult:
    params.validated(loaded.shape)
    if params.recognition_engine in AI_RECOGNITION_ENGINES:
        # The adapter preserves the calibrated source-pixel scale even for a
        # GUI preview; legacy max_side never resizes a learned-model input.
        module = ai_engine_module(params.recognition_engine)
        return module.segment_loaded(loaded, params, reference_loaded=reference_loaded)
    image = loaded.image
    source_shape = loaded.shape
    if params.validated().scale_mode == "relative_grid":
        return _segment_loaded_relative_grid(
            loaded,
            params,
            max_side=max_side,
        )
    if max_side is None or max(source_shape) <= max_side:
        return segment_image(
            image,
            params.validated(source_shape),
            analysis_scale=1.0,
            source_shape=source_shape,
        )
    scale = float(max_side) / float(max(source_shape))
    width = max(1, int(round(source_shape[1] * scale)))
    height = max(1, int(round(source_shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    scaled_params = params.validated(source_shape).scaled_for_preview(scale).validated(
        (height, width)
    )
    return segment_image(
        resized,
        scaled_params,
        analysis_scale=scale,
        source_shape=source_shape,
        source_params=params.validated(source_shape),
    )


_MANUAL_CONTRIBUTOR_ID = "manual"


def _manual_rebuild_object_provenance(
    objects: list[dict[str, object]],
    final_labels: np.ndarray,
    previous: UniversalResult | None,
) -> list[dict[str, object]]:
    """Carry branch attribution through a full-resolution manual mask edit.

    Pixels retained from exactly one previous object keep that object's branch
    contributors. New pixels, merges, or objects whose previous source cannot
    be established receive the explicit ``manual`` contributor marker.
    """

    if (
        previous is None
        or previous.is_preview
        or previous.mask.shape != final_labels.shape
        or previous.labels.shape != final_labels.shape
    ):
        return [
            {
                **record,
                "contributing_branch_ids": [_MANUAL_CONTRIBUTOR_ID],
                "manual_provenance": "manual_unattributed",
            }
            for record in objects
        ]

    fallback_contributors: list[str] = []
    quick_plan = getattr(previous, "quick_plan", None)
    plan_branches = getattr(quick_plan, "branches", ())
    if len(plan_branches) == 1:
        branch_id = str(getattr(plan_branches[0], "branch_id", "")).strip()
        if branch_id:
            fallback_contributors.append(branch_id)
    if not fallback_contributors:
        compiled_plan = getattr(previous, "compiled_plan", None)
        if isinstance(compiled_plan, Mapping):
            branches = compiled_plan.get("branches", [])
            if isinstance(branches, Sequence) and len(branches) == 1:
                branch = branches[0]
                if isinstance(branch, Mapping):
                    branch_id = str(branch.get("branch_id", "")).strip()
                    if branch_id:
                        fallback_contributors.append(branch_id)

    contributors_by_old_id: dict[int, list[str]] = {}
    for record in previous.objects:
        try:
            old_id = int(record.get("id", 0))
        except (TypeError, ValueError):
            continue
        raw_contributors = record.get("contributing_branch_ids", [])
        if isinstance(raw_contributors, str):
            values = [
                item.strip()
                for item in raw_contributors.replace(",", ";").split(";")
                if item.strip()
            ]
        elif isinstance(raw_contributors, Sequence):
            values = [str(item).strip() for item in raw_contributors if str(item).strip()]
        else:
            values = []
        contributors_by_old_id[old_id] = list(dict.fromkeys(values))

    previous_mask = np.asarray(previous.mask, dtype=bool)
    previous_labels = np.asarray(previous.labels, dtype=np.int32)
    new_labels = np.asarray(final_labels, dtype=np.int32)

    # Build the new-object -> old-object overlap table in bounded tiles.
    # Comparing the entire label image separately for every object makes one
    # brush stroke O(object_count * pixel_count), while packing the whole image
    # at once can allocate several gigabytes on a dense 80 MP PSB.  One-million
    # pixel tiles keep temporary arrays small; the Python sets grow only with
    # actual object-pair/new-object counts.
    target_tile_pixels = 1_000_000
    height, width = new_labels.shape
    tile_width = max(1, min(width, target_tile_pixels))
    tile_height = max(1, min(height, target_tile_pixels // tile_width))
    stride = int(previous_labels.max(initial=0)) + 1
    packed_pairs: set[int] = set()
    new_pixel_ids: set[int] = set()
    for y0 in range(0, height, tile_height):
        y1 = min(height, y0 + tile_height)
        for x0 in range(0, width, tile_width):
            x1 = min(width, x0 + tile_width)
            new_tile = new_labels[y0:y1, x0:x1]
            old_tile = previous_labels[y0:y1, x0:x1]
            previous_mask_tile = previous_mask[y0:y1, x0:x1]

            overlap = (new_tile > 0) & (old_tile > 0)
            if np.any(overlap):
                packed = new_tile[overlap].astype(np.uint64, copy=False)
                packed *= np.uint64(stride)
                packed += old_tile[overlap].astype(np.uint64, copy=False)
                packed_pairs.update(int(value) for value in np.unique(packed).tolist())

            new_pixels = (new_tile > 0) & (~previous_mask_tile)
            if np.any(new_pixels):
                new_pixel_ids.update(
                    int(value)
                    for value in np.unique(new_tile[new_pixels]).tolist()
                    if int(value) > 0
                )

    old_ids_by_new_id: dict[int, list[int]] = {}
    for pair in sorted(packed_pairs):
        new_id = int(pair // stride)
        old_id = int(pair % stride)
        old_ids_by_new_id.setdefault(new_id, []).append(old_id)
    attributed: list[dict[str, object]] = []
    for record in objects:
        try:
            new_id = int(record.get("id", 0))
        except (TypeError, ValueError):
            new_id = 0
        old_ids = old_ids_by_new_id.get(new_id, [])
        contributors: list[str] = []
        for old_id in old_ids:
            values = contributors_by_old_id.get(old_id) or fallback_contributors
            for value in values:
                if value not in contributors:
                    contributors.append(value)

        has_new_pixels = new_id in new_pixel_ids
        if not old_ids:
            status = "manual_new"
            contributors = [_MANUAL_CONTRIBUTOR_ID]
        elif len(old_ids) > 1:
            status = "manual_merge"
            if _MANUAL_CONTRIBUTOR_ID not in contributors:
                contributors.append(_MANUAL_CONTRIBUTOR_ID)
        elif not contributors:
            status = "manual_unattributed"
            contributors = [_MANUAL_CONTRIBUTOR_ID]
        elif has_new_pixels:
            status = "manual_extended"
            if _MANUAL_CONTRIBUTOR_ID not in contributors:
                contributors.append(_MANUAL_CONTRIBUTOR_ID)
        else:
            status = "preserved"

        attributed.append(
            {
                **record,
                "contributing_branch_ids": contributors,
                "manual_provenance": status,
            }
        )
    return attributed


def rebuild_result_from_mask(
    loaded: LoadedImage,
    mask: np.ndarray,
    params: UniversalParams,
    previous: UniversalResult | None = None,
) -> UniversalResult:
    started = time.perf_counter()
    params = enforce_shape_compatibility(params.validated(loaded.shape))
    is_ai = params.recognition_engine in AI_RECOGNITION_ENGINES
    if is_ai and (previous is None or previous.ai_metadata is None):
        previous = segment_loaded_image(loaded, params)
    clean_mask = np.asarray(mask, dtype=bool)
    if clean_mask.shape != loaded.shape:
        raise ValueError(
            T(
                "The manual mask size {0} differs from the original image size {1}",
                clean_mask.shape,
                loaded.shape,
            )
        )
    same_params = bool(
        previous is not None
        and previous.params.validated(loaded.shape) == params
    )
    can_reuse = (
        previous is not None
        and not previous.is_preview
        and previous.signal.shape == loaded.shape
        and previous.normalized_signal.shape == loaded.shape
        and previous.response.shape == loaded.shape
        and (same_params or is_ai)
    )
    if can_reuse:
        assert previous is not None
        signal = previous.signal
        normalized = previous.normalized_signal
        response = previous.response
        mode_used = previous.signal_mode_used
        black_value = previous.input_black_value
        white_value = previous.input_white_value
        diagnostics = (
            previous.input_saturation_fraction,
            previous.input_high_plateau_fraction,
            previous.input_largest_tie_fraction,
            previous.input_unique_levels,
            previous.intensity_warnings,
        )
    else:
        signal, mode_used = extract_signal(loaded.image, params)
        normalized, black_value, white_value = normalize_signal(signal, params)
        diagnostics = _input_intensity_diagnostics(signal, loaded.image.dtype)
        if params.intensity_space == "robust_snr":
            # Manual rebuilds of a robust-SNR result must recompute the same
            # noise-sigma response, not the ranked/background response used by
            # every other intensity space, or the exported response columns
            # would describe a different physical quantity than the analysis.
            response = _robust_snr_response(signal, params)
        else:
            denoised = _denoise(normalized, params.denoise_method, params.denoise_strength)
            response = robust_normalize(
                _background_response(
                    denoised,
                    params.background_method,
                    params.background_radius,
                    params.background_strength,
                )
            )
    labels, label_boxes = (
        _plain_component_labels_with_boxes(clean_mask)
        if is_ai else _component_labels_with_boxes(clean_mask, params)
    )
    rebuild_reference = (
        AI_RESPONSE_REFERENCE.get(params.recognition_engine, 3.5) if is_ai else _snr_seed_high(params)
        if is_ai or params.threshold_method == "snr_hysteresis"
        else None
    )
    reusable = None
    if (
        can_reuse
        and previous is not None
        and previous.labels.shape == loaded.shape
        and previous.objects
        # A multi-branch union measures its objects against the union
        # response's Otsu reference rather than this recipe's own reference,
        # so its records are not what a rebuild would compute; measure afresh.
        and int(getattr(previous, "quick_variant_count", 1)) <= 1
    ):
        # Same planes, same params, same confidence reference: any object the
        # edit did not touch keeps its exact previous measurements.
        reusable = (previous.labels, previous.objects)
    final_mask, final_labels, objects = measure_and_filter_labels(
        labels,
        signal,
        normalized,
        response,
        params,
        apply_filters=False,
        response_reference=rebuild_reference,
        reusable=reusable,
        bounding_boxes=label_boxes,
    )
    inherit_plan = bool(
        previous is not None
        and not previous.is_preview
        and previous.mask.shape == loaded.shape
        and previous.labels.shape == loaded.shape
        and same_params
    )
    objects = _manual_rebuild_object_provenance(
        objects,
        final_labels,
        previous if inherit_plan else None,
    )
    rebuilt = UniversalResult(
        mask=final_mask,
        labels=final_labels,
        objects=objects,
        signal=signal,
        normalized_signal=normalized,
        response=response,
        threshold_value=(previous.threshold_value if previous else float("nan")),
        low_threshold=(previous.low_threshold if previous else float("nan")),
        positive_fraction=float(final_mask.mean()),
        params=params,
        signal_mode_used=mode_used,
        input_black_value=float(black_value),
        input_white_value=float(white_value),
        processing_seconds=float(time.perf_counter() - started),
        analysis_scale=1.0,
        source_shape=loaded.shape,
        manually_edited=True,
        input_saturation_fraction=float(diagnostics[0]),
        input_high_plateau_fraction=float(diagnostics[1]),
        input_largest_tie_fraction=float(diagnostics[2]),
        input_unique_levels=int(diagnostics[3]),
        intensity_warnings=tuple(diagnostics[4]),
        # The edited mask still carries whatever boundary offset produced the
        # previous result, so keep documenting it.
        edge_offset_px=int(getattr(previous, "edge_offset_px", 0)) if previous else 0,
        edge_offset_grid_px=(
            int(getattr(previous, "edge_offset_grid_px", 0)) if previous else 0
        ),
    )
    if inherit_plan:
        assert previous is not None
        for attribute in (
            "quick_plan_key",
            "quick_variant_count",
            "quick_plan",
            "compiled_plan",
        ):
            if hasattr(previous, attribute):
                setattr(rebuilt, attribute, getattr(previous, attribute))
    if is_ai and previous is not None:
        rebuilt = _copy_result_attributes(previous, rebuilt)
        assert rebuilt.ai_metadata is not None
        rebuilt.ai_metadata.setdefault("postprocessing", []).append({
            "operation": "manual_mask_edit", "main_area_px": int(rebuilt.mask.sum()),
            "review_area_px": int(rebuilt.review_mask.sum()) if rebuilt.review_mask is not None else 0,
        })
    return rebuilt


def _display_curve(values: np.ndarray, low: float, high: float, gamma: float) -> np.ndarray:
    """The percentile stretch + gamma curve, element-wise, to 8-bit."""

    scaled = (values.astype(np.float32, copy=False) - np.float32(low)) / np.float32(
        high - low
    )
    scaled = np.clip(scaled, 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, np.float32(gamma), dtype=np.float32)
    return np.rint(scaled * 255.0).astype(np.uint8)


MAX_EDGE_OFFSET_PX = 256


def _inherit_branch_contributors(
    objects: list[dict[str, object]],
    new_labels: np.ndarray,
    previous_labels: np.ndarray,
    previous_objects: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Carry ``contributing_branch_ids`` across a relabelling by pixel overlap."""

    contributors_by_old_id: dict[int, list[str]] = {}
    for record in previous_objects:
        raw = record.get("contributing_branch_ids")
        if not isinstance(raw, (list, tuple)) or not raw:
            continue
        try:
            contributors_by_old_id[int(record["id"])] = [str(item) for item in raw]
        except (KeyError, TypeError, ValueError):
            continue
    if not contributors_by_old_id or previous_labels.shape != new_labels.shape:
        return objects
    overlap = (new_labels > 0) & (previous_labels > 0)
    if not bool(overlap.any()):
        return objects
    stride = int(previous_labels.max(initial=0)) + 1
    packed = new_labels[overlap].astype(np.int64) * stride + previous_labels[overlap].astype(np.int64)
    contributors_by_new_id: dict[int, list[str]] = {}
    for value in np.unique(packed).tolist():
        new_id, old_id = divmod(int(value), stride)
        for item in contributors_by_old_id.get(old_id, ()):
            bucket = contributors_by_new_id.setdefault(new_id, [])
            if item not in bucket:
                bucket.append(item)
    inherited: list[dict[str, object]] = []
    for record in objects:
        try:
            new_id = int(record.get("id", 0))
        except (TypeError, ValueError):
            new_id = 0
        contributors = contributors_by_new_id.get(new_id)
        if contributors:
            inherited.append({**record, "contributing_branch_ids": list(contributors)})
        else:
            inherited.append(record)
    return inherited


def apply_edge_offset(result: UniversalResult, offset_px: int) -> UniversalResult:
    """Offset the final selection boundary by ``offset_px`` source pixels.

    This is a post-processing step on the *result*, not a recognition
    parameter: positive values expand every selected region outward, negative
    values contract it, using the exact Euclidean distance of
    :func:`adjust_mask_boundary`.  Objects are relabelled and re-measured on
    the result's own grid (no recipe filters are re-applied, so the step never
    removes or adds objects other than by geometry), the recipe parameters are
    left untouched, and the offset is recorded on the returned result so
    exports can document it.  ``offset_px == 0`` returns an equivalent result
    with the offset fields reset.
    """

    started = time.perf_counter()
    if isinstance(offset_px, (bool, np.bool_)):
        raise TypeError(T("The edge distance must be a whole number of pixels"))
    offset = int(operator.index(offset_px))
    if abs(offset) > MAX_EDGE_OFFSET_PX:
        raise ValueError(T("The edge distance must be within ±{0} pixels", MAX_EDGE_OFFSET_PX))
    if offset == 0:
        if result.edge_offset_px == 0 and result.edge_offset_grid_px == 0:
            return result
        return _copy_result_attributes(
            result,
            replace(result, edge_offset_px=0, edge_offset_grid_px=0),
        )

    mask_shape = tuple(int(value) for value in result.mask.shape[:2])
    source_shape = tuple(
        int(value) for value in (result.source_shape or mask_shape)[:2]
    )
    analysis_scale = float(result.analysis_scale) if mask_shape != source_shape else 1.0
    grid_offset = int(round(offset * analysis_scale))
    if grid_offset == 0:
        return _copy_result_attributes(
            result,
            replace(result, edge_offset_px=offset, edge_offset_grid_px=0),
        )

    params = enforce_shape_compatibility(result.params.validated())
    if mask_shape == source_shape:
        grid_params = params.validated(mask_shape)
    elif params.scale_mode == "relative_grid":
        grid_params = params.scaled_spatial(
            min(mask_shape) / float(max(16, params.reference_short_side))
        ).validated(mask_shape)
    else:
        grid_params = (
            params.validated(source_shape)
            .scaled_for_preview(analysis_scale)
            .validated(mask_shape)
        )
    adjusted = adjust_mask_boundary(result.mask, grid_offset)
    is_ai = result.ai_metadata is not None
    labels, label_boxes = (
        _plain_component_labels_with_boxes(adjusted)
        if is_ai else _component_labels_with_boxes(adjusted, grid_params)
    )
    reference = (
        AI_RESPONSE_REFERENCE.get(result.params.recognition_engine, 3.5) if is_ai else _snr_seed_high(params)
        if is_ai or params.threshold_method == "snr_hysteresis"
        else None
    )
    final_mask, final_labels, objects = measure_and_filter_labels(
        labels,
        result.signal,
        result.normalized_signal,
        result.response,
        grid_params,
        apply_filters=False,
        response_reference=reference,
        bounding_boxes=label_boxes,
    )
    objects = _inherit_branch_contributors(
        objects,
        final_labels,
        np.asarray(result.labels),
        result.objects,
    )
    if mask_shape != source_shape:
        objects = _rescale_preview_objects(objects, analysis_scale)
    offset_result = replace(
        result,
        mask=final_mask,
        labels=final_labels,
        objects=objects,
        positive_fraction=float(final_mask.mean()),
        processing_seconds=float(result.processing_seconds)
        + float(time.perf_counter() - started),
        edge_offset_px=offset,
        edge_offset_grid_px=grid_offset,
    )
    offset_result = _copy_result_attributes(result, offset_result)
    if is_ai and offset_result.review_base_mask is not None:
        offset_result.review_base_mask = adjust_mask_boundary(offset_result.review_base_mask, grid_offset)
        offset_result.review_mask = offset_result.review_base_mask & ~offset_result.mask
        assert offset_result.ai_metadata is not None
        offset_result.ai_metadata["review_area_px"] = int(offset_result.review_mask.sum())
        offset_result.ai_metadata.setdefault("postprocessing", []).append({
            "operation": "edge_offset", "source_pixels": offset, "grid_pixels": grid_offset,
            "main_area_px": int(offset_result.mask.sum()), "review_area_px": int(offset_result.review_mask.sum()),
        })
    return offset_result


def _copy_result_attributes(source: UniversalResult, target: UniversalResult) -> UniversalResult:
    """Carry the ad-hoc attributes (quick plan, analysis signature) to a copy."""

    for attribute in (
        "quick_plan_key",
        "quick_variant_count",
        "quick_plan",
        "compiled_plan",
        "analysis_input_signature",
    ):
        if hasattr(source, attribute):
            setattr(target, attribute, getattr(source, attribute))
    if source.ai_metadata is not None:
        target.ai_metadata = copy.deepcopy(source.ai_metadata)
        if target.params.recognition_engine in AI_RECOGNITION_ENGINES:
            target.ai_metadata["engine"] = target.params.recognition_engine
            target.ai_metadata["scope_note"] = (
                T(
                    "Linear discriminant on six features normalized per image; calibration data: "
                    "16 image sets, 1315 manually judged candidates.",
                )
            )
            target.signal_mode_used = target.params.recognition_engine
        base = source.review_base_mask if source.review_base_mask is not None else source.review_mask
        if base is not None:
            if base.shape != target.mask.shape:
                raise ValueError(T(
                    "The AI review base and the main selection differ in size; they are not "
                    "rescaled silently.",
                ))
            target.review_base_mask = np.asarray(base, dtype=bool).copy()
            target.review_mask = target.review_base_mask & ~target.mask
        target.ai_metadata["main_area_px"] = int(target.mask.sum())
        target.ai_metadata["review_area_px"] = int(target.review_mask.sum()) if target.review_mask is not None else 0
        target.ai_metadata["review_in_main_quantification"] = False
    return target


def restore_ai_result_sidecars(
    mask_path: str | Path,
    loaded: LoadedImage,
    result: UniversalResult,
) -> UniversalResult:
    """Restore an exported learned result; classic exports remain unchanged."""
    path = Path(mask_path)
    suffix = "positive_mask.tif"
    if not path.name.casefold().endswith(suffix):
        return result
    prefix = path.name[:-len(suffix)]
    sidecar = path.with_name(prefix + "ai_analysis.json")
    if not sidecar.is_file():
        return result
    try:
        engine = normalize_engine_id(
            json.loads(sidecar.read_text(encoding="utf-8")).get("ai_metadata", {}).get("engine")
        )
    except (OSError, ValueError):
        return result
    if engine not in AI_RECOGNITION_ENGINES:
        return result
    return ai_engine_module(engine).restore_sidecars(sidecar, loaded, result)


def export_bundle_exists(output_parent: str | Path, source: str | Path) -> bool:
    """True when a FluoroAnts result bundle for ``source`` already sits in ``output_parent``.

    Both the legacy result directory and the prefixed-file layout are
    recognised, exactly like the reservation logic used while exporting.
    """

    parent = Path(output_parent).expanduser()
    if not parent.is_dir():
        return False
    return _bundle_prefix_in_use(parent, export_bundle_base_name(source))


def normalize_for_display(image: np.ndarray, gamma: float = 0.82) -> np.ndarray:
    sample = _finite_sample(image)
    low, high = (float(value) for value in np.percentile(sample, (0.2, 99.8)))
    if high <= low:
        low = float(sample.min(initial=0.0))
        high = float(sample.max(initial=1.0))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    if image.dtype == np.uint8 or image.dtype == np.uint16:
        # The curve is a pure per-value function, so evaluate it once per grey
        # level and gather: identical bytes, without a full-frame float32
        # stretch, clip, power and rounding pass over a 24 MP frame.
        levels = int(image.max(initial=0)) + 1
        table = _display_curve(np.arange(levels, dtype=image.dtype), low, high, gamma)
        if image.dtype == np.uint8:
            table256 = np.zeros(256, dtype=np.uint8)
            table256[:levels] = table
            return cv2.LUT(np.ascontiguousarray(image), table256)
        return table[image]
    return _display_curve(image, low, high, gamma)


def image_to_rgb_display(image: np.ndarray, max_side: int | None = None) -> tuple[np.ndarray, float]:
    source_height, source_width = image.shape[:2]
    scale = 1.0
    display_image = image
    if max_side is not None and max(source_height, source_width) > max_side:
        scale = float(max_side) / float(max(source_height, source_width))
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        display_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if display_image.ndim == 2:
        gray = normalize_for_display(display_image)
        rgb = np.repeat(gray[..., None], 3, axis=2)
    elif display_image.shape[2] == 1:
        gray = normalize_for_display(display_image[..., 0])
        rgb = np.repeat(gray[..., None], 3, axis=2)
    else:
        channels = [normalize_for_display(display_image[..., index]) for index in range(3)]
        bgr = np.stack(channels, axis=2)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8), scale


def image_to_rgb_original_values(
    image: np.ndarray,
    max_side: int | None = None,
) -> tuple[np.ndarray, float]:
    """Render original channel values without percentile or per-channel stretch.

    This is intended for reopening a clean Merge TIFF beside an exported
    selection-review image. Integer inputs use their complete dtype range;
    floating microscopy inputs use the conventional 0..1 display range.
    Resizing is display-only and happens after that fixed conversion.
    """

    source = np.asarray(image)
    if source.ndim not in {2, 3}:
        raise ValueError(T(
            "Original-brightness display supports 2-D greyscale or colour images only.",
        ))
    if source.ndim == 3 and source.shape[2] < 1:
        raise ValueError(T("A colour image needs at least one channel."))

    if source.dtype == np.uint8:
        converted = source.copy()
    elif source.dtype == np.uint16:
        converted = np.rint(source.astype(np.float64) / 257.0).astype(np.uint8)
    elif np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        span = float(info.max) - float(info.min)
        if span <= 0.0:
            converted = np.zeros(source.shape, dtype=np.uint8)
        else:
            converted = np.rint(
                (source.astype(np.float64) - float(info.min)) * (255.0 / span)
            ).clip(0.0, 255.0).astype(np.uint8)
    elif np.issubdtype(source.dtype, np.floating):
        converted = np.rint(
            np.clip(source.astype(np.float64), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    else:
        raise TypeError(T("Unsupported image type for original brightness: {0}", source.dtype))

    if converted.ndim == 2:
        rgb = np.repeat(converted[..., None], 3, axis=2)
    elif converted.shape[2] == 1:
        rgb = np.repeat(converted[..., :1], 3, axis=2)
    else:
        rgb = cv2.cvtColor(converted[..., :3], cv2.COLOR_BGR2RGB)

    source_height, source_width = rgb.shape[:2]
    scale = 1.0
    if max_side is not None and max(source_height, source_width) > max_side:
        scale = float(max_side) / float(max(source_height, source_width))
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8), scale


def display_channel_for_signal_mode(signal_mode: str) -> str:
    """Map a requested analysis signal to its human-facing RGB view channel."""

    return {
        "red": "r",
        "green": "g",
        "blue": "b",
    }.get(str(signal_mode).casefold(), "rgb")


def select_rgb_display_channel(rgb: np.ndarray, channel: str) -> np.ndarray:
    """Return RGB or one color-preserving R/G/B plane for display/export.

    A selected red plane is encoded as ``[R, 0, 0]`` rather than copied into
    all three planes as gray. The same rule is shared by the live view, normal
    exports and batch exports.
    """

    source = np.asarray(rgb, dtype=np.uint8)
    if source.ndim == 2:
        source = np.repeat(source[..., None], 3, axis=2)
    if source.ndim != 3 or source.shape[2] < 1:
        raise ValueError(T("The display image must contain at least one channel."))
    if source.shape[2] < 3:
        plane = source[..., 0]
        source = np.repeat(plane[..., None], 3, axis=2)
    else:
        source = source[..., :3]

    channel_index = {"r": 0, "g": 1, "b": 2}.get(str(channel).casefold())
    if channel_index is None:
        return np.ascontiguousarray(source, dtype=np.uint8)
    output = np.zeros(source.shape, dtype=np.uint8)
    output[..., channel_index] = source[..., channel_index]
    return output


def image_to_signal_rgb_display(
    image: np.ndarray,
    signal_mode: str,
    max_side: int | None = None,
) -> tuple[np.ndarray, float]:
    """Render the same requested channel that drives segmentation."""

    rgb, scale = image_to_rgb_display(image, max_side=max_side)
    channel = display_channel_for_signal_mode(signal_mode)
    return select_rgb_display_channel(rgb, channel), scale


def scalar_to_rgb_display(image: np.ndarray, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    gray = normalize_for_display(image, gamma=0.9)
    if target_shape is not None and gray.shape != target_shape:
        gray = cv2.resize(
            gray,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.repeat(gray[..., None], 3, axis=2)


def _mask_as_uint8(mask: np.ndarray) -> np.ndarray:
    """Return a contiguous 0/1 uint8 view of a boolean mask without copying."""

    source = np.asarray(mask)
    if source.dtype == np.bool_:
        return np.ascontiguousarray(source).view(np.uint8)
    return source.astype(np.uint8)


def resized_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool, copy=False)
    source = np.asarray(mask)
    resized = cv2.resize(
        _mask_as_uint8(source),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    if source.dtype == np.bool_:
        # Nearest sampling only copies the 0/1 bytes of a boolean source, so
        # the result can be reinterpreted in place instead of copied.
        return resized.view(np.bool_)
    return resized.astype(bool)


def resized_labels(labels: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if labels.shape == target_shape:
        return labels.astype(np.int32, copy=False)
    # Nearest-neighbour sampling copies whole elements, so the label plane can
    # be resized directly as int32.  This is the same source-pixel mapping as
    # the former float32 round trip but without two full-frame conversions,
    # and it stays exact above 2**24 labels where float32 could not.
    resized = cv2.resize(
        np.ascontiguousarray(labels, dtype=np.int32),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.int32, copy=False)


def render_overlay(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    fill: bool = True,
    contour: bool = True,
    object_records: list[dict[str, object]] | None = None,
    object_scale: float = 1.0,
) -> np.ndarray:
    # One guaranteed C-contiguous writable copy. The previous
    # ascontiguousarray(...).copy() copied twice for a non-contiguous input; a
    # plain copy would alias a contiguous input, which callers must not mutate.
    output = np.array(base_rgb, dtype=np.uint8, order="C")
    target_shape = output.shape[:2]
    mask_display = resized_mask(mask, target_shape)
    if fill and bool(mask_display.any()):
        overlay = output.copy()
        overlay[mask_display] = np.asarray((50, 205, 145), dtype=np.uint8)
        output = cv2.addWeighted(output, 0.68, overlay, 0.32, 0)
    if contour and bool(mask_display.any()):
        contours, _ = cv2.findContours(
            mask_display.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (255, 195, 60), 1, cv2.LINE_AA)
    if object_records and len(object_records) <= 400:
        for record in object_records:
            x = int(round(float(record["centroid_x_px"]) * object_scale))
            y = int(round(float(record["centroid_y_px"]) * object_scale))
            cv2.putText(
                output,
                str(record["id"]),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return output


def _light_contour_dash_segments(
    contour: np.ndarray,
    *,
    dash_length: int,
    gap_length: int,
) -> tuple[list[np.ndarray], tuple[int, int] | None]:
    """Return the light dash segments along one closed contour.

    The walk is the same as before (dash phase carries across polyline
    vertices, every dash piece is clipped at a vertex and rounded to integer
    pixels) but runs on Python floats instead of one-element NumPy arrays,
    which was the dominant cost when exporting a mask with thousands of
    objects.  ``round`` and ``np.rint`` both round half to even, so the
    integer endpoints are identical.  A one-point contour is returned as a
    dot centre instead.
    """

    points = np.asarray(contour).reshape(-1, 2)
    if len(points) == 0:
        return [], None
    if len(points) == 1:
        return [], (int(points[0, 0]), int(points[0, 1]))

    coordinates = points.astype(np.float64, copy=False).tolist()
    coordinates.append(coordinates[0])
    segments: list[np.ndarray] = []
    draw_light = True
    run_remaining = float(dash_length)
    epsilon = 1e-9
    for index in range(len(coordinates) - 1):
        start_x, start_y = coordinates[index]
        end_x, end_y = coordinates[index + 1]
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        segment_length = float(math.hypot(delta_x, delta_y))
        if segment_length <= epsilon:
            continue
        direction_x = delta_x / segment_length
        direction_y = delta_y / segment_length
        offset = 0.0
        while offset < segment_length - epsilon:
            step = min(run_remaining, segment_length - offset)
            if draw_light and step > epsilon:
                dash_start_x = round(start_x + direction_x * offset)
                dash_start_y = round(start_y + direction_y * offset)
                dash_end_x = round(start_x + direction_x * (offset + step))
                dash_end_y = round(start_y + direction_y * (offset + step))
                segments.append(
                    np.array(
                        [[dash_start_x, dash_start_y], [dash_end_x, dash_end_y]],
                        dtype=np.int32,
                    ).reshape(-1, 1, 2)
                )
            offset += step
            run_remaining -= step
            if run_remaining <= epsilon:
                draw_light = not draw_light
                run_remaining = float(dash_length if draw_light else gap_length)
    return segments, None


def render_selection_merge(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    contour_color: tuple[int, int, int] = (255, 255, 255),
    contour_thickness: int | None = None,
) -> np.ndarray:
    """Render a readable Merge image with selection boundaries and no fill.

    The image interior is deliberately left untouched: no translucent color and
    no object numbers are added.  Equal-width white segments overwrite a black
    contour at regular intervals, forming static black-and-white marching ants
    without the heavy black halo of a wider under-stroke.  All boundaries are
    drawn, including holes inside ring-shaped selections.
    """

    output = np.array(base_rgb, dtype=np.uint8, order="C")
    if output.ndim != 3 or output.shape[2] != 3:
        raise ValueError(T("The Merge base must be a three-channel RGB image."))
    mask_display = resized_mask(mask, output.shape[:2])
    if not bool(mask_display.any()):
        return output
    contours, _ = cv2.findContours(
        mask_display.astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contour_thickness is None:
        contour_thickness = max(1, int(round(min(output.shape[:2]) / 1024.0)))
    thickness = max(1, int(contour_thickness))
    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    light_color = tuple(int(value) for value in contour_color)
    dash_length = max(6, thickness * 6)
    gap_length = dash_length
    pending: list[np.ndarray] = []
    for contour in contours:
        segments, dot = _light_contour_dash_segments(
            contour,
            dash_length=dash_length,
            gap_length=gap_length,
        )
        if dot is not None:
            # Keep drawing order: flush the dashes gathered so far first.
            if pending:
                cv2.polylines(output, pending, False, light_color, thickness, cv2.LINE_AA)
                pending = []
            cv2.circle(
                output,
                dot,
                max(1, int(math.ceil(thickness / 2.0))),
                light_color,
                -1,
                cv2.LINE_AA,
            )
            continue
        pending.extend(segments)
    if pending:
        cv2.polylines(output, pending, False, light_color, thickness, cv2.LINE_AA)
    return output


def _write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix.casefold()
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(T("Cannot encode the image: {0}", path.name))
    encoded.tofile(path)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value).strip(" ._")
    shortened = (cleaned[:120] or "image")
    # Keep historical ASCII names unchanged while respecting common 255-byte
    # component limits for multibyte Unicode names plus the longest artifact
    # suffix. A hash prevents distinct long names from collapsing after the
    # byte-safe cut.
    max_stem_bytes = 160
    if len(shortened.encode("utf-8")) <= max_stem_bytes:
        return shortened
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:10]
    suffix = f"_{digest}"
    byte_budget = max_stem_bytes - len(suffix.encode("ascii"))
    prefix = shortened.encode("utf-8")[:byte_budget].decode(
        "utf-8",
        errors="ignore",
    ).rstrip(" ._")
    return f"{prefix or 'image'}{suffix}"


def export_bundle_base_name(source: str | Path) -> str:
    """Return the deterministic, filesystem-safe prefix for one source image."""

    return f"{_safe_name(Path(source).stem)}_FluoroAnts"


def _csv_safe(value: object) -> object:
    """Neutralize spreadsheet formula injection for user-controlled text cells.

    A source filename such as ``=SUM(...)`` or ``-2+3`` is otherwise interpreted
    as a formula when the exported CSV is opened in Excel/Sheets.  Non-string
    cells (numbers, booleans) are returned unchanged so numeric output is intact.
    """

    if not isinstance(value, str) or not value:
        return value
    if value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _unique_output_directory(output_root: Path, source: Path) -> Path:
    stem = _safe_name(source.stem)
    base = output_root / f"{stem}_FluoroAnts"
    candidate = base
    index = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            candidate = output_root / f"{base.name}_{index:03d}"
            index += 1
        else:
            return candidate


def normalize_export_bundle_mode(value: object) -> str:
    """Validate whether one image exports as a directory or sibling files."""

    if not isinstance(value, str):
        raise TypeError(T("The export bundle mode must be a string"))
    if value not in EXPORT_BUNDLE_MODES:
        choices = T(", ").join(sorted(EXPORT_BUNDLE_MODES))
        raise ValueError(T(
            "Unknown export bundle mode: {0}; allowed values: {1}",
            value or T("empty value"),
            choices,
        ))
    return value


def _prefixed_export_filename(prefix: str, leaf_name: str) -> str:
    return f"{prefix}_{leaf_name}"


def _bundle_prefix_in_use(output_root: Path, prefix: str) -> bool:
    """Check both legacy result directories and every new bundle artifact."""

    if os.path.lexists(output_root / prefix):
        return True
    return any(
        os.path.lexists(output_root / _prefixed_export_filename(prefix, leaf_name))
        for leaf_name in _EXPORT_MARKER_FILENAMES
    )


def _reserve_output_bundle(
    output_root: Path,
    source: Path,
    *,
    allowed_output_root: str | Path | None = None,
) -> tuple[str, Path]:
    """Atomically reserve a bundle prefix with a temporary staging directory."""

    base = export_bundle_base_name(source)
    index = 1
    while True:
        checked_root = _resolve_contained_output_parent(
            output_root,
            allowed_output_root,
        )
        prefix = base if index == 1 else f"{base}_{index:03d}"
        staging = checked_root / f".{prefix}.exporting"
        try:
            staging.mkdir(exist_ok=False)
        except FileExistsError:
            index += 1
            continue
        try:
            # Validate again before writing the marker. A destination replaced
            # between the first check and mkdir is rejected and this exact
            # staging path is cleaned without traversing unexpected children.
            checked_root = _resolve_contained_output_parent(
                checked_root,
                allowed_output_root,
            )
            checked_staging = _resolve_contained_output_parent(
                staging,
                allowed_output_root,
            )
            if checked_staging != staging:
                raise ValueError(T("The temporary export folder changed while it was reserved"))
            if _bundle_prefix_in_use(checked_root, prefix):
                staging.rmdir()
                index += 1
                continue
            (staging / "INCOMPLETE.txt").write_text(
                T(
                    "The export is not finished; this temporary folder is removed automatically "
                    "when it is.",
                ),
                encoding="utf-8",
            )
        except Exception:
            try:
                _cleanup_export_staging(staging)
            except OSError:
                pass
            raise
        return prefix, staging


def _cleanup_export_staging(staging: Path) -> None:
    """Remove only files created inside one exact private staging directory."""

    try:
        if _path_is_link_or_reparse_point(staging):
            raise OSError(T(
                "The temporary export folder has become a link; refusing to clean its target",
            ))
    except FileNotFoundError:
        return
    try:
        entries = tuple(staging.iterdir())
    except FileNotFoundError:
        return
    for entry in sorted(
        entries,
        key=lambda item: (item.name == "INCOMPLETE.txt", item.name.casefold()),
    ):
        if entry.is_dir() and not entry.is_symlink():
            raise OSError(T(
                "The temporary export folder contains unexpected subfolders; refusing to clean "
                "recursively",
            ))
        entry.unlink(missing_ok=True)
    staging.rmdir()


def _copy_file_exclusive(source: Path, destination: Path, *, fsync: bool = True) -> None:
    """Copy one staged artifact without ever replacing an existing path."""

    created = False
    try:
        with source.open("rb") as input_handle, destination.open("xb") as output_handle:
            created = True
            for chunk in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                output_handle.write(chunk)
            output_handle.flush()
            if fsync:
                os.fsync(output_handle.fileno())
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def _commit_file_exclusive(source: Path, destination: Path) -> None:
    """Publish one staged artifact under its final name without replacing.

    The staged file is first flushed to disk, then hard-linked to the final
    name: ``os.link`` never overwrites an existing path and, unlike the former
    byte copy, costs nothing for a 70 MB PNG.  File systems without hard links
    (FAT/exFAT, some network shares) fall back to the exclusive copy.
    """

    try:
        # Read/write mode: Windows only flushes handles opened with write
        # access, and nothing is written here.
        with source.open("r+b") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        _copy_file_exclusive(source, destination)


def _resolve_contained_output_parent(
    output_parent: str | Path,
    allowed_output_root: str | Path | None,
) -> Path:
    """Resolve a destination immediately before writing and recheck its root.

    Batch/GUI callers pass the canonical root frozen when the export was
    planned.  Re-resolving only the destination catches a mirror subdirectory
    or output root that was replaced by a symlink/junction during analysis.
    """

    requested = Path(output_parent).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    if allowed_output_root is None:
        return requested.resolve()

    allowed = Path(allowed_output_root).expanduser()
    if not allowed.is_absolute():
        raise ValueError(T("The safe export root must be an absolute path"))
    try:
        requested.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(T("The export path lies outside the confirmed output root")) from exc

    cursor = requested
    while True:
        try:
            if _path_is_link_or_reparse_point(cursor):
                raise ValueError(T(
                    "The export path became a link or reparse point while the task was running",
                ))
        except FileNotFoundError:
            pass
        if cursor == allowed:
            break
        if cursor == cursor.parent:
            raise ValueError(T(
                "The export path cannot be traced back to the confirmed output root",
            ))
        cursor = cursor.parent

    resolved = requested.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(T(
            "The export path left the confirmed output root while the task was running",
        )) from exc
    return resolved


def normalize_export_location_mode(value: object) -> str:
    """Validate a batch export layout without silently changing its meaning."""

    if not isinstance(value, str):
        raise TypeError(T("The export location mode must be a string"))
    if value not in EXPORT_LOCATION_MODES:
        choices = T(", ").join(sorted(EXPORT_LOCATION_MODES))
        raise ValueError(T(
            "Unknown export location mode: {0}; allowed values: {1}",
            value or T("empty value"),
            choices,
        ))
    return value


def _normalize_export_relative_path(
    source: Path,
    relative_path: str | Path | None,
) -> Path:
    """Return a safe relative source path used only to mirror its parents."""

    if relative_path is None:
        return Path(source.name)
    if isinstance(relative_path, (bytes, bytearray)):
        raise TypeError(T("The relative import path must not be bytes"))
    try:
        candidate = Path(relative_path)
    except TypeError as exc:
        raise TypeError(T("The relative import path must be a string or a Path")) from exc
    raw = str(relative_path)
    windows_absolute = bool(re.match(r"(?i)^[a-z]:[\\/]", raw)) or raw.startswith(
        ("\\\\", "//")
    )
    if not raw or candidate.is_absolute() or candidate.anchor or windows_absolute:
        raise ValueError(T("The relative import path must lie inside the import root"))
    if any(part == ".." for part in candidate.parts):
        raise ValueError(T("The relative import path must not step up to a parent folder"))
    if candidate.name in ("", ".", ".."):
        raise ValueError(T("The relative import path must contain a file name"))
    # Only the trusted source filename controls the result-directory stem.  The
    # supplied relative value contributes directory structure, never a rename.
    return candidate.parent / source.name


def resolve_export_parent(
    source: str | Path,
    output_root: str | Path | None,
    *,
    relative_path: str | Path | None = None,
    location_mode: str = EXPORT_LOCATION_FLAT,
) -> Path:
    """Resolve one image's output parent for flat, mirrored or source export."""

    mode = normalize_export_location_mode(location_mode)
    source_path = Path(source).expanduser().resolve()
    relative = _normalize_export_relative_path(source_path, relative_path)
    if mode == EXPORT_LOCATION_SOURCE:
        return source_path.parent
    if output_root is None:
        raise ValueError(T("Flat or mirror export needs an output root folder"))
    root = Path(output_root).expanduser().resolve()
    existing = root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir():
        raise NotADirectoryError(T("An existing parent of the output path is not a folder"))
    if mode == EXPORT_LOCATION_FLAT:
        return root

    parent = (root / relative.parent).resolve()
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(T("The mirror export path lies outside the chosen output root")) from exc
    existing = parent
    while not existing.exists() and existing != root:
        existing = existing.parent
    if existing.exists() and not existing.is_dir():
        raise NotADirectoryError(T("An existing parent of the mirror export path is not a folder"))
    return parent


def _verify_file_fingerprint(
    path: Path,
    fingerprint: Mapping[str, object],
    *,
    label: str,
) -> None:
    stat = path.stat()
    if int(stat.st_size) != int(fingerprint["size"]):
        raise RuntimeError(T("{0}: the size has changed; export refused: {1}", label, path.name))
    if int(stat.st_mtime_ns) != int(fingerprint["mtime_ns"]):
        raise RuntimeError(T(
            "{0}: the modification time has changed; export refused: {1}",
            label,
            path.name,
        ))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != str(fingerprint["sha256"]):
        raise RuntimeError(T("{0}: the content has changed; export refused: {1}", label, path.name))


_PUBLIC_IMAGE_RECORD_KEYS = ("size", "sha256", "dtype", "shape", "color_order")


def public_image_record(record: object) -> dict[str, object] | None:
    """Describe an input image in an export file without its location on disk.

    Exported records are meant to be shared, so they name the file and carry
    its content fingerprint, but never the folder it was read from.
    """

    if not isinstance(record, Mapping):
        return None
    name = record.get("file_name") or Path(str(record.get("path") or "")).name
    public: dict[str, object] = {"file_name": str(name)}
    for key in _PUBLIC_IMAGE_RECORD_KEYS:
        if key in record:
            public[key] = copy.deepcopy(record[key])
    return public


def _verify_source(loaded: LoadedImage) -> None:
    _verify_file_fingerprint(loaded.path, loaded.metadata, label=T("Source image"))
    reference = loaded.metadata.get("_crosstalk_reference")
    if reference is None:
        return
    if not isinstance(reference, Mapping):
        raise RuntimeError(T(
            "The verification record of the crosstalk reference image is damaged; export refused",
        ))
    reference_path = Path(str(reference.get("path", ""))).expanduser().resolve()
    if not reference_path.is_file():
        raise RuntimeError(
            T(
                "The crosstalk reference image no longer exists; export refused: {0}",
                reference_path.name or T("reference image"),
            )
        )
    _verify_file_fingerprint(reference_path, reference, label=T("Crosstalk reference image"))


def _normalize_export_artifacts(
    artifacts: Iterable[str] | None,
) -> frozenset[str]:
    """Validate and freeze one artifact request before any output is created."""

    if artifacts is None:
        return EXPORT_ARTIFACT_IDS
    if isinstance(artifacts, str):
        values: tuple[object, ...] = (artifacts,) if artifacts else ()
    elif isinstance(artifacts, (bytes, bytearray)):
        raise TypeError(T("An export artifact ID must be a string, not bytes"))
    else:
        try:
            values = tuple(artifacts)
        except TypeError as exc:
            raise TypeError(T("Export artifacts must be an iterable of string IDs")) from exc

    if not values:
        raise ValueError(T("Choose at least one export artifact"))
    non_strings = [value for value in values if not isinstance(value, str)]
    if non_strings:
        raise TypeError(T("All export artifact IDs must be strings"))
    requested = frozenset(values)
    unknown = requested - EXPORT_ARTIFACT_IDS
    if unknown:
        choices = T(", ").join(DEFAULT_EXPORT_ARTIFACTS)
        invalid = T(", ").join(sorted(unknown))
        raise ValueError(T("Unknown export artifact: {0}; allowed values: {1}", invalid, choices))
    return requested


def _normalize_outline_expand_px(value: int) -> int:
    """Validate and freeze the export-only review-outline expansion."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(T(
            "The outline expansion for export must be a whole number of pixels from 0 to 512",
        ))
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(T(
            "The outline expansion for export must be a whole number of pixels from 0 to 512",
        )) from exc
    if not 0 <= normalized <= 512:
        raise ValueError(T("The outline expansion for export must be between 0 and 512 pixels"))
    return int(normalized)


def normalize_merge_base_mode(value: object) -> str:
    """Validate the background used only for exported selection-review images."""

    if not isinstance(value, str):
        raise TypeError(T("The Merge base mode must be a string"))
    if value not in MERGE_BASE_MODES:
        choices = T(", ").join(sorted(MERGE_BASE_MODES))
        raise ValueError(T(
            "Unknown Merge base mode: {0}; allowed values: {1}",
            value or T("empty value"),
            choices,
        ))
    return value


def _validate_original_merge_source(
    analysis_loaded: LoadedImage,
    source_loaded: LoadedImage | None,
) -> LoadedImage:
    """Return the verified full-resolution source paired with an analysis input."""

    if source_loaded is None:
        raise ValueError(T(
            "The full-resolution original image is missing although the original was chosen as "
            "the Merge base",
        ))
    if source_loaded is analysis_loaded:
        raise ValueError(T(
            "The Merge original must not be the same object as the crosstalk-corrected analysis "
            "input",
        ))
    source_preprocessing = source_loaded.metadata.get("analysis_preprocessing")
    if (
        isinstance(source_preprocessing, Mapping)
        or "analysis_input_signature" in source_loaded.metadata
        or "_crosstalk_reference" in source_loaded.metadata
    ):
        raise ValueError(T(
            "The Merge original still carries an analysis preprocessing mark; it is not the "
            "unmodified original",
        ))
    metadata_shape = tuple(source_loaded.metadata.get("shape", ()))
    if metadata_shape != tuple(source_loaded.image.shape):
        raise ValueError(T(
            "The in-memory size of the Merge original differs from the metadata of the source file",
        ))
    if str(source_loaded.metadata.get("dtype", "")) != str(source_loaded.image.dtype):
        raise ValueError(T(
            "The in-memory pixel type of the Merge original differs from the metadata of the "
            "source file",
        ))
    if source_loaded.path.expanduser().resolve() != analysis_loaded.path.expanduser().resolve():
        raise ValueError(T(
            "The Merge original and the current analysis input are not the same source file",
        ))
    if source_loaded.shape != analysis_loaded.shape:
        raise ValueError(
            T(
                "The Merge original and the current analysis input differ in size: {0}×{1} versus "
                "{2}×{3}",
                source_loaded.shape[1],
                source_loaded.shape[0],
                analysis_loaded.shape[1],
                analysis_loaded.shape[0],
            )
        )
    for key, label in (
        ("sha256", T("Content fingerprint")),
        ("size", T("File size")),
        ("mtime_ns", T("Modification time")),
    ):
        if str(source_loaded.metadata.get(key, "")) != str(
            analysis_loaded.metadata.get(key, "")
        ):
            raise ValueError(T(
                "The Merge original and the current analysis input differ: {0}",
                label,
            ))
    _verify_source(source_loaded)
    return source_loaded


def _length_unit_slug(unit: str) -> str:
    """Column-safe slug for a length unit (``µm`` → ``um``)."""

    slug = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    slug = re.sub(r"[^a-z0-9]+", "", slug)
    # ``pixel_size_unit`` is the fixed column containing the human-readable
    # unit.  Never generate that same name for the numeric pixel-size column.
    return "customunit" if not slug or slug == "unit" else slug


def _resolve_pixel_calibration(
    pixel_size: float | None, length_unit: str
) -> dict[str, object] | None:
    """Return physical-area calibration metadata for a positive pixel size.

    The program never guesses spatial calibration; a physical pixel size must be
    supplied explicitly (e.g. from the microscope), after which object and image
    areas can be reported in real units alongside the pixel counts.
    """

    if pixel_size is None:
        return None
    if isinstance(pixel_size, (bool, np.bool_)):
        raise TypeError(T("The pixel size must be a finite number greater than 0"))
    try:
        size = float(pixel_size)
    except (TypeError, ValueError) as exc:
        raise TypeError(T("The pixel size must be a finite number greater than 0")) from exc
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError(T("The pixel size must be a finite number greater than 0"))
    pixel_area = size * size
    if not math.isfinite(pixel_area) or pixel_area <= 0.0:
        raise ValueError(T(
            "The pixel size is too large or too small to compute physical areas reliably",
        ))
    if not isinstance(length_unit, str):
        raise TypeError(T("The unit of length must be a string"))
    normalized_unit = length_unit.strip()
    if not normalized_unit:
        raise ValueError(T("When a pixel size is given, the unit of length must not be empty"))
    if len(normalized_unit) > 32 or any(
        ord(character) < 32 for character in normalized_unit
    ):
        raise ValueError(T(
            "The unit of length must not exceed 32 characters or contain control characters",
        ))
    slug = _length_unit_slug(normalized_unit)
    return {
        "pixel_size": size,
        "unit": normalized_unit,
        "unit_slug": slug,
        "pixel_area": pixel_area,
        "area_column": f"area_{slug}2",
        "positive_area_column": f"positive_area_{slug}2",
        "pixel_size_column": f"pixel_size_{slug}",
    }


def export_result(
    loaded: LoadedImage,
    result: UniversalResult,
    output_root: str | Path,
    *,
    quick_settings: Mapping[str, object] | None = None,
    quick_base_params: UniversalParams | None = None,
    artifacts: Iterable[str] | None = None,
    outline_expand_px: int = 0,
    merge_base_mode: str = MERGE_BASE_MODE_ANALYSIS_INPUT,
    merge_source_loaded: LoadedImage | None = None,
    preserve_merge_original_values: bool = False,
    pixel_size: float | None = None,
    length_unit: str = "µm",
    allowed_output_root: str | Path | None = None,
    bundle_mode: str = EXPORT_BUNDLE_DIRECTORY,
) -> Path | ExportReceipt:
    """Write selected artifacts as a legacy directory or prefixed sibling files.

    ``allowed_output_root`` may pin a canonical containment root that was
    frozen when a long-running task was planned.  When provided, the actual
    destination is checked again immediately before directory creation and
    writing so a replaced symlink or Windows reparse point is rejected.

    ``directory`` preserves the historical return value and fixed leaf names.
    ``prefixed_files`` returns an :class:`ExportReceipt`; no per-image result
    directory remains after the staged files are exclusively committed.
    """

    requested_artifacts = _normalize_export_artifacts(artifacts)
    frozen_outline_expand_px = _normalize_outline_expand_px(outline_expand_px)
    frozen_merge_base_mode = normalize_merge_base_mode(merge_base_mode)
    if not isinstance(preserve_merge_original_values, (bool, np.bool_)):
        raise TypeError(T("\"Keep original Merge brightness\" must be a boolean"))
    frozen_preserve_merge_values = bool(preserve_merge_original_values)
    frozen_bundle_mode = normalize_export_bundle_mode(bundle_mode)
    calibration = _resolve_pixel_calibration(pixel_size, length_unit)
    if (quick_settings is None) != (quick_base_params is None):
        raise ValueError(T(
            "Exporting a quick recipe needs both the quick settings and the professional base.",
        ))
    if result.is_preview:
        raise ValueError(T(
            "A preview result cannot be exported; wait for the full-resolution run to finish",
        ))
    _verify_source(loaded)
    if result.ai_metadata is not None:
        if result.params.recognition_engine not in AI_RECOGNITION_ENGINES:
            raise ValueError(T(
                "The recognition mode of the AI result does not match the current original image; "
                "check the saved mode.",
            ))
        reference_fingerprint = result.ai_metadata.get("reference_image")
        if not (isinstance(reference_fingerprint, Mapping) and reference_fingerprint.get("path")):
            # The reference channel is optional, and a result restored from an
            # export without its reference image has nothing left to verify.
            reference_fingerprint = None
        if reference_fingerprint is not None:
            _verify_file_fingerprint(Path(str(reference_fingerprint["path"])), reference_fingerprint, label=T("Reference-channel image"))
    has_render_artifacts = EXPORT_ARTIFACT_MERGE in requested_artifacts
    merge_render_loaded = loaded
    if (
        frozen_merge_base_mode == MERGE_BASE_MODE_SOURCE_ORIGINAL
        and has_render_artifacts
    ):
        merge_render_loaded = _validate_original_merge_source(
            loaded,
            merge_source_loaded,
        )
    root = _resolve_contained_output_parent(output_root, allowed_output_root)
    root.mkdir(parents=True, exist_ok=True)
    # Recheck after mkdir as well: a pre-existing ancestor could have been
    # swapped between the first validation and directory creation.
    root = _resolve_contained_output_parent(root, allowed_output_root)
    bundle_prefix: str | None = None
    committed_files: list[Path] = []
    final_incomplete_marker: Path | None = None
    if frozen_bundle_mode == EXPORT_BUNDLE_DIRECTORY:
        output = _unique_output_directory(root, loaded.path)
        _resolve_contained_output_parent(output, allowed_output_root)
        incomplete = output / "INCOMPLETE.txt"
        incomplete.write_text(
            T("The export is not finished; do not treat this folder as a complete result."),
            encoding="utf-8",
        )
    else:
        bundle_prefix, output = _reserve_output_bundle(
            root,
            loaded.path,
            allowed_output_root=allowed_output_root,
        )
        try:
            _resolve_contained_output_parent(output, allowed_output_root)
        except Exception:
            try:
                _cleanup_export_staging(output)
            except OSError:
                pass
            raise
        incomplete = output / "INCOMPLETE.txt"

    def final_filename(leaf_name: str) -> str:
        if bundle_prefix is None:
            return leaf_name
        return _prefixed_export_filename(bundle_prefix, leaf_name)

    try:
        if result.ai_metadata is not None:
            _ai_module = ai_engine_module(result.params.recognition_engine)
            render_review_rgb = _ai_module.render_review_rgb
            render_original_ants16 = _ai_module.render_original_ants16
            if result.review_mask is None or result.review_base_mask is None:
                raise ValueError(T(
                    "The AI result lacks its review selection or review base; an incomplete "
                    "result cannot be exported.",
                ))
            if result.review_mask.shape != result.mask.shape or result.review_base_mask.shape != result.mask.shape:
                raise ValueError(T(
                    "The AI main selection and the review selection differ in size.",
                ))
            review_mask = np.asarray(result.review_base_mask, dtype=bool) & ~result.mask
            _write_image(output / "ai_review_mask.tif", review_mask.astype(np.uint8) * 255)
            _write_image(output / "ai_review_base_mask.tif", np.asarray(result.review_base_mask, dtype=np.uint8) * 255)
            _write_image(output / "ai_review_overlay.png", cv2.cvtColor(render_review_rgb(loaded, result), cv2.COLOR_RGB2BGR))
            _write_image(output / "ai_ants_16bit.tif", render_original_ants16(loaded, adjust_mask_boundary(result.mask, frozen_outline_expand_px)))
            public_ai_metadata = copy.deepcopy(result.ai_metadata)
            for image_key in ("source_image", "reference_image"):
                public_ai_metadata[image_key] = public_image_record(public_ai_metadata.get(image_key))
            public_ai_metadata["main_area_px"] = int(result.mask.sum())
            public_ai_metadata["review_area_px"] = int(review_mask.sum())
            public_ai_metadata["review_in_main_quantification"] = False
            ai_record = {
                "schema_version": 1, "ai_metadata": public_ai_metadata,
                "params": params_to_dict(result.params), "source_shape": list(loaded.shape),
                "main_mask_file": final_filename("positive_mask.tif") if EXPORT_ARTIFACT_POSITIVE_MASK in requested_artifacts else None,
                "review_mask_file": final_filename("ai_review_mask.tif"),
                "review_base_mask_file": final_filename("ai_review_base_mask.tif"),
                "review_overlay_file": final_filename("ai_review_overlay.png"),
                "ants_16bit_file": final_filename("ai_ants_16bit.tif"),
                "manually_edited": result.manually_edited,
                "edge_offset_px": result.edge_offset_px, "edge_offset_grid_px": result.edge_offset_grid_px,
                "display_legend": (
                    T(
                        "Green = main selection; yellow = review area (score within 1.5 below the "
                        "main threshold). Yellow is not counted in the main objects or the main "
                        "area.",
                    )
                ),
            }
            (output / "ai_analysis.json").write_text(json.dumps(ai_record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if EXPORT_ARTIFACT_POSITIVE_MASK in requested_artifacts:
            mask8 = result.mask.astype(np.uint8) * 255
            _write_image(output / "positive_mask.tif", mask8)

        labels_filename: str | None = None
        if EXPORT_ARTIFACT_OBJECT_LABELS in requested_artifacts:
            max_label = int(result.labels.max(initial=0))
            if max_label <= 65535:
                labels_filename = "object_labels.tif"
                _write_image(output / labels_filename, result.labels.astype(np.uint16))
            else:
                labels_filename = "object_labels.npy"
                np.save(output / labels_filename, result.labels.astype(np.int32))

        display_channel = display_channel_for_signal_mode(result.params.signal_mode)
        display_label = {
            "rgb": T("RGB composite"),
            "r": T("Red channel"),
            "g": T("Green channel"),
            "b": T("Blue channel"),
        }[display_channel]
        preprocessing_payload = loaded.metadata.get("analysis_preprocessing")
        has_crosstalk_correction = bool(
            isinstance(preprocessing_payload, Mapping)
            and preprocessing_payload.get("type") == "crosstalk_subtraction"
        )
        display_input_label = (
            T("Original RGB composite without automatic stretching")
            if frozen_preserve_merge_values
            else
            T("{0} of the original image", display_label)
            if frozen_merge_base_mode == MERGE_BASE_MODE_SOURCE_ORIGINAL
            else (
                T("{0} after crosstalk correction", display_label)
                if has_crosstalk_correction
                else T("Full-resolution {0}", display_label)
            )
        )
        if has_render_artifacts:
            if frozen_preserve_merge_values:
                base_rgb, _ = image_to_rgb_original_values(
                    merge_render_loaded.image,
                    max_side=None,
                )
            else:
                base_rgb, _ = image_to_signal_rgb_display(
                    merge_render_loaded.image,
                    result.params.signal_mode,
                    max_side=None,
                )
            outline_mask = adjust_mask_boundary(
                result.mask,
                frozen_outline_expand_px,
            )
            selection_merge_rgb = render_selection_merge(base_rgb, outline_mask)
            selection_merge_bgr = cv2.cvtColor(
                selection_merge_rgb,
                cv2.COLOR_RGB2BGR,
            )
            _write_image(output / MERGE_IMAGE_FILENAME, selection_merge_bgr)

        if EXPORT_ARTIFACT_BACKGROUND_CORRECTED_RESPONSE in requested_artifacts:
            response8 = normalize_for_display(result.response, gamma=0.85)
            _write_image(output / "background_corrected_response.png", response8)

        quality_artifacts = {
            EXPORT_ARTIFACT_QUALITY_REPORT,
            EXPORT_ARTIFACT_SUMMARY,
            EXPORT_ARTIFACT_PROJECT,
        }
        quality = None
        if requested_artifacts.intersection(quality_artifacts):
            quality = assess_segmentation(
                result.mask,
                result.objects,
                signal_mode_used=result.signal_mode_used,
                structure_method=result.params.background_method,
                confidence_floor=result.params.min_response_confidence,
                extra_warnings=result.intensity_warnings,
                force_warning=bool(
                    result.input_saturation_fraction >= 0.10
                    or result.input_high_plateau_fraction >= 0.10
                    or result.input_largest_tie_fraction >= 0.50
                ),
            )
        if EXPORT_ARTIFACT_QUALITY_REPORT in requested_artifacts:
            assert quality is not None
            (output / "quality_report.json").write_text(
                json.dumps(
                    quality.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

        # For a signal-free (all-zero) image the threshold is +inf.  `Infinity` is
        # not valid JSON (RFC 8259) and "inf" is not a portable CSV number, so emit
        # an empty / null field that strict external consumers can still parse.
        safe_threshold_value = (
            result.threshold_value if math.isfinite(result.threshold_value) else None
        )
        safe_low_threshold = (
            result.low_threshold if math.isfinite(result.low_threshold) else None
        )

        if EXPORT_ARTIFACT_OBJECTS in requested_artifacts:
            object_columns = [
                "id",
                "area_px",
                "perimeter_px",
                "outer_perimeter_px",
                "circularity",
                "outer_circularity",
                "major_axis_px",
                "minor_axis_px",
                "mean_thickness_px",
                "max_thickness_px",
                "elongation_index",
                "aspect_ratio",
                "eccentricity",
                "solidity",
                "outer_solidity",
                "hole_fraction",
                "max_hole_fraction",
                "hole_count",
                "shape_class",
                "shape_tags",
                "centroid_x_px",
                "centroid_y_px",
                "bbox_x_px",
                "bbox_y_px",
                "bbox_width_px",
                "bbox_height_px",
                "touches_border",
                "raw_mean",
                "raw_max",
                "raw_integrated",
                "normalized_mean",
                "normalized_max",
                "response_mean",
                "response_max",
                "response_confidence",
                "saturated_fraction",
                "contributing_branch_ids",
                "manual_provenance",
            ]
            if calibration is not None:
                object_columns.insert(
                    object_columns.index("area_px") + 1,
                    str(calibration["area_column"]),
                )
            with (output / "objects.csv").open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=object_columns,
                    extrasaction="ignore",
                )
                writer.writeheader()
                for record in result.objects:
                    row = {
                        **record,
                        "contributing_branch_ids": ";".join(
                            str(item)
                            for item in record.get("contributing_branch_ids", [])
                        ),
                    }
                    if calibration is not None:
                        row[str(calibration["area_column"])] = (
                            float(record.get("area_px", 0) or 0)
                            * float(calibration["pixel_area"])
                        )
                    writer.writerow(row)

        if EXPORT_ARTIFACT_SUMMARY in requested_artifacts:
            assert quality is not None
            with (output / "summary.csv").open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                summary_header = [
                    "source_file",
                    "width_px",
                    "height_px",
                    "object_count",
                    "positive_area_px",
                    "positive_fraction",
                    "positive_percent",
                    "threshold",
                    "low_threshold",
                    "signal_mode_used",
                    "display_channel",
                    "processing_seconds",
                    "manually_edited",
                    "quality_level",
                    "quality_warning_count",
                    "largest_object_fraction",
                    "border_area_fraction",
                    "small_object_fraction",
                    "saturated_object_fraction",
                    "response_confidence_median",
                    "edge_offset_px",
                ]
                summary_row = [
                    _csv_safe(loaded.path.name),
                    loaded.shape[1],
                    loaded.shape[0],
                    result.object_count,
                    int(result.mask.sum()),
                    result.positive_fraction,
                    result.positive_fraction * 100.0,
                    safe_threshold_value,
                    safe_low_threshold,
                    result.signal_mode_used,
                    display_channel,
                    result.processing_seconds,
                    result.manually_edited,
                    quality.level,
                    len(quality.warnings),
                    quality.largest_object_fraction,
                    quality.border_area_fraction,
                    quality.small_object_fraction,
                    quality.saturated_object_fraction,
                    quality.response_confidence_median,
                    int(getattr(result, "edge_offset_px", 0)),
                ]
                if result.ai_metadata is not None:
                    summary_header += ["recognition_engine", "ai_main_threshold", "ai_review_threshold", "review_area_px", "review_in_main_quantification"]
                    summary_row += [result.params.recognition_engine,
                                    result.ai_metadata.get("main_threshold", result.threshold_value),
                                    result.ai_metadata.get("review_threshold", result.low_threshold),
                                    int(result.review_mask.sum()) if result.review_mask is not None else 0, False]
                if calibration is not None:
                    summary_header += [
                        str(calibration["pixel_size_column"]),
                        "pixel_size_unit",
                        str(calibration["positive_area_column"]),
                    ]
                    summary_row += [
                        float(calibration["pixel_size"]),
                        _csv_safe(calibration["unit"]),
                        float(int(result.mask.sum()) * float(calibration["pixel_area"])),
                    ]
                writer.writerow(summary_header)
                writer.writerow(summary_row)

        if EXPORT_ARTIFACT_RECIPE in requested_artifacts:
            save_recipe(
                output / "recipe.json",
                result.params,
                name=T("Recipe used for the export"),
                description=T(
                    "Complete, reproducible parameters exported by FluoroAnts Universal.",
                ),
                quick_settings=quick_settings,
                quick_base_params=quick_base_params,
                analysis_preprocessing=(
                    preprocessing_payload
                    if isinstance(preprocessing_payload, Mapping)
                    else None
                ),
            )

        if EXPORT_ARTIFACT_PROJECT in requested_artifacts:
            assert quality is not None
            outline_display_description = (
                T(
                    "The black-and-white outline follows the boundary of the final mask exactly; "
                    "the mask, the object statistics and the quality results are unchanged.",
                )
                if frozen_outline_expand_px == 0
                else (
                    T(
                        "The black-and-white outline is for review only and is drawn {0} source "
                        "pixels outside the final mask; the mask, the object statistics and the "
                        "quality results are unchanged.",
                        frozen_outline_expand_px,
                    )
                )
            )
            artifact_manifest: dict[str, object] = {}
            if EXPORT_ARTIFACT_MERGE in requested_artifacts:
                artifact_manifest["merge"] = {
                    "file": final_filename(MERGE_IMAGE_FILENAME),
                    "description": (
                        T(
                            "{0} as the base, overlaid with the black-and-white selection outline "
                            "derived from the final mask (outline drawn {1} px outside); no "
                            "translucent fill, no object numbers.",
                            display_input_label,
                            frozen_outline_expand_px,
                        )
                    ),
                }
            if EXPORT_ARTIFACT_POSITIVE_MASK in requested_artifacts:
                artifact_manifest["positive_mask"] = final_filename(
                    "positive_mask.tif"
                )
            if EXPORT_ARTIFACT_OBJECT_LABELS in requested_artifacts:
                assert labels_filename is not None
                artifact_manifest["object_labels"] = final_filename(labels_filename)
            if EXPORT_ARTIFACT_BACKGROUND_CORRECTED_RESPONSE in requested_artifacts:
                artifact_manifest["background_corrected_response"] = (
                    final_filename("background_corrected_response.png")
                )
            if EXPORT_ARTIFACT_QUALITY_REPORT in requested_artifacts:
                artifact_manifest["quality_report"] = final_filename(
                    "quality_report.json"
                )
            if EXPORT_ARTIFACT_OBJECTS in requested_artifacts:
                artifact_manifest["objects"] = final_filename("objects.csv")
            if EXPORT_ARTIFACT_SUMMARY in requested_artifacts:
                artifact_manifest["summary"] = final_filename("summary.csv")
            if EXPORT_ARTIFACT_RECIPE in requested_artifacts:
                artifact_manifest["recipe"] = final_filename("recipe.json")

            project = {
                "application": "FluoroAnts Universal",
                "release_version": FLUOROANTS_RELEASE_VERSION,
                "algorithm_version": UNIVERSAL_ALGORITHM_VERSION,
                "export_display": {
                    "outline_expand_px": frozen_outline_expand_px,
                    "merge_base_mode": frozen_merge_base_mode,
                    "preserve_original_channel_values": (
                        frozen_preserve_merge_values
                    ),
                    "merge_background_used": (
                        None
                        if not has_render_artifacts
                        else (
                            "source_original"
                            if frozen_merge_base_mode
                            == MERGE_BASE_MODE_SOURCE_ORIGINAL
                            or not has_crosstalk_correction
                            else "crosstalk_corrected"
                        )
                    ),
                    "description": outline_display_description,
                },
                "source": {
                    "file_name": loaded.path.name,
                    "size_bytes": int(loaded.metadata["size"]),
                    "dtype": str(loaded.metadata["dtype"]),
                    "shape": list(loaded.metadata["shape"]),
                },
                "result": {
                    "object_count": result.object_count,
                    "positive_fraction": result.positive_fraction,
                    "threshold_value": safe_threshold_value,
                    "low_threshold": safe_low_threshold,
                    "signal_mode_used": result.signal_mode_used,
                    "display_channel": display_channel,
                    "processing_seconds": result.processing_seconds,
                    "manually_edited": result.manually_edited,
                    # Post-recognition boundary offset (source pixels); it is a
                    # display/export setting, so it is documented here and not
                    # inside the recipe parameters.
                    "selection_edge_offset_px": int(getattr(result, "edge_offset_px", 0)),
                    "intensity_diagnostics": {
                        "saturation_fraction": result.input_saturation_fraction,
                        "high_plateau_fraction": result.input_high_plateau_fraction,
                        "largest_tie_fraction": result.input_largest_tie_fraction,
                        "unique_levels": result.input_unique_levels,
                        "warnings": list(result.intensity_warnings),
                    },
                    "quality": quality.to_dict(),
                },
                "params": params_to_dict(result.params),
                "artifacts": artifact_manifest,
            }
            if calibration is not None:
                project["calibration"] = {
                    "pixel_size": float(calibration["pixel_size"]),
                    "unit": str(calibration["unit"]),
                    "pixel_area": float(calibration["pixel_area"]),
                    "positive_area": float(
                        int(result.mask.sum()) * float(calibration["pixel_area"])
                    ),
                    "note": T(
                        "Physical areas are computed from the pixel size given by the user; the "
                        "program does not calibrate automatically.",
                    ),
                }
            if result.ai_metadata is not None:
                project["ai_analysis"] = public_ai_metadata
                project["result"]["review_area_px"] = int(result.review_mask.sum()) if result.review_mask is not None else 0
                project["result"]["review_in_main_quantification"] = False
                project["artifacts"]["ai_analysis"] = final_filename("ai_analysis.json")
                project["artifacts"]["ai_review_mask"] = final_filename("ai_review_mask.tif")
                project["artifacts"]["ai_review_base_mask"] = final_filename("ai_review_base_mask.tif")
                project["artifacts"]["ai_review_overlay"] = final_filename("ai_review_overlay.png")
                project["artifacts"]["ai_ants_16bit"] = final_filename("ai_ants_16bit.tif")
            if isinstance(preprocessing_payload, Mapping):
                public_preprocessing_keys = (
                    "type",
                    "version",
                    "formula",
                    "coefficient",
                    "target_signal_mode_requested",
                    "target_signal_mode_used",
                    "reference_file_name",
                    "reference_signal_mode_requested",
                    "reference_signal_mode_used",
                    "preserve_reference_background",
                    "reference_background_percentile",
                    "reference_background_normalized",
                    "zero_clipped_fraction",
                    "size_policy",
                )
                project["analysis_preprocessing"] = {
                    key: preprocessing_payload[key]
                    for key in public_preprocessing_keys
                    if key in preprocessing_payload
                }
            if quick_settings is not None and quick_base_params is not None:
                compiled_plan = getattr(
                    result,
                    "compiled_plan",
                    getattr(result, "quick_plan", None),
                )
                if hasattr(compiled_plan, "to_dict"):
                    compiled_plan = compiled_plan.to_dict()
                project.update(
                    {
                        "mode": "quick",
                        "quick_schema_version": QUICK_SCHEMA_VERSION,
                        "quick_settings": dict(quick_settings),
                        "quick_base_params": params_to_dict(quick_base_params),
                        "quick_variant_count": int(
                            getattr(result, "quick_variant_count", 1)
                        ),
                        "compiled_quick_plan": compiled_plan,
                    }
                )
            else:
                project["mode"] = "professional"
            (output / "project.json").write_text(
                json.dumps(project, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        _verify_source(loaded)
        if result.ai_metadata is not None and reference_fingerprint is not None:
            _verify_file_fingerprint(Path(str(reference_fingerprint["path"])), reference_fingerprint, label=T("Reference-channel image"))
        if bundle_prefix is None:
            incomplete.unlink(missing_ok=True)
            return output

        root = _resolve_contained_output_parent(root, allowed_output_root)
        output = _resolve_contained_output_parent(output, allowed_output_root)
        if not output.is_dir() or not incomplete.is_file():
            raise RuntimeError(T("The temporary export folder changed before it was committed"))
        if _bundle_prefix_in_use(root, bundle_prefix):
            raise FileExistsError(T(
                "The export file prefix was taken by another task while this one was running",
            ))
        staged_files = sorted(
            (path for path in output.iterdir() if path != incomplete),
            key=lambda path: (path.name == "project.json", path.name.casefold()),
        )
        final_incomplete_marker = root / final_filename("INCOMPLETE.txt")
        with final_incomplete_marker.open("x", encoding="utf-8") as marker_handle:
            marker_handle.write(
                T(
                    "Result files are being committed; if the program is interrupted, do not "
                    "treat files with this prefix as a complete result.",
                )
            )
            marker_handle.flush()
            os.fsync(marker_handle.fileno())
        for staged_path in staged_files:
            if not staged_path.is_file() or staged_path.is_symlink():
                raise RuntimeError(T("The temporary export folder contains an invalid item"))
            final_path = root / final_filename(staged_path.name)
            _commit_file_exclusive(staged_path, final_path)
            committed_files.append(final_path)
        _cleanup_export_staging(output)
        final_incomplete_marker.unlink()
        final_incomplete_marker = None
        return ExportReceipt(
            parent=root,
            prefix=bundle_prefix,
            files=tuple(committed_files),
        )
    except Exception:
        if bundle_prefix is not None:
            for committed_path in committed_files:
                try:
                    committed_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if final_incomplete_marker is not None:
                try:
                    final_incomplete_marker.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                _cleanup_export_staging(output)
            except OSError:
                pass
        raise


def batch_process(
    image_paths: Iterable[str | Path],
    params: UniversalParams,
    output_root: str | Path | None,
    *,
    relative_paths: Mapping[str | Path, str | Path] | None = None,
    export_location_mode: str = EXPORT_LOCATION_FLAT,
    export_bundle_mode: str = EXPORT_BUNDLE_DIRECTORY,
    params_for_shape: Callable[[tuple[int, int]], UniversalParams] | None = None,
    segmenter: Callable[[LoadedImage, UniversalParams], UniversalResult] | None = None,
    export_quick_settings: Mapping[str, object] | None = None,
    export_quick_base_params: UniversalParams | None = None,
    export_artifacts: Iterable[str] | None = None,
    outline_expand_px: int = 0,
    pixel_size: float | None = None,
    length_unit: str = "µm",
    edge_offset_px: int = 0,
) -> dict[str, list[dict[str, object]]]:
    """Segment and export multiple images with an optional custom segmenter.

    ``segmenter`` receives the loaded image and the parameters selected for its
    shape.  Leaving it unset preserves the original single-branch workflow.
    Keeping this as a callback avoids importing higher-level quick-setting
    orchestration into the core module.  ``flat`` remains the API-compatible
    default; callers can pass ``mirror`` plus per-image ``relative_paths`` or
    ``source`` to place each result beside its original image.  The API keeps
    the historical ``directory`` bundle default; desktop and CLI callers use
    ``prefixed_files`` to avoid an extra per-image folder.  Destination
    configuration is preflighted for the whole batch and raises immediately;
    per-image load, segmentation, and export errors are collected in
    ``summary["failures"]``. In prefixed-file mode each exported record keeps
    the compatibility ``output`` label, repeats it as ``output_prefix``, and
    lists the actual safe relative paths in ``files``.
    """

    # Freeze generators and mutable caller collections once so every batch item
    # receives exactly the same validated request and destination.
    frozen_paths = tuple(Path(value).expanduser().resolve() for value in image_paths)
    frozen_location_mode = normalize_export_location_mode(export_location_mode)
    frozen_bundle_mode = normalize_export_bundle_mode(export_bundle_mode)
    frozen_relative_map: dict[Path, Path] = {}
    for key, value in (relative_paths or {}).items():
        source_key = Path(key).expanduser().resolve()
        frozen_relative_map[source_key] = _normalize_export_relative_path(
            source_key,
            value,
        )
    frozen_relative_paths = {
        path: frozen_relative_map.get(path, Path(path.name)) for path in frozen_paths
    }
    frozen_output_root = (
        None if output_root is None else Path(output_root).expanduser().resolve()
    )
    if (
        frozen_location_mode != EXPORT_LOCATION_SOURCE
        and frozen_output_root is None
    ):
        raise ValueError(T("Flat or mirror export needs an output root folder"))
    output_parents = {
        path: resolve_export_parent(
            path,
            frozen_output_root,
            relative_path=frozen_relative_paths[path],
            location_mode=frozen_location_mode,
        )
        for path in frozen_paths
    }
    allowed_output_roots = {
        path: (
            path.parent
            if frozen_location_mode == EXPORT_LOCATION_SOURCE
            else frozen_output_root
        )
        for path in frozen_paths
    }
    frozen_export_artifacts = _normalize_export_artifacts(export_artifacts)
    frozen_outline_expand_px = _normalize_outline_expand_px(outline_expand_px)
    # Validate calibration once before reading or segmenting any image.  A bad
    # batch option must fail fast and must not create a partial output folder.
    _resolve_pixel_calibration(pixel_size, length_unit)
    if isinstance(edge_offset_px, (bool, np.bool_)):
        raise TypeError(T("The edge distance must be a whole number of pixels"))
    frozen_edge_offset = int(operator.index(edge_offset_px))
    if abs(frozen_edge_offset) > MAX_EDGE_OFFSET_PX:
        raise ValueError(T("The edge distance must be within ±{0} pixels", MAX_EDGE_OFFSET_PX))
    exported: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for path in frozen_paths:
        try:
            loaded = load_image(path)
            current_params = (
                params_for_shape(loaded.shape) if params_for_shape is not None else params
            )
            result = (
                segment_loaded_image(loaded, current_params, max_side=None)
                if segmenter is None
                else segmenter(loaded, current_params)
            )
            if frozen_edge_offset:
                result = apply_edge_offset(result, frozen_edge_offset)
            output = export_result(
                loaded,
                result,
                output_parents[path],
                quick_settings=export_quick_settings,
                quick_base_params=export_quick_base_params,
                artifacts=frozen_export_artifacts,
                outline_expand_px=frozen_outline_expand_px,
                pixel_size=pixel_size,
                length_unit=length_unit,
                allowed_output_root=allowed_output_roots[path],
                bundle_mode=frozen_bundle_mode,
            )
            source_label = frozen_relative_paths[path].as_posix()
            if isinstance(output, ExportReceipt):
                output_label_path = output.label_path
                output_label = output.prefix
                if (
                    frozen_location_mode != EXPORT_LOCATION_SOURCE
                    and frozen_output_root is not None
                ):
                    output_label = output_label_path.relative_to(
                        frozen_output_root
                    ).as_posix()
                file_labels = (
                    [path.name for path in output.files]
                    if frozen_location_mode == EXPORT_LOCATION_SOURCE
                    or frozen_output_root is None
                    else [
                        path.relative_to(frozen_output_root).as_posix()
                        for path in output.files
                    ]
                )
                exported.append(
                    {
                        "source": source_label,
                        # Kept for compatibility with the historical summary;
                        # in file-bundle mode this is a logical prefix label.
                        "output": output_label,
                        "output_prefix": output_label,
                        "files": file_labels,
                    }
                )
            else:
                output_label = output.name
                if (
                    frozen_location_mode != EXPORT_LOCATION_SOURCE
                    and frozen_output_root is not None
                ):
                    output_label = output.relative_to(frozen_output_root).as_posix()
                exported.append({"source": source_label, "output": output_label})
        except Exception as exc:
            # Scrub absolute paths at the source so the privacy invariant holds
            # regardless of how the caller surfaces the failure message.
            failures.append(
                {
                    "source": frozen_relative_paths[path].as_posix(),
                    "error": scrub_local_paths(str(exc)),
                }
            )
    return {"exported": exported, "failures": failures}

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from itertools import product

import numpy as np

from .i18n import T
from .universal_core import (
    AI_RECOGNITION_ENGINES,
    BUILTIN_PRESETS,
    LoadedImage,
    UniversalParams,
    UniversalResult,
    _plain_component_labels_with_boxes,
    _rescale_preview_objects,
    enforce_shape_compatibility,
    measure_and_filter_labels,
    segment_loaded_image,
)


QUICK_LOGIC_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "balanced": T("General balanced"),
    "strict": T("Strict positive"),
    "low_fp": T("Very low false positives"),
    "texture_guard": T("Texture-background suppression"),
    "sparse_spots": T("Sparse tiny spots"),
    "dense_spots": T("Dense tiny spots"),
    "low_contrast": T("Low contrast"),
}

_LOGIC_PRESETS: dict[str, str] = {
    "balanced": "General balanced",
    "strict": "Strict positive",
    "low_fp": "Very low false positives",
    "sparse_spots": "Sparse tiny spots",
    "dense_spots": "Dense tiny spots",
    "low_contrast": "Low contrast",
}

QUICK_MULTI_FIELDS = frozenset({"logic", "shape", "size"})
MAX_QUICK_VARIANTS = 12

QUICK_SIGNAL_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "auto": T("Auto"),
    "luma": T("Luminance"),
    "max_rgb": T("Brightest colour"),
    "red": T("Red signal"),
    "green": T("Green signal"),
    "blue": T("Blue signal"),
    "hsv_saturation": T("Colour saturation"),
}

QUICK_POLARITY_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "bright": T("Bright targets"),
    "dark": T("Dark targets"),
}

QUICK_SENSITIVITY_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "very_strict": T("Very strict"),
    "strict": T("Rather strict"),
    "balanced": T("Balanced"),
    "sensitive": T("Rather sensitive"),
    "very_sensitive": T("Very sensitive"),
}

QUICK_SIZE_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "any": T("Any size"),
    "tiny": T("Tiny"),
    "small": T("Small"),
    "medium": T("Medium"),
    "large": T("Large"),
}

QUICK_SHAPE_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "any": T("Any / mixed"),
    "round": T("Round / compact"),
    "ellipse": T("Elliptic / spindle"),
    "fiber": T("Elongated / strip / fibre"),
    "patch": T("Patch / irregular"),
    "ring": T("Ring / keep inner holes"),
}

QUICK_IMAGING_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "adaptive": T("Adaptive SNR (recommended)"),
    "normal": T("Normal"),
    "dim": T("Dim overall"),
    "low_contrast": T("Low contrast"),
    "overexposed": T("Overexposed"),
}

QUICK_SCALE_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "relative_grid": T("Relative to the frame (recommended)"),
    "pixel_locked": T("Fixed source pixels"),
}

QUICK_BACKGROUND_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "none": T("No correction"),
    "normal": T("Ordinary background"),
    "gradient": T("Slow gradient"),
    "uneven": T("Strongly uneven"),
    "high": T("High background fluorescence"),
    "diffuse": T("Diffuse background"),
}

QUICK_NOISE_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "off": T("Off"),
    "detail": T("Keep fine detail"),
    "light": T("Light"),
    "balanced": T("Balanced"),
    "granular": T("Granular noise"),
    "strong": T("Strong"),
}

QUICK_EDGE_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "core": T("Clear cores only"),
    "tight": T("Tight boundary"),
    "balanced": T("Balanced"),
    "wide": T("Wide boundary"),
    "full": T("Keep the full outline"),
}

QUICK_SPECK_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "off": T("No extra cleanup"),
    "light": T("Light cleanup"),
    "balanced": T("Balanced cleanup"),
    "strong": T("Strong cleanup"),
}

QUICK_OPENING_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "off": T("Off"),
    "light": T("Light despeckle"),
    "strong": T("Strong despeckle"),
}

QUICK_GAP_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "off": T("Do not connect"),
    "light": T("Connect small gaps"),
    "medium": T("Connect medium gaps"),
    "strong": T("Connect larger gaps"),
}

QUICK_CONTOUR_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "shrink_2": T("Shrink clearly"),
    "shrink_1": T("Shrink slightly"),
    "none": T("No adjustment"),
    "grow_1": T("Grow slightly"),
    "grow_2": T("Grow clearly"),
}

QUICK_HOLE_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "keep": T("Keep holes"),
    "small": T("Fill small holes only"),
    "medium": T("Fill medium holes"),
    "large": T("Fill holes where possible"),
}

QUICK_TOUCHING_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "keep": T("Keep whole"),
    "conservative": T("Separate conservatively"),
    "balanced": T("Separate moderately"),
    "strong": T("Separate where possible"),
}

QUICK_BORDER_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "keep": T("Keep border objects"),
    "remove": T("Remove border objects"),
}

QUICK_INTENSITY_LABELS: dict[str, str] = {
    "follow": T("Follow preset"),
    "none": T("No restriction"),
    "weak": T("Keep weak signal too"),
    "balanced": T("Balanced"),
    "clear": T("Clear positives only"),
    "strict": T("Extremely strict"),
}

QUICK_SHAPE_STRICTNESS_LABELS: dict[str, str] = {
    "follow": T("Default (standard)"),
    "loose": T("Loose"),
    "standard": T("Standard"),
    "strict": T("Strict"),
}


@dataclass(frozen=True)
class QuickSettings:
    """Semantic overrides layered on a complete professional parameter set."""

    logic: str = "follow"
    signal: str = "follow"
    polarity: str = "follow"
    sensitivity: str = "follow"
    size: str = "follow"
    shape: str = "follow"
    # New sessions use an effect-size gate which is allowed to return an empty
    # mask.  ``quick_settings_from_dict`` supplies ``follow`` for absent keys.
    imaging: str = "adaptive"
    scale: str = "relative_grid"
    background: str = "follow"
    noise: str = "follow"
    edge: str = "follow"
    speck_cleanup: str = "follow"
    opening: str = "follow"
    gap_connect: str = "follow"
    contour_adjust: str = "follow"
    holes: str = "follow"
    touching: str = "follow"
    border: str = "follow"
    intensity: str = "follow"
    shape_strictness: str = "follow"

    @property
    def override_count(self) -> int:
        total = 0
        for name, value in asdict(self).items():
            selected = quick_setting_values(name, value)
            if selected != ("follow",):
                total += len(selected)
        return total


QUICK_CHOICE_LABELS: dict[str, Mapping[str, str]] = {
    "logic": QUICK_LOGIC_LABELS,
    "signal": QUICK_SIGNAL_LABELS,
    "polarity": QUICK_POLARITY_LABELS,
    "sensitivity": QUICK_SENSITIVITY_LABELS,
    "size": QUICK_SIZE_LABELS,
    "shape": QUICK_SHAPE_LABELS,
    "imaging": QUICK_IMAGING_LABELS,
    "scale": QUICK_SCALE_LABELS,
    "background": QUICK_BACKGROUND_LABELS,
    "noise": QUICK_NOISE_LABELS,
    "edge": QUICK_EDGE_LABELS,
    "speck_cleanup": QUICK_SPECK_LABELS,
    "opening": QUICK_OPENING_LABELS,
    "gap_connect": QUICK_GAP_LABELS,
    "contour_adjust": QUICK_CONTOUR_LABELS,
    "holes": QUICK_HOLE_LABELS,
    "touching": QUICK_TOUCHING_LABELS,
    "border": QUICK_BORDER_LABELS,
    "intensity": QUICK_INTENSITY_LABELS,
    "shape_strictness": QUICK_SHAPE_STRICTNESS_LABELS,
}

_CHOICES = QUICK_CHOICE_LABELS


@dataclass(frozen=True)
class QuickPlanIssue:
    """One stable, serializable planner diagnostic."""

    code: str
    severity: str
    message: str
    fields: tuple[str, ...] = ()
    branch_id: str = ""
    details: tuple[tuple[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "fields": list(self.fields),
            "branch_id": self.branch_id,
            "details": {key: value for key, value in self.details},
        }


@dataclass(frozen=True)
class QuickBranch:
    """One fully resolved scalar branch and the provenance of every override."""

    branch_id: str
    logic: str
    shape: str
    size: str
    params: UniversalParams
    origins: tuple[tuple[str, str, str], ...] = ()
    provenance: tuple[tuple[str, tuple[str, ...]], ...] = ()
    derivations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "logic": self.logic,
            "shape": self.shape,
            "size": self.size,
            "origins": [
                {"logic": logic, "shape": shape, "size": size}
                for logic, shape, size in self.origins
            ],
            "params": asdict(self.params),
            "provenance": {
                key: list(sources) for key, sources in self.provenance
            },
            "derivations": list(self.derivations),
        }


@dataclass(frozen=True)
class QuickPlan:
    """Conflict-aware, deterministic execution plan for one image shape."""

    settings: QuickSettings
    image_shape: tuple[int, int] | None
    branches: tuple[QuickBranch, ...]
    issues: tuple[QuickPlanIssue, ...]
    fingerprint: str

    @property
    def errors(self) -> tuple[QuickPlanIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[QuickPlanIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "settings": quick_settings_to_dict(self.settings),
            "image_shape": (
                list(self.image_shape) if self.image_shape is not None else None
            ),
            "branches": [branch.to_dict() for branch in self.branches],
            "issues": [issue.to_dict() for issue in self.issues],
            "fingerprint": self.fingerprint,
            "is_valid": self.is_valid,
        }

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        first = self.errors[0]
        suffix = T(" ({0} more conflict(s))", len(self.errors) - 1) if len(self.errors) > 1 else ""
        raise ValueError(f"{first.message}{suffix}")


def _identity_values(params: UniversalParams) -> dict[str, object]:
    """The parameters that define what a classic branch computes.

    The learned model's own setting takes no part in it; leaving it out keeps
    branch ids and plan fingerprints equal to those of earlier versions.
    """
    values = asdict(params)
    values.pop("learned_sensitivity", None)
    return values


def _identity_record(record: dict[str, object]) -> dict[str, object]:
    """A branch record as it enters the plan fingerprint (see ``_identity_values``)."""
    cleaned = dict(record)
    for key, value in record.items():
        if isinstance(value, dict) and "learned_sensitivity" in value:
            cleaned[key] = {k: v for k, v in value.items() if k != "learned_sensitivity"}
    return cleaned


def quick_setting_values(field_name: str, value: object) -> tuple[str, ...]:
    """Return one deterministic selection tuple; empty/none means follow."""

    choices = _CHOICES.get(field_name)
    if choices is None:
        raise ValueError(T("Unknown quick setting: {0}", field_name))
    if isinstance(value, str):
        raw_values = value.split("+") if field_name in QUICK_MULTI_FIELDS else [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = [str(item) for item in value]
    elif value is None:
        raw_values = []
    else:
        raw_values = [str(value)]
    requested = [str(item).strip() for item in raw_values if str(item).strip()]
    invalid = [item for item in requested if item not in choices]
    if invalid:
        raise ValueError(T("Invalid choice for quick setting {0}: {1}", field_name, invalid[0]))
    selected = tuple(key for key in choices if key in requested)
    if field_name not in QUICK_MULTI_FIELDS and len(selected) > 1:
        raise ValueError(T("Quick setting {0} allows one choice only.", field_name))
    if "follow" in selected and len(selected) > 1:
        selected = tuple(item for item in selected if item != "follow")
    if field_name in {"shape", "size"} and "any" in selected:
        return ("any",)
    return selected or ("follow",)


def quick_settings_to_dict(settings: QuickSettings) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in asdict(settings).items():
        selected = quick_setting_values(key, value)
        payload[key] = list(selected) if len(selected) > 1 else selected[0]
    return payload


def quick_settings_from_dict(payload: Mapping[str, object]) -> QuickSettings:
    allowed = {field.name for field in fields(QuickSettings)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(T("Unknown quick setting: {0}", ', '.join(unknown)))
    values: dict[str, str] = {}
    for name in _CHOICES:
        selected = quick_setting_values(name, payload.get(name, "follow"))
        values[name] = "+".join(selected)
    return QuickSettings(**values)


def _shape(image_shape: tuple[int, int] | None) -> tuple[int, int]:
    if image_shape is None:
        return 1024, 1024
    height, width = (max(1, int(value)) for value in image_shape[:2])
    return height, width


# Spatial values of the quick choices are canonical; the core materialises
# them once for the selected scale mode.
def _linear(value: float) -> int:
    if value <= 0:
        return 0
    return max(1, round(value))


def _area(value: float) -> int:
    if value <= 0:
        return 0
    return max(1, round(value))


def _odd(value: float) -> int:
    result = max(3, round(value))
    return result if result % 2 else result + 1


_SIZE_VALUES: dict[str, tuple[int, int, int]] = {
    "any": (0, 0, 24),
    "tiny": (2, 240, 10),
    "small": (8, 2000, 24),
    "medium": (25, 8000, 64),
    "large": (220, 0, 144),
}

_FIBER_THICKNESS_VALUES: dict[str, tuple[float, float]] = {
    "any": (0.0, 0.0),
    "tiny": (0.0, 7.0),
    "small": (4.0, 13.0),
    "medium": (10.0, 24.0),
    "large": (20.0, 0.0),
}

_SENSITIVITY_DELTAS = {
    "very_strict": -26,
    "strict": -14,
    "balanced": 0,
    "sensitive": 16,
    "very_sensitive": 28,
}

_EDGE_DELTAS = {
    "core": 0.22,
    "tight": 0.12,
    "balanced": 0.0,
    "wide": -0.11,
    "full": -0.20,
}

_HYSTERESIS_THRESHOLD_METHODS = {
    "snr_hysteresis",
    "hysteresis_otsu",
    "manual",
    "percentile",
    "local_gaussian",
    "local_mean",
    "mean_std",
    "sauvola",
}

_CONTINUOUS_SHAPES = {"fiber", "patch", "ring"}


def normalize_quick_settings(
    settings: QuickSettings,
    base_shape_profile: str = "any",
) -> tuple[QuickSettings, tuple[str, ...]]:
    """Canonicalize representation without changing the user's choices.

    Contradictory selections are not rewritten here; the planner preserves
    intent and reports conflicts from :func:`compile_quick_plan`.
    """

    del base_shape_profile
    return quick_settings_from_dict(quick_settings_to_dict(settings)), ()


def _write_params(
    params: UniversalParams,
    provenance: dict[str, list[str]],
    source: str,
    **changes: object,
) -> UniversalParams:
    """Replace supported fields and retain an ordered write history."""

    supported = {key: value for key, value in changes.items() if hasattr(params, key)}
    for key in supported:
        history = provenance.setdefault(key, [])
        if source not in history:
            history.append(source)
    return replace(params, **supported) if supported else params


def _constraint_intersection(
    base_minimum: float,
    base_maximum: float,
    requested_minimum: float,
    requested_maximum: float,
) -> tuple[float, float, bool]:
    minimum = max(float(base_minimum), float(requested_minimum))
    finite_maxima = [
        float(value) for value in (base_maximum, requested_maximum) if value > 0
    ]
    maximum = min(finite_maxima) if finite_maxima else 0.0
    return minimum, maximum, not maximum or minimum <= maximum


def _planner_issue(
    code: str,
    message: str,
    *field_names: str,
    severity: str = "error",
    details: Mapping[str, object] | None = None,
) -> QuickPlanIssue:
    return QuickPlanIssue(
        code=code,
        severity=severity,
        message=message,
        fields=tuple(field_names),
        branch_id="__branch__",
        details=tuple(sorted((details or {}).items())),
    )




def _compile_scalar_branch(
    base_params: UniversalParams,
    settings: QuickSettings,
    image_shape: tuple[int, int] | None,
    *,
    logic: str,
    shape: str,
    size: str,
    base_source: str,
) -> tuple[
    UniversalParams,
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[str, ...],
    tuple[QuickPlanIssue, ...],
]:
    """Compile one scalar branch without silently resolving contradictions."""

    for field_name in QUICK_MULTI_FIELDS:
        selected = quick_setting_values(field_name, getattr(settings, field_name))
        if len(selected) > 1:
            raise ValueError(T(
                "Quick setting {0} must first be expanded into single-choice branches.",
                field_name,
            ))

    height, width = _shape(image_shape)
    params = replace(base_params)
    provenance: dict[str, list[str]] = {
        item.name: [base_source] for item in fields(UniversalParams)
    }
    derivations: list[str] = []
    issues: list[QuickPlanIssue] = []

    if settings.scale != "follow":
        params = _write_params(
            params,
            provenance,
            f"scale:{settings.scale}",
            scale_mode=settings.scale,
            reference_short_side=1024,
        )

    if settings.signal != "follow":
        params = _write_params(
            params,
            provenance,
            f"signal:{settings.signal}",
            signal_mode=settings.signal,
            red_weight=1.0,
            green_weight=1.0,
            blue_weight=1.0,
        )
    if settings.polarity != "follow":
        params = _write_params(
            params,
            provenance,
            f"polarity:{settings.polarity}",
            invert=settings.polarity == "dark",
        )

    if settings.imaging == "adaptive":
        params = _write_params(
            params,
            provenance,
            "imaging:adaptive",
            intensity_space="robust_snr",
            threshold_method="snr_hysteresis",
            background_method="gaussian",
            background_strength=1.0,
            min_mean_intensity=0.0,
            min_response_confidence=0.0,
        )
    elif settings.imaging != "follow":
        imaging = {
            "normal": (0.2, 99.8, 1.00, "hysteresis_otsu", 0.20),
            "dim": (0.1, 99.9, 0.80, "hysteresis_otsu", 0.20),
            "low_contrast": (0.5, 99.5, 0.82, "mean_std", 0.35),
            "overexposed": (0.5, 98.8, 1.25, "hysteresis_otsu", 0.20),
        }[settings.imaging]
        params = _write_params(
            params,
            provenance,
            f"imaging:{settings.imaging}",
            intensity_space="percentile_linear",
            black_percentile=imaging[0],
            white_percentile=imaging[1],
            gamma=imaging[2],
            threshold_method=imaging[3],
            manual_threshold=0.25,
            threshold_percentile=98.5,
            local_window=_odd(61),
            local_k=imaging[4],
        )

    original_shape = params.shape_profile
    if settings.shape != "follow":
        profile = "any" if settings.shape == "any" else settings.shape
        if settings.shape_strictness in {"loose", "standard", "strict"}:
            strictness = settings.shape_strictness
        elif original_shape == profile:
            strictness = params.shape_profile_strictness
        else:
            strictness = "standard"
            derivations.append(
                T("No strictness was given for shape {0}; derived as standard", profile)
            )
        if profile == "any" and settings.shape_strictness != "follow":
            issues.append(
                _planner_issue(
                    "SHAPE_STRICTNESS_WITHOUT_SHAPE",
                    T(
                        "\"Any shape\" cannot be combined with a shape requirement; clear one of "
                        "them.",
                    ),
                    "shape",
                    "shape_strictness",
                )
            )
        shape_changes: dict[str, object] = {"shape_profile": profile}
        if profile != "any":
            shape_changes.update(
                {
                    "shape_profile_strictness": strictness,
                    "min_thickness": 0.0,
                    "max_thickness": 0.0,
                    "min_circularity": 0.0,
                    "min_solidity": 0.0,
                    "max_eccentricity": 1.0,
                    "max_aspect_ratio": 1000.0,
                }
            )
            derivations.append(
                T(
                    "An explicit shape uses the unified shape rules; duplicate geometric limits "
                    "of the preset are cleared",
                )
            )
        params = _write_params(
            params,
            provenance,
            f"shape:{profile}",
            **shape_changes,
        )
    elif settings.shape_strictness != "follow":
        if params.shape_profile == "any":
            issues.append(
                _planner_issue(
                    "SHAPE_STRICTNESS_WITHOUT_SHAPE",
                    T(
                        "The current base has no specific shape, so the shape requirement has no "
                        "effect; choose a shape first.",
                    ),
                    "shape",
                    "shape_strictness",
                )
            )
        else:
            params = _write_params(
                params,
                provenance,
                f"shape_strictness:{settings.shape_strictness}",
                shape_profile_strictness=settings.shape_strictness,
            )

    if settings.size != "follow" and settings.size != "any":
        if params.shape_profile == "fiber":
            requested_min, requested_max = _FIBER_THICKNESS_VALUES[settings.size]
            minimum, maximum, valid = _constraint_intersection(
                params.min_thickness,
                params.max_thickness,
                requested_min,
                requested_max,
            )
            if valid:
                params = _write_params(
                    params,
                    provenance,
                    f"size:{settings.size}:thickness_intersection",
                    min_thickness=minimum,
                    max_thickness=maximum,
                )
            else:
                issues.append(
                    _planner_issue(
                        "EMPTY_SIZE_CONSTRAINT",
                        T(
                            "The target thickness does not overlap the thickness range of the "
                            "detector / professional base.",
                        ),
                        "size",
                        details={
                            "base_min": params.min_thickness,
                            "base_max": params.max_thickness,
                            "requested_min": requested_min,
                            "requested_max": requested_max,
                        },
                    )
                )
        else:
            requested_min, requested_max, _background_radius = _SIZE_VALUES[settings.size]
            requested_min = _area(requested_min)
            requested_max = _area(requested_max)
            minimum, maximum, valid = _constraint_intersection(
                params.min_area,
                params.max_area,
                requested_min,
                requested_max,
            )
            if valid:
                params = _write_params(
                    params,
                    provenance,
                    f"size:{settings.size}:area_intersection",
                    min_area=int(round(minimum)),
                    max_area=int(round(maximum)),
                    min_thickness=0.0,
                    max_thickness=0.0,
                )
            else:
                issues.append(
                    _planner_issue(
                        "EMPTY_SIZE_CONSTRAINT",
                        T(
                            "The target size does not overlap the area range of the detector / "
                            "professional base.",
                        ),
                        "size",
                        details={
                            "base_min": params.min_area,
                            "base_max": params.max_area,
                            "requested_min": requested_min,
                            "requested_max": requested_max,
                        },
                    )
                )

    if settings.background != "follow":
        base_radius = {
            "any": 24,
            "tiny": 10,
            "small": 24,
            "medium": 64,
            "large": 144,
            "follow": max(1, round(params.background_radius)),
        }[settings.size]
        if params.intensity_space == "robust_snr":
            # SNR evidence needs a signed subtraction.  In this mode the
            # background choices describe only the spatial scale; switching to
            # flat-field or multiscale response would silently destroy the
            # offset/gain invariance promised by the mode.
            background = {
                "none": ("none", 0, 1.0),
                "normal": ("gaussian", base_radius, 1.0),
                "gradient": ("gaussian", base_radius * 2, 1.0),
                "uneven": ("gaussian", max(96, base_radius * 3), 1.0),
                "high": ("gaussian", max(72, round(base_radius * 1.25)), 1.0),
                "diffuse": ("gaussian", max(96, base_radius * 2), 1.0),
            }[settings.background]
        else:
            background = {
                "none": ("none", 0, 0.0),
                "normal": ("multiscale", base_radius, 1.0),
                "gradient": ("multiscale", base_radius * 2, 1.15),
                "uneven": ("multiscale", max(96, base_radius * 3), 1.25),
                "high": ("flat_field", max(72, round(base_radius * 1.25)), 1.0),
                "diffuse": ("gaussian", max(96, base_radius * 2), 0.85),
            }[settings.background]
        params = _write_params(
            params,
            provenance,
            f"background:{settings.background}",
            background_method=background[0],
            background_radius=_linear(background[1]),
            background_strength=background[2],
        )
    if (
        logic == "texture_guard"
        and settings.background != "follow"
        and params.background_method != "gaussian"
    ):
        issues.append(
            _planner_issue(
                "TEXTURE_GUARD_BACKGROUND_CONFLICT",
                T(
                    "Texture-background suppression needs \"Follow preset\" or a Gaussian "
                    "background; the current background choice would break this detector.",
                ),
                "logic",
                "background",
                details={"resolved_background_method": params.background_method},
            )
        )

    if settings.noise != "follow":
        noise = {
            "off": ("none", 0.0),
            "detail": ("gaussian", 0.25),
            "light": ("gaussian", 0.50),
            "balanced": ("gaussian", 0.80),
            "granular": ("median", 1.20),
            "strong": ("median", 1.80),
        }[settings.noise]
        params = _write_params(
            params,
            provenance,
            f"noise:{settings.noise}",
            denoise_method=noise[0],
            denoise_strength=noise[1],
        )

    if settings.sensitivity != "follow":
        if params.threshold_method == "manual":
            issues.append(
                _planner_issue(
                    "MANUAL_SENSITIVITY_CONFLICT",
                    T(
                        "A manual threshold does not use the sensitivity setting; choose an "
                        "automatic threshold or set sensitivity to \"Follow preset\".",
                    ),
                    "sensitivity",
                    "threshold_method",
                )
            )
        params = _write_params(
            params,
            provenance,
            f"sensitivity:{settings.sensitivity}:relative",
            sensitivity=params.sensitivity + _SENSITIVITY_DELTAS[settings.sensitivity],
        )

    if settings.edge != "follow":
        if params.threshold_method not in _HYSTERESIS_THRESHOLD_METHODS:
            issues.append(
                _planner_issue(
                    "EDGE_THRESHOLD_CONFLICT",
                    T(
                        "The current threshold method has no hysteresis growth stage, so the "
                        "boundary setting has no effect.",
                    ),
                    "edge",
                    "threshold_method",
                    details={"threshold_method": params.threshold_method},
                )
            )
        params = _write_params(
            params,
            provenance,
            f"edge:{settings.edge}:relative",
            grow_ratio=params.grow_ratio + _EDGE_DELTAS[settings.edge],
        )

    if settings.speck_cleanup != "follow":
        cleanup = {"off": 0, "light": 2, "balanced": 8, "strong": 20}[
            settings.speck_cleanup
        ]
        requested_min = _area(cleanup)
        minimum, maximum, valid = _constraint_intersection(
            params.min_area,
            params.max_area,
            requested_min,
            0,
        )
        if valid:
            params = _write_params(
                params,
                provenance,
                f"speck_cleanup:{settings.speck_cleanup}:intersection",
                min_area=int(round(minimum)),
                max_area=int(round(maximum)),
            )
        else:
            issues.append(
                _planner_issue(
                    "EMPTY_SIZE_CONSTRAINT",
                    T(
                        "The lower limit of speck cleanup exceeds the largest area the target / "
                        "detector allows.",
                    ),
                    "speck_cleanup",
                    "size",
                    details={
                        "current_min": params.min_area,
                        "current_max": params.max_area,
                        "cleanup_min": requested_min,
                    },
                )
            )

    if settings.opening != "follow":
        params = _write_params(
            params,
            provenance,
            f"opening:{settings.opening}",
            open_radius=_linear({"off": 0, "light": 1, "strong": 2}[settings.opening]),
        )

    if settings.gap_connect != "follow":
        params = _write_params(
            params,
            provenance,
            f"gap_connect:{settings.gap_connect}",
            close_radius=_linear(
                {"off": 0, "light": 1, "medium": 2, "strong": 4}[settings.gap_connect]
            ),
        )

    if settings.contour_adjust != "follow":
        erosion, dilation = {
            "shrink_2": (2, 0),
            "shrink_1": (1, 0),
            "none": (0, 0),
            "grow_1": (0, 1),
            "grow_2": (0, 2),
        }[settings.contour_adjust]
        params = _write_params(
            params,
            provenance,
            f"contour_adjust:{settings.contour_adjust}",
            erode_radius=_linear(erosion),
            dilate_radius=_linear(dilation),
        )

    if settings.holes != "follow":
        params = _write_params(
            params,
            provenance,
            f"holes:{settings.holes}",
            hole_area=_area(
                {"keep": 0, "small": 4, "medium": 30, "large": 300}[settings.holes]
            ),
        )

    if settings.touching != "follow":
        split, distance, height_value = {
            "keep": (False, 6, 0.28),
            "conservative": (True, 10, 0.38),
            "balanced": (True, 6, 0.28),
            "strong": (True, 3, 0.20),
        }[settings.touching]
        params = _write_params(
            params,
            provenance,
            f"touching:{settings.touching}",
            split_touching=split,
            watershed_peak_distance=_linear(distance),
            watershed_peak_height=height_value,
        )

    if settings.border != "follow":
        params = _write_params(
            params,
            provenance,
            f"border:{settings.border}",
            remove_border=settings.border == "remove",
        )

    if settings.intensity != "follow":
        if params.intensity_space == "robust_snr":
            requested_mean = 0.0
            requested_confidence = {
                "none": 0.0,
                "weak": 0.40,
                "balanced": 0.50,
                "clear": 0.65,
                "strict": 0.85,
            }[settings.intensity]
        else:
            requested_mean = {
                "none": 0.0,
                "weak": 0.02,
                "balanced": 0.04,
                "clear": 0.10,
                "strict": 0.18,
            }[settings.intensity]
            requested_confidence = {
                "none": 0.0,
                "weak": 0.55,
                "balanced": 0.80,
                "clear": 1.15,
                "strict": 1.70,
            }[settings.intensity]
        retained_mean = max(params.min_mean_intensity, requested_mean)
        retained_confidence = max(
            params.min_response_confidence,
            requested_confidence,
        )
        params = _write_params(
            params,
            provenance,
            f"intensity:{settings.intensity}:minimum_evidence",
            min_mean_intensity=retained_mean,
            min_response_confidence=retained_confidence,
        )
        if retained_mean > requested_mean or retained_confidence > requested_confidence:
            derivations.append(
                T(
                    "Positive intensity only raises the lower evidence limit; the detector's "
                    "stricter existing limit was kept",
                )
            )

    profile = params.shape_profile
    if profile == "ring":
        if settings.holes not in {"follow", "keep"}:
            issues.append(
                _planner_issue(
                    "RING_HOLE_FILL_CONFLICT",
                    T(
                        "Ring targets conflict with explicit hole filling; choose \"Keep holes\" "
                        "or drop the ring shape.",
                    ),
                    "shape",
                    "holes",
                )
            )
        if settings.gap_connect not in {"follow", "off"}:
            issues.append(
                _planner_issue(
                    "RING_CONNECT_CONFLICT",
                    T(
                        "Ring targets cannot use ordinary gap connection; it may close the "
                        "central hole.",
                    ),
                    "shape",
                    "gap_connect",
                )
            )
        if settings.contour_adjust in {"grow_1", "grow_2"}:
            issues.append(
                _planner_issue(
                    "RING_DILATION_CONFLICT",
                    T("Ring targets cannot be dilated explicitly; it may close the central hole."),
                    "shape",
                    "contour_adjust",
                )
            )
        ring_changes: dict[str, object] = {"hole_area": 0}
        if settings.gap_connect == "follow":
            ring_changes["close_radius"] = 0
        if settings.contour_adjust == "follow":
            ring_changes["dilate_radius"] = 0
        params = _write_params(
            params,
            provenance,
            "derived:ring_topology_preservation",
            **ring_changes,
        )
        derivations.append(T(
            "The ring branch keeps holes automatically and turns off closing/dilation that was "
            "not chosen explicitly",
        ))

    if profile in _CONTINUOUS_SHAPES:
        if settings.touching not in {"follow", "keep"}:
            issues.append(
                _planner_issue(
                    "CONTINUOUS_SPLIT_CONFLICT",
                    T(
                        "Continuous structures (fibres, patches or rings) cannot use watershed "
                        "separation explicitly.",
                    ),
                    "shape",
                    "touching",
                )
            )
        params = _write_params(
            params,
            provenance,
            "derived:continuous_keep_whole",
            split_touching=False,
        )
        derivations.append(T("Continuous-shape branches turn off the watershed automatically"))

    validation_shape = (
        None
        if params.scale_mode == "relative_grid"
        else ((height, width) if image_shape is not None else None)
    )
    before_validation = params
    params = params.validated(validation_shape)
    for item in fields(UniversalParams):
        if getattr(before_validation, item.name) != getattr(params, item.name):
            provenance.setdefault(item.name, []).append("validated")
    before_compatibility = params
    params = enforce_shape_compatibility(params)
    for item in fields(UniversalParams):
        if getattr(before_compatibility, item.name) != getattr(params, item.name):
            provenance.setdefault(item.name, []).append("core_shape_compatibility")

    return (
        params,
        tuple(
            (key, tuple(value)) for key, value in sorted(provenance.items())
        ),
        tuple(dict.fromkeys(derivations)),
        tuple(issues),
    )


def _merge_provenance(
    left: tuple[tuple[str, tuple[str, ...]], ...],
    right: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    merged: dict[str, list[str]] = {}
    for key, sources in (*left, *right):
        values = merged.setdefault(key, [])
        for source in sources:
            if source not in values:
                values.append(source)
    return tuple((key, tuple(values)) for key, values in sorted(merged.items()))


def compile_quick_plan(
    base_params: UniversalParams,
    settings: QuickSettings,
    image_shape: tuple[int, int] | None = None,
) -> QuickPlan:
    """Compile settings into deduplicated branches plus explicit diagnostics."""

    canonical, _notes = normalize_quick_settings(settings, base_params.shape_profile)
    logic_values = quick_setting_values("logic", canonical.logic)
    shape_values = quick_setting_values("shape", canonical.shape)
    size_values = quick_setting_values("size", canonical.size)

    branches_by_key: dict[str, QuickBranch] = {}
    issues: list[QuickPlanIssue] = []
    for logic, shape, size in product(logic_values, shape_values, size_values):
        if logic == "follow":
            branch_base = base_params
            base_source = "base"
        elif logic == "texture_guard":
            branch_base = replace(
                BUILTIN_PRESETS["Strict positive"].params,
                background_method="gaussian",
                background_radius=24,
                min_response_confidence=1.70,
            )
            base_source = "detector:texture_guard"
        else:
            branch_base = BUILTIN_PRESETS[_LOGIC_PRESETS[logic]].params
            base_source = f"detector:{logic}"
        branch_settings = replace(
            canonical,
            logic="follow",
            shape=shape,
            size=size,
        )
        params, provenance, derivations, branch_issues = _compile_scalar_branch(
            branch_base,
            branch_settings,
            image_shape,
            logic=logic,
            shape=shape,
            size=size,
            base_source=base_source,
        )
        execution_key = json.dumps(
            _identity_values(params),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        branch_id = "qb-" + hashlib.sha256(execution_key.encode("utf-8")).hexdigest()[:12]
        origin = (logic, shape, size)
        branch = QuickBranch(
            branch_id=branch_id,
            logic=logic,
            shape=shape,
            size=size,
            params=params,
            origins=(origin,),
            provenance=provenance,
            derivations=derivations,
        )
        existing = branches_by_key.get(execution_key)
        if existing is None:
            branches_by_key[execution_key] = branch
        else:
            branches_by_key[execution_key] = replace(
                existing,
                origins=tuple(dict.fromkeys((*existing.origins, origin))),
                provenance=_merge_provenance(existing.provenance, provenance),
                derivations=tuple(
                    dict.fromkeys((*existing.derivations, *derivations))
                ),
            )
        issues.extend(
            replace(
                issue,
                branch_id=(branch_id if issue.branch_id == "__branch__" else issue.branch_id),
            )
            for issue in branch_issues
        )

    branches = tuple(branches_by_key.values())
    scale_signatures = tuple(
        sorted(
            {
                (
                    str(branch.params.scale_mode),
                    int(branch.params.reference_short_side),
                )
                for branch in branches
            }
        )
    )
    if len(scale_signatures) > 1:
        issues.append(
            QuickPlanIssue(
                code="MIXED_SCALE_MODE",
                severity="error",
                message=(
                    T(
                        "One union plan contains different scale modes, so the preview and the "
                        "full-resolution run cannot share it safely; choose one scale mode "
                        "explicitly (relative or fixed pixels).",
                    )
                ),
                fields=("scale", "logic"),
                details=(
                    (
                        "scale_signatures",
                        [
                            {"mode": mode, "reference_short_side": reference}
                            for mode, reference in scale_signatures
                        ],
                    ),
                ),
            )
        )
    if len(branches) > MAX_QUICK_VARIANTS:
        issues.append(
            QuickPlanIssue(
                code="QUICK_VARIANT_LIMIT",
                severity="error",
                message=(
                    T(
                        "After removing duplicates this combination still gives {0} recognition "
                        "branches, more than the safe limit of {1}; clear some choices of the "
                        "same group.",
                        len(branches),
                        MAX_QUICK_VARIANTS,
                    )
                ),
                fields=("logic", "shape", "size"),
                details=(("deduplicated_branch_count", len(branches)),),
            )
        )

    unique_issues: list[QuickPlanIssue] = []
    seen_issues: set[str] = set()
    for issue in issues:
        key = json.dumps(issue.to_dict(), ensure_ascii=False, sort_keys=True)
        if key not in seen_issues:
            seen_issues.add(key)
            unique_issues.append(issue)

    fingerprint_payload = {
        "schema_version": 3,
        "base": _identity_values(base_params),
        "settings": quick_settings_to_dict(canonical),
        "image_shape": list(image_shape) if image_shape is not None else None,
        "branches": [_identity_record(branch.to_dict()) for branch in branches],
        "issues": [issue.to_dict() for issue in unique_issues],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return QuickPlan(
        settings=canonical,
        image_shape=(
            (int(image_shape[0]), int(image_shape[1]))
            if image_shape is not None
            else None
        ),
        branches=branches,
        issues=tuple(unique_issues),
        fingerprint=fingerprint,
    )


def resolve_quick_settings_variants(
    base_params: UniversalParams,
    settings: QuickSettings,
    image_shape: tuple[int, int] | None = None,
) -> tuple[UniversalParams, ...]:
    """Return deduplicated executable branches after conflict validation."""

    plan = compile_quick_plan(base_params, settings, image_shape)
    plan.raise_for_errors()
    return tuple(branch.params for branch in plan.branches)


def resolve_quick_settings(
    base_params: UniversalParams,
    settings: QuickSettings,
    image_shape: tuple[int, int] | None = None,
) -> UniversalParams:
    """Return the representative branch used by the professional parameter panel."""

    return resolve_quick_settings_variants(base_params, settings, image_shape)[0]


def quick_plan_key(
    base_params: UniversalParams,
    settings: QuickSettings,
    image_shape: tuple[int, int],
) -> str:
    plan = compile_quick_plan(base_params, settings, image_shape)
    plan.raise_for_errors()
    return plan.fingerprint


def quick_variant_count(
    base_params: UniversalParams,
    settings: QuickSettings,
    image_shape: tuple[int, int] | None = None,
) -> int:
    plan = compile_quick_plan(base_params, settings, image_shape)
    plan.raise_for_errors()
    return len(plan.branches)


def segment_loaded_image_with_quick(
    loaded: LoadedImage,
    base_params: UniversalParams,
    settings: QuickSettings,
    *,
    max_side: int | None = None,
) -> UniversalResult:
    """Run each selected branch and return their exact pixel-wise mask union."""

    if getattr(base_params, "recognition_engine", "classic") in AI_RECOGNITION_ENGINES:
        # A saved model recipe must never turn into classical branches when
        # it happens to carry quick controls.
        return segment_loaded_image(loaded, base_params, max_side=max_side)

    started = time.perf_counter()
    plan = compile_quick_plan(base_params, settings, loaded.shape)
    plan.raise_for_errors()
    results = [
        segment_loaded_image(loaded, branch.params, max_side=max_side)
        for branch in plan.branches
    ]
    compiled_plan = plan.to_dict()
    if len(results) == 1:
        result = results[0]
        branch_id = plan.branches[0].branch_id
        result.objects = [
            {**record, "contributing_branch_ids": [branch_id]}
            for record in result.objects
        ]
        setattr(result, "quick_plan_key", plan.fingerprint)
        setattr(result, "quick_variant_count", 1)
        setattr(result, "quick_plan", plan)
        setattr(result, "compiled_plan", compiled_plan)
        return result

    primary = results[0]
    mask = np.logical_or.reduce([result.mask for result in results])
    labels, label_boxes = _plain_component_labels_with_boxes(mask)
    response = np.maximum.reduce([result.response for result in results])
    normalized = np.maximum.reduce([result.normalized_signal for result in results])
    final_mask, final_labels, objects = measure_and_filter_labels(
        labels,
        primary.signal,
        normalized,
        response,
        primary.params,
        apply_filters=False,
        bounding_boxes=label_boxes,
    )
    contributions: dict[int, list[str]] = {
        int(record["id"]): [] for record in objects
    }
    for branch, branch_result in zip(plan.branches, results, strict=True):
        label_ids = np.unique(final_labels[branch_result.mask & final_mask])
        for label_id in label_ids.tolist():
            if int(label_id) > 0:
                contributions.setdefault(int(label_id), []).append(branch.branch_id)
    objects = [
        {
            **record,
            "contributing_branch_ids": contributions.get(int(record["id"]), []),
        }
        for record in objects
    ]
    objects = _rescale_preview_objects(objects, primary.analysis_scale)
    result = UniversalResult(
        mask=final_mask,
        labels=final_labels,
        objects=objects,
        signal=primary.signal,
        normalized_signal=normalized,
        response=response,
        threshold_value=float("nan"),
        low_threshold=float("nan"),
        positive_fraction=float(final_mask.mean()),
        params=primary.params,
        signal_mode_used=T("{0} (union of {1} branches)", primary.signal_mode_used, len(results)),
        input_black_value=min(result.input_black_value for result in results),
        input_white_value=max(result.input_white_value for result in results),
        processing_seconds=float(time.perf_counter() - started),
        analysis_scale=primary.analysis_scale,
        source_shape=primary.source_shape,
        manually_edited=False,
        input_saturation_fraction=max(
            item.input_saturation_fraction for item in results
        ),
        input_high_plateau_fraction=max(
            item.input_high_plateau_fraction for item in results
        ),
        input_largest_tie_fraction=max(
            item.input_largest_tie_fraction for item in results
        ),
        input_unique_levels=min(item.input_unique_levels for item in results),
        intensity_warnings=tuple(
            dict.fromkeys(
                warning
                for item in results
                for warning in item.intensity_warnings
            )
        ),
    )
    setattr(result, "quick_plan_key", plan.fingerprint)
    setattr(result, "quick_variant_count", len(results))
    setattr(result, "quick_plan", plan)
    setattr(result, "compiled_plan", compiled_plan)
    return result


def result_matches_quick_plan(
    result: UniversalResult | None,
    base_params: UniversalParams,
    settings: QuickSettings,
    image_shape: tuple[int, int],
) -> bool:
    return bool(
        result is not None
        and getattr(result, "quick_plan_key", "")
        == quick_plan_key(base_params, settings, image_shape)
    )


def quick_settings_summary(settings: QuickSettings) -> str:
    values = quick_settings_to_dict(settings)
    labels: list[str] = []
    for field_name, value in values.items():
        selected = quick_setting_values(field_name, value)
        if selected != ("follow",):
            labels.append(" + ".join(_CHOICES[field_name][item] for item in selected))
    return T(", ").join(labels) if labels else T("Follows the preset completely")

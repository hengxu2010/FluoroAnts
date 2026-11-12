from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .i18n import T


QUALITY_LEVEL_LABELS = {
    "ok": T("Routine review"),
    "review": T("Careful review advised"),
    "warning": T("High risk"),
}


@dataclass(frozen=True)
class QualityAssessment:
    level: str
    positive_area_px: int
    positive_fraction: float
    object_count: int
    touches_border_count: int
    touches_border_fraction: float
    touches_border_area_px: int
    border_area_fraction: float
    largest_object_area_px: int
    largest_object_fraction: float
    small_object_count: int
    small_object_fraction: float
    saturated_object_count: int
    saturated_object_fraction: float
    response_confidence_median: float | None
    response_confidence_p10: float | None
    near_confidence_floor_fraction: float | None
    warnings: tuple[str, ...]

    @property
    def level_label(self) -> str:
        return QUALITY_LEVEL_LABELS.get(self.level, self.level)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["level_label"] = self.level_label
        payload["warnings"] = list(self.warnings)
        payload["warning_count"] = len(self.warnings)
        payload["is_ground_truth_assessment"] = False
        return payload


def _finite_number(record: Mapping[str, object], key: str) -> float | None:
    try:
        value = float(record.get(key, float("nan")))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def assess_segmentation(
    mask: np.ndarray,
    objects: Sequence[Mapping[str, object]],
    *,
    mixed_channels: bool = False,
    signal_mode_used: str = "",
    recipe_name: str = "",
    structure_method: str = "",
    confidence_floor: float = 0.0,
    extra_warnings: Sequence[str] = (),
    force_warning: bool = False,
) -> QualityAssessment:
    """Return deterministic review hints without treating any mask as biological truth."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(T("The quality check accepts 2-D masks only."))
    height, width = binary.shape
    image_pixels = max(1, int(height * width))
    positive_area = int(np.count_nonzero(binary))
    positive_fraction = float(positive_area / image_pixels)
    object_count = len(objects)

    areas: list[float] = []
    border_areas: list[float] = []
    confidence_values: list[float] = []
    saturated_count = 0
    border_count = 0
    tiny_threshold = max(9.0, min(64.0, image_pixels * 0.00001))
    small_count = 0
    for record in objects:
        area = _finite_number(record, "area_px")
        safe_area = max(0.0, area or 0.0)
        areas.append(safe_area)
        if safe_area <= tiny_threshold:
            small_count += 1
        if bool(record.get("touches_border", False)):
            border_count += 1
            border_areas.append(safe_area)
        confidence = _finite_number(record, "response_confidence")
        if confidence is not None:
            confidence_values.append(confidence)
        saturated = _finite_number(record, "saturated_fraction")
        if saturated is not None and saturated >= 0.25:
            saturated_count += 1

    border_area = int(round(sum(border_areas)))
    largest_area = int(round(max(areas, default=0.0)))
    border_object_fraction = float(border_count / object_count) if object_count else 0.0
    border_area_fraction = float(border_area / positive_area) if positive_area else 0.0
    largest_fraction = float(largest_area / image_pixels)
    small_fraction = float(small_count / object_count) if object_count else 0.0
    saturated_fraction = float(saturated_count / object_count) if object_count else 0.0
    confidence_median = float(np.median(confidence_values)) if confidence_values else None
    confidence_p10 = float(np.percentile(confidence_values, 10)) if confidence_values else None
    near_floor_fraction: float | None = None
    if confidence_values and confidence_floor > 0:
        near_limit = float(confidence_floor) + max(0.08, float(confidence_floor) * 0.10)
        near_floor_fraction = float(
            np.count_nonzero(np.asarray(confidence_values) <= near_limit) / len(confidence_values)
        )

    warnings: list[str] = [str(item) for item in extra_warnings if str(item).strip()]
    high_risk = bool(force_warning)
    if object_count == 0 or positive_area == 0:
        warnings.append(
            T(
                "No candidate objects were detected; no empty alpha is created and the existing "
                "selection is kept. Check the source, the signal polarity and the threshold.",
            )
        )
    if positive_fraction >= 0.35:
        high_risk = True
        warnings.append(
            T(
                "The candidate area covers {0:.1%} of the canvas; it may include background, a "
                "wrong source or too wide a threshold.",
                positive_fraction,
            )
        )
    elif positive_fraction >= 0.15:
        warnings.append(
            T(
                "The candidate area covers {0:.1%} of the canvas; check diffuse background "
                "carefully against a negative control.",
                positive_fraction,
            )
        )
    if largest_fraction >= 0.20:
        high_risk = True
        warnings.append(
            T(
                "The largest connected object covers {0:.1%} of the canvas; it may join large "
                "areas of background into one object.",
                largest_fraction,
            )
        )
        if positive_fraction >= 0.35:
            warnings.append(
                T(
                    "Compare \"Texture-background suppression\" first; if the expected targets "
                    "are mainly tiny spots, also add \"Sparse tiny spots + Dense tiny spots\"; if "
                    "large or diffuse regions are expected, keep the current result and review it "
                    "against control images.",
                )
            )
    elif largest_fraction >= 0.08:
        warnings.append(
            T(
                "The largest connected object covers {0:.1%} of the canvas; confirm that it is "
                "one structure and not merged background.",
                largest_fraction,
            )
        )
    if border_area_fraction >= 0.30:
        high_risk = True
        warnings.append(
            T(
                "Objects touching the border contribute {0:.1%} of the candidate area; the risk "
                "of edge artefacts is high.",
                border_area_fraction,
            )
        )
    elif border_area_fraction >= 0.10:
        warnings.append(
            T(
                "Objects touching the border contribute {0:.1%} of the candidate area; check the "
                "canvas edges.",
                border_area_fraction,
            )
        )
    if object_count >= 8 and border_object_fraction >= 0.35:
        warnings.append(T(
            "{0:.1%} of the objects touch the canvas edge; check cropping or edge noise.",
            border_object_fraction,
        ))
    if object_count >= 100 and small_fraction >= 0.75:
        warnings.append(
            T(
                "{0:.1%} of the objects are no larger than {1:.0f} px²; the result may be "
                "fragmented.",
                small_fraction,
                tiny_threshold,
            )
        )
    if object_count >= 8 and saturated_fraction >= 0.40:
        warnings.append(T(
            "{0:.1%} of the objects contain many saturated pixels; treat boundaries and intensity "
            "measurements with caution.",
            saturated_fraction,
        ))
    if object_count >= 20 and near_floor_fraction is not None and near_floor_fraction >= 0.65:
        warnings.append(
            T(
                "{0:.1%} of the objects are close to the current confidence threshold; small "
                "parameter changes may change the count noticeably.",
                near_floor_fraction,
            )
        )
    if mixed_channels:
        warnings.append(
            T(
                "The whole visible composite is used and several top-level channels are mixed; "
                "this suits only explorations that explicitly need merged channels.",
            )
        )
    if signal_mode_used.startswith("max_rgb"):
        warnings.append(T(
            "The current signal mode takes the brightest RGB channel; to quantify a single "
            "marker, choose its colour or single-channel source explicitly.",
        ))
    if recipe_name == "Very low false positives":
        warnings.append(
            T(
                "\"Very low false positives\" may shrink true object boundaries and reduce "
                "objects to their bright cores; compare side by side with \"Strict positive\".",
            )
        )
    if structure_method == "log":
        warnings.append(
            T(
                "Spot-specific enhancement is in use; it also responds to spotty texture and to "
                "bright cores inside larger targets. If the targets are not spots, use a general "
                "preset.",
            )
        )
    elif structure_method == "vesselness":
        warnings.append(
            T(
                "Fibre-specific enhancement is in use; it also responds to target outlines, "
                "bright edges and curved boundaries. If the targets are not fibres, tubes or "
                "networks, use a general preset.",
            )
        )

    unique_warnings = tuple(dict.fromkeys(warnings))
    level = "warning" if high_risk else ("review" if unique_warnings else "ok")
    return QualityAssessment(
        level=level,
        positive_area_px=positive_area,
        positive_fraction=positive_fraction,
        object_count=object_count,
        touches_border_count=border_count,
        touches_border_fraction=border_object_fraction,
        touches_border_area_px=border_area,
        border_area_fraction=border_area_fraction,
        largest_object_area_px=largest_area,
        largest_object_fraction=largest_fraction,
        small_object_count=small_count,
        small_object_fraction=small_fraction,
        saturated_object_count=saturated_count,
        saturated_object_fraction=saturated_fraction,
        response_confidence_median=confidence_median,
        response_confidence_p10=confidence_p10,
        near_confidence_floor_fraction=near_floor_fraction,
        warnings=unique_warnings,
    )

# -*- coding: utf-8 -*-
"""
Learned model v6.3: local-contrast target recognition.

The model looks for small targets that are clearly brighter than their
immediate surroundings and sit in a quiet neighbourhood.  The decision uses six
interpretable quantities.  Each one is first converted into a robust score
relative to the other candidates of the same image, so images that differ in
overall brightness, exposure or gain can be compared directly.

The six quantities
    ringnear  how much brighter than the adjacent ring of background
              (in units of the local noise)
    zmean     mean local contrast inside the target
    zmax      peak local contrast of the target
    noiserel  noise level of the neighbourhood (relative to the image median;
              the quieter, the more target-like)
    reftone   brightness of the reference channel at that place (the brighter,
              the more likely crosstalk or a cross-channel artefact)
    aspect    ratio of the long to the short axis (the more elongated, the more
              fibre-like)

Why the features are normalised per image
    Targets are stable in absolute feature space, but the population of
    non-targets differs enormously between images: one image has large
    over-exposed structures in the background, another is clean.  The decision
    therefore has to rest on "relative to this image's own candidates", not on
    absolute brightness.
    The scale of the normalisation is estimated from the lower half of each
    distribution only (the background side): a MAD over the whole distribution
    is inflated by true signal, which would push images with many targets back
    and flatten real differences between images.

Scope
    Single-channel images, or images that can be reduced to one plane, of any
    size and bit depth.  The detector works at a fixed pixel scale (see
    SCALE_NOTE) and suits targets of about 3 to 30 pixels in diameter.
    A reference channel is optional: when present it is used to subtract
    crosstalk and as ``reftone``; without it that term takes the training mean
    and the other five are unaffected.
    The threshold was calibrated on one particular set of images.  On another
    kind of image the shape discrimination still holds, but how strict the
    threshold is has to be checked against the new data.

Depends only on numpy, opencv and scipy.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from .i18n import T

ENGINE = "learned_v6"
MODEL_DIR = Path(__file__).with_name("learned_v6_data")
MODEL_FILENAME = "model_v6.json"

SEED_THRESHOLD = 3.5          # candidate threshold: local-contrast z score
REVIEW_MARGIN = 1.5           # candidates this far below the main threshold go to "review"
FULL_SCALE = 65535.           # common internal intensity range
SATURATION_LEVEL = 65000.     # intensities from here on count as clipped
NOISE_FLOOR = 32.             # lowest image noise of the calibration set; below it see image_noise
SENSITIVITY_LIMIT = 3.        # how far the user may move the main threshold
_PIXEL_VARIANCE = 1. / 12.    # second moment of one pixel: a line one pixel wide is not infinitely thin
_COARSE_STEP = .04            # grey step / typical neighbour difference from which data count as coarse
NOISE_TERM_MIN_SCALE = .10    # differences in noise level below 10 % are estimation error, not structure
SCALE_NOTE = (T(
    "Candidate radius 9 px; ring background 13/27 px; noise estimated in 50 px blocks; "
    "multi-scale differences σ=0.65/1.15/2.2/4.2 px. Best for targets about 3–30 px in diameter.",
))

# File-name rules for pairing a reference channel found in the same folder:
# (pattern in the target name, replacement that gives the reference name).
# A replacement ending in "*" accepts any channel name after the number, as
# long as exactly one file fits.  These are naming conventions, not
# requirements: without a match the image is analysed without a reference
# channel.
PAIRING_RULES = (
    (r"(?i)_04_(.+)$", "_03_ref"),
    (r"(?i)_c0*4$", "_C03"),
    (r"(?i)_ch0*2$", "_ch01"),
    (r"(?i)_ch0*1$", "_ch00"),
    (r"(?i)_04_[^_]+$", "_03_*"),
)
_EXPORT_NAME_MARKER = "_fluoroants"     # files written by this program are never a reference

_MODEL = None
_MANIFEST = None
_K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
_K5 = np.ones((5, 5), np.uint8)
_MARGIN = 16
_CANDIDATE_RADIUS = 9          # half width of a candidate's support window (19×19), as in SCALE_NOTE
_RING_RADIUS = 6               # radius of the 13×13 elliptical kernel: the ring reaches 6 px from the target
_CHUNK = 16384                 # candidates per batch: bounds the peak memory of the vectorised steps


def load_model(path=None):
    global _MODEL
    if _MODEL is None or path is not None:
        import hashlib
        p = Path(path) if path else MODEL_DIR / MODEL_FILENAME
        raw = p.read_bytes()
        manifest = json.loads((p.parent / "bundle_manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(raw).hexdigest() != manifest["model_sha256"]:
            raise RuntimeError(T(
                "The model file failed its integrity check; please reinstall the complete model "
                "resources.",
            ))
        _MODEL = json.loads(raw.decode("utf-8"))
    return _MODEL


def _load_manifest():
    """Read the model manifest once; it is small, but re-reading it for every image is pointless."""
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.loads((MODEL_DIR / "bundle_manifest.json").read_text(encoding="utf-8"))
    return _MANIFEST


# --------------------------------------------------------------- input adaptation
def _plane(loaded, *, target):
    """Any size and bit depth -> float32 on the common 0..65535 range."""
    name = T("Target image") if target else T("Reference image")
    a = np.asarray(loaded.image)
    if loaded.metadata.get("analysis_preprocessing"):
        raise ValueError(T(
            "{0}: this mode needs an image without crosstalk subtraction or other preprocessing; "
            "please load the original image again.",
            name,
        ))
    if a.ndim == 3:
        if a.shape[2] == 4 and not np.all(a[..., 3] == np.iinfo(a.dtype).max if a.dtype.kind == "u" else True):
            a = a[..., :3]
        a = a[..., :3].max(axis=2) if a.shape[2] >= 3 else a[..., 0]
        note = T("maximum over channels")
    elif a.ndim == 2:
        note = T("greyscale")
    else:
        raise ValueError(T("{0}: a 2-D greyscale or multi-channel image is required.", name))
    if a.dtype.kind in "ui":
        info = np.iinfo(a.dtype)
        scale = FULL_SCALE / float(info.max) if info.max != int(FULL_SCALE) else 1.
        out = a.astype(np.float32) * np.float32(scale)
        note += T(" · {0}→16-bit range", a.dtype) if scale != 1. else f" · {a.dtype}"
    else:
        f = a.astype(np.float32)
        top = float(np.nanmax(f)) if f.size else 1.
        out = f * np.float32(FULL_SCALE) if top <= 1.5 else f
        note += T(" · float, normalized") if top <= 1.5 else T(" · float, original values")
    if out.shape[0] < 64 or out.shape[1] < 64:
        raise ValueError(T(
            "{0}: the size is {1}, smaller than the detector's minimum working size of 64×64.",
            name,
            out.shape,
        ))
    return np.nan_to_num(out, nan=0., posinf=FULL_SCALE, neginf=0.), note


def find_reference(loaded, reference_loaded):
    """Return the reference channel, or None: it is optional.

    A reference passed in is used as is; otherwise the folder of the image is
    searched for a partner named by PAIRING_RULES.  The folder is scanned once
    with ``os.scandir`` for all rules together (earlier rules win), which keeps
    the number of system calls low in batch folders with thousands of images.
    """
    if reference_loaded is not None:
        return reference_loaded if reference_loaded.shape == loaded.shape else None
    from .universal_core import SUPPORTED_IMAGE_SUFFIXES, load_image
    stem = loaded.path.stem.casefold()
    wanted = []            # exact reference names, in rule order
    prefixes = []          # "name up to the channel number" for the "*" rules
    for pattern, repl in PAIRING_RULES:
        alt = re.sub(pattern, repl, loaded.path.stem).casefold()
        if alt == stem:
            continue
        if alt.endswith("*"):
            if alt[:-1] not in prefixes:
                prefixes.append(alt[:-1])
        elif alt not in wanted:
            wanted.append(alt)
    if not wanted and not prefixes:
        return None
    found = {}
    loose = {prefix: [] for prefix in prefixes}
    try:
        with os.scandir(loaded.path.parent) as entries:
            for entry in entries:
                name = entry.name
                dot = name.rfind(".")
                if dot <= 0 or name[dot:].casefold() not in SUPPORTED_IMAGE_SUFFIXES:
                    continue
                key = name[:dot].casefold()
                exact = key in wanted and key not in found
                fitting = [
                    prefix for prefix in prefixes
                    if key.startswith(prefix) and key != stem and _EXPORT_NAME_MARKER not in key
                ]
                if not exact and not fitting:
                    continue
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                if exact:
                    found[key] = Path(entry.path)
                    if not prefixes and len(found) == len(wanted):
                        break
                for prefix in fitting:
                    loose[prefix].append(Path(entry.path))
    except OSError:
        return None
    candidates = [found[key] for key in wanted if key in found]
    candidates += [loose[prefix][0] for prefix in prefixes if len(loose[prefix]) == 1]
    for path in candidates:
        try:
            ref = load_image(path)
        except Exception:
            return None
        return ref if ref.shape == loaded.shape else None
    return None


# --------------------------------------------------------------- candidates
def _gauss(a, s):
    return cv2.GaussianBlur(a, (0, 0), s, borderType=cv2.BORDER_REFLECT_101)


def _grey_step(differences):
    """Smallest positive value: the grey step of the data when they are quantised."""
    positive = differences[differences > 0]
    return float(positive.min()) if positive.size else 0.


def image_noise(raw):
    """Noise of the whole image, from the residual against a slight blur.

    Down to NOISE_FLOOR this is the plain robust estimate, as in the
    calibration.  Below it the fixed floor used to take over: an absolute
    number of counts on the 16-bit range, so the same scene recorded at a
    lower gain (a 12-bit camera writing 16-bit files, say) looked several
    times fainter in noise units and lost its candidates.  Such an image is
    now measured by itself: stretches without any variation (clipped
    background) are left out, and the result is at least half a grey step.
    """
    residual = np.abs(raw - _gauss(raw, 1))
    noise = float(np.median(residual)) * 1.4826
    if noise >= NOISE_FLOOR:
        return noise
    step = _grey_step(np.abs(np.diff(raw, axis=1)))
    live = residual[residual > max(1e-6 * FULL_SCALE, .01 * step)]
    estimate = float(np.median(live)) * 1.4826 if live.size >= 1024 else 0.
    return max(estimate, .5 * step, 1e-3)


def response_from_planes(raw, ref):
    """Multi-scale difference-of-Gaussians response map; without ``ref`` no crosstalk is subtracted."""
    if ref is None:
        corrected = raw
    else:
        corrected = np.maximum(raw - np.maximum(ref - np.percentile(ref, 10), 0), 0)
    noise = image_noise(raw)
    fine = (_gauss(raw, .65) - _gauss(raw, 2.2)) / noise
    medium = (_gauss(raw, 1.15) - _gauss(raw, 4.2)) / noise
    return corrected, np.maximum(fine, medium).astype("float32"), noise


def propose(response, cutoff=SEED_THRESHOLD):
    """Pixels that are 5×5 local maxima above the cutoff; one peak per plateau (4-connected).

    ``cv2.dilate`` as a 5×5 maximum filter gives, pixel for pixel, the same
    answer to "equals the neighbourhood maximum" as
    ``ndi.maximum_filter(mode="reflect")`` (reflected pixels are already inside
    the window) and is an order of magnitude faster.  A plateau is represented
    by its highest response; ties go to the first pixel in row-major order.
    """
    response = np.ascontiguousarray(response, dtype=np.float32)
    maxima = (response == cv2.dilate(response, _K5)) & (response >= cutoff)
    ys, xs = np.nonzero(maxima)
    if ys.size == 0:
        return np.zeros((0, 2), dtype="int32")
    lab, count = ndi.label(maxima)
    if count == ys.size:                            # single-pixel peaks only: nothing to merge
        return np.stack([ys, xs], axis=1).astype("int32")
    ids = lab[ys, xs]
    order = np.lexsort((xs, ys, -response[ys, xs], ids))
    ids_sorted = ids[order]
    keep = np.empty(order.size, dtype=bool)
    keep[0] = True
    np.not_equal(ids_sorted[1:], ids_sorted[:-1], out=keep[1:])
    chosen = order[keep]
    return np.stack([ys[chosen], xs[chosen]], axis=1).astype("int32")


def _tile_grid(count):
    """Arrange ``count`` tiles in a near-square mosaic: (rows, columns, row of each tile, column of each tile)."""
    cols = max(1, int(math.ceil(math.sqrt(count))))
    rows = (count + cols - 1) // cols
    index = np.arange(count)
    return rows, cols, index // cols, index % cols


def candidate_objects(response):
    """Grow each peak inside its 19×19 window (8-connected, adaptive low threshold) into its support pixels.

    All windows are laid out in one mosaic with 1 px zero seams, so
    ``connectedComponents`` runs once and the component that contains the seed
    is kept.  Pixel for pixel this equals labelling each window on its own
    (outside the image the window is filled with -inf, which amounts to
    cropping it at the border) without tens of thousands of small-array calls.
    """
    response = np.ascontiguousarray(response, dtype=np.float32)
    coords = propose(response)
    total = int(coords.shape[0])
    if total == 0:
        return []
    h, w = response.shape
    r = _CANDIDATE_RADIUS
    size = 2 * r + 1
    pitch = size + 1
    padded = np.full((h + 2 * r, w + 2 * r), -np.inf, dtype=np.float32)
    padded[r:r + h, r:r + w] = response
    offsets = np.arange(size)
    objects = []
    for start in range(0, total, _CHUNK):
        cy = coords[start:start + _CHUNK, 0].astype(np.int64)
        cx = coords[start:start + _CHUNK, 1].astype(np.int64)
        n = cy.size
        rows = cy[:, None] + offsets[None, :]
        cols = cx[:, None] + offsets[None, :]
        crops = padded[rows[:, :, None], cols[:, None, :]]          # (n, 19, 19)
        peak = response[cy, cx].astype(np.float64)
        # Compute the threshold in float64 and then convert to float32, so the
        # rounding equals plain Python float arithmetic per candidate.
        low = np.maximum(np.maximum(SEED_THRESHOLD * .4, peak * .18), .8).astype(np.float32)
        inside = crops >= low[:, None, None]
        grid_rows, grid_cols, tile_r, tile_c = _tile_grid(n)
        mosaic = np.zeros((grid_rows * pitch, grid_cols * pitch), dtype=np.uint8)
        mosaic.reshape(grid_rows, pitch, grid_cols, pitch)[tile_r, :size, tile_c, :size] = inside
        _, labels = cv2.connectedComponents(mosaic, connectivity=8, ltype=cv2.CV_32S)
        tiles = labels.reshape(grid_rows, pitch, grid_cols, pitch)[tile_r, :size, tile_c, :size]
        member = tiles == tiles[:, r, r][:, None, None]             # the seed is always above the threshold
        obj, dy, dx = np.nonzero(member)                            # C order: candidate, row, column
        ys_all = cy[obj] - r + dy
        xs_all = cx[obj] - r + dx
        bounds = np.cumsum(np.bincount(obj, minlength=n))[:-1]
        ys_split = np.split(ys_all, bounds)
        xs_split = np.split(xs_all, bounds)
        objects.extend(
            (ys_split[i], xs_split[i], int(cy[i]), int(cx[i])) for i in range(n)
        )
    return objects


# --------------------------------------------------------------- features and scores
def _grouped_median(blocks, step):
    """Median of quantised values, interpolated inside the grey step that holds it.

    ``blocks`` holds absolute differences: the lowest class is [0, step/2) and
    every other one is centred on a multiple of ``step``.
    """
    n = blocks.shape[-1]
    middle = np.partition(blocks, (n - 1) // 2, axis=-1)[..., (n - 1) // 2]
    below = (blocks < (middle - .25 * step)[..., None]).sum(axis=-1)
    equal = (np.abs(blocks - middle[..., None]) <= .25 * step).sum(axis=-1)
    lowest = middle < .5 * step
    lower_edge = np.where(lowest, 0., middle - .5 * step)
    width = np.where(lowest, .5 * step, step)
    return lower_edge + width * (n / 2. - below) / np.maximum(equal, 1)


def local_noise(img, tile=50):
    """Local noise per ``tile`` block, from the MAD of neighbouring-pixel differences.

    On coarse data (8-bit files, very low counts) the plain median of the
    differences can only take whole grey steps, so the noise of neighbouring
    blocks would jump by ten per cent and more for no reason.  There the
    median is interpolated inside its grey step, as for grouped data; finely
    stepped data are estimated exactly as before.
    """
    dx = np.abs(np.diff(img, axis=1))
    step = _grey_step(dx)
    h, w = dx.shape
    ph, pw = (-h) % tile, (-w) % tile
    if ph or pw:
        dx = np.pad(dx, ((0, ph), (0, pw)), mode="reflect")
    r, c = dx.shape[0] // tile, dx.shape[1] // tile
    blocks = dx.reshape(r, tile, c, tile).transpose(0, 2, 1, 3).reshape(r, c, -1)
    medians = np.median(blocks, axis=2)
    if step > _COARSE_STEP * float(np.median(medians)):
        medians = _grouped_median(blocks, step)
    s = (medians * 1.4826 / np.sqrt(2)).astype("float32")
    s = ndi.median_filter(s, 3, mode="nearest")
    p = s[s > 0]
    s = np.maximum(s, .25 * float(np.median(p)) if p.size else 0.)
    return np.maximum(cv2.resize(s, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR), 1e-3)


def _raw_features_loop(objects, corrected, response, noise, reftone):
    """Per-candidate implementation, the fallback for supports larger than the usual window."""
    h, w = response.shape
    out = np.zeros((len(objects), 6), "float32")
    for i, (ys, xs, _, _) in enumerate(objects):
        area = ys.size
        y0, y1 = max(0, ys.min() - _MARGIN), min(h, ys.max() + _MARGIN + 1)
        x0, x1 = max(0, xs.min() - _MARGIN), min(w, xs.max() + _MARGIN + 1)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[ys - y0, xs - x0] = True
        mu = m.view(np.uint8)
        near = cv2.dilate(mu, _K1) > cv2.dilate(mu, _K5)
        nz = float(noise[ys, xs].mean())
        rn = (float((corrected[ys, xs].mean() - np.median(corrected[y0:y1, x0:x1][near])) / nz)
              if near.sum() > 20 else 0.)
        if area > 2:
            yf = ys.astype(np.float64); xf = xs.astype(np.float64)
            dy = yf - yf.mean(); dx = xf - xf.mean()
            n1 = area - 1
            a = dy @ dy / n1; b = dy @ dx / n1; c = dx @ dx / n1
            half = .5 * (a + c)
            disc = np.sqrt(max(.25 * (a - c) ** 2 + b * b, 0.))
            aspect = float(np.sqrt(max(half + disc, _PIXEL_VARIANCE) / max(half - disc, _PIXEL_VARIANCE)))
        else:
            aspect = 1.
        zin = response[ys, xs]
        rt = float(reftone[ys, xs].mean()) if reftone is not None else 0.
        out[i] = (rn, float(zin.mean()), float(zin.max()), nz, rt, aspect)
    return out


def raw_features(objects, corrected, response, noise, reftone):
    """The six quantities, in the order of ``feats`` in model_v6.json.

    Candidates are processed in batches: means and maxima over the support use
    grouped ``bincount``/``reduceat``; for the ring background every candidate
    goes into one mosaic that is dilated twice, and the median is taken from
    row-sorted values.  Each candidate sees exactly the pixels of the
    per-candidate implementation; means are accumulated in float64, which
    differs from float32 accumulation by about 1e-6, far below the resolution
    of the decision threshold.
    """
    total = len(objects)
    out = np.zeros((total, 6), "float32")
    if total == 0:
        return out
    h, w = response.shape
    for start in range(0, total, _CHUNK):
        chunk = objects[start:start + _CHUNK]
        n = len(chunk)
        counts = np.fromiter((o[0].size for o in chunk), dtype=np.int64, count=n)
        if counts.min() < 1:
            out[start:start + n] = _raw_features_loop(chunk, corrected, response, noise, reftone)
            continue
        cy = np.fromiter((o[2] for o in chunk), dtype=np.int64, count=n)
        cx = np.fromiter((o[3] for o in chunk), dtype=np.int64, count=n)
        ys = np.concatenate([o[0] for o in chunk]).astype(np.int64, copy=False)
        xs = np.concatenate([o[1] for o in chunk]).astype(np.int64, copy=False)
        obj = np.repeat(np.arange(n), counts)
        dy = ys - cy[obj]
        dx = xs - cx[obj]
        extent = int(max(np.abs(dy).max(), np.abs(dx).max()))
        if extent > 4 * _CANDIDATE_RADIUS:
            # Very large supports (not produced by this module) would make the
            # mosaic costly in memory: use the per-candidate implementation.
            out[start:start + n] = _raw_features_loop(chunk, corrected, response, noise, reftone)
            continue
        starts = np.zeros(n, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        inv = 1.0 / counts

        zin = response[ys, xs]
        nz = np.bincount(obj, weights=noise[ys, xs], minlength=n) * inv
        zmean = np.bincount(obj, weights=zin, minlength=n) * inv
        zmax = np.maximum.reduceat(zin, starts)
        corr_mean = np.bincount(obj, weights=corrected[ys, xs], minlength=n) * inv
        if reftone is not None:
            rt = np.bincount(obj, weights=reftone[ys, xs], minlength=n) * inv
        else:
            rt = np.zeros(n, dtype=np.float64)

        # Axis ratio: coordinates relative to the seed are small integers, so
        # their sums are exact in float64.
        dyf = dy.astype(np.float64)
        dxf = dx.astype(np.float64)
        sy = np.bincount(obj, weights=dyf, minlength=n)
        sx = np.bincount(obj, weights=dxf, minlength=n)
        syy = np.bincount(obj, weights=dyf * dyf, minlength=n)
        sxx = np.bincount(obj, weights=dxf * dxf, minlength=n)
        sxy = np.bincount(obj, weights=dyf * dxf, minlength=n)
        n1 = np.maximum(counts - 1, 1).astype(np.float64)
        my = sy * inv
        mx = sx * inv
        a = (syy - counts * my * my) / n1
        b = (sxy - counts * my * mx) / n1
        c = (sxx - counts * mx * mx) / n1
        half = .5 * (a + c)
        disc = np.sqrt(np.maximum(.25 * (a - c) ** 2 + b * b, 0.))
        # Pixels in one straight line have no spread across it; without the
        # width of the pixel itself the ratio would be about 30000.
        aspect = np.sqrt(np.maximum(half + disc, _PIXEL_VARIANCE) / np.maximum(half - disc, _PIXEL_VARIANCE))
        aspect[counts <= 2] = 1.

        # Ring background: mosaic plus two dilations gives the same pixel set
        # as dilating each candidate's own crop.
        tile_r = extent + _RING_RADIUS
        size = 2 * tile_r + 1
        pitch = size + 1
        grid_rows, grid_cols, tr, tc = _tile_grid(n)
        mosaic = np.zeros((grid_rows * pitch, grid_cols * pitch), dtype=np.uint8)
        mosaic[tr[obj] * pitch + dy + tile_r, tc[obj] * pitch + dx + tile_r] = 1
        near = cv2.dilate(mosaic, _K1) > cv2.dilate(mosaic, _K5)
        near = near.reshape(grid_rows, pitch, grid_cols, pitch)[tr, :size, tc, :size]
        offsets = np.arange(size) - tile_r
        img_y = cy[:, None] + offsets[None, :]
        img_x = cx[:, None] + offsets[None, :]
        near &= ((img_y >= 0) & (img_y < h))[:, :, None]
        near &= ((img_x >= 0) & (img_x < w))[:, None, :]
        ring = near.sum(axis=(1, 2))
        values = corrected[
            np.clip(img_y, 0, h - 1)[:, :, None], np.clip(img_x, 0, w - 1)[:, None, :]
        ]
        values = np.where(near, values, np.float32(np.inf)).reshape(n, -1)
        values.sort(axis=1)
        rows = np.arange(n)
        lo = np.maximum(ring - 1, 0) // 2
        hi = np.minimum(ring // 2, values.shape[1] - 1)
        median = (values[rows, lo] + values[rows, hi]) / 2      # float32, as np.median
        rn = np.where(ring > 20, (corr_mean - median.astype(np.float64)) / nz, 0.)

        out[start:start + n, 0] = rn
        out[start:start + n, 1] = zmean
        out[start:start + n, 2] = zmax
        out[start:start + n, 3] = nz
        out[start:start + n, 4] = rt
        out[start:start + n, 5] = aspect
    return out


def score_objects(objects, corrected, response, ref, model=None):
    """Discriminant score of every candidate; a score >= model['threshold'] is a target."""
    M = model or load_model()
    if not objects:
        return np.zeros(0, "float32")
    noise = local_noise(corrected)
    nmed = float(np.median(noise))
    reftone = (ref - _gauss(ref, 4.)) / local_noise(ref) if ref is not None else None
    F = raw_features(objects, corrected, response, noise, reftone)
    F[:, 3] /= nmed                                   # noiserel: relative to this image's median noise
    ref_col = M["feats"].index("reftone") if "reftone" in M["feats"] else 4
    logset = set(M["log_feats"])
    for j, name in enumerate(M["feats"]):
        if name in logset:
            F[:, j] = np.log1p(np.maximum(F[:, j], 0))
    med = np.median(F, 0)
    sd = np.empty(F.shape[1], "float64")
    for j in range(F.shape[1]):                       # scale from the lower (background) half only
        lo = F[:, j][F[:, j] <= med[j]]
        sd[j] = np.median(med[j] - lo) * 1.4826 if lo.size > 4 else 1.
    if "noiserel" in M["feats"]:
        # Uniform noise: the spread among the candidates is then only the
        # error of the estimate, and must not be blown up to a full score unit.
        noise_col = M["feats"].index("noiserel")
        sd[noise_col] = max(sd[noise_col], NOISE_TERM_MIN_SCALE)
    Z = np.clip((F - med) / np.maximum(sd, 1e-9), -M["clip"], M["clip"])
    Z = (Z - np.asarray(M["mean"])) / np.asarray(M["scale"])
    if ref is None:
        Z[:, ref_col] = 0.                            # missing term: training mean, no shift of the score
    return (Z @ np.asarray(M["coef"]) + M["intercept"]).astype("float32")


def sensitivity_shift(params):
    """How far the user moved the main threshold; positive accepts lower scores (more targets)."""
    value = getattr(params, "learned_sensitivity", 0.) or 0.
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.
    if not math.isfinite(value):
        return 0.
    return float(min(max(value, -SENSITIVITY_LIMIT), SENSITIVITY_LIMIT))


def _paint(target, objects, selected):
    """Write the supports of the selected candidates into a boolean image; return how many were selected."""
    index = np.flatnonzero(selected)
    if index.size:
        target[
            np.concatenate([objects[i][0] for i in index]),
            np.concatenate([objects[i][1] for i in index]),
        ] = True
    return int(index.size)


# --------------------------------------------------------------- program interface
def segment_loaded(loaded, params, *, reference_loaded=None):
    from . import universal_core as core
    started = time.perf_counter()
    params = params.validated(loaded.shape)
    raw, raw_note = _plane(loaded, target=True)
    reference = find_reference(loaded, reference_loaded)
    ref, ref_note = (_plane(reference, target=False) if reference is not None else (None, T("not provided")))
    corrected, response, global_noise = response_from_planes(raw, ref)

    objects = candidate_objects(response)
    model = load_model()
    scores = score_objects(objects, corrected, response, ref, model)
    shift = sensitivity_shift(params)
    main_t = float(model["threshold"]) - shift
    review_t = main_t - REVIEW_MARGIN

    mask = np.zeros(loaded.shape, bool)
    review = np.zeros(loaded.shape, bool)
    accepted = scores >= main_t
    n_main = _paint(mask, objects, accepted)
    n_review = _paint(review, objects, (scores >= review_t) & ~accepted)
    review &= ~mask

    # Measurement only: the learned Mask does not pass through the classic
    # morphology, shape filters or watershed.
    labels, boxes = core._plain_component_labels_with_boxes(mask)
    normalized = (raw / np.float32(FULL_SCALE)).astype(np.float32)
    mask, labels, objs = core.measure_and_filter_labels(
        labels, raw, normalized, response, params, apply_filters=False,
        response_reference=SEED_THRESHOLD, bounding_boxes=boxes)
    # ``raw`` is on the common 16-bit range whatever the bit depth of the file
    # (an 8-bit 255 arrives here as 65535), so clipping is judged on that range.
    source_dtype = np.dtype(loaded.image.dtype)
    diagnostics = core._input_intensity_diagnostics(
        raw, np.uint16 if source_dtype.kind in "ui" else source_dtype)

    manifest = _load_manifest()
    metadata = {
        "engine": ENGINE, "model_filename": MODEL_FILENAME,
        "model_sha256": manifest["model_sha256"],
        "model_version": model.get("version", "v6.3"),
        "main_threshold": main_t, "review_threshold": review_t,
        "model_threshold": float(model["threshold"]), "sensitivity": shift,
        "feature_names": model["feats"], "coefficients": model["coef"],
        "normalization": T(
            "Per-image median + lower-half MAD (anchored to the background, not inflated by true "
            "signal)",
        ),
        "detector_scale": SCALE_NOTE,
        "candidate_count": len(objects),
        "accepted_candidate_count": n_main, "review_candidate_count": n_review,
        "main_area_px": int(mask.sum()), "review_area_px": int(review.sum()),
        "original_main_area_px": int(mask.sum()), "original_review_area_px": int(review.sum()),
        "review_in_main_quantification": False,
        "saturated_fraction": float((raw >= SATURATION_LEVEL).mean()),
        "analysis_shape": [int(v) for v in loaded.shape], "resized": False,
        "global_noise": float(global_noise),
        "target_plane": raw_note, "reference_plane": ref_note,
        "reference_used": ref is not None,
        "source_image": {"path": str(loaded.path), **copy.deepcopy(loaded.metadata)},
        "reference_image": ({"path": str(reference.path), **copy.deepcopy(reference.metadata)}
                            if reference is not None else None),
        "feature_formula":
            T(
                "ringnear: how much brighter than the adjacent ring background / zmean: mean "
                "local contrast inside the target / zmax: peak / noiserel: neighbourhood noise / "
                "reftone: reference-channel brightness / aspect: long-to-short axis ratio; each "
                "is converted to a robust score relative to this image's candidates (scale from "
                "the lower half of the distribution)",
            ),
        "score_interpretation":
            T(
                "A linear discriminant score, not a probability. The threshold maximizes F1 on "
                "the calibration images; on another kind of image its strictness must be checked.",
            ),
        "known_limitations":
            T(
                "Edges of large bright stripe-like structures are easily mistaken for targets; "
                "inside clipped regions the contrast was lost at acquisition and cannot be "
                "recovered.",
            ),
        "support_definition": T(
            "The pixel extent comes from growing each candidate on the response map.",
        ),
        "postprocessing": [],
    }
    warnings = tuple(diagnostics[4]) + (
        T(
            "Yellow marks candidates scoring within 1.5 below the main threshold; they are for "
            "review and not part of the main quantification.",
        ),
        T(
            "Edges of large bright stripe-like structures are a known source of false positives; "
            "review such fields image by image.",
        ),
    )
    if ref is None:
        warnings += (T(
            "No reference channel was paired: crosstalk subtraction was skipped and the reftone "
            "term uses the training mean.",
        ),)
    result = core.UniversalResult(
        mask=mask, labels=labels, objects=objs, signal=raw,
        normalized_signal=normalized, response=response, threshold_value=SEED_THRESHOLD,
        low_threshold=SEED_THRESHOLD, positive_fraction=float(mask.mean()), params=params,
        signal_mode_used=ENGINE, input_black_value=0., input_white_value=FULL_SCALE,
        processing_seconds=time.perf_counter() - started, analysis_scale=1.,
        source_shape=loaded.shape,
        input_saturation_fraction=float(diagnostics[0]),
        input_high_plateau_fraction=float(diagnostics[1]),
        input_largest_tie_fraction=float(diagnostics[2]),
        input_unique_levels=int(diagnostics[3]),
        intensity_warnings=warnings,
        review_mask=review, review_base_mask=review.copy(), ai_metadata=metadata)
    return result


def render_review_rgb(loaded, result):
    from .universal_core import normalize_for_display
    gray = normalize_for_display(result.signal)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    main = np.asarray(result.mask, dtype=bool)
    review = np.asarray(result.review_mask, dtype=bool) & ~main
    rgb[main] = (rgb[main] * .45 + np.array([35, 255, 80]) * .55).astype(np.uint8)
    rgb[review] = (rgb[review] * .45 + np.array([255, 190, 35]) * .55).astype(np.uint8)
    return rgb


def render_original_ants16(loaded, mask):
    from .universal_core import _light_contour_dash_segments
    raw, _ = _plane(loaded, target=True)
    output = np.repeat(raw.astype(np.uint16)[..., None], 3, axis=2)
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 0, 0), 1, cv2.LINE_8)
    for contour in contours:
        segments, dot = _light_contour_dash_segments(contour, dash_length=6, gap_length=6)
        if dot is not None:
            cv2.circle(output, dot, 1, (65535, 65535, 65535), -1, cv2.LINE_8)
        elif segments:
            cv2.polylines(output, segments, False, (65535, 65535, 65535), 1, cv2.LINE_8)
    return output


def _recorded_reference(recorded, loaded):
    """Find the reference image named in an export record, or return None.

    Records name the file and its content fingerprint.  The file is looked up
    beside the original image and must match the fingerprint and the image size.
    """
    from .universal_core import load_image
    candidates = []
    former_location = recorded.get("path")          # written by earlier builds
    if isinstance(former_location, str) and former_location:
        candidates.append(Path(former_location))
    name = Path(str(recorded.get("file_name") or former_location or "")).name
    if name:
        candidates.append(loaded.path.parent / name)
    expected = recorded.get("sha256")
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            reference = load_image(candidate)
        except (OSError, ValueError):
            continue
        if reference.shape != loaded.shape:
            continue
        if expected and reference.metadata.get("sha256") != expected:
            continue
        return reference
    return None


def restore_sidecars(sidecar_path, loaded, result):
    from . import universal_core as core
    sidecar_path = Path(sidecar_path)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    metadata = payload.get("ai_metadata")
    if (payload.get("schema_version") != 1 or not isinstance(metadata, dict)
            or core.normalize_engine_id(metadata.get("engine")) != ENGINE):
        raise ValueError(T(
            "The version or recognition mode of the export record does not match; it cannot be "
            "restored.",
        ))
    if metadata.get("source_image", {}).get("sha256") != loaded.metadata.get("sha256"):
        raise ValueError(T(
            "The current original image differs from the one in the export record; the review "
            "area cannot be applied.",
        ))
    review_files = []
    for key in ["review_mask_file", "review_base_mask_file"]:
        filename = payload.get(key)
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(T("The path of the review file is invalid."))
        path = sidecar_path.parent / filename
        if not path.is_file():
            raise ValueError(T("The export lacks the review file: {0}", filename))
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.shape != loaded.shape:
            raise ValueError(T("The review selection and the original image differ in size."))
        review_files.append(image > 0)
    params = core.params_from_dict(payload.get("params", {}))
    if params.recognition_engine != ENGINE:
        raise ValueError(T("The recipe in the export record is not marked as learned mode."))
    raw, _ = _plane(loaded, target=True)
    recorded = metadata.get("reference_image") or {}
    reference = _recorded_reference(recorded, loaded)
    ref = _plane(reference, target=False)[0] if reference is not None else None
    _, response, _ = response_from_planes(raw, ref)
    restored_metadata = copy.deepcopy(metadata)
    restored_metadata["source_image"] = {"path": str(loaded.path), **copy.deepcopy(loaded.metadata)}
    if reference is not None:
        restored_metadata["reference_image"] = {"path": str(reference.path), **copy.deepcopy(reference.metadata)}
    elif recorded:
        # Keep what the record says about the reference, without a location:
        # there is no file left to verify when this result is exported again.
        restored_metadata["reference_image"] = core.public_image_record(recorded)
    seed = replace(result, params=params, signal=raw,
                   normalized_signal=(raw / np.float32(FULL_SCALE)).astype(np.float32),
                   response=response, threshold_value=SEED_THRESHOLD, low_threshold=SEED_THRESHOLD,
                   signal_mode_used=params.recognition_engine, input_black_value=0.,
                   input_white_value=FULL_SCALE, source_shape=loaded.shape, analysis_scale=1.,
                   review_mask=review_files[0], review_base_mask=review_files[1],
                   ai_metadata=restored_metadata,
                   edge_offset_px=int(payload.get("edge_offset_px", 0)),
                   edge_offset_grid_px=int(payload.get("edge_offset_grid_px", 0)), objects=[])
    restored = core.rebuild_result_from_mask(loaded, result.mask, params, previous=seed)
    restored.manually_edited = bool(payload.get("manually_edited", False))
    restored.ai_metadata["postprocessing"] = copy.deepcopy(metadata.get("postprocessing", []))
    return restored

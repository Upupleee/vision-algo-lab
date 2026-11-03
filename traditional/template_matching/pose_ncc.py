from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np
from scipy.signal import fftconvolve

from traditional.template_matching.ncc_matching import parabolic_subpixel


def _gray(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须为非空、有限数值的二维灰度图")
    return array


def _grid(bounds: tuple[float, float], step: float, name: str) -> list[float]:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} 必须包含两个有限数值")
    lo, hi = values
    if lo > hi or not np.isfinite(step) or step <= 0:
        raise ValueError(f"{name} 范围必须有序，步长必须为正有限数")
    count = int(np.floor((hi - lo) / step + 1e-10))
    if count > 10000:
        raise ValueError(f"{name} 采样过密")
    grid = [float(lo + i * step) for i in range(count + 1)]
    if not np.isclose(grid[-1], hi, rtol=0, atol=1e-10):
        grid.append(float(hi))
    return grid


def _warp(template: np.ndarray, mask: np.ndarray, angle: float, scale: float):
    h, w = template.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    rotated = corners @ matrix[:, :2].T + matrix[:, 2]
    lower = rotated.min(axis=0)
    extent = np.ceil(rotated.max(axis=0) - lower - 1e-9).astype(int) + 1
    matrix[:, 2] -= lower
    size = tuple(int(v) for v in extent)
    support = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_LINEAR)
    weighted = cv2.warpAffine(template * mask, matrix, size, flags=cv2.INTER_LINEAR)
    warped = np.divide(weighted, support, out=np.zeros_like(weighted), where=support > 1e-9)
    return warped, (support >= 1 - 1e-6).astype(np.float64), matrix


def _score(image: np.ndarray, template: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum()
    if count < 4:
        raise ValueError("有效模板像素不足")
    original = image
    selected = template[mask > 0]
    selected = selected / max(float(np.max(np.abs(selected))), np.finfo(float).tiny)
    selected = selected - selected[0]
    selected = selected - selected.mean()
    amplitude = np.max(np.abs(selected))
    if amplitude == 0:
        raise ValueError("有效模板为常量或对比度不足")
    selected /= amplitude
    centered = np.zeros_like(template)
    centered[mask > 0] = selected
    energy = np.sum(centered * centered)
    image = image / max(float(np.max(np.abs(image))), np.finfo(float).tiny)
    image = image - image.mean()
    kernel = mask[::-1, ::-1]
    sums = fftconvolve(image, kernel, mode="valid")
    squares = fftconvolve(image * image, kernel, mode="valid")
    variance = np.maximum(squares - sums * sums / count, 0)
    numerator = fftconvolve(image, centered[::-1, ::-1], mode="valid")
    valid = variance > np.finfo(float).eps * np.maximum(1.0, np.abs(squares)) * 256
    result = np.full(variance.shape, -np.inf)
    np.divide(numerator, np.sqrt(variance) * np.sqrt(energy), out=result, where=valid)
    if not valid.all() and np.any(original != original.flat[0]):
        windows = np.lib.stride_tricks.sliding_window_view(original, template.shape)
        indices = np.argwhere(~valid)
        for start in range(0, len(indices), 128):
            yy, xx = indices[start:start + 128].T
            patches = windows[yy, xx][:, mask > 0]
            patches = patches / np.maximum(np.max(np.abs(patches), axis=1, keepdims=True), np.finfo(float).tiny)
            patches -= patches[:, :1].copy()
            patches -= patches.mean(axis=1, keepdims=True)
            amplitudes = np.max(np.abs(patches), axis=1)
            usable = amplitudes > 0
            patches[usable] /= amplitudes[usable, None]
            lengths = np.linalg.norm(patches, axis=1)
            result[yy[usable], xx[usable]] = (patches[usable] @ selected) / (lengths[usable] * np.sqrt(energy))
    return np.clip(result, -1, 1, out=result, where=np.isfinite(result))


def _peaks(score: np.ndarray, count: int, radius: float) -> list[tuple[int, int, float]]:
    working = score.copy()
    peaks = []
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(working), working.shape)
        value = float(working[y, x])
        if not np.isfinite(value):
            break
        peaks.append((int(y), int(x), value))
        r = int(np.ceil(radius))
        y0, y1 = max(0, y - r), min(working.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(working.shape[1], x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        working[y0:y1, x0:x1][(yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] = -np.inf
    return peaks


def _search_pose(images, template, mask, levels, candidate_limit, nms_radius, exhaustive_verification):
    templates = [(template, mask)]
    for level in range(1, levels):
        previous, support = templates[-1]
        h, w = previous.shape
        if min(h, w) < 16:
            break
        size = ((w + 1) // 2, (h + 1) // 2)
        reduced_mask = cv2.resize(support, size, interpolation=cv2.INTER_AREA)
        weighted = cv2.resize(previous * support, size, interpolation=cv2.INTER_AREA)
        reduced = np.divide(weighted, reduced_mask, out=np.zeros_like(weighted), where=reduced_mask > 1e-9)
        reduced_mask = (reduced_mask >= 1 - 1e-6).astype(np.float64)
        if reduced_mask.sum() < 16:
            break
        try:
            _score(reduced, reduced, reduced_mask)
        except ValueError:
            break
        templates.append((reduced, reduced_mask))
    top = len(templates) - 1
    score = _score(images[top], *templates[top])
    peaks = _peaks(score, candidate_limit, nms_radius / 2 ** top)
    final = [(y, x, value, score, y, x) for y, x, value in peaks]
    for level in range(top - 1, -1, -1):
        image = images[level]
        tpl, support = templates[level]
        h, w = tpl.shape
        options = []
        for y, x, *_ in final:
            x0, y0 = max(0, 2 * x - 4), max(0, 2 * y - 4)
            x1 = min(image.shape[1] - w, 2 * x + 4)
            y1 = min(image.shape[0] - h, 2 * y + 4)
            if x0 > x1 or y0 > y1:
                continue
            local = _score(image[y0:y1 + h, x0:x1 + w], tpl, support)
            for ly, lx, value in _peaks(local, 2, nms_radius / 2 ** level):
                options.append((y0 + ly, x0 + lx, value, local, ly, lx))
        final = []
        for option in sorted(options, key=lambda item: -item[2]):
            if all((option[0] - p[0]) ** 2 + (option[1] - p[1]) ** 2 > (nms_radius / 2 ** level) ** 2 for p in final):
                final.append(option)
            if len(final) >= candidate_limit:
                break
    if top > 0 and exhaustive_verification:
        score = _score(images[0], template, mask)
        final = [(y, x, value, score, y, x) for y, x, value in _peaks(score, candidate_limit, nms_radius)]
    return final


def match_pose_ncc(
    image: np.ndarray,
    template: np.ndarray,
    *,
    angle_range: tuple[float, float] = (0.0, 0.0),
    angle_step: float = 5.0,
    scale_range: tuple[float, float] = (1.0, 1.0),
    scale_step: float = 0.1,
    min_score: float = 0.8,
    min_score_gap: float = 0.03,
    nms_radius: float = 8.0,
    max_count: int = 1,
    pyramid_levels: int = 3,
    candidate_limit: int = 8,
    subpixel: bool = True,
    mask: np.ndarray | None = None,
    exhaustive_verification: bool = True,
) -> dict:
    started = perf_counter()
    image = _gray(image, "image")
    template = _gray(template, "template")
    angles = _grid(angle_range, angle_step, "angle_range")
    scales = _grid(scale_range, scale_step, "scale_range")
    if scales[0] <= 0:
        raise ValueError("scale_range 必须为正")
    for name, value in (("max_count", max_count), ("pyramid_levels", pyramid_levels), ("candidate_limit", candidate_limit)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    if not np.isfinite(min_score) or not -1 <= min_score <= 1:
        raise ValueError("min_score 必须在 [-1, 1] 内")
    if not np.isfinite(min_score_gap) or not 0 <= min_score_gap <= 2:
        raise ValueError("min_score_gap 必须在 [0, 2] 内")
    if not np.isfinite(nms_radius) or nms_radius <= 0:
        raise ValueError("nms_radius 必须为正有限数")
    if max_count > 1 and min_score_gap > 0:
        raise ValueError("多目标模式请显式设置 min_score_gap=0")
    if candidate_limit < max(max_count, 2 if min_score_gap > 0 else 1):
        raise ValueError("candidate_limit 不足以支持输出数量或歧义检测")
    if pyramid_levels > 8 or len(angles) * len(scales) > 10000:
        raise ValueError("搜索配置过大：最多 8 层、10000 个姿态")
    if mask is None:
        mask = np.ones(template.shape, dtype=np.float64)
    else:
        mask = _gray(mask, "mask")
        if mask.shape != template.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("mask 必须与 template 同形且仅含 0/1")
    _score(template, template, mask)
    images = [image]
    for _ in range(1, pyramid_levels):
        h, w = images[-1].shape
        images.append(cv2.resize(images[-1], ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA))
    candidates = []
    searched = 0
    h, w = template.shape
    center = np.array([(w - 1) / 2, (h - 1) / 2, 1.0])
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]])
    for angle in angles:
        for scale in scales:
            theta = np.deg2rad(angle)
            bounds_w = scale * (abs(np.cos(theta)) * (w - 1) + abs(np.sin(theta)) * (h - 1))
            bounds_h = scale * (abs(np.sin(theta)) * (w - 1) + abs(np.cos(theta)) * (h - 1))
            if np.ceil(bounds_w - 1e-9) + 1 > image.shape[1] or np.ceil(bounds_h - 1e-9) + 1 > image.shape[0]:
                continue
            warped, support, matrix = _warp(template, mask, angle, scale)
            try:
                peaks = _search_pose(images, warped, support, pyramid_levels, candidate_limit, nms_radius, exhaustive_verification)
            except ValueError:
                continue
            searched += 1
            for y, x, score, local, ly, lx in peaks:
                neighborhood = local[max(0, ly - 1):ly + 2, max(0, lx - 1):lx + 2]
                refine = subpixel and np.isfinite(neighborhood).all()
                dy, dx = parabolic_subpixel(local, ly, lx) if refine else (0.0, 0.0)
                transform = matrix.copy()
                transform[:, 2] += (x + dx, y + dy)
                cx, cy = transform @ center
                candidates.append({
                    "x": float(transform[0, 2]), "y": float(transform[1, 2]),
                    "center_x": float(cx), "center_y": float(cy),
                    "bbox_x": float(x + dx), "bbox_y": float(y + dy),
                    "angle_deg": angle, "scale": scale, "score": score,
                    "transform": np.vstack([transform, [0, 0, 1]]).tolist(),
                    "corners": (corners @ transform.T).tolist(),
                })
    distinct = []
    for candidate in sorted(candidates, key=lambda item: -item["score"]):
        if all(np.hypot(candidate["center_x"] - p["center_x"], candidate["center_y"] - p["center_y"]) > nms_radius for p in distinct):
            distinct.append(candidate)
    best = distinct[0]["score"] if distinct else None
    second = distinct[1]["score"] if len(distinct) > 1 else None
    gap = best - second if second is not None else None
    matches = [item for item in distinct if item["score"] >= min_score][:max_count]
    reason = "matched"
    if searched == 0:
        reason = "no_valid_pose"
    elif not matches:
        reason = "below_min_score"
    elif min_score_gap > 0 and gap is not None and gap < min_score_gap:
        reason = "ambiguous"
        matches = []
    return {
        "matched": bool(matches), "reason": reason, "matches": matches,
        "best_score": best, "second_score": second, "score_gap": gap,
        "elapsed_ms": (perf_counter() - started) * 1000,
        "searched_poses": searched,
    }
